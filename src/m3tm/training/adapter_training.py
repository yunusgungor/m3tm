"""
Adapter ve Görev Başlığı Eğitim Yöneticisi

Bu modül, dondurulmuş çekirdek model üzerinde sadece adapter'ları ve
görev başlıklarını eğitmek için gerekli mekanizmaları sağlar.

Örüntüler:
- TrainingLoopTemplate (PT-013): Eğitim döngüsü şablonu
- ModelComposite (PT-003): Modüler model mimarisi
- FactoryMethod (PT-002): Eğitim yöneticisi oluşturma
- MetricsCollector (PT-008): Eğitim metriklerini toplama
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, Tuple, List, Set

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from m3tm.config.model_config import TrainingConfig
from m3tm.adapters.adapter import Adapter
from m3tm.adapters.adapter_manager import AdapterManager
from m3tm.task_heads.classification import ClassificationHead
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.training.metrics import MetricsCollector, calculate_classification_metrics

logger = logging.getLogger(__name__)


class AdapterTrainingConfig(TrainingConfig):
    """
    Adapter eğitimi için özelleştirilmiş yapılandırma sınıfı.
    """
    def __init__(
        self,
        train_adapter_names: List[str] = None,
        train_task_heads: bool = True,
        freeze_core_model: bool = True,
        checkpoint_frequency: int = 1,
        checkpoint_keep_best_n: int = 3,
        memory_optimization_level: str = "moderate",
        **kwargs
    ):
        """
        Args:
            train_adapter_names: Eğitilecek adapter isimlerinin listesi. None ise tüm adapter'lar eğitilir.
            train_task_heads: Görev başlıklarını eğit
            freeze_core_model: Çekirdek modeli dondur
            checkpoint_frequency: Kaç epoch'ta bir checkpoint oluşturulacak
            checkpoint_keep_best_n: Saklanacak en iyi checkpoint sayısı
            memory_optimization_level: Bellek optimizasyon seviyesi ("low", "moderate", "aggressive")
            **kwargs: Temel sınıfın diğer parametreleri
        """
        super().__init__(**kwargs)
        self.train_adapter_names = train_adapter_names
        self.train_task_heads = train_task_heads
        self.freeze_core_model = freeze_core_model
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_keep_best_n = checkpoint_keep_best_n
        self.memory_optimization_level = memory_optimization_level


class TrainingCallback:
    """
    Eğitim sürecindeki çeşitli aşamalarda çağrılabilecek callback sınıfı.
    """
    def on_training_start(self, trainer: 'AdapterTrainingManager', **kwargs) -> None:
        """Eğitim başladığında çağrılır."""
        pass
    
    def on_epoch_start(self, trainer: 'AdapterTrainingManager', epoch: int, **kwargs) -> None:
        """Her epoch başında çağrılır."""
        pass
    
    def on_batch_start(self, trainer: 'AdapterTrainingManager', batch: Any, **kwargs) -> None:
        """Her batch başında çağrılır."""
        pass
    
    def on_batch_end(self, trainer: 'AdapterTrainingManager', batch_metrics: Dict[str, float], **kwargs) -> None:
        """Her batch sonunda çağrılır."""
        pass
    
    def on_epoch_end(
        self, 
        trainer: 'AdapterTrainingManager', 
        epoch: int, 
        metrics: Dict[str, float], 
        **kwargs
    ) -> None:
        """Her epoch sonunda çağrılır."""
        pass
    
    def on_training_end(self, trainer: 'AdapterTrainingManager', **kwargs) -> None:
        """Eğitim bittiğinde çağrılır."""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """
    Erken durdurma callback'i.
    """
    def __init__(self, patience: int = 3, monitor: str = "val_loss", mode: str = "min"):
        """
        Args:
            patience: Kaç epoch boyunca iyileşme olmazsa durdurulacak
            monitor: İzlenecek metrik
            mode: "min" veya "max" metriğin optimize edilme yönü
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.stopped_epoch = 0
        self.should_stop = False
    
    def on_epoch_end(
        self, 
        trainer: 'AdapterTrainingManager', 
        epoch: int, 
        metrics: Dict[str, float], 
        **kwargs
    ) -> None:
        """Her epoch sonunda metriği kontrol eder ve gerekirse erken durdurma sinyali verir."""
        # Metriği al
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return
        
        # Daha iyi bir değer mi kontrol et
        if (self.mode == "min" and current_value < self.best_value) or \
           (self.mode == "max" and current_value > self.best_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                logger.info(f"Erken durdurma sinyali: Epoch {epoch}")


class LearningRateSchedulerCallback(TrainingCallback):
    """
    Öğrenme oranı zamanlayıcısı callback'i.
    """
    def __init__(
        self, 
        scheduler: torch.optim.lr_scheduler._LRScheduler, 
        monitor: Optional[str] = "val_loss",
        schedule_on_batch: bool = False
    ):
        """
        Args:
            scheduler: Öğrenme oranı zamanlayıcısı
            monitor: İzlenecek metrik (ReduceLROnPlateau için)
            schedule_on_batch: Batch sonunda mı yoksa epoch sonunda mı güncelleme yapılacak
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.schedule_on_batch = schedule_on_batch
    
    def on_batch_end(self, trainer: 'AdapterTrainingManager', batch_metrics: Dict[str, float], **kwargs) -> None:
        """Her batch sonunda zamanlayıcıyı günceller (eğer schedule_on_batch True ise)."""
        if self.schedule_on_batch and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
    
    def on_epoch_end(
        self, 
        trainer: 'AdapterTrainingManager', 
        epoch: int, 
        metrics: Dict[str, float], 
        **kwargs
    ) -> None:
        """Her epoch sonunda zamanlayıcıyı günceller."""
        if not self.schedule_on_batch:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                value = metrics.get(self.monitor, None)
                if value is not None:
                    self.scheduler.step(value)
            else:
                self.scheduler.step()


class AdapterTrainingManager:
    """
    Adapter ve görev başlığı eğitimini yöneten sınıf.
    
    Bu sınıf, çekirdek modeli dondurarak sadece adapter'ları ve görev başlıklarını
    eğitmek için gerekli işlevselliği sağlar.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: AdapterTrainingConfig,
        adapter_manager: Optional[AdapterManager] = None,
        save_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = False,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        """
        Args:
            model: Eğitilecek model
            config: Eğitim yapılandırması
            adapter_manager: Adapter yöneticisi (varsa)
            save_dir: Modelin ve metriklerin kaydedileceği dizin
            use_tensorboard: TensorBoard kullanılsın mı?
            callbacks: Eğitim sırasında çağrılacak callback'ler listesi
        """
        self.model = model
        self.config = config
        self.adapter_manager = adapter_manager
        self.callbacks = callbacks or []
        
        # Cihazı belirle
        self.device = self._get_device(config.device)
        
        # Metrik toplayıcı
        self.metrics_collector = MetricsCollector(save_dir, use_tensorboard)
        
        # Kaydetme dizini
        if save_dir is not None:
            if isinstance(save_dir, str):
                save_dir = Path(save_dir)
                
            self.save_dir = save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        
        # Eğitim durumu
        self.current_epoch = 0
        self.best_checkpoint_metrics = []
        self.global_step = 0
        self.training_active = False
        self.best_val_loss = float('inf')
        
        # Bellek optimizasyon düzeyi
        self.memory_optimized = False
    
    def _get_device(self, device_name: str) -> torch.device:
        """
        Eğitim cihazını belirler.
        
        Args:
            device_name: Cihaz adı ('cpu', 'cuda', 'mps')
            
        Returns:
            torch.device: Eğitim cihazı
        """
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def prepare_model_for_training(self) -> None:
        """
        Modeli eğitim için hazırlar:
        1. Çekirdek model parametrelerini dondurur (eğer yapılandırılmışsa)
        2. Sadece adapter'ları ve görev başlıklarını eğitilebilir yapar
        """
        # Önce tüm parametreleri dondur
        if self.config.freeze_core_model:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Adapter'ları eğitilebilir yap
        if self.adapter_manager is not None:
            adapters_to_train = self.config.train_adapter_names
            self.adapter_manager.set_adapters_trainable(adapters_to_train)
        else:
            # Model içinde adapter'ları bul
            for name, module in self.model.named_modules():
                if isinstance(module, Adapter):
                    if (self.config.train_adapter_names is None or 
                        any(adapter_name in name for adapter_name in self.config.train_adapter_names)):
                        for param in module.parameters():
                            param.requires_grad = True
        
        # Görev başlıklarını eğitilebilir yap
        if self.config.train_task_heads:
            for name, module in self.model.named_modules():
                if isinstance(module, ClassificationHead):
                    for param in module.parameters():
                        param.requires_grad = True
        
        # Eğitilebilir parametre sayısını loglama
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Toplam parametre sayısı: {total_params:,}")
        logger.info(f"Eğitilebilir parametre sayısı: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Optimizer oluşturur.
        
        Returns:
            optim.Optimizer: Optimizer
        """
        # Eğitilebilir parametreleri al
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer seç
        if self.config.optimizer == "adam":
            return optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        num_training_steps: int
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Öğrenme oranı zamanlayıcısı oluşturur.
        
        Args:
            optimizer: Optimizer
            num_training_steps: Toplam eğitim adımı sayısı
            
        Returns:
            optim.lr_scheduler._LRScheduler: Öğrenme oranı zamanlayıcısı
        """
        # Isınma adımı sayısı
        warmup_steps = self.config.warmup_steps
        
        # Zamanlayıcı seç
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - warmup_steps
            )
        elif self.config.scheduler == "linear":
            return optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_training_steps - warmup_steps
            )
        elif self.config.scheduler == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2
            )
        elif self.config.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _save_checkpoint(
        self,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Model checkpoint'i kaydeder.
        
        Args:
            optimizer: Optimizer
            epoch: Epoch numarası
            metrics: Güncel metrikler
            is_best: En iyi model mi?
        """
        if self.save_dir is None:
            return
        
        # Checkpoint dosya adı
        file_name = f"checkpoint_epoch_{epoch}.pt"
        file_path = self.save_dir / file_name
        
        # Modelin tamamı değil, sadece eğitilen kısımları kaydet
        model_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_state[name] = param.data.cpu().clone()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict_partial": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else self.config,
        }
        
        # Kaydet
        torch.save(checkpoint, file_path)
        
        # En iyi modelse, ayrıca "best_model.pt" olarak da kaydet
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"En iyi model kaydedildi: {best_path}")
        
        # En iyi N checkpoint'i tut
        if self.config.checkpoint_keep_best_n > 0:
            self.best_checkpoint_metrics.append((file_path, metrics.get("val_loss", float('inf'))))
            self.best_checkpoint_metrics.sort(key=lambda x: x[1])  # val_loss'a göre sırala
            
            # Fazla checkpoint'leri sil
            while len(self.best_checkpoint_metrics) > self.config.checkpoint_keep_best_n:
                worst_checkpoint_path, _ = self.best_checkpoint_metrics.pop()
                if worst_checkpoint_path.exists() and worst_checkpoint_path.name != "best_model.pt":
                    worst_checkpoint_path.unlink()
    
    def _load_checkpoint(
        self,
        optimizer: Optional[optim.Optimizer] = None,
        file_path: Optional[Union[str, Path]] = None
    ) -> int:
        """
        Checkpoint'ten model ve optimizer durumunu yükler.
        
        Args:
            optimizer: Optimizer
            file_path: Checkpoint dosya yolu
            
        Returns:
            int: Yüklenen epoch numarası
        """
        if file_path is None:
            if self.save_dir is None:
                return 0
            
            # En son checkpoint'i bul
            checkpoints = list(self.save_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                return 0
            
            # En son epoch'u bul
            epochs = [int(cp.stem.split("_")[-1]) for cp in checkpoints]
            max_epoch_idx = epochs.index(max(epochs))
            file_path = checkpoints[max_epoch_idx]
        
        # Checkpoint'i yükle
        checkpoint = torch.load(file_path, map_location=self.device)
        
        # Modeli yükle (sadece eğitilen kısımları)
        model_state_dict = checkpoint.get("model_state_dict_partial", {})
        model_param_dict = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        
        for name, param in model_param_dict.items():
            if name in model_state_dict:
                param.data.copy_(model_state_dict[name])
        
        # Optimizer'ı yükle
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Metrikleri yükle
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            self.metrics_collector.best_val_loss = metrics.get("val_loss", float('inf'))
            self.metrics_collector.best_val_accuracy = metrics.get("val_accuracy", 0.0)
        
        logger.info(f"Checkpoint yüklendi: {file_path}")
        return checkpoint.get("epoch", 0)

    def _optimize_memory(self, level: str = "moderate") -> None:
        """
        Eğitim sırasında bellek kullanımını optimize eder.
        
        Args:
            level: Optimizasyon seviyesi ("low", "moderate", "aggressive")
        """
        # PyTorch'un hafıza izlemesini devre dışı bırak
        torch.backends.cudnn.benchmark = True
        
        if level == "low":
            # Minimum optimizasyon
            pass
        
        elif level == "moderate":
            # Orta seviye optimizasyon
            if self.device.type == "cuda":
                # CUDA akış önbelleklerini boşalt
                torch.cuda.empty_cache()
        
        elif level == "aggressive":
            # Agresif optimizasyon
            if self.device.type == "cuda":
                # CUDA akış önbelleklerini boşalt
                torch.cuda.empty_cache()
                
                # Gradient hesaplamasını optimize et
                torch.backends.cudnn.deterministic = False
                
                # Bellek kullanımını azaltmak için belirli PyTorch operasyonlarını kullan
                torch._C._jit_set_bailout_depth(2)
    
    def _train_one_batch(
        self,
        batch: Union[Dict[str, torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, ...]],
        optimizer: optim.Optimizer,
        criterion: Callable
    ) -> Dict[str, float]:
        """
        Tek bir batch üzerinde eğitim adımı gerçekleştirir.
        
        Args:
            batch: Eğitim verisi (sözlük veya liste/tuple)
            optimizer: Optimizer
            criterion: Kayıp fonksiyonu
            
        Returns:
            Dict[str, float]: Eğitim metrikleri
        """
        # Callback'leri çağır
        for callback in self.callbacks:
            callback.on_batch_start(self, batch=batch)
        
        # Bellek optimizasyonu
        self._optimize_memory(self.config.memory_optimization_level)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # Batch'i işle
        if isinstance(batch, (list, tuple)):
            # TensorDataset'ten gelen veriler genellikle (inputs, labels) formatındadır
            if len(batch) >= 2:
                inputs = batch[0].to(self.device)  # Girişler genellikle ilk öğedir
                labels = batch[1].to(self.device)  # Etiketler genellikle ikinci öğedir
                
                # Model forward geçişi
                self.model.train()
                with torch.set_grad_enabled(True):
                    outputs = self.model(input_ids=inputs, labels=labels)
                
                # Eğer outputs sözlük değilse veya loss içermiyorsa, criterion kullan
                if not isinstance(outputs, dict) or "loss" not in outputs:
                    loss = criterion(outputs, {"input_ids": inputs, "labels": labels})
                else:
                    loss = outputs["loss"]
            else:
                # Sadece girdi var, etiket yok
                inputs = batch[0].to(self.device)
                
                # Model forward geçişi
                self.model.train()
                with torch.set_grad_enabled(True):
                    outputs = self.model(input_ids=inputs)
                
                # Eğer outputs sözlük değilse veya loss içermiyorsa, criterion kullan
                if not isinstance(outputs, dict) or "loss" not in outputs:
                    loss = criterion(outputs, {"input_ids": inputs})
                else:
                    loss = outputs["loss"]
        else:
            # Dictionary formatı
            inputs = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            
            # Model forward geçişi
            self.model.train()
            with torch.set_grad_enabled(True):
                outputs = self.model(**inputs)
            
            # Eğer outputs sözlük değilse veya loss içermiyorsa, criterion kullan
            if not isinstance(outputs, dict) or "loss" not in outputs:
                loss = criterion(outputs, inputs)
            else:
                loss = outputs["loss"]
        
        # Geriye yayılım
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # Optimize et
        optimizer.step()
        
        # Metrikleri oluştur
        metrics = {"train_loss": loss.item()}
        
        # Callback'leri çağır
        for callback in self.callbacks:
            callback.on_batch_end(self, batch_metrics=metrics)
        
        # Global adım sayısını artır
        self.global_step += 1
        
        return metrics
    
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: Callable
    ) -> Dict[str, float]:
        """
        Doğrulama değerlendirmesini gerçekleştirir.
        
        Args:
            val_loader: Doğrulama veri yükleyicisi
            criterion: Kayıp fonksiyonu
            
        Returns:
            Dict[str, float]: Doğrulama metrikleri
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        # Tüm çıktıları ve etiketleri topla
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Batch'i doğru formata dönüştür
                if isinstance(batch, (list, tuple)):
                    # TensorDataset'ten gelen liste/tuple formatı
                    inputs = batch[0].to(self.device)  # İlk tensor genellikle girdiler
                    if len(batch) > 1:
                        labels = batch[1].to(self.device)  # İkinci tensor genellikle etiketler
                        outputs = self.model(input_ids=inputs, labels=labels)
                    else:
                        outputs = self.model(input_ids=inputs)
                else:
                    # Dictionary formatı
                    inputs = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                    outputs = self.model(**inputs)
                
                # Kayıp hesapla
                if isinstance(batch, (list, tuple)):
                    batch_input = {"input_ids": inputs, "labels": labels} if len(batch) > 1 else {"input_ids": inputs}
                    batch_loss = criterion(outputs, batch_input)
                else:
                    batch_loss = criterion(outputs, inputs)
                
                val_loss += batch_loss.item()
                num_batches += 1
                
                # Tahminleri ve etiketleri topla (sınıflandırma için)
                if isinstance(outputs, dict) and "logits" in outputs:
                    all_preds.append(outputs["logits"].detach().cpu())
                    
                    if isinstance(batch, (list, tuple)) and len(batch) > 1:
                        all_labels.append(labels.detach().cpu())
                    elif isinstance(batch, dict) and "labels" in batch:
                        all_labels.append(batch["labels"].detach().cpu())
        
        # Ortalama kayıp hesapla
        val_loss /= max(num_batches, 1)
        
        # Doğrulama metriklerini oluştur
        val_metrics = {
            "val_loss": val_loss
        }
        
        # Sınıflandırma metrikleri
        if all_preds and all_labels:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            val_metrics.update(
                calculate_classification_metrics(all_preds, all_labels)
            )
        
        return val_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        criterion: Callable,
        val_loader: Optional[DataLoader] = None,
        resume: bool = False
    ) -> Dict[str, float]:
        """
        Modeli eğitir.
        
        Args:
            train_loader: Eğitim veri yükleyicisi
            criterion: Kayıp fonksiyonu
            val_loader: Doğrulama veri yükleyicisi (opsiyonel)
            resume: Eğitime kaldığı yerden devam etsin mi?
            
        Returns:
            Dict[str, float]: Son eğitim metrikleri
        """
        # Modeli eğitim cihazına taşı
        self.model.to(self.device)
        
        # Optimizer oluştur
        optimizer = self._create_optimizer()
        
        # Zamanlayıcı oluştur (scheduler)
        num_training_steps = len(train_loader) * self.config.epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)
        
        # LR scheduler callback'i ekle (eğer verilmemişse)
        if scheduler is not None and not any(isinstance(cb, LearningRateSchedulerCallback) for cb in self.callbacks):
            scheduler_cb = LearningRateSchedulerCallback(
                scheduler,
                monitor="val_loss" if val_loader is not None else None,
                schedule_on_batch=False
            )
            self.callbacks.append(scheduler_cb)
        
        # Eğitime devam et
        start_epoch = 0
        if resume:
            start_epoch = self._load_checkpoint(optimizer)
            self.current_epoch = start_epoch
        
        # Callback'leri çağır
        for callback in self.callbacks:
            callback.on_training_start(self)
        
        # Eğitim durumunu ayarla
        self.training_active = True
        
        # Eğitim döngüsü
        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Callback'leri çağır
            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch=epoch)
            
            # Epoch başlangıç zamanı
            epoch_start_time = time.time()
            
            # Eğitim
            self.model.train()
            train_metrics_sum = {}
            num_batches = 0
            
            # Batch'ler üzerinde döngü
            for batch in train_loader:
                # Batch eğitimi
                batch_metrics = self._train_one_batch(batch, optimizer, criterion)
                
                # Toplam metrikleri güncelle
                for k, v in batch_metrics.items():
                    train_metrics_sum[k] = train_metrics_sum.get(k, 0.0) + v
                
                num_batches += 1
            
            # Ortalama eğitim metriklerini hesapla
            train_metrics = {}
            for k, v in train_metrics_sum.items():
                train_metrics[k] = v / num_batches
            
            # Doğrulama
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate(val_loader, criterion)
                
                # En iyi model mi kontrol et
                is_best = False
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    is_best = True
                
                if "val_accuracy" in val_metrics and val_metrics["val_accuracy"] > self.metrics_collector.best_val_accuracy:
                    self.metrics_collector.best_val_accuracy = val_metrics["val_accuracy"]
            
            # Epoch sonu
            epoch_time = time.time() - epoch_start_time
            
            # Mevcut öğrenme oranını al
            current_lr = optimizer.param_groups[0]['lr']
            
            # Epoch sonunda metrikleri birleştir ve kaydet
            all_metrics = {**train_metrics, **val_metrics}
            
            # Epoch sonunda metrikleri koleksiyona ekle
            if self.metrics_collector is not None:
                # Eğitim ve doğrulama metriklerini koleksiyona ekle
                self.metrics_collector.update_train_metrics(
                    loss=train_metrics['train_loss'],
                    accuracy=train_metrics.get('accuracy', None),
                    learning_rate=current_lr,
                    train_time=epoch_time
                )
                
                if val_loader is not None:
                    self.metrics_collector.update_val_metrics(
                        loss=val_metrics['val_loss'],
                        accuracy=val_metrics.get('accuracy', None)
                    )
                
                # Metrikleri dosyaya kaydet
                if self.save_dir:
                    self.metrics_collector.save_metrics()
            
            # En iyi modeli kaydet
            is_best = False
            if val_loader is not None:
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    is_best = True
                
                self._save_checkpoint(
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=all_metrics,
                    is_best=is_best
                )
            
            # Metrikleri loglama
            logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                       f"Train Loss: {train_metrics['train_loss']:.4f} - "
                       f"Val Loss: {val_metrics.get('val_loss', 'N/A')} - "
                       f"Time: {epoch_time:.2f}s")
            
            # Callback'leri çağır
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch=epoch, metrics=all_metrics)
            
            # Erken durdurma kontrolü
            early_stopping_callback = next((cb for cb in self.callbacks if isinstance(cb, EarlyStoppingCallback)), None)
            if early_stopping_callback and early_stopping_callback.should_stop:
                logger.info(f"Erken durdurma: Epoch {epoch+1}")
                break
        
        # Eğitim durumunu güncelle
        self.training_active = False
        
        # En son checkpoint'i kaydet (eğer yapılandırılmışsa)
        self._save_checkpoint(
            optimizer=optimizer,
            epoch=self.config.epochs - 1,
            metrics=all_metrics,
            is_best=False
        )
        
        # Callback'leri çağır
        for callback in self.callbacks:
            callback.on_training_end(self)
        
        # Son metrikleri döndür
        return all_metrics
    
    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: Callable
    ) -> Dict[str, float]:
        """
        Test seti üzerinde değerlendirme yapar.
        
        Args:
            test_loader: Test veri yükleyicisi
            criterion: Kayıp fonksiyonu
            
        Returns:
            Dict[str, float]: Test metrikleri
        """
        self.model.to(self.device)
        self.model.eval()
        
        test_metrics = self._validate(test_loader, criterion)
        
        # "val_" ile başlayan metrik adlarını "test_" ile değiştir
        test_metrics_renamed = {}
        for k, v in test_metrics.items():
            if k.startswith("val_"):
                test_metrics_renamed["test_" + k[4:]] = v
            else:
                test_metrics_renamed[k] = v
        
        logger.info(f"Test sonuçları: {test_metrics_renamed}")
        return test_metrics_renamed
    
    def get_trainable_parameter_count(self) -> Dict[str, int]:
        """
        Eğitilebilir parametre sayılarını bileşen bazında döndürür.
        
        Returns:
            Dict[str, int]: Bileşen adı ve parametre sayısı
        """
        param_counts = {}
        
        # Adapter'ları kontrol et
        for name, module in self.model.named_modules():
            if isinstance(module, Adapter) and any(p.requires_grad for p in module.parameters()):
                param_counts[f"adapter.{name}"] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Görev başlıklarını kontrol et
        for name, module in self.model.named_modules():
            if isinstance(module, ClassificationHead) and any(p.requires_grad for p in module.parameters()):
                param_counts[f"task_head.{name}"] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Toplam
        param_counts["total"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return param_counts


def create_training_manager(
    model: nn.Module,
    config: Union[AdapterTrainingConfig, Dict[str, Any]],
    adapter_manager: Optional[AdapterManager] = None,
    save_dir: Optional[Union[str, Path]] = None,
    use_tensorboard: bool = False,
    callbacks: Optional[List[TrainingCallback]] = None
) -> AdapterTrainingManager:
    """
    AdapterTrainingManager oluşturur.
    
    Args:
        model: Eğitilecek model
        config: Eğitim yapılandırması
        adapter_manager: Adapter yöneticisi
        save_dir: Kaydetme dizini
        use_tensorboard: TensorBoard kullanılsın mı?
        callbacks: Callback'ler listesi
        
    Returns:
        AdapterTrainingManager: Adapter eğitim yöneticisi
    """
    # Yapılandırma nesnesini oluştur
    if isinstance(config, dict):
        config = AdapterTrainingConfig(**config)
    
    # Adapter eğitim yöneticisi oluştur
    return AdapterTrainingManager(
        model=model,
        config=config,
        adapter_manager=adapter_manager,
        save_dir=save_dir,
        use_tensorboard=use_tensorboard,
        callbacks=callbacks
    ) 