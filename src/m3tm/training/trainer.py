"""
Eğitim Döngüsü Modülü

Bu modül, M³TM modeli için eğitim döngüleri ve ilgili yardımcı sınıfları içerir.
Farklı görevler için özelleştirilmiş eğitim stratejileri ve döngüleri sağlar.

Örüntüler:
- TrainingLoopTemplate: Standart eğitim döngüsü şablonu
- MetricsCollector (PT-008): Performans metriklerini toplama ve raporlama
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from m3tm.config.model_config import TrainingConfig
from m3tm.task_heads import ClassificationHead
from m3tm.embedding import TextEmbedding
from m3tm.transformer import ProtoTransformerBlock
from .metrics import MetricsCollector, calculate_classification_metrics


class TrainingLoopTemplate:
    """
    Eğitim döngüsü için şablon sınıf.
    
    Bu sınıf, eğitim döngüsünün genel yapısını tanımlar ve alt sınıfların
    özelleştirmesi için kancalar (hooks) sağlar.
    
    Örüntü: Template Method
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        save_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = False
    ):
        """
        TrainingLoopTemplate sınıfını başlatır.
        
        Args:
            config: Eğitim yapılandırması
            save_dir: Modelin ve metriklerin kaydedileceği dizin
            use_tensorboard: TensorBoard kullanılsın mı?
        """
        self.config = config
        
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
        
        # Erken durdurma sayacı
        self.early_stopping_counter = 0
    
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
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Optimizer oluşturur.
        
        Args:
            model: Eğitilecek model
            
        Returns:
            optim.Optimizer: Optimizer
        """
        # Eğitilebilir parametreleri al
        parameters = [p for p in model.parameters() if p.requires_grad]
        
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
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        file_name: str = "checkpoint.pt"
    ) -> None:
        """
        Model checkpoint'i kaydeder.
        
        Args:
            model: Model
            optimizer: Optimizer
            epoch: Epoch numarası
            file_name: Kaydedilecek dosya adı
        """
        if self.save_dir is not None:
            file_path = self.save_dir / file_name
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": self.metrics_collector.best_val_loss,
                "best_val_accuracy": self.metrics_collector.best_val_accuracy
            }
            
            torch.save(checkpoint, file_path)
    
    def _load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        file_path: Optional[Union[str, Path]] = None
    ) -> int:
        """
        Model checkpoint'i yükler.
        
        Args:
            model: Model
            optimizer: Optimizer
            file_path: Yüklenecek dosya yolu
            
        Returns:
            int: Yüklenen epoch numarası
        """
        if file_path is None and self.save_dir is not None:
            file_path = self.save_dir / "checkpoint.pt"
        
        if file_path is None or not os.path.exists(file_path):
            return 0
        
        checkpoint = torch.load(file_path, map_location=self.device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "best_val_loss" in checkpoint:
            self.metrics_collector.best_val_loss = checkpoint["best_val_loss"]
            
        if "best_val_accuracy" in checkpoint:
            self.metrics_collector.best_val_accuracy = checkpoint["best_val_accuracy"]
        
        return checkpoint["epoch"]
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: Callable,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Bir epoch eğitim yapar.
        
        Alt sınıflar bu metodu override etmelidir.
        
        Args:
            model: Eğitilecek model
            train_loader: Eğitim veri yükleyicisi
            optimizer: Optimizer
            criterion: Kayıp fonksiyonu
            scheduler: Öğrenme oranı zamanlayıcısı
            
        Returns:
            Dict[str, float]: Eğitim metrikleri
        """
        raise NotImplementedError("Alt sınıflar bu metodu uygulamalıdır.")
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: Callable
    ) -> Dict[str, float]:
        """
        Doğrulama yapar.
        
        Alt sınıflar bu metodu override etmelidir.
        
        Args:
            model: Doğrulanacak model
            val_loader: Doğrulama veri yükleyicisi
            criterion: Kayıp fonksiyonu
            
        Returns:
            Dict[str, float]: Doğrulama metrikleri
        """
        raise NotImplementedError("Alt sınıflar bu metodu uygulamalıdır.")
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[Callable] = None,
        resume: bool = False
    ) -> nn.Module:
        """
        Modeli eğitir.
        
        Args:
            model: Eğitilecek model
            train_loader: Eğitim veri yükleyicisi
            val_loader: Doğrulama veri yükleyicisi
            criterion: Kayıp fonksiyonu
            resume: Eğitimi kaldığı yerden devam ettir
            
        Returns:
            nn.Module: Eğitilmiş model
        """
        # Cihaza taşı
        model = model.to(self.device)
        
        # Parametre sayısını kaydet
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.metrics_collector.set_parameter_count(param_count)
        
        # Optimizer ve scheduler oluştur
        optimizer = self._create_optimizer(model)
        num_training_steps = len(train_loader) * self.config.epochs
        scheduler = self._create_scheduler(optimizer, num_training_steps)
        
        # Kayıp fonksiyonu belirtilmemişse varsayılan olarak CrossEntropyLoss kullan
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # Eğitimi kaldığı yerden devam ettir
        start_epoch = 0
        if resume:
            start_epoch = self._load_checkpoint(model, optimizer)
        
        # Eğitim döngüsü
        for epoch in range(start_epoch, self.config.epochs):
            # Eğitim
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion, scheduler)
            
            # Doğrulama
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate(model, val_loader, criterion)
                
                # Metrikleri güncelle
                improved = self.metrics_collector.update_val_metrics(
                    val_metrics["loss"],
                    val_metrics.get("accuracy"),
                    val_metrics.get("precision"),
                    val_metrics.get("recall"),
                    val_metrics.get("f1")
                )
                
                # En iyi modeli kaydet
                if improved:
                    self._save_checkpoint(model, optimizer, epoch, "best_model.pt")
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # Erken durdurma
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Her epoch'ta checkpoint kaydet
            self._save_checkpoint(model, optimizer, epoch)
            
            # Metrikleri ve özeti kaydet
            self.metrics_collector.save_metrics()
            self.metrics_collector.save_summary()
            
            # Epoch sonuçlarını yazdır
            self._print_epoch_results(epoch, train_metrics, val_metrics)
        
        # Eğitim sonunda grafiği çiz
        self.metrics_collector.plot_metrics()
        
        # TensorBoard writer'ı kapat
        self.metrics_collector.close()
        
        # En iyi modeli yükle
        if val_loader is not None and os.path.exists(self.save_dir / "best_model.pt"):
            self._load_checkpoint(model, file_path=self.save_dir / "best_model.pt")
        
        return model
    
    def _print_epoch_results(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """
        Epoch sonuçlarını yazdırır.
        
        Args:
            epoch: Epoch numarası
            train_metrics: Eğitim metrikleri
            val_metrics: Doğrulama metrikleri
        """
        print(f"Epoch {epoch + 1}/{self.config.epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics.get('accuracy', 0):.4f}")
        
        if val_metrics:
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics.get('accuracy', 0):.4f}")
        
        print("-" * 50)


class TextClassificationTrainer(TrainingLoopTemplate):
    """
    Metin sınıflandırma eğitimi için özelleştirilmiş eğitim döngüsü.
    
    Bu sınıf, TextEmbedding, ProtoTransformerBlock ve ClassificationHead
    bileşenlerini birleştiren bir modeli eğitmek için kullanılır.
    
    Örüntüler:
    - TrainingLoopTemplate: Standart eğitim döngüsü şablonu
    - ModelComposite (PT-003): Alt modülleri birleştiren kompozit model yapısı
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        save_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = False
    ):
        """
        TextClassificationTrainer sınıfını başlatır.
        
        Args:
            config: Eğitim yapılandırması
            save_dir: Modelin ve metriklerin kaydedileceği dizin
            use_tensorboard: TensorBoard kullanılsın mı?
        """
        super().__init__(config, save_dir, use_tensorboard)
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: Callable,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Bir epoch metin sınıflandırma eğitimi yapar.
        
        Args:
            model: Eğitilecek model (TextEmbedding, ProtoTransformerBlock ve ClassificationHead bileşenlerini içerir)
            train_loader: Eğitim veri yükleyicisi
            optimizer: Optimizer
            criterion: Kayıp fonksiyonu
            scheduler: Öğrenme oranı zamanlayıcısı
            
        Returns:
            Dict[str, float]: Eğitim metrikleri
        """
        model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Veriyi cihaza taşı
            inputs = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            targets = batch["labels"].to(self.device)
            
            # Gradyanları sıfırla
            optimizer.zero_grad()
            
            # İleri geçiş
            outputs = model(inputs, attention_mask)
            
            # Kayıp hesapla
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            
            loss = criterion(logits, targets)
            
            # Geri yayılım
            loss.backward()
            
            # Gradyan kırpma
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.gradient_clip
                )
            
            # Parametreleri güncelle
            optimizer.step()
            
            # Öğrenme oranını güncelle
            if scheduler is not None and self.config.scheduler != "reduce_on_plateau":
                scheduler.step()
            
            # İstatistikleri topla
            total_loss += loss.item()
            
            # Tahminleri topla
            if isinstance(outputs, dict) and "probs" in outputs:
                predictions = outputs["probs"]
            else:
                if logits.size(-1) > 1:  # Çok sınıflı
                    predictions = torch.softmax(logits, dim=-1)
                else:  # İkili sınıflandırma
                    predictions = torch.sigmoid(logits)
            
            all_predictions.append(predictions.detach())
            all_targets.append(targets)
        
        # Eğitim süresi
        train_time = time.time() - start_time
        
        # Tahminleri birleştir
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Metrikleri hesapla
        metrics = calculate_classification_metrics(all_predictions, all_targets)
        metrics["loss"] = total_loss / len(train_loader)
        
        # Bellek kullanımını hesapla (sadece CUDA için)
        memory_usage = 0
        if self.device.type == "cuda":
            memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # Öğrenme oranını al
        learning_rate = optimizer.param_groups[0]["lr"]
        
        # Metrikleri güncelle
        self.metrics_collector.update_train_metrics(
            metrics["loss"],
            metrics.get("accuracy"),
            learning_rate,
            train_time,
            memory_usage,
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("f1")
        )
        
        return metrics
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: Callable
    ) -> Dict[str, float]:
        """
        Metin sınıflandırma doğrulaması yapar.
        
        Args:
            model: Doğrulanacak model (TextEmbedding, ProtoTransformerBlock ve ClassificationHead bileşenlerini içerir)
            val_loader: Doğrulama veri yükleyicisi
            criterion: Kayıp fonksiyonu
            
        Returns:
            Dict[str, float]: Doğrulama metrikleri
        """
        model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Veriyi cihaza taşı
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                targets = batch["labels"].to(self.device)
                
                # İleri geçiş
                outputs = model(inputs, attention_mask)
                
                # Kayıp hesapla
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs
                
                loss = criterion(logits, targets)
                
                # İstatistikleri topla
                total_loss += loss.item()
                
                # Tahminleri topla
                if isinstance(outputs, dict) and "probs" in outputs:
                    predictions = outputs["probs"]
                else:
                    if logits.size(-1) > 1:  # Çok sınıflı
                        predictions = torch.softmax(logits, dim=-1)
                    else:  # İkili sınıflandırma
                        predictions = torch.sigmoid(logits)
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Tahminleri birleştir
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Metrikleri hesapla
        metrics = calculate_classification_metrics(all_predictions, all_targets)
        metrics["loss"] = total_loss / len(val_loader)
        
        return metrics
    
    @staticmethod
    def create_composite_model(
        text_embedding: TextEmbedding,
        transformer_block: ProtoTransformerBlock,
        classification_head: ClassificationHead
    ) -> nn.Module:
        """
        TextEmbedding, ProtoTransformerBlock ve ClassificationHead bileşenlerini
        birleştiren bir kompozit model oluşturur.
        
        Args:
            text_embedding: Metin gömme modülü
            transformer_block: Transformer bloğu
            classification_head: Sınıflandırma başlığı
            
        Returns:
            nn.Module: Kompozit model
            
        Örüntü: ModelComposite (PT-003)
        """
        class TextClassificationModel(nn.Module):
            """
            Metin sınıflandırma için kompozit model.
            
            Örüntü: ModelComposite (PT-003)
            """
            
            def __init__(
                self,
                embedding: TextEmbedding,
                transformer: ProtoTransformerBlock,
                head: ClassificationHead
            ):
                """
                TextClassificationModel sınıfını başlatır.
                
                Args:
                    embedding: Metin gömme modülü
                    transformer: Transformer bloğu
                    head: Sınıflandırma başlığı
                """
                super().__init__()
                self.embedding = embedding
                self.transformer = transformer
                self.head = head
            
            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
            ) -> Dict[str, torch.Tensor]:
                """
                TextClassificationModel modülünün ileri geçişi.
                
                Args:
                    input_ids: Token ID'leri [batch_size, seq_length]
                    attention_mask: Dikkat maskesi [batch_size, seq_length]
                    
                Returns:
                    Dict[str, torch.Tensor]: Çıktı sözlüğü
                """
                # Metin gömme
                embedding_outputs = self.embedding(
                    input_ids,
                    attention_mask,
                    return_dict=True
                )
                
                embeddings = embedding_outputs["embeddings"]
                if attention_mask is None and "attention_mask" in embedding_outputs:
                    attention_mask = embedding_outputs["attention_mask"]
                
                # Transformer bloğu
                transformer_outputs, transformer_metrics = self.transformer(
                    embeddings,
                    attention_mask
                )
                
                # Sınıflandırma başlığı
                head_outputs = self.head(
                    transformer_outputs,
                    attention_mask,
                    return_dict=True
                )
                
                # Metrikleri birleştir
                metrics = {}
                metrics.update(transformer_metrics)
                if "metrics" in head_outputs:
                    metrics.update(head_outputs["metrics"])
                
                return {
                    "logits": head_outputs["logits"],
                    "probs": head_outputs["probs"],
                    "metrics": metrics
                }
            
            def count_parameters(self) -> Dict[str, int]:
                """
                Modül başına eğitilebilir parametre sayısını hesaplar.
                
                Returns:
                    Dict[str, int]: Modül başına parametre sayısı
                """
                return {
                    "embedding": sum(p.numel() for p in self.embedding.parameters() if p.requires_grad),
                    "transformer": sum(p.numel() for p in self.transformer.parameters() if p.requires_grad),
                    "head": sum(p.numel() for p in self.head.parameters() if p.requires_grad),
                    "total": sum(p.numel() for p in self.parameters() if p.requires_grad)
                }
        
        return TextClassificationModel(text_embedding, transformer_block, classification_head) 