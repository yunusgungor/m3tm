"""
Genel Eğitim Yöneticisi

Bu modül, M³TM modeli için genel eğitim yönetimi sağlar.
TrainingManager sınıfı, hem çekirdek modelin hem de adapters ve task heads'in
tamamının veya belirli kısımlarının eğitilmesini koordine eder.

Örüntüler:
- FactoryMethod (PT-002): Farklı eğitim yöneticileri oluşturma
- DecoratorPattern (PT-017): İşlevselliği dinamik olarak genişletme
- ModelComposite (PT-003): Modüler model mimarisi
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, Tuple, List, Set

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from m3tm.config.model_config import TrainingConfig
from m3tm.adapters.adapter import Adapter
from m3tm.adapters.adapter_manager import AdapterManager
from m3tm.task_heads.base import TaskHead
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.training.adapter_training import AdapterTrainingManager, AdapterTrainingConfig, TrainingCallback
from m3tm.training.adapter_training_utils import (
    create_adapter_training_callbacks,
    create_adapter_criterion,
    find_task_heads_and_adapters
)

logger = logging.getLogger(__name__)

class TrainingManager:
    """
    M³TM modeli için genel eğitim yönetimi sağlayan sınıf.
    
    Bu sınıf, AdapterTrainingManager'ı kapsayarak yüksek seviye bir arayüz sunar
    ve eğitim sürecini koordine eder. Adapter'lar, görev başlıkları ve çekirdek modelin
    farklı kısımlarının eğitilmesini yönetir.
    
    Ayrıca eğitim ilerleme takibi, bellek optimizasyonu ve checkpointing için
    gelişmiş mekanizmalar sağlar.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        adapter_manager: Optional[AdapterManager] = None,
        save_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = False,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        """
        TrainingManager sınıfını başlatır.
        
        Args:
            model: Eğitilecek M³TM modeli
            config: Eğitim yapılandırması
            adapter_manager: AdapterManager örneği (varsa)
            save_dir: Model ve metriklerin kaydedileceği dizin
            use_tensorboard: TensorBoard kullanılsın mı
            callbacks: Eğitim sürecindeki çeşitli aşamalarda çağrılacak callbacks
        """
        self.model = model
        self.config = config
        self.adapter_manager = adapter_manager
        
        if save_dir is not None:
            if isinstance(save_dir, str):
                save_dir = Path(save_dir)
            self.save_dir = save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        
        self.use_tensorboard = use_tensorboard
        self.callbacks = callbacks or []
        
        # Modeldeki task_heads ve adapter'ları bul
        self.task_heads, self.adapters = find_task_heads_and_adapters(model)
        
        # Eğitim stratejisini belirle
        self._determine_training_strategy()
        
        logger.info(f"TrainingManager başlatıldı")
        logger.info(f"Bulunan görev başlıkları: {len(self.task_heads)}")
        logger.info(f"Bulunan adapter'lar: {len(self.adapters)}")
    
    def _determine_training_strategy(self) -> None:
        """
        Eğitim stratejisini yapılandırmaya göre belirler.
        """
        # Eğer AdapterTrainingConfig değilse, varsayılan değerlerle bir tane oluştur
        if not isinstance(self.config, AdapterTrainingConfig):
            adapter_config = AdapterTrainingConfig(
                train_adapter_names=None,  # Tüm adapter'ları eğit
                train_task_heads=True,     # Tüm görev başlıklarını eğit
                freeze_core_model=True,    # Çekirdek modeli dondur
                
                # Diğer yapılandırma parametrelerini aktar
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                weight_decay=self.config.weight_decay,
                optimizer=self.config.optimizer,
                scheduler=self.config.scheduler,
                warmup_steps=self.config.warmup_steps,
                gradient_clip_val=self.config.gradient_clip_val,
                device=self.config.device
            )
            self.config = adapter_config
        
        logger.info(f"Eğitim stratejisi: Çekirdek model{'in dondurulması' if self.config.freeze_core_model else 'in tamamının eğitilmesi'}")
        if self.config.train_adapter_names:
            logger.info(f"Eğitilecek adapter'lar: {', '.join(self.config.train_adapter_names)}")
        else:
            logger.info(f"Tüm adapter'lar eğitilecek")
        
        logger.info(f"Görev başlıkları {'eğitilecek' if self.config.train_task_heads else 'dondurulacak'}")
    
    def prepare_model_for_training(self) -> None:
        """
        Modeli eğitim için hazırlar:
        - Çekirdek modeli dondurur (eğer belirtilmişse)
        - Sadece belirli adapter'ları eğitir (eğer belirtilmişse)
        - Görev başlıklarını eğitir/dondurur (yapılandırmaya göre)
        """
        # AdapterTrainingManager kullanarak adapter ve görev başlığı eğitimi için hazırla
        self.adapter_trainer = AdapterTrainingManager(
            model=self.model,
            config=self.config,
            adapter_manager=self.adapter_manager,
            save_dir=self.save_dir,
            use_tensorboard=self.use_tensorboard,
            callbacks=self.callbacks
        )
        
        # Modeli eğitim için hazırla
        self.adapter_trainer.prepare_model_for_training()
        
        # Eğitilebilir parametre sayısını raporla
        trainable_params = self.adapter_trainer.get_trainable_parameter_count()
        logger.info(f"Toplam eğitilebilir parametre: {trainable_params['total']:,}")
    
    def train(
        self,
        train_loader: DataLoader,
        criterion: Optional[Callable] = None,
        val_loader: Optional[DataLoader] = None,
        resume: bool = False
    ) -> Dict[str, float]:
        """
        Modeli eğitir.
        
        Args:
            train_loader: Eğitim veri yükleyicisi
            criterion: Kayıp fonksiyonu (belirtilmezse otomatik olarak oluşturulur)
            val_loader: Doğrulama veri yükleyicisi (opsiyonel)
            resume: Eğitimi bir checkpoint'ten devam ettir
            
        Returns:
            Eğitim sonuçları içeren sözlük
        """
        # Modeli eğitim için hazırla (eğer henüz hazırlanmamışsa)
        if not hasattr(self, 'adapter_trainer'):
            self.prepare_model_for_training()
        
        # Kayıp fonksiyonunu oluştur (belirtilmemişse)
        if criterion is None:
            # Model çıktısına göre uygun kriteri belirle
            # Bu bir tahmin, gerçek uygulamada model çıktı türüne bakılmalı
            if self.task_heads and hasattr(self.task_heads[0], 'task_type'):
                task_type = self.task_heads[0].task_type
                criterion = create_adapter_criterion(task_type)
            else:
                # Varsayılan olarak sınıflandırma kriteri kullan
                criterion = create_adapter_criterion('classification')
        
        # Adapter eğitim yöneticisini kullanarak eğitimi gerçekleştir
        return self.adapter_trainer.train(
            train_loader=train_loader,
            criterion=criterion,
            val_loader=val_loader,
            resume=resume
        )
    
    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Modeli değerlendirir.
        
        Args:
            test_loader: Test veri yükleyicisi
            criterion: Kayıp fonksiyonu (belirtilmezse otomatik olarak oluşturulur)
            
        Returns:
            Değerlendirme sonuçları içeren sözlük
        """
        if not hasattr(self, 'adapter_trainer'):
            self.prepare_model_for_training()
        
        # Kayıp fonksiyonunu oluştur (belirtilmemişse)
        if criterion is None:
            if self.task_heads and hasattr(self.task_heads[0], 'task_type'):
                task_type = self.task_heads[0].task_type
                criterion = create_adapter_criterion(task_type)
            else:
                criterion = create_adapter_criterion('classification')
        
        return self.adapter_trainer.evaluate(
            test_loader=test_loader,
            criterion=criterion
        )
    
    def save_checkpoint(
        self,
        file_path: Optional[Union[str, Path]] = None,
        is_best: bool = False
    ) -> None:
        """
        Model durumunu kaydeder.
        
        Args:
            file_path: Kaydedilecek dosya yolu (belirtilmezse varsayılan ad kullanılır)
            is_best: Bu checkpoint en iyi sonucu veren model mi
        """
        if not hasattr(self, 'adapter_trainer'):
            logger.warning("Model henüz eğitilmedi, checkpoint kaydedilemiyor")
            return
        
        self.adapter_trainer._save_checkpoint(
            optimizer=self.adapter_trainer.optimizer,
            epoch=self.adapter_trainer.current_epoch,
            metrics=self.adapter_trainer.metrics_collector.current_metrics,
            is_best=is_best
        )
    
    def load_checkpoint(
        self,
        file_path: Optional[Union[str, Path]] = None
    ) -> int:
        """
        Model durumunu yükler.
        
        Args:
            file_path: Yüklenecek dosya yolu (belirtilmezse en son checkpoint kullanılır)
            
        Returns:
            Yüklenen checkpoint'in epoch numarası
        """
        if not hasattr(self, 'adapter_trainer'):
            self.prepare_model_for_training()
        
        return self.adapter_trainer._load_checkpoint(
            optimizer=self.adapter_trainer.optimizer,
            file_path=file_path
        )
    
    @staticmethod
    def create(
        model: nn.Module,
        config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
        adapter_manager: Optional[AdapterManager] = None,
        save_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = False,
        use_early_stopping: bool = True,
        use_lr_scheduler: bool = True,
        memory_optimization_level: str = "moderate",
        additional_callbacks: Optional[List[TrainingCallback]] = None
    ) -> 'TrainingManager':
        """
        TrainingManager örneği oluşturur.
        
        Args:
            model: Eğitilecek model
            config: Eğitim yapılandırması (dict olarak verilebilir)
            adapter_manager: AdapterManager örneği (varsa)
            save_dir: Model checkpoint'lerinin kaydedileceği dizin
            use_tensorboard: TensorBoard kullanılsın mı
            use_early_stopping: Erken durdurma kullanılsın mı
            use_lr_scheduler: Öğrenme oranı zamanlayıcısı kullanılsın mı
            memory_optimization_level: Bellek optimizasyon seviyesi
            additional_callbacks: Ek callback'ler
            
        Returns:
            TrainingManager örneği
        """
        # Yapılandırmayı hazırla
        if config is None:
            config = AdapterTrainingConfig()
        elif isinstance(config, dict):
            config = AdapterTrainingConfig(**config)
        
        # Callback'leri oluştur
        callbacks = create_adapter_training_callbacks(
            use_early_stopping=use_early_stopping,
            use_stats_logging=True,
            use_grad_check=True,
            use_memory_tracking=True,
            use_lr_monitor=use_lr_scheduler,
            additional_callbacks=additional_callbacks
        )
        
        # Bellek optimizasyon seviyesini ayarla
        if isinstance(config, AdapterTrainingConfig):
            config.memory_optimization_level = memory_optimization_level
        
        return TrainingManager(
            model=model,
            config=config,
            adapter_manager=adapter_manager,
            save_dir=save_dir,
            use_tensorboard=use_tensorboard,
            callbacks=callbacks
        ) 