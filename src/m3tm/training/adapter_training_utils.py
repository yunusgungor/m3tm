"""
Adapter ve Görev Başlığı Eğitimi için Yardımcı Fonksiyonlar

Bu modül, adapter ve görev başlığı eğitimi için yardımcı fonksiyonlar ve
özelleştirilmiş bileşenler içerir.

Örüntüler:
- FactoryMethod (PT-002): Çeşitli alt bileşenler oluşturma
- DecoratorPattern (PT-017): İşlevselliği dinamik olarak genişletme
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Callable, Tuple, List, Set, Type

import torch
import torch.nn as nn

from m3tm.adapters.adapter import Adapter
from m3tm.adapters.adapter_manager import AdapterManager
from m3tm.task_heads.base import TaskHead
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.training.adapter_training import (
    AdapterTrainingConfig, AdapterTrainingManager, TrainingCallback, 
    EarlyStoppingCallback, LearningRateSchedulerCallback
)

logger = logging.getLogger(__name__)


class ModelStatisticsCallback(TrainingCallback):
    """
    Model istatistiklerini toplayan ve loglayan callback.
    """
    def __init__(self, log_frequency: int = 1):
        """
        Args:
            log_frequency: Kaç epoch'ta bir istatistiklerin loglanacağı
        """
        self.log_frequency = log_frequency
    
    def on_epoch_end(
        self, 
        trainer: AdapterTrainingManager, 
        epoch: int, 
        metrics: Dict[str, float], 
        **kwargs
    ) -> None:
        """Her epoch sonunda model istatistiklerini loglar."""
        if (epoch + 1) % self.log_frequency != 0:
            return
        
        # Eğitilebilir parametre sayılarını al
        trainable_params = trainer.get_trainable_parameter_count()
        
        logger.info(f"Model istatistikleri (Epoch {epoch+1}):")
        logger.info(f"  Toplam eğitilebilir parametre: {trainable_params['total']:,}")
        
        # Adapter ve görev başlıklarına göre grupla
        adapter_params = {k: v for k, v in trainable_params.items() if k.startswith("adapter.")}
        task_head_params = {k: v for k, v in trainable_params.items() if k.startswith("task_head.")}
        
        if adapter_params:
            logger.info(f"  Adapter parametreleri:")
            for name, count in adapter_params.items():
                logger.info(f"    {name}: {count:,}")
        
        if task_head_params:
            logger.info(f"  Görev başlığı parametreleri:")
            for name, count in task_head_params.items():
                logger.info(f"    {name}: {count:,}")


class GradientCheckCallback(TrainingCallback):
    """
    Gradient değerlerini kontrol eden ve aşırı/sıfır gradient problemlerini tespit eden callback.
    """
    def __init__(
        self, 
        check_frequency: int = 10, 
        log_histogram: bool = False,
        gradient_threshold: float = 10.0,
        zero_fraction_threshold: float = 0.9
    ):
        """
        Args:
            check_frequency: Kaç batch'te bir gradient kontrolü yapılacağı
            log_histogram: Gradient histogramını logla
            gradient_threshold: Aşırı gradient eşik değeri
            zero_fraction_threshold: Sıfır gradient oranı eşik değeri
        """
        self.check_frequency = check_frequency
        self.log_histogram = log_histogram
        self.gradient_threshold = gradient_threshold
        self.zero_fraction_threshold = zero_fraction_threshold
        self.batch_count = 0
    
    def on_batch_end(self, trainer: AdapterTrainingManager, batch_metrics: Dict[str, float], **kwargs) -> None:
        """Her batch sonunda gradientleri kontrol eder."""
        self.batch_count += 1
        if self.batch_count % self.check_frequency != 0:
            return
        
        # Eğitilebilir parametreler ve gradientleri al
        for name, param in trainer.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            
            # Gradient değerlerini kontrol et
            grad_data = param.grad.data
            
            # İstatistikler
            max_grad = grad_data.abs().max().item()
            mean_grad = grad_data.abs().mean().item()
            zero_fraction = (grad_data.abs() < 1e-8).float().mean().item()
            
            # Aşırı gradient kontrolü
            if max_grad > self.gradient_threshold:
                logger.warning(f"Aşırı gradient tespit edildi: {name}, max={max_grad:.4f}, mean={mean_grad:.4f}")
            
            # Sıfır gradient kontrolü
            if zero_fraction > self.zero_fraction_threshold:
                logger.warning(f"Yüksek sıfır gradient oranı: {name}, zero_fraction={zero_fraction:.4f}")
            
            # Histogram loglama
            if self.log_histogram and hasattr(trainer.metrics_collector, "writer") and trainer.metrics_collector.writer:
                trainer.metrics_collector.writer.add_histogram(
                    f"gradients/{name}", 
                    grad_data.cpu().numpy(), 
                    trainer.global_step
                )


class MemoryTrackingCallback(TrainingCallback):
    """
    Bellek kullanımını takip eden callback.
    """
    def __init__(self, log_frequency: int = 10):
        """
        Args:
            log_frequency: Kaç batch'te bir bellek kullanımını loglanacağı
        """
        self.log_frequency = log_frequency
        self.batch_count = 0
    
    def on_batch_end(self, trainer: AdapterTrainingManager, batch_metrics: Dict[str, float], **kwargs) -> None:
        """Her batch sonunda bellek kullanımını kontrol eder."""
        self.batch_count += 1
        if self.batch_count % self.log_frequency != 0:
            return
        
        # GPU bellek kullanımını al
        if trainer.device.type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(trainer.device) / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved(trainer.device) / (1024 ** 2)  # MB
            max_allocated = torch.cuda.max_memory_allocated(trainer.device) / (1024 ** 2)  # MB
            
            logger.info(f"GPU Bellek - Ayrılan: {allocated:.2f} MB, "
                       f"Rezerve: {reserved:.2f} MB, "
                       f"Maksimum: {max_allocated:.2f} MB")
            
            # TensorBoard'a yaz
            if hasattr(trainer.metrics_collector, "writer") and trainer.metrics_collector.writer:
                trainer.metrics_collector.writer.add_scalar(
                    "memory/gpu_allocated_mb", 
                    allocated, 
                    trainer.global_step
                )
                trainer.metrics_collector.writer.add_scalar(
                    "memory/gpu_reserved_mb", 
                    reserved, 
                    trainer.global_step
                )


class LearningRateMonitorCallback(TrainingCallback):
    """
    Öğrenme oranını takip eden callback.
    """
    def __init__(self, log_frequency: int = 10):
        """
        Args:
            log_frequency: Kaç batch'te bir öğrenme oranını loglanacağı
        """
        self.log_frequency = log_frequency
        self.batch_count = 0
    
    def on_batch_end(self, trainer: AdapterTrainingManager, batch_metrics: Dict[str, float], **kwargs) -> None:
        """Her batch sonunda öğrenme oranını kontrol eder."""
        self.batch_count += 1
        if self.batch_count % self.log_frequency != 0:
            return
        
        # Mevcut optimizer'ı bul
        optimizer = None
        for callback in trainer.callbacks:
            if isinstance(callback, LearningRateSchedulerCallback):
                # Scheduler'dan optimizer'a eriş
                if hasattr(callback.scheduler, "optimizer"):
                    optimizer = callback.scheduler.optimizer
                    break
        
        if optimizer is None:
            return
        
        # Öğrenme oranlarını al
        lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        if not lrs:
            return
        
        # Ortalama öğrenme oranını logla
        avg_lr = sum(lrs) / len(lrs)
        logger.debug(f"Öğrenme oranı: {avg_lr:.8f}")
        
        # TensorBoard'a yaz
        if hasattr(trainer.metrics_collector, "writer") and trainer.metrics_collector.writer:
            trainer.metrics_collector.writer.add_scalar(
                "learning_rate", 
                avg_lr, 
                trainer.global_step
            )


def create_adapter_training_callbacks(
    use_early_stopping: bool = True,
    use_stats_logging: bool = True,
    use_grad_check: bool = False,
    use_memory_tracking: bool = False,
    use_lr_monitor: bool = True,
    early_stopping_config: Optional[Dict[str, Any]] = None,
    stats_config: Optional[Dict[str, Any]] = None,
    grad_check_config: Optional[Dict[str, Any]] = None,
    memory_config: Optional[Dict[str, Any]] = None,
    lr_monitor_config: Optional[Dict[str, Any]] = None,
    additional_callbacks: Optional[List[TrainingCallback]] = None
) -> List[TrainingCallback]:
    """
    Adapter eğitimi için callback'ler oluşturur.
    
    Args:
        use_early_stopping: Erken durdurma kullanılsın mı?
        use_stats_logging: İstatistik loglama kullanılsın mı?
        use_grad_check: Gradient kontrolü kullanılsın mı?
        use_memory_tracking: Bellek takibi kullanılsın mı?
        use_lr_monitor: Öğrenme oranı takibi kullanılsın mı?
        early_stopping_config: Erken durdurma yapılandırması
        stats_config: İstatistik yapılandırması
        grad_check_config: Gradient kontrol yapılandırması
        memory_config: Bellek takip yapılandırması
        lr_monitor_config: Öğrenme oranı takip yapılandırması
        additional_callbacks: Ek callback'ler
        
    Returns:
        List[TrainingCallback]: Callback'ler listesi
    """
    callbacks = []
    
    # Erken durdurma
    if use_early_stopping:
        early_stopping_params = early_stopping_config or {}
        callbacks.append(EarlyStoppingCallback(**early_stopping_params))
    
    # İstatistik loglama
    if use_stats_logging:
        stats_params = stats_config or {}
        callbacks.append(ModelStatisticsCallback(**stats_params))
    
    # Gradient kontrolü
    if use_grad_check:
        grad_check_params = grad_check_config or {}
        callbacks.append(GradientCheckCallback(**grad_check_params))
    
    # Bellek takibi
    if use_memory_tracking:
        memory_params = memory_config or {}
        callbacks.append(MemoryTrackingCallback(**memory_params))
    
    # Öğrenme oranı takibi
    if use_lr_monitor:
        lr_monitor_params = lr_monitor_config or {}
        callbacks.append(LearningRateMonitorCallback(**lr_monitor_params))
    
    # Ek callback'ler
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    return callbacks


def create_adapter_criterion(
    task_type: str,
    loss_config: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Belirtilen görev türüne uygun kayıp fonksiyonu oluşturur.
    
    Args:
        task_type: Görev türü ("classification", "regression", "multi_label")
        loss_config: Kayıp fonksiyonu yapılandırması
        
    Returns:
        Callable: Kayıp fonksiyonu
    """
    loss_config = loss_config or {}
    
    if task_type == "classification":
        weight = None
        if "class_weights" in loss_config:
            weight = torch.tensor(loss_config["class_weights"])
        
        def criterion(outputs, inputs):
            return nn.CrossEntropyLoss(weight=weight)(outputs["logits"], inputs["labels"])
        
        return criterion
    
    elif task_type == "regression":
        def criterion(outputs, inputs):
            return nn.MSELoss()(outputs["logits"], inputs["labels"])
        
        return criterion
    
    elif task_type == "multi_label":
        def criterion(outputs, inputs):
            return nn.BCEWithLogitsLoss()(outputs["logits"], inputs["labels"])
        
        return criterion
    
    else:
        raise ValueError(f"Bilinmeyen görev türü: {task_type}")


def find_task_heads_and_adapters(model: nn.Module) -> Tuple[List[TaskHead], List[Adapter]]:
    """
    Modeldeki görev başlıklarını ve adapter'ları bulur.
    
    Args:
        model: Model
        
    Returns:
        Tuple[List[TaskHead], List[Adapter]]: Görev başlıkları ve adapter'lar
    """
    task_heads = []
    adapters = []
    
    for module in model.modules():
        if isinstance(module, TaskHead):
            task_heads.append(module)
        elif isinstance(module, Adapter):
            adapters.append(module)
    
    return task_heads, adapters


def create_training_example():
    """
    Adapter eğitimi için örnek kod akışı.
    
    Not: Bu fonksiyon doğrudan çalıştırılmak için değil, kullanıcılara örnek teşkil etmesi için eklenmiştir.
    """
    # Örnek model oluşturma (gerçekte kullanıcı kendi modelini sağlayacak)
    from m3tm.transformer.proto_transformer import ProtoTransformerBlock, ProtoTransformerConfig
    from m3tm.embedding.text_embedding import TextEmbedding, TextEmbeddingConfig
    from m3tm.task_heads.classification import ClassificationHead, ClassificationHeadConfig
    from m3tm.adapters.adapter import BottleneckAdapter, AdapterConfig, AdapterType
    from m3tm.adapters.adapter_manager import AdapterManager
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    
    # Örnek model oluşturma
    text_config = TextEmbeddingConfig(vocab_size=10000, embed_dim=64)
    text_embedding = TextEmbedding(text_config)
    
    transformer_config = ProtoTransformerConfig(embed_dim=64, num_heads=2, ffn_hidden_dim=128)
    transformer_block = ProtoTransformerBlock(transformer_config)
    
    # Adapter oluşturma ve ekleme
    adapter_config = AdapterConfig(adapter_type=AdapterType.BOTTLENECK, bottleneck_dim=16)
    adapter = BottleneckAdapter(adapter_config, input_dim=64)
    
    # Adapter'ı transformer bloğuna ekle
    adapter_manager = AdapterManager()
    adapter_manager.add_adapter(transformer_block, "adapter1", adapter)
    
    # Görev başlığı oluşturma
    head_config = ClassificationHeadConfig(input_dim=64, num_classes=10, hidden_dim=32)
    classification_head = ClassificationHead(head_config)
    
    # Model kompozisyonu
    class TextModel(nn.Module):
        def __init__(self, embedding, transformer, head):
            super().__init__()
            self.embedding = embedding
            self.transformer = transformer
            self.head = head
        
        def forward(self, input_ids, labels=None):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            outputs = self.head(x)
            return outputs
    
    model = TextModel(text_embedding, transformer_block, classification_head)
    
    # Örnek veri oluşturma
    batch_size = A
    x = torch.randint(0, 10000, (100, 20))  # (num_samples, seq_len)
    y = torch.randint(0, 10, (100,))  # (num_samples,)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Eğitim yapılandırması
    train_config = AdapterTrainingConfig(
        train_adapter_names=["adapter1"],
        train_task_heads=True,
        freeze_core_model=True,
        learning_rate=1e-3,
        num_epochs=5,
        batch_size=batch_size,
        checkpoint_frequency=1,
        memory_optimization_level="moderate"
    )
    
    # Callback'ler oluşturma
    callbacks = create_adapter_training_callbacks(
        use_early_stopping=True,
        use_stats_logging=True,
        early_stopping_config={"patience": 3, "monitor": "val_loss"}
    )
    
    # Eğitim yöneticisi oluşturma
    training_manager = AdapterTrainingManager(
        model=model,
        config=train_config,
        adapter_manager=adapter_manager,
        callbacks=callbacks
    )
    
    # Kayıp fonksiyonu oluşturma
    criterion = create_adapter_criterion("classification")
    
    # Eğitimi başlat
    training_manager.train(train_loader, criterion) 