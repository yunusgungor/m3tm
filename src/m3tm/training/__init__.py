"""
M³TM Eğitim Modülü

Bu modül, M³TM modeli için eğitim döngüleri, veri yükleyiciler ve ilgili yardımcı sınıfları içerir.
Farklı görevler için özelleştirilmiş eğitim stratejileri ve döngüleri sağlar.

Örüntüler:
- TrainingLoopTemplate (PT-013): Eğitim döngüsü şablonu
- MetricsCollector (PT-008): Performans metriklerini toplama ve raporlama
"""

from m3tm.training.metrics import (
    MetricsCollector,
    calculate_classification_metrics
)

from m3tm.training.dataset import (
    TextClassificationDataset,
    DatasetFactory
)

from m3tm.training.trainer import (
    TrainingLoopTemplate,
    TextClassificationTrainer
)

from m3tm.training.adapter_training import (
    AdapterTrainingConfig,
    AdapterTrainingManager,
    TrainingCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    create_training_manager
)

from m3tm.training.adapter_training_utils import (
    ModelStatisticsCallback,
    GradientCheckCallback,
    MemoryTrackingCallback,
    LearningRateMonitorCallback,
    create_adapter_training_callbacks,
    create_adapter_criterion,
    find_task_heads_and_adapters
)

__all__ = [
    # Metrics
    'MetricsCollector',
    'calculate_classification_metrics',
    
    # Dataset
    'TextClassificationDataset',
    'DatasetFactory',
    
    # Trainer
    'TrainingLoopTemplate',
    'TextClassificationTrainer',
    
    # Adapter Training
    'AdapterTrainingConfig',
    'AdapterTrainingManager',
    'TrainingCallback',
    'EarlyStoppingCallback', 
    'LearningRateSchedulerCallback',
    'create_training_manager',
    
    # Adapter Training Utils
    'ModelStatisticsCallback',
    'GradientCheckCallback',
    'MemoryTrackingCallback',
    'LearningRateMonitorCallback',
    'create_adapter_training_callbacks',
    'create_adapter_criterion',
    'find_task_heads_and_adapters'
]
