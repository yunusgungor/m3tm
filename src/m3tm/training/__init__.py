"""
Eğitim Modülü

Bu modül, M³TM modelinin eğitimi için gerekli sınıfları ve fonksiyonları içerir.
Eğitim döngüleri, optimizasyon stratejileri ve metrik toplama araçları bulunur.
"""

from .metrics import (
    TrainingMetrics,
    MetricsCollector,
    calculate_accuracy,
    calculate_classification_metrics
)

from .dataset import (
    TextClassificationDataset,
    DatasetFactory
)

from .trainer import (
    TrainingLoopTemplate,
    TextClassificationTrainer
)

__all__ = [
    # Metrics
    'TrainingMetrics',
    'MetricsCollector',
    'calculate_accuracy',
    'calculate_classification_metrics',
    
    # Dataset
    'TextClassificationDataset',
    'DatasetFactory',
    
    # Trainer
    'TrainingLoopTemplate',
    'TextClassificationTrainer',
]
