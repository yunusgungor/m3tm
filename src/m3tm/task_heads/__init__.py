"""
Görev Başlıkları (Task Heads) Modülü

Bu modül, modelin farklı görevleri gerçekleştirmesi için özelleştirilmiş 
görev başlıkları (task heads) içerir.
"""

from .config import ClassificationHeadConfig, TaskHeadFactory
from .classification import ClassificationHead

__all__ = [
    'ClassificationHeadConfig',
    'TaskHeadFactory',
    'ClassificationHead',
]
