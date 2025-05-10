"""
Görev Başlıkları (Task Heads) Modülü

Bu modül, modelin farklı görevleri gerçekleştirmesi için özelleştirilmiş 
görev başlıkları (task heads) içerir.
"""

from .base import TaskHead
from .config import ClassificationHeadConfig, RegressionHeadConfig, MultiLabelHeadConfig, TaskHeadFactory
from .classification import ClassificationHead
from .regression import RegressionHead
from .multi_label import MultiLabelHead

__all__ = [
    'TaskHead',
    'ClassificationHeadConfig',
    'RegressionHeadConfig',
    'MultiLabelHeadConfig',
    'TaskHeadFactory',
    'ClassificationHead',
    'RegressionHead',
    'MultiLabelHead',
]
