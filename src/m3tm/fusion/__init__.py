"""
Füzyon Modülü

Bu modül, farklı modalitelerin (metin, görüntü) birleştirilmesi için mekanizmalar sağlar.
"""

from .config import (
    FusionConfig,
    ConcatenationFusionConfig,
    WeightedSumFusionConfig,
    GatedFusionConfig,
    FusionType
)
from .base_fusion import BaseFusion
from .fusion_strategies import (
    ConcatenationFusion,
    WeightedSumFusion,
    GatedFusion
)
from .factory import FusionFactory

__all__ = [
    'FusionConfig',
    'ConcatenationFusionConfig',
    'WeightedSumFusionConfig',
    'GatedFusionConfig',
    'FusionType',
    'BaseFusion',
    'ConcatenationFusion',
    'WeightedSumFusion',
    'GatedFusion',
    'FusionFactory'
]
