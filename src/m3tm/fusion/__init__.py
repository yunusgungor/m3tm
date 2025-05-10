"""
M³TM Fusion Module

Bu modül, farklı modaliteleri (metin, görüntü) birleştirmek için kullanılan
füzyon mekanizmalarını sağlar.
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
