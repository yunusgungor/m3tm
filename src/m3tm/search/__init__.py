"""
M³TM Arama Modülü

Bu modül, M³TM modelinin semantik arama özelliklerini sağlar. Arama gömme projeksiyonu,
indeksleme ve sorgu işleme bileşenlerini içerir.
"""

from .projection import (
    SearchProjectionConfig,
    M3TMSearchProjection,
    SearchProjectionFactory
)

__all__ = [
    'SearchProjectionConfig',
    'M3TMSearchProjection',
    'SearchProjectionFactory'
]
