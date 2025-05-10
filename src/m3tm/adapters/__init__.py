"""
M³TM Adapter Modülü.

Bu modül, M³TM modeli için adapter mekanizmalarını içerir.
Adapter'lar, çekirdek modelin üzerine küçük, eğitilebilir parametreler ekleyerek
modeli özelleştirmeyi sağlar, ana modeli değiştirmeden.

Örüntüler:
- ModelComposite (PT-003): Ana modelin içine küçük ve özelleştirilmiş modül ekleme
"""

from m3tm.adapters.adapter import Adapter, AdapterConfig, BottleneckAdapter, ParallelAdapter
from m3tm.adapters.adapter_manager import AdapterManager, AdapterRegistration
from m3tm.adapters.adapter_utils import create_adapter_for_module, get_adapter_positions

__all__ = [
    "Adapter",
    "AdapterConfig",
    "BottleneckAdapter",
    "ParallelAdapter",
    "AdapterManager",
    "AdapterRegistration",
    "create_adapter_for_module",
    "get_adapter_positions"
]
