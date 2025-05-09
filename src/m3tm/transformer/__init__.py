"""
M³TM Transformer modülü.

Bu modül, M³TM modelinin çekirdek Transformer bileşenlerini içerir.
"""

from m3tm.transformer.config import (
    ProtoTransformerConfig,
    AttentionConfig,
    FeedForwardConfig,
    AdapterConfig
)

from m3tm.transformer.ffn import (
    get_ffn_mechanism,
    list_available_ffn_mechanisms,
    StandardFFN,
    MobileFFN,
    GhostFFN
)

from m3tm.transformer.adapter import (
    AdapterSlot,
    Adapter,
    ParallelAdapter,
    create_adapter_slots
)

from m3tm.transformer.proto_transformer import (
    ProtoTransformerBlock,
    ProtoTransformer
)


__all__ = [
    # Yapılandırma
    'ProtoTransformerConfig',
    'AttentionConfig',
    'FeedForwardConfig',
    'AdapterConfig',
    
    # FFN
    'get_ffn_mechanism',
    'list_available_ffn_mechanisms',
    'StandardFFN',
    'MobileFFN',
    'GhostFFN',
    
    # Adapter
    'AdapterSlot',
    'Adapter',
    'ParallelAdapter',
    'create_adapter_slots',
    
    # Transformer
    'ProtoTransformerBlock',
    'ProtoTransformer',
]
