"""
Yapılandırma modülü için birim testler.
"""

import pytest

from m3tm.config.model_config import (
    AdapterConfig,
    FusionConfig,
    ImageConfig,
    M3TMConfig,
    SearchConfig,
    TextConfig,
    TransformerConfig,
    get_default_config,
    get_tiny_config,
)


def test_default_config():
    """Varsayılan yapılandırmanın doğru değerlere sahip olduğunu kontrol eder."""
    config = get_default_config()
    
    # M3TMConfig için temel kontroller
    assert config.num_core_blocks == 2
    assert config.fused_embed_dim == 64
    assert config.search_embed_dim == 64
    assert config.use_text is True
    assert config.use_image is True
    
    # Alt yapılandırmalar için kontroller
    assert config.text_config.vocab_size == 4000
    assert config.text_config.embed_dim == 32
    assert config.image_config.embed_dim == 32
    assert config.transformer_config.embed_dim == 32  # __post_init__ tarafından güncellenmeli
    
    # Tutarlılık kontrolü
    assert config.fusion_config.text_embed_dim == config.text_config.embed_dim
    assert config.fusion_config.image_embed_dim == config.image_config.embed_dim
    assert config.fusion_config.fused_embed_dim == config.fused_embed_dim
    assert config.search_config.search_embed_dim == config.search_embed_dim
    assert config.adapter_config.input_dim == config.transformer_config.embed_dim


def test_tiny_config():
    """Küçük yapılandırmanın doğru değerlere sahip olduğunu kontrol eder."""
    config = get_tiny_config()
    
    # Küçük yapılandırma için özel değerler
    assert config.text_config.vocab_size == 1000
    assert config.text_config.embed_dim == 16
    assert config.text_config.max_seq_len == 128
    assert config.image_config.embed_dim == 16
    assert config.image_config.patch_size == 8
    assert config.image_config.image_size == 112
    assert config.transformer_config.embed_dim == 16
    assert config.transformer_config.ffn_hidden_dim == 64
    assert config.adapter_config.input_dim == 16
    assert config.adapter_config.bottleneck_dim == 4
    
    # Tutarlılık kontrolü
    assert config.fusion_config.text_embed_dim == config.text_config.embed_dim
    assert config.fusion_config.image_embed_dim == config.image_config.embed_dim
    assert config.adapter_config.input_dim == config.transformer_config.embed_dim


def test_config_post_init():
    """__post_init__ metodunun yapılandırma tutarlılığını sağladığını kontrol eder."""
    # Tutarsız embed_dim değerleri ile yapılandırma oluştur
    text_config = TextConfig(embed_dim=24)
    transformer_config = TransformerConfig(embed_dim=32)
    adapter_config = AdapterConfig(input_dim=48)
    
    # Yapılandırma
    config = M3TMConfig(
        text_config=text_config,
        transformer_config=transformer_config,
        adapter_config=adapter_config
    )
    
    # __post_init__ çağrısından sonra tutarlılık kontrolü
    assert config.transformer_config.embed_dim == config.text_config.embed_dim == 24
    assert config.adapter_config.input_dim == config.transformer_config.embed_dim == 24
    assert config.fusion_config.text_embed_dim == config.text_config.embed_dim == 24 