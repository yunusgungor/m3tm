"""
M³TM modeli için yapılandırma sınıfları.
Tüm model parametreleri ve yapılandırma seçenekleri burada tanımlanır.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TextConfig:
    """Metin işleme modülü için yapılandırma."""
    
    vocab_size: int = 4000  # Küçük dağarcık boyutu
    embed_dim: int = 32
    max_seq_len: int = 512
    pad_token_id: int = 0
    tokenizer_type: str = "sentencepiece"  # "character", "sentencepiece"


@dataclass
class ImageConfig:
    """Görüntü işleme modülü için yapılandırma."""
    
    in_channels: int = 3  # RGB
    patch_size: int = 4  # 4x4 yamalar
    image_size: int = 224  # 224x224 görüntüler
    embed_dim: int = 32


@dataclass
class TransformerConfig:
    """Proto-Transformer blokları için yapılandırma."""
    
    embed_dim: int = 32
    num_heads: int = 1  # Çok az dikkat başlığı
    ffn_hidden_dim: int = 128
    dropout: float = 0.1
    attention_type: str = "lightweight_conv"  # "self_attention", "lightweight_conv"
    conv_kernel_size: int = 3  # Hafif konvolüsyon için
    use_adapter_slots: bool = True


@dataclass
class FusionConfig:
    """Modalite füzyon mekanizması için yapılandırma."""
    
    text_embed_dim: int = 32
    image_embed_dim: int = 32
    fused_embed_dim: int = 64
    fusion_type: str = "simple_concat"  # "simple_concat", "cross_attention", "gated"


@dataclass
class SearchConfig:
    """Arama modülü için yapılandırma."""
    
    search_embed_dim: int = 64
    index_type: str = "brute_force"  # "brute_force", "lsh"
    similarity_metric: str = "cosine"  # "cosine", "dot", "l2"


@dataclass
class AdapterConfig:
    """Adapter modülü için yapılandırma."""
    
    input_dim: int = 32
    bottleneck_dim: int = 8
    activation: str = "hardswish"  # "relu", "gelu", "hardswish"


@dataclass
class M3TMConfig:
    """M³TM modeli için ana yapılandırma sınıfı."""
    
    text_config: TextConfig = TextConfig()
    image_config: ImageConfig = ImageConfig()
    transformer_config: TransformerConfig = TransformerConfig()
    fusion_config: FusionConfig = FusionConfig()
    search_config: SearchConfig = SearchConfig()
    adapter_config: AdapterConfig = AdapterConfig()
    
    num_core_blocks: int = 2
    fused_embed_dim: int = 64
    search_embed_dim: int = 64
    
    # Etkin modaliteler
    use_text: bool = True
    use_image: bool = True
    
    def __post_init__(self):
        """Yapılandırmayı doğrula ve tutarlılığı sağla."""
        # Yapılandırma tutarlılığını sağlamak için basit kontroller
        self.fusion_config.text_embed_dim = self.text_config.embed_dim
        self.fusion_config.image_embed_dim = self.image_config.embed_dim
        self.fusion_config.fused_embed_dim = self.fused_embed_dim
        self.search_config.search_embed_dim = self.search_embed_dim
        
        # Transformer yapılandırmasını modalite embed_dim'leri ile uyumlu hale getir
        # Basitlik için tüm modaliteler aynı embed_dim'i kullanabilir
        self.transformer_config.embed_dim = self.text_config.embed_dim
        
        # Adapter yapılandırmasını güncelle
        self.adapter_config.input_dim = self.transformer_config.embed_dim


def get_default_config() -> M3TMConfig:
    """Varsayılan M³TM yapılandırmasını döndürür."""
    return M3TMConfig()


def get_tiny_config() -> M3TMConfig:
    """Çok küçük boyutlu bir M³TM yapılandırması döndürür (test için)."""
    config = M3TMConfig(
        text_config=TextConfig(
            vocab_size=1000,
            embed_dim=16,
            max_seq_len=128
        ),
        image_config=ImageConfig(
            embed_dim=16,
            patch_size=8,
            image_size=112
        ),
        transformer_config=TransformerConfig(
            embed_dim=16,
            num_heads=1,
            ffn_hidden_dim=64
        ),
        adapter_config=AdapterConfig(
            input_dim=16,
            bottleneck_dim=4
        )
    )
    
    # __post_init__ metodunu elle çağırarak yapılandırma tutarlılığını sağla
    config.__post_init__()
    
    return config 