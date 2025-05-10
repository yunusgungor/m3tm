"""
M³TM transformer modülü için yapılandırma sınıfları.

Bu modül, M³TM transformer modelleri için yapılandırma sınıflarını içerir.

Örüntüler:
- ConfigurationDataclass (PT-001): Yapılandırma parametrelerini dataclass olarak modelleme
- ConfigValidationPipeline (PT-009): Yapılandırma doğrulama adımlarını bir boru hattında birleştirme
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

from m3tm.config.config_base import ConfigBase, validate_positive_integer, validate_probability


@dataclass
class AdapterConfig(ConfigBase):
    """
    Transformer adapter yapılandırması.
    
    Örüntüler:
    - ConfigurationDataclass (PT-001): Yapılandırma parametrelerini dataclass olarak modelleme
    """
    
    enabled: bool = False  # Adapter etkinleştirme bayrağı
    bottleneck_dim: int = 16  # Darboğaz boyutu
    use_layer_norm: bool = True  # Layer normalization kullanımı
    adapter_type: str = "bottleneck"  # Adapter türü
    init_scale: float = 1e-3  # Başlangıç ölçeği
    activation: str = "gelu"  # Aktivasyon fonksiyonu
    dropout: float = 0.0  # Dropout oranı
    residual_connection: bool = True  # Artık bağlantı kullanımı
    alpha: float = 1.0  # Paralel adapter için ölçekleme faktörü (adapter_type = "parallel" için)
    adapter_positions: List[str] = field(default_factory=lambda: ["post_attention", "post_ffn"])
    additional_params: Dict[str, Any] = field(default_factory=dict)  # Ek parametreler
    
    def validate(self) -> bool:
        """Adapter yapılandırma parametrelerini doğrular."""
        is_valid = True
        
        # Sadece adapter etkinleştirilmişse diğer parametreleri kontrol et
        if self.enabled:
            is_valid &= validate_positive_integer(self, "bottleneck_dim")
            is_valid &= validate_probability(self, "dropout")
            
            # Alpha değeri pozitif olmalı
            if self.alpha <= 0:
                self.validation_errors.append(f"alpha must be positive, got {self.alpha}")
                is_valid = False
            
            # Adapter türleri kontrol et
            valid_types = ["bottleneck", "parallel", "series", "scaled", "prefix"]
            if self.adapter_type not in valid_types:
                self.validation_errors.append(f"adapter_type must be one of {valid_types}, got {self.adapter_type}")
                is_valid = False
                
            # Pozisyonları kontrol et
            valid_positions = ["pre_attention", "post_attention", "pre_ffn", "post_ffn"]
            for pos in self.adapter_positions:
                if pos not in valid_positions:
                    self.validation_errors.append(f"invalid adapter position: {pos}")
                    is_valid = False
                    
            # Aktivasyon fonksiyonları kontrol et
            valid_activations = ["relu", "gelu", "swish", "silu", "hardswish", "leaky_relu"]
            if self.activation not in valid_activations:
                self.validation_errors.append(f"activation must be one of {valid_activations}, got {self.activation}")
                is_valid = False
                
        return is_valid


@dataclass
class AttentionConfig(ConfigBase):
    """
    Dikkat mekanizması yapılandırması.
    """
    
    num_heads: int = 8  # Dikkat başlığı sayısı
    head_dim: Optional[int] = None  # Başlık boyutu (None ise hidden_size / num_heads)
    dropout: float = 0.1  # Dikkat dropout oranı
    attention_type: str = "scaled_dot_product"  # Dikkat tipi
    use_rotary_embeddings: bool = False  # Rotary pozisyon gömme kullanımı
    max_position_embeddings: int = 512  # Maksimum pozisyon gömme sayısı
    use_bias: bool = True  # Projeksiyon katmanında bias kullanımı
    
    def validate(self) -> bool:
        """Dikkat yapılandırma parametrelerini doğrular."""
        is_valid = True
        
        is_valid &= validate_positive_integer(self, "num_heads")
        
        if self.head_dim is not None:
            is_valid &= validate_positive_integer(self, "head_dim")
            
        is_valid &= validate_probability(self, "dropout")
        
        valid_attention_types = ["scaled_dot_product", "linear", "local", "longformer"]
        if self.attention_type not in valid_attention_types:
            self.validation_errors.append(f"attention_type must be one of {valid_attention_types}, got {self.attention_type}")
            is_valid = False
            
        return is_valid


@dataclass
class FeedForwardConfig(ConfigBase):
    """
    Feed-forward ağ yapılandırması.
    """
    
    intermediate_dim: Optional[int] = None  # Ara boyut (None ise hidden_size * 4)
    activation: str = "gelu"  # Aktivasyon fonksiyonu
    dropout: float = 0.1  # FFN dropout oranı
    use_bias: bool = True  # Katmanlarda bias kullanımı
    gated: bool = False  # Gated FFN kullanımı
    
    def validate(self) -> bool:
        """Feed-forward yapılandırma parametrelerini doğrular."""
        is_valid = True
        
        if self.intermediate_dim is not None:
            is_valid &= validate_positive_integer(self, "intermediate_dim")
            
        is_valid &= validate_probability(self, "dropout")
        
        valid_activations = ["relu", "gelu", "swish", "silu", "hardswish", "leaky_relu"]
        if self.activation not in valid_activations:
            self.validation_errors.append(f"activation must be one of {valid_activations}, got {self.activation}")
            is_valid = False
            
        return is_valid


@dataclass
class ProtoTransformerConfig(ConfigBase):
    """
    Prototip transformer model yapılandırması.
    """
    
    hidden_size: int = 768  # Gizli durum boyutu
    num_layers: int = 12  # Transformer blok sayısı
    max_sequence_length: int = 512  # Maksimum dizi uzunluğu
    vocab_size: int = 30000  # Kelime dağarcığı boyutu
    dropout: float = 0.1  # Genel dropout oranı
    layer_norm_epsilon: float = 1e-12  # LayerNorm epsilon değeri
    embedding_dim: Optional[int] = None  # Gömme boyutu (None ise hidden_size)
    pre_layer_norm: bool = True  # Pre-LayerNorm kullanımı
    use_flash_attention: bool = False  # Flash Attention kullanımı
    fuse_operations: bool = False  # Operasyonları birleştirme
    layer_norm_eps: float = 1e-12  # LayerNorm epsilon değeri (geriye uyumluluk için)
    
    # Alt yapılandırmalar
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    ffn_config: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)
    
    # Opsiyonel parametreler
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Başlangıç sonrası başlatma."""
        # Varsayılan değerleri doldur
        if self.embedding_dim is None:
            self.embedding_dim = self.hidden_size
            
        if self.attention_config.head_dim is None:
            self.attention_config.head_dim = self.hidden_size // self.attention_config.num_heads
            
        if self.ffn_config.intermediate_dim is None:
            self.ffn_config.intermediate_dim = self.hidden_size * 4
            
        # Geriye uyumluluk için layer_norm_epsilon'u layer_norm_eps'e eşitle
        self.layer_norm_eps = self.layer_norm_epsilon
    
    def validate(self) -> bool:
        """Tüm yapılandırma parametrelerini doğrular."""
        is_valid = True
        
        # Ana parametreler
        is_valid &= validate_positive_integer(self, "hidden_size")
        is_valid &= validate_positive_integer(self, "num_layers")
        is_valid &= validate_positive_integer(self, "max_sequence_length")
        is_valid &= validate_positive_integer(self, "vocab_size")
        is_valid &= validate_probability(self, "dropout")
        
        # Embedding boyutunu kontrol et
        is_valid &= validate_positive_integer(self, "embedding_dim")
        
        # Alt yapılandırmaları doğrula
        if not self.attention_config.validate():
            self.validation_errors.extend([f"attention_config.{error}" for error in self.attention_config.validation_errors])
            is_valid = False
            
        if not self.ffn_config.validate():
            self.validation_errors.extend([f"ffn_config.{error}" for error in self.ffn_config.validation_errors])
            is_valid = False
            
        if not self.adapter_config.validate():
            self.validation_errors.extend([f"adapter_config.{error}" for error in self.adapter_config.validation_errors])
            is_valid = False
            
        # Tutarlılık kontrolleri
        if self.attention_config.num_heads > 0 and self.hidden_size % self.attention_config.num_heads != 0:
            self.validation_errors.append(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.attention_config.num_heads})")
            is_valid = False
            
        return is_valid 