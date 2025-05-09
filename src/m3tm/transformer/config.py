"""
M³TM Transformer modülü için yapılandırma sınıfları.

Bu modül, ProtoTransformerBlock ve ilgili alt bileşenler için yapılandırma sınıflarını içerir.

Örüntüler:
- ConfigurationDataclass (PT-001): Yapılandırma parametrelerini dataclass olarak modelleme
- ConfigValidationPipeline (PT-009): Yapılandırma doğrulama adımlarını bir boru hattında birleştirme
- ConfigurationComposite (PT-012): Alt yapılandırma bileşenlerini hiyerarşik olarak organize etme
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Tuple, Any

import torch

from m3tm.config.config_base import ConfigBase, validate_positive_integer, validate_probability


@dataclass
class AttentionConfig(ConfigBase):
    """ProtoTransformerBlock içindeki dikkat mekanizması yapılandırması."""
    
    mechanism_name: str = "MobileAttention"  # Dikkat mekanizması adı
    head_dim: int = 32  # Dikkat başı boyutu
    num_heads: int = 4  # Dikkat başı sayısı
    dropout: float = 0.1  # Dropout oranı
    use_causal_mask: bool = False  # Nedensel maskeleme kullanımı
    
    # Mekanizma özel parametreleri
    mechanism_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Yapılandırma parametrelerini doğrula."""
        is_valid = True
        is_valid &= validate_positive_integer(self, "head_dim")
        is_valid &= validate_positive_integer(self, "num_heads")
        is_valid &= validate_probability(self, "dropout")
        
        # Dikkat mekanizması adı boş olmamalı
        if not self.mechanism_name:
            self.validation_errors.append("mechanism_name must not be empty")
            is_valid = False
            
        return is_valid
        

@dataclass
class FeedForwardConfig(ConfigBase):
    """ProtoTransformerBlock içindeki feed-forward network yapılandırması."""
    
    mechanism_name: str = "MobileFFN"  # FFN mekanizması adı
    expansion_factor: float = 2.0  # Genişleme faktörü
    hidden_act: str = "gelu"  # Gizli aktivasyon fonksiyonu
    dropout: float = 0.1  # Dropout oranı
    
    # Mekanizma özel parametreleri
    mechanism_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Yapılandırma parametrelerini doğrula."""
        is_valid = True
        
        # Genişleme faktörü pozitif olmalı
        if self.expansion_factor <= 0:
            self.validation_errors.append(f"expansion_factor must be positive, got {self.expansion_factor}")
            is_valid = False
            
        is_valid &= validate_probability(self, "dropout")
        
        # Mekanizma adı boş olmamalı
        if not self.mechanism_name:
            self.validation_errors.append("mechanism_name must not be empty")
            is_valid = False
            
        return is_valid


@dataclass
class AdapterConfig(ConfigBase):
    """Adapter yuva yapılandırması (temel, ileride genişletilebilir)."""
    
    enabled: bool = False  # Adapter'ların etkinleştirilip etkinleştirilmediği
    bottleneck_dim: int = 8  # Darboğaz boyutu
    adapter_positions: List[str] = field(default_factory=lambda: ["post_attention", "post_ffn"])
    
    def validate(self) -> bool:
        """Yapılandırma parametrelerini doğrula."""
        is_valid = True
        
        if self.enabled:
            is_valid &= validate_positive_integer(self, "bottleneck_dim")
            
            # Geçerli adapter pozisyonları
            valid_positions = ["pre_attention", "post_attention", "pre_ffn", "post_ffn"]
            for pos in self.adapter_positions:
                if pos not in valid_positions:
                    self.validation_errors.append(f"Invalid adapter position: {pos}")
                    is_valid = False
                    
        return is_valid


@dataclass
class ProtoTransformerConfig(ConfigBase):
    """ProtoTransformerBlock için ana yapılandırma sınıfı."""
    
    # Genel parametreler
    hidden_size: int = 256  # Gizli durum boyutu
    intermediate_size: int = 512  # Ara boyut (FFN için)
    layer_norm_eps: float = 1e-12  # LayerNorm epsilon değeri
    pre_layer_norm: bool = True  # Pre-LN mimarisi kullanımı
    dropout: float = 0.1  # Genel dropout oranı
    
    # Alt yapılandırmalar (ConfigurationComposite)
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    ffn_config: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)
    
    # Hesaplama optimizasyonları
    use_flash_attention: bool = False  # Flash Attention kullanımı
    fuse_operations: bool = True  # Uygun olduğunda operasyonları birleştirme
    
    def validate(self) -> bool:
        """Yapılandırma parametrelerini doğrula."""
        is_valid = True
        
        # Temel parametreleri doğrula
        is_valid &= validate_positive_integer(self, "hidden_size")
        is_valid &= validate_positive_integer(self, "intermediate_size")
        is_valid &= validate_probability(self, "dropout")
        
        if self.layer_norm_eps <= 0:
            self.validation_errors.append(f"layer_norm_eps must be positive, got {self.layer_norm_eps}")
            is_valid = False
            
        # Alt yapılandırmaları doğrula
        if not self.attention_config.validate():
            self.validation_errors.append(f"Invalid attention_config: {self.attention_config.validation_errors}")
            is_valid = False
            
        if not self.ffn_config.validate():
            self.validation_errors.append(f"Invalid ffn_config: {self.ffn_config.validation_errors}")
            is_valid = False
            
        if not self.adapter_config.validate():
            self.validation_errors.append(f"Invalid adapter_config: {self.adapter_config.validation_errors}")
            is_valid = False
            
        return is_valid 