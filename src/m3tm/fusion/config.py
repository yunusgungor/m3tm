"""
Füzyon Mekanizmaları Yapılandırma Modülü

Bu modül, farklı modaliteleri (metin, görüntü) birleştirmek için kullanılan
füzyon mekanizmalarının yapılandırma sınıflarını içerir.

Örüntü: ConfigurationDataclass (PT-001), ConfigurationComposite (PT-012)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union


class FusionType(Enum):
    """Desteklenen füzyon türleri."""
    CONCATENATION = "concatenation"
    WEIGHTED_SUM = "weighted_sum"
    GATED = "gated"
    CROSS_ATTENTION = "cross_attention"
    ADAPTIVE_WEIGHTING = "adaptive_weighting"


@dataclass
class FusionConfig:
    """
    Temel füzyon mekanizması yapılandırması.
    
    Örüntü: ConfigurationDataclass (PT-001)
    """
    # Füzyon tipi
    fusion_type: FusionType = FusionType.CONCATENATION
    
    # Boyut parametreleri (girdi ve çıktı)
    text_dim: Optional[int] = None  # Metin girdi boyutu (dinamik olarak ayarlanabilir)
    image_dim: Optional[int] = None  # Görüntü girdi boyutu (dinamik olarak ayarlanabilir)
    output_dim: Optional[int] = None  # Çıktı boyutu (füzyon tipine göre hesaplanabilir)
    
    # Normalizasyon ve dropout ayarları
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-12
    dropout_rate: float = 0.1
    
    # İlklendirme standart sapması
    init_std: float = 0.02
    
    # Ek parametreler
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Yapılandırma doğrulamasını gerçekleştirir."""
        # Örüntü: ConfigValidationPipeline (PT-009)
        self._validate_fusion_type()
        self._validate_dimensions()
    
    def _validate_fusion_type(self):
        """Füzyon tipi doğrulaması."""
        if not isinstance(self.fusion_type, FusionType):
            try:
                self.fusion_type = FusionType(self.fusion_type)
            except ValueError:
                valid_types = [t.value for t in FusionType]
                raise ValueError(
                    f"fusion_type şunlardan biri olmalıdır: {valid_types}, "
                    f"alınan: {self.fusion_type}"
                )
    
    def _validate_dimensions(self):
        """Boyut parametreleri doğrulaması."""
        # Boyutlar opsiyonel olabilir, init anında ayarlanabilirler
        dimensions = [self.text_dim, self.image_dim, self.output_dim]
        for dim_name, dim_value in zip(["text_dim", "image_dim", "output_dim"], dimensions):
            if dim_value is not None and dim_value <= 0:
                raise ValueError(f"{dim_name} pozitif olmalıdır, alınan: {dim_value}")


@dataclass
class ConcatenationFusionConfig(FusionConfig):
    """
    Concatenation füzyon mekanizması için özel yapılandırma.
    
    Örüntü: ConfigurationComposite (PT-012)
    """
    fusion_type: FusionType = FusionType.CONCATENATION
    
    # Concatenation özel parametreleri
    use_projection: bool = True  # Concatenation sonrası projeksiyon kullanılsın mı?


@dataclass
class WeightedSumFusionConfig(FusionConfig):
    """
    Weighted Sum füzyon mekanizması için özel yapılandırma.
    
    Örüntü: ConfigurationComposite (PT-012)
    """
    fusion_type: FusionType = FusionType.WEIGHTED_SUM
    
    # Weighted Sum özel parametreleri
    learnable_weights: bool = True  # Ağırlıklar öğrenilebilir mi?
    initial_text_weight: float = 0.5  # Metin modalitesi için başlangıç ağırlığı
    initial_image_weight: float = 0.5  # Görüntü modalitesi için başlangıç ağırlığı
    normalize_weights: bool = True  # Ağırlıklar toplamı 1'e normalize edilsin mi?


@dataclass
class GatedFusionConfig(FusionConfig):
    """
    Gated füzyon mekanizması için özel yapılandırma.
    
    Örüntü: ConfigurationComposite (PT-012)
    """
    fusion_type: FusionType = FusionType.GATED
    
    # Gated Fusion özel parametreleri
    gate_activation: str = "sigmoid"  # Gate aktivasyon fonksiyonu
    hidden_dim: Optional[int] = None  # Geçiş için gizli boyut
    use_residual: bool = True  # Artık bağlantı kullanılsın mı?
    
    def __post_init__(self):
        """Yapılandırma doğrulamasını gerçekleştirir."""
        super().__post_init__()
        self._validate_gate_activation()
        
    def _validate_gate_activation(self):
        """Gate aktivasyon fonksiyonu doğrulaması."""
        valid_activations = ["sigmoid", "tanh", "relu", "leaky_relu", "hardswish"]
        if self.gate_activation not in valid_activations:
            raise ValueError(
                f"gate_activation şunlardan biri olmalıdır: {valid_activations}, "
                f"alınan: {self.gate_activation}"
            )


@dataclass
class CrossAttentionFusionConfig(FusionConfig):
    """
    CrossAttention füzyon mekanizması için özel yapılandırma.
    
    Örüntü: ConfigurationComposite (PT-012), MultimodalAttention (PT-027)
    """
    fusion_type: FusionType = FusionType.CROSS_ATTENTION
    
    # CrossAttention özel parametreleri
    num_heads: int = 4  # Dikkat mekanizmasındaki kafa sayısı
    dropout: float = 0.1  # Dropout oranı


@dataclass
class AdaptiveWeightingFusionConfig(FusionConfig):
    """
    AdaptiveWeighting füzyon mekanizması için özel yapılandırma.
    
    Örüntü: ConfigurationComposite (PT-012), AdaptiveWeighting (PT-028)
    """
    fusion_type: FusionType = FusionType.ADAPTIVE_WEIGHTING
    
    # AdaptiveWeighting özel parametreleri
    hidden_dim: int = 512  # Ağırlık tahmin ağındaki gizli katman boyutu
    dropout: float = 0.1  # Dropout oranı 