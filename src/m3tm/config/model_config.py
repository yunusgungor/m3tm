"""
M³TM Model Yapılandırma Sınıfları

Bu modül, M³TM modelinin tüm bileşenleri için yapılandırma sınıflarını içerir.
ConfigurationDataclass (PT-001) örüntüsünü uygular.
"""

from typing import List, Optional, Dict, Any, Tuple, Union, ClassVar
from dataclasses import dataclass, field, InitVar

from .config_base import ConfigBase, ConfigValidationError


@dataclass
class TextEmbeddingConfig(ConfigBase):
    """
    Metin gömme (embedding) bileşeni yapılandırması.
    """
    vocab_size: int = 4000  # Sözlük boyutu (token sayısı)
    embed_dim: int = 64  # Gömme boyutu
    max_seq_len: int = 256  # Maksimum dizi uzunluğu
    pad_token_id: int = 0  # Dolgu (padding) token ID'si
    use_positional_embedding: bool = True  # Konumsal gömme kullanımı
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['vocab_size', 'embed_dim']
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.vocab_size <= 0:
            raise ConfigValidationError(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.embed_dim <= 0:
            raise ConfigValidationError(f"embed_dim must be positive, got {self.embed_dim}")
        
        if self.max_seq_len <= 0:
            raise ConfigValidationError(f"max_seq_len must be positive, got {self.max_seq_len}")


@dataclass
class ImagePatchEmbeddingConfig(ConfigBase):
    """
    Görüntü yama gömme bileşeni yapılandırması.
    """
    in_channels: int = 3  # Girdi kanal sayısı (genellikle RGB için 3)
    patch_size: int = 4  # Yama boyutu (örn: 4x4 piksel)
    embed_dim: int = 64  # Gömme boyutu
    image_size: Tuple[int, int] = (224, 224)  # Görüntü boyutu (yükseklik, genişlik)
    use_positional_embedding: bool = True  # Konumsal gömme kullanımı
    dropout: float = 0.1  # Dropout oranı
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['in_channels', 'patch_size', 'embed_dim']
    
    def __post_init__(self):
        """Ek alanları hesaplar."""
        # Yama sayısını hesapla
        self.num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        super().__post_init__()
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.in_channels <= 0:
            raise ConfigValidationError(f"in_channels must be positive, got {self.in_channels}")
        
        if self.patch_size <= 0:
            raise ConfigValidationError(f"patch_size must be positive, got {self.patch_size}")
        
        if self.embed_dim <= 0:
            raise ConfigValidationError(f"embed_dim must be positive, got {self.embed_dim}")
        
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ConfigValidationError(f"image_size dimensions must be positive, got {self.image_size}")
        
        if self.image_size[0] % self.patch_size != 0 or self.image_size[1] % self.patch_size != 0:
            raise ConfigValidationError(
                f"image_size dimensions ({self.image_size}) must be divisible by patch_size {self.patch_size}"
            )
        
        if self.dropout < 0 or self.dropout > 1:
            raise ConfigValidationError(f"dropout must be between 0 and 1, got {self.dropout}")


@dataclass
class TransformerConfig(ConfigBase):
    """
    Transformer blok yapılandırması.
    """
    embed_dim: int = 64  # Gömme boyutu
    num_heads: int = 2  # Dikkat başı sayısı
    mlp_ratio: float = 2.0  # MLP gizli boyutu = embed_dim * mlp_ratio
    dropout: float = 0.1  # Dropout oranı
    attention_dropout: float = 0.1  # Dikkat dropout oranı
    use_bias: bool = True  # Doğrusal katmanlarda bias kullanımı
    use_adapter_slots: bool = True  # Adapter yuvaları kullanımı
    activation: str = "gelu"  # Aktivasyon fonksiyonu
    # Dikkat ve FFN alternatifleri
    attention_type: str = "MobileAttention"  # Dikkat mekanizması
    ffn_type: str = "GLU"  # FFN alternatifi
    
    # Opsiyonel config nesneleri - gerçek değerler alt modüllerden gelecek
    attention_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    ffn_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['embed_dim', 'num_heads']
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.embed_dim <= 0:
            raise ConfigValidationError(f"embed_dim must be positive, got {self.embed_dim}")
        
        if self.num_heads <= 0:
            raise ConfigValidationError(f"num_heads must be positive, got {self.num_heads}")
        
        if self.embed_dim % self.num_heads != 0:
            raise ConfigValidationError(
                f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}"
            )
        
        if self.mlp_ratio <= 0:
            raise ConfigValidationError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ConfigValidationError(f"dropout must be between 0 and 1, got {self.dropout}")
        
        if self.attention_dropout < 0 or self.attention_dropout > 1:
            raise ConfigValidationError(
                f"attention_dropout must be between 0 and 1, got {self.attention_dropout}"
            )


@dataclass
class FusionConfig(ConfigBase):
    """
    Modalite füzyon (birleştirme) bileşeni yapılandırması.
    """
    fusion_type: str = "concat"  # Füzyon tipi: concat, add, gate, bilinear, vb.
    text_dim: int = 64  # Metin gömme boyutu
    image_dim: int = 64  # Görüntü gömme boyutu
    output_dim: int = 128  # Füzyon çıktı boyutu
    dropout: float = 0.1  # Dropout oranı
    use_layer_norm: bool = True  # Katman normalizasyonu kullanımı
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['fusion_type', 'text_dim', 'image_dim', 'output_dim']
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.fusion_type not in ["concat", "add", "gate", "bilinear", "cross_attention"]:
            raise ConfigValidationError(
                f"fusion_type must be one of ['concat', 'add', 'gate', 'bilinear', 'cross_attention'], "
                f"got {self.fusion_type}"
            )
        
        if self.text_dim <= 0:
            raise ConfigValidationError(f"text_dim must be positive, got {self.text_dim}")
        
        if self.image_dim <= 0:
            raise ConfigValidationError(f"image_dim must be positive, got {self.image_dim}")
        
        if self.output_dim <= 0:
            raise ConfigValidationError(f"output_dim must be positive, got {self.output_dim}")
        
        if self.fusion_type == "add" and self.text_dim != self.image_dim:
            raise ConfigValidationError(
                f"For 'add' fusion, text_dim {self.text_dim} must equal image_dim {self.image_dim}"
            )
        
        if self.dropout < 0 or self.dropout > 1:
            raise ConfigValidationError(f"dropout must be between 0 and 1, got {self.dropout}")


@dataclass
class SearchConfig(ConfigBase):
    """
    Arama bileşeni yapılandırması.
    """
    input_dim: int = 128  # Girdi boyutu (füzyon çıktısı)
    search_dim: int = 128  # Arama gömme uzayı boyutu
    index_type: str = "faiss"  # Indeksleme tipi: faiss, annoy, vb.
    metric: str = "cosine"  # Benzerlik metriği: cosine, l2, vb.
    use_normalization: bool = True  # Normalizasyon kullanımı
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['input_dim', 'search_dim', 'index_type', 'metric']
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.input_dim <= 0:
            raise ConfigValidationError(f"input_dim must be positive, got {self.input_dim}")
        
        if self.search_dim <= 0:
            raise ConfigValidationError(f"search_dim must be positive, got {self.search_dim}")
        
        if self.index_type not in ["faiss", "annoy", "simple"]:
            raise ConfigValidationError(
                f"index_type must be one of ['faiss', 'annoy', 'simple'], got {self.index_type}"
            )
        
        if self.metric not in ["cosine", "l2", "inner_product"]:
            raise ConfigValidationError(
                f"metric must be one of ['cosine', 'l2', 'inner_product'], got {self.metric}"
            )


@dataclass
class AdapterConfig(ConfigBase):
    """
    Adapter bileşeni yapılandırması.
    """
    embed_dim: int = 64  # Gömme boyutu (model boyutuyla eşleşmeli)
    reduction_factor: int = 16  # Düşürme faktörü (bottleneck size = embed_dim / reduction_factor)
    activation: str = "gelu"  # Aktivasyon fonksiyonu
    init_scale: float = 0.001  # Başlangıç ölçekleme faktörü
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['embed_dim', 'reduction_factor']
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.embed_dim <= 0:
            raise ConfigValidationError(f"embed_dim must be positive, got {self.embed_dim}")
        
        if self.reduction_factor <= 0:
            raise ConfigValidationError(f"reduction_factor must be positive, got {self.reduction_factor}")
        
        if self.embed_dim % self.reduction_factor != 0:
            raise ConfigValidationError(
                f"embed_dim {self.embed_dim} must be divisible by reduction_factor {self.reduction_factor}"
            )
        
        if self.init_scale <= 0:
            raise ConfigValidationError(f"init_scale must be positive, got {self.init_scale}")


@dataclass
class TaskHeadConfig(ConfigBase):
    """
    Görev başlığı bileşeni yapılandırması.
    """
    task_type: str = "classification"  # Görev tipi: classification, regression, vb.
    input_dim: int = 128  # Girdi boyutu
    hidden_dim: int = 64  # Gizli katman boyutu (0 ise doğrudan son katman)
    output_dim: int = 2  # Çıktı boyutu (sınıflandırma için sınıf sayısı)
    use_bias: bool = True  # Bias kullanımı
    dropout: float = 0.1  # Dropout oranı
    activation: str = "gelu"  # Aktivasyon fonksiyonu
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['task_type', 'input_dim', 'output_dim']
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.task_type not in ["classification", "regression", "multilabel"]:
            raise ConfigValidationError(
                f"task_type must be one of ['classification', 'regression', 'multilabel'], "
                f"got {self.task_type}"
            )
        
        if self.input_dim <= 0:
            raise ConfigValidationError(f"input_dim must be positive, got {self.input_dim}")
        
        if self.hidden_dim < 0:
            raise ConfigValidationError(f"hidden_dim must be non-negative, got {self.hidden_dim}")
        
        if self.output_dim <= 0:
            raise ConfigValidationError(f"output_dim must be positive, got {self.output_dim}")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ConfigValidationError(f"dropout must be between 0 and 1, got {self.dropout}")


@dataclass
class TrainingConfig(ConfigBase):
    """
    Eğitim yapılandırması.
    """
    batch_size: int = 16  # Parti (batch) boyutu
    learning_rate: float = 0.001  # Öğrenme oranı
    weight_decay: float = 0.01  # Ağırlık bozunması
    epochs: int = 10  # Devir (epoch) sayısı
    optimizer: str = "adamw"  # Optimizer tipi
    scheduler: str = "cosine"  # Öğrenme oranı zamanlayıcısı
    warmup_steps: int = 100  # Isınma adımları
    gradient_clip: float = 1.0  # Gradyan kırpma
    early_stopping_patience: int = 5  # Erken durdurma sabır sayısı
    device: str = "cpu"  # Eğitim cihazı: cpu, cuda, mps
    mixed_precision: bool = False  # Karışık hassasiyet eğitimi
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['batch_size', 'learning_rate', 'epochs']
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.batch_size <= 0:
            raise ConfigValidationError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.learning_rate <= 0:
            raise ConfigValidationError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ConfigValidationError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        if self.epochs <= 0:
            raise ConfigValidationError(f"epochs must be positive, got {self.epochs}")
        
        if self.warmup_steps < 0:
            raise ConfigValidationError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        
        if self.gradient_clip < 0:
            raise ConfigValidationError(f"gradient_clip must be non-negative, got {self.gradient_clip}")
        
        if self.early_stopping_patience < 0:
            raise ConfigValidationError(
                f"early_stopping_patience must be non-negative, got {self.early_stopping_patience}"
            )
        
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ConfigValidationError(
                f"device must be one of ['cpu', 'cuda', 'mps'], got {self.device}"
            )


@dataclass
class M3TMModelConfig(ConfigBase):
    """
    M³TM modeli ana yapılandırması.
    
    Tüm model bileşenlerinin yapılandırmalarını içerir.
    """
    name: str = "m3tm-v2.3-base"  # Model adı
    description: str = "Mobil Multi-Modal Modüler Transformer"  # Model açıklaması
    version: str = "2.3.0"  # Model versiyonu
    
    # Alt bileşen yapılandırmaları
    text_config: TextEmbeddingConfig = field(default_factory=TextEmbeddingConfig)
    image_config: ImagePatchEmbeddingConfig = field(default_factory=ImagePatchEmbeddingConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    search_config: SearchConfig = field(default_factory=SearchConfig)
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)
    
    # Eğitim yapılandırması
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Model spesifik parametreler
    num_transformer_blocks: int = 3  # Transformer blok sayısı
    use_text_modality: bool = True  # Metin modalitesi kullanımı
    use_image_modality: bool = True  # Görüntü modalitesi kullanımı
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['name', 'version']
    CONFIG_VERSION: ClassVar[str] = "1.0.0"
    
    def validate(self) -> None:
        """Yapılandırma değerlerini doğrular."""
        super().validate()
        
        if self.num_transformer_blocks <= 0:
            raise ConfigValidationError(
                f"num_transformer_blocks must be positive, got {self.num_transformer_blocks}"
            )
        
        # Alt yapılandırmaların tutarlılığını kontrol et
        # Örn: text_config.embed_dim ve transformer_config.embed_dim'in uyumluluğu
        if self.text_config.embed_dim != self.transformer_config.embed_dim:
            raise ConfigValidationError(
                f"text_config.embed_dim {self.text_config.embed_dim} must equal "
                f"transformer_config.embed_dim {self.transformer_config.embed_dim}"
            )
        
        if self.image_config.embed_dim != self.transformer_config.embed_dim:
            raise ConfigValidationError(
                f"image_config.embed_dim {self.image_config.embed_dim} must equal "
                f"transformer_config.embed_dim {self.transformer_config.embed_dim}"
            )
        
        # Füzyon yapılandırmasının tutarlılığını kontrol et
        if self.fusion_config.text_dim != self.text_config.embed_dim:
            raise ConfigValidationError(
                f"fusion_config.text_dim {self.fusion_config.text_dim} must equal "
                f"text_config.embed_dim {self.text_config.embed_dim}"
            )
        
        if self.fusion_config.image_dim != self.image_config.embed_dim:
            raise ConfigValidationError(
                f"fusion_config.image_dim {self.fusion_config.image_dim} must equal "
                f"image_config.embed_dim {self.image_config.embed_dim}"
            )
        
        # Arama yapılandırmasının tutarlılığını kontrol et
        if self.search_config.input_dim != self.fusion_config.output_dim:
            raise ConfigValidationError(
                f"search_config.input_dim {self.search_config.input_dim} must equal "
                f"fusion_config.output_dim {self.fusion_config.output_dim}"
            )
        
        # Adapter yapılandırmasının tutarlılığını kontrol et
        if self.adapter_config.embed_dim != self.transformer_config.embed_dim:
            raise ConfigValidationError(
                f"adapter_config.embed_dim {self.adapter_config.embed_dim} must equal "
                f"transformer_config.embed_dim {self.transformer_config.embed_dim}"
            )
    
    @classmethod
    def get_default_config(cls) -> 'M3TMModelConfig':
        """Varsayılan model yapılandırmasını döndürür."""
        config = cls()
        
        # Boyut tutarlılığını sağla
        config.text_config.embed_dim = 64
        config.image_config.embed_dim = 64
        config.transformer_config.embed_dim = 64
        config.fusion_config.text_dim = 64
        config.fusion_config.image_dim = 64
        config.fusion_config.output_dim = 128
        config.search_config.input_dim = 128
        config.search_config.search_dim = 128
        config.adapter_config.embed_dim = 64
    
    return config 