"""
Embedding Modülü Yapılandırma Sınıfları

Bu modül, metin embedding ve tokenizer için yapılandırma sınıflarını içerir.
ConfigurationDataclass (PT-001) ve ConfigurationComposite (PT-012) örüntülerini uygular.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, ClassVar, Set, Tuple, Union

from m3tm.config.config_base import ConfigBase, ConfigValidationError


@dataclass
class TokenizerConfig(ConfigBase):
    """
    Tokenizer için yapılandırma sınıfı.
    
    Bu sınıf, metin tokenizasyonu için gerekli yapılandırma parametrelerini içerir.
    
    Örüntü: ConfigurationDataclass (PT-001)
    """
    
    # Sınıf özellikleri
    CONFIG_VERSION: ClassVar[str] = "1.0.0"
    REQUIRED_FIELDS: ClassVar[List[str]] = ["vocab_size", "max_seq_length"]
    
    # Yapılandırma parametreleri
    vocab_size: int = 10000
    max_seq_length: int = 512
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    })
    lowercase: bool = True
    strip_accents: bool = True
    add_bos_token: bool = True
    add_eos_token: bool = True
    
    def validate(self) -> None:
        """
        Yapılandırma değerlerinin geçerliliğini doğrular.
        
        Raises:
            ConfigValidationError: Yapılandırma geçersizse
        """
        super().validate()
        
        if self.vocab_size <= 0:
            raise ConfigValidationError("vocab_size must be positive")
        
        if self.max_seq_length <= 0:
            raise ConfigValidationError("max_seq_length must be positive")
        
        if not self.special_tokens or len(self.special_tokens) == 0:
            raise ConfigValidationError("At least one special token must be defined")
        
        # Özel token ID'lerinin sıfır veya pozitif olduğunu kontrol et
        for token, token_id in self.special_tokens.items():
            if token_id < 0:
                raise ConfigValidationError(f"Token ID must be non-negative: {token} has ID {token_id}")
        
        # Özel token ID'lerinin benzersiz olduğunu kontrol et
        token_ids = list(self.special_tokens.values())
        if len(token_ids) != len(set(token_ids)):
            raise ConfigValidationError("Token IDs must be unique")


@dataclass
class TextEmbeddingConfig(ConfigBase):
    """
    Metin embedding için yapılandırma sınıfı.
    
    Bu sınıf, metin embedding işlemi için gerekli yapılandırma parametrelerini içerir.
    Tokenizer yapılandırmasını da barındırır (kompozit yapılandırma).
    
    Örüntü: ConfigurationDataclass (PT-001), ConfigurationComposite (PT-012)
    """
    
    # Sınıf özellikleri
    CONFIG_VERSION: ClassVar[str] = "1.0.0"
    REQUIRED_FIELDS: ClassVar[List[str]] = ["embed_dim", "tokenizer_config"]
    
    # Yapılandırma parametreleri
    embed_dim: int = 384
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    padding_idx: int = 0
    use_position_embedding: bool = True
    max_position_embeddings: int = 512
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    use_embedding_projection: bool = False
    projection_dim: Optional[int] = None
    
    def validate(self) -> None:
        """
        Yapılandırma değerlerinin geçerliliğini doğrular.
        
        Raises:
            ConfigValidationError: Yapılandırma geçersizse
        """
        super().validate()
        
        if self.embed_dim <= 0:
            raise ConfigValidationError("embed_dim must be positive")
        
        if self.use_position_embedding and (self.max_position_embeddings <= 0):
            raise ConfigValidationError("max_position_embeddings must be positive when use_position_embedding is True")
        
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ConfigValidationError("dropout_rate must be between 0.0 and 1.0")
        
        if self.layer_norm_eps <= 0:
            raise ConfigValidationError("layer_norm_eps must be positive")
        
        if self.use_embedding_projection and (self.projection_dim is None or self.projection_dim <= 0):
            raise ConfigValidationError("projection_dim must be positive when use_embedding_projection is True")
        
        # tokenizer_config doğrulamaya gerek yok, çünkü bu sınıfın init aşamasında zaten doğrulanacak


@dataclass
class ImagePatchEmbeddingConfig(ConfigBase):
    """
    Görüntü yama embedding için yapılandırma sınıfı.
    
    Bu sınıf, görüntü yama embedding işlemi için gerekli yapılandırma parametrelerini içerir.
    
    Örüntü: ConfigurationDataclass (PT-001)
    """
    
    # Sınıf özellikleri
    CONFIG_VERSION: ClassVar[str] = "1.0.0"
    REQUIRED_FIELDS: ClassVar[List[str]] = ["embed_dim", "patch_size", "image_size"]
    
    # Yapılandırma parametreleri
    embed_dim: int = 384
    patch_size: int = 16
    image_size: Union[int, Tuple[int, int]] = (224, 224)
    in_channels: int = 3
    channels: int = 3  # Geriye dönük uyumluluk için
    use_position_embedding: bool = True
    position_embedding_type: str = "learned"  # "learned", "sincos" veya "alibi"
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    use_embedding_projection: bool = False
    projection_dim: Optional[int] = None
    
    def __post_init__(self):
        """Yapılandırma değerlerini sonrasında işle."""
        super().__post_init__()
        
        # in_channels ve channels parametrelerini senkronize et
        if hasattr(self, 'in_channels'):
            self.channels = self.in_channels
        elif hasattr(self, 'channels'):
            self.in_channels = self.channels
        
        # image_size tek bir sayı ise, kare görüntü boyutu olarak kabul et
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
    
    def validate(self) -> None:
        """
        Yapılandırma değerlerinin geçerliliğini doğrular.
        
        Raises:
            ConfigValidationError: Yapılandırma geçersizse
        """
        super().validate()
        
        if self.embed_dim <= 0:
            raise ConfigValidationError("embed_dim must be positive")
        
        if self.patch_size <= 0:
            raise ConfigValidationError("patch_size must be positive")
        
        # Görüntü boyutunun yama boyutuna tam bölünüp bölünmediğini kontrol et
        height, width = self.image_size
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ConfigValidationError(
                f"Image dimensions ({height}, {width}) must be divisible by patch_size ({self.patch_size})"
            )
        
        if self.in_channels <= 0:
            raise ConfigValidationError("in_channels must be positive")
        
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ConfigValidationError("dropout_rate must be between 0.0 and 1.0")
        
        if self.position_embedding_type not in ["learned", "sincos", "alibi"]:
            raise ConfigValidationError("position_embedding_type must be one of: learned, sincos, alibi")
        
        if self.use_embedding_projection and (self.projection_dim is None or self.projection_dim <= 0):
            raise ConfigValidationError("projection_dim must be positive when use_embedding_projection is True") 