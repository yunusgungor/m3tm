"""
Görev Başlıkları (TaskHeads) Yapılandırma Modülü

Bu modül, görev başlıkları için gerekli yapılandırma sınıflarını içerir.

Örüntüler:
- ConfigurationDataclass (PT-001): Tip güvenliği, doğrulama ve varsayılan değerler sağlar
- ConfigValidationPipeline (PT-009): Kapsamlı doğrulama kuralları oluşturur
- ConfigurationComposite (PT-012): Alt yapılandırmaları organize eder
"""

from dataclasses import dataclass, field
from typing import List, ClassVar, Optional, Dict, Any, Union

from m3tm.config.config_base import ConfigBase, ConfigValidationError


@dataclass
class ClassificationHeadConfig(ConfigBase):
    """
    Sınıflandırma başlığı yapılandırması.
    
    Metin sınıflandırma için özelleştirilmiş görev başlığı.
    
    Örüntü: ConfigurationDataclass (PT-001)
    """
    input_dim: int = 64  # Girdi öznitelik boyutu
    hidden_dim: int = 32  # Ara katman boyutu (0 ise doğrudan çıktı)
    num_classes: int = 2  # Sınıf sayısı
    dropout_rate: float = 0.1  # Dropout oranı
    activation: str = "gelu"  # Aktivasyon fonksiyonu (gelu, relu, swish)
    pooling: str = "mean"  # Havuzlama yöntemi (mean, max, cls, first)
    layer_norm_eps: float = 1e-12  # Katman normalizasyonu epsilon değeri
    use_bias: bool = True  # Bias kullanımı
    use_layer_norm: bool = True  # Katman normalizasyonu kullanımı
    
    REQUIRED_FIELDS: ClassVar[List[str]] = ['input_dim', 'num_classes']
    CONFIG_VERSION: ClassVar[str] = "1.0.0"
    
    def validate(self) -> None:
        """Yapılandırma değerlerinin geçerliliğini doğrular."""
        super().validate()
        
        if self.input_dim <= 0:
            raise ConfigValidationError(f"input_dim must be positive, got {self.input_dim}")
        
        if self.hidden_dim < 0:
            raise ConfigValidationError(f"hidden_dim must be non-negative, got {self.hidden_dim}")
        
        if self.num_classes <= 1:
            raise ConfigValidationError(f"num_classes must be at least 2, got {self.num_classes}")
        
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ConfigValidationError(f"dropout_rate must be in range [0, 1), got {self.dropout_rate}")
        
        if self.activation not in ["gelu", "relu", "swish", "sigmoid", "tanh"]:
            raise ConfigValidationError(
                f"activation must be one of ['gelu', 'relu', 'swish', 'sigmoid', 'tanh'], "
                f"got {self.activation}"
            )
        
        if self.pooling not in ["mean", "max", "cls", "first"]:
            raise ConfigValidationError(
                f"pooling must be one of ['mean', 'max', 'cls', 'first'], "
                f"got {self.pooling}"
            )


@dataclass
class TaskHeadFactory:
    """
    Görev başlıkları için fabrika sınıfı.
    
    Farklı görev başlığı yapılandırmalarını oluşturmak için yardımcı metotlar sağlar.
    
    Örüntü: Factory Method
    """
    
    @staticmethod
    def create_binary_classification_config(
        input_dim: int = 64,
        hidden_dim: int = 0
    ) -> ClassificationHeadConfig:
        """
        İkili sınıflandırma yapılandırması oluşturur.
        
        Args:
            input_dim: Girdi boyutu
            hidden_dim: Gizli katman boyutu
            
        Returns:
            ClassificationHeadConfig: İkili sınıflandırma yapılandırması
        """
        return ClassificationHeadConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=2
        )
    
    @staticmethod
    def create_multiclass_classification_config(
        input_dim: int = 64,
        hidden_dim: int = 32,
        num_classes: int = 5
    ) -> ClassificationHeadConfig:
        """
        Çok sınıflı sınıflandırma yapılandırması oluşturur.
        
        Args:
            input_dim: Girdi boyutu
            hidden_dim: Gizli katman boyutu
            num_classes: Sınıf sayısı
            
        Returns:
            ClassificationHeadConfig: Çok sınıflı sınıflandırma yapılandırması
        """
        return ClassificationHeadConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ) 