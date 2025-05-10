"""
Temel Yapılandırma Sınıfları

Bu modül, tüm yapılandırma sınıfları için temel sınıfları ve yardımcı fonksiyonları içerir.
ConfigurationDataclass (PT-001) örüntüsünü uygular.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, ClassVar, Type, TypeVar, List, Union
from dataclasses import dataclass, field, asdict, is_dataclass


class ConfigValidationError(Exception):
    """Yapılandırma doğrulama hatası."""
    pass


T = TypeVar('T', bound='ConfigBase')

@dataclass
class ConfigBase:
    """
    Tüm yapılandırma sınıfları için temel sınıf.
    
    Bu sınıf, yapılandırma nesnelerinin JSON/YAML formatında kaydedilmesi,
    bu formatlardan yüklenmesi ve doğrulanması için temel fonksiyonelliği sağlar.
    
    Örüntü: ConfigurationDataclass (PT-001)
    """
    
    # Alt sınıfların override edebileceği sınıf değişkenleri
    CONFIG_VERSION: ClassVar[str] = "1.0.0"
    REQUIRED_FIELDS: ClassVar[List[str]] = []
    
    validation_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """
        Yapılandırma tutarlılığını doğrular.
        
        ConfigValidationError hatası, yapılandırma geçersizse fırlatılır.
        """
        self.validate()
    
    def validate(self) -> bool:
        """
        Yapılandırma parametrelerini doğrular.
        Alt sınıflar bu metodu genişletmelidir.
        
        Returns:
            Yapılandırma geçerliyse True, değilse False
        """
        self.validation_errors = []
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Yapılandırma sınıfını sözlük olarak döndürür.
        
        Returns:
            Yapılandırma parametrelerini içeren sözlük
        """
        result = asdict(self)
        result["__config_type__"] = self.__class__.__name__
        result["__config_version__"] = self.CONFIG_VERSION
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """
        Yapılandırmayı JSON formatında döndürür.
        
        Args:
            indent: JSON girintileme seviyesi
            
        Returns:
            str: JSON formatında yapılandırma
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_yaml(self) -> str:
        """
        Yapılandırmayı YAML formatında döndürür.
        
        Returns:
            str: YAML formatında yapılandırma
        """
        return yaml.dump(self.to_dict(), sort_keys=False)
    
    def save(self, file_path: Union[str, Path], format: str = "json") -> None:
        """
        Yapılandırmayı dosyaya kaydeder.
        
        Args:
            file_path: Kaydedilecek dosya yolu
            format: Kaydetme formatı ('json' veya 'yaml')
            
        Raises:
            ValueError: Desteklenmeyen format belirtilirse
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
        elif format == "yaml":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Sözlükten yapılandırma nesnesi oluşturur.
        
        Args:
            config_dict: Yapılandırma değerlerini içeren sözlük
            
        Returns:
            T: Oluşturulan yapılandırma nesnesi
        """
        # __config_type__ ve __config_version__ alanlarını kaldır
        clean_dict = config_dict.copy()
        clean_dict.pop("__config_type__", None)
        clean_dict.pop("__config_version__", None)
        
        # Dataclass alt sınıfların dönüştürülmesi için yinelemeli işlem
        for key, value in list(clean_dict.items()):
            if isinstance(value, dict) and "__config_type__" in value:
                # Alt yapılandırma sınıfının tipini belirle ve oluştur
                sub_config_type = value["__config_type__"]
                # Alt sınıf tipini bul ve oluştur (basit yöntem - geliştirilebilir)
                # Not: Daha gelişmiş bir çözüm için bir kayıt mekanizması kullanılabilir
                from . import model_config
                if hasattr(model_config, sub_config_type):
                    sub_class = getattr(model_config, sub_config_type)
                    clean_dict[key] = sub_class.from_dict(value)
        
        # Yapılandırma sınıfı örneğini oluştur
        return cls(**clean_dict)
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """
        JSON formatından yapılandırma nesnesi oluşturur.
        
        Args:
            json_str: JSON formatında yapılandırma
            
        Returns:
            T: Oluşturulan yapılandırma nesnesi
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls: Type[T], yaml_str: str) -> T:
        """
        YAML formatından yapılandırma nesnesi oluşturur.
        
        Args:
            yaml_str: YAML formatında yapılandırma
            
        Returns:
            T: Oluşturulan yapılandırma nesnesi
        """
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load(cls: Type[T], file_path: Union[str, Path], format: Optional[str] = None) -> T:
        """
        Dosyadan yapılandırma nesnesi yükler.
        
        Args:
            file_path: Yapılandırma dosyasının yolu
            format: Dosya formatı ('json' veya 'yaml'). Belirtilmezse uzantıdan çıkarılır.
            
        Returns:
            T: Oluşturulan yapılandırma nesnesi
            
        Raises:
            ValueError: Desteklenmeyen veya tanımlanamayan format
            FileNotFoundError: Dosya bulunamazsa
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Format belirtilmemişse, dosya uzantısından çıkar
        if format is None:
            if file_path.suffix.lower() in ['.json']:
                format = "json"
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                format = "yaml"
            else:
                raise ValueError(f"Cannot determine format from file extension: {file_path.suffix}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if format == "json":
            return cls.from_json(content)
        elif format == "yaml":
            return cls.from_yaml(content)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")


def validate_positive_integer(config: ConfigBase, field_name: str) -> bool:
    """
    Bir alanın pozitif tamsayı olduğunu doğrular.
    
    Args:
        config: Yapılandırma nesnesi
        field_name: Kontrol edilecek alan adı
        
    Returns:
        Alan geçerliyse True, değilse False
    """
    field_value = getattr(config, field_name, None)
    
    if field_value is None:
        return True
    
    if not isinstance(field_value, int) or field_value <= 0:
        config.validation_errors.append(f"{field_name} must be a positive integer, got {field_value}")
        return False
    
    return True


def validate_non_negative_integer(config: ConfigBase, field_name: str) -> bool:
    """
    Bir alanın negatif olmayan tamsayı olduğunu doğrular.
    
    Args:
        config: Yapılandırma nesnesi
        field_name: Kontrol edilecek alan adı
        
    Returns:
        Alan geçerliyse True, değilse False
    """
    field_value = getattr(config, field_name, None)
    
    if field_value is None:
        return True
    
    if not isinstance(field_value, int) or field_value < 0:
        config.validation_errors.append(f"{field_name} must be a non-negative integer, got {field_value}")
        return False
    
    return True


def validate_probability(config: ConfigBase, field_name: str) -> bool:
    """
    Bir alanın olasılık değeri (0.0 ile 1.0 arasında) olduğunu doğrular.
    
    Args:
        config: Yapılandırma nesnesi
        field_name: Kontrol edilecek alan adı
        
    Returns:
        Alan geçerliyse True, değilse False
    """
    field_value = getattr(config, field_name, None)
    
    if field_value is None:
        return True
    
    if not isinstance(field_value, (int, float)) or field_value < 0 or field_value > 1:
        config.validation_errors.append(f"{field_name} must be a float between 0 and 1, got {field_value}")
        return False
    
    return True


def validate_positive_float(config: ConfigBase, field_name: str) -> bool:
    """
    Bir alanın pozitif ondalıklı sayı olduğunu doğrular.
    
    Args:
        config: Yapılandırma nesnesi
        field_name: Kontrol edilecek alan adı
        
    Returns:
        Alan geçerliyse True, değilse False
    """
    field_value = getattr(config, field_name, None)
    
    if field_value is None:
        return True
    
    if not isinstance(field_value, (int, float)) or field_value <= 0:
        config.validation_errors.append(f"{field_name} must be a positive float, got {field_value}")
        return False
    
    return True


def validate_non_negative_float(config: ConfigBase, field_name: str) -> bool:
    """
    Bir alanın negatif olmayan ondalıklı sayı olduğunu doğrular.
    
    Args:
        config: Yapılandırma nesnesi
        field_name: Kontrol edilecek alan adı
        
    Returns:
        Alan geçerliyse True, değilse False
    """
    field_value = getattr(config, field_name, None)
    
    if field_value is None:
        return True
    
    if not isinstance(field_value, (int, float)) or field_value < 0:
        config.validation_errors.append(f"{field_name} must be a non-negative float, got {field_value}")
        return False
    
    return True


def validate_string_in_list(config: ConfigBase, field_name: str, valid_values: List[str]) -> bool:
    """
    Bir dize alanının izin verilen değerler listesinde olduğunu doğrular.
    
    Args:
        config: Yapılandırma nesnesi
        field_name: Kontrol edilecek alan adı
        valid_values: İzin verilen değerler listesi
        
    Returns:
        Alan geçerliyse True, değilse False
    """
    field_value = getattr(config, field_name, None)
    
    if field_value is None:
        return True
    
    if not isinstance(field_value, str) or field_value not in valid_values:
        config.validation_errors.append(f"{field_name} must be one of {valid_values}, got {field_value}")
        return False
    
    return True


def validate_range(config: ConfigBase, field_name: str, min_value: Union[int, float], max_value: Union[int, float]) -> bool:
    """
    Bir alanın belirli bir aralıkta olduğunu doğrular.
    
    Args:
        config: Yapılandırma nesnesi
        field_name: Kontrol edilecek alan adı
        min_value: Minimum değer (dahil)
        max_value: Maksimum değer (dahil)
        
    Returns:
        Alan geçerliyse True, değilse False
    """
    field_value = getattr(config, field_name, None)
    
    if field_value is None:
        return True
    
    if not isinstance(field_value, (int, float)) or field_value < min_value or field_value > max_value:
        config.validation_errors.append(f"{field_name} must be between {min_value} and {max_value}, got {field_value}")
        return False
    
    return True 