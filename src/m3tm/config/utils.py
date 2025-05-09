"""
Yapılandırma Yardımcı Fonksiyonları

Bu modül, M³TM yapılandırma sistemi için yardımcı fonksiyonları içerir.
ConfigurationDataclass (PT-001) ve MechanismRegistry (PT-007) örüntülerini destekler.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TypeVar, Type

from .config_base import ConfigBase


# ConfigValidationPipeline potansiyel örüntüsünün basit bir uygulaması
class ConfigValidator:
    """
    Yapılandırma doğrulama sınıfı.
    
    Bir yapılandırma nesnesi veya dosyası için bir dizi doğrulama kontrolü uygular.
    
    Potansiyel Örüntü: ConfigValidationPipeline
    """
    
    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> bool:
        """
        Yapılandırma dosyasının var olup olmadığını kontrol eder.
        
        Args:
            file_path: Kontrol edilecek dosya yolu
            
        Returns:
            bool: Dosya varsa True, yoksa False
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        return file_path.exists() and file_path.is_file()
    
    @staticmethod
    def validate_json_format(json_str: str) -> bool:
        """
        JSON formatının geçerli olup olmadığını kontrol eder.
        
        Args:
            json_str: Kontrol edilecek JSON string
            
        Returns:
            bool: Format geçerliyse True, değilse False
        """
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    @staticmethod
    def validate_yaml_format(yaml_str: str) -> bool:
        """
        YAML formatının geçerli olup olmadığını kontrol eder.
        
        Args:
            yaml_str: Kontrol edilecek YAML string
            
        Returns:
            bool: Format geçerliyse True, değilse False
        """
        try:
            yaml.safe_load(yaml_str)
            return True
        except yaml.YAMLError:
            return False
    
    @staticmethod
    def validate_required_fields(config_data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """
        Yapılandırma verisinin gerekli alanları içerip içermediğini kontrol eder.
        
        Args:
            config_data: Kontrol edilecek yapılandırma verisi
            required_fields: Gerekli alanların listesi
            
        Returns:
            List[str]: Eksik alanların listesi (boşsa tüm gerekli alanlar mevcut)
        """
        missing_fields = []
        
        for field in required_fields:
            if field not in config_data or config_data[field] is None:
                missing_fields.append(field)
        
        return missing_fields


# Yapılandırma araçları
def find_config_file(base_dir: Union[str, Path], config_name: str = "m3tm_config", 
                    file_types: List[str] = None) -> Optional[Path]:
    """
    Belirtilen dizinde ve üst dizinlerde yapılandırma dosyasını arar.
    
    Args:
        base_dir: Aramaya başlanacak dizin
        config_name: Yapılandırma dosyası adı (uzantısız)
        file_types: Aranacak dosya uzantıları (None ise, varsayılan olarak ['.json', '.yaml', '.yml'])
        
    Returns:
        Optional[Path]: Bulunan yapılandırma dosyasının yolu veya None
    """
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    
    if file_types is None:
        file_types = ['.json', '.yaml', '.yml']
    
    # Var olan dizin olup olmadığını kontrol et
    if not base_dir.exists() or not base_dir.is_dir():
        return None
    
    # Mevcut dizinde ara
    for file_type in file_types:
        config_file = base_dir / f"{config_name}{file_type}"
        if config_file.exists() and config_file.is_file():
            return config_file
    
    # Üst dizine geç (kök dizine ulaşıncaya kadar)
    parent_dir = base_dir.parent
    if parent_dir == base_dir:  # Kök dizine ulaştık
        return None
    
    # Üst dizinde yinelemeli olarak ara
    return find_config_file(parent_dir, config_name, file_types)


def get_env_config_prefix() -> str:
    """
    Çevre değişkenleri için yapılandırma önekini döndürür.
    
    Çevre değişkeni M3TM_CONFIG_PREFIX ile özelleştirilebilir.
    
    Returns:
        str: Yapılandırma öneki
    """
    return os.environ.get("M3TM_CONFIG_PREFIX", "M3TM_CONFIG")


def print_config_summary(config: ConfigBase) -> str:
    """
    Bir yapılandırma nesnesinin özet bilgilerini formatlar.
    
    Args:
        config: Yapılandırma nesnesi
        
    Returns:
        str: Formatlanmış özet bilgiler
    """
    config_dict = config.to_dict()
    
    # Temel bilgileri topla
    if hasattr(config, 'name') and hasattr(config, 'version'):
        summary_lines = [
            f"Model: {config.name}",
            f"Versiyon: {config.version}"
        ]
    else:
        summary_lines = [f"Yapılandırma Türü: {config.__class__.__name__}"]
    
    # Alt yapılandırmalar için kısa özetler
    nested_configs = []
    
    for key, value in config_dict.items():
        if isinstance(value, dict) and "__config_type__" in value:
            nested_configs.append(f"- {key}: {value['__config_type__']}")
    
    if nested_configs:
        summary_lines.append("\nAlt Yapılandırmalar:")
        summary_lines.extend(nested_configs)
    
    return "\n".join(summary_lines)


def compare_configs(config1: ConfigBase, config2: ConfigBase) -> Dict[str, Dict[str, Any]]:
    """
    İki yapılandırma nesnesini karşılaştırır ve farklılıkları belirler.
    
    Args:
        config1: İlk yapılandırma nesnesi
        config2: İkinci yapılandırma nesnesi
        
    Returns:
        Dict[str, Dict[str, Any]]: Farklılıkları içeren sözlük
            (parametre adı -> {'old': eski değer, 'new': yeni değer})
    """
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    # Özel meta alanları kaldır
    for d in [dict1, dict2]:
        d.pop("__config_type__", None)
        d.pop("__config_version__", None)
    
    differences = {}
    
    # Düz parametreleri karşılaştır (alt yapılandırmalar hariç)
    for key in set(dict1.keys()) | set(dict2.keys()):
        # Alt yapılandırma nesneleri için yinelemeli karşılaştırma gerekebilir
        # Burada sadece basit bir implementasyon gösterilmiştir
        if key in dict1 and key in dict2:
            if dict1[key] != dict2[key]:
                # Alt yapılandırmalar için yinelemeli karşılaştırma
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict) and \
                   "__config_type__" in dict1[key] and "__config_type__" in dict2[key]:
                    # Bu alt yapılandırmaların karşılaştırılması için daha karmaşık işlem gerekiyor
                    # Bu örnekte basitçe farklı olarak işaretliyoruz
                    differences[key] = {'old': '...', 'new': '...'}
                else:
                    differences[key] = {'old': dict1[key], 'new': dict2[key]}
        elif key in dict1:
            differences[key] = {'old': dict1[key], 'new': None}
        else:
            differences[key] = {'old': None, 'new': dict2[key]}
    
    return differences


def merge_configs(base_config: ConfigBase, override_config: Dict[str, Any]) -> ConfigBase:
    """
    Temel yapılandırma nesnesinin üzerine belirtilen değerleri uygular.
    
    Args:
        base_config: Temel yapılandırma nesnesi
        override_config: Üzerine yazılacak değerleri içeren sözlük
        
    Returns:
        ConfigBase: Birleştirilmiş yapılandırma nesnesi
    """
    # Basit bir yaklaşım: Sözlüğe dönüştür, override değerleri uygula, tekrar nesneye dönüştür
    config_dict = base_config.to_dict()
    
    # Düz parametreleri birleştir (alt yapılandırmalar için daha karmaşık işlem gerekebilir)
    for key, value in override_config.items():
        if "." in key:
            # Nokta notasyonuyla alt yapılandırmalara erişim (örn: "text_config.embed_dim")
            parts = key.split(".")
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config_dict[key] = value
    
    # Nesneye geri dönüştür
    return base_config.__class__.from_dict(config_dict)


T = TypeVar('T', bound=ConfigBase)

def load_config_from_file(config_class: Type[T], file_path: Union[str, Path]) -> T:
    """
    Dosyadan yapılandırma nesnesi yükler.
    
    Args:
        config_class: Yapılandırma sınıfı
        file_path: Dosya yolu
        
    Returns:
        T: Oluşturulan yapılandırma nesnesi
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        ValueError: Desteklenmeyen format veya doğrulama başarısız olursa
    """
    return config_class.load(file_path) 