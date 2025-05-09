"""
Yapılandırma Yönetimi

Bu modül, yapılandırma kaynakları ve değişiklikleri izleyen mekanizmalar sağlar.
MechanismRegistry (PT-007) örüntüsünü yapılandırma kaynakları için uygular.
ConfigurationObserver potansiyel örüntüsünü de içerir.
"""

import os
import json
import yaml
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable, Set, Union, TypeVar, Type, Callable

from .config_base import ConfigBase


class ConfigSource(Enum):
    """Yapılandırma kaynağı türleri."""
    FILE_JSON = auto()
    FILE_YAML = auto()
    ENVIRONMENT = auto()
    COMMAND_LINE = auto()
    MEMORY = auto()


@runtime_checkable
class ConfigChangeListener(Protocol):
    """
    Yapılandırma değişikliklerini dinleyen protokol.
    
    Bu protokolü uygulayan sınıflar, yapılandırma değişikliklerini
    otomatik olarak işleyebilir.
    
    Potansiyel Örüntü: ConfigurationObserver
    """
    def on_config_changed(self, parameter: str, old_value: Any, new_value: Any) -> None:
        """
        Yapılandırma değiştiğinde çağrılır.
        
        Args:
            parameter: Değişen parametre adı (nokta ile ayrılmış)
            old_value: Parametrenin eski değeri
            new_value: Parametrenin yeni değeri
        """
        ...


T = TypeVar('T', bound=ConfigBase)


class ConfigManager:
    """
    Yapılandırma yönetimi sınıfı.
    
    Bu sınıf, farklı kaynaklardan yapılandırma yüklemeyi, değişiklikleri
    izlemeyi ve ilgili bileşenlere bildirmeyi sağlar.
    
    Örüntüler:
    - MechanismRegistry (PT-007): Farklı yapılandırma kaynaklarını merkezi bir sistemde yönetir
    - ConfigurationObserver: Yapılandırma değişikliklerini dinleyicilere bildirir
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton örüntüsü uygular."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """ConfigManager örneğini başlatır."""
        if self._initialized:
            return
            
        self._config_sources: Dict[str, ConfigSource] = {}
        self._configs: Dict[str, ConfigBase] = {}
        self._listeners: Dict[str, Set[ConfigChangeListener]] = {}
        self._source_loaders: Dict[ConfigSource, Callable] = {
            ConfigSource.FILE_JSON: self._load_from_json_file,
            ConfigSource.FILE_YAML: self._load_from_yaml_file,
            ConfigSource.ENVIRONMENT: self._load_from_environment,
            ConfigSource.COMMAND_LINE: self._load_from_command_line,
            ConfigSource.MEMORY: lambda *args, **kwargs: None  # Bellek zaten yüklü
        }
        
        self._initialized = True
    
    def register_config(self, name: str, config: ConfigBase, source: ConfigSource = ConfigSource.MEMORY, 
                       source_path: Optional[str] = None) -> None:
        """
        Yapılandırma nesnesini kaydeder.
        
        Args:
            name: Yapılandırma adı
            config: Yapılandırma nesnesi
            source: Yapılandırma kaynağı
            source_path: Kaynak dosya yolu (dosya kaynakları için)
        """
        self._configs[name] = config
        self._config_sources[name] = source
        
        if source != ConfigSource.MEMORY and source_path:
            # Kaynaktan yükle (dosya, çevre değişkenleri vb.)
            self._load_from_source(name, config.__class__, source, source_path)
    
    def get_config(self, name: str) -> Optional[ConfigBase]:
        """
        İsme göre yapılandırma nesnesini döndürür.
        
        Args:
            name: Yapılandırma adı
            
        Returns:
            Optional[ConfigBase]: Yapılandırma nesnesi veya None
        """
        return self._configs.get(name)
    
    def update_config(self, name: str, updates: Dict[str, Any]) -> None:
        """
        Yapılandırma değerlerini günceller.
        
        Args:
            name: Yapılandırma adı
            updates: Güncellenecek değerleri içeren sözlük
            
        Raises:
            KeyError: Belirtilen isimde yapılandırma bulunamazsa
        """
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        
        config = self._configs[name]
        
        # Değerleri güncelle ve değişiklikleri bildir
        for key, new_value in updates.items():
            if hasattr(config, key):
                old_value = getattr(config, key)
                if old_value != new_value:
                    setattr(config, key, new_value)
                    self._notify_listeners(f"{name}.{key}", old_value, new_value)
    
    def save_config(self, name: str, file_path: Union[str, Path], format: str = "json") -> None:
        """
        Yapılandırmayı dosyaya kaydeder.
        
        Args:
            name: Yapılandırma adı
            file_path: Kaydedilecek dosya yolu
            format: Kaydetme formatı ('json' veya 'yaml')
            
        Raises:
            KeyError: Belirtilen isimde yapılandırma bulunamazsa
        """
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        
        config = self._configs[name]
        config.save(file_path, format)
    
    def add_listener(self, parameter_path: str, listener: ConfigChangeListener) -> None:
        """
        Yapılandırma değişikliği dinleyicisi ekler.
        
        Args:
            parameter_path: Dinlenecek parametre yolu (örn: "model_config.learning_rate")
            listener: Dinleyici nesne
        """
        if parameter_path not in self._listeners:
            self._listeners[parameter_path] = set()
        self._listeners[parameter_path].add(listener)
    
    def remove_listener(self, parameter_path: str, listener: ConfigChangeListener) -> None:
        """
        Yapılandırma değişikliği dinleyicisini kaldırır.
        
        Args:
            parameter_path: Parametre yolu
            listener: Kaldırılacak dinleyici
        """
        if parameter_path in self._listeners:
            if listener in self._listeners[parameter_path]:
                self._listeners[parameter_path].remove(listener)
                
            # Boş kümeyi temizle
            if not self._listeners[parameter_path]:
                del self._listeners[parameter_path]
    
    def _notify_listeners(self, parameter: str, old_value: Any, new_value: Any) -> None:
        """
        Yapılandırma değişikliği dinleyicilerine bildirim yapar.
        
        Args:
            parameter: Değişen parametre adı
            old_value: Eski değer
            new_value: Yeni değer
        """
        # Tam parametreye özel dinleyiciler
        if parameter in self._listeners:
            for listener in self._listeners[parameter]:
                listener.on_config_changed(parameter, old_value, new_value)
        
        # Prefix eşleşmelerine bak (örn: "model_config." ile başlayan tüm parametreler)
        for prefix, listeners in self._listeners.items():
            if prefix.endswith('.*') and parameter.startswith(prefix[:-2] + '.'):
                for listener in listeners:
                    listener.on_config_changed(parameter, old_value, new_value)
    
    def _load_from_source(self, name: str, config_class: Type[T], source: ConfigSource, 
                         source_path: str) -> None:
        """
        Belirtilen kaynaktan yapılandırma yükler.
        
        Args:
            name: Yapılandırma adı
            config_class: Yapılandırma sınıfı
            source: Kaynak türü
            source_path: Kaynak yolu veya öneki
        """
        if source in self._source_loaders:
            self._source_loaders[source](name, config_class, source_path)
    
    def _load_from_json_file(self, name: str, config_class: Type[T], file_path: str) -> None:
        """JSON dosyasından yapılandırma yükler."""
        try:
            config = config_class.load(file_path, format="json")
            self._configs[name] = config
        except Exception as e:
            # Hata durumunda logla, mevcut yapılandırmayı değiştirme
            # Gerçek uygulamada daha iyi bir hata yönetimi kullanılmalı
            print(f"Error loading configuration '{name}' from JSON file: {e}")
    
    def _load_from_yaml_file(self, name: str, config_class: Type[T], file_path: str) -> None:
        """YAML dosyasından yapılandırma yükler."""
        try:
            config = config_class.load(file_path, format="yaml")
            self._configs[name] = config
        except Exception as e:
            print(f"Error loading configuration '{name}' from YAML file: {e}")
    
    def _load_from_environment(self, name: str, config_class: Type[T], prefix: str) -> None:
        """
        Çevre değişkenlerinden yapılandırma yükler.
        
        Örn: prefix=MODEL_CONFIG, MODEL_CONFIG_LEARNING_RATE=0.01 için
        config.learning_rate=0.01 olarak ayarlanır.
        """
        updates = {}
        prefix_upper = prefix.upper() + "_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix_upper):
                param_name = key[len(prefix_upper):].lower()
                # Nokta içeren parametre adlarını işle
                if '.' in param_name:
                    # Karmaşık nested yapılandırma - bu basit implementasyonda desteklenmiyor
                    # Gerçek uygulamada burada daha karmaşık bir mantık gerekebilir
                    continue
                
                # Basit tip dönüşümü
                if value.lower() == 'true':
                    parsed_value = True
                elif value.lower() == 'false':
                    parsed_value = False
                elif value.isdigit():
                    parsed_value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                    parsed_value = float(value)
                else:
                    parsed_value = value
                
                updates[param_name] = parsed_value
        
        # Yapılandırma nesnesini güncelle
        if name in self._configs and updates:
            self.update_config(name, updates)
    
    def _load_from_command_line(self, name: str, config_class: Type[T], prefix: str) -> None:
        """
        Komut satırı argümanlarından yapılandırma yükler.
        
        Bu, basit bir uygulama - gerçek uygulamada argparse veya click gibi
        kütüphanelerle daha kapsamlı bir çözüm kullanılabilir.
        """
        # Bu basit implementasyon bir örnek olarak eklendi
        # Gerçek bir uygulama uygulandığında değiştirilecek
        pass 