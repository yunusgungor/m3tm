"""
M³TM adapter_manager modülü.

Bu modül, modelin farklı kısımlarına takılan adapter'ların yönetimi için
bir arabirim sağlar. Manager, adapter'ların kaydedilmesi, kaldırılması,
etkinleştirilmesi veya devre dışı bırakılması gibi işlemleri kolaylaştırır.

Örüntüler:
- FactoryMethod (PT-002): Farklı adapter türlerini oluşturmak için fabrika metodu
- MetricsCollector (PT-008): Adapter parametrelerini toplamak ve izlemek için
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import os
import json

import torch
import torch.nn as nn

from m3tm.adapters.adapter import AdapterConfig, Adapter, create_adapter


@dataclass
class AdapterRegistration:
    """Bir adapter'ın kayıt bilgilerini tutan veri sınıfı."""
    
    adapter_name: str  # Adapter'ın benzersiz adı
    module_name: str  # Adapter'ın takıldığı modül
    position: str  # Adapter'ın modül içindeki konumu (ör: "pre_attention", "post_ffn")
    adapter_config: AdapterConfig  # Adapter yapılandırması
    adapter_id: str = ""  # Benzersiz adapter tanımlayıcısı (opsiyonel)
    is_active: bool = True  # Adapter'ın etkinlik durumu
    metadata: Dict[str, Any] = field(default_factory=dict)  # Ek bilgiler
    
    def __post_init__(self):
        """Yapılandırma sonrası başlatma."""
        if not self.adapter_id:
            # Benzersiz bir adapter_id oluştur
            self.adapter_id = f"{self.adapter_name}_{self.module_name}_{self.position}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Kayıt bilgilerini sözlük olarak döndürür."""
        return {
            "adapter_id": self.adapter_id,
            "adapter_name": self.adapter_name,
            "module_name": self.module_name,
            "position": self.position,
            "adapter_type": self.adapter_config.adapter_type,
            "bottleneck_dim": self.adapter_config.bottleneck_dim,
            "is_active": self.is_active,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: AdapterConfig) -> 'AdapterRegistration':
        """Sözlükten AdapterRegistration nesnesi oluşturur."""
        return cls(
            adapter_id=data.get("adapter_id", ""),
            adapter_name=data["adapter_name"],
            module_name=data["module_name"],
            position=data["position"],
            adapter_config=config,
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {})
        )


class AdapterManager:
    """
    Model adaptörlerini yöneten sınıf.
    
    Bu sınıf, bir model içindeki tüm adaptörleri ve bağlantı noktalarını
    takip eder, yönetir ve gerektiğinde etkinleştirir veya devre dışı bırakır.
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: Adaptörlerin ekleneceği model
        """
        self.model = model
        self.registered_adapters: Dict[str, AdapterRegistration] = {}  # adapter_id -> registration
        self.adapter_instances: Dict[str, Adapter] = {}  # adapter_id -> adapter instance
        self.module_registry: Dict[str, List[Tuple[str, str]]] = {}  # module_name -> [(adapter_id, position)]
    
    def register_adapter(self, 
                        module: nn.Module, 
                        adapter_name: str, 
                        position: str, 
                        config: AdapterConfig,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Bir modüle adapter kaydeder.
        
        Args:
            module: Adapter'ın ekleneceği modül
            adapter_name: Adapter'ın adı
            position: Adapter'ın modül içindeki konumu
            config: Adapter yapılandırması
            metadata: Adapter ile ilgili ek bilgiler
            
        Returns:
            Başarılı ise adapter_id, başarısız ise None
        """
        # Modül adını al
        module_name = self._get_module_name(module)
        
        # Desteklenen bir pozisyon mu kontrol et
        if not hasattr(module, "register_adapter") or not self._is_valid_position(module, position):
            return None
        
        # Kayıt oluştur
        registration = AdapterRegistration(
            adapter_name=adapter_name,
            module_name=module_name,
            position=position,
            adapter_config=config,
            metadata=metadata or {}
        )
        adapter_id = registration.adapter_id
        
        # Adapter'ı oluştur
        input_dim = self._get_module_input_dim(module)
        adapter = create_adapter(config, input_dim)
        
        # Modüle adapter'ı kaydet
        if not module.register_adapter(adapter, position):
            return None
        
        # Yönetici kayıtlarını güncelle
        self.registered_adapters[adapter_id] = registration
        self.adapter_instances[adapter_id] = adapter
        
        # Modül kayıtlarını güncelle
        if module_name not in self.module_registry:
            self.module_registry[module_name] = []
        self.module_registry[module_name].append((adapter_id, position))
        
        return adapter_id
    
    def remove_adapter(self, adapter_id: str) -> bool:
        """
        Belirli bir adapter'ı kaldırır.
        
        Args:
            adapter_id: Kaldırılacak adapter'ın ID'si
            
        Returns:
            İşlem başarılı ise True, değilse False
        """
        if adapter_id not in self.registered_adapters:
            return False
        
        # Kayıt bilgilerini al
        registration = self.registered_adapters[adapter_id]
        module_name = registration.module_name
        position = registration.position
        
        # Modülü bul
        module = self._find_module_by_name(module_name)
        if module is None or not hasattr(module, "remove_adapter"):
            return False
        
        # Adapter'ı kaldır
        if not module.remove_adapter(position):
            return False
        
        # Kayıtları güncelle
        del self.registered_adapters[adapter_id]
        del self.adapter_instances[adapter_id]
        
        # Modül kayıtlarını güncelle
        if module_name in self.module_registry:
            self.module_registry[module_name] = [
                (aid, pos) for aid, pos in self.module_registry[module_name]
                if aid != adapter_id
            ]
            
        return True
    
    def activate_adapter(self, adapter_id: str) -> bool:
        """
        Belirli bir adapter'ı etkinleştirir.
        
        Args:
            adapter_id: Etkinleştirilecek adapter'ın ID'si
            
        Returns:
            İşlem başarılı ise True, değilse False
        """
        if adapter_id not in self.registered_adapters:
            return False
        
        registration = self.registered_adapters[adapter_id]
        if registration.is_active:
            return True  # Zaten etkin
        
        # Modülü bul
        module = self._find_module_by_name(registration.module_name)
        if module is None:
            return False
        
        # Adapter'ı bul
        adapter = self.adapter_instances.get(adapter_id)
        if adapter is None:
            return False
        
        # Adapter'ı etkinleştir
        if not hasattr(module, "register_adapter") or not module.register_adapter(adapter, registration.position):
            return False
        
        # Kayıt bilgilerini güncelle
        registration.is_active = True
        return True
    
    def deactivate_adapter(self, adapter_id: str) -> bool:
        """
        Belirli bir adapter'ı devre dışı bırakır (silmeden).
        
        Args:
            adapter_id: Devre dışı bırakılacak adapter'ın ID'si
            
        Returns:
            İşlem başarılı ise True, değilse False
        """
        if adapter_id not in self.registered_adapters:
            return False
        
        registration = self.registered_adapters[adapter_id]
        if not registration.is_active:
            return True  # Zaten devre dışı
        
        # Modülü bul
        module = self._find_module_by_name(registration.module_name)
        if module is None or not hasattr(module, "remove_adapter"):
            return False
        
        # Adapter'ı devre dışı bırak
        if not module.remove_adapter(registration.position):
            return False
        
        # Kayıt bilgilerini güncelle
        registration.is_active = False
        return True
    
    def get_adapter_info(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Belirli bir adapter hakkında bilgi döndürür.
        
        Args:
            adapter_id: Bilgileri alınacak adapter'ın ID'si
            
        Returns:
            Adapter bilgilerini içeren sözlük veya None
        """
        if adapter_id not in self.registered_adapters:
            return None
        
        registration = self.registered_adapters[adapter_id]
        adapter = self.adapter_instances.get(adapter_id)
        
        if adapter is None:
            return None
        
        info = registration.to_dict()
        info.update({
            "parameter_count": adapter.count_parameters(),
            "model_percentage": self._get_model_percentage(adapter) if adapter else 0.0
        })
        
        return info
    
    def list_adapters(self, active_only: bool = False) -> List[str]:
        """
        Kayıtlı adapter ID'lerinin listesini döndürür.
        
        Args:
            active_only: Sadece etkin adapter'ları listelemek için True
            
        Returns:
            Adapter ID'lerinin listesi
        """
        if active_only:
            return [
                adapter_id for adapter_id, registration in self.registered_adapters.items()
                if registration.is_active
            ]
        return list(self.registered_adapters.keys())
    
    def get_module_adapters(self, module_name: str) -> List[str]:
        """
        Belirli bir modüldeki tüm adapter'ların ID'lerini döndürür.
        
        Args:
            module_name: Modül adı
            
        Returns:
            Adapter ID'lerinin listesi
        """
        if module_name not in self.module_registry:
            return []
        
        return [adapter_id for adapter_id, _ in self.module_registry[module_name]]
    
    def get_adapters_by_name(self, adapter_name: str) -> List[str]:
        """
        Belirli bir ada sahip tüm adapter'ların ID'lerini döndürür.
        
        Args:
            adapter_name: Adapter adı
            
        Returns:
            Adapter ID'lerinin listesi
        """
        return [
            adapter_id for adapter_id, registration in self.registered_adapters.items()
            if registration.adapter_name == adapter_name
        ]
    
    def save_adapters(self, save_dir: str, adapter_ids: Optional[List[str]] = None) -> bool:
        """
        Belirtilen adapter'ları kaydeder.
        
        Args:
            save_dir: Kayıt dizini
            adapter_ids: Kaydedilecek adapter ID'leri (None ise tümü)
            
        Returns:
            İşlem başarılı ise True, değilse False
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Hangi adapter'ların kaydedileceğini belirle
        if adapter_ids is None:
            adapter_ids = list(self.registered_adapters.keys())
        else:
            # Sadece var olan ID'leri filtrele
            adapter_ids = [aid for aid in adapter_ids if aid in self.registered_adapters]
        
        if not adapter_ids:
            return False
        
        # Adapter kayıt bilgilerini kaydet
        registrations = {
            adapter_id: self.registered_adapters[adapter_id].to_dict()
            for adapter_id in adapter_ids
        }
        
        with open(os.path.join(save_dir, "adapter_registry.json"), "w") as f:
            json.dump(registrations, f, indent=2)
        
        # Adapter model durumlarını kaydet
        for adapter_id in adapter_ids:
            adapter = self.adapter_instances.get(adapter_id)
            if adapter is not None:
                torch.save(
                    adapter.state_dict(),
                    os.path.join(save_dir, f"{adapter_id}.pt")
                )
        
        return True
    
    def load_adapters(self, load_dir: str, adapter_ids: Optional[List[str]] = None) -> List[str]:
        """
        Belirtilen adapter'ları yükler.
        
        Args:
            load_dir: Yükleme dizini
            adapter_ids: Yüklenecek adapter ID'leri (None ise tümü)
            
        Returns:
            Başarıyla yüklenen adapter ID'lerinin listesi
        """
        registry_path = os.path.join(load_dir, "adapter_registry.json")
        if not os.path.exists(registry_path):
            return []
        
        # Kayıt bilgilerini yükle
        with open(registry_path, "r") as f:
            registrations = json.load(f)
        
        # Hangi adapter'ların yükleneceğini belirle
        if adapter_ids is None:
            adapter_ids = list(registrations.keys())
        else:
            # Sadece kayıtta olan ID'leri filtrele
            adapter_ids = [aid for aid in adapter_ids if aid in registrations]
        
        loaded_adapters = []
        
        # Her bir adapter'ı yükle
        for adapter_id in adapter_ids:
            # Adapter yapılandırmasını yükle
            adapter_data = registrations[adapter_id]
            adapter_config = AdapterConfig(
                adapter_type=adapter_data.get("adapter_type", "bottleneck"),
                bottleneck_dim=adapter_data.get("bottleneck_dim", 16)
            )
            
            # Kayıt bilgileri
            registration = AdapterRegistration.from_dict(adapter_data, adapter_config)
            
            # Modülü bul
            module = self._find_module_by_name(registration.module_name)
            if module is None:
                continue
            
            # Adapter'ı oluştur
            input_dim = self._get_module_input_dim(module)
            adapter = create_adapter(adapter_config, input_dim)
            
            # Adapter durumunu yükle
            adapter_path = os.path.join(load_dir, f"{adapter_id}.pt")
            if not os.path.exists(adapter_path):
                continue
                
            adapter.load_state_dict(torch.load(adapter_path))
            
            # Modüle adapter'ı kaydet (aktifse)
            if registration.is_active:
                if not hasattr(module, "register_adapter") or not module.register_adapter(adapter, registration.position):
                    continue
            
            # Yönetici kayıtlarını güncelle
            self.registered_adapters[adapter_id] = registration
            self.adapter_instances[adapter_id] = adapter
            
            # Modül kayıtlarını güncelle
            if registration.module_name not in self.module_registry:
                self.module_registry[registration.module_name] = []
            self.module_registry[registration.module_name].append((adapter_id, registration.position))
            
            loaded_adapters.append(adapter_id)
        
        return loaded_adapters
    
    def get_adapter_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """
        Tüm aktif adapter'ların parametrelerini döndürür.
        
        Returns:
            Adapter parametrelerini içeren sözlük (adapter_id -> parametre listesi)
        """
        return {
            adapter_id: list(adapter.parameters())
            for adapter_id, adapter in self.adapter_instances.items()
            if self.registered_adapters[adapter_id].is_active
        }
    
    def merge_adapter_to_model(self, adapter_id: str) -> bool:
        """
        Belirli bir adapter'ı ana modele birleştirir (desteklenirse).
        
        Args:
            adapter_id: Birleştirilecek adapter'ın ID'si
            
        Returns:
            İşlem başarılı ise True, değilse False
        """
        # Bu işlevi uygulamak gelecek çalışmalara bırakılmıştır
        # Mevcut durumda adapters'ı temel modelden ayrı tutmayı tercih ediyoruz
        return False
    
    def _get_module_name(self, module: nn.Module) -> str:
        """Modülün adını döndürür."""
        if hasattr(module, "name"):
            return module.name
        return module.__class__.__name__
    
    def _find_module_by_name(self, module_name: str) -> Optional[nn.Module]:
        """Adına göre modülü bulur."""
        # İsimlendirme sözleşmesine göre model içinde modülü ara
        if hasattr(self.model, module_name):
            return getattr(self.model, module_name)
        
        # İç içe modüller arasında ara
        for name, module in self.model.named_modules():
            if name == module_name or self._get_module_name(module) == module_name:
                return module
        
        return None
    
    def _get_module_input_dim(self, module: nn.Module) -> int:
        """Modülün girdi boyutunu tahmin eder."""
        if hasattr(module, "hidden_size"):
            return module.hidden_size
        if hasattr(module, "config") and hasattr(module.config, "hidden_size"):
            return module.config.hidden_size
        if hasattr(module, "input_dim"):
            return module.input_dim
            
        # Varsayılan bir değer (ileride daha sofistike bir yaklaşım gerekebilir)
        return 256
    
    def _is_valid_position(self, module: nn.Module, position: str) -> bool:
        """Pozisyonun geçerli olup olmadığını kontrol eder."""
        # SimpleModule için test adaptasyonu
        if hasattr(module, "_is_valid_position") and callable(module._is_valid_position):
            return module._is_valid_position(position)
            
        # get_adapter_positions aracılığıyla kontrol et
        positions = get_adapter_positions(module)
        return position in positions
    
    def _get_model_percentage(self, adapter: Adapter) -> float:
        """Adapter parametre sayısının modele göre yüzdesini döndürür."""
        adapter_params = adapter.count_parameters()
        
        # Model toplam parametre sayısını hesapla
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if model_params == 0:
            return 0.0
            
        return (adapter_params / model_params) * 100.0
    
    def summary(self) -> Dict[str, Any]:
        """
        Adapter yöneticisinin özet bilgilerini döndürür.
        
        Returns:
            Özet bilgileri içeren sözlük
        """
        active_adapters = [aid for aid in self.registered_adapters if self.registered_adapters[aid].is_active]
        
        # Adapter parametreleri
        total_adapter_params = sum(
            adapter.count_parameters() for adapter in self.adapter_instances.values()
            if self.registered_adapters[list(self.adapter_instances.keys())[list(self.adapter_instances.values()).index(adapter)]].is_active
        )
        
        # Model toplam parametre sayısı
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_adapters": len(self.registered_adapters),
            "active_adapters": len(active_adapters),
            "total_adapter_parameters": total_adapter_params,
            "model_parameters": model_params,
            "adapter_percentage": (total_adapter_params / model_params * 100) if model_params > 0 else 0.0,
            "modules_with_adapters": list(self.module_registry.keys()),
            "adapter_types": {
                adapter_id: self.registered_adapters[adapter_id].adapter_config.adapter_type
                for adapter_id in self.registered_adapters
            }
        } 