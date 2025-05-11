"""
M³TM adaptör mekanizmaları.

Bu modül, ProtoTransformerBlock için temel adaptör mekanizmalarını içerir.
Adaptörler, çekirdek modele küçük, eğitilebilir parametre kümeleri ekleyerek
modeli özelleştirmeye olanak tanır.

Örüntüler:
- ConfigurationDataclass (PT-001): Yapılandırma parametrelerini dataclass olarak modelleme
- ModelComposite (PT-003): Ana modelin içine küçük ve özelleştirilmiş modül ekleme
- CompositeAdapter (PT-014): Birden fazla adaptörü sıralı olarak uygulama
"""

from typing import Dict, Tuple, Optional, List, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from m3tm.transformer.config import AdapterConfig


class AdapterSlot(nn.Module):
    """
    Gelişmiş adaptör yuva sınıfı.
    
    Bu sınıf, adaptörleri takıp çıkarmaya olanak sağlayan bir yuva mekanizması sunar.
    Adaptör yoksa, girdiyi doğrudan çıktıya iletir. Birden fazla adaptör eklendiğinde,
    bunları sıralı olarak uygular. Ayrıca eğitim ve çıkarım modları arasında geçiş yapabilir.
    
    Örüntüler:
    - ModelComposite (PT-003): Ana modelin içine küçük ve özelleştirilmiş modül ekleme
    - CompositeAdapter (PT-014): Birden fazla adaptörü sıralı olarak uygulama
    """
    
    def __init__(self, config: AdapterConfig, hidden_size: int):
        """
        Args:
            config: Adaptör yapılandırması
            hidden_size: Gizli durum boyutu
        """
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.bottleneck_dim = config.bottleneck_dim
        
        # Birden fazla adaptörü desteklemek için liste kullanılıyor
        self.adapters = nn.ModuleList([])
        self.adapter_names = []
        
        # Eğitim modu bayrağı
        self.training_mode = True
        
        # config sınıfları arasında uyumluluk: 'enabled' ve 'initial_adapter_type' özelliklerini kontrol et
        config_enabled = getattr(config, 'enabled', False)
        initial_adapter_type = getattr(config, 'initial_adapter_type', None)
        
        if config_enabled and initial_adapter_type is not None:
            adapter = self._create_adapter(initial_adapter_type)
            if adapter:
                self.adapters.append(adapter)
                self.adapter_names.append("default")
    
    def _create_adapter(self, adapter_type: str) -> Optional[nn.Module]:
        """Adaptör türüne göre uygun adaptör oluşturur.
        
        Args:
            adapter_type: Adaptör türü
            
        Returns:
            Oluşturulan adaptör modülü
        """
        if adapter_type == "bottleneck":
            return Adapter(self.hidden_size, self.bottleneck_dim, 
                           use_layer_norm=self.config.use_layer_norm,
                           activation=self.config.activation,
                           dropout=self.config.dropout,
                           init_scale=self.config.init_scale)
        elif adapter_type == "parallel":
            return ParallelAdapter(self.hidden_size, self.bottleneck_dim,
                                   scale=self.config.alpha,
                                   use_layer_norm=self.config.use_layer_norm,
                                   activation=self.config.activation,
                                   dropout=self.config.dropout,
                                   init_scale=self.config.init_scale)
        else:
            return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            Adaptör çıktısı, şekil (batch_size, seq_len, hidden_size)
        """
        # x bir sözlük ise, girdi olarak doğru değeri al
        if isinstance(x, dict):
            print(f"AdapterSlot input is dict with keys: {list(x.keys())}")
            attention_mask = None
            if 'attention_mask' in x:
                attention_mask = x['attention_mask']  # Eğer kullanmamız gerekirse sakla
            
            if 'hidden_states' in x:
                x = x['hidden_states']
                print(f"  Using 'hidden_states' key, type: {type(x)}")
            elif 'embeddings' in x:
                x = x['embeddings']
                print(f"  Using 'embeddings' key, type: {type(x)}")
            elif len(x) == 1:  # Tek bir anahtar varsa, değeri doğrudan al
                key = list(x.keys())[0]
                print(f"  Using single key: {key}, type: {type(x[key])}")
                x = list(x.values())[0]
        
        # Performans için hızlı kontrol: Eğer adaptör yoksa veya eğitim modu kapalıysa ve çıkarım modundaysak
        # doğrudan girdiyi döndür
        if not self.adapters or (not self.training_mode and not self.training):
            return x
        
        # Adaptörleri sıralı olarak uygula
        output = x
        for i, adapter in enumerate(self.adapters):
            if isinstance(output, dict):
                print(f"  Output is dict before adapter {i} call with keys: {list(output.keys())}")
            output = adapter(output)
        
        return output
    
    def register_adapter(self, adapter: nn.Module, name: str = None) -> bool:
        """Yeni bir adaptör kaydeder.
        
        Args:
            adapter: Kaydedilecek adaptör
            name: Adaptör adı (None ise otomatik oluşturulur)
            
        Returns:
            Kayıt başarılı ise True, değilse False
        """
        if name is None:
            name = f"adapter_{len(self.adapters)}"
        
        # Aynı isimde adaptör var mı kontrol et
        if name in self.adapter_names:
            return False
        
        self.adapters.append(adapter)
        self.adapter_names.append(name)
        return True
    
    def remove_adapter(self, name: str = None) -> bool:
        """Belirli bir adaptörü kaldırır.
        
        Args:
            name: Kaldırılacak adaptör adı (None ise son eklenen adaptör kaldırılır)
            
        Returns:
            Kaldırma başarılı ise True, değilse False
        """
        if len(self.adapters) == 0:
            return False
        
        if name is None:
            # Son adaptörü kaldır
            self.adapters.pop()
            self.adapter_names.pop()
            return True
        
        # İsme göre adaptörü bul ve kaldır
        if name in self.adapter_names:
            idx = self.adapter_names.index(name)
            self.adapter_names.pop(idx)
            # ModuleList'ten doğrudan öğe kaldırmak için yeni liste oluştur
            new_adapters = nn.ModuleList()
            for i, adapter in enumerate(self.adapters):
                if i != idx:
                    new_adapters.append(adapter)
            self.adapters = new_adapters
            return True
        
        return False
    
    def get_adapter(self, name: str = None) -> Optional[nn.Module]:
        """İsimle bir adaptörü alır.
        
        Args:
            name: Adaptör adı (None ise son eklenen adaptör döndürülür)
            
        Returns:
            Adaptör modülü veya None
        """
        if len(self.adapters) == 0:
            return None
        
        if name is None:
            return self.adapters[-1]
        
        if name in self.adapter_names:
            idx = self.adapter_names.index(name)
            return self.adapters[idx]
        
        return None
    
    def has_adapter(self, name: str = None) -> bool:
        """Belirli bir adaptörün olup olmadığını kontrol eder.
        
        Args:
            name: Kontrol edilecek adaptör adı (None ise herhangi bir adaptör olup olmadığı)
            
        Returns:
            Adaptör varsa True, yoksa False
        """
        if name is None:
            return len(self.adapters) > 0
        return name in self.adapter_names
    
    def get_adapter_names(self) -> List[str]:
        """Kayıtlı adaptör isimlerini döndürür."""
        return self.adapter_names.copy()
    
    def set_training_mode(self, mode: bool) -> None:
        """Eğitim modunu ayarlar.
        
        Args:
            mode: True ise eğitim modu, False ise çıkarım modu
        """
        self.training_mode = mode

    def count_parameters(self) -> int:
        """Tüm adaptörlerdeki toplam eğitilebilir parametre sayısını hesaplar."""
        return sum(sum(p.numel() for p in adapter.parameters() if p.requires_grad) 
                  for adapter in self.adapters)


class Adapter(nn.Module):
    """
    Temel adaptör implementasyonu.
    
    Referans: Parameter-Efficient Transfer Learning for NLP
    https://arxiv.org/abs/1902.00751
    
    Bu sınıf, basit bir darboğaz adaptörü uygular.
    """
    
    def __init__(self, hidden_size: int, bottleneck_dim: int, 
                 use_layer_norm: bool = True, 
                 activation: str = "gelu",
                 dropout: float = 0.0,
                 init_scale: float = 1e-3):
        """
        Args:
            hidden_size: Gizli durum boyutu
            bottleneck_dim: Darboğaz boyutu
            use_layer_norm: Layer normalization kullanımı
            activation: Aktivasyon fonksiyonu
            dropout: Dropout oranı
            init_scale: Başlangıç ölçeği
        """
        super().__init__()
        self._down_project = nn.Linear(hidden_size, bottleneck_dim)
        self._up_project = nn.Linear(bottleneck_dim, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(bottleneck_dim) if use_layer_norm else None
        
        # Aktivasyon fonksiyonu
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self._init_weights(init_scale)
    
    def _get_activation(self, activation_name):
        """Aktivasyon fonksiyonunu adına göre döndürür."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name in ["swish", "silu"]:
            return nn.SiLU()
        elif activation_name == "hardswish":
            return nn.Hardswish()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        else:
            # Varsayılan olarak GELU kullan
            return nn.GELU()
    
    def _init_weights(self, init_scale) -> None:
        """Ağırlıkları başlat."""
        nn.init.normal_(self._down_project.weight, std=init_scale)
        nn.init.normal_(self._up_project.weight, std=init_scale)
        nn.init.zeros_(self._down_project.bias)
        nn.init.zeros_(self._up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            Adaptor çıktısı, şekil (batch_size, seq_len, hidden_size)
        """
        # x bir sözlük ise, girdi olarak doğru değeri al
        if isinstance(x, dict):
            if 'hidden_states' in x:
                x = x['hidden_states']
            elif 'embeddings' in x:
                x = x['embeddings']
            elif len(x) == 1:  # Tek bir anahtar varsa, değeri doğrudan al
                x = list(x.values())[0]
        
        # Down projection
        residual = x
        hidden_states = self._down_project(x)
        
        # Layer normalization (if enabled)
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        
        # Activation
        hidden_states = self.activation(hidden_states)
        
        # Dropout (if enabled)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        
        # Up projection
        output = self._up_project(hidden_states)
        
        # Residual connection
        output = output + residual
        
        return output
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ParallelAdapter(nn.Module):
    """
    Paralel adaptör implementasyonu.
    
    Referans: MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer
    https://arxiv.org/abs/2005.00052
    
    Bu sınıf, paralel adaptör mimarisini uygular, burada adaptör çıktısı
    ve normal çıktı doğrudan birleştirilir.
    """
    
    def __init__(self, hidden_size: int, bottleneck_dim: int, 
                 scale: float = 1.0,
                 use_layer_norm: bool = True,
                 activation: str = "gelu",
                 dropout: float = 0.0,
                 init_scale: float = 1e-3):
        """
        Args:
            hidden_size: Gizli durum boyutu
            bottleneck_dim: Darboğaz boyutu
            scale: Ölçekleme faktörü
            use_layer_norm: Layer normalization kullanımı
            activation: Aktivasyon fonksiyonu
            dropout: Dropout oranı
            init_scale: Başlangıç ölçeği
        """
        super().__init__()
        self.scale = scale
        self.layer_norm = nn.LayerNorm(hidden_size)
        self._down_project = nn.Linear(hidden_size, bottleneck_dim)
        self._up_project = nn.Linear(bottleneck_dim, hidden_size)
        
        # Aktivasyon fonksiyonu
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self._init_weights(init_scale)
    
    def _get_activation(self, activation_name):
        """Aktivasyon fonksiyonunu adına göre döndürür."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name in ["swish", "silu"]:
            return nn.SiLU()
        elif activation_name == "hardswish":
            return nn.Hardswish()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        else:
            # Varsayılan olarak GELU kullan
            return nn.GELU()
    
    def _init_weights(self, init_scale) -> None:
        """Ağırlıkları başlat."""
        nn.init.normal_(self._down_project.weight, std=init_scale)
        nn.init.normal_(self._up_project.weight, std=init_scale)
        nn.init.zeros_(self._down_project.bias)
        nn.init.zeros_(self._up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            Adaptör çıktısı, şekil (batch_size, seq_len, hidden_size)
        """
        # x bir sözlük ise, girdi olarak doğru değeri al
        if isinstance(x, dict):
            if 'hidden_states' in x:
                x = x['hidden_states']
            elif 'embeddings' in x:
                x = x['embeddings']
            elif len(x) == 1:  # Tek bir anahtar varsa, değeri doğrudan al
                x = list(x.values())[0]
        
        # Layer normalization
        hidden_states = self.layer_norm(x)
        
        # Down projection
        hidden_states = self._down_project(hidden_states)
        
        # Activation
        hidden_states = self.activation(hidden_states)
        
        # Dropout (if enabled)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        
        # Up projection
        hidden_states = self._up_project(hidden_states)
        
        # Scaled addition
        output = x + self.scale * hidden_states
        
        return output
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_adapter_slots(config: AdapterConfig, hidden_size: int, positions: List[str]) -> Dict[str, AdapterSlot]:
    """Belirtilen pozisyonlarda adaptör yuvaları oluşturur.
    
    Args:
        config: Adaptör yapılandırması
        hidden_size: Gizli durum boyutu
        positions: Adaptör yuvası pozisyonları
        
    Returns:
        Pozisyon-AdapterSlot sözlüğü
    """
    slots = {}
    for pos in positions:
        slots[pos] = AdapterSlot(config, hidden_size)
    return slots 