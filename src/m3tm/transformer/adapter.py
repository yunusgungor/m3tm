"""
M³TM adaptör mekanizmaları.

Bu modül, ProtoTransformerBlock için temel adaptör mekanizmalarını içerir.
Adaptörler, çekirdek modele küçük, eğitilebilir parametre kümeleri ekleyerek
modeli özelleştirmeye olanak tanır.

Not: Bu, hikaye_6 için temel bir implementasyondur. Daha kapsamlı implementasyon hikaye_12'de yapılacaktır.

Örüntüler:
- ConfigurationDataclass (PT-001): Yapılandırma parametrelerini dataclass olarak modelleme
"""

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from m3tm.transformer.config import AdapterConfig


class AdapterSlot(nn.Module):
    """
    Temel adaptör yuva sınıfı.
    
    Bu sınıf, adaptörleri takıp çıkarmaya olanak sağlayan bir yuva mekanizması sunar.
    Adaptör yoksa, girdiyi doğrudan çıktıya iletir.
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
        
        self.adapter = None
        if config.enabled:
            self.adapter = Adapter(hidden_size, self.bottleneck_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            Adaptör çıktısı, şekil (batch_size, seq_len, hidden_size)
        """
        if self.adapter is not None:
            return self.adapter(x)
        return x
    
    def register_adapter(self, adapter: nn.Module) -> None:
        """Yeni bir adaptör kaydeder."""
        self.adapter = adapter
    
    def remove_adapter(self) -> None:
        """Mevcut adaptörü kaldırır."""
        self.adapter = None
    
    def has_adapter(self) -> bool:
        """Bir adaptör takılı olup olmadığını kontrol eder."""
        return self.adapter is not None


class Adapter(nn.Module):
    """
    Temel adaptör implementasyonu.
    
    Referans: Parameter-Efficient Transfer Learning for NLP
    https://arxiv.org/abs/1902.00751
    
    Bu sınıf, basit bir darboğaz adaptörü uygular.
    """
    
    def __init__(self, hidden_size: int, bottleneck_dim: int):
        """
        Args:
            hidden_size: Gizli durum boyutu
            bottleneck_dim: Darboğaz boyutu
        """
        super().__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)
        
        self.layer_norm = nn.LayerNorm(bottleneck_dim)
        self.activation = nn.GELU()
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Ağırlıkları başlat."""
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.normal_(self.up_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            Adaptör çıktısı (girdi + adaptör çıktısı),
            şekil (batch_size, seq_len, hidden_size)
        """
        # Darboğaz mimarisi + artık bağlantı
        h = self.down_project(x)
        h = self.layer_norm(h)
        h = self.activation(h)
        h = self.up_project(h)
        
        # Artık bağlantı
        return x + h
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ParallelAdapter(nn.Module):
    """
    Paralel adaptör implementasyonu.
    
    Referans: MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer
    https://arxiv.org/abs/2005.00052
    
    Bu sınıf, parallel adaptör mimarisini uygular, burada adaptör çıktısı
    ve normal çıktı doğrudan birleştirilir.
    """
    
    def __init__(self, hidden_size: int, bottleneck_dim: int, scale: float = 1.0):
        """
        Args:
            hidden_size: Gizli durum boyutu
            bottleneck_dim: Darboğaz boyutu
            scale: Adaptör çıktısını ölçeklendirme faktörü
        """
        super().__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)
        
        self.layer_norm = nn.LayerNorm(bottleneck_dim)
        self.activation = nn.GELU()
        self.scale = scale
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Ağırlıkları başlat."""
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.normal_(self.up_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            Adaptör çıktısı, şekil (batch_size, seq_len, hidden_size)
        """
        # Paralel implementasyon
        h = self.layer_norm(x)
        h = self.down_project(h)
        h = self.activation(h)
        h = self.up_project(h)
        
        # Ölçeklendirilmiş toplama
        return x + self.scale * h
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_adapter_slots(config: AdapterConfig, hidden_size: int, positions: List[str]) -> Dict[str, AdapterSlot]:
    """Belirtilen pozisyonlar için adaptör yuvaları oluşturur."""
    return {pos: AdapterSlot(config, hidden_size) for pos in positions if pos in config.adapter_positions} 