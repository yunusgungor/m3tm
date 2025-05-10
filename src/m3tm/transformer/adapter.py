"""
M³TM adaptör mekanizmaları.

Bu modül, ProtoTransformerBlock için temel adaptör mekanizmalarını içerir.
Adaptörler, çekirdek modele küçük, eğitilebilir parametre kümeleri ekleyerek
modeli özelleştirmeye olanak tanır.

Örüntüler:
- ConfigurationDataclass (PT-001): Yapılandırma parametrelerini dataclass olarak modelleme
- ModelComposite (PT-003): Ana modelin içine küçük ve özelleştirilmiş modül ekleme
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
    
    Örüntüler:
    - ModelComposite (PT-003): Ana modelin içine küçük ve özelleştirilmiş modül ekleme
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
            self.adapter = self._create_adapter()
    
    def _create_adapter(self):
        """Adaptör yapılandırmasına göre uygun adaptör oluşturur."""
        if self.config.adapter_type == "bottleneck":
            return Adapter(self.hidden_size, self.bottleneck_dim, 
                           use_layer_norm=self.config.use_layer_norm,
                           activation=self.config.activation,
                           dropout=self.config.dropout,
                           init_scale=self.config.init_scale)
        elif self.config.adapter_type == "parallel":
            return ParallelAdapter(self.hidden_size, self.bottleneck_dim,
                                   scale=self.config.alpha,
                                   use_layer_norm=self.config.use_layer_norm,
                                   activation=self.config.activation,
                                   dropout=self.config.dropout,
                                   init_scale=self.config.init_scale)
        else:
            raise ValueError(f"Geçersiz adapter türü: {self.config.adapter_type}")
    
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
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)
        
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
        nn.init.normal_(self.down_project.weight, std=init_scale)
        nn.init.normal_(self.up_project.weight, std=init_scale)
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
        residual = x
        
        # Darboğaz mimarisi
        h = self.down_project(x)
        
        # Layer normalization (varsa)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        
        # Aktivasyon
        h = self.activation(h)
        
        # Dropout (varsa)
        if self.dropout is not None:
            h = self.dropout(h)
        
        # Up-projeksiyon
        h = self.up_project(h)
        
        # Artık bağlantı
        return residual + h
    
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
            scale: Adaptör çıktısını ölçeklendirme faktörü
            use_layer_norm: Layer normalization kullanımı
            activation: Aktivasyon fonksiyonu adı
            dropout: Dropout oranı
            init_scale: Başlangıç ölçeği
        """
        super().__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)
        
        # Layer normalization her zaman kullanılır
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Aktivasyon fonksiyonu
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Ölçekleme faktörü (alpha)
        self.scale = scale
        
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
        nn.init.normal_(self.down_project.weight, std=init_scale)
        nn.init.normal_(self.up_project.weight, std=init_scale)
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
        
        # Dropout (varsa)
        if self.dropout is not None:
            h = self.dropout(h)
            
        h = self.up_project(h)
        
        # Ölçeklendirilmiş toplama
        return x + self.scale * h
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_adapter_slots(config: AdapterConfig, hidden_size: int, positions: List[str]) -> Dict[str, AdapterSlot]:
    """
    Belirtilen pozisyonlar için adaptör yuvaları oluşturur.
    
    Args:
        config: Adaptör yapılandırması
        hidden_size: Gizli durum boyutu 
        positions: Desteklenen adapter pozisyonları
        
    Returns:
        Pozisyon -> AdapterSlot eşleştirmesi içeren sözlük
    """
    # Sadece adapter_positions'ta belirtilen pozisyonlar için slot oluştur
    return {pos: AdapterSlot(config, hidden_size) for pos in positions if pos in config.adapter_positions} 