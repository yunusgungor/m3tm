"""
M³TM için Feed-Forward Network mekanizmaları.

Bu modül, ProtoTransformerBlock için alternatif Feed-Forward Network 
implementasyonlarını içerir. 

Örüntüler:
- ConfigurationDataclass (PT-001): Yapılandırma parametrelerini dataclass olarak modelleme
- MechanismRegistry (PT-007): FFN mekanizmalarını kaydetme ve isimle erişme
- PluggableComponentStrategy: Farklı FFN mekanizmalarını değiştirilebilir kılmak için
"""

from typing import Dict, Tuple, Optional, Any, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from m3tm.transformer.config import FeedForwardConfig


class FFNMechanismRegistry:
    """FFN mekanizmalarını kaydeden ve isimle erişim sağlayan kayıt sınıfı."""
    
    _registry: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """Yeni bir FFN mekanizmasını kaydeder."""
        def decorator(ffn_class: Type[nn.Module]):
            mechanism_name = name if name else ffn_class.__name__
            cls._registry[mechanism_name] = ffn_class
            return ffn_class
        return decorator
    
    @classmethod
    def get_ffn(cls, config: FeedForwardConfig) -> nn.Module:
        """İsme göre FFN mekanizmasını oluşturup döndürür."""
        if config.mechanism_name not in cls._registry:
            raise ValueError(f"FFN mechanism '{config.mechanism_name}' not registered")
        
        ffn_class = cls._registry[config.mechanism_name]
        return ffn_class(config)
    
    @classmethod
    def list_mechanisms(cls) -> list:
        """Kayıtlı tüm FFN mekanizmalarının listesini döndürür."""
        return list(cls._registry.keys())


@FFNMechanismRegistry.register(name="StandardFFN")
class StandardFFN(nn.Module):
    """Standart iki katmanlı Feed-Forward Network.
    
    Bu temel implementasyon, karşılaştırma için referans olarak kullanılacaktır.
    """
    
    def __init__(self, config: FeedForwardConfig):
        """
        Args:
            config: FFN yapılandırması
        """
        super().__init__()
        self.config = config
        hidden_size = config.mechanism_params.get("hidden_size", 256)
        intermediate_size = int(hidden_size * config.expansion_factor)
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Ağırlıkları başlat."""
        nn.init.normal_(self.intermediate.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.intermediate.bias)
        nn.init.zeros_(self.output.bias)
    
    def get_activation(self, activation: str):
        """Aktivasyon fonksiyonunu isimle al."""
        if activation == "gelu":
            return F.gelu
        elif activation == "relu":
            return F.relu
        elif activation == "silu" or activation == "swish":
            return F.silu
        else:
            raise ValueError(f"Activation {activation} not supported")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            FFN çıktısı, şekil (batch_size, seq_len, hidden_size)
        """
        h = self.intermediate(x)
        h = self.get_activation(self.config.hidden_act)(h)
        h = self.output(h)
        h = self.dropout(h)
        
        metrics = {
            "parameter_count": self.count_parameters(),
            "flops": self._calculate_flops(x.shape),
        }
        
        return h, metrics
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _calculate_flops(self, input_shape: Tuple) -> int:
        """Yaklaşık FLOP sayısını hesaplar."""
        batch_size, seq_len, hidden_size = input_shape
        intermediate_size = int(hidden_size * self.config.expansion_factor)
        
        # Linear(hidden -> intermediate) + activation + Linear(intermediate -> hidden)
        flops = batch_size * seq_len * (
            2 * hidden_size * intermediate_size + intermediate_size + 
            2 * intermediate_size * hidden_size + hidden_size
        )
        
        return flops


@FFNMechanismRegistry.register(name="MobileFFN")
class MobileFFN(nn.Module):
    """Mobil cihazlar için optimize edilmiş Feed-Forward Network.
    
    Standart FFN'den farklı olarak:
    1. Daha küçük genişleme faktörü (tipik olarak 2.0 vs. orijinal transformer'ın 4.0)
    2. Daha hafif aktivasyon (GELU yerine SiLU/Swish)
    3. Opsiyonel olarak nokta-nokta konvolüsyon kullanımı (1x1 konv)
    """
    
    def __init__(self, config: FeedForwardConfig):
        """
        Args:
            config: FFN yapılandırması
        """
        super().__init__()
        self.config = config
        hidden_size = config.mechanism_params.get("hidden_size", 256)
        intermediate_size = int(hidden_size * config.expansion_factor)
        use_conv = config.mechanism_params.get("use_conv", False)
        
        # Konvolüsyon veya lineer katman kullanımı
        if use_conv:
            self.intermediate = nn.Conv1d(
                hidden_size, intermediate_size, kernel_size=1, stride=1, padding=0, bias=True
            )
            self.output = nn.Conv1d(
                intermediate_size, hidden_size, kernel_size=1, stride=1, padding=0, bias=True
            )
        else:
            self.intermediate = nn.Linear(hidden_size, intermediate_size)
            self.output = nn.Linear(intermediate_size, hidden_size)
            
        self.dropout = nn.Dropout(config.dropout)
        self.use_conv = use_conv
        
        self._init_weights()
    
    def _init_weights(self):
        """Ağırlıkları başlat."""
        if self.use_conv:
            nn.init.kaiming_normal_(self.intermediate.weight, nonlinearity="linear")
            nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
            nn.init.zeros_(self.intermediate.bias)
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.normal_(self.intermediate.weight, std=0.02)
            nn.init.normal_(self.output.weight, std=0.02)
            nn.init.zeros_(self.intermediate.bias)
            nn.init.zeros_(self.output.bias)
    
    def get_activation(self, activation: str):
        """Aktivasyon fonksiyonunu isimle al."""
        # Mobil için SiLU/Swish tercih edilir
        if activation == "gelu":
            return F.gelu
        elif activation == "relu":
            return F.relu
        elif activation == "silu" or activation == "swish":
            return F.silu
        elif activation == "hardswish":
            return F.hardswish
        else:
            raise ValueError(f"Activation {activation} not supported")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            FFN çıktısı, şekil (batch_size, seq_len, hidden_size)
        """
        if self.use_conv:
            # Conv1d için boyut düzenlemesi (B, L, H) -> (B, H, L)
            x_conv = x.transpose(1, 2)
            h = self.intermediate(x_conv)
            h = self.get_activation(self.config.hidden_act)(h)
            h = self.output(h)
            # Sonucu tekrar düzenle (B, H, L) -> (B, L, H)
            h = h.transpose(1, 2)
        else:
            h = self.intermediate(x)
            h = self.get_activation(self.config.hidden_act)(h)
            h = self.output(h)
            
        h = self.dropout(h)
        
        metrics = {
            "parameter_count": self.count_parameters(),
            "flops": self._calculate_flops(x.shape),
        }
        
        return h, metrics
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _calculate_flops(self, input_shape: Tuple) -> int:
        """Yaklaşık FLOP sayısını hesaplar."""
        batch_size, seq_len, hidden_size = input_shape
        intermediate_size = int(hidden_size * self.config.expansion_factor)
        
        # Konv veya lineer hesaplama
        if self.use_conv:
            # 1x1 conv için FLOPs hesaplama (her pikselde bir nokta çarpımı)
            flops = batch_size * seq_len * (
                2 * hidden_size * intermediate_size + intermediate_size + 
                2 * intermediate_size * hidden_size + hidden_size
            )
        else:
            # Linear(hidden -> intermediate) + activation + Linear(intermediate -> hidden)
            flops = batch_size * seq_len * (
                2 * hidden_size * intermediate_size + intermediate_size + 
                2 * intermediate_size * hidden_size + hidden_size
            )
        
        return flops


@FFNMechanismRegistry.register(name="GhostFFN")
class GhostFFN(nn.Module):
    """Ghost module tabanlı FFN.
    
    Referans: GhostNet: More Features from Cheap Operations
    https://arxiv.org/abs/1911.11907
    
    Bu mekanizma, standart FFN'i şu şekilde değiştirir:
    1. Birincil özellikler ve daha ucuz hesaplanmış "hayalet" özellikler kullanır
    2. Toplam parametre sayısını önemli ölçüde azaltır
    """
    
    def __init__(self, config: FeedForwardConfig):
        """
        Args:
            config: FFN yapılandırması
        """
        super().__init__()
        self.config = config
        hidden_size = config.mechanism_params.get("hidden_size", 256)
        ratio = config.mechanism_params.get("ghost_ratio", 2)
        intermediate_size = int(hidden_size * config.expansion_factor)
        
        # Ghost module birincil özellikler için daha az filtre kullanır
        self.primary_channels = intermediate_size // ratio
        
        # İlk ghost modülü
        self.primary_fc = nn.Linear(hidden_size, self.primary_channels)
        self.cheap_operation = nn.Sequential(
            nn.Linear(self.primary_channels, self.primary_channels),
            nn.BatchNorm1d(self.primary_channels),
            nn.ReLU(inplace=True)
        )
        
        # İkinci ghost modülü
        self.primary_out = nn.Linear(intermediate_size, hidden_size // ratio)
        self.cheap_out = nn.Sequential(
            nn.Linear(hidden_size // ratio, hidden_size // ratio),
            nn.BatchNorm1d(hidden_size // ratio),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Ağırlıkları başlat."""
        nn.init.normal_(self.primary_fc.weight, std=0.02)
        nn.init.normal_(self.primary_out.weight, std=0.02)
        nn.init.zeros_(self.primary_fc.bias)
        nn.init.zeros_(self.primary_out.bias)
    
    def get_activation(self, activation: str):
        """Aktivasyon fonksiyonunu isimle al."""
        if activation == "gelu":
            return F.gelu
        elif activation == "relu":
            return F.relu
        elif activation == "silu" or activation == "swish":
            return F.silu
        else:
            raise ValueError(f"Activation {activation} not supported")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            
        Returns:
            FFN çıktısı, şekil (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # İlk ghost modülü
        primary_features = self.primary_fc(x)  # (B, L, primary_channels)
        
        # Birincil özellikleri yeniden şekillendir - BatchNorm1d için
        primary_features_reshaped = primary_features.reshape(-1, self.primary_channels)
        
        # Ucuz operasyon
        cheap_features = self.cheap_operation(primary_features_reshaped)
        cheap_features = cheap_features.reshape(batch_size, seq_len, self.primary_channels)
        
        # Birincil ve hayalet özellikleri birleştir
        h_intermediate = torch.cat([primary_features, cheap_features], dim=2)  # (B, L, intermediate_size)
        h_intermediate = self.get_activation(self.config.hidden_act)(h_intermediate)
        
        # İkinci ghost modülü
        primary_out = self.primary_out(h_intermediate)  # (B, L, hidden_size//ratio)
        
        # Yeniden şekillendir - BatchNorm1d için
        primary_out_reshaped = primary_out.reshape(-1, hidden_size // ratio)
        
        # Ucuz operasyon
        cheap_out = self.cheap_out(primary_out_reshaped)
        cheap_out = cheap_out.reshape(batch_size, seq_len, hidden_size // ratio)
        
        # Birleştir ve çıktı ver
        h = torch.cat([primary_out, cheap_out], dim=2)  # (B, L, hidden_size)
        h = self.dropout(h)
        
        metrics = {
            "parameter_count": self.count_parameters(),
            "flops": self._calculate_flops(x.shape),
        }
        
        return h, metrics
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _calculate_flops(self, input_shape: Tuple) -> int:
        """Yaklaşık FLOP sayısını hesaplar."""
        batch_size, seq_len, hidden_size = input_shape
        intermediate_size = int(hidden_size * self.config.expansion_factor)
        
        # Birincil özellikler + ucuz operasyonlar + çıktı katmanı
        flops = batch_size * seq_len * (
            2 * hidden_size * self.primary_channels + self.primary_channels +  # İlk birincil
            2 * self.primary_channels * self.primary_channels + 2 * self.primary_channels +  # Ucuz op
            2 * intermediate_size * (hidden_size // 2) + (hidden_size // 2) +  # İkinci birincil
            2 * (hidden_size // 2) * (hidden_size // 2) + (hidden_size // 2)  # İkinci ucuz op
        )
        
        return flops


def get_ffn_mechanism(config: FeedForwardConfig) -> nn.Module:
    """FFN mekanizmasını yapılandırmaya göre oluşturur."""
    return FFNMechanismRegistry.get_ffn(config)


def list_available_ffn_mechanisms() -> list:
    """Kullanılabilir tüm FFN mekanizmalarını listeler."""
    return FFNMechanismRegistry.list_mechanisms() 