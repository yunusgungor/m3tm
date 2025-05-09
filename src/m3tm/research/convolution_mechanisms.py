"""
Mobil Konvolüsyon Mekanizmaları Araştırması

Bu modül, mobil cihazlarda verimli çalışabilecek alternatif 
konvolüsyon mekanizmalarının araştırılması ve prototiplenmesi için kullanılır.

Örüntü: ConfigurationDataclass (PT-001)
"""

from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConvolutionMechanismConfig:
    """Konvolüsyon mekanizması yapılandırması için temel sınıf.
    
    Tüm konvolüsyon mekanizmaları için ortak parametreleri içerir.
    Özel konvolüsyon mekanizmaları bu sınıftan türetilmelidir.
    """
    name: str = ""  # Mekanizma adı
    in_channels: int = 32  # Girdi kanal sayısı
    out_channels: int = 32  # Çıktı kanal sayısı
    kernel_size: int = 3  # Çekirdek boyutu
    stride: int = 1  # Adım boyutu
    padding: int = 1  # Dolgu boyutu
    groups: int = 1  # Grup sayısı
    bias: bool = True  # Bias kullanılıp kullanılmayacağı
    
    def __post_init__(self):
        """Yapılandırma parametrelerinin tutarlılığını kontrol eder."""
        if not self.name:
            raise ValueError("Convolution mechanism name must be specified")
        
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        
        if self.in_channels % self.groups != 0:
            raise ValueError(f"in_channels {self.in_channels} must be divisible by groups {self.groups}")
            
        if self.out_channels % self.groups != 0:
            raise ValueError(f"out_channels {self.out_channels} must be divisible by groups {self.groups}")


@dataclass
class StandardConvConfig(ConvolutionMechanismConfig):
    """Standart konvolüsyon yapılandırması."""
    name: str = "StandardConv"


@dataclass
class DepthwiseSeparableConvConfig(ConvolutionMechanismConfig):
    """Derinlik yönlü ayrılabilir konvolüsyon yapılandırması."""
    name: str = "DepthwiseSeparableConv"
    # Derinlik yönlü = giriş kanalı sayısı kadar grup
    # (groups özelliği __post_init__ içinde ayarlanacak)
    
    def __post_init__(self):
        super().__post_init__()
        # Derinlik yönlü konvolüsyon için grupları girdi kanalına eşitle
        # Not: Bu, yalnızca ilk katman içindir; nokta-yönlü konvolüsyon için
        # groups=1 olarak kalır (bu sınıf her iki aşamayı da içeriyor)


@dataclass
class MobileConvBlockConfig(ConvolutionMechanismConfig):
    """MobileNet konvolüsyon bloğu yapılandırması."""
    name: str = "MobileConvBlock"
    expansion_factor: float = 6.0  # Genişleme faktörü
    use_residual: bool = True  # Artık bağlantı kullanımı
    activation: str = "relu"  # Aktivasyon fonksiyonu


@dataclass
class ShuffleConvConfig(ConvolutionMechanismConfig):
    """ShuffleNet konvolüsyon yapılandırması."""
    name: str = "ShuffleConv"
    shuffle_groups: int = 2  # Karıştırma grupları
    use_residual: bool = True  # Artık bağlantı kullanımı


@dataclass
class GhostConvConfig(ConvolutionMechanismConfig):
    """Ghost konvolüsyon yapılandırması."""
    name: str = "GhostConv"
    ratio: int = 2  # Hayalet özellik oranı
    dw_kernel_size: int = 3  # Derinlik yönlü çekirdek boyutu
    use_act: bool = True  # Aktivasyon kullanımı


class ConvolutionMechanismRegistry:
    """Konvolüsyon mekanizmalarını kaydeden ve erişim sağlayan kayıt sınıfı."""
    
    _registry: Dict[str, Dict] = {
        "models": {},
        "configs": {}
    }
    
    @classmethod
    def register(cls, conv_class: nn.Module, config_class):
        """Yeni bir konvolüsyon mekanizmasını kaydeder."""
        name = config_class().name
        cls._registry["models"][name] = conv_class
        cls._registry["configs"][name] = config_class
        return conv_class
    
    @classmethod
    def get_convolution(cls, name: str, **kwargs):
        """İsme göre konvolüsyon mekanizması sınıfını ve yapılandırmasını döndürür."""
        if name not in cls._registry["models"]:
            raise ValueError(f"Convolution mechanism '{name}' not registered")
        
        config_class = cls._registry["configs"][name]
        config = config_class(**kwargs)
        conv_class = cls._registry["models"][name]
        
        return conv_class, config
    
    @classmethod
    def list_mechanisms(cls):
        """Kayıtlı tüm konvolüsyon mekanizmalarının listesini döndürür."""
        return list(cls._registry["models"].keys())


@ConvolutionMechanismRegistry.register
class StandardConv(nn.Module):
    """Standart konvolüsyon implementasyonu.
    
    Bu temel implementasyon, karşılaştırma için referans olarak kullanılacaktır.
    """
    
    def __init__(self, config: StandardConvConfig):
        """
        Args:
            config: Konvolüsyon mekanizması yapılandırması
        """
        super().__init__()
        self.config = config
        
        # Standart 2B konvolüsyon katmanı
        self.conv = nn.Conv2d(
            config.in_channels,
            config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            groups=config.groups,
            bias=config.bias
        )
        
        # İşlem sonrası normalizasyon
        self.batch_norm = nn.BatchNorm2d(config.out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, in_channels, height, width)
            
        Returns:
            Konvolüsyon çıktısı, şekil (batch_size, out_channels, out_height, out_width)
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Performans metrikleri
        metrics = {
            "memory_complexity": self.config.kernel_size ** 2 * self.config.in_channels * self.config.out_channels // self.config.groups,
            "compute_complexity": self.config.kernel_size ** 2 * self.config.in_channels * self.config.out_channels // self.config.groups,
            "parameter_count": self.count_parameters()
        }
        
        return x, metrics
    
    def count_parameters(self):
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@ConvolutionMechanismRegistry.register
class DepthwiseSeparableConv(nn.Module):
    """Derinlik yönlü ayrılabilir konvolüsyon implementasyonu.
    
    Referans: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
    
    Bu mekanizma, standart konvolüsyonu iki aşamaya ayırır:
    1. Derinlik yönlü konvolüsyon: Her girdi kanalını ayrı ayrı işler
    2. Nokta-yönlü (1x1) konvolüsyon: Kanallar arası bilgiyi karıştırır
    """
    
    def __init__(self, config: DepthwiseSeparableConvConfig):
        """
        Args:
            config: Derinlik yönlü ayrılabilir konvolüsyon yapılandırması
        """
        super().__init__()
        self.config = config
        
        # 1. Aşama: Derinlik yönlü konvolüsyon (her kanala ayrı konvolüsyon uygular)
        self.depthwise = nn.Conv2d(
            config.in_channels,
            config.in_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            groups=config.in_channels,  # Her kanal için ayrı filtre
            bias=False  # Genellikle derinlik katmanında bias kullanılmaz
        )
        self.bn1 = nn.BatchNorm2d(config.in_channels)
        
        # 2. Aşama: Nokta-yönlü konvolüsyon (1x1 konvolüsyon ile kanal karıştırma)
        self.pointwise = nn.Conv2d(
            config.in_channels,
            config.out_channels,
            kernel_size=1,  # 1x1 çekirdek
            stride=1,
            padding=0,
            bias=config.bias
        )
        self.bn2 = nn.BatchNorm2d(config.out_channels)
        
        # Aktivasyon
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, in_channels, height, width)
            
        Returns:
            Konvolüsyon çıktısı, şekil (batch_size, out_channels, out_height, out_width)
        """
        # Derinlik yönlü aşama
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Nokta-yönlü aşama
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        # Performans metrikleri
        # DW + PW karmaşıklığı: (k² * C_in) + (C_in * C_out)
        metrics = {
            "memory_complexity": (self.config.kernel_size ** 2 * self.config.in_channels) + (self.config.in_channels * self.config.out_channels),
            "compute_complexity": (self.config.kernel_size ** 2 * self.config.in_channels) + (self.config.in_channels * self.config.out_channels),
            "parameter_count": self.count_parameters()
        }
        
        return x, metrics
    
    def count_parameters(self):
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@ConvolutionMechanismRegistry.register
class MobileConvBlock(nn.Module):
    """MobileNetV2 tipi ters darboğaz (inverted bottleneck) konvolüsyon bloğu.
    
    Referans: MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
    
    Bu blok şu aşamalardan oluşur:
    1. Genişleme: 1x1 konvolüsyon ile kanal sayısını artırma (expansion)
    2. Derinlik yönlü: 3x3 derinlik yönlü konvolüsyon
    3. Projeksiyon: 1x1 konvolüsyon ile kanal sayısını azaltma (projection)
    """
    
    def __init__(self, config: MobileConvBlockConfig):
        """
        Args:
            config: MobileNet konvolüsyon bloğu yapılandırması
        """
        super().__init__()
        self.config = config
        
        hidden_dim = int(config.in_channels * config.expansion_factor)
        self.use_residual = config.use_residual and config.in_channels == config.out_channels and config.stride == 1
        
        layers = []
        
        # 1. Genişleme katmanı (yoksa atlayabilir)
        if hidden_dim != config.in_channels:
            layers.extend([
                # 1x1 konvolüsyon ile genişleme
                nn.Conv2d(config.in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                self._get_activation(config.activation)
            ])
        
        # 2. Derinlik yönlü konvolüsyon
        layers.extend([
            # 3x3 derinlik yönlü konvolüsyon
            nn.Conv2d(
                hidden_dim, 
                hidden_dim, 
                kernel_size=config.kernel_size, 
                stride=config.stride, 
                padding=config.padding,
                groups=hidden_dim,  # Derinlik yönlü
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            self._get_activation(config.activation)
        ])
        
        # 3. Projeksiyon katmanı
        layers.extend([
            # 1x1 konvolüsyon ile projeksiyon (DoğrusalBN)
            nn.Conv2d(hidden_dim, config.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def _get_activation(self, activation_name: str):
        """Aktivasyon fonksiyonunu ismine göre döndürür."""
        if activation_name.lower() == "relu":
            return nn.ReLU(inplace=True)
        elif activation_name.lower() == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation_name.lower() == "swish":
            return nn.SiLU(inplace=True)  # PyTorch'un Swish/SiLU uygulaması
        elif activation_name.lower() == "hardswish":
            return nn.Hardswish(inplace=True)
        else:
            return nn.ReLU(inplace=True)  # Varsayılan
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, in_channels, height, width)
            
        Returns:
            Konvolüsyon çıktısı, şekil (batch_size, out_channels, out_height, out_width)
        """
        # Konvolüsyon bloğu
        out = self.conv(x)
        
        # Artık bağlantı
        if self.use_residual:
            out = out + x
        
        hidden_dim = int(self.config.in_channels * self.config.expansion_factor)
        
        # Performans metrikleri
        # 1x1 expansion + k² DW + 1x1 projection
        metrics = {
            "memory_complexity": (self.config.in_channels * hidden_dim) + 
                               (self.config.kernel_size ** 2 * hidden_dim) + 
                               (hidden_dim * self.config.out_channels),
            "compute_complexity": (self.config.in_channels * hidden_dim) + 
                                (self.config.kernel_size ** 2 * hidden_dim) + 
                                (hidden_dim * self.config.out_channels),
            "parameter_count": self.count_parameters()
        }
        
        return out, metrics
    
    def count_parameters(self):
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_convolution_mechanism_by_name(name: str, **kwargs):
    """İsme göre konvolüsyon mekanizması sınıfını ve uygun yapılandırmasını döndürür.
    
    Args:
        name: Konvolüsyon mekanizması adı
        **kwargs: Konvolüsyon mekanizması yapılandırması için ek parametreler
        
    Returns:
        Konvolüsyon mekanizması sınıfı ve yapılandırması
    """
    return ConvolutionMechanismRegistry.get_convolution(name, **kwargs)


def get_available_mechanisms():
    """Kullanılabilir tüm konvolüsyon mekanizmalarının listesini döndürür."""
    return ConvolutionMechanismRegistry.list_mechanisms() 