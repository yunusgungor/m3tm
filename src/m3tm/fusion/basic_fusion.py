"""
Temel Füzyon Modülü

Bu modül, görüntü ve metin modaliteleri için temel füzyon operasyonlarını içerir.
"""

import torch
import torch.nn as nn

from m3tm.config.model_config import FusionConfig

class BasicFusion(nn.Module):
    """
    Temel füzyon modülü sınıfı.
    
    Bu sınıf, metin ve görüntü özelliklerini basit bir yöntemle birleştirir.
    """
    
    def __init__(self, config: FusionConfig):
        """
        Args:
            config: Füzyon yapılandırması
        """
        super(BasicFusion, self).__init__()
        
        # Konfigürasyondan boyut bilgilerini al
        self.text_dim = config.text_dim
        self.image_dim = config.image_dim
        self.output_dim = config.output_dim
        self.use_layer_norm = config.use_layer_norm
        
        # Metin projeksiyon
        self.text_proj = nn.Linear(self.text_dim, self.output_dim // 2)
        
        # Görüntü projeksiyon
        self.image_proj = nn.Linear(self.image_dim, self.output_dim // 2)
        
        # Birleştirilmiş özellik projeksiyonu
        layers = [nn.Linear(self.output_dim, self.output_dim)]
        
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(self.output_dim))
        
        layers.append(nn.GELU())
        
        if hasattr(config, 'dropout') and config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
            
        self.combined_proj = nn.Sequential(*layers)
    
    def forward(self, text_features: torch.Tensor = None, image_features: torch.Tensor = None) -> torch.Tensor:
        """
        İleri geçiş.
        
        Args:
            text_features: Metin özellikleri [batch_size, text_dim] (opsiyonel)
            image_features: Görüntü özellikleri [batch_size, image_dim] (opsiyonel)
            
        Returns:
            torch.Tensor: Füzyon özellikleri [batch_size, output_dim]
        """
        # Her iki modalite de yoksa hata fırlat
        if text_features is None and image_features is None:
            raise ValueError("En az bir modalite (metin veya görüntü) sağlanmalıdır.")
            
        # Yalnızca metin varsa
        if text_features is not None and image_features is None:
            text_proj = self.text_proj(text_features)
            # Görüntü yerine metin özelliklerini doldur
            image_proj = torch.zeros_like(text_proj)
            
        # Yalnızca görüntü varsa
        elif image_features is not None and text_features is None:
            image_proj = self.image_proj(image_features)
            # Metin yerine görüntü özelliklerini doldur
            text_proj = torch.zeros_like(image_proj)
            
        # Her iki modalite de varsa
        else:
            # Metni ve görüntüyü ayrı ayrı project et
            text_proj = self.text_proj(text_features)      # [batch_size, output_dim//2]
            image_proj = self.image_proj(image_features)   # [batch_size, output_dim//2]
        
        # Projeksiyonları birleştir
        combined = torch.cat([text_proj, image_proj], dim=1)  # [batch_size, output_dim]
        
        # Birleştirilmiş özelliklerin son projeksiyonu
        fused_features = self.combined_proj(combined)  # [batch_size, output_dim]
        
        return fused_features 