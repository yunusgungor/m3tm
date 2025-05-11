"""
Temel Füzyon Modülü

Bu modül, görüntü ve metin modaliteleri için temel füzyon operasyonlarını içerir.
"""

import torch
import torch.nn as nn

class BasicFusion(nn.Module):
    """
    Temel füzyon modülü sınıfı.
    
    Bu sınıf, metin ve görüntü özelliklerini basit bir yöntemle birleştirir.
    """
    
    def __init__(self, text_dim: int, image_dim: int, output_dim: int):
        """
        Args:
            text_dim: Metin özelliklerinin boyutu
            image_dim: Görüntü özelliklerinin boyutu
            output_dim: Çıktı özelliklerinin boyutu
        """
        super(BasicFusion, self).__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim
        
        # Metin projeksiyon
        self.text_proj = nn.Linear(text_dim, output_dim // 2)
        
        # Görüntü projeksiyon
        self.image_proj = nn.Linear(image_dim, output_dim // 2)
        
        # Birleştirilmiş özellik projeksiyon
        self.combined_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        İleri geçiş.
        
        Args:
            text_features: Metin özellikleri [batch_size, text_dim]
            image_features: Görüntü özellikleri [batch_size, image_dim]
            
        Returns:
            torch.Tensor: Füzyon özellikleri [batch_size, output_dim]
        """
        # Metni ve görüntüyü ayrı ayrı project et
        text_proj = self.text_proj(text_features)    # [batch_size, output_dim//2]
        image_proj = self.image_proj(image_features) # [batch_size, output_dim//2]
        
        # Projeksiyonları birleştir
        combined = torch.cat([text_proj, image_proj], dim=1)  # [batch_size, output_dim]
        
        # Birleştirilmiş özelliklerin son projeksiyonu
        fused_features = self.combined_proj(combined)  # [batch_size, output_dim]
        
        return fused_features 