"""
Temel Füzyon Mekanizması Modülü

Bu modül, görüntü ve metin modaliteleri için temel füzyon operasyonlarını içerir.
"""

from typing import Optional, Union, Tuple, Dict, Any

import torch
import torch.nn as nn

from .base_fusion import BaseFusion
from .config import FusionConfig, FusionType


class BasicFusion(BaseFusion):
    """
    Temel füzyon modülü. 
    
    Bu füzyon modülü, metin ve görüntü özelliklerini 
    doğrusal projeksiyonlar ve basit birleştirme ile füzyon yapar.
    """
    
    def __init__(
        self, 
        config_or_text_dim: Union[FusionConfig, int], 
        image_dim: Optional[int] = None, 
        output_dim: Optional[int] = None
    ):
        """
        BasicFusion sınıfını başlatır.
        
        Args:
            config_or_text_dim: Füzyon konfigürasyonu ya da metin gömmelerinin boyutu
            image_dim: Görüntü gömmelerinin boyutu (config kullanılmadığında gerekli)
            output_dim: Çıktı gömmelerinin boyutu (config kullanılmadığında gerekli)
        """
        # BaseFusion'ı başlat
        super().__init__(config_or_text_dim, image_dim, output_dim)
        
        # Konfigürasyondan boyutları al
        self.text_dim = self.config.text_dim
        self.image_dim = self.config.image_dim
        self.output_dim = self.config.output_dim
        
        # Projeksiyon katmanları
        self.text_projection = nn.Linear(self.text_dim, self.output_dim // 2)
        self.image_projection = nn.Linear(self.image_dim, self.output_dim // 2)
        
        # Çıktı aktivasyonu
        self.activation = nn.ReLU()
    
    def fuse(
        self, 
        text_embeddings: torch.Tensor, 
        image_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Metin ve görüntü gömmelerini temel füzyon ile birleştirir.
        BaseFusion sınıfının soyut metodunu uygular.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, text_seq_len, text_dim] veya [batch_size, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, image_seq_len, image_dim] veya [batch_size, image_dim]
            text_mask: Metin maskesi [batch_size, text_seq_len]
            image_mask: Görüntü maskesi [batch_size, image_seq_len]
            
        Returns:
            torch.Tensor: Birleştirilmiş gömmeler [batch_size, output_dim]
        """
        # Global gömmeleri elde et
        if text_embeddings is not None:
            if text_embeddings.dim() > 2:
                # Sequence boyutunda ortalama al
                text_features = torch.mean(text_embeddings, dim=1)
            else:
                text_features = text_embeddings
                
            # Projeksiyon uygula
            text_proj = self.text_projection(text_features)  # [batch_size, output_dim//2]
        else:
            batch_size = image_embeddings.size(0)
            # Metin embeddingi yoksa sıfır vektörü kullan
            text_proj = torch.zeros(batch_size, self.output_dim // 2, device=image_embeddings.device)
            
        # Global gömmeleri elde et
        if image_embeddings is not None:
            if image_embeddings.dim() > 2:
                # Sequence boyutunda ortalama al
                image_features = torch.mean(image_embeddings, dim=1)
            else:
                image_features = image_embeddings
                
            # Projeksiyon uygula
            image_proj = self.image_projection(image_features)  # [batch_size, output_dim//2]
        else:
            batch_size = text_embeddings.size(0)
            # Görüntü embeddingi yoksa sıfır vektörü kullan
            image_proj = torch.zeros(batch_size, self.output_dim // 2, device=text_embeddings.device)
        
        # Gömmeleri birleştir (concatenate)
        fused_features = torch.cat([text_proj, image_proj], dim=1)  # [batch_size, output_dim]
        
        # Aktivasyon uygula
        fused_features = self.activation(fused_features)
        
        return fused_features
    
    def _handle_text_only(self, text_embeddings: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece metin modalitesi için özel işlem.
        BaseFusion'ın varsayılan uygulamasını override eder.
        
        Args:
            text_embeddings: Metin gömmeleri
            text_mask: Metin maskesi
            
        Returns:
            torch.Tensor: İşlenmiş çıktı
        """
        # Global features elde et
        if text_embeddings.dim() > 2:
            # Sequence boyutunda ortalama al
            text_features = torch.mean(text_embeddings, dim=1)
        else:
            text_features = text_embeddings
            
        # Projeksiyon uygula
        text_proj = self.text_projection(text_features)
        
        # Görüntü kısmı için sıfır doldur ve birleştir
        batch_size = text_embeddings.size(0)
        device = text_embeddings.device
        image_proj = torch.zeros(batch_size, self.output_dim // 2, device=device)
        
        # Birleştir
        fused_features = torch.cat([text_proj, image_proj], dim=1)
        
        # Aktivasyon uygula
        fused_features = self.activation(fused_features)
        
        return fused_features
    
    def _handle_image_only(self, image_embeddings: torch.Tensor, image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece görüntü modalitesi için özel işlem.
        BaseFusion'ın varsayılan uygulamasını override eder.
        
        Args:
            image_embeddings: Görüntü gömmeleri
            image_mask: Görüntü maskesi
            
        Returns:
            torch.Tensor: İşlenmiş çıktı
        """
        # Global features elde et
        if image_embeddings.dim() > 2:
            # Sequence boyutunda ortalama al
            image_features = torch.mean(image_embeddings, dim=1)
        else:
            image_features = image_embeddings
            
        # Projeksiyon uygula
        image_proj = self.image_projection(image_features)
        
        # Metin kısmı için sıfır doldur ve birleştir
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device
        text_proj = torch.zeros(batch_size, self.output_dim // 2, device=device)
        
        # Birleştir
        fused_features = torch.cat([text_proj, image_proj], dim=1)
        
        # Aktivasyon uygula
        fused_features = self.activation(fused_features)
        
        return fused_features
    
    def forward(self, 
               text_embeddings: Optional[torch.Tensor] = None, 
               image_embeddings: Optional[torch.Tensor] = None,
               text_mask: Optional[torch.Tensor] = None,
               image_mask: Optional[torch.Tensor] = None,
               return_dict: bool = True,
               return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Füzyon modülünün ileri geçişi.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim] veya [batch_size, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim] veya [batch_size, image_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            return_dict: Çıktı sözlük formatında döndürülsün mü?
            return_attention: Dikkat ağırlıkları döndürülsün mü?
            
        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | Dict[str, Any]: 
                Birleştirilmiş gömmeler, opsiyonel dikkat ağırlıklarıyla birlikte
        """
        # BaseFusion'ın forward metodunu çağır
        output = super().forward(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            text_mask=text_mask,
            image_mask=image_mask,
            return_dict=False  # Önce tensor olarak al
        )
        
        # Sabit dikkat ağırlıkları (temel füzyon için eşit ağırlık)
        if return_attention:
            batch_size = output.size(0)
            device = output.device
            
            if text_embeddings is None:
                # Sadece görüntü varsa
                attention_weights = torch.cat([
                    torch.zeros(batch_size, 1, device=device),
                    torch.ones(batch_size, 1, device=device)
                ], dim=1)
            elif image_embeddings is None:
                # Sadece metin varsa
                attention_weights = torch.cat([
                    torch.ones(batch_size, 1, device=device),
                    torch.zeros(batch_size, 1, device=device)
                ], dim=1)
            else:
                # Her ikisi de varsa
                attention_weights = torch.cat([
                    torch.full((batch_size, 1), 0.5, device=device),
                    torch.full((batch_size, 1), 0.5, device=device)
                ], dim=1)
            
            if return_dict:
                return {
                    "fused_embeddings": output,
                    "attention_weights": attention_weights
                }
            else:
                return output, attention_weights
        
        if return_dict:
            return {
                "fused_embeddings": output
            }
        else:
            return output 