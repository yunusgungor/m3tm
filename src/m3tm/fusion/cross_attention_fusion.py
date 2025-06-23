"""
Çapraz Dikkat Füzyon Modülü

Bu modül, metin ve görüntü modalitelerini çapraz dikkat mekanizması ile 
birleştirmek için kullanılır. Bu yaklaşım, modaliteler arasında dikkat 
ağırlıkları hesaplayarak etkileşimli bir füzyon sağlar.

Örüntü: MultimodalAttention (PT-027)
"""

from typing import Dict, Optional, Tuple, Union, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_fusion import BaseFusion
from .config import FusionConfig, FusionType


class CrossAttentionFusion(BaseFusion):
    """
    Metin ve görüntü gömmelerini çapraz dikkat mekanizması ile birleştiren modül.
    Bu yaklaşım, her modalite için dikkat skoru hesaplar ve bu skorları 
    ağırlıklandırma olarak kullanır.
    
    Örüntü: MultimodalAttention (PT-027)
    """
    
    def __init__(
        self, 
        config: Optional[FusionConfig] = None,
        text_dim: int = 768, 
        image_dim: int = 1024, 
        output_dim: int = 768, 
        num_heads: int = 4,
        dropout: float = 0.1,
        head_dim: Optional[int] = None
    ):
        """
        CrossAttentionFusion sınıfını başlatır.
        
        Args:
            config: Füzyon yapılandırması
            text_dim: Metin özelliklerinin boyutu
            image_dim: Görüntü özelliklerinin boyutu
            output_dim: Çıktı özelliklerinin boyutu
            num_heads: Dikkat mekanizmasındaki kafa sayısı
            dropout: Dropout oranı
            head_dim: Her kafa için dikkat boyutu (belirtilmezse otomatik hesaplanır)
        """
        super().__init__(config or FusionConfig(fusion_type=FusionType.CROSS_ATTENTION))
        
        # Config'den ya da argümanlardan boyutları belirle
        if config is not None:
            self.text_dim = config.text_dim
            self.image_dim = config.image_dim
            self.output_dim = config.output_dim
            self.num_heads = num_heads  # Argümandan alıyoruz, çünkü config'de olmayabilir
        else:
            self.text_dim = text_dim
            self.image_dim = image_dim
            self.output_dim = output_dim
            self.num_heads = num_heads
        
        # Text-to-Image çapraz dikkat
        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Image-to-Text çapraz dikkat
        self.image_to_text_attention = nn.MultiheadAttention(
            embed_dim=self.image_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projeksiyon katmanları
        self.text_proj = nn.Linear(self.text_dim, self.output_dim)
        self.image_proj = nn.Linear(self.image_dim, self.output_dim)
        
        # Son projeksiyon
        self.output_proj = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
    
    def get_parameter_count(self) -> int:
        """Toplam parametre sayısını döndürür."""
        return sum(p.numel() for p in self.parameters())

    def fuse(
        self, 
        text_embeddings: torch.Tensor, 
        image_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Metin ve görüntü gömmelerini çapraz dikkat mekanizması ile birleştirir.
        BaseFusion sınıfının soyut metodunu uygular.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, text_seq_len, text_dim] veya [batch_size, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, image_seq_len, image_dim] veya [batch_size, image_dim]
            text_mask: Metin maskesi [batch_size, text_seq_len]
            image_mask: Görüntü maskesi [batch_size, image_seq_len]
            
        Returns:
            torch.Tensor: Birleştirilmiş gömmeler [batch_size, output_dim]
        """
        return self.forward(text_embeddings, image_embeddings, text_mask, image_mask)

    def forward(
        self, 
        text_features: torch.Tensor, 
        image_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        İki modaliteyi çapraz dikkat mekanizması ile birleştirir.
        
        Args:
            text_features: Metin özellikleri [batch_size, text_seq_len, text_dim] veya [batch_size, text_dim]
            image_features: Görüntü özellikleri [batch_size, image_seq_len, image_dim] veya [batch_size, image_dim]
            text_mask: Metin özellikleri için maske [batch_size, text_seq_len]
            image_mask: Görüntü özellikleri için maske [batch_size, image_seq_len]
            return_attention: Dikkat ağırlıklarının döndürülüp döndürülmeyeceği
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                Birleştirilmiş özellikler ve opsiyonel olarak dikkat ağırlıkları
        """
        # Tek modalite durumunu ele al
        if image_features is None:
            return self._handle_text_only(text_features, text_mask)
        elif text_features is None:
            return self._handle_image_only(image_features, image_mask)
        
        batch_size = text_features.size(0)
        
        # 2D girdi (global özellikler) için 3D'ye genişlet
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
            
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        
        # Maskeleri key padding mask formatına dönüştür
        text_key_padding_mask = None
        image_key_padding_mask = None
        
        if text_mask is not None:
            # True değerleri padding'i gösterir (dikkat edilmeyecek kısım)
            text_key_padding_mask = ~text_mask.bool()
            
        if image_mask is not None:
            # True değerleri padding'i gösterir (dikkat edilmeyecek kısım)
            image_key_padding_mask = ~image_mask.bool()
        
        # Text-to-Image çapraz dikkat
        text_attn_output, text_attn_weights = self.text_to_image_attention(
            query=text_features,
            key=image_features,
            value=image_features,
            key_padding_mask=image_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Image-to-Text çapraz dikkat
        image_attn_output, image_attn_weights = self.image_to_text_attention(
            query=image_features,
            key=text_features,
            value=text_features,
            key_padding_mask=text_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Çıkışları projeksiyon
        text_output = self.text_proj(text_attn_output)
        image_output = self.image_proj(image_attn_output)
        
        # Global özellikler (CLS token'ları veya ortalamaları)
        global_text = text_output[:, 0] if text_output.dim() > 2 else text_output.mean(dim=1)
        global_image = image_output[:, 0] if image_output.dim() > 2 else image_output.mean(dim=1)
        
        # Özellikleri birleştir
        combined_features = torch.cat([global_text, global_image], dim=1)
        
        # Son projeksiyon
        output_features = self.output_proj(combined_features)
        
        # Modalite ağırlıklarını hesapla (çapraz dikkat ağırlıklarının ortalamasını alarak)
        # [batch_size, 2] boyutunda, [metin_ağırlığı, görüntü_ağırlığı] içerir
        text_weight = torch.mean(text_attn_weights, dim=1)  # [batch_size, image_seq_len] -> [batch_size] 
        image_weight = torch.mean(image_attn_weights, dim=1)  # [batch_size, text_seq_len] -> [batch_size]
        
        # Eğer text/image sequence uzunluğu 1 ise, boyut düşürmeye gerek yok
        if text_attn_weights.dim() > 2 and text_attn_weights.size(1) > 1:
            text_weight = text_weight.mean(dim=1)
        if image_attn_weights.dim() > 2 and image_attn_weights.size(1) > 1:
            image_weight = image_weight.mean(dim=1)
            
        # Normalize et ve [batch_size, 2] boyutuna getir
        total_weight = text_weight + image_weight
        modality_weights = torch.stack([
            text_weight / total_weight, 
            image_weight / total_weight
        ], dim=1)
        
        if return_attention:
            return output_features, modality_weights
        
        return output_features
    
    def _handle_text_only(self, text_features: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sadece metin modalitesi durumunu ele alır.
        
        Args:
            text_features: Metin özellikleri [batch_size, text_seq_len, text_dim] veya [batch_size, text_dim]
            text_mask: Metin özellikleri için maske [batch_size, text_seq_len]
            
        Returns:
            torch.Tensor: İşlenmiş metin özellikleri
        """
        batch_size = text_features.size(0)
        
        # 2D girdi (global özellikler) için 3D'ye genişlet
        if text_features.dim() == 2:
            text_features_3d = text_features.unsqueeze(1)
        else:
            text_features_3d = text_features
            
        # Global pooling (gerekiyorsa)
        if text_features_3d.size(1) > 1:
            if text_mask is not None:
                expanded_mask = text_mask.unsqueeze(-1)
                masked_sum = (text_features_3d * expanded_mask).sum(dim=1)
                text_sum = expanded_mask.sum(dim=1).clamp(min=1e-9)
                text_pooled = masked_sum / text_sum
            else:
                text_pooled = text_features_3d.mean(dim=1)
        else:
            text_pooled = text_features_3d.squeeze(1)
            
        # Son projeksiyon
        text_proj = self.text_proj(text_pooled)
        
        # Çıktı projeksiyonu - sadece metin için ikinci yerine aynı özelliği kullan
        combined_features = torch.cat([text_proj, text_proj], dim=1)
        output_features = self.output_proj(combined_features)
        
        # Dikkat ağırlıkları (sadece metin için [1.0, 0.0])
        modality_weights = torch.zeros(batch_size, 2, device=text_features.device)
        modality_weights[:, 0] = 1.0  # Tüm ağırlık metin modalitesine
        
        return output_features, modality_weights
    
    def _handle_image_only(self, image_features: torch.Tensor, image_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sadece görüntü modalitesi durumunu ele alır.
        
        Args:
            image_features: Görüntü özellikleri [batch_size, image_seq_len, image_dim] veya [batch_size, image_dim]
            image_mask: Görüntü özellikleri için maske [batch_size, image_seq_len]
            
        Returns:
            torch.Tensor: İşlenmiş görüntü özellikleri
        """
        batch_size = image_features.size(0)
        
        # 2D girdi (global özellikler) için 3D'ye genişlet
        if image_features.dim() == 2:
            image_features_3d = image_features.unsqueeze(1)
        else:
            image_features_3d = image_features
            
        # Global pooling (gerekiyorsa)
        if image_features_3d.size(1) > 1:
            if image_mask is not None:
                expanded_mask = image_mask.unsqueeze(-1)
                masked_sum = (image_features_3d * expanded_mask).sum(dim=1)
                image_sum = expanded_mask.sum(dim=1).clamp(min=1e-9)
                image_pooled = masked_sum / image_sum
            else:
                image_pooled = image_features_3d.mean(dim=1)
        else:
            image_pooled = image_features_3d.squeeze(1)
            
        # Son projeksiyon
        image_proj = self.image_proj(image_pooled)
        
        # Çıktı projeksiyonu - sadece görüntü için ikinci yerine aynı özelliği kullan
        combined_features = torch.cat([image_proj, image_proj], dim=1)
        output_features = self.output_proj(combined_features)
        
        # Dikkat ağırlıkları (sadece görüntü için [0.0, 1.0])
        modality_weights = torch.zeros(batch_size, 2, device=image_features.device)
        modality_weights[:, 1] = 1.0  # Tüm ağırlık görüntü modalitesine
        
        return output_features, modality_weights 