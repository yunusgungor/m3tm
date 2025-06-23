"""
Adaptif Ağırlıklandırma Füzyon Modülü

Bu modül, metin ve görüntü modalitelerini adaptif ağırlıklandırma ile
birleştirmek için kullanılır. Bu yaklaşım, modalitelerin katkısını 
içeriklerine göre dinamik olarak ayarlar.

Örüntü: AdaptiveWeighting (PT-028)
"""

from typing import Dict, Optional, Tuple, Union, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_fusion import BaseFusion
from .config import FusionConfig, FusionType


class AdaptiveWeightingFusion(BaseFusion):
    """
    Metin ve görüntü gömmelerini adaptif ağırlıklandırma ile birleştiren modül.
    Bu yaklaşım, her modalite için içeriğe bağlı olarak önem ağırlıkları hesaplar
    ve bu ağırlıkları kullanarak füzyon yapar.
    
    Örüntü: AdaptiveWeighting (PT-028)
    """
    
    def __init__(
        self, 
        config: Optional[FusionConfig] = None,
        text_dim: int = 768, 
        image_dim: int = 1024, 
        output_dim: int = 768, 
        dropout: float = 0.1
    ):
        """
        AdaptiveWeightingFusion sınıfını başlatır.
        
        Args:
            config: Füzyon yapılandırması
            text_dim: Metin özelliklerinin boyutu
            image_dim: Görüntü özelliklerinin boyutu
            output_dim: Çıktı özelliklerinin boyutu
            dropout: Dropout oranı
        """
        super().__init__(config or FusionConfig(fusion_type=FusionType.ADAPTIVE_WEIGHTING))
        
        # Config'den ya da argümanlardan boyutları belirle
        if config is not None:
            self.text_dim = config.text_dim
            self.image_dim = config.image_dim
            self.output_dim = config.output_dim
        else:
            self.text_dim = text_dim
            self.image_dim = image_dim
            self.output_dim = output_dim
            
        # Metin projeksiyon
        self.text_proj = nn.Linear(self.text_dim, self.output_dim)
        self.text_norm = nn.LayerNorm(self.output_dim)
        
        # Görüntü projeksiyon
        self.image_proj = nn.Linear(self.image_dim, self.output_dim)
        self.image_norm = nn.LayerNorm(self.output_dim)
        
        # Modalite ağırlıklarını hesaplayan MLP
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.output_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)  # İki modalite için softmax
        )

    def get_parameter_count(self) -> int:
        """Toplam parametre sayısını döndürür."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, 
        text_features: torch.Tensor, 
        image_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        İki modaliteyi adaptif ağırlıklandırma ile birleştirir.
        
        Args:
            text_features: Metin özellikleri [batch_size, text_seq_len, text_dim] veya [batch_size, text_dim]
            image_features: Görüntü özellikleri [batch_size, image_seq_len, image_dim] veya [batch_size, image_dim]
            text_mask: Metin özellikleri için maske [batch_size, text_seq_len]
            image_mask: Görüntü özellikleri için maske [batch_size, image_seq_len]
            return_weights: Modalite ağırlıklarının döndürülüp döndürülmeyeceği
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                Birleştirilmiş özellikler ve opsiyonel olarak modalite ağırlıkları
        """
        # Tek modalite durumunu ele al
        if image_features is None:
            return self._handle_text_only(text_features, text_mask)
        elif text_features is None:
            return self._handle_image_only(image_features, image_mask)
        
        batch_size = text_features.size(0)
        device = text_features.device
        
        # Metin kalitesi bilgisini başlangıçta None olarak ayarla
        text_quality_factor = None
        
        # Global metin özellikleri
        if text_features.dim() > 2:
            # Maske varsa, metin kalitesini hesapla (Padding token oranı)
            if text_mask is not None:
                # Maskeyi [0,1] olarak kullan; 1: gerçek token, 0: padding
                text_quality_factor = text_mask.float().mean(dim=1)  # [batch_size]
            
            # [CLS] token veya maskeleme ile ortalama
            if text_mask is not None:
                # Maskeyi genişlet
                expanded_mask = text_mask.unsqueeze(-1)
                # Maskelenen kısımların 0 olduğunu varsayarak ortalama al
                masked_sum = (text_features * expanded_mask).sum(dim=1)
                text_sum = expanded_mask.sum(dim=1).clamp(min=1e-9)
                global_text = masked_sum / text_sum
            else:
                # [CLS] token'ı kullan (ilk token) veya ortalama al
                global_text = text_features[:, 0] if text_features.size(1) > 1 else text_features.mean(dim=1)
        else:
            # Zaten global özellik
            global_text = text_features
            
            # Özellik vektöründen kalite tahmini yapabilmek için
            # L2 normunu hesaplayarak metin kalitesini tahmin etmeyi dene 
            # (Zayıf metin genellikle daha düşük norma sahip olur)
            if text_quality_factor is None and text_mask is None:
                text_feature_norm = torch.norm(global_text, p=2, dim=1)
                max_norm = text_feature_norm.max()
                if max_norm > 0:
                    text_quality_factor = text_feature_norm / max_norm  # [0, 1] aralığına normalize et
        
        # Global görüntü özellikleri
        if image_features.dim() > 2:
            # Patch embedding veya maskeleme ile ortalama
            if image_mask is not None:
                # Maskeyi genişlet
                expanded_mask = image_mask.unsqueeze(-1)
                # Maskelenen kısımların 0 olduğunu varsayarak ortalama al
                masked_sum = (image_features * expanded_mask).sum(dim=1)
                image_sum = expanded_mask.sum(dim=1).clamp(min=1e-9)
                global_image = masked_sum / image_sum
            else:
                # İlk token'ı kullan (varsa) veya ortalama al
                global_image = image_features[:, 0] if image_features.size(1) > 1 else image_features.mean(dim=1)
        else:
            # Zaten global özellik
            global_image = image_features
        
        # Projecsiyon ve normalizasyon
        text_projected = self.text_norm(self.text_proj(global_text))
        image_projected = self.image_norm(self.image_proj(global_image))
        
        # Ağırlık tahmini için özellikleri birleştir
        combined_features = torch.cat([text_projected, image_projected], dim=1)
        
        # Adaptif ağırlıkları hesapla
        weights = self.attention_mlp(combined_features)
        
        # Debug ve karşılaştırma amaçlı orijinal ağırlıkları kopyala
        original_weights = weights.clone()
        
        # Metin kalitesi bilgisi varsa, güçlü bir şekilde uygula
        if text_quality_factor is not None:
            # print(f"Text quality factor: {text_quality_factor}")  # Debug
            
            # Kalite faktörünü daha büyük bir etki yaratacak şekilde güçlendir
            # 0.5'in altındaki değerler daha da düşürülsün
            enhanced_factor = torch.pow(text_quality_factor, 2.0)  # Karesi alınarak etkisi artırılır
            
            # Ağırlıkları yeniden dengele
            adjusted_weights = torch.zeros_like(weights, device=device)
            
            # 0-1 arası normalize edilmiş text_quality_factor değerine göre ağırlıkları güncelle
            # Düşük kalite (düşük factor) -> Görüntü ağırlığı artar
            # Yüksek kalite (yüksek factor) -> Metin ağırlığı korunur
            adjusted_weights[:, 0] = weights[:, 0] * enhanced_factor  # Metin ağırlığını azalt
            adjusted_weights[:, 1] = 1.0 - adjusted_weights[:, 0]  # Görüntü ağırlığını artır
            
            # Test için ağırlıkların değişimini yazdır
            # print(f"Original weights: {weights}")
            # print(f"Adjusted weights: {adjusted_weights}")
            
            weights = adjusted_weights
            
        # Ağırlıklı toplam
        weighted_text = text_projected * weights[:, 0].unsqueeze(1)
        weighted_image = image_projected * weights[:, 1].unsqueeze(1)
        
        # Modaliteleri birleştir
        fused_features = weighted_text + weighted_image
        
        # Son işleme
        output_features = fused_features
        
        if return_weights:
            return output_features, weights
        
        return output_features
    
    def fuse(
        self, 
        text_embeddings: torch.Tensor, 
        image_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None, 
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Metin ve görüntü gömmelerini adaptif ağırlıklandırma ile füzyon yapar.
        BaseFusion arayüzü ile uyumlu kalması için forward'a yönlendirir.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, image_dim]
            text_mask: Metin maskesi
            image_mask: Görüntü maskesi
            return_weights: Modalite ağırlıklarının döndürülüp döndürülmeyeceği
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                Füzyon çıktısı ve opsiyonel olarak modalite ağırlıkları
        """
        return self.forward(
            text_embeddings, 
            image_embeddings, 
            text_mask, 
            image_mask, 
            return_weights
        )
    
    def _handle_text_only(
        self, 
        text_features: torch.Tensor, 
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sadece metin modalitesi durumunu ele alır.
        
        Args:
            text_features: Metin özellikleri [batch_size, text_seq_len, text_dim] veya [batch_size, text_dim]
            text_mask: Metin özellikleri için maske [batch_size, text_seq_len]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: İşlenmiş metin özellikleri ve modalite ağırlıkları
        """
        batch_size = text_features.size(0)
        
        # Global metin özellikleri
        if text_features.dim() > 2:
            if text_mask is not None:
                expanded_mask = text_mask.unsqueeze(-1)
                masked_sum = (text_features * expanded_mask).sum(dim=1)
                text_sum = expanded_mask.sum(dim=1).clamp(min=1e-9)
                global_text = masked_sum / text_sum
            else:
                global_text = text_features[:, 0] if text_features.size(1) > 1 else text_features.mean(dim=1)
        else:
            global_text = text_features
        
        # Projeksiyon
        text_projected = self.text_norm(self.text_proj(global_text))
        
        # Son işleme
        output_features = text_projected
        
        # Ağırlıklar (sadece metin için [1.0, 0.0])
        weights = torch.zeros(batch_size, 2, device=text_features.device)
        weights[:, 0] = 1.0  # Tüm ağırlık metin modalitesine
        
        return output_features, weights
    
    def _handle_image_only(
        self, 
        image_features: torch.Tensor, 
        image_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sadece görüntü modalitesi durumunu ele alır.
        
        Args:
            image_features: Görüntü özellikleri [batch_size, image_seq_len, image_dim] veya [batch_size, image_dim]
            image_mask: Görüntü özellikleri için maske [batch_size, image_seq_len]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: İşlenmiş görüntü özellikleri ve modalite ağırlıkları
        """
        batch_size = image_features.size(0)
        
        # Global görüntü özellikleri
        if image_features.dim() > 2:
            if image_mask is not None:
                expanded_mask = image_mask.unsqueeze(-1)
                masked_sum = (image_features * expanded_mask).sum(dim=1)
                image_sum = expanded_mask.sum(dim=1).clamp(min=1e-9)
                global_image = masked_sum / image_sum
            else:
                global_image = image_features[:, 0] if image_features.size(1) > 1 else image_features.mean(dim=1)
        else:
            global_image = image_features
        
        # Projeksiyon
        image_projected = self.image_norm(self.image_proj(global_image))
        
        # Son işleme
        output_features = image_projected
        
        # Ağırlıklar (sadece görüntü için [0.0, 1.0])
        weights = torch.zeros(batch_size, 2, device=image_features.device)
        weights[:, 1] = 1.0  # Tüm ağırlık görüntü modalitesine
        
        return output_features, weights 