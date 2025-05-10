"""
Füzyon Stratejileri Modülü

Bu modül, farklı modaliteleri (metin, görüntü) birleştirmek için kullanılan
çeşitli füzyon stratejilerini uygulamaktadır.

Örüntü: ModelComposite (PT-003)
"""

from typing import Dict, Optional, Tuple, Union, Any, List, Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    FusionConfig, 
    ConcatenationFusionConfig, 
    WeightedSumFusionConfig, 
    GatedFusionConfig, 
    FusionType
)
from .base_fusion import BaseFusion


class ConcatenationFusion(BaseFusion):
    """
    Metin ve görüntü gömmelerini birleştirme yoluyla füzyon yapan modül.
    Bu basit strateji, gömmeler tek bir tensöre birleştirilir ve isteğe bağlı
    olarak bir projeksiyon uygulanır.
    
    Örüntü: ModelComposite (PT-003)
    """
    
    def __init__(self, config: ConcatenationFusionConfig):
        """
        ConcatenationFusion sınıfını başlatır.
        
        Args:
            config: Concatenation füzyon yapılandırması
        """
        super().__init__(config)
        self.config = config
        
        # Projeksiyon katmanı (opsiyonel)
        self.projection = None
        if config.use_projection:
            # Concatenation sonucu oluşan vektörü output_dim boyutuna projekte et
            concat_dim = self.config.text_dim + self.config.image_dim
            self.projection = nn.Linear(concat_dim, self.config.output_dim)
            
            # İlklendirme
            nn.init.normal_(self.projection.weight, std=self.config.init_std)
            nn.init.zeros_(self.projection.bias)
    
    def _global_pooling(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gömmelere global pooling uygulayarak tek bir vektöre dönüştürür.
        
        Args:
            embeddings: Gömmeler [batch_size, seq_len/num_patches, dim]
            mask: Maske [batch_size, seq_len/num_patches]
            
        Returns:
            torch.Tensor: Global pooling uygulanmış gömmeler [batch_size, dim]
        """
        if mask is not None:
            # Maskeyi genişlet [batch_size, seq_len/num_patches, 1]
            mask = mask.unsqueeze(-1)
            # Maskeli ortalama al
            masked_embeddings = embeddings * mask
            # Maskenin toplamı (epsilon ekleyerek sıfıra bölünmeyi önle)
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            # Maskeli ortalama hesapla
            pooled = masked_embeddings.sum(dim=1) / mask_sum.squeeze(-1)
        else:
            # Basit ortalama
            pooled = embeddings.mean(dim=1)
        
        return pooled
    
    def fuse(self, 
            text_embeddings: torch.Tensor, 
            image_embeddings: torch.Tensor,
            text_mask: Optional[torch.Tensor] = None,
            image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Metin ve görüntü gömmelerini birleştirme yoluyla füzyon yapar.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: Birleştirilmiş gömmeler
        """
        # Sequence/patch boyutunda ortalama al
        text_pooled = self._global_pooling(text_embeddings, text_mask)
        image_pooled = self._global_pooling(image_embeddings, image_mask)
        
        # Birleştir [batch_size, text_dim + image_dim]
        fused = torch.cat([text_pooled, image_pooled], dim=-1)
        
        # Projeksiyon uygula (varsa)
        if self.projection is not None:
            fused = self.projection(fused)
        
        # Çıktı boyutu: [batch_size, output_dim]
        return fused
    
    def _handle_text_only(self, 
                         text_embeddings: torch.Tensor, 
                         text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece metin modalitesi durumunu ele alır.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            
        Returns:
            torch.Tensor: İşlenmiş metin gömmeleri
        """
        text_pooled = self._global_pooling(text_embeddings, text_mask)
        
        # Projeksiyon varsa ve boyutlar uyuşmuyorsa
        if self.projection is not None and text_pooled.size(-1) != self.config.output_dim:
            # Dummy image gömmeleri oluştur (sıfır tensörü)
            batch_size = text_pooled.size(0)
            dummy_image = torch.zeros(batch_size, self.config.image_dim, device=text_pooled.device)
            
            # Concatenate ve projekte et
            concat = torch.cat([text_pooled, dummy_image], dim=-1)
            return self.projection(concat)
        
        return text_pooled
    
    def _handle_image_only(self, 
                          image_embeddings: torch.Tensor, 
                          image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece görüntü modalitesi durumunu ele alır.
        
        Args:
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: İşlenmiş görüntü gömmeleri
        """
        image_pooled = self._global_pooling(image_embeddings, image_mask)
        
        # Projeksiyon varsa ve boyutlar uyuşmuyorsa
        if self.projection is not None and image_pooled.size(-1) != self.config.output_dim:
            # Dummy text gömmeleri oluştur (sıfır tensörü)
            batch_size = image_pooled.size(0)
            dummy_text = torch.zeros(batch_size, self.config.text_dim, device=image_pooled.device)
            
            # Concatenate ve projekte et
            concat = torch.cat([dummy_text, image_pooled], dim=-1)
            return self.projection(concat)
        
        return image_pooled


class WeightedSumFusion(BaseFusion):
    """
    Metin ve görüntü gömmelerini ağırlıklı toplama yoluyla füzyon yapan modül.
    Bu strateji, gömmeler ağırlıklı olarak toplanır, isteğe bağlı olarak
    ağırlıklar öğrenilebilir olabilir.
    
    Örüntü: ModelComposite (PT-003)
    """
    
    def __init__(self, config: WeightedSumFusionConfig):
        """
        WeightedSumFusion sınıfını başlatır.
        
        Args:
            config: WeightedSum füzyon yapılandırması
        """
        super().__init__(config)
        self.config = config
        
        # Projeksiyon katmanları (farklı boyutta gömmeler için)
        self.text_projection = None
        self.image_projection = None
        
        # Giriş boyutları çıkış boyutundan farklıysa, projeksiyon kullan
        if self.config.text_dim != self.config.output_dim:
            self.text_projection = nn.Linear(self.config.text_dim, self.config.output_dim)
            nn.init.normal_(self.text_projection.weight, std=self.config.init_std)
            nn.init.zeros_(self.text_projection.bias)
            
        if self.config.image_dim != self.config.output_dim:
            self.image_projection = nn.Linear(self.config.image_dim, self.config.output_dim)
            nn.init.normal_(self.image_projection.weight, std=self.config.init_std)
            nn.init.zeros_(self.image_projection.bias)
        
        # Ağırlıklar
        if config.learnable_weights:
            # Öğrenilebilir ağırlıklar
            self.text_weight = nn.Parameter(torch.tensor(config.initial_text_weight))
            self.image_weight = nn.Parameter(torch.tensor(config.initial_image_weight))
        else:
            # Sabit ağırlıklar
            self.register_buffer("text_weight", torch.tensor(config.initial_text_weight))
            self.register_buffer("image_weight", torch.tensor(config.initial_image_weight))
    
    def _global_pooling(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gömmelere global pooling uygulayarak tek bir vektöre dönüştürür.
        
        Args:
            embeddings: Gömmeler [batch_size, seq_len/num_patches, dim]
            mask: Maske [batch_size, seq_len/num_patches]
            
        Returns:
            torch.Tensor: Global pooling uygulanmış gömmeler [batch_size, dim]
        """
        if mask is not None:
            # Maskeyi genişlet [batch_size, seq_len/num_patches, 1]
            mask = mask.unsqueeze(-1)
            # Maskeli ortalama al
            masked_embeddings = embeddings * mask
            # Maskenin toplamı (epsilon ekleyerek sıfıra bölünmeyi önle)
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            # Maskeli ortalama hesapla
            pooled = masked_embeddings.sum(dim=1) / mask_sum.squeeze(-1)
        else:
            # Basit ortalama
            pooled = embeddings.mean(dim=1)
        
        return pooled
    
    def _normalize_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ağırlıkları normalize eder.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Normalize edilmiş ağırlıklar
        """
        if self.config.normalize_weights:
            weight_sum = self.text_weight + self.image_weight
            return self.text_weight / weight_sum, self.image_weight / weight_sum
        else:
            return self.text_weight, self.image_weight
    
    def fuse(self, 
            text_embeddings: torch.Tensor, 
            image_embeddings: torch.Tensor,
            text_mask: Optional[torch.Tensor] = None,
            image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Metin ve görüntü gömmelerini ağırlıklı toplama yoluyla füzyon yapar.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: Ağırlıklı toplanmış gömmeler
        """
        # Sequence/patch boyutunda ortalama al
        text_pooled = self._global_pooling(text_embeddings, text_mask)
        image_pooled = self._global_pooling(image_embeddings, image_mask)
        
        # Projeksiyon uygula (gerekiyorsa)
        if self.text_projection is not None:
            text_pooled = self.text_projection(text_pooled)
            
        if self.image_projection is not None:
            image_pooled = self.image_projection(image_pooled)
        
        # Ağırlıkları normalize et
        norm_text_weight, norm_image_weight = self._normalize_weights()
        
        # Ağırlıklı toplama
        fused = norm_text_weight * text_pooled + norm_image_weight * image_pooled
        
        # Çıktı boyutu: [batch_size, output_dim]
        return fused
    
    def _handle_text_only(self, 
                         text_embeddings: torch.Tensor, 
                         text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece metin modalitesi durumunu ele alır.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            
        Returns:
            torch.Tensor: İşlenmiş metin gömmeleri
        """
        text_pooled = self._global_pooling(text_embeddings, text_mask)
        
        # Projeksiyon uygula (gerekiyorsa)
        if self.text_projection is not None:
            text_pooled = self.text_projection(text_pooled)
        
        return text_pooled
    
    def _handle_image_only(self, 
                          image_embeddings: torch.Tensor, 
                          image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece görüntü modalitesi durumunu ele alır.
        
        Args:
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: İşlenmiş görüntü gömmeleri
        """
        image_pooled = self._global_pooling(image_embeddings, image_mask)
        
        # Projeksiyon uygula (gerekiyorsa)
        if self.image_projection is not None:
            image_pooled = self.image_projection(image_pooled)
        
        return image_pooled


class GatedFusion(BaseFusion):
    """
    Metin ve görüntü gömmelerini gate mekanizması ile birleştiren modül.
    Bu daha gelişmiş strateji, bir modaliteye diğerine göre dinamik olarak
    farklı önem ağırlıkları atamak için öğrenilebilir gate mekanizması kullanır.
    
    Örüntü: ModelComposite (PT-003)
    """
    
    def __init__(self, config: GatedFusionConfig):
        """
        GatedFusion sınıfını başlatır.
        
        Args:
            config: Gated füzyon yapılandırması
        """
        super().__init__(config)
        self.config = config
        
        # Projeksiyon katmanları (farklı boyutta gömmeler için)
        self.text_projection = None
        self.image_projection = None
        
        # Giriş boyutları çıkış boyutundan farklıysa, projeksiyon kullan
        if self.config.text_dim != self.config.output_dim:
            self.text_projection = nn.Linear(self.config.text_dim, self.config.output_dim)
            nn.init.normal_(self.text_projection.weight, std=self.config.init_std)
            nn.init.zeros_(self.text_projection.bias)
            
        if self.config.image_dim != self.config.output_dim:
            self.image_projection = nn.Linear(self.config.image_dim, self.config.output_dim)
            nn.init.normal_(self.image_projection.weight, std=self.config.init_std)
            nn.init.zeros_(self.image_projection.bias)
        
        # Gate mekanizması için kullanılacak hidden boyut
        if config.hidden_dim is None:
            hidden_dim = self.config.output_dim // 2
        else:
            hidden_dim = config.hidden_dim
        
        # Gate ağı - Metin ve görüntü gate'leri ayrı ayrı öğrenir
        self.gate_network = nn.Sequential(
            nn.Linear(self.config.output_dim * 2, hidden_dim),
            self._get_activation(activation=config.gate_activation)
        )
        
        # Gate çıkış projeksiyonu - Son gate değerlerini üretir
        self.gate_output = nn.Linear(hidden_dim, self.config.output_dim)
        
        # İlklendirme
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                nn.init.zeros_(module.bias)
        
        nn.init.normal_(self.gate_output.weight, std=self.config.init_std)
        nn.init.zeros_(self.gate_output.bias)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """
        İstenilen aktivasyon fonksiyonunu döndürür.
        
        Args:
            activation: Aktivasyon fonksiyonu adı
            
        Returns:
            nn.Module: PyTorch aktivasyon modülü
        """
        if activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif activation == "hardswish":
            return nn.Hardswish()
        else:
            raise ValueError(f"Bilinmeyen aktivasyon fonksiyonu: {activation}")
    
    def _global_pooling(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gömmelere global pooling uygulayarak tek bir vektöre dönüştürür.
        
        Args:
            embeddings: Gömmeler [batch_size, seq_len/num_patches, dim]
            mask: Maske [batch_size, seq_len/num_patches]
            
        Returns:
            torch.Tensor: Global pooling uygulanmış gömmeler [batch_size, dim]
        """
        if mask is not None:
            # Maskeyi genişlet [batch_size, seq_len/num_patches, 1]
            mask = mask.unsqueeze(-1)
            # Maskeli ortalama al
            masked_embeddings = embeddings * mask
            # Maskenin toplamı (epsilon ekleyerek sıfıra bölünmeyi önle)
            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            # Maskeli ortalama hesapla
            pooled = masked_embeddings.sum(dim=1) / mask_sum.squeeze(-1)
        else:
            # Basit ortalama
            pooled = embeddings.mean(dim=1)
        
        return pooled
    
    def fuse(self, 
            text_embeddings: torch.Tensor, 
            image_embeddings: torch.Tensor,
            text_mask: Optional[torch.Tensor] = None,
            image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Metin ve görüntü gömmelerini gate mekanizması ile füzyon yapar.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: Gate mekanizması ile füzyon yapılmış gömmeler
        """
        # Sequence/patch boyutunda ortalama al
        text_pooled = self._global_pooling(text_embeddings, text_mask)
        image_pooled = self._global_pooling(image_embeddings, image_mask)
        
        # Projeksiyon uygula (gerekiyorsa)
        if self.text_projection is not None:
            text_pooled = self.text_projection(text_pooled)
            
        if self.image_projection is not None:
            image_pooled = self.image_projection(image_pooled)
        
        # Gate mekanizması için birleşik gömmeler
        concat_features = torch.cat([text_pooled, image_pooled], dim=-1)
        
        # Gate değerlerini hesapla
        gate_hidden = self.gate_network(concat_features)
        gate_values = self.gate_output(gate_hidden)
        
        # Sigmoid (0-1 arası) gate değerleri - text vs image
        if self.config.gate_activation == "sigmoid":
            # Gate'in sigmoid olması durumunda, metin için gate_values,
            # görüntü için (1 - gate_values) kullan
            fused = gate_values * text_pooled + (1.0 - gate_values) * image_pooled
        else:
            # Gate'in diğer aktivasyonlar için, her iki gömmayı farklı gate'le ölçekle
            gate_values_complement = 1.0 - torch.abs(gate_values)  # Tamamlayıcı gate
            fused = torch.abs(gate_values) * text_pooled + gate_values_complement * image_pooled
        
        # Artık bağlantı (opsiyonel)
        if self.config.use_residual:
            # Basit ortalama
            avg_embedding = (text_pooled + image_pooled) / 2.0
            fused = fused + avg_embedding
        
        # Çıktı boyutu: [batch_size, output_dim]
        return fused
    
    def _handle_text_only(self, 
                         text_embeddings: torch.Tensor, 
                         text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece metin modalitesi durumunu ele alır.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            
        Returns:
            torch.Tensor: İşlenmiş metin gömmeleri
        """
        text_pooled = self._global_pooling(text_embeddings, text_mask)
        
        # Projeksiyon uygula (gerekiyorsa)
        if self.text_projection is not None:
            text_pooled = self.text_projection(text_pooled)
        
        return text_pooled
    
    def _handle_image_only(self, 
                          image_embeddings: torch.Tensor, 
                          image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece görüntü modalitesi durumunu ele alır.
        
        Args:
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: İşlenmiş görüntü gömmeleri
        """
        image_pooled = self._global_pooling(image_embeddings, image_mask)
        
        # Projeksiyon uygula (gerekiyorsa)
        if self.image_projection is not None:
            image_pooled = self.image_projection(image_pooled)
        
        return image_pooled 