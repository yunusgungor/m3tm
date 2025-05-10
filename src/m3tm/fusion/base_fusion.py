"""
Temel Füzyon Mekanizması Modülü

Bu modül, farklı modaliteleri (metin, görüntü) birleştirmek için kullanılan
temel füzyon mekanizmasının soyut sınıfını içerir. Tüm füzyon stratejileri
bu temel sınıftan türetilir.

Örüntü: ModelComposite (PT-003)
"""

import abc
from typing import Dict, Optional, Tuple, Union, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig, FusionType


class BaseFusion(nn.Module, abc.ABC):
    """
    Farklı modalite gömmelerini birleştirmek için temel füzyon sınıfı.
    Bu soyut sınıf, tüm füzyon stratejileri için temel işlevleri ve
    arayüzü tanımlar.
    
    Örüntü: ModelComposite (PT-003)
    """
    
    def __init__(self, config: FusionConfig):
        """
        BaseFusion sınıfını başlatır.
        
        Args:
            config: Füzyon yapılandırması
        """
        super().__init__()
        self.config = config
        
        # Boyutları ayarla veya doğrula
        self._validate_and_set_dimensions()
        
        # Ortak katmanlar
        
        # Normalizasyon
        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(
                self.config.output_dim,
                eps=self.config.layer_norm_eps
            )
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout_rate)
    
    def _validate_and_set_dimensions(self):
        """
        Boyutları doğrular ve gerekirse varsayılan değerleri ayarlar.
        Alt sınıflar bu metodu override edebilir.
        """
        # text_dim ve image_dim her zaman belirtilmelidir
        if self.config.text_dim is None or self.config.image_dim is None:
            raise ValueError(
                "text_dim ve image_dim belirtilmelidir."
            )
        
        # output_dim belirtilmemişse, füzyon tipine göre varsayılan değeri hesapla
        if self.config.output_dim is None:
            if self.config.fusion_type == FusionType.CONCATENATION:
                self.config.output_dim = self.config.text_dim + self.config.image_dim
            else:  # WEIGHTED_SUM, GATED ve diğerleri için
                # Giriş boyutları farklı ise uyumlu olmayabilir
                if self.config.text_dim != self.config.image_dim:
                    raise ValueError(
                        f"{self.config.fusion_type.value} füzyon için text_dim ve image_dim "
                        f"aynı olmalıdır veya output_dim açıkça belirtilmelidir."
                    )
                self.config.output_dim = self.config.text_dim
    
    def _check_input_dimensions(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> None:
        """
        Girdi tensor boyutlarını kontrol eder.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
        
        Raises:
            ValueError: Girdi boyutları yapılandırma ile uyuşmazsa
        """
        if text_embeddings.size(-1) != self.config.text_dim:
            raise ValueError(
                f"text_embeddings boyutu ({text_embeddings.size(-1)}) yapılandırma ile "
                f"uyuşmuyor (text_dim={self.config.text_dim})."
            )
        
        if image_embeddings.size(-1) != self.config.image_dim:
            raise ValueError(
                f"image_embeddings boyutu ({image_embeddings.size(-1)}) yapılandırma ile "
                f"uyuşmuyor (image_dim={self.config.image_dim})."
            )
        
        # Batch boyutları uyumlu olmalı
        if text_embeddings.size(0) != image_embeddings.size(0):
            raise ValueError(
                f"Batch boyutları uyuşmuyor: text_embeddings batch_size={text_embeddings.size(0)}, "
                f"image_embeddings batch_size={image_embeddings.size(0)}."
            )
    
    def _process_masks(self, 
                       text_mask: Optional[torch.Tensor] = None, 
                       image_mask: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Metin ve görüntü maskelerini işler ve füzyon maskeleri oluşturur.
        
        Alt sınıfların ihtiyaçlarına göre override edilebilir.
        
        Args:
            text_mask: Metin maskesi [batch_size, seq_len]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            Tuple[Optional[torch.Tensor], ...]: İşlenmiş maskeler
        """
        return text_mask, image_mask
    
    @abc.abstractmethod
    def fuse(self, 
            text_embeddings: torch.Tensor, 
            image_embeddings: torch.Tensor,
            text_mask: Optional[torch.Tensor] = None,
            image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Metin ve görüntü gömmelerini birleştirir.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: Birleştirilmiş gömmeler
        """
        pass
    
    def forward(self, 
               text_embeddings: Optional[torch.Tensor] = None, 
               image_embeddings: Optional[torch.Tensor] = None,
               text_mask: Optional[torch.Tensor] = None,
               image_mask: Optional[torch.Tensor] = None,
               return_dict: bool = True) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Füzyon modülünün ileri geçişi.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            return_dict: Çıktı sözlük formatında döndürülsün mü?
            
        Returns:
            torch.Tensor | Dict[str, Any]: Birleştirilmiş gömmeler veya çıktı sözlüğü
            
        Raises:
            ValueError: text_embeddings ve image_embeddings aynı anda None ise
        """
        # Girdileri kontrol et
        if text_embeddings is None and image_embeddings is None:
            raise ValueError("En az bir modalite (text_embeddings veya image_embeddings) sağlanmalıdır.")
        
        # Tek modalite durumlarını ele al
        if text_embeddings is None:
            # Sadece görüntü modalitesi sağlanmış
            # Görüntü gömmelerini çıkış boyutuna projekte et
            output = self._handle_image_only(image_embeddings, image_mask)
        elif image_embeddings is None:
            # Sadece metin modalitesi sağlanmış
            # Metin gömmelerini çıkış boyutuna projekte et
            output = self._handle_text_only(text_embeddings, text_mask)
        else:
            # Her iki modalite de sağlanmış, boyutları kontrol et
            self._check_input_dimensions(text_embeddings, image_embeddings)
            
            # Maskeleri işle
            processed_masks = self._process_masks(text_mask, image_mask)
            
            # Füzyon işlemini gerçekleştir
            output = self.fuse(
                text_embeddings, 
                image_embeddings,
                processed_masks[0] if processed_masks else None,
                processed_masks[1] if len(processed_masks) > 1 else None
            )
            
            # Normalizasyon ve dropout uygula
            if hasattr(self, 'layer_norm'):
                output = self.layer_norm(output)
            
            output = self.dropout(output)
        
        # Çıktıyı oluştur
        if return_dict:
            return {
                "fused_embeddings": output,
                "fusion_type": self.config.fusion_type.value
            }
        else:
            return output
    
    def _handle_text_only(self, 
                         text_embeddings: torch.Tensor, 
                         text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece metin modalitesi durumunu ele alır.
        Alt sınıflar bu metodu override edebilir.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            
        Returns:
            torch.Tensor: İşlenmiş metin gömmeleri
        """
        # Varsayılan davranış: metin gömmelerini olduğu gibi döndür
        # Alt sınıflar, metin gömmelerini çıkış boyutuna projekte eden bir projeksiyon tanımlayabilir
        return text_embeddings
    
    def _handle_image_only(self, 
                          image_embeddings: torch.Tensor, 
                          image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece görüntü modalitesi durumunu ele alır.
        Alt sınıflar bu metodu override edebilir.
        
        Args:
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: İşlenmiş görüntü gömmeleri
        """
        # Varsayılan davranış: görüntü gömmelerini olduğu gibi döndür
        # Alt sınıflar, görüntü gömmelerini çıkış boyutuna projekte eden bir projeksiyon tanımlayabilir
        return image_embeddings 