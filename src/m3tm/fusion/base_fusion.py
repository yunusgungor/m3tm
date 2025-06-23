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
    
    def __init__(self, 
                config_or_text_dim: Union[FusionConfig, int],
                image_dim: Optional[int] = None,
                output_dim: Optional[int] = None):
        """
        BaseFusion sınıfını başlatır.
        
        Args:
            config_or_text_dim: Füzyon yapılandırması (FusionConfig) veya metin boyutu
            image_dim: Görüntü boyutu (config_or_text_dim bir FusionConfig değilse kullanılır)
            output_dim: Çıktı boyutu (config_or_text_dim bir FusionConfig değilse kullanılır)
        """
        super().__init__()
        
        # FusionConfig veya boyut parametrelerini kullanarak yapılandırma oluştur
        if isinstance(config_or_text_dim, FusionConfig):
            self.config = config_or_text_dim
        else:
            if image_dim is None or output_dim is None:
                raise ValueError("image_dim ve output_dim, config yerine text_dim kullanıldığında gereklidir")
            
            # Basit bir yapılandırma nesnesi oluştur (FusionConfig değil)
            self.config = type('SimpleConfig', (), {
                'text_dim': config_or_text_dim,
                'image_dim': image_dim,
                'output_dim': output_dim,
                'use_layer_norm': False,
                'layer_norm_eps': 1e-12,
                'dropout_rate': 0.1,
                'fusion_type': None,
                'num_attention_heads': 4  # Test için gereken özellik
            })
        
        # Varsayılan değerleri ayarla (custom config nesneleri için)
        if not hasattr(self.config, 'use_layer_norm'):
            self.config.use_layer_norm = False
        
        if not hasattr(self.config, 'layer_norm_eps'):
            self.config.layer_norm_eps = 1e-12
            
        if not hasattr(self.config, 'dropout_rate'):
            self.config.dropout_rate = 0.1
            
        if not hasattr(self.config, 'num_attention_heads'):
            self.config.num_attention_heads = 4
        
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
        if not hasattr(self.config, 'text_dim') or not hasattr(self.config, 'image_dim'):
            raise ValueError(
                "config nesnesi text_dim ve image_dim özelliklerine sahip olmalıdır."
            )
        
        # output_dim belirtilmemişse hata ver
        if not hasattr(self.config, 'output_dim'):
            raise ValueError("config nesnesi output_dim özelliğine sahip olmalıdır.")
    
    def _check_input_dimensions(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> None:
        """
        Girdi tensor boyutlarını kontrol eder.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim] veya [batch_size, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim] veya [batch_size, image_dim]
        
        Raises:
            ValueError: Girdi boyutları yapılandırma ile uyuşmazsa
        """
        # Son boyutu kontrol et
        text_dim = text_embeddings.size(-1)
        if text_dim != self.config.text_dim:
            raise ValueError(
                f"text_embeddings boyutu ({text_dim}) yapılandırma ile "
                f"uyuşmuyor (text_dim={self.config.text_dim})."
            )
        
        # Son boyutu kontrol et
        image_dim = image_embeddings.size(-1)
        if image_dim != self.config.image_dim:
            raise ValueError(
                f"image_embeddings boyutu ({image_dim}) yapılandırma ile "
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
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim] veya [batch_size, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim] veya [batch_size, image_dim]
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
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim] veya [batch_size, text_dim]
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim] veya [batch_size, image_dim]
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
        
        # Çıktı formatını belirle
        if return_dict:
            return {
                "fused_embeddings": output
            }
        else:
            return output
    
    def _handle_text_only(self, 
                         text_embeddings: torch.Tensor, 
                         text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece metin modalitesi durumunu ele alır.
        
        Args:
            text_embeddings: Metin gömmeleri [batch_size, seq_len, text_dim] veya [batch_size, text_dim]
            text_mask: Metin maskesi [batch_size, seq_len]
            
        Returns:
            torch.Tensor: İşlenmiş metin gömmeleri
        """
        # Varsayılan davranış - alt sınıflar bu metodu override edebilir
        batch_size = text_embeddings.size(0)
        
        # Metin gömmelerini düzleştir (eğer sequence ise)
        if text_embeddings.dim() > 2:
            # Text masking
            if text_mask is not None:
                text_mask = text_mask.unsqueeze(-1)
                text_embeddings = (text_embeddings * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1e-6)
            else:
                text_embeddings = text_embeddings.mean(dim=1)
        
        # Görüntü modalitesi için sıfır vektörü
        dummy_image_embeddings = torch.zeros(
            batch_size, self.config.image_dim, 
            device=text_embeddings.device, 
            dtype=text_embeddings.dtype
        )
        
        # İki modaliteyi birleştir, bu durumda image modalitesi sıfır
        return self.fuse(text_embeddings, dummy_image_embeddings)
    
    def _handle_image_only(self, 
                          image_embeddings: torch.Tensor, 
                          image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sadece görüntü modalitesi durumunu ele alır.
        
        Args:
            image_embeddings: Görüntü gömmeleri [batch_size, num_patches, image_dim] veya [batch_size, image_dim]
            image_mask: Görüntü maskesi [batch_size, num_patches]
            
        Returns:
            torch.Tensor: İşlenmiş görüntü gömmeleri
        """
        # Varsayılan davranış - alt sınıflar bu metodu override edebilir
        batch_size = image_embeddings.size(0)
        
        # Görüntü gömmelerini düzleştir (eğer sequence ise)
        if image_embeddings.dim() > 2:
            # Image masking
            if image_mask is not None:
                image_mask = image_mask.unsqueeze(-1)
                image_embeddings = (image_embeddings * image_mask).sum(dim=1) / image_mask.sum(dim=1).clamp(min=1e-6)
            else:
                image_embeddings = image_embeddings.mean(dim=1)
        
        # Metin modalitesi için sıfır vektörü
        dummy_text_embeddings = torch.zeros(
            batch_size, self.config.text_dim, 
            device=image_embeddings.device, 
            dtype=image_embeddings.dtype
        )
        
        # İki modaliteyi birleştir, bu durumda text modalitesi sıfır
        return self.fuse(dummy_text_embeddings, image_embeddings) 