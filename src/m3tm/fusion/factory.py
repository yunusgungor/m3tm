"""
Füzyon Fabrikası Modülü

Bu modül, farklı füzyon stratejilerini oluşturmak için fabrika deseni uygular.
İstenilen füzyon tipi ve yapılandırmasına göre uygun füzyon sınıfını örnekler.

Örüntü: FactoryMethod (PT-002)
"""

from typing import Dict, Optional, Type, Union, Any, List

import torch.nn as nn

from .config import (
    FusionConfig, 
    ConcatenationFusionConfig, 
    WeightedSumFusionConfig, 
    GatedFusionConfig, 
    FusionType
)
from .base_fusion import BaseFusion
from .fusion_strategies import ConcatenationFusion, WeightedSumFusion, GatedFusion


class FusionFactory:
    """
    Farklı füzyon mekanizmaları oluşturmak için fabrika sınıfı.
    
    Bu sınıf, yapılandırma nesnesindeki füzyon tipine göre uygun 
    füzyon modülünü oluşturur ve geri döndürür.
    
    Örüntü: FactoryMethod (PT-002)
    """
    
    @staticmethod
    def get_fusion_class(fusion_type: FusionType) -> Type[BaseFusion]:
        """
        Belirtilen füzyon tipine göre füzyon sınıfını döndürür.
        
        Args:
            fusion_type: Füzyon tipi (FusionType enum)
            
        Returns:
            Type[BaseFusion]: Füzyon sınıfı
            
        Raises:
            ValueError: Bilinmeyen füzyon tipi için
        """
        if fusion_type == FusionType.CONCATENATION:
            return ConcatenationFusion
        elif fusion_type == FusionType.WEIGHTED_SUM:
            return WeightedSumFusion
        elif fusion_type == FusionType.GATED:
            return GatedFusion
        else:
            raise ValueError(f"Bilinmeyen füzyon tipi: {fusion_type}")
    
    @staticmethod
    def get_config_class(fusion_type: FusionType) -> Type[FusionConfig]:
        """
        Belirtilen füzyon tipine göre yapılandırma sınıfını döndürür.
        
        Args:
            fusion_type: Füzyon tipi (FusionType enum)
            
        Returns:
            Type[FusionConfig]: Yapılandırma sınıfı
            
        Raises:
            ValueError: Bilinmeyen füzyon tipi için
        """
        if fusion_type == FusionType.CONCATENATION:
            return ConcatenationFusionConfig
        elif fusion_type == FusionType.WEIGHTED_SUM:
            return WeightedSumFusionConfig
        elif fusion_type == FusionType.GATED:
            return GatedFusionConfig
        else:
            raise ValueError(f"Bilinmeyen füzyon tipi: {fusion_type}")
    
    @classmethod
    def create_fusion(cls, config: FusionConfig) -> BaseFusion:
        """
        Yapılandırmaya göre füzyon modülü oluşturur.
        
        Args:
            config: Füzyon yapılandırması
            
        Returns:
            BaseFusion: Oluşturulan füzyon modülü
            
        Raises:
            ValueError: Bilinmeyen füzyon tipi veya yapılandırma tipi uyumsuzluğu için
        """
        fusion_type = config.fusion_type
        
        # Füzyon sınıfını al
        fusion_class = cls.get_fusion_class(fusion_type)
        
        # Yapılandırma tipini kontrol et
        expected_config_class = cls.get_config_class(fusion_type)
        if not isinstance(config, expected_config_class):
            # Basit yapılandırmayı genişlet
            config_kwargs = {
                key: getattr(config, key) 
                for key in dir(config) 
                if not key.startswith('_') and not callable(getattr(config, key))
            }
            config = expected_config_class(**config_kwargs)
        
        # Füzyon modülünü oluştur
        return fusion_class(config)
    
    @classmethod
    def create_concatenation_fusion(cls, 
                                   text_dim: int, 
                                   image_dim: int, 
                                   output_dim: Optional[int] = None,
                                   use_projection: bool = True,
                                   **kwargs) -> BaseFusion:
        """
        Concatenation füzyon modülü oluşturur.
        
        Args:
            text_dim: Metin girdi boyutu
            image_dim: Görüntü girdi boyutu
            output_dim: Çıktı boyutu (belirtilmezse text_dim + image_dim kullanılır)
            use_projection: Projeksiyon kullanılsın mı?
            **kwargs: Diğer yapılandırma parametreleri
            
        Returns:
            BaseFusion: Oluşturulan füzyon modülü
        """
        if output_dim is None:
            output_dim = text_dim + image_dim if not use_projection else min(512, text_dim + image_dim)
        
        config = ConcatenationFusionConfig(
            fusion_type=FusionType.CONCATENATION,
            text_dim=text_dim,
            image_dim=image_dim,
            output_dim=output_dim,
            use_projection=use_projection,
            **kwargs
        )
        
        return cls.create_fusion(config)
    
    @classmethod
    def create_weighted_sum_fusion(cls,
                                  text_dim: int,
                                  image_dim: int,
                                  output_dim: Optional[int] = None,
                                  learnable_weights: bool = True,
                                  initial_text_weight: float = 0.5,
                                  initial_image_weight: float = 0.5,
                                  normalize_weights: bool = True,
                                  **kwargs) -> BaseFusion:
        """
        Weighted Sum füzyon modülü oluşturur.
        
        Args:
            text_dim: Metin girdi boyutu
            image_dim: Görüntü girdi boyutu
            output_dim: Çıktı boyutu (belirtilmezse, giriş boyutları aynıysa o değer, 
                       farklıysa max(text_dim, image_dim) kullanılır)
            learnable_weights: Ağırlıklar öğrenilebilir mi?
            initial_text_weight: Metin modalitesi için başlangıç ağırlığı
            initial_image_weight: Görüntü modalitesi için başlangıç ağırlığı
            normalize_weights: Ağırlıklar toplamı 1'e normalize edilsin mi?
            **kwargs: Diğer yapılandırma parametreleri
            
        Returns:
            BaseFusion: Oluşturulan füzyon modülü
        """
        if output_dim is None:
            if text_dim == image_dim:
                output_dim = text_dim
            else:
                output_dim = max(text_dim, image_dim)
        
        config = WeightedSumFusionConfig(
            fusion_type=FusionType.WEIGHTED_SUM,
            text_dim=text_dim,
            image_dim=image_dim,
            output_dim=output_dim,
            learnable_weights=learnable_weights,
            initial_text_weight=initial_text_weight,
            initial_image_weight=initial_image_weight,
            normalize_weights=normalize_weights,
            **kwargs
        )
        
        return cls.create_fusion(config)
    
    @classmethod
    def create_gated_fusion(cls,
                           text_dim: int,
                           image_dim: int,
                           output_dim: Optional[int] = None,
                           gate_activation: str = "sigmoid",
                           hidden_dim: Optional[int] = None,
                           use_residual: bool = True,
                           **kwargs) -> BaseFusion:
        """
        Gated füzyon modülü oluşturur.
        
        Args:
            text_dim: Metin girdi boyutu
            image_dim: Görüntü girdi boyutu
            output_dim: Çıktı boyutu (belirtilmezse, giriş boyutları aynıysa o değer, 
                       farklıysa max(text_dim, image_dim) kullanılır)
            gate_activation: Gate aktivasyon fonksiyonu
            hidden_dim: Geçiş için gizli boyut (belirtilmezse output_dim // 2 kullanılır)
            use_residual: Artık bağlantı kullanılsın mı?
            **kwargs: Diğer yapılandırma parametreleri
            
        Returns:
            BaseFusion: Oluşturulan füzyon modülü
        """
        if output_dim is None:
            if text_dim == image_dim:
                output_dim = text_dim
            else:
                output_dim = max(text_dim, image_dim)
        
        config = GatedFusionConfig(
            fusion_type=FusionType.GATED,
            text_dim=text_dim,
            image_dim=image_dim,
            output_dim=output_dim,
            gate_activation=gate_activation,
            hidden_dim=hidden_dim,
            use_residual=use_residual,
            **kwargs
        )
        
        return cls.create_fusion(config)
    
    @classmethod
    def list_available_fusion_types(cls) -> List[str]:
        """
        Kullanılabilir füzyon tiplerini listeler.
        
        Returns:
            List[str]: Kullanılabilir füzyon tipleri listesi
        """
        return [fusion_type.value for fusion_type in FusionType] 