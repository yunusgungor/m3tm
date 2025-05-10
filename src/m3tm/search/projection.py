"""
M³TM Arama Gömme Projeksiyonu

Bu modül, model çıktılarını semantik arama için normalize edilmiş bir gömme uzayına
projekte eden M3TMSearchProjection sınıfını içerir. Bu modül, farklı modalitelerden (metin, görüntü)
gelen temsilleri arama için optimize edilmiş bir uzaya dönüştürür ve benzerlik aramaları için 
kullanılabilir hale getirir.

Örüntü: EmbeddingNormalizer (PT-007)
"""

from typing import Optional, Dict, Any, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchProjectionConfig:
    """
    Arama gömme projeksiyonu yapılandırması.
    
    Attributes:
        input_dim (int): Girdi boyutu.
        embedding_dim (int): Arama gömme uzayının boyutu.
        dropout_rate (float): Dropout oranı.
        use_layer_norm (bool): Layer normalization kullanılıp kullanılmayacağı.
        layer_norm_eps (float): Layer normalization için epsilon değeri.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-12,
    ):
        """
        SearchProjectionConfig sınıfını başlatır.
        
        Args:
            input_dim: Girdi boyutu.
            embedding_dim: Arama gömme uzayının boyutu (varsayılan: 128).
            dropout_rate: Dropout oranı (varsayılan: 0.1).
            use_layer_norm: Layer normalization kullanılıp kullanılmayacağı (varsayılan: True).
            layer_norm_eps: Layer normalization için epsilon değeri (varsayılan: 1e-12).
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.layer_norm_eps = layer_norm_eps


class M3TMSearchProjection(nn.Module):
    """
    Model çıktılarını arama için optimize edilmiş bir gömme uzayına projekte eden sınıf.
    
    Bu sınıf, farklı modalitelerden gelen model çıktılarını bir lineer projeksiyon ve 
    L2 normalizasyonu kullanarak birim vektörlere dönüştürür. Bu vektörler, semantik arama
    için benzerlik hesaplamalarında kullanılabilir.
    
    Örüntü: EmbeddingNormalizer (PT-007)
    """
    
    def __init__(self, config: SearchProjectionConfig):
        """
        M3TMSearchProjection sınıfını başlatır.
        
        Args:
            config: Arama gömme projeksiyonu yapılandırması.
        """
        super().__init__()
        self.config = config
        
        # Lineer projeksiyon
        self.projection = nn.Linear(config.input_dim, config.embedding_dim)
        
        # Opsiyonel layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Gömmeleri arama uzayına projekte eder.
        
        Args:
            embeddings: Projekte edilecek gömmeler [batch_size, input_dim].
            return_dict: Çıktıyı sözlük olarak döndür, False ise sadece arama gömmesini döndür.
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 
                Normalize edilmiş arama gömmeleri veya çıktı sözlüğü.
                Sözlük olarak döndürülürse şu anahtarları içerir:
                - 'search_embeddings': L2 normalleştirilmiş arama gömmeleri
                - 'projections': Normalleştirilmemiş projeksiyon çıktıları
        """
        # Girdi boyutunu kontrol et
        if embeddings.dim() == 3:
            # [batch_size, seq_len, input_dim] durumunun ortalama havuzlama ile işlenmesi
            # Bu, füzyon çıktısı gibi dizi formunda gömmeleri işler
            embeddings = embeddings.mean(dim=1)
        
        # Boyut kontrolü
        if embeddings.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Girdi boyutu ({embeddings.size(-1)}) yapılandırma ile uyuşmuyor "
                f"(input_dim={self.config.input_dim})."
            )
        
        # Lineer projeksiyon
        projections = self.projection(embeddings)
        
        # Normalizasyon ve dropout
        if hasattr(self, 'layer_norm'):
            projections = self.layer_norm(projections)
        
        projections = self.dropout(projections)
        
        # L2 normalizasyonu (p=2, dim=-1)
        # Birim vektörler oluşturmak için
        search_embeddings = F.normalize(projections, p=2, dim=-1)
        
        if return_dict:
            return {
                'search_embeddings': search_embeddings,
                'projections': projections
            }
        
        return search_embeddings


class SearchProjectionFactory:
    """
    Arama gömme projeksiyonu oluşturmak için fabrika sınıfı.
    
    Örüntü: Factory (PT-002)
    """
    
    @staticmethod
    def create(
        input_dim: int,
        embedding_dim: int = 128,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-12,
    ) -> M3TMSearchProjection:
        """
        Yeni bir M3TMSearchProjection örneği oluşturur.
        
        Args:
            input_dim: Girdi boyutu.
            embedding_dim: Arama gömme uzayının boyutu (varsayılan: 128).
            dropout_rate: Dropout oranı (varsayılan: 0.1).
            use_layer_norm: Layer normalization kullanılıp kullanılmayacağı (varsayılan: True).
            layer_norm_eps: Layer normalization için epsilon değeri (varsayılan: 1e-12).
            
        Returns:
            M3TMSearchProjection: Arama gömme projeksiyonu modülü.
        """
        config = SearchProjectionConfig(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
        )
        
        return M3TMSearchProjection(config) 