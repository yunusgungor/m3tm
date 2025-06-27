"""
Arama Gömme Modülü

Bu modül, arama için özelleştirilmiş gömme projeksiyonlarını içerir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple, Optional

from ..config.model_config import SearchConfig

class SearchEmbeddingProjection(nn.Module):
    """
    Arama gömme projeksiyonu sınıfı.
    
    Bu sınıf, çok modlu özellikleri arama uzayına göre yeniden biçimlendirir.
    Arama gömme vektörleri birim uzunluğa normalize edilir (L2 norm).
    """
    
    def __init__(self, 
                config_or_input_dim: Union[SearchConfig, int], 
                output_dim: Optional[int] = None,
                use_normalization: bool = True,
                metric: str = "cosine"):
        """
        Args:
            config_or_input_dim: Arama yapılandırması veya girdi boyutu
            output_dim: Çıktı boyutu (config kullanılmadığında gerekli)
            use_normalization: Normalizasyon kullanılıp kullanılmayacağı
            metric: Kullanılacak metrik ("cosine", "inner_product" vs.)
        """
        super(SearchEmbeddingProjection, self).__init__()
        
        # SearchConfig kontrolü - import path farklılıkları nedeniyle attribute kontrolü yapalım
        if hasattr(config_or_input_dim, 'input_dim') and hasattr(config_or_input_dim, 'search_dim'):
            self.config = config_or_input_dim
            self.input_dim = self.config.input_dim
            self.output_dim = self.config.search_dim
            self.use_normalization = getattr(self.config, 'use_normalization', True)
            self.metric = getattr(self.config, 'metric', 'cosine')
        else:
            if output_dim is None:
                raise ValueError("output_dim, SearchConfig yerine input_dim kullanıldığında gereklidir")
            self.config = None
            self.input_dim = config_or_input_dim
            self.output_dim = output_dim
            self.use_normalization = use_normalization
            self.metric = metric
        
        # Doğrusal projeksiyon katmanı
        self.projection = nn.Linear(self.input_dim, self.output_dim)
        
        # Katman normalizasyonu
        self.layer_norm = nn.LayerNorm(self.output_dim, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(0.0)  # Default olarak dropout yok
    
    def forward(self, x: torch.Tensor, return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Girdi tensörü, shape: [batch_size, hidden_dim]
            return_dict: Sözlük dönmeyi belirler, False ise doğrudan embedding tensörü döner
            
        Returns:
            return_dict=True ise:
                Dict[str, Tensor]: {'projections': çıktı tensörü}
                
            return_dict=False ise:
                Tensor: çıktı tensörü, [batch_size, search_dim]
        """
        # Doğrusal projeksiyon
        x = self.projection(x)
        
        # Normalizasyon (eğer isteniyorsa)
        if self.use_normalization:
            # Katman normalizasyonu
            x = self.layer_norm(x)
            
            # Dropout
            x = self.dropout(x)
            
            # L2 normalizasyon
            if self.metric == "cosine":
                x = F.normalize(x, p=2, dim=-1)
        
        if return_dict:
            return {"projections": x}
        return x 