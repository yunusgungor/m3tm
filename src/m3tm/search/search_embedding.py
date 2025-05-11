"""
Arama Gömme Modülü

Bu modül, arama için özelleştirilmiş gömme projeksiyonlarını içerir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple

from m3tm.config.model_config import SearchConfig

class SearchEmbeddingProjection(nn.Module):
    """
    Arama gömme projeksiyonu sınıfı.
    
    Bu sınıf, çok modlu özellikleri arama uzayına göre yeniden biçimlendirir.
    Arama gömme vektörleri birim uzunluğa normalize edilir (L2 norm).
    """
    
    def __init__(self, config: SearchConfig):
        """
        Args:
            config: Arama yapılandırması
        """
        super(SearchEmbeddingProjection, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.search_dim
        self.use_normalization = config.use_normalization
        self.metric = config.metric
        
        # Doğrusal projeksiyon katmanı
        self.projection = nn.Linear(self.input_dim, self.output_dim)
        
        # Katman normalizasyonu
        self.layer_norm = nn.LayerNorm(self.output_dim, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(0.0)  # Default olarak dropout yok
    
    def forward(self, x: torch.Tensor, return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        İleri geçiş.
        
        Args:
            x: Girdi özellikleri [batch_size, input_dim] veya [batch_size, seq_len, input_dim]
            return_dict: True ise çıktıyı sözlük olarak döndürür, False ise doğrudan tensor olarak
            
        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 
                Normalize edilmiş arama gömmeleri [batch_size, output_dim]
                veya bunları içeren bir sözlük
        """
        # Giriş boyutlarını kontrol et
        if x.dim() == 3:
            # [batch_size, seq_len, input_dim] -> Ortalama havuzlama yap
            x = x.mean(dim=1)
        
        # Girdi boyutunu doğrula
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[-1]}")
        
        # Doğrusal projeksiyon
        projections = self.projection(x)
        
        # Layer norm
        projections = self.layer_norm(projections)
        
        # Dropout
        projections = self.dropout(projections)
        
        # Konfigürasyona göre normalizasyon
        if self.use_normalization:
            if self.metric == "cosine" or self.metric == "inner_product":
                # L2 normalizasyonu
                search_embeddings = F.normalize(projections, p=2, dim=1)
            else:
                # Diğer metrikler için normalizasyon yok
                search_embeddings = projections
        else:
            search_embeddings = projections
        
        if return_dict:
            return {
                "search_embedding": search_embeddings,  # backward compatibility için search_embedding
                "search_embeddings": search_embeddings,
                "projections": projections
            }
        else:
            return search_embeddings 