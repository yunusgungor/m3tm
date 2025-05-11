"""
Arama Gömme Modülü

Bu modül, arama için özelleştirilmiş gömme projeksiyonlarını içerir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple

class SearchEmbeddingProjection(nn.Module):
    """
    Arama gömme projeksiyonu sınıfı.
    
    Bu sınıf, çok modlu özellikleri arama uzayına göre yeniden biçimlendirir.
    Arama gömme vektörleri birim uzunluğa normalize edilir (L2 norm).
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Girdi özelliklerinin boyutu
            output_dim: Çıktı özelliklerinin boyutu (arama boyutu)
        """
        super(SearchEmbeddingProjection, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Doğrusal projeksiyon katmanı
        self.projection = nn.Linear(input_dim, output_dim)
        
        # Katman normalizasyonu
        self.layer_norm = nn.LayerNorm(output_dim, eps=1e-12)
        
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
        
        # L2 normalizasyonu
        search_embeddings = F.normalize(projections, p=2, dim=1)
        
        if return_dict:
            return {
                "search_embeddings": search_embeddings,
                "projections": projections
            }
        else:
            return search_embeddings 