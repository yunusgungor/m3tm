"""
Arama Gömme Modülü

Bu modül, arama için özelleştirilmiş gömme projeksiyonlarını içerir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri geçiş.
        
        Args:
            x: Girdi özellikleri [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Normalize edilmiş arama gömmeleri [batch_size, output_dim]
        """
        # Doğrusal projeksiyon
        embeddings = self.projection(x)
        
        # L2 normalizasyonu
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return normalized_embeddings 