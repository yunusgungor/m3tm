"""
Arama Gömme Projeksiyonu Örneği

Bu örnek, M3TMSearchProjection sınıfının temel kullanımını gösterir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from m3tm.search import SearchProjectionFactory


def main():
    # Rastgele giriş vektörleri oluşturma
    batch_size = 3
    input_dim = 64
    inputs = torch.randn(batch_size, input_dim)
    
    # Arama projeksiyonu oluştur
    search_projection = SearchProjectionFactory.create(
        input_dim=input_dim,
        embedding_dim=32,
        dropout_rate=0.1,
        use_layer_norm=True
    )
    
    # Değerlendirme modu (inference)
    search_projection.eval()
    
    # Gömme projeksiyonu
    with torch.no_grad():
        search_embeddings = search_projection(inputs, return_dict=False)
    
    # Sonuçları görüntüle
    print(f"Girdi Şekli: {inputs.shape}")
    print(f"Gömme Şekli: {search_embeddings.shape}")
    
    # L2 normalizasyonu kontrolü (Her vektör bir birim vektör olmalı)
    for i in range(batch_size):
        norm = torch.norm(search_embeddings[i], p=2).item()
        print(f"Vektör {i} Normu: {norm:.6f}")
    
    # Benzerlik hesaplamaları
    print("\nEmbedding Benzerlik Matrisi (Kosinüs Benzerliği):")
    similarity = F.cosine_similarity(
        search_embeddings.unsqueeze(1), 
        search_embeddings.unsqueeze(0), 
        dim=2
    )
    print(similarity)


if __name__ == "__main__":
    main() 