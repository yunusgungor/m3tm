#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
M³TM Arama İndeksleme Örneği

Bu örnek, M3TM Arama İndeksleme Modülünün temel işlevselliğini gösterir.
Rastgele gömme vektörleri oluşturur, bunları indeksler ve basit bir arama yapar.
"""

import os
import numpy as np
import torch
import time
import tempfile
from pathlib import Path

from m3tm.search import (
    SearchIndexFactory,
    SearchProjectionFactory,
    SearchIndex
)


def main():
    """Arama indeksleme modülünü test eden ana fonksiyon."""
    print("M³TM Arama İndeksleme Örneği\n")
    
    # Yapılandırma parametreleri
    embedding_dim = 128
    num_samples = 1000
    
    # SearchProjection örneği oluştur
    projection = SearchProjectionFactory.create(
        input_dim=512,  # Simüle edilmiş model çıktı boyutu
        embedding_dim=embedding_dim
    )
    print(f"Arama projeksiyonu oluşturuldu: {embedding_dim} boyutlu")
    
    # SearchIndex örneği oluştur
    use_gpu = torch.cuda.is_available()
    with tempfile.TemporaryDirectory() as temp_dir:
        index = SearchIndexFactory.create(
            embedding_dim=embedding_dim,
            index_type="flat",  # "flat", "hnsw", "ivf" arasından seçim yapılabilir
            metric_type="cosine",
            use_gpu=use_gpu,
            storage_path=temp_dir
        )
        print(f"Arama indeksi oluşturuldu: type={index.config.index_type}, "
              f"metric={index.config.metric_type}, GPU={index.config.use_gpu}")
        
        # Rastgele "görüntü" gömmelerini simüle et
        print(f"\nRastgele {num_samples} görüntü gömmesi oluşturuluyor...")
        simulated_embeddings = torch.randn(num_samples, 512)  # Simüle edilmiş model çıktıları
        
        # Görüntü gömmelerini arama uzayına projekte et
        print("Gömmeler arama uzayına projekte ediliyor...")
        search_embeddings = projection(simulated_embeddings, return_dict=False)
        
        # Örnek metadata hazırla
        metadata_list = [
            {"type": "image", "content": f"Sample image {i}", "timestamp": time.time()}
            for i in range(num_samples)
        ]
        
        # Gömmeleri indeksle
        print("Gömmeler indeksleniyor...")
        start_time = time.time()
        ids = index.add(search_embeddings, metadata_list)
        end_time = time.time()
        print(f"İndeksleme tamamlandı. Geçen süre: {end_time - start_time:.4f} saniye")
        print(f"İndeks büyüklüğü: {index.get_size()} öğe")
        
        # Rastgele bir sorgu vektörü oluştur
        print("\nRastgele bir sorgu vektörü oluşturuluyor...")
        query_embedding = torch.randn(1, 512)  # Simüle edilmiş model çıktısı
        query_search_embedding = projection(query_embedding, return_dict=False)
        
        # En yakın k sonucu ara
        k = 5
        print(f"En yakın {k} sonuç aranıyor...")
        start_time = time.time()
        distances, result_ids, result_metadata = index.search(query_search_embedding, k=k)
        end_time = time.time()
        print(f"Arama tamamlandı. Geçen süre: {end_time - start_time:.4f} saniye")
        
        # Sonuçları göster
        print("\nArama Sonuçları:")
        for i in range(min(k, len(result_ids[0]))):
            rid = int(result_ids[0][i])
            distance = distances[0][i]
            metadata = result_metadata[i]
            print(f"#{i+1}: ID={rid}, Benzerlik={distance:.4f}, İçerik={metadata.get('content', 'N/A')}")
        
        # İndeksi kaydet ve yeniden yükle
        print("\nİndeks kaydediliyor...")
        index_path = Path(temp_dir) / "test_index"
        os.makedirs(index_path, exist_ok=True)
        index.save(index_path)
        print(f"İndeks kaydedildi: {index_path}")
        
        print("\nKaydedilen indeks yükleniyor...")
        loaded_index = SearchIndex.load(index_path)
        print(f"Yüklenen indeks büyüklüğü: {loaded_index.get_size()} öğe")
        
        # Yüklenen indeks ile aynı sorguyu tekrarla
        print(f"Yüklenen indekste arama yapılıyor...")
        distances, result_ids, result_metadata = loaded_index.search(query_search_embedding, k=k)
        
        # Sonuçları göster
        print("\nYüklenen İndeks - Arama Sonuçları:")
        for i in range(min(k, len(result_ids[0]))):
            rid = int(result_ids[0][i])
            distance = distances[0][i]
            metadata = result_metadata[i]
            print(f"#{i+1}: ID={rid}, Benzerlik={distance:.4f}, İçerik={metadata.get('content', 'N/A')}")
        
        print("\nTest başarıyla tamamlandı!")


if __name__ == "__main__":
    main() 