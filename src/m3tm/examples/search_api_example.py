#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
M³TM Arama API Örneği

Bu örnek, M3TM Arama Servisi API'sinin temel işlevselliğini gösterir.
Rastgele gömme vektörleri oluşturur, bunları indeksler ve farklı sorgu türleriyle arama yapar.
"""

import os
import numpy as np
import torch
import time
import tempfile
import asyncio
from pathlib import Path

from m3tm.search import (
    SearchServiceFactory,
    SearchFilter,
    Pagination,
    SearchResultType
)


class DummyModel(torch.nn.Module):
    """
    Test için sahte bir model sınıfı.
    Gerçek bir M³TM modelini simüle eder.
    """
    
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
    
    def encode_text(self, text):
        """Metin gömmesi oluşturur."""
        # Metin uzunluğuna göre deterministik bir gömme oluştur
        text_hash = sum(ord(c) for c in text)
        torch.manual_seed(text_hash)
        return torch.randn(1, self.embed_dim)
    
    def encode_image(self, image):
        """Görüntü gömmesi oluşturur."""
        # Görüntü içeriğine göre deterministik bir gömme oluştur
        image_hash = int(torch.sum(image).item())
        torch.manual_seed(image_hash)
        return torch.randn(1, self.embed_dim)
    
    def fuse_embeddings(self, text_embedding, image_embedding):
        """Metin ve görüntü gömmelerini birleştirir."""
        if text_embedding is None:
            return image_embedding
        elif image_embedding is None:
            return text_embedding
        else:
            # Basit bir ortalama alma işlemi
            return (text_embedding + image_embedding) / 2


async def run_async_example(search_service):
    """Asenkron arama örneklerini çalıştırır."""
    print("\n=== Asenkron Arama Örnekleri ===")
    
    # Asenkron metin araması
    print("\nAsenkron metin araması yapılıyor...")
    text_results = await search_service.search_by_text_async(
        "asenkron metin sorgusu",
        top_k=3
    )
    print(f"Sonuç sayısı: {text_results.total_count}")
    print(f"Sorgu süresi: {text_results.query_time:.4f} saniye")
    
    # Asenkron görüntü araması
    print("\nAsenkron görüntü araması yapılıyor...")
    dummy_image = torch.randn(3, 224, 224)
    image_results = await search_service.search_by_image_async(
        dummy_image,
        top_k=3
    )
    print(f"Sonuç sayısı: {image_results.total_count}")
    print(f"Sorgu süresi: {image_results.query_time:.4f} saniye")
    
    # Asenkron çok modlu arama
    print("\nAsenkron çok modlu arama yapılıyor...")
    multimodal_results = await search_service.search_multimodal_async(
        text_query="asenkron çok modlu sorgu",
        image_tensor=dummy_image,
        top_k=3
    )
    print(f"Sonuç sayısı: {multimodal_results.total_count}")
    print(f"Sorgu süresi: {multimodal_results.query_time:.4f} saniye")


def main():
    """Arama API'sini test eden ana fonksiyon."""
    print("M³TM Arama API Örneği\n")
    
    # Sahte model oluştur
    model = DummyModel(embed_dim=512)
    
    # SearchService oluştur
    print("Arama servisi oluşturuluyor...")
    search_service = SearchServiceFactory.create(
        model=model,
        embedding_dim=128,
        index_type="flat",
        metric_type="cosine",
        use_gpu=False,
        enable_async=True
    )
    
    # Örnek veri ekle
    print("\n=== Veri Ekleme ===")
    
    # Metin verileri ekle
    print("\nMetin verileri ekleniyor...")
    text_ids = []
    for i in range(100):
        text = f"Örnek metin {i}"
        metadata = {
            "type": SearchResultType.TEXT.value,
            "title": f"Metin {i}",
            "source": "örnek",
            "timestamp": time.time(),
            "language": "tr",
            "category": ["örnek", "test"][i % 2]
        }
        text_id = search_service.add_text(text, metadata)
        text_ids.append(text_id)
    print(f"{len(text_ids)} metin eklendi.")
    
    # Görüntü verileri ekle
    print("\nGörüntü verileri ekleniyor...")
    image_ids = []
    for i in range(50):
        # Sahte görüntü tensörü
        image = torch.randn(3, 224, 224)
        metadata = {
            "type": SearchResultType.IMAGE.value,
            "title": f"Görüntü {i}",
            "source": "örnek",
            "timestamp": time.time(),
            "resolution": "224x224",
            "category": ["doğa", "şehir", "insan"][i % 3]
        }
        image_id = search_service.add_image(image, metadata)
        image_ids.append(image_id)
    print(f"{len(image_ids)} görüntü eklendi.")
    
    # Temel arama örneği
    print("\n=== Temel Arama Örnekleri ===")
    
    # Metin araması
    print("\nMetin araması yapılıyor...")
    text_results = search_service.search_by_text(
        "örnek metin sorgusu",
        top_k=5
    )
    print(f"Sonuç sayısı: {text_results.total_count}")
    print(f"Sorgu süresi: {text_results.query_time:.4f} saniye")
    print("İlk 3 sonuç:")
    for i, result in enumerate(text_results.results[:3]):
        print(f"  {i+1}. ID: {result.id}, Skor: {result.score:.4f}, Başlık: {result.metadata.get('title')}")
    
    # Görüntü araması
    print("\nGörüntü araması yapılıyor...")
    dummy_image = torch.randn(3, 224, 224)
    image_results = search_service.search_by_image(
        dummy_image,
        top_k=5
    )
    print(f"Sonuç sayısı: {image_results.total_count}")
    print(f"Sorgu süresi: {image_results.query_time:.4f} saniye")
    print("İlk 3 sonuç:")
    for i, result in enumerate(image_results.results[:3]):
        print(f"  {i+1}. ID: {result.id}, Skor: {result.score:.4f}, Başlık: {result.metadata.get('title')}")
    
    # Çok modlu arama
    print("\nÇok modlu arama yapılıyor...")
    multimodal_results = search_service.search_multimodal(
        text_query="çok modlu sorgu",
        image_tensor=dummy_image,
        top_k=5
    )
    print(f"Sonuç sayısı: {multimodal_results.total_count}")
    print(f"Sorgu süresi: {multimodal_results.query_time:.4f} saniye")
    print("İlk 3 sonuç:")
    for i, result in enumerate(multimodal_results.results[:3]):
        print(f"  {i+1}. ID: {result.id}, Skor: {result.score:.4f}, Başlık: {result.metadata.get('title')}")
    
    # Filtreleme örneği
    print("\n=== Filtreleme Örnekleri ===")
    
    # Kategori filtresi
    print("\nKategori filtresi ile arama yapılıyor...")
    category_filter = SearchFilter(
        metadata_filters={"category": "örnek"},
        min_score=0.5
    )
    filtered_results = search_service.search_by_text(
        "filtreli sorgu",
        top_k=10,
        filter_params=category_filter
    )
    print(f"Filtrelenmiş sonuç sayısı: {filtered_results.total_count}")
    print(f"Sorgu süresi: {filtered_results.query_time:.4f} saniye")
    
    # Tür filtresi
    print("\nTür filtresi ile arama yapılıyor...")
    type_filter = SearchFilter(
        result_type=SearchResultType.IMAGE
    )
    type_filtered_results = search_service.search_by_text(
        "görüntü sorgusu",
        top_k=10,
        filter_params=type_filter
    )
    print(f"Filtrelenmiş sonuç sayısı: {type_filtered_results.total_count}")
    print(f"Sorgu süresi: {type_filtered_results.query_time:.4f} saniye")
    
    # Sayfalama örneği
    print("\n=== Sayfalama Örnekleri ===")
    
    # İlk sayfa
    print("\nİlk sayfa sonuçları getiriliyor...")
    pagination = Pagination(page=0, page_size=5)
    page1_results = search_service.search_by_text(
        "sayfalama sorgusu",
        top_k=20,
        pagination=pagination
    )
    print(f"Toplam sonuç sayısı: {page1_results.total_count}")
    print(f"Toplam sayfa sayısı: {page1_results.total_pages}")
    print(f"Mevcut sayfa: {page1_results.page + 1}")
    print(f"Sayfadaki sonuç sayısı: {len(page1_results.results)}")
    
    # İkinci sayfa
    print("\nİkinci sayfa sonuçları getiriliyor...")
    pagination = Pagination(page=1, page_size=5)
    page2_results = search_service.search_by_text(
        "sayfalama sorgusu",
        top_k=20,
        pagination=pagination
    )
    print(f"Toplam sonuç sayısı: {page2_results.total_count}")
    print(f"Toplam sayfa sayısı: {page2_results.total_pages}")
    print(f"Mevcut sayfa: {page2_results.page + 1}")
    print(f"Sayfadaki sonuç sayısı: {len(page2_results.results)}")
    
    # İndeksi kaydet ve yükle
    print("\n=== İndeks Kaydetme ve Yükleme ===")
    
    # Geçici bir dizin oluştur
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = os.path.join(temp_dir, "search_index")
        
        # İndeksi kaydet
        print(f"\nİndeks kaydediliyor: {index_path}")
        search_service.save(index_path)
        
        # İndeksi yükle
        print(f"\nİndeks yükleniyor: {index_path}")
        loaded_service = search_service.load(index_path, model=model)
        
        # Yüklenen indeks ile arama yap
        print("\nYüklenen indeks ile arama yapılıyor...")
        loaded_results = loaded_service.search_by_text(
            "yüklenen indeks sorgusu",
            top_k=5
        )
        print(f"Sonuç sayısı: {loaded_results.total_count}")
        print(f"Sorgu süresi: {loaded_results.query_time:.4f} saniye")
    
    # Asenkron arama örnekleri
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_async_example(search_service))
    except RuntimeError as e:
        print(f"\nAsenkron örnekler çalıştırılamadı: {e}")
    
    print("\nArama API örneği tamamlandı.")


if __name__ == "__main__":
    main() 