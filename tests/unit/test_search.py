"""
Search modülü için birim testleri
"""
import os
import torch
import pytest
import numpy as np
import tempfile
import json

from m3tm.search.search_embedding import SearchEmbeddingProjection
from m3tm.search.search_index import SearchIndex
from m3tm.search.search_service import SearchService

class TestSearchEmbeddingProjection:
    """SearchEmbeddingProjection sınıfı testleri"""
    
    def test_embedding_projection_initialization(self):
        """SearchEmbeddingProjection başlatma testi"""
        input_dim = 64
        output_dim = 128
        projection = SearchEmbeddingProjection(input_dim, output_dim)
        
        assert projection.input_dim == input_dim
        assert projection.output_dim == output_dim
        assert isinstance(projection.projection, torch.nn.Linear)
        assert projection.projection.in_features == input_dim
        assert projection.projection.out_features == output_dim
    
    def test_embedding_projection_forward(self):
        """SearchEmbeddingProjection forward testi"""
        input_dim = 64
        output_dim = 128
        projection = SearchEmbeddingProjection(input_dim, output_dim)
        
        # Test girdisi
        batch_size = 4
        x = torch.rand(batch_size, input_dim)
        
        # Forward geçişi - dictionary yerine tensor almak için return_dict=False
        output = projection(x, return_dict=False)
        
        # Çıktı kontrolü
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
        
        # Normalizasyon kontrolü (çıktı vektörleri birim uzunlukta olmalı)
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

class TestSearchIndex:
    """SearchIndex sınıfı testleri"""
    
    def test_search_index_initialization(self):
        """SearchIndex başlatma testi"""
        embedding_dim = 128
        index = SearchIndex(embedding_dim)
        
        assert index.embedding_dim == embedding_dim
        assert len(index.embeddings) == 0
        assert len(index.metadata) == 0
    
    def test_search_index_add_and_search(self):
        """SearchIndex'e ekleme ve arama testi"""
        embedding_dim = 16  # Test için daha küçük boyut kullanıyoruz
        index = SearchIndex(embedding_dim)
        
        # 10 örnek embedding oluştur
        num_samples = 10
        embeddings = []
        for i in range(num_samples):
            # Birim vektör oluştur
            emb = torch.randn(embedding_dim)
            emb = emb / torch.norm(emb, p=2)
            embeddings.append(emb)
            
            # Metaveri
            metadata = {"id": f"item_{i}", "type": "test"}
            
            # Endekse ekle
            index.add(emb, metadata)
        
        # Doğrulama
        assert len(index.embeddings) == num_samples
        assert len(index.metadata) == num_samples
        
        # İlk eklenen embedding ile arama yap
        query_emb = embeddings[0]
        results = index.search(query_emb, top_k=3)
        
        # Sonuç kontrolü
        assert len(results) == 3
        # İlk sonuç, query_emb ile aynı olmalı (kendisi)
        assert results[0]['score'] > 0.99  # Benzerlik neredeyse 1 olmalı
        assert results[0]['metadata']['id'] == "item_0"
    
    def test_search_index_save_load(self):
        """SearchIndex kaydetme ve yükleme testi"""
        embedding_dim = 16
        index = SearchIndex(embedding_dim)
        
        # Endekse ekle
        for i in range(5):
            emb = torch.randn(embedding_dim)
            emb = emb / torch.norm(emb, p=2)
            index.add(emb, {"id": f"item_{i}"})
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Endeksi kaydet
            index.save(temp_filename)
            
            # Yeni endeks oluştur ve yükle
            loaded_index = SearchIndex(embedding_dim)
            loaded_index.load(temp_filename)
            
            # Doğrulama
            assert len(loaded_index.embeddings) == 5
            assert len(loaded_index.metadata) == 5
            
            # Orijinal ve yüklenen endeksteki embedding'leri karşılaştır
            for i in range(5):
                original_emb = index.embeddings[i]
                loaded_emb = loaded_index.embeddings[i]
                assert torch.allclose(original_emb, loaded_emb)
                
                original_meta = index.metadata[i]
                loaded_meta = loaded_index.metadata[i]
                assert original_meta == loaded_meta
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def test_search_index_batch_search(self):
        """SearchIndex toplu arama testi"""
        embedding_dim = 16
        index = SearchIndex(embedding_dim)
        
        # 20 örnek embedding oluştur
        for i in range(20):
            emb = torch.randn(embedding_dim)
            emb = emb / torch.norm(emb, p=2)
            index.add(emb, {"id": f"item_{i}"})
        
        # 5 sorgu oluştur
        query_batch = torch.stack([torch.randn(embedding_dim) for _ in range(5)])
        # Normalize
        query_batch = torch.nn.functional.normalize(query_batch, p=2, dim=1)
        
        # Toplu arama
        results = index.batch_search(query_batch, top_k=3)
        
        # Doğrulama
        assert len(results) == 5  # 5 sorgu
        for res in results:
            assert len(res) == 3  # Her sorgu için 3 sonuç
            # Skorların sıralı olduğunu kontrol et (azalan)
            scores = [item['score'] for item in res]
            assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

class TestSearchService:
    """SearchService sınıfı testleri"""
    
    def test_search_service_initialization(self):
        """SearchService başlatma testi"""
        embedding_dim = 128
        service = SearchService(embedding_dim)
        
        assert service.embedding_dim == embedding_dim
        assert isinstance(service.index, SearchIndex)
        assert service.index.embedding_dim == embedding_dim
    
    def test_search_service_embedding_and_search(self, mocker):
        """SearchService embedding hesaplama ve arama testi"""
        embedding_dim = 16
        service = SearchService(embedding_dim)
        
        # Model taklidi oluştur
        mock_embeddings = torch.nn.functional.normalize(torch.randn(10, embedding_dim), p=2, dim=1)
        
        # SearchService._get_embedding metodunu mockla
        mocker.patch.object(
            service, 
            '_get_embedding', 
            side_effect=lambda text, image=None: mock_embeddings[len(service.index.embeddings)]
        )
        
        # İçerik ekle
        for i in range(5):
            if i % 2 == 0:
                # Metin içeriği
                service.add_content(text=f"Sample text {i}", metadata={"id": f"text_{i}"})
            else:
                # Görüntü içeriği (görüntüyü simüle et)
                service.add_content(image="fake_image_tensor", metadata={"id": f"image_{i}"})
        
        # Doğrulama
        assert len(service.index.embeddings) == 5
        
        # Metinle arama
        text_results = service.search(text="sample query", top_k=2)
        assert len(text_results) == 2
        
        # Görüntüyle arama
        image_results = service.search(image="fake_image_tensor", top_k=3)
        assert len(image_results) == 3
        
        # Hatalı arama (hiçbir parametre girilmediğinde)
        with pytest.raises(ValueError):
            service.search()
            
        # Hatalı arama (hem metin hem görüntü verildiğinde - desteklenmiyor)
        with pytest.raises(ValueError):
            service.search(text="query", image="image")
    
    def test_search_service_save_load(self, mocker):
        """SearchService kaydetme ve yükleme testi"""
        embedding_dim = 16
        service = SearchService(embedding_dim)
        
        # Model taklidi oluştur
        mock_embeddings = torch.nn.functional.normalize(torch.randn(5, embedding_dim), p=2, dim=1)
        
        # SearchService._get_embedding metodunu mockla
        mocker.patch.object(
            service, 
            '_get_embedding', 
            side_effect=lambda text, image=None: mock_embeddings[len(service.index.embeddings)]
        )
        
        # İçerik ekle
        for i in range(5):
            service.add_content(text=f"Sample text {i}", metadata={"id": f"item_{i}"})
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Servisi kaydet
            service.save_index(temp_filename)
            
            # Yeni servis oluştur ve yükle
            new_service = SearchService(embedding_dim)
            new_service.load_index(temp_filename)
            
            # Doğrulama
            assert len(new_service.index.embeddings) == 5
            assert len(new_service.index.metadata) == 5
            
            # Arama yaparak doğrula
            mocker.patch.object(
                new_service, 
                '_get_embedding', 
                return_value=mock_embeddings[0]  # İlk embedding'i döndür
            )
            
            results = new_service.search(text="test query", top_k=3)
            assert len(results) == 3
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename) 