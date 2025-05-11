"""
M³TM modelinin end-to-end entegrasyon testleri
"""
import os
import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path

from m3tm.config.model_config import get_tiny_config
from m3tm.config.config_base import ConfigBase
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.embedding.text_embedding import TextEmbedding
from m3tm.embedding.image_embedding import ImagePatchEmbedding
from m3tm.fusion.basic_fusion import BasicFusion
from m3tm.search.search_embedding import SearchEmbeddingProjection
from m3tm.search.search_service import SearchService
from m3tm.adapters.adapter_manager import create_adapter
from m3tm.task_heads.classification_head import ClassificationHead
from m3tm.training.trainer import TextClassificationModel, Trainer
from m3tm.data_export.export_manager import ExportManager

class TestEndToEndModel:
    """End-to-end model entegrasyon testleri"""
    
    def setup_tiny_model(self):
        """Test için küçük M³TM modeli oluşturur"""
        config = get_tiny_config()
        
        # Metin gömme
        text_embedding = TextEmbedding(config.text_config)
        
        # Görüntü gömme
        image_embedding = ImagePatchEmbedding(config.image_config)
        
        # Transformer blokları
        transformer_blocks = torch.nn.ModuleList([
            ProtoTransformerBlock(config.transformer_config) 
            for _ in range(config.num_core_blocks)
        ])
        
        # Füzyon
        fusion = BasicFusion(
            config.fusion_config.text_embed_dim,
            config.fusion_config.image_embed_dim,
            config.fusion_config.fused_embed_dim
        )
        
        # Arama gömme projeksiyonu
        search_projection = SearchEmbeddingProjection(
            config.fusion_config.fused_embed_dim,
            config.search_config.search_embed_dim
        )
        
        return {
            'config': config,
            'text_embedding': text_embedding,
            'image_embedding': image_embedding,
            'transformer_blocks': transformer_blocks,
            'fusion': fusion,
            'search_projection': search_projection
        }
    
    @pytest.mark.slow
    def test_multimodal_text_image_flow(self):
        """Metin ve görüntü verisiyle end-to-end akış testi"""
        model_components = self.setup_tiny_model()
        config = model_components['config']
        
        # Test girdileri
        batch_size = 2
        text_input = torch.randint(0, config.text_config.vocab_size, (batch_size, 16))
        image_input = torch.rand(batch_size, 3, 32, 32)  # Küçük test görüntüleri
        
        # Metin akışı
        text_features = model_components['text_embedding'](text_input)
        assert text_features.shape == (batch_size, 16, config.text_config.embed_dim)
        
        # Görüntü akışı
        image_features = model_components['image_embedding'](image_input)
        # Yamaların sayısı (32/4)² = 64 olmalı (eğer patch_size=4 ise)
        expected_num_patches = (32 // config.image_config.patch_size) ** 2
        assert image_features.shape == (batch_size, expected_num_patches, config.image_config.embed_dim)
        
        # Transformer işleme (her modalite için ayrı)
        for block in model_components['transformer_blocks']:
            text_features = block(text_features)
            image_features = block(image_features)
        
        # Füzyon
        # Global pooling
        text_pooled = torch.mean(text_features, dim=1)
        image_pooled = torch.mean(image_features, dim=1)
        
        fused_features = model_components['fusion'](text_pooled, image_pooled)
        assert fused_features.shape == (batch_size, config.fusion_config.fused_embed_dim)
        
        # Arama gömme
        search_embeddings = model_components['search_projection'](fused_features)
        assert search_embeddings.shape == (batch_size, config.search_config.search_embed_dim)
        
        # Normalizasyon kontrolü
        norms = torch.norm(search_embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    @pytest.mark.slow
    def test_adapter_integration(self):
        """Adaptör entegrasyonu testi"""
        model_components = self.setup_tiny_model()
        config = model_components['config']
        transformer_blocks = model_components['transformer_blocks']
        
        # Test girdisi
        batch_size = 2
        seq_len = 16
        input_tensor = torch.rand(batch_size, seq_len, config.transformer_config.embed_dim)
        
        # Adaptör ekleme öncesi forward
        original_outputs = []
        for block in transformer_blocks:
            input_tensor = block(input_tensor)
            original_outputs.append(input_tensor.clone())
        
        # Adaptör ekleyip yeniden çalıştır
        adapter_size = 8
        for i, block in enumerate(transformer_blocks):
            block.add_adapter(f"adapter_{i}", adapter_size=adapter_size)
        
        # Adaptör ekleme sonrası forward
        input_tensor = torch.rand(batch_size, seq_len, config.transformer_config.embed_dim)  # Aynı giriş şekli
        adapter_outputs = []
        for block in transformer_blocks:
            input_tensor = block(input_tensor)
            adapter_outputs.append(input_tensor.clone())
        
        # Çıktıların boyutları aynı kalmalı ama değerleri farklılaşmalı
        for i in range(len(original_outputs)):
            assert original_outputs[i].shape == adapter_outputs[i].shape
            # Çıktılar farklı olmalı (adaptör değişiklik yapmalı)
            assert not torch.allclose(original_outputs[i], adapter_outputs[i], atol=1e-4)
    
    @pytest.mark.slow
    def test_classification_head_integration(self):
        """Sınıflandırma başlığı ve eğitim entegrasyonu testi"""
        model_components = self.setup_tiny_model()
        config = model_components['config']
        
        # Sınıflandırma başlığı
        num_classes = 5
        classification_head = ClassificationHead(
            config.fusion_config.fused_embed_dim,
            hidden_dim=32,
            num_classes=num_classes,
            dropout=0.0  # Test için determinizmi koruyoruz
        )
        
        # Test girdileri
        batch_size = 4
        text_input = torch.randint(0, config.text_config.vocab_size, (batch_size, 16))
        
        # Text classification model oluştur
        model = TextClassificationModel(
            text_embedding=model_components['text_embedding'],
            transformer_blocks=model_components['transformer_blocks'],
            classification_head=classification_head
        )
        
        # Forward
        logits = model(text_input)
        assert logits.shape == (batch_size, num_classes)
        
        # Eğitim simülasyonu (tek bir adım)
        labels = torch.randint(0, num_classes, (batch_size,))
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward ve backward
        optimizer.zero_grad()
        logits = model(text_input)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Pozitif kayıplar olmalı ve gradyanlar hesaplanmış olmalı
        assert loss.item() > 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} parametresinin gradyanları yok"
    
    @pytest.mark.slow
    def test_search_service_integration(self):
        """Arama servisi entegrasyonu testi"""
        model_components = self.setup_tiny_model()
        config = model_components['config']
        
        # SearchService oluştur
        search_dim = config.search_config.search_embed_dim
        search_service = SearchService(search_dim)
        
        # Sahte embedding döndüren mock fonksiyon
        def mock_get_embedding(text=None, image=None):
            return torch.nn.functional.normalize(torch.randn(search_dim), p=2, dim=0)
        
        # Mock metodu yerleştir
        original_method = search_service._get_embedding
        search_service._get_embedding = mock_get_embedding
        
        try:
            # İçerik ekle
            for i in range(10):
                search_service.add_content(
                    text=f"Sample content {i}",
                    metadata={"id": f"doc_{i}", "type": "text"}
                )
            
            # Arama yap
            results = search_service.search(text="test query", top_k=3)
            
            # Doğrulama
            assert len(results) == 3
            assert "score" in results[0]
            assert "metadata" in results[0]
            assert results[0]["metadata"]["id"].startswith("doc_")
            
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            try:
                # İndeksi kaydet
                search_service.save_index(temp_filename)
                
                # Yeni servis oluştur, indeksi yükle
                new_service = SearchService(search_dim)
                new_service._get_embedding = mock_get_embedding
                new_service.load_index(temp_filename)
                
                # Doğrulama
                assert len(new_service.index.embeddings) == 10
                
                # Yeni servisle arama
                new_results = new_service.search(text="another query", top_k=3)
                assert len(new_results) == 3
                
            finally:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                    
        finally:
            # Orijinal metodu geri yükle
            search_service._get_embedding = original_method
    
    @pytest.mark.slow
    def test_export_integration(self):
        """Veri dışa aktarma entegrasyonu testi"""
        # Test verisi
        data = [
            {
                "id": f"doc_{i}", 
                "content": f"Test content {i}", 
                "metadata": {
                    "type": "text",
                    "created": f"2024-01-{i+1:02d}T10:00:00Z"
                }
            }
            for i in range(10)
        ]
        
        # ExportManager oluştur
        export_manager = ExportManager()
        
        # Her format için test
        for format_extension in ["json", "csv", "txt"]:
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(suffix=f".{format_extension}", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Dışa aktar
                export_manager.export(data, temp_filename)
                
                # Dosya varlığını ve boyutunu kontrol et
                assert os.path.exists(temp_filename)
                assert os.path.getsize(temp_filename) > 0
                
            finally:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
    
    @pytest.mark.slow
    def test_complete_pipeline(self):
        """Tam model pipeline'ı entegrasyon testi"""
        # Küçük model ve tüm bileşenleri oluştur
        model_components = self.setup_tiny_model()
        config = model_components['config']
        
        # Sınıflandırma başlığı
        num_classes = 3
        classification_head = ClassificationHead(
            config.fusion_config.fused_embed_dim,
            hidden_dim=32,
            num_classes=num_classes
        )
        
        # Test verileri
        batch_size = 4
        text_input = torch.randint(0, config.text_config.vocab_size, (batch_size, 16))
        image_input = torch.rand(batch_size, 3, 32, 32)
        
        # 1. Gömme aşaması
        text_features = model_components['text_embedding'](text_input)
        image_features = model_components['image_embedding'](image_input)
        
        # 2. Transformer işleme
        for block in model_components['transformer_blocks']:
            text_features = block(text_features)
            image_features = block(image_features)
        
        # 3. Füzyon
        text_pooled = torch.mean(text_features, dim=1)
        image_pooled = torch.mean(image_features, dim=1)
        fused_features = model_components['fusion'](text_pooled, image_pooled)
        
        # 4. İki farklı görev için çıktı oluşturma
        
        # 4a. Arama gömme
        search_embeddings = model_components['search_projection'](fused_features)
        
        # 4b. Sınıflandırma
        logits = classification_head(fused_features)
        
        # Çıktı kontrolü
        assert search_embeddings.shape == (batch_size, config.search_config.search_embed_dim)
        assert torch.allclose(torch.norm(search_embeddings, p=2, dim=1), torch.ones(batch_size), atol=1e-6)
        assert logits.shape == (batch_size, num_classes)
        
        # 5. Arama servisi entegrasyonu
        search_service = SearchService(config.search_config.search_embed_dim)
        
        # SearchService._get_embedding metodunu geçici olarak override et
        original_method = search_service._get_embedding
        search_service._get_embedding = lambda text=None, image=None: search_embeddings[0].detach()
        
        try:
            # İçerik ekle ve ara
            for i in range(5):
                search_service.add_content(text=f"Content {i}", metadata={"id": f"item_{i}"})
            
            results = search_service.search(text="query")
            assert len(results) > 0
            
            # 6. Dışa aktarma entegrasyonu
            export_data = [
                {
                    "id": result["metadata"]["id"],
                    "content": f"Content for {result['metadata']['id']}",
                    "score": result["score"],
                    "metadata": result["metadata"]
                }
                for result in results
            ]
            
            export_manager = ExportManager()
            
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                temp_filename = temp_file.name
            
            try:
                # JSON olarak dışa aktar
                export_manager.export(export_data, temp_filename)
                assert os.path.exists(temp_filename)
                assert os.path.getsize(temp_filename) > 0
            finally:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                    
        finally:
            # Orijinal metodu geri yükle
            search_service._get_embedding = original_method 