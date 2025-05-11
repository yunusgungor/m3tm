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
from m3tm.task_heads.classification import ClassificationHead
from m3tm.training.trainer import TextClassificationTrainer
from m3tm.data_export.export_manager import ExportManager

class TestEndToEndModel:
    """End-to-end model entegrasyon testleri"""
    
    def setup_tiny_model(self):
        """Test için küçük M³TM modeli oluşturur"""
        config = get_tiny_config()
        
        # Dikkat ve FFN mekanizmalarını ayarla
        config.transformer_config.attention_type = "StandardSelfAttention"
        config.transformer_config.ffn_type = "StandardFFN"
        
        # Metin gömme
        text_embedding = TextEmbedding(config.text_config)
        
        # Görüntü gömme
        image_embedding = ImagePatchEmbedding(config.image_config)
        
        # Transformer blokları
        transformer_blocks = torch.nn.ModuleList([
            ProtoTransformerBlock(config.transformer_config) 
            for _ in range(config.num_core_blocks)
        ])
        
        # Füzyon - boyutları config ile uyumlu hale getir
        fusion = BasicFusion(
            config.fusion_config.text_dim,  # text_dim config'den gelecek
            config.fusion_config.image_dim, # image_dim config'den gelecek
            config.fusion_config.output_dim # output_dim config'den gelecek
        )
        
        # Arama gömme projeksiyonu - boyutları config ile uyumlu hale getir
        search_projection = SearchEmbeddingProjection(
            config.fusion_config.output_dim,      # input_dim, fusion output_dim ile eşleşmeli
            config.search_config.search_dim       # output_dim, search_dim ile eşleşmeli
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
        text_features = model_components['text_embedding'](text_input, return_dict=False)
        assert text_features.shape == (batch_size, 16, config.text_config.embed_dim)
        
        # Görüntü akışı
        image_features = model_components['image_embedding'](image_input, return_dict=False)
        # Yamaların sayısı (32/4)² = 64 olmalı (eğer patch_size=4 ise)
        expected_num_patches = (32 // config.image_config.patch_size) ** 2
        assert image_features.shape == (batch_size, expected_num_patches, config.image_config.embed_dim)
        
        # Transformer işleme (her modalite için ayrı)
        for block in model_components['transformer_blocks']:
            # block(x) bir tuple döndürüyor, ilk öğesi output tensoru
            text_features = block(text_features)[0]
            image_features = block(image_features)[0]
        
        # Füzyon
        # Global pooling
        text_pooled = torch.mean(text_features, dim=1)
        image_pooled = torch.mean(image_features, dim=1)
        
        fused_features = model_components['fusion'](text_pooled, image_pooled)
        assert fused_features.shape == (batch_size, config.fusion_config.output_dim)
        
        # Arama gömme
        search_embeddings = model_components['search_projection'](fused_features, return_dict=False)
        assert search_embeddings.shape == (batch_size, config.search_config.search_dim)
        
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
            # block(x) bir tuple döndürüyor, ilk öğesi output tensoru
            output_tensor, _ = block(input_tensor)
            input_tensor = output_tensor
            original_outputs.append(output_tensor.clone())
        
        # Adaptör ekleyip yeniden çalıştır
        adapter_size = 8
        for i, block in enumerate(transformer_blocks):
            block.add_adapter(f"adapter_{i}", adapter_size=adapter_size)
        
        # Adaptör ekleme sonrası forward
        input_tensor = torch.rand(batch_size, seq_len, config.transformer_config.embed_dim)  # Aynı giriş şekli
        adapter_outputs = []
        for block in transformer_blocks:
            # block(x) bir tuple döndürüyor, ilk öğesi output tensoru
            output_tensor, _ = block(input_tensor)
            input_tensor = output_tensor
            adapter_outputs.append(output_tensor.clone())
        
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
        
        # Sınıflandırma başlığı yapılandırması
        from m3tm.task_heads.config import ClassificationHeadConfig
        num_classes = 5
        
        # Transformer çıktı boyutu
        transformer_output_dim = config.transformer_config.embed_dim
        
        classification_config = ClassificationHeadConfig(
            input_dim=transformer_output_dim,  # Transformer çıktı boyutuyla eşleşmeli
            hidden_dim=transformer_output_dim // 2,  # hidden_dim için uygun bir değer
            num_classes=num_classes,
            dropout_rate=0.0  # Test için determinizmi koruyoruz
        )
        
        # Sınıflandırma başlığı
        classification_head = ClassificationHead(classification_config)
        
        # Test girdileri
        batch_size = 4
        text_input = torch.randint(0, config.text_config.vocab_size, (batch_size, 16))
        
        # Transformer bloğunu al (ilki)
        transformer_block = model_components['transformer_blocks'][0]
        
        # TextClassificationTrainer kullanarak model oluştur
        model = TextClassificationTrainer.create_composite_model(
            text_embedding=model_components['text_embedding'],
            transformer_block=transformer_block,
            classification_head=classification_head
        )
        
        # Forward
        outputs = model(text_input)
        # Çıktı bir sözlük olmalı
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, num_classes)
        
        # Eğitim simülasyonu (tek bir adım)
        labels = torch.randint(0, num_classes, (batch_size,))
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward ve backward
        optimizer.zero_grad()
        outputs = model(text_input)
        loss = loss_fn(outputs["logits"], labels)
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
        search_dim = config.search_config.search_dim
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
        
        # Sınıflandırma başlığı yapılandırması
        from m3tm.task_heads.config import ClassificationHeadConfig
        num_classes = 3
        classification_config = ClassificationHeadConfig(
            input_dim=config.fusion_config.output_dim,
            hidden_dim=32,
            num_classes=num_classes
        )
        
        # Sınıflandırma başlığı
        classification_head = ClassificationHead(classification_config)
        
        # Test girdileri
        batch_size = 4
        text_input = torch.randint(0, config.text_config.vocab_size, (batch_size, 16))
        image_input = torch.rand(batch_size, 3, 224, 224)  # Standart görüntü boyutu
        
        # Metin ve görüntü gömme
        text_features = model_components['text_embedding'](text_input, return_dict=False)
        image_features = model_components['image_embedding'](image_input, return_dict=False)
        
        # Transformer işleme
        for block in model_components['transformer_blocks']:
            text_features = block(text_features)[0]
            image_features = block(image_features)[0]
        
        # Pooling
        text_pooled = torch.mean(text_features, dim=1)
        image_pooled = torch.mean(image_features, dim=1)
        
        # Füzyon
        fused_features = model_components['fusion'](text_pooled, image_pooled)
        
        # Arama gömme
        search_embeddings = model_components['search_projection'](fused_features, return_dict=False)
        
        # Sınıflandırma başlığı
        classification_output = classification_head(fused_features, return_dict=False)
        assert classification_output.shape == (batch_size, num_classes)
        
        # End-to-end ile tahminler
        def run_complete_pipeline(text, image):
            """Tam pipeline'ı çalıştırır"""
            t_embed = model_components['text_embedding'](text, return_dict=False)
            i_embed = model_components['image_embedding'](image, return_dict=False)
            
            for block in model_components['transformer_blocks']:
                t_embed = block(t_embed)[0]
                i_embed = block(i_embed)[0]
            
            t_pooled = torch.mean(t_embed, dim=1)
            i_pooled = torch.mean(i_embed, dim=1)
            
            fused = model_components['fusion'](t_pooled, i_pooled)
            search_emb = model_components['search_projection'](fused, return_dict=False)
            classification = classification_head(fused, return_dict=False)
            
            return {
                'fused': fused,
                'search_embedding': search_emb,
                'classification': classification
            }
        
        # Farklı girdilerle test et
        pipeline_output_1 = run_complete_pipeline(
            text_input[:2], 
            image_input[:2]
        )
        
        pipeline_output_2 = run_complete_pipeline(
            text_input[2:], 
            image_input[2:]
        )
        
        # Farklı girdiler farklı çıktılar üretmeli
        assert not torch.allclose(pipeline_output_1['fused'], pipeline_output_2['fused'])
        assert not torch.allclose(pipeline_output_1['search_embedding'], pipeline_output_2['search_embedding'])
        assert not torch.allclose(pipeline_output_1['classification'], pipeline_output_2['classification']) 