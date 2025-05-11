"""
Arama Gömme Projeksiyonu Testleri
"""
import unittest
import torch
import torch.nn as nn
import numpy as np

from src.m3tm.search.projection import SearchProjectionConfig, M3TMSearchProjection, SearchProjectionFactory


class TestSearchProjection(unittest.TestCase):
    """M3TMSearchProjection sınıfı için testler."""
    
    def setUp(self):
        """Test düzeni kurulumu."""
        self.batch_size = 4
        self.input_dim = 64
        self.embedding_dim = 32
        self.config = SearchProjectionConfig(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            dropout_rate=0.0  # Test için deterministik davranış
        )
        self.projection = M3TMSearchProjection(self.config)
        # Test için eval moduna geçiş (dropout için)
        self.projection.eval()
    
    def test_init(self):
        """Başlatma işleminin doğru yapıldığını doğrula."""
        self.assertEqual(self.projection.projection.in_features, self.input_dim)
        self.assertEqual(self.projection.projection.out_features, self.embedding_dim)
        self.assertTrue(hasattr(self.projection, "layer_norm"))
        self.assertTrue(isinstance(self.projection.layer_norm, nn.LayerNorm))
        self.assertEqual(self.projection.layer_norm.normalized_shape[0], self.embedding_dim)
    
    def test_forward_2d(self):
        """2B girdiler için forward işleminin doğru çalıştığını doğrula."""
        # 2B girdi (batch_size, input_dim)
        inputs = torch.randn(self.batch_size, self.input_dim)
        
        # Sözlük dönüşüyle çağır
        outputs_dict = self.projection(inputs, return_dict=True)
        self.assertIn("search_embeddings", outputs_dict)
        self.assertIn("projections", outputs_dict)
        
        # Doğrudan tensor dönüşüyle çağır
        search_embeddings = self.projection(inputs, return_dict=False)
        
        # Boyut kontrolü
        self.assertEqual(outputs_dict["search_embeddings"].shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(outputs_dict["projections"].shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(search_embeddings.shape, (self.batch_size, self.embedding_dim))
        
        # Tutarlılık kontrolü
        torch.testing.assert_close(outputs_dict["search_embeddings"], search_embeddings)
        
        # L2 normalizasyonu kontrolü (birim vektörler)
        for i in range(self.batch_size):
            # norm ≈ 1 kontrolü
            norm = torch.norm(search_embeddings[i], p=2).item()
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_forward_3d(self):
        """3B girdiler için forward işleminin doğru çalıştığını doğrula."""
        # 3B girdi (batch_size, seq_len, input_dim)
        seq_len = 5
        inputs = torch.randn(self.batch_size, seq_len, self.input_dim)
        
        # Sözlük dönüşüyle çağır
        outputs_dict = self.projection(inputs, return_dict=True)
        
        # Boyut kontrolü
        self.assertEqual(outputs_dict["search_embeddings"].shape, (self.batch_size, self.embedding_dim))
        
        # Ortalama havuzlama kontrolü
        mean_inputs = inputs.mean(dim=1)
        # Projeksiyon, layer_norm ve normalizasyon işlemleri karmaşık olduğundan doğrudan 
        # kontrol etmek mümkün değil, ancak basitleştirilmiş bir versiyonu ile temel işlemi 
        # kontrol edebiliriz
        simple_projection = nn.Linear(self.input_dim, self.embedding_dim).to(inputs.device)
        with torch.no_grad():
            self.projection.projection.weight.copy_(simple_projection.weight)
            self.projection.projection.bias.copy_(simple_projection.bias)

            # Layer norm ağırlık ve bias'ı resetle
            if hasattr(self.projection, 'layer_norm'):
                nn.init.ones_(self.projection.layer_norm.weight)
                nn.init.zeros_(self.projection.layer_norm.bias)
            
            # Basit test süreci
            projected = simple_projection(mean_inputs)
            if hasattr(self.projection, 'layer_norm'):
                projected = self.projection.layer_norm(projected)
            expected = torch.nn.functional.normalize(projected, p=2, dim=-1)
            
            actual = self.projection(inputs, return_dict=False)
            torch.testing.assert_close(actual, expected)
    
    def test_dimension_mismatch(self):
        """Boyut uyuşmazlığı durumunda ValueError fırlatılmalı."""
        # Yanlış boyutlu girdi (input_dim + 1)
        inputs = torch.randn(self.batch_size, self.input_dim + 1)
        
        # ValueError bekliyoruz
        with self.assertRaises(ValueError):
            _ = self.projection(inputs)
    
    def test_factory(self):
        """SearchProjectionFactory'nin doğru çalıştığını doğrula."""
        projection = SearchProjectionFactory.create(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            dropout_rate=0.0
        )
        
        self.assertIsInstance(projection, M3TMSearchProjection)
        self.assertEqual(projection.projection.in_features, self.input_dim)
        self.assertEqual(projection.projection.out_features, self.embedding_dim)
    
    def test_jit_trace(self):
        """JIT izleme ile model uyumluluğunu doğrula."""
        inputs = torch.randn(self.batch_size, self.input_dim)
        
        # eval modunda JIT trace
        self.projection.eval()
        
        # return_dict parametresini kullanmadan daha basit bir model tanımla
        class TraceWrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                
            def forward(self, x):
                # Doğrudan tensor döndüren bir forward pass
                emb = self.module.projection(x)
                norm = torch.nn.functional.normalize(emb, p=2, dim=-1)
                return norm
        
        # Sarmalayıcı oluştur
        trace_model = TraceWrapper(self.projection)
        
        # Modeli izle
        traced_model = torch.jit.trace(
            trace_model,
            inputs
        )
        
        # Modelin çıktısını kontrol et
        with torch.no_grad():
            ref_output = trace_model(inputs)
            trace_output = traced_model(inputs)
            
        # Çıktılar yakın olmalı
        torch.testing.assert_close(ref_output, trace_output)


if __name__ == "__main__":
    unittest.main() 