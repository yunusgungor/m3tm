"""
ProtoTransformerBlock birim testleri.
"""

import unittest
import torch

from m3tm.transformer import (
    ProtoTransformerConfig,
    AttentionConfig,
    FeedForwardConfig,
    AdapterConfig,
    ProtoTransformerBlock,
    ProtoTransformer
)


class TestProtoTransformerConfig(unittest.TestCase):
    """ProtoTransformerConfig test durumları."""
    
    def test_default_config(self):
        """Varsayılan yapılandırmanın geçerli olup olmadığını kontrol eder."""
        config = ProtoTransformerConfig()
        self.assertTrue(config.validate())
        self.assertEqual(len(config.validation_errors), 0)
    
    def test_custom_config(self):
        """Özel yapılandırmanın geçerli olup olmadığını kontrol eder."""
        config = ProtoTransformerConfig(
            hidden_size=128,
            intermediate_size=256,
            attention_config=AttentionConfig(
                mechanism_name="MobileAttention",
                head_dim=32,
                num_heads=4
            ),
            ffn_config=FeedForwardConfig(
                mechanism_name="MobileFFN",
                expansion_factor=2.0
            ),
            adapter_config=AdapterConfig(
                enabled=True,
                bottleneck_dim=16
            )
        )
        self.assertTrue(config.validate())
        self.assertEqual(len(config.validation_errors), 0)
    
    def test_invalid_config(self):
        """Geçersiz yapılandırmaların doğru şekilde algılanıp algılanmadığını kontrol eder."""
        # Negatif hidden_size
        config = ProtoTransformerConfig(hidden_size=-1)
        self.assertFalse(config.validate())
        
        # Geçersiz dropout
        config = ProtoTransformerConfig(dropout=2.0)
        self.assertFalse(config.validate())
        
        # Geçersiz dikkat mekanizması
        config = ProtoTransformerConfig(
            attention_config=AttentionConfig(mechanism_name="")
        )
        self.assertFalse(config.validate())


class TestProtoTransformerBlock(unittest.TestCase):
    """ProtoTransformerBlock test durumları."""
    
    def setUp(self):
        """Her test için gerekli nesneleri oluşturur."""
        self.config = ProtoTransformerConfig(
            hidden_size=64,
            intermediate_size=128,
            attention_config=AttentionConfig(
                mechanism_name="StandardSelfAttention",
                head_dim=16,
                num_heads=4,
                mechanism_params={"input_dim": 64}
            ),
            ffn_config=FeedForwardConfig(
                mechanism_name="StandardFFN",
                expansion_factor=2.0,
                mechanism_params={"hidden_size": 64}
            )
        )
        self.model = ProtoTransformerBlock(self.config)
        
        # Test girdileri
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 64
        self.x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.mask = torch.ones(self.batch_size, self.seq_len)
    
    def test_forward_pass(self):
        """İleri beslemeli geçişin çalışıp çalışmadığını kontrol eder."""
        output, metrics = self.model(self.x)
        
        # Çıktı şeklini kontrol et
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        
        # Metrikler var mı kontrol et
        self.assertIn("total_parameters", metrics)
        self.assertIn("attention_parameter_count", metrics)
        self.assertIn("ffn_parameter_count", metrics)
    
    def test_with_mask(self):
        """Maskelemenin çalışıp çalışmadığını kontrol eder."""
        # Maske oluştur (ilk token'lar görünür, diğerleri maskeli)
        mask = torch.zeros(self.batch_size, 1, self.seq_len)
        mask[:, :, 0] = 1  # Sadece ilk token görünür
        
        output_masked, _ = self.model(self.x, mask)
        output_normal, _ = self.model(self.x)
        
        # Maskeleme işe yararsa, çıktılar farklı olmalı
        self.assertFalse(torch.allclose(output_masked, output_normal, atol=1e-5))
    
    def test_parameter_count(self):
        """Parametre sayısının doğru hesaplanıp hesaplanmadığını kontrol eder."""
        param_count = self.model.count_parameters()
        
        # Manuel hesaplama
        manual_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertEqual(param_count, manual_count)
    
    def test_with_adapters(self):
        """Adapter'ların çalışıp çalışmadığını kontrol eder."""
        # Adapter'lı yapılandırma
        config_with_adapters = ProtoTransformerConfig(
            hidden_size=64,
            intermediate_size=128,
            attention_config=AttentionConfig(
                mechanism_name="StandardSelfAttention",
                head_dim=16,
                num_heads=4,
                mechanism_params={"input_dim": 64}
            ),
            ffn_config=FeedForwardConfig(
                mechanism_name="StandardFFN",
                expansion_factor=2.0,
                mechanism_params={"hidden_size": 64}
            ),
            adapter_config=AdapterConfig(
                enabled=True,
                bottleneck_dim=8
            )
        )
        model_with_adapters = ProtoTransformerBlock(config_with_adapters)
        
        output_normal, _ = self.model(self.x)
        output_with_adapters, _ = model_with_adapters(self.x)
        
        # Adapter'lar etkinse, çıktılar farklı olmalı
        self.assertFalse(torch.allclose(output_normal, output_with_adapters, atol=1e-5))
        
        # Adapter yuvaları var mı kontrol et
        self.assertTrue(len(model_with_adapters.adapter_slots) > 0)
        
        # Adapter kaldırma işlemini test et
        for position in model_with_adapters.adapter_slots:
            self.assertTrue(model_with_adapters.remove_adapter(position))


class TestProtoTransformer(unittest.TestCase):
    """ProtoTransformer test durumları."""
    
    def setUp(self):
        """Her test için gerekli nesneleri oluşturur."""
        self.config = ProtoTransformerConfig(
            hidden_size=64,
            intermediate_size=128,
            attention_config=AttentionConfig(
                mechanism_name="StandardSelfAttention",
                head_dim=16,
                num_heads=4,
                mechanism_params={"input_dim": 64}
            ),
            ffn_config=FeedForwardConfig(
                mechanism_name="StandardFFN",
                expansion_factor=2.0,
                mechanism_params={"hidden_size": 64}
            )
        )
        
        # Test girdileri
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 64
        self.x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    
    def test_single_layer(self):
        """Tek katmanlı modelin çalışıp çalışmadığını kontrol eder."""
        model = ProtoTransformer(self.config, num_layers=1)
        output, metrics = model(self.x)
        
        # Çıktı şeklini kontrol et
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        
        # Layer metriklerini kontrol et
        self.assertIn("layer_0", metrics)
    
    def test_multi_layer(self):
        """Çok katmanlı modelin çalışıp çalışmadığını kontrol eder."""
        model = ProtoTransformer(self.config, num_layers=3)
        output, metrics = model(self.x)
        
        # Çıktı şeklini kontrol et
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
        
        # Layer metriklerini kontrol et
        self.assertIn("layer_0", metrics)
        self.assertIn("layer_1", metrics)
        self.assertIn("layer_2", metrics)
        
        # Toplam parametre sayısını kontrol et
        self.assertIn("total_parameters", metrics)


if __name__ == "__main__":
    unittest.main() 