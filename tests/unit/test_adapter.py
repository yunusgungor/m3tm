"""
M³TM Adapters birim testleri.

Bu modül, M³TM adaptör uygulamalarının birim testlerini içerir.
"""

import unittest
import torch
import os
import tempfile
import shutil

from m3tm.adapters.adapter import AdapterConfig, AdapterType
from m3tm.adapters.adapter import Adapter, BottleneckAdapter, ParallelAdapter, create_adapter
from m3tm.adapters.adapter_manager import AdapterManager, AdapterRegistration
from m3tm.adapters.adapter_utils import (
    get_adapter_positions, 
    create_adapter_for_module, 
    find_adapter_modules, 
    count_adapter_parameters,
    get_trainable_adapter_parameters,
    freeze_model_except_adapters
)


class SimpleModule(torch.nn.Module):
    """Test için basit bir modül."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        self.adapter_slots = {}
        
    def register_adapter(self, adapter, position):
        """Adapter ekle."""
        self.adapter_slots[position] = adapter
        return True
        
    def remove_adapter(self, position):
        """Adapter kaldır."""
        if position in self.adapter_slots:
            del self.adapter_slots[position]
            return True
        return False
        
    def forward(self, x):
        """İleri geçiş."""
        if "input" in self.adapter_slots:
            x = self.adapter_slots["input"](x)
        
        x = self.linear(x)
        
        if "output" in self.adapter_slots:
            x = self.adapter_slots["output"](x)
        
        return x


class TestAdapterConfig(unittest.TestCase):
    """AdapterConfig sınıfı için birim testleri."""
    
    def test_default_values(self):
        """Varsayılan değerlerin doğru olduğunu test et."""
        config = AdapterConfig()
        self.assertEqual(config.adapter_type, AdapterType.BOTTLENECK)
        self.assertEqual(config.bottleneck_dim, 16)
        self.assertTrue(config.use_layer_norm)
        self.assertEqual(config.activation, "gelu")
        
    def test_validation(self):
        """Doğrulama işlevinin doğru çalıştığını test et."""
        # Geçerli yapılandırma
        config = AdapterConfig(bottleneck_dim=32, dropout=0.1)
        self.assertTrue(config.validate())
        
        # Geçersiz yapılandırma
        invalid_config = AdapterConfig(bottleneck_dim=-1, alpha=-0.5)
        self.assertFalse(invalid_config.validate())
        self.assertTrue(any("bottleneck_dim" in error for error in invalid_config.validation_errors))
        self.assertTrue(any("alpha" in error for error in invalid_config.validation_errors))


class TestAdapterImplementations(unittest.TestCase):
    """Adapter implementasyonları için birim testleri."""
    
    def setUp(self):
        """Test için hazırlık."""
        self.batch_size = 2
        self.seq_len = 8
        self.input_dim = 64
        self.bottleneck_dim = 16
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
    def test_bottleneck_adapter(self):
        """BottleneckAdapter'ın doğru çalıştığını test et."""
        config = AdapterConfig(adapter_type=AdapterType.BOTTLENECK, bottleneck_dim=self.bottleneck_dim)
        adapter = BottleneckAdapter(config, self.input_dim)
        
        # Parametre sayılarını kontrol et
        expected_params = 2 * (self.input_dim * self.bottleneck_dim + self.bottleneck_dim)
        # LayerNorm için ek parametreler
        if config.use_layer_norm:
            expected_params += 2 * self.bottleneck_dim
        self.assertEqual(adapter.count_parameters(), expected_params)
        
        # İleri geçişi kontrol et
        output = adapter(self.x)
        self.assertEqual(output.shape, self.x.shape)
        self.assertFalse(torch.allclose(output, self.x))  # Çıktı değişmeli
        
    def test_parallel_adapter(self):
        """ParallelAdapter'ın doğru çalıştığını test et."""
        config = AdapterConfig(adapter_type=AdapterType.PARALLEL, bottleneck_dim=self.bottleneck_dim)
        adapter = ParallelAdapter(config, self.input_dim)
        
        # İleri geçişi kontrol et
        output = adapter(self.x)
        self.assertEqual(output.shape, self.x.shape)
        self.assertFalse(torch.allclose(output, self.x))  # Çıktı değişmeli
        
    def test_create_adapter_factory(self):
        """Adapter fabrika metodunu test et."""
        # Bottleneck adapter oluştur
        config1 = AdapterConfig(adapter_type=AdapterType.BOTTLENECK)
        adapter1 = create_adapter(config1, self.input_dim)
        self.assertIsInstance(adapter1, BottleneckAdapter)
        
        # Parallel adapter oluştur
        config2 = AdapterConfig(adapter_type=AdapterType.PARALLEL)
        adapter2 = create_adapter(config2, self.input_dim)
        self.assertIsInstance(adapter2, ParallelAdapter)
        
        # Geçersiz tür
        with self.assertRaises(ValueError):
            invalid_config = AdapterConfig()
            invalid_config.adapter_type = "invalid_type"
            create_adapter(invalid_config, self.input_dim)


class TestAdapterManager(unittest.TestCase):
    """AdapterManager sınıfı için birim testleri."""
    
    def setUp(self):
        """Test için hazırlık."""
        self.module = SimpleModule(hidden_size=64)
        self.manager = AdapterManager(self.module)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Temizlik."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_register_and_remove_adapter(self):
        """Adapter kaydetme ve kaldırma işlevselliğini test et."""
        # Adapter kaydet
        adapter_id = self.manager.register_adapter(
            self.module,
            "test_adapter",
            "input",
            AdapterConfig(bottleneck_dim=8)
        )
        
        self.assertIsNotNone(adapter_id)
        self.assertIn(adapter_id, self.manager.registered_adapters)
        
        # Adapter'ı kaldır
        result = self.manager.remove_adapter(adapter_id)
        self.assertTrue(result)
        self.assertNotIn(adapter_id, self.manager.registered_adapters)
    
    def test_activate_deactivate_adapter(self):
        """Adapter etkinleştirme ve devre dışı bırakma işlevselliğini test et."""
        # Adapter kaydet
        adapter_id = self.manager.register_adapter(
            self.module,
            "test_adapter",
            "input",
            AdapterConfig(bottleneck_dim=8)
        )
        
        # Devre dışı bırak
        result = self.manager.deactivate_adapter(adapter_id)
        self.assertTrue(result)
        self.assertFalse(self.manager.registered_adapters[adapter_id].is_active)
        
        # Etkinleştir
        result = self.manager.activate_adapter(adapter_id)
        self.assertTrue(result)
        self.assertTrue(self.manager.registered_adapters[adapter_id].is_active)
    
    def test_save_load_adapters(self):
        """Adapter kaydetme ve yükleme işlevselliğini test et."""
        # Adapter kaydet
        adapter_id = self.manager.register_adapter(
            self.module,
            "test_adapter",
            "input",
            AdapterConfig(bottleneck_dim=8)
        )
        
        # Dosyaya kaydet
        result = self.manager.save_adapters(self.temp_dir)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "adapter_registry.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{adapter_id}.pt")))
        
        # Adapter'ı kaldır
        self.manager.remove_adapter(adapter_id)
        
        # Yükle
        loaded_adapters = self.manager.load_adapters(self.temp_dir)
        self.assertEqual(len(loaded_adapters), 1)
        self.assertIn(adapter_id, loaded_adapters)
        self.assertIn(adapter_id, self.manager.registered_adapters)


class TestAdapterUtils(unittest.TestCase):
    """Adapter yardımcı fonksiyonları için birim testleri."""
    
    def setUp(self):
        """Test için hazırlık."""
        self.module = SimpleModule(hidden_size=64)
        
    def test_adapter_positions(self):
        """get_adapter_positions fonksiyonunu test et."""
        positions = get_adapter_positions(self.module)
        self.assertIsInstance(positions, list)
        self.assertTrue(len(positions) > 0)
        
    def test_create_adapter_for_module(self):
        """create_adapter_for_module fonksiyonunu test et."""
        adapter = create_adapter_for_module(
            self.module,
            "input",
            AdapterConfig(bottleneck_dim=8)
        )
        
        self.assertIsNotNone(adapter)
        self.assertIsInstance(adapter, Adapter)
        
    def test_find_adapter_modules(self):
        """find_adapter_modules fonksiyonunu test et."""
        model = torch.nn.Sequential(
            SimpleModule(32),
            SimpleModule(64)
        )
        
        adapter_modules = find_adapter_modules(model)
        self.assertTrue(len(adapter_modules) >= 2)
        
    def test_freeze_model_except_adapters(self):
        """freeze_model_except_adapters fonksiyonunu test et."""
        # Adapter ekle
        manager = AdapterManager(self.module)
        manager.register_adapter(
            self.module,
            "test_adapter",
            "input",
            AdapterConfig(bottleneck_dim=8)
        )
        
        # Dondur
        freeze_model_except_adapters(self.module)
        
        # Adapter parametreleri eğitilebilir olmalı
        adapter_params = get_trainable_adapter_parameters(self.module)
        for param in adapter_params:
            self.assertTrue(param.requires_grad)
            
        # Diğer parametreler dondurulmuş olmalı
        adapter_param_ids = [id(p) for p in adapter_params]
        for param in self.module.parameters():
            if id(param) not in adapter_param_ids:
                self.assertFalse(param.requires_grad)


if __name__ == "__main__":
    unittest.main() 