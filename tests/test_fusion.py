"""
Fusion modülü için test dosyası.
"""

import unittest
import torch
import torch.nn as nn

from m3tm.fusion import (
    FusionConfig,
    ConcatenationFusionConfig,
    WeightedSumFusionConfig,
    GatedFusionConfig,
    FusionType,
    BaseFusion,
    ConcatenationFusion,
    WeightedSumFusion,
    GatedFusion,
    FusionFactory
)


class TestFusionConfig(unittest.TestCase):
    """Fusion konfigürasyon sınıfları için testler."""
    
    def test_fusion_config_validation(self):
        """Temel FusionConfig doğrulamasını test eder."""
        # Geçerli yapılandırma
        config = FusionConfig(
            fusion_type=FusionType.CONCATENATION,
            text_dim=128,
            image_dim=192,
            output_dim=256
        )
        self.assertEqual(config.fusion_type, FusionType.CONCATENATION)
        self.assertEqual(config.text_dim, 128)
        self.assertEqual(config.image_dim, 192)
        self.assertEqual(config.output_dim, 256)
        
        # String füzyon tipi ile test
        config = FusionConfig(
            fusion_type="weighted_sum",
            text_dim=128,
            image_dim=192
        )
        self.assertEqual(config.fusion_type, FusionType.WEIGHTED_SUM)
        
        # Geçersiz füzyon tipi
        with self.assertRaises(ValueError):
            FusionConfig(
                fusion_type="invalid_type",
                text_dim=128,
                image_dim=192
            )
        
        # Geçersiz boyutlar
        with self.assertRaises(ValueError):
            FusionConfig(
                fusion_type=FusionType.CONCATENATION,
                text_dim=-1,
                image_dim=192
            )
    
    def test_concatenation_fusion_config(self):
        """ConcatenationFusionConfig sınıfını test eder."""
        config = ConcatenationFusionConfig(
            text_dim=128,
            image_dim=192,
            use_projection=True
        )
        self.assertEqual(config.fusion_type, FusionType.CONCATENATION)
        self.assertTrue(config.use_projection)
    
    def test_weighted_sum_fusion_config(self):
        """WeightedSumFusionConfig sınıfını test eder."""
        config = WeightedSumFusionConfig(
            text_dim=128,
            image_dim=128,
            learnable_weights=True,
            initial_text_weight=0.6,
            initial_image_weight=0.4
        )
        self.assertEqual(config.fusion_type, FusionType.WEIGHTED_SUM)
        self.assertTrue(config.learnable_weights)
        self.assertEqual(config.initial_text_weight, 0.6)
        self.assertEqual(config.initial_image_weight, 0.4)
    
    def test_gated_fusion_config(self):
        """GatedFusionConfig sınıfını test eder."""
        config = GatedFusionConfig(
            text_dim=128,
            image_dim=128,
            gate_activation="sigmoid",
            hidden_dim=64,
            use_residual=True
        )
        self.assertEqual(config.fusion_type, FusionType.GATED)
        self.assertEqual(config.gate_activation, "sigmoid")
        self.assertEqual(config.hidden_dim, 64)
        self.assertTrue(config.use_residual)
        
        # Geçersiz aktivasyon fonksiyonu
        with self.assertRaises(ValueError):
            GatedFusionConfig(
                text_dim=128,
                image_dim=128,
                gate_activation="invalid_activation"
            )


class TestBaseFusion(unittest.TestCase):
    """BaseFusion sınıfı için testler."""
    
    def setUp(self):
        """Test için ortak verileri hazırla."""
        # Test tensörleri oluştur
        self.batch_size = 2
        self.seq_len = 10
        self.num_patches = 8
        self.text_dim = 64
        self.image_dim = 96
        
        # Rastgele text ve image embeddings oluştur
        self.text_embeddings = torch.randn(
            self.batch_size, self.seq_len, self.text_dim
        )
        self.image_embeddings = torch.randn(
            self.batch_size, self.num_patches, self.image_dim
        )
        
        # Maskeleri oluştur
        self.text_mask = torch.ones(self.batch_size, self.seq_len)
        self.text_mask[0, -2:] = 0  # İlk batch'in son iki token'ı maskelenmiş
        
        self.image_mask = torch.ones(self.batch_size, self.num_patches)
        self.image_mask[1, -3:] = 0  # İkinci batch'in son üç yamasi maskelenmiş


class TestConcatenationFusion(TestBaseFusion):
    """ConcatenationFusion sınıfı için testler."""
    
    def test_concatenation_fusion(self):
        """ConcatenationFusion'ın temel işlevini test eder."""
        # Yapılandırma ve modül oluştur
        config = ConcatenationFusionConfig(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            use_projection=True,
            output_dim=128
        )
        fusion_module = ConcatenationFusion(config)
        
        # İleri geçiş
        output_dict = fusion_module(
            text_embeddings=self.text_embeddings,
            image_embeddings=self.image_embeddings,
            text_mask=self.text_mask,
            image_mask=self.image_mask
        )
        
        # Çıktıyı kontrol et
        self.assertIn("fused_embeddings", output_dict)
        self.assertIn("fusion_type", output_dict)
        
        fused = output_dict["fused_embeddings"]
        self.assertEqual(fused.shape, (self.batch_size, 128))
        self.assertEqual(output_dict["fusion_type"], "concatenation")
        
        # Tensor olarak çıktı
        fused_tensor = fusion_module(
            text_embeddings=self.text_embeddings,
            image_embeddings=self.image_embeddings,
            text_mask=self.text_mask,
            image_mask=self.image_mask,
            return_dict=False
        )
        self.assertEqual(fused_tensor.shape, (self.batch_size, 128))
    
    def test_concatenation_single_modality(self):
        """ConcatenationFusion'ın tek modalite durumunu test eder."""
        # Yapılandırma ve modül oluştur
        config = ConcatenationFusionConfig(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            use_projection=True,
            output_dim=128
        )
        fusion_module = ConcatenationFusion(config)
        
        # Sadece metin embeddings ile test
        output_dict = fusion_module(
            text_embeddings=self.text_embeddings,
            text_mask=self.text_mask
        )
        self.assertEqual(output_dict["fused_embeddings"].shape, (self.batch_size, 128))
        
        # Sadece image embeddings ile test
        output_dict = fusion_module(
            image_embeddings=self.image_embeddings,
            image_mask=self.image_mask
        )
        self.assertEqual(output_dict["fused_embeddings"].shape, (self.batch_size, 128))


class TestWeightedSumFusion(TestBaseFusion):
    """WeightedSumFusion sınıfı için testler."""
    
    def test_weighted_sum_fusion(self):
        """WeightedSumFusion'ın temel işlevini test eder."""
        # Yapılandırma ve modül oluştur - Giriş boyutları farklı, çıkış boyutu belirtilmiş
        config = WeightedSumFusionConfig(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            output_dim=80,
            learnable_weights=True,
            initial_text_weight=0.6,
            initial_image_weight=0.4
        )
        fusion_module = WeightedSumFusion(config)
        
        # İleri geçiş
        output_dict = fusion_module(
            text_embeddings=self.text_embeddings,
            image_embeddings=self.image_embeddings,
            text_mask=self.text_mask,
            image_mask=self.image_mask
        )
        
        # Çıktıyı kontrol et
        fused = output_dict["fused_embeddings"]
        self.assertEqual(fused.shape, (self.batch_size, 80))
        self.assertEqual(output_dict["fusion_type"], "weighted_sum")
    
    def test_weighted_sum_same_dimensions(self):
        """WeightedSumFusion'ın aynı boyutlu gömmelerle kullanımını test eder."""
        # Aynı boyutlu gömme vektörleri oluştur
        text_embeddings = torch.randn(self.batch_size, self.seq_len, 128)
        image_embeddings = torch.randn(self.batch_size, self.num_patches, 128)
        
        # Yapılandırma ve modül oluştur - Giriş boyutları aynı, çıkış boyutu belirtilmemiş
        config = WeightedSumFusionConfig(
            text_dim=128,
            image_dim=128,
            learnable_weights=False,
            initial_text_weight=0.5,
            initial_image_weight=0.5
        )
        fusion_module = WeightedSumFusion(config)
        
        # İleri geçiş
        output_dict = fusion_module(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings
        )
        
        # Çıktıyı kontrol et
        fused = output_dict["fused_embeddings"]
        self.assertEqual(fused.shape, (self.batch_size, 128))


class TestGatedFusion(TestBaseFusion):
    """GatedFusion sınıfı için testler."""
    
    def test_gated_fusion(self):
        """GatedFusion'ın temel işlevini test eder."""
        # Yapılandırma ve modül oluştur
        config = GatedFusionConfig(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            output_dim=80,
            gate_activation="sigmoid",
            hidden_dim=40,
            use_residual=True
        )
        fusion_module = GatedFusion(config)
        
        # İleri geçiş
        output_dict = fusion_module(
            text_embeddings=self.text_embeddings,
            image_embeddings=self.image_embeddings,
            text_mask=self.text_mask,
            image_mask=self.image_mask
        )
        
        # Çıktıyı kontrol et
        fused = output_dict["fused_embeddings"]
        self.assertEqual(fused.shape, (self.batch_size, 80))
        self.assertEqual(output_dict["fusion_type"], "gated")
    
    def test_gated_fusion_different_activation(self):
        """GatedFusion'ın farklı aktivasyon fonksiyonları ile kullanımını test eder."""
        # Farklı aktivasyon fonksiyonları için yapılandırmalar oluştur
        activations = ["tanh", "relu", "leaky_relu", "hardswish"]
        
        for activation in activations:
            config = GatedFusionConfig(
                text_dim=self.text_dim,
                image_dim=self.image_dim,
                output_dim=80,
                gate_activation=activation,
                hidden_dim=40,
                use_residual=True
            )
            fusion_module = GatedFusion(config)
            
            # İleri geçiş
            output_dict = fusion_module(
                text_embeddings=self.text_embeddings,
                image_embeddings=self.image_embeddings
            )
            
            # Çıktıyı kontrol et
            fused = output_dict["fused_embeddings"]
            self.assertEqual(fused.shape, (self.batch_size, 80))


class TestFusionFactory(TestBaseFusion):
    """FusionFactory sınıfı için testler."""
    
    def test_get_fusion_class(self):
        """get_fusion_class metodunu test eder."""
        self.assertEqual(FusionFactory.get_fusion_class(FusionType.CONCATENATION), ConcatenationFusion)
        self.assertEqual(FusionFactory.get_fusion_class(FusionType.WEIGHTED_SUM), WeightedSumFusion)
        self.assertEqual(FusionFactory.get_fusion_class(FusionType.GATED), GatedFusion)
        
        with self.assertRaises(ValueError):
            # Geçersiz füzyon tipi nesnesi oluştur ve test et
            invalid_type = object()
            FusionFactory.get_fusion_class(invalid_type)
    
    def test_get_config_class(self):
        """get_config_class metodunu test eder."""
        self.assertEqual(FusionFactory.get_config_class(FusionType.CONCATENATION), ConcatenationFusionConfig)
        self.assertEqual(FusionFactory.get_config_class(FusionType.WEIGHTED_SUM), WeightedSumFusionConfig)
        self.assertEqual(FusionFactory.get_config_class(FusionType.GATED), GatedFusionConfig)
    
    def test_create_fusion(self):
        """create_fusion metodunu test eder."""
        # FusionConfig ile
        config = FusionConfig(
            fusion_type=FusionType.CONCATENATION,
            text_dim=self.text_dim,
            image_dim=self.image_dim
        )
        fusion_module = FusionFactory.create_fusion(config)
        self.assertIsInstance(fusion_module, ConcatenationFusion)
        
        # Özel konfigürasyonla
        config = WeightedSumFusionConfig(
            text_dim=self.text_dim,
            image_dim=self.image_dim
        )
        fusion_module = FusionFactory.create_fusion(config)
        self.assertIsInstance(fusion_module, WeightedSumFusion)
    
    def test_convenience_methods(self):
        """Kolaylık metodlarını (create_XXX_fusion) test eder."""
        # Concatenation fusion
        fusion = FusionFactory.create_concatenation_fusion(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            output_dim=128
        )
        self.assertIsInstance(fusion, ConcatenationFusion)
        
        # Weighted sum fusion
        fusion = FusionFactory.create_weighted_sum_fusion(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            learnable_weights=True
        )
        self.assertIsInstance(fusion, WeightedSumFusion)
        
        # Gated fusion
        fusion = FusionFactory.create_gated_fusion(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            gate_activation="tanh"
        )
        self.assertIsInstance(fusion, GatedFusion)
    
    def test_list_available_fusion_types(self):
        """list_available_fusion_types metodunu test eder."""
        fusion_types = FusionFactory.list_available_fusion_types()
        self.assertIsInstance(fusion_types, list)
        self.assertIn("concatenation", fusion_types)
        self.assertIn("weighted_sum", fusion_types)
        self.assertIn("gated", fusion_types)


if __name__ == "__main__":
    unittest.main() 