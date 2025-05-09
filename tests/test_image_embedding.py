"""
ImagePatchEmbedding bileşeni için birim testler.
"""

import unittest
import torch
import pytest
import numpy as np

from m3tm.embedding.image_embedding import ImagePatchEmbeddingConfig, ImagePatchEmbedding, ImagePatchEmbeddingFactory
from m3tm.image.patch import patchify_image, unpatchify_image, get_2d_sincos_pos_embed


class TestImagePatchEmbeddingConfig(unittest.TestCase):
    """ImagePatchEmbeddingConfig sınıfı için testler."""
    
    def test_default_config(self):
        """Varsayılan yapılandırmanın doğru değerlere sahip olduğunu kontrol eder."""
        config = ImagePatchEmbeddingConfig()
        
        self.assertEqual(config.embed_dim, 384)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.image_size, (224, 224))
        self.assertEqual(config.channels, 3)
        self.assertTrue(config.use_position_embedding)
        self.assertEqual(config.dropout_rate, 0.1)
        self.assertEqual(config.layer_norm_eps, 1e-12)
        self.assertFalse(config.use_embedding_projection)
        self.assertIsNone(config.projection_dim)
    
    def test_custom_config(self):
        """Özel yapılandırmanın doğru değerlere sahip olduğunu kontrol eder."""
        config = ImagePatchEmbeddingConfig(
            embed_dim=256,
            patch_size=8,
            image_size=(160, 160),
            channels=1,
            use_position_embedding=False,
            dropout_rate=0.2,
            use_embedding_projection=True,
            projection_dim=128
        )
        
        self.assertEqual(config.embed_dim, 256)
        self.assertEqual(config.patch_size, 8)
        self.assertEqual(config.image_size, (160, 160))
        self.assertEqual(config.channels, 1)
        self.assertFalse(config.use_position_embedding)
        self.assertEqual(config.dropout_rate, 0.2)
        self.assertTrue(config.use_embedding_projection)
        self.assertEqual(config.projection_dim, 128)
    
    def test_validation(self):
        """Yapılandırma doğrulamasının beklendiği gibi çalıştığını kontrol eder."""
        # Geçerli yapılandırma
        config = ImagePatchEmbeddingConfig()
        config.validate()  # Hata fırlatmamalı
        
        # Geçersiz embed_dim
        config = ImagePatchEmbeddingConfig(embed_dim=0)
        with self.assertRaises(Exception):
            config.validate()
        
        # Geçersiz patch_size
        config = ImagePatchEmbeddingConfig(patch_size=0)
        with self.assertRaises(Exception):
            config.validate()
        
        # Görüntü boyutu yama boyutuna bölünmüyor
        config = ImagePatchEmbeddingConfig(image_size=(100, 100), patch_size=16)
        with self.assertRaises(Exception):
            config.validate()
        
        # Geçersiz kanal sayısı
        config = ImagePatchEmbeddingConfig(channels=0)
        with self.assertRaises(Exception):
            config.validate()
        
        # Geçersiz dropout oranı
        config = ImagePatchEmbeddingConfig(dropout_rate=1.5)
        with self.assertRaises(Exception):
            config.validate()
        
        # Projeksiyon kullanılıyor ama boyutu belirtilmemiş
        config = ImagePatchEmbeddingConfig(use_embedding_projection=True, projection_dim=None)
        with self.assertRaises(Exception):
            config.validate()


class TestImagePatchEmbedding(unittest.TestCase):
    """ImagePatchEmbedding sınıfı için testler."""
    
    def setUp(self):
        """Test için gerekli nesneleri hazırlar."""
        self.config = ImagePatchEmbeddingConfig(
            embed_dim=192,
            patch_size=16,
            image_size=(224, 224),
            channels=3,
            use_position_embedding=True,
            dropout_rate=0.1,
            use_embedding_projection=False
        )
        
        self.model = ImagePatchEmbedding(self.config)
        self.batch_size = 4
    
    def test_model_init(self):
        """Model başlatmanın beklendiği gibi çalıştığını kontrol eder."""
        model = self.model
        
        # Temel parametreleri kontrol et
        self.assertEqual(model.config.embed_dim, 192)
        self.assertEqual(model.patch_size, 16)
        self.assertEqual(model.num_patches, (224 // 16) * (224 // 16))  # 196
        
        # Alt modüllerin varlığını kontrol et
        self.assertIsNotNone(model.patch_embedding)
        self.assertIsNotNone(model.position_embedding)
        self.assertIsNone(model.projection)  # Kullanılmıyor
        self.assertIsNotNone(model.layer_norm)
        self.assertIsNotNone(model.dropout)
    
    def test_forward_pass(self):
        """İleri geçişin beklendiği gibi çalıştığını kontrol eder."""
        # Test girişi oluştur
        x = torch.randn(self.batch_size, 3, 224, 224)
        
        # İleri geçiş
        output_dict = self.model(x)
        
        # Çıktı şeklini kontrol et
        self.assertIn("embeddings", output_dict)
        self.assertIn("attention_mask", output_dict)
        
        embeddings = output_dict["embeddings"]
        attention_mask = output_dict["attention_mask"]
        
        self.assertEqual(embeddings.shape, (self.batch_size, self.model.num_patches, self.config.embed_dim))
        self.assertEqual(attention_mask.shape, (self.batch_size, self.model.num_patches))
    
    def test_forward_with_mask(self):
        """Dikkat maskesiyle ileri geçişin beklendiği gibi çalıştığını kontrol eder."""
        # Test girişi oluştur
        x = torch.randn(self.batch_size, 3, 224, 224)
        
        # Dikkat maskesi oluştur (bazı yamaları maskelemek için)
        mask = torch.ones(self.batch_size, self.model.num_patches)
        mask[:, 50:100] = 0  # Bazı yamaları maskele
        
        # İleri geçiş
        output_dict = self.model(x, attention_mask=mask)
        
        # Çıktı şeklini kontrol et
        embeddings = output_dict["embeddings"]
        attention_mask = output_dict["attention_mask"]
        
        # Maskelenen bölgelerin 0 olduğunu kontrol et
        self.assertTrue(torch.all(embeddings[:, 50:100, :] == 0))
        self.assertEqual(attention_mask.sum().item(), (self.batch_size * self.model.num_patches) - (self.batch_size * 50))
    
    def test_non_dict_output(self):
        """return_dict=False durumunda çıktının beklendiği gibi olduğunu kontrol eder."""
        # Test girişi oluştur
        x = torch.randn(self.batch_size, 3, 224, 224)
        
        # İleri geçiş
        embeddings = self.model(x, return_dict=False)
        
        # Çıktı şeklini kontrol et
        self.assertTrue(isinstance(embeddings, torch.Tensor))
        self.assertEqual(embeddings.shape, (self.batch_size, self.model.num_patches, self.config.embed_dim))
    
    def test_with_projection(self):
        """Projeksiyon katmanının beklendiği gibi çalıştığını kontrol eder."""
        # Projeksiyon ile yapılandırma oluştur
        config = ImagePatchEmbeddingConfig(
            embed_dim=192,
            patch_size=16,
            image_size=(224, 224),
            channels=3,
            use_position_embedding=True,
            dropout_rate=0.1,
            use_embedding_projection=True,
            projection_dim=128
        )
        
        model = ImagePatchEmbedding(config)
        
        # Test girişi oluştur
        x = torch.randn(self.batch_size, 3, 224, 224)
        
        # İleri geçiş
        output_dict = model(x)
        embeddings = output_dict["embeddings"]
        
        # Çıktı şeklini kontrol et
        self.assertEqual(embeddings.shape, (self.batch_size, model.num_patches, config.projection_dim))


class TestImagePatchEmbeddingFactory(unittest.TestCase):
    """ImagePatchEmbeddingFactory sınıfı için testler."""
    
    def test_create_image_patch_embedding(self):
        """create_image_patch_embedding metodunun beklendiği gibi çalıştığını kontrol eder."""
        config = ImagePatchEmbeddingConfig()
        model = ImagePatchEmbeddingFactory.create_image_patch_embedding(config)
        
        self.assertIsInstance(model, ImagePatchEmbedding)
        self.assertEqual(model.config, config)
    
    def test_create_compressed_embedding(self):
        """create_compressed_embedding metodunun beklendiği gibi çalıştığını kontrol eder."""
        config = ImagePatchEmbeddingConfig(embed_dim=192)
        compression_ratio = 0.5
        
        model = ImagePatchEmbeddingFactory.create_compressed_embedding(config, compression_ratio)
        
        self.assertIsInstance(model, ImagePatchEmbedding)
        self.assertEqual(model.config.embed_dim, 192)
        self.assertTrue(model.config.use_embedding_projection)
        self.assertEqual(model.config.projection_dim, int(192 * compression_ratio))
        
        # Test girişi oluştur
        x = torch.randn(2, 3, 224, 224)
        
        # İleri geçiş
        output_dict = model(x)
        embeddings = output_dict["embeddings"]
        
        # Çıktı şeklini kontrol et
        self.assertEqual(embeddings.shape, (2, model.num_patches, model.config.projection_dim))
    
    def test_invalid_compression_ratio(self):
        """Geçersiz sıkıştırma oranıyla create_compressed_embedding çağrıldığında hata fırlatıldığını kontrol eder."""
        config = ImagePatchEmbeddingConfig()
        
        with self.assertRaises(ValueError):
            ImagePatchEmbeddingFactory.create_compressed_embedding(config, compression_ratio=0)
        
        with self.assertRaises(ValueError):
            ImagePatchEmbeddingFactory.create_compressed_embedding(config, compression_ratio=1.0)


class TestPatchImageFunctions(unittest.TestCase):
    """Görüntü yama fonksiyonları için testler."""
    
    def test_patchify_image_tensor(self):
        """patchify_image fonksiyonunun tensor girişlerle beklendiği gibi çalıştığını kontrol eder."""
        # Test görüntüsü (tek görüntü)
        image = torch.randn(3, 32, 32)  # C, H, W
        patch_size = 8
        
        patches = patchify_image(image, patch_size)
        
        # Şekil kontrolü
        self.assertEqual(patches.shape, (16, 8, 8, 3))  # num_patches, P, P, C
        
        # Test görüntüsü (batch)
        batch_image = torch.randn(4, 3, 32, 32)  # B, C, H, W
        
        batch_patches = patchify_image(batch_image, patch_size)
        
        # Şekil kontrolü
        self.assertEqual(batch_patches.shape, (4, 16, 8, 8, 3))  # B, num_patches, P, P, C
    
    def test_unpatchify_image_tensor(self):
        """unpatchify_image fonksiyonunun tensor girişlerle beklendiği gibi çalıştığını kontrol eder."""
        # Test yamaları (tek görüntü)
        patches = torch.randn(16, 8, 8, 3)  # num_patches, P, P, C
        image_size = (32, 32)
        patch_size = 8
        
        image = unpatchify_image(patches, image_size, patch_size)
        
        # Şekil kontrolü
        self.assertEqual(image.shape, (3, 32, 32))  # C, H, W
        
        # Test yamaları (batch)
        batch_patches = torch.randn(4, 16, 8, 8, 3)  # B, num_patches, P, P, C
        
        batch_image = unpatchify_image(batch_patches, image_size, patch_size)
        
        # Şekil kontrolü
        self.assertEqual(batch_image.shape, (4, 3, 32, 32))  # B, C, H, W
    
    def test_patchify_unpatchify_roundtrip(self):
        """patchify_image ve unpatchify_image fonksiyonlarının birlikte çalıştığını kontrol eder."""
        # Test görüntüsü
        image = torch.randn(3, 32, 32)  # C, H, W
        patch_size = 8
        
        # Gidiş-dönüş dönüşümü
        patches = patchify_image(image, patch_size)
        reconstructed = unpatchify_image(patches, (32, 32), patch_size)
        
        # Orijinal görüntü ile yeniden oluşturulmuş görüntünün aynı olduğunu kontrol et
        self.assertTrue(torch.allclose(image, reconstructed, rtol=1e-5, atol=1e-5))
    
    def test_get_2d_sincos_pos_embed(self):
        """get_2d_sincos_pos_embed fonksiyonunun beklendiği gibi çalıştığını kontrol eder."""
        embed_dim = 64
        grid_size = (7, 7)
        
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
        
        # Şekil kontrolü
        self.assertEqual(pos_embed.shape, (grid_size[0] * grid_size[1], embed_dim))
        
        # Değer sınırları kontrolü (sin/cos değerleri -1 ve 1 arasında olmalı)
        self.assertTrue(np.all(pos_embed >= -1.0))
        self.assertTrue(np.all(pos_embed <= 1.0))


def test_image_patch_embedding_config():
    """ImagePatchEmbeddingConfig doğrulama testi"""
    # Varsayılan yapılandırma
    config = ImagePatchEmbeddingConfig()
    assert config.image_size == 224
    assert config.patch_size == 16
    assert config.embed_dim == 192
    assert config.in_channels == 3
    assert config.num_patches == (224 // 16) ** 2 == 196
    
    # Özel yapılandırma
    config = ImagePatchEmbeddingConfig(
        image_size=160,
        patch_size=8,
        embed_dim=128,
        in_channels=1,
        use_position_embedding=True,
        position_embedding_type="learned"
    )
    assert config.image_size == 160
    assert config.patch_size == 8
    assert config.embed_dim == 128
    assert config.in_channels == 1
    assert config.use_position_embedding == True
    assert config.position_embedding_type == "learned"
    assert config.num_patches == (160 // 8) ** 2 == 400
    
    # Geçersiz yapılandırma testleri
    with pytest.raises(ValueError):
        # Görüntü boyutu yama boyutunun tam katı değil
        ImagePatchEmbeddingConfig(image_size=100, patch_size=16)
    
    with pytest.raises(ValueError):
        # Negatif değer
        ImagePatchEmbeddingConfig(embed_dim=-128)
    
    with pytest.raises(ValueError):
        # Yanlış konum gömmesi türü
        ImagePatchEmbeddingConfig(position_embedding_type="invalid")
    
    with pytest.raises(ValueError):
        # Projeksiyon boyutu eksik
        ImagePatchEmbeddingConfig(use_embedding_projection=True, projection_dim=None)


def test_image_patch_embedding_forward():
    """ImagePatchEmbedding modülünün ileri geçişi testi"""
    batch_size = 4
    image_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 192
    
    config = ImagePatchEmbeddingConfig(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels
    )
    
    # Modeli oluştur
    model = ImagePatchEmbedding(config)
    
    # Test girdisi oluştur
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    
    # İleri geçiş yap
    output_dict = model(x)
    output = output_dict["embeddings"]
    
    # Çıktı boyutlarını doğrula
    num_patches = (image_size // patch_size) ** 2
    assert output.shape == (batch_size, num_patches, embed_dim)
    
    # Doğrudan tensör dönüşünü doğrula
    output_tensor = model(x, return_dict=False)
    assert torch.allclose(output, output_tensor)


def test_position_embedding_types():
    """Farklı konum gömmesi türleri testi"""
    batch_size = 2
    image_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 192
    
    # Öğrenilen konum gömmesi
    config_learned = ImagePatchEmbeddingConfig(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        use_position_embedding=True,
        position_embedding_type="learned"
    )
    model_learned = ImagePatchEmbedding(config_learned)
    
    # Sinüzoidal konum gömmesi
    config_sincos = ImagePatchEmbeddingConfig(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        use_position_embedding=True,
        position_embedding_type="sincos"
    )
    model_sincos = ImagePatchEmbedding(config_sincos)
    
    # Konum gömmesi yok
    config_none = ImagePatchEmbeddingConfig(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        use_position_embedding=False
    )
    model_none = ImagePatchEmbedding(config_none)
    
    # Test girdisi
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    
    # Tüm modeller için ileri geçiş
    output_learned = model_learned(x)["embeddings"]
    output_sincos = model_sincos(x)["embeddings"]
    output_none = model_none(x)["embeddings"]
    
    # Tüm çıktıların doğru boyutlarda olduğunu doğrula
    num_patches = (image_size // patch_size) ** 2
    assert output_learned.shape == (batch_size, num_patches, embed_dim)
    assert output_sincos.shape == (batch_size, num_patches, embed_dim)
    assert output_none.shape == (batch_size, num_patches, embed_dim)
    
    # Çıktıların farklı olduğunu doğrula
    assert not torch.allclose(output_learned, output_sincos, rtol=1e-3)
    assert not torch.allclose(output_learned, output_none, rtol=1e-3)
    assert not torch.allclose(output_sincos, output_none, rtol=1e-3)


def test_embedding_projection():
    """Embedding projelendirme testi"""
    batch_size = 2
    image_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 192
    projection_dim = 96
    
    # Projeksiyon ile
    config_proj = ImagePatchEmbeddingConfig(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        use_embedding_projection=True,
        projection_dim=projection_dim
    )
    model_proj = ImagePatchEmbedding(config_proj)
    
    # Projeksiyon olmadan
    config_no_proj = ImagePatchEmbeddingConfig(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        use_embedding_projection=False
    )
    model_no_proj = ImagePatchEmbedding(config_no_proj)
    
    # Test girdisi
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    
    # İleri geçiş
    output_proj = model_proj(x)["embeddings"]
    output_no_proj = model_no_proj(x)["embeddings"]
    
    # Çıktı boyutlarını doğrula
    num_patches = (image_size // patch_size) ** 2
    assert output_proj.shape == (batch_size, num_patches, projection_dim)
    assert output_no_proj.shape == (batch_size, num_patches, embed_dim)


def test_factory_methods():
    """ImagePatchEmbeddingFactory test"""
    config = ImagePatchEmbeddingConfig(
        image_size=224,
        patch_size=16,
        embed_dim=192,
        in_channels=3
    )
    
    # Normal embedding oluştur
    model_standard = ImagePatchEmbeddingFactory.create_image_patch_embedding(config)
    assert isinstance(model_standard, ImagePatchEmbedding)
    assert model_standard.config.embed_dim == 192
    
    # Verimli embedding oluştur
    model_efficient = ImagePatchEmbeddingFactory.create_efficient_embedding(
        config,
        efficiency_factor=0.5
    )
    assert isinstance(model_efficient, ImagePatchEmbedding)
    assert model_efficient.config.embed_dim == 96  # 192 * 0.5
    assert model_efficient.config.use_embedding_projection == True
    assert model_efficient.config.position_embedding_type == "sincos"


def test_patch_functions():
    """Görüntü yama fonksiyonlarının testi"""
    # Test girdisi
    batch_size = 2
    channels = 3
    image_size = 32
    patch_size = 8
    
    # Rastgele görüntü oluştur
    images = torch.randn(batch_size, channels, image_size, image_size)
    
    # Görüntüleri yamalara böl (flatten=True)
    patches_flat = patchify_image(images, patch_size, flatten=True)
    assert patches_flat.shape == (batch_size, (image_size // patch_size) ** 2, channels * patch_size * patch_size)
    
    # Görüntüleri yamalara böl (flatten=False)
    patches_unflat = patchify_image(images, patch_size, flatten=False)
    assert patches_unflat.shape == (batch_size, (image_size // patch_size) ** 2, channels, patch_size, patch_size)
    
    # Yamalardan görüntülere dönüştür
    recon_images = unpatchify_image(patches_unflat, patch_size)
    assert recon_images.shape == images.shape
    
    # Dönüşümün doğruluğunu kontrol et (sayısal hassasiyet nedeniyle tam eşitlik beklemiyoruz)
    assert torch.allclose(images, recon_images, rtol=1e-5, atol=1e-5)


def test_sincos_pos_embed():
    """Sinüzoidal konum kodlaması testi"""
    # 2D konum kodlaması
    embed_dim = 64
    grid_h = 5
    grid_w = 5
    
    pos_embed_2d = get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w)
    assert pos_embed_2d.shape == (grid_h * grid_w, embed_dim)
    
    # [CLS] token'lı
    pos_embed_2d_cls = get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=True)
    assert pos_embed_2d_cls.shape == (1 + grid_h * grid_w, embed_dim)
    assert torch.allclose(pos_embed_2d_cls[0], torch.zeros(embed_dim))  # [CLS] token 0 olmalı


if __name__ == "__main__":
    unittest.main() 