"""
M³TM modelinin sağlamlık (robustness) entegrasyon testleri

Bu modul, modelin çeşitli zorlu durumlarda sağlamlığını test eder:
- Gürültülü girdilerle başa çıkma
- Eksik modalitelerde çalışma
- Düşük kaliteli görüntülerle çalışma
- Farklı boyutlarda girdilere adaptasyon
- Farklı dil ve karakter setleriyle çalışma
- Cihaz kısıtlarını simüle etme (düşük RAM, düşük işlemci)
"""
import os
import pytest
import torch
import tempfile
import numpy as np
from PIL import Image, ImageFilter
import random
import string
from pathlib import Path

from m3tm.config.model_config import get_tiny_config
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.embedding.text_embedding import TextEmbedding
from m3tm.embedding.image_embedding import ImagePatchEmbedding
from m3tm.fusion.basic_fusion import BasicFusion
from m3tm.search.search_embedding import SearchEmbeddingProjection
from m3tm.task_heads.classification import ClassificationHead
from m3tm.core.base_model import BaseModel

class TestModelRobustness:
    """M³TM modelinin sağlamlık testleri"""
    
    def setup_model(self):
        """Test için tam M³TM modeli oluşturur"""
        config = get_tiny_config()
        model = BaseModel(config)
        return model, config
    
    @pytest.mark.integration
    def test_noisy_text_input(self):
        """Gürültülü metin girişiyle model performansı testi"""
        model, config = self.setup_model()
        
        # Temiz metin girdisi
        batch_size = 2
        seq_len = 16
        clean_text = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        
        # Gürültülü metin girdisi (bazı token'ları UNK ile değiştir)
        noisy_text = clean_text.clone()
        # Her token'ın %20'sini UNK (genellikle 1) ile değiştir
        mask = torch.rand_like(noisy_text, dtype=torch.float) < 0.2
        noisy_text[mask] = 1  # UNK token ID'si
        
        # İki girdiyle model çıktılarını karşılaştır
        with torch.no_grad():
            clean_output = model(text_input=clean_text)
            noisy_output = model(text_input=noisy_text)
        
        # Çıktılar farklı olmalı ancak tamamen farklı olmamalı
        clean_emb = clean_output["search_embedding"]
        noisy_emb = noisy_output["search_embedding"]
        
        # Kosinus benzerliği hesapla
        similarity = torch.nn.functional.cosine_similarity(clean_emb, noisy_emb, dim=1)
        
        # Benzerlik belirli bir eşiğin üzerinde olmalı (örn: 0.5)
        # Bu, modelin gürültüye kısmen dayanıklı olduğunu gösterir
        assert torch.all(similarity > 0.5), f"Gürültülü metin girdisiyle benzerlik çok düşük: {similarity}"
    
    @pytest.mark.integration
    def test_noisy_image_input(self):
        """Gürültülü görüntü girdisiyle model performansı"""
        model, config = self.setup_model()
        
        # Temiz görüntü girişi
        batch_size = 2
        img_size = 32
        clean_image = torch.rand(batch_size, 3, img_size, img_size)
        
        # Gürültülü görüntü girişi
        noisy_image = clean_image + 0.1 * torch.randn_like(clean_image)
        # [0,1] aralığına kliplemek gerekiyor
        noisy_image = torch.clamp(noisy_image, 0, 1)
        
        # İki girdiyle model çıktılarını karşılaştır
        with torch.no_grad():
            clean_output = model(image_input=clean_image)
            noisy_output = model(image_input=noisy_image)
        
        # Çıktılar farklı olmalı ancak tamamen farklı olmamalı
        clean_emb = clean_output["search_embedding"]
        noisy_emb = noisy_output["search_embedding"]
        
        # Kosinus benzerliği hesapla
        similarity = torch.nn.functional.cosine_similarity(clean_emb, noisy_emb, dim=1)
        
        # Benzerlik belirli bir eşiğin üzerinde olmalı (örn: 0.7)
        assert torch.all(similarity > 0.7), f"Gürültülü görüntü girdisiyle benzerlik çok düşük: {similarity}"
    
    @pytest.mark.integration
    def test_missing_modality(self):
        """Eksik modalite durumunda model davranışı testi"""
        model, config = self.setup_model()
        
        batch_size = 2
        seq_len = 16
        img_size = 32
        
        # Test girdileri
        text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # Üç farklı durum için model çıktılarını al
        with torch.no_grad():
            text_only_output = model(text_input=text_input)
            image_only_output = model(image_input=image_input)
            multimodal_output = model(text_input=text_input, image_input=image_input)
        
        # Çıktılar geçerli olmalı
        for output in [text_only_output, image_only_output, multimodal_output]:
            assert "search_embedding" in output
            assert output["search_embedding"].shape == (batch_size, config.search_config.search_dim)
            
            # Embeddingler normalize edilmiş olmalı
            norms = torch.norm(output["search_embedding"], p=2, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        
        # Çok-modlu çıktı, tekli modlu çıktılardan farklı olmalı
        text_emb = text_only_output["search_embedding"]
        image_emb = image_only_output["search_embedding"]
        multi_emb = multimodal_output["search_embedding"]
        
        # İki tekli çıktıyla çok modlu çıktı arasındaki benzerlik orta düzeyde olmalı
        text_multi_sim = torch.nn.functional.cosine_similarity(text_emb, multi_emb, dim=1)
        image_multi_sim = torch.nn.functional.cosine_similarity(image_emb, multi_emb, dim=1)
        
        # Her iki modaliteye de biraz benzemeli ama aynı olmamalı
        assert torch.all(text_multi_sim > 0.3) and torch.all(text_multi_sim < 0.95)
        assert torch.all(image_multi_sim > 0.3) and torch.all(image_multi_sim < 0.95)
    
    @pytest.mark.integration
    def test_variable_input_sizes(self):
        """Farklı giriş boyutlarıyla model davranışı testi"""
        model, config = self.setup_model()
        
        # Farklı metin uzunlukları
        batch_size = 2
        text_lens = [8, 16, 32, 64]
        image_sizes = [16, 32, 48, 64]  # Patch size'a bölünebilir olmalı
        
        # Her boyut kombinasyonunu test et
        for text_len in text_lens:
            text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, 
                                     (batch_size, text_len))
            
            # Metin girişi ile model çağrımı
            with torch.no_grad():
                text_output = model(text_input=text_input)
            
            # Çıktı doğrulama
            assert "search_embedding" in text_output
            assert text_output["search_embedding"].shape == (batch_size, config.search_config.search_dim)
            
        for img_size in image_sizes:
            # Patch size'a bölünebilir olduğundan emin olalım
            if img_size % config.image_config.patch_size != 0:
                continue
                
            image_input = torch.rand(batch_size, 3, img_size, img_size)
            
            # Görüntü girişi ile model çağrımı
            with torch.no_grad():
                image_output = model(image_input=image_input)
            
            # Çıktı doğrulama
            assert "search_embedding" in image_output
            assert image_output["search_embedding"].shape == (batch_size, config.search_config.search_dim)
    
    @pytest.mark.integration
    def test_low_quality_images(self):
        """Düşük kaliteli görüntülerle model davranışı testi"""
        model, config = self.setup_model()
        
        batch_size = 2
        img_size = 32
        
        # Orijinal görüntü
        original_image = torch.rand(batch_size, 3, img_size, img_size)
        
        # Bulanık görüntü (PIL Image üzerinden)
        blurred_images = []
        for i in range(batch_size):
            # Tensor'dan PIL'e
            img = (original_image[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Bulanıklaştır
            blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
            
            # PIL'den Tensor'a
            blurred_np = np.array(blurred_pil).astype(np.float32) / 255.0
            blurred_tensor = torch.from_numpy(blurred_np).permute(2, 0, 1)
            blurred_images.append(blurred_tensor)
            
        blurred_image = torch.stack(blurred_images)
        
        # Düşük kontrastlı görüntü
        low_contrast_image = original_image * 0.5 + 0.25
        
        # Düşük çözünürlüklü görüntü (küçültüp, tekrar boyutlandır)
        downscaled_images = []
        for i in range(batch_size):
            # Tensor'dan PIL'e
            img = (original_image[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Küçült ve tekrar boyutlandır
            small_pil = pil_img.resize((img_size // 4, img_size // 4), Image.BICUBIC)
            back_pil = small_pil.resize((img_size, img_size), Image.BICUBIC)
            
            # PIL'den Tensor'a
            downscaled_np = np.array(back_pil).astype(np.float32) / 255.0
            downscaled_tensor = torch.from_numpy(downscaled_np).permute(2, 0, 1)
            downscaled_images.append(downscaled_tensor)
            
        downscaled_image = torch.stack(downscaled_images)
        
        # Farklı görüntü kaliteleriyle modeli test et
        with torch.no_grad():
            original_output = model(image_input=original_image)
            blurred_output = model(image_input=blurred_image)
            low_contrast_output = model(image_input=low_contrast_image)
            downscaled_output = model(image_input=downscaled_image)
        
        # Orijinal vs. bozulmuş görüntüler arasındaki benzerlikler
        for name, degraded_output in [
            ("bulanık", blurred_output), 
            ("düşük kontrastlı", low_contrast_output), 
            ("düşük çözünürlüklü", downscaled_output)
        ]:
            original_emb = original_output["search_embedding"]
            degraded_emb = degraded_output["search_embedding"]
            
            # Kosinus benzerliği
            similarity = torch.nn.functional.cosine_similarity(original_emb, degraded_emb, dim=1)
            
            # Benzerlik belirli bir eşiğin üzerinde olmalı
            assert torch.all(similarity > 0.6), f"{name} görüntü girişiyle benzerlik çok düşük: {similarity}"
    
    @pytest.mark.integration
    def test_multilingual_text(self):
        """Çoklu dil desteği testi"""
        model, config = self.setup_model()
        tokenizer = model.text_embedding.tokenizer
        
        # Farklı dillerde özel karakterler içeren metinler
        test_texts = [
            "Bu bir Türkçe testtir. Şğıöçü karakterleri içerir.",
            "これは日本語のテストです。漢字とカタカナとひらがなを含みます。",
            "Это тест на русском языке с кириллицей.",
            "هذا اختبار للغة العربية مع الحروف العربية.",
            "這是中文測試，包含簡體和繁體字符。"
        ]
        
        # Modeli her metinle test et
        for text in test_texts:
            # Tokenizerı kullanarak token ID'leri oluştur
            # Tokenizer yoksa veya farklı bir API kullanıyorsa bu kısım uyarlanabilir
            try:
                token_ids = tokenizer.encode(text)
                token_tensor = torch.tensor(token_ids).unsqueeze(0)  # batch dim ekle
                
                # Modeli çalıştır
                with torch.no_grad():
                    output = model(text_input=token_tensor)
                
                # Çıktı doğrulama
                assert "search_embedding" in output
                assert output["search_embedding"].shape[0] == 1
                assert output["search_embedding"].shape[1] == config.search_config.search_dim
                
                # Embeddingler normalize edilmiş olmalı
                norms = torch.norm(output["search_embedding"], p=2, dim=1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
                
            except Exception as e:
                pytest.fail(f"'{text}' metni için test başarısız oldu: {str(e)}")
    
    @pytest.mark.integration
    def test_resource_constraints(self):
        """Kaynak kısıtları altında model davranışı testi"""
        model, config = self.setup_model()
        
        # Düşük bellek senaryosu - küçük batch_size kullan
        batch_size = 1
        seq_len = 16
        img_size = 32
        
        text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # Düşük bellek senaryosunu simüle et
        try:
            # Gradyanları devre dışı bırak
            with torch.no_grad():
                # Zaman ölçümü ve bellek kullanımını gözlemle
                import time
                import psutil
                
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss / 1024 / 1024  # MB cinsinden
                
                start_time = time.time()
                output = model(text_input=text_input, image_input=image_input)
                end_time = time.time()
                
                end_memory = process.memory_info().rss / 1024 / 1024  # MB cinsinden
                
                # Çalışma süresi ve bellek kullanımı
                inference_time = end_time - start_time
                memory_usage = end_memory - start_memory
                
                print(f"Çıkarım süresi: {inference_time:.4f} saniye")
                print(f"Tahmini bellek kullanımı: {memory_usage:.2f} MB")
                
                # Mobil için makul değer mi?
                # Not: Gerçek sınırlar cihaza bağlı olacaktır
                assert inference_time < 1.0, f"Çıkarım çok uzun sürüyor: {inference_time:.4f} saniye"
                
                # Çıktı doğrulama
                assert "search_embedding" in output
                
        except Exception as e:
            pytest.fail(f"Kaynak kısıtları testi başarısız oldu: {str(e)}")
    
    @pytest.mark.integration
    def test_adversarial_inputs(self):
        """Modeli saldırgan (adversarial) girdilerle test eder"""
        model, config = self.setup_model()
        
        batch_size = 2
        seq_len = 16
        img_size = 32
        
        # Normal girdiler
        normal_text = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        normal_image = torch.rand(batch_size, 3, img_size, img_size)
        
        # Aşırı tekrarlayan içerik (tüm tokenleri aynı)
        repetitive_text = torch.full((batch_size, seq_len), 
                                  42 % config.text_config.tokenizer_config.vocab_size, 
                                  dtype=torch.long)
        
        # Aşırı görüntü (tamamen beyaz/parlak veya siyah/karanlık)
        bright_image = torch.ones((batch_size, 3, img_size, img_size))
        dark_image = torch.zeros((batch_size, 3, img_size, img_size))
        
        # Saldırgan paternler (yüksek kontrastlı ızgaralar)
        checkerboard_image = torch.zeros((batch_size, 3, img_size, img_size))
        for i in range(img_size):
            for j in range(img_size):
                if (i + j) % 2 == 0:
                    checkerboard_image[:, :, i, j] = 1.0
        
        # Tüm girdilerle modeli test et
        with torch.no_grad():
            # Normal girdiler
            normal_output = model(text_input=normal_text, image_input=normal_image)
            
            # Aşırı tekrarlayan/uç/saldırgan girdiler
            repetitive_text_output = model(text_input=repetitive_text)
            bright_image_output = model(image_input=bright_image)
            dark_image_output = model(image_input=dark_image)
            checkerboard_output = model(image_input=checkerboard_image)
            
            # Uç koşullar: tekrarlayan metin + aşırı görüntüler
            repetitive_bright_output = model(text_input=repetitive_text, image_input=bright_image)
            repetitive_dark_output = model(text_input=repetitive_text, image_input=dark_image)
            repetitive_checkerboard_output = model(text_input=repetitive_text, image_input=checkerboard_image)
        
        # Tüm çıktılar geçerli ve normalize edilmiş olmalı
        for name, output in [
            ("normal", normal_output),
            ("tekrarlayan metin", repetitive_text_output),
            ("parlak görüntü", bright_image_output),
            ("karanlık görüntü", dark_image_output),
            ("dama tahtası görüntü", checkerboard_output),
            ("tekrarlayan metin + parlak görüntü", repetitive_bright_output),
            ("tekrarlayan metin + karanlık görüntü", repetitive_dark_output),
            ("tekrarlayan metin + dama tahtası", repetitive_checkerboard_output)
        ]:
            # Temel çıktı kontrolü
            assert "search_embedding" in output, f"{name} çıktısında search_embedding yok"
            
            # Embeddingler normalize edilmiş olmalı
            emb = output["search_embedding"]
            norms = torch.norm(emb, p=2, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), f"{name} için embeddingler normalize edilmemiş"
            
            # NaN veya Inf değerler olmamalı
            assert not torch.isnan(emb).any(), f"{name} çıktısında NaN değerler var"
            assert not torch.isinf(emb).any(), f"{name} çıktısında Inf değerler var" 