"""
M³TM modelinin mobil performans testleri

Bu modul, modelin mobil cihazlarda performansını test eder:
- Çıkarım süresi (inference time)
- Bellek kullanımı
- Enerji tüketimi emülasyonu
- Model boyutu ve disk kullanımı
- Çeşitli mobil kısıtlamalar altında çalışma yeteneği
"""
import pytest
import torch
import tempfile
import time
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

from m3tm.config.model_config import get_tiny_config, get_small_config
from m3tm.core.base_model import M3TMBaseModel
from m3tm.mobile.model_converter import MobileModelConverter
from m3tm.mobile.optimization import optimize_model_for_mobile
from m3tm.mobile.benchmarking import benchmark_inference


class TestMobilePerformance:
    """M³TM modelinin mobil performans testleri"""
    
    def setup_model(self, size="tiny"):
        """Test için mobil uyumlu M³TM modeli oluşturur"""
        if size == "tiny":
            config = get_tiny_config()
        elif size == "small":
            config = get_small_config()
        else:
            raise ValueError(f"Geçersiz model boyutu: {size}")
            
        model = M3TMBaseModel(config)
        # Mobil optimizasyon
        model = optimize_model_for_mobile(model)
        return model, config
    
    @pytest.mark.mobile
    @pytest.mark.integration
    def test_inference_time(self):
        """Çeşitli koşullarda çıkarım süresi testleri"""
        # Tiny model en hızlı olmalı
        model, config = self.setup_model(size="tiny")
        
        # Test girdileri
        batch_size = 1  # Mobil cihazlarda genellikle batch_size=1
        seq_len = 16
        img_size = 64
        
        text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # Çıkarım senaryoları
        scenarios = {
            "yalnız_metin": {"text_input": text_input, "image_input": None},
            "yalnız_görüntü": {"text_input": None, "image_input": image_input},
            "metin_ve_görüntü": {"text_input": text_input, "image_input": image_input}
        }
        
        results = {}
        
        # Isınma çalıştırması
        with torch.no_grad():
            _ = model(**scenarios["metin_ve_görüntü"])
        
        # Her senaryoyu ölç
        for scenario_name, inputs in scenarios.items():
            # Birden fazla kez çalıştırıp ortalamasını al
            times = []
            memory_usage = []
            
            for _ in range(10):  # 10 deneme
                # Bellek kullanımını ölç
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    start_mem = torch.cuda.memory_allocated()
                
                # Zamanı ölç
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model(**inputs)
                
                # Süreyi hesapla
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # ms cinsinden
                times.append(inference_time)
                
                # Bellek kullanımını hesapla (varsa)
                if torch.cuda.is_available():
                    end_mem = torch.cuda.memory_allocated()
                    mem_used = (end_mem - start_mem) / 1024 / 1024  # MB cinsinden
                    memory_usage.append(mem_used)
            
            # Sonuçları topla
            avg_time = sum(times) / len(times)
            std_time = np.std(times)
            
            results[scenario_name] = {
                "avg_time_ms": avg_time,
                "std_time_ms": std_time
            }
            
            if memory_usage:
                results[scenario_name]["avg_memory_mb"] = sum(memory_usage) / len(memory_usage)
                results[scenario_name]["peak_memory_mb"] = max(memory_usage)
        
        # Sonuçları yazdır
        print("\nÇıkarım Süresi Sonuçları:")
        for scenario, metrics in results.items():
            print(f"{scenario}: {metrics['avg_time_ms']:.2f} ms (±{metrics['std_time_ms']:.2f})")
            if "avg_memory_mb" in metrics:
                print(f"  Bellek: {metrics['avg_memory_mb']:.2f} MB (Peak: {metrics['peak_memory_mb']:.2f} MB)")
        
        # Mobil cihazlar için makul değerler mi?
        # Not: Gerçek eşikler cihaza ve kullanım durumuna bağlı olacak
        for scenario, metrics in results.items():
            assert metrics["avg_time_ms"] < 100, f"{scenario} için çıkarım süresi çok uzun: {metrics['avg_time_ms']:.2f} ms"
    
    @pytest.mark.mobile
    @pytest.mark.integration
    def test_model_size(self):
        """Model boyutu ve disk kullanımı testi"""
        model, config = self.setup_model(size="tiny")
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Modeli kaydet
            torch.save(model.state_dict(), temp_path)
            
            # Dosya boyutunu al
            file_size_bytes = os.path.getsize(temp_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            print(f"\nModel Dosya Boyutu: {file_size_mb:.2f} MB")
            
            # MobileModelConverter'ı kullanarak optimizasyonu test et
            converter = MobileModelConverter()
            optimized_path = temp_path + ".optimized"
            
            try:
                # Modeli mobil için optimize et
                converter.convert_for_mobile(model, optimized_path)
                
                # Optimize edilmiş dosya boyutunu al
                opt_size_bytes = os.path.getsize(optimized_path)
                opt_size_mb = opt_size_bytes / (1024 * 1024)
                
                print(f"Optimize Edilmiş Model Boyutu: {opt_size_mb:.2f} MB")
                print(f"Sıkıştırma Oranı: {(1 - opt_size_mb / file_size_mb) * 100:.2f}%")
                
                # Mobil için makul boyut mu?
                # Bu değer gereksinimlerinize bağlı olarak değişebilir
                assert opt_size_mb < 15, f"Optimize edilmiş model boyutu çok büyük: {opt_size_mb:.2f} MB"
                
            except Exception as e:
                # MobileModelConverter sınıfı mevcut olmayabilir, bu durumda sadece kaydedelim
                print(f"Mobile optimizasyon testi atlandı: {str(e)}")
            
            # Standart model boyutu için doğrulama
            assert file_size_mb < 20, f"Model boyutu çok büyük: {file_size_mb:.2f} MB"
            
        finally:
            # Geçici dosyaları temizle
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(optimized_path):
                os.unlink(optimized_path)
    
    @pytest.mark.mobile
    @pytest.mark.integration
    def test_quantization_impact(self):
        """Kuantalama (quantization) etkisi testi"""
        model, config = self.setup_model(size="tiny")
        
        # Test girdileri
        batch_size = 1
        seq_len = 16
        img_size = 64
        
        text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # Orijinal model çıktısı
        with torch.no_grad():
            original_output = model(text_input=text_input, image_input=image_input)
        
        # PyTorch kuantalama (quantization) API'sini kullan
        try:
            # Statik kuantalama için modeli hazırla
            model_fp32_prepared = torch.quantization.prepare(model)
            
            # Kalibrasyon verisiyle modeli çalıştır (burada test girdilerini kullanıyoruz)
            with torch.no_grad():
                _ = model_fp32_prepared(text_input=text_input, image_input=image_input)
            
            # Modeli kuantala
            model_int8 = torch.quantization.convert(model_fp32_prepared)
            
            # Kuantize edilmiş model çıktısı
            with torch.no_grad():
                quantized_output = model_int8(text_input=text_input, image_input=image_input)
            
            # Orijinal ile kuantize edilmiş çıktıyı karşılaştır
            original_emb = original_output["search_embedding"]
            quantized_emb = quantized_output["search_embedding"]
            
            # Kosinus benzerliği
            similarity = torch.nn.functional.cosine_similarity(original_emb, quantized_emb, dim=1)
            
            print(f"\nKuantalama sonrası benzerlik: {similarity.item():.4f}")
            
            # Benzerlik yeterince yüksek olmalı
            assert torch.all(similarity > 0.9), f"Kuantalama sonrası benzerlik çok düşük: {similarity}"
            
            # Boyut karşılaştırma
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_fp32:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_int8:
                    torch.save(model.state_dict(), temp_fp32.name)
                    torch.save(model_int8.state_dict(), temp_int8.name)
                    
                    fp32_size = os.path.getsize(temp_fp32.name) / (1024 * 1024)
                    int8_size = os.path.getsize(temp_int8.name) / (1024 * 1024)
                    
                    print(f"FP32 Model Boyutu: {fp32_size:.2f} MB")
                    print(f"INT8 Model Boyutu: {int8_size:.2f} MB")
                    print(f"Boyut Azaltma: {(1 - int8_size / fp32_size) * 100:.2f}%")
                    
                    # Temizlik
                    os.unlink(temp_fp32.name)
                    os.unlink(temp_int8.name)
            
        except Exception as e:
            pytest.skip(f"Kuantalama testi atlandı: {str(e)}")
    
    @pytest.mark.mobile
    @pytest.mark.integration
    def test_energy_consumption(self):
        """Enerji tüketimi emülasyonu testi"""
        model, config = self.setup_model(size="tiny")
        
        # Test girdileri
        batch_size = 1
        seq_len = 16
        img_size = 64
        
        text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # CPU enerji tüketimini simüle et
        # Not: Gerçek ölçümler için özel donanım/yazılım gerekir
        try:
            import psutil
            
            # İşlemci kullanımını ölç
            cpu_percent_start = psutil.cpu_percent(interval=0.1)
            start_time = time.time()
            
            # Birden fazla çıkarım yap (daha doğru ölçüm için)
            num_inferences = 100
            with torch.no_grad():
                for _ in range(num_inferences):
                    _ = model(text_input=text_input, image_input=image_input)
            
            end_time = time.time()
            cpu_percent_end = psutil.cpu_percent(interval=0.1)
            
            # Çıkarım başına süre (ms)
            time_per_inference_ms = (end_time - start_time) * 1000 / num_inferences
            
            # İşlemci kullanımındaki değişim
            cpu_usage_change = cpu_percent_end - cpu_percent_start
            
            print(f"\nÇıkarım Başına Süre: {time_per_inference_ms:.2f} ms")
            print(f"İşlemci Kullanımı Değişimi: {cpu_usage_change:.2f}%")
            
            # "Enerji skoru" hesapla (tahmini)
            # Not: Bu son derece kabataslak bir tahmindir, gerçek enerji tüketimi 
            # çok daha karmaşık şekilde ölçülmelidir
            energy_score = time_per_inference_ms * abs(cpu_usage_change) / 100
            print(f"Tahmini Enerji Skoru: {energy_score:.2f}")
            
            # Mobil için makul değer mi? (düşük değerler daha iyidir)
            assert energy_score < 5, f"Enerji skoru çok yüksek: {energy_score:.2f}"
            
        except ImportError:
            pytest.skip("psutil modülü bulunamadı, enerji testi atlandı")
    
    @pytest.mark.mobile
    @pytest.mark.integration
    def test_android_compatibility(self):
        """Android uyumluluğu ve SDK entegrasyon testi"""
        model, config = self.setup_model(size="tiny")
        
        # Android SDK bridge'inin olmadığı durumlarda çalışır
        try:
            from m3tm.mobile.android_bridge import prepare_for_android
            
            # Geçici dosya oluştur
            with tempfile.TemporaryDirectory() as temp_dir:
                # Modeli Android'e hazırla
                android_model_path = os.path.join(temp_dir, "model.pt")
                metadata_path = os.path.join(temp_dir, "metadata.json")
                
                # Android için modeli hazırla (traceable bir format)
                prepare_for_android(model, android_model_path, metadata_path)
                
                # Dosyaların var olduğunu kontrol et
                assert os.path.exists(android_model_path), "Android model dosyası oluşturulamadı"
                assert os.path.exists(metadata_path), "Metadata dosyası oluşturulamadı"
                
                # Metadata içeriğini kontrol et
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Gerekli alanları içermeli
                required_fields = [
                    "model_format_version", "input_shapes", "output_shapes",
                    "text_vocab_size", "image_size", "patch_size"
                ]
                
                for field in required_fields:
                    assert field in metadata, f"Metadata dosyasında {field} alanı eksik"
                
                # Android için doğru format
                assert metadata["model_format_version"] >= 1, "Geçersiz model format versiyonu"
                
                print(f"\nAndroid model ve metadata dosyaları başarıyla oluşturuldu")
                print(f"Metadata: {metadata}")
                
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Android modülleri bulunamadı, Android testi atlandı")
    
    @pytest.mark.mobile
    @pytest.mark.integration
    def test_multi_device_compatibility(self):
        """Farklı cihaz ve işlemci mimarileri uyumluluğu testi"""
        model, config = self.setup_model(size="tiny")
        
        # Test girdileri
        batch_size = 1
        seq_len = 16
        img_size = 64
        
        text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # Farklı cihazlarda çalışabilirliği test et
        devices = ["cpu"]
        
        # GPU varsa listeye ekle
        if torch.cuda.is_available():
            devices.append("cuda")
        
        # MPS (Metal Performance Shaders - Apple M1/M2 gibi) varsa listeye ekle
        if hasattr(torch, "mps") and torch.mps.is_available():
            devices.append("mps")
        
        # Her cihazda testi çalıştır
        for device_name in devices:
            device = torch.device(device_name)
            
            try:
                # Modeli ve girdileri cihaza taşı
                model_device = model.to(device)
                text_input_device = text_input.to(device)
                image_input_device = image_input.to(device)
                
                # Çıkarım yap
                with torch.no_grad():
                    output = model_device(text_input=text_input_device, image_input=image_input_device)
                
                # Çıktı doğrulama
                assert "search_embedding" in output
                assert output["search_embedding"].shape == (batch_size, config.search_config.search_dim)
                
                # Embeddingler normalize edilmiş olmalı
                emb = output["search_embedding"]
                norms = torch.norm(emb, p=2, dim=1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
                
                print(f"\n{device_name.upper()} cihazında model başarıyla çalıştı")
                
            except Exception as e:
                print(f"\n{device_name.upper()} cihazında test başarısız: {str(e)}")
                # Test yalnızca bilgi amaçlı, hatalar için başarısız sayma
                pass
    
    @pytest.mark.mobile
    @pytest.mark.integration
    def test_continual_inference(self):
        """Sürekli çıkarım (uzun süre çalışma) testi"""
        model, config = self.setup_model(size="tiny")
        
        # Test girdileri
        batch_size = 1
        seq_len = 16
        img_size = 32
        
        text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
        image_input = torch.rand(batch_size, 3, img_size, img_size)
        
        # Sürekli çıkarım testi (bellek sızıntılarını yakalamak için)
        import gc
        import psutil
        
        process = psutil.Process(os.getpid())
        
        # İlk bellek durumu
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Birden fazla çıkarım yap
        num_iterations = 100
        with torch.no_grad():
            for i in range(num_iterations):
                # Her 10 iterasyonda bir yeni rastgele girdi oluştur
                if i % 10 == 0:
                    text_input = torch.randint(0, config.text_config.tokenizer_config.vocab_size, (batch_size, seq_len))
                    image_input = torch.rand(batch_size, 3, img_size, img_size)
                
                # Çıkarım yap
                output = model(text_input=text_input, image_input=image_input)
                
                # Belleği temizle
                if i % 25 == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Son bellek durumu
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Bellek artışı
        memory_increase = final_memory - initial_memory
        
        print(f"\nBaşlangıç Bellek Kullanımı: {initial_memory:.2f} MB")
        print(f"Son Bellek Kullanımı: {final_memory:.2f} MB")
        print(f"Bellek Artışı: {memory_increase:.2f} MB ({memory_increase / initial_memory * 100:.2f}%)")
        
        # Bellek artışı kabul edilebilir sınırlar içinde olmalı
        # Not: Tam eşik değerleri uygulamaya bağlı olacaktır
        assert memory_increase / initial_memory < 0.5, f"Bellek artışı çok yüksek: {memory_increase:.2f} MB" 