# M³TM v2.3 Mimari Prensipler

**Versiyon:** 1.0  
**Oluşturma Tarihi:** 23 Mayıs 2024  
**Son Güncelleme:** 23 Mayıs 2024

## 1. Temel Mimari Prensipler

### 1.1. Ultra-Hafif Tasarım
- **Prensip:** Model ve bileşenleri mobil cihazların kısıtlı kaynakları göz önünde bulundurularak tasarlanmalıdır.
- **Zorunluluk:** Tüm modüller, bellek ve işlemci kullanımını en aza indirgeyecek şekilde optimize edilmelidir.
- **Doğrulama:** Model bileşenlerinin bellek ve işlemci kullanımı düzenli olarak ölçülmeli ve belgelenmelidir.

### 1.2. Modülerlik ve Genişletilebilirlik
- **Prensip:** Model, bağımsız modüller halinde tasarlanmalı ve bu modüller açık bir şekilde tanımlanmış arayüzler üzerinden iletişim kurmalıdır.
- **Zorunluluk:** Her modül, minimum bağımlılıkla bağımsız olarak test edilebilmeli ve değiştirilebilmelidir.
- **Doğrulama:** Modül bağımlılıkları sürekli izlenmeli ve döngüsel bağımlılıklardan kaçınılmalıdır.

### 1.3. Gizlilik Öncelikli Tasarım
- **Prensip:** Tüm model mimarisi ve veri akışı, kullanıcı verilerinin gizliliğini koruyacak şekilde tasarlanmalıdır.
- **Zorunluluk:** Kullanıcı verileri, kullanıcının açık izni olmadan cihazdan çıkmamalıdır.
- **Doğrulama:** Veri akışı düzenli olarak analiz edilmeli ve veri sızıntıları potansiyeli kontrol edilmelidir.

### 1.4. Adaptif Öğrenme Mimarisi
- **Prensip:** Model mimarisi, çekirdek yapıyı değiştirmeden kullanıcı verileriyle büyüyebilir olmalıdır.
- **Zorunluluk:** Çekirdek model yapısı dondurulmalı, öğrenme adaptörler ve görev başlıkları üzerinden gerçekleşmelidir.
- **Doğrulama:** Adaptif öğrenme bileşenleri işlevselliği düzenli olarak test edilmelidir.

## 2. Mimari Katmanlar

### 2.1. Çekirdek Model Katmanı
- **Sorumluluklar:** Temel veri akışını ve modalite etkileşimini sağlar.
- **Bileşenler:** 
  - Metin İşleme (TextEmbedding)
  - Görüntü İşleme (ImagePatchEmbedding)
  - Proto-Transformer Blokları
  - Füzyon Mekanizması
  - Arama Gömme Projeksiyonu
- **Kısıtlamalar:** Çekirdek model, minimum parametre sayısına sahip olmalı ve eğitim sırasında dondurulmalıdır.

### 2.2. Büyüme Modülleri Katmanı
- **Sorumluluklar:** Modelin kullanıcı verileriyle öğrenme ve genişleme yeteneğini sağlar.
- **Bileşenler:**
  - Adaptörler (Adapter)
  - Görev Başlıkları (Task Heads)
  - Eğitim Yöneticisi (TrainingManager)
- **Kısıtlamalar:** Büyüme modülleri, çekirdek modele dokunmadan eklenebilmeli ve çıkarılabilmelidir.

### 2.3. Veri İşleme Katmanı
- **Sorumluluklar:** Veri önişleme, zenginleştirme ve akış yönetimini sağlar.
- **Bileşenler:**
  - Veri Önişleyiciler (Preprocessors)
  - Veri Yükleyiciler (DataLoaders)
  - Veri Dönüştürücüler (Transformers)
- **Kısıtlamalar:** Veri işleme, cihaz üzerinde verimli bir şekilde gerçekleştirilmelidir.

### 2.4. Uygulama Katmanı (SDK)
- **Sorumluluklar:** Geliştiricilere model yeteneklerini kullanma imkanı sağlar.
- **Bileşenler:**
  - API Sarmalayıcılar
  - Platform-Spesifik Entegrasyonlar
  - Dokümantasyon ve Örnekler
- **Kısıtlamalar:** SDK, basit ve tutarlı bir API sağlamalı, platform detaylarını geliştiricilerden gizlemelidir.

## 3. Veri Akışları

### 3.1. Eğitim Veri Akışı
1. Kullanıcı verisi girişi
2. Veri önişleme ve dönüştürme
3. Çekirdek model üzerinden geçiş (inference-only)
4. Adaptörler ve görev başlıkları üzerinden geçiş ve güncelleme (training)
5. Sonuçların değerlendirilmesi ve metriklerin toplanması

### 3.2. Çıkarım Veri Akışı
1. Kullanıcı sorgusu girişi
2. Sorgu önişleme
3. Çekirdek model üzerinden geçiş
4. İlgili adaptörler üzerinden geçiş
5. İlgili görev başlığı üzerinden geçiş
6. Sonuçların formatlanması ve sunulması

### 3.3. Arama Veri Akışı
1. Kullanıcı sorgusu girişi
2. Sorgu önişleme ve gömme oluşturma
3. İndeks üzerinde benzerlik araması
4. Sonuçların sıralanması ve filtrelenmesi
5. Sonuçların formatlanması ve sunulması

### 3.4. Veri İndirme Akışı
1. Kullanıcı veri indirme talebi
2. Verilerin toplanması ve filtrelenmesi
3. Verilerin formatlanması
4. İndirme paketinin oluşturulması ve sunulması

## 4. Mimari Kısıtlamalar ve Kalite Nitelikleri

### 4.1. Performans Kısıtlamaları
- Model çıkarım süresi, 256x256 görüntü veya 100 token'lık metin için orta seviye bir mobil cihazda 500ms'yi geçmemelidir.
- Eğitim döngüsü, örnek başına 2 saniyeyi geçmemelidir.
- Model boyutu, 50MB'yi geçmemelidir.

### 4.2. Bellek Kısıtlamaları
- Çalışma zamanı bellek kullanımı, 200MB'yi geçmemelidir.
- Eğitim sırasında bellek kullanımı, 500MB'yi geçmemelidir.
- Minimum hedef cihaz belleği: 2GB RAM.

### 4.3. Batarya Kısıtlamaları
- Sürekli model kullanımı, cihaz bataryasını saatte %5'ten fazla tüketmemelidir.
- Eğitim işlemleri, enerji tasarrufu modunda veya düşük batarya durumunda otomatik olarak kısıtlanmalıdır.

### 4.4. Ölçeklenebilirlik
- Mimari, ileride yeni modalitelerin (ses, video) eklenmesine olanak tanımalıdır.
- Adaptör sayısı ve büyüklüğü, kullanıcının cihaz kapasitesine göre otomatik olarak ayarlanabilmelidir.

### 4.5. Test Edilebilirlik
- Her mimari bileşen, bağımsız olarak test edilebilmelidir.
- Mimari, simüle edilmiş mobil ortamlar kullanılarak test edilebilmelidir.
- Performans test araçları, mimari içine entegre edilmelidir.

### 4.6. Güvenlik ve Güvenilirlik
- Model, kötü amaçlı girdilere karşı savunmaya sahip olmalıdır.
- Çökme durumunda, model durumu güvenli bir şekilde kurtarılabilmelidir.
- Tutarsız model durumları otomatik olarak tespit edilmeli ve düzeltilmelidir.

## 5. Teknoloji Seçimleri ve Kısıtlamalar

### 5.1. Temel Teknoloji Yığını
- **PyTorch:** Model geliştirme ve eğitim için ana çerçeve
- **PyTorch Mobile:** Mobil cihazlarda çalıştırma için optimizasyon
- **TorchScript:** Model serileştirme ve çalışma zamanı optimizasyon
- **ONNX:** Platform bağımsız model dışa aktarımı (opsiyonel)

### 5.2. Mobil Platform Entegrasyonları
- **Android:** Java/Kotlin API sarmalayıcıları, C++ JNI köprüsü
- **iOS:** Swift/Objective-C API sarmalayıcıları, C++ köprüsü

### 5.3. Geliştirme Araçları
- **Birim Test:** PyTest
- **Performans Profili:** PyTorch Profiler
- **Lint:** Flake8, Pylint
- **Belgeleme:** Sphinx
- **Bağımlılık Yönetimi:** Poetry veya Pip

## 6. Mimari Kararlar

Mimari kararlar, `.project_meta/.architecture/adr_log.json` dosyasında belgelenecek ve düzenli olarak güncellenecektir. Önemli mimari kararlar aşağıdakileri içerir:

1. Proto-Transformer mimarisi seçimi
2. Adaptör-bazlı büyüme mekanizması
3. Mobil optimizasyon stratejileri
4. Çoklu-modalite füzyon yaklaşımı
5. Arama indeksleme ve sorgu mekanizması
6. SDK API tasarımı

## 7. Mimari Değerlendirme ve Gözden Geçirme

### 7.1. Mimari Değerlendirme Kriterleri
- Performans hedeflerine uygunluk
- Bellek kısıtlamalarına uygunluk
- Modülerlik ve bağımlılık yönetimi
- Mimari prensiplere uygunluk
- Kod kalitesi ve standartlara uyum

### 7.2. Değerlendirme Süreci
- Her iterasyon sonunda mimari değerlendirme yapılacaktır.
- Mimari değişiklikler, mimari kararlar (ADR) olarak belgelenecektir.
- Mimari drift ölçümleri düzenli olarak yapılacak ve raporlanacaktır. 