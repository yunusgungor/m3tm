# M3TM Sayısal Kararlılık Bakım Rehberi

## Genel Bakış

Bu dokümantasyon, M3TM modelinde sayısal kararlılık sorunlarını belirleme, önleme ve çözme süreçlerini kapsar. Özellikle geniş değer aralıklarıyla çalışan gömme modülleri ve model bileşenleri için kritik öneme sahiptir.

## Sayısal Kararsızlık Belirtileri

Aşağıdaki belirtiler, muhtemelen modelde sayısal kararsızlık sorunları olduğunu gösterir:

1. **Aşırı benzer çıktılar**: Farklı girişler için neredeyse aynı çıktılar üretilmesi
2. **Aşırı farklı çıktılar**: Benzer girişler için çok farklı çıktılar üretilmesi
3. **NaN veya Inf değerleri**: İşlemler sırasında sayısal taşma veya alttan taşma
4. **Gradyan patlaması/kaybı**: Eğitim sırasında anormal büyük veya küçük gradyanlar
5. **Test tutarsızlıkları**: Aynı testin farklı çalıştırmalarda farklı sonuçlar vermesi
6. **Model davranış farklılıkları**: Farklı donanım veya çalışma zamanları arasında tutarsızlıklar

## Yaygın Sayısal Kararsızlık Nedenleri

1. **Normalizasyon eksikliği**: Girdi değerlerinin normalizasyon olmadan işlenmesi
2. **Aşırı ölçekleme**: Çok büyük ölçekleme faktörleri kullanılması
3. **Yetersiz sayısal hassasiyet**: Düşük hassasiyetli veri tipleri kullanımı
4. **Aktivasyon doygunluğu**: Aktivasyon fonksiyonlarının doğrusal olmayan bölgelerinde çalışma
5. **Dengesiz başlatma**: Parametrelerin uygun bir aralıkta başlatılmaması
6. **Gürültülü gradyanlar**: Optimizasyon sırasında dengesiz gradyan güncellemeleri

## Sayısal Kararlılık Desenleri

### InputNormalizationWithRobustness (PT-101)

Bu desen, giriş tensörlerinin işlenme öncesinde normalizasyonunu sağlayarak sayısal kararlılık sorunlarını önler.

**Uygulama Örneği**:

```python
def forward(self, x: torch.Tensor, ...):
    # Sayısal kararlılık için normalizasyon
    max_val = torch.max(torch.abs(x))
    is_large_input = max_val > 100.0
    is_small_input = max_val < 1e-2
    
    # Normalizasyon uygula
    if is_large_input:
        # Büyük değerler için güçlü normalizasyon ve kontrollü rastgelelik
        scale_factor = 1.0 / (max_val + 1e-5)
        x = x * scale_factor
        # Çok küçük bir rastgelelik ekle
        if self.training:
            noise = torch.randn_like(x) * 0.001
            x = x + noise
    elif is_small_input:
        # Küçük değerler için ölçekleme ve minimum eşik
        x = x * 10.0
        # Minimum değer eşiği
        x = torch.where(torch.abs(x) < 1e-6, torch.sign(x) * 1e-6, x)
        
    # İşlemeye devam et...
```

### RobustParameterInitialization (PT-016)

Model parametrelerinin uygun aralıkta başlatılmasını sağlayarak sayısal kararlılığı artırır.

### Diğer Yararlı Desenler

1. **GradientClipping**: Geri yayılım sırasında gradyanların sınırlanması
2. **MixedPrecisionTraining**: Farklı hassasiyetlerde hesaplama yaparak bellek kullanımı ve performansı dengeler
3. **StableActivationFunctions**: Sayısal olarak daha kararlı aktivasyon fonksiyonları kullanımı

## Sayısal Kararlılık Test Teknikleri

1. **Aşırı değer testleri**: Çok büyük ve çok küçük giriş değerleri ile test
2. **Benzerlik kıyaslamaları**: Farklı girişler arasındaki çıktı benzerliklerinin analizi
3. **Gradyan kararlılık testleri**: Geri yayılım sırasında gradyanların izlenmesi
4. **Aktivasyon doygunluk testleri**: Aktivasyon değerlerinin analizi
5. **Tekrarlanabilirlik testleri**: Aynı girişle çoklu çalıştırmaların tutarlılığı

## M3TM Testleri Entegrasyonu

`test_numerical_stability.py` modülü dört test içerir:

1. `test_extreme_gradients`: Çok büyük gradyanların önlenmesi testi
2. `test_vanishing_gradients`: Çok küçük gradyanların önlenmesi testi
3. `test_numerical_range_stability`: Farklı büyüklüklerdeki giriş değerlerinde model tepkisi testi
4. `test_activation_saturation`: Aktivasyon fonksiyonlarının doygunluk testi

Bu testler yeni bileşenler eklendiğinde veya mevcut bileşenler değiştirildiğinde kullanılmalıdır.

## En İyi Uygulama Yönergeleri

1. **Her zaman giriş normalizasyonu yapın**: Tüm girdi işleyen fonksiyon ve modüllerde normalizasyon uygulayın
2. **Değer aralığı kısıtlamalarını belirtin**: API dokümantasyonunda değer aralığı varsayımlarını belirtin
3. **İstatistiksel normalizasyon kullanın**: Veri dağılımına uygun normalizasyon yöntemleri seçin
4. **Minimum/Maksimum eşikler tanımlayın**: Aşırı değerler için güvenli sınırlar koyun
5. **Düzenli sayısal testler ekleyin**: Her modül için kararlılık ve sağlamlık testleri yazın
6. **Rastgelelik ekleme dikkatli olun**: Sisteme eklenen rastgelelik kontrollü ve tekrarlanabilir olmalı

## Sorun Giderme

### Yaygın Hatalar ve Çözümleri

1. **Sayısal taşma/alttan taşma hataları**:
   - Çözüm: Tensorü ölçekleyin ve minimum/maksimum eşikler uygulayın

2. **Tutarsız çıktılar veya düşük benzerlik**:
   - Çözüm: Normalizasyon stratejilerini gözden geçirin, aşırı rastgelelik olup olmadığını kontrol edin

3. **NaN veya Infinity değerleri**:
   - Çözüm: İşlemler sırasında bölmeleri ve üsleri kontrol edin, 0'a bölmeyi önleyin

4. **Aktivasyon satürasyonu**:
   - Çözüm: Girdileri aktivasyon fonksiyonunun dinamik aralığına ölçekleyin

### Diyagnostik Araçlar

1. **torch.autograd.detect_anomaly()**: Geri yayılım sırasında NaN/Inf değerlerini tespit etme
2. **torch.isnan/torch.isinf**: Tensor içinde problemli değerleri kontrol etme
3. **torch.histc()**: Tensor değerlerinin dağılımını görselleştirme

## Referanslar

1. [InputNormalizationWithRobustness Desen Dokümantasyonu](./../patterns_learnings/InputNormalizationWithRobustness.md)
2. [Sayısal Kararlılık Entegrasyon Test Raporları](./../../integration/reports/failure_analysis.json)
3. [PT-101 Desen İncelemesi](./../../patterns/reviews/pattern_review_PT101.json)

## Son Güncelleme

Bu dokümantasyon, InputNormalizationWithRobustness (PT-101) deseninin keşfedilmesi ve uygulanması sonrasında, 12 Temmuz 2024 tarihinde güncellenmiştir. 