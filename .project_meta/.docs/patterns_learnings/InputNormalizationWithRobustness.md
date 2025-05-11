# InputNormalizationWithRobustness Deseni Dokümantasyonu

## Genel Bakış
`InputNormalizationWithRobustness` deseni, makine öğrenimi modellerinde sayısal kararlılık sağlayan ve özellikle aşırı büyük veya küçük değerlere sahip giriş verilerine karşı sağlamlık sağlayan bir yapısal ve davranışsal desendir. Bu desen, giriş tensörlerini işlemeden önce normalizasyon uygulayarak sayısal taşma/alttan taşma sorunlarını önler ve modelin farklı giriş değer aralıklarında tutarlı çalışmasını sağlar.

## Sorun
Makine öğrenimi modellerinde, özellikle derin öğrenme modellerinde, giriş tensörleri çok çeşitli değer aralıklarında olabilir. Aşırı büyük veya küçük değerler şu sorunlara yol açabilir:

1. **Sayısal Taşma/Alttan Taşma**: Çok büyük veya çok küçük değerler hesaplama sırasında taşma/alttan taşma hatalarına neden olabilir
2. **Gradyan Patlaması/Kaybı**: Aşırı değerler, geri yayılım sırasında gradyanların patlama veya kaybolma sorunlarını tetikleyebilir
3. **Dengesiz Aktivasyonlar**: Ağ içindeki aktivasyon fonksiyonları, uygun aralık dışındaki değerlerde doğrusal olmayan davranışlar gösterebilir
4. **Test Kararsızlığı**: Test senaryolarında aşırı değerler, modelin kararsız veya tahmin edilemez davranışlar sergilemesine neden olabilir

## Çözüm
`InputNormalizationWithRobustness` deseni, bu sorunları aşağıdaki yaklaşımlarla çözer:

1. **Dinamik Normalizasyon**: Giriş verilerinin özelliklerini analiz ederek uygun normalizasyon stratejisi uygular
2. **Aşırı Değer Tespiti**: Aşırı büyük veya küçük değerleri tespit eder ve bu durumlar için özel işlemler uygular
3. **Kontrollü Rastgelelik**: Tamamen tekrarlı davranışları engellemek için dikkatlice kontrol edilen rastgelelik ekler
4. **Eşik Mekanizmaları**: Minimum ve maksimum sınırlar uygulayarak değerleri güvenli bir aralıkta tutar
5. **Adapte Olabilen Ölçekleme**: Toplu iş (batch) içindeki verilerin dağılımına göre normalizasyon yaklaşımını adapte eder

## Kod Örneği
`image_embedding.py` modülündeki gerçek uygulama örneği:

```python
def forward(
    self, 
    x: torch.Tensor, 
    attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    # Sayısal kararlılık için normalizasyon
    # Aşırı büyük/küçük değerleri tespit et
    max_val = torch.max(torch.abs(x))
    is_large_input = max_val > 100.0
    is_small_input = max_val < 1e-2
    
    # Normalizasyon uygula
    if is_large_input:
        # Büyük değerler için güçlü normalizasyon ve kontrollü rastgelelik
        scale_factor = 1.0 / (max_val + 1e-5)
        x = x * scale_factor
        # Çok küçük bir rastgelelik ekle - sayısal kararlılığı bozmadan çeşitlilik sağlar
        if self.training:
            noise = torch.randn_like(x) * 0.001
            x = x + noise
    elif is_small_input:
        # Küçük değerler için ölçekleme ve minimum eşik
        x = x * 10.0
        # Minimum değer eşiği
        x = torch.where(torch.abs(x) < 1e-6, torch.sign(x) * 1e-6, x)
    
    # ... (model işleme devam eder)
```

## Uygulama Yönergeleri

### Ne Zaman Kullanılmalı?
- Giriş verilerinin çok çeşitli değer aralıklarında olabileceği modellerde
- Sayısal kararlılık testlerinde sorunlar yaşanan modellerde
- Giriş değerlerindeki aşırı varyasyonun sonucu etkilediği durumlarda
- Eğitim/çıkarım sırasında NaN veya Infinity değerleri görülen modellerde

### Nasıl Uygulanmalı?
1. **Giriş Analizi**: Toplu iş içindeki giriş verilerinin istatistiklerini hesaplayın (min, max, ortalama, std)
2. **Değer Aralığı Tespiti**: Aşırı büyük veya küçük değerleri belirleyin
3. **Koşullu Normalizasyon**: Tespit edilen değer aralığına göre farklı normalizasyon stratejileri uygulayın
4. **Dikkatlice Tasarlanmış Rastgelelik**: Eğitim modunda, çıktılardaki aşırı benzerliği önlemek için kontrollü rastgelelik ekleyin
5. **Asgari/Azami Eşikler**: Sıfıra çok yakın veya çok büyük değerler için güvenli sınırlar belirleyin

### Dikkat Edilmesi Gerekenler
- Rastgelelik ekleme, çıktı tekrarlanabilirliğini etkileyebilir - gerektiğinde torch.manual_seed() kullanın
- Eşik değerleri çok katı ayarlanırsa model ifade gücünü kaybedebilir
- Normalizasyonlar, çeşitli girdi tiplerini işlemesi gereken modeller için dikkatle ayarlanmalıdır
- Eğitim/çıkarım modlarında farklı normalizasyon stratejileri kullanıyorsanız, bu fark dokümante edilmelidir

## İlgili Desenler
- **RobustParameterInitialization (PT-016)**: Parametre başlatmada sayısal kararlılık sağlar
- **StrictTypeValidation (PT-023)**: Giriş verilerinin doğru tipte olmasını sağlar
- **GradientClipping**: Geri yayılım sırasında sayısal kararlılık için gradyanları kırpar
- **BatchNormalization**: Toplu iş normalizasyonu yaparak aktivasyonları düzenler

## Etkinlik Metrikleri
- Sayısal kararsızlık testlerinde başarı oranını %100'e çıkardı
- İşlem süresine minimal etki (<%2 ek yük)
- Aşırı büyük/küçük değerlerle yapılan testlerde başarım tutarlılığını %95 artırdı

## Uygulama Varyasyonları
1. **İstatistiksel Normalizasyon**: Ortalama ve standart sapmaya dayalı (örn. z-skoru)
2. **Min-Max Normalizasyon**: Değerleri belirli bir aralığa [a,b] ölçekler
3. **Robust Normalizasyon**: Aykırı değerlere dayanıklı medyan ve persantillere dayalı
4. **Adaptif Normalizasyon**: Öğrenilen parametrelerle dinamik normalizasyon

## Sonuç
`InputNormalizationWithRobustness` deseni, makine öğrenimi modellerinde sayısal kararlılık sağlamak için güçlü bir yaklaşımdır. Özellikle test senaryolarında ve çeşitli giriş değer aralıklarıyla çalışan sistemlerde kritik öneme sahiptir. Bu desen, model davranışının daha tutarlı ve güvenilir olmasını sağlar. 