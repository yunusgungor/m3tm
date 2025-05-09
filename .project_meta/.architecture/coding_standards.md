# M³TM v2.3 Kodlama Standartları

**Versiyon:** 1.1  
**Oluşturma Tarihi:** 23 Mayıs 2024  
**Son Güncelleme:** 23 Mayıs 2024

## 1. Genel Prensipler

### 1.1. Tek Sorumluluk Prensibi (SRP)
- Her sınıf ve dosya tek bir sorumluluk alanına sahip olmalıdır.
- Fonksiyonlar idealde 5-10 satır, maksimum 20 satırla sınırlandırılmalıdır.
- Sınıflar genellikle 100 satırın altında tutulmalıdır.
- 200-500+ satır uzunluğundaki dosyalar, refactoring gerektiren anti-pattern olarak değerlendirilecektir.

### 1.2. Modülerlik
- Fonksiyonalite, mantıksal modüller halinde gruplandırılmalıdır.
- Her modül, `.project_meta/.architecture/module_definitions.json` dosyasında açıkça tanımlanmalıdır.
- Modüller arası bağımlılıklar en aza indirilmeli ve daima `.project_meta/.dependencies/dependency_graph.json` içinde belgelenmelidir.

### 1.3. Kod Örüntüleri
- Tanımlanan kod örüntüleri, `.project_meta/.patterns/pattern_catalog.json` içinde belgelenmelidir.
- Anti-örüntüler `.project_meta/.patterns/anti_patterns.json` içinde belgelenmeli ve düzeltilmelidir.
- Yeni örüntüler tespit edildiğinde kataloglanmalı ve mevcut kodla tutarlı kullanılmalıdır.
- **Uygulama İlkesi:** Katalogdaki örüntüler, ilgili kullanım senaryolarında tutarlı bir şekilde uygulanmalıdır. Aynı problemi çözen birden fazla örüntü varsa, proje içinde tek bir örüntü standardı belirlenmelidir.
- **Örüntü Referansı:** Karmaşık örüntüleri uygulayan dosyalarda dosya başında örüntü referansı belirtilmelidir.

### 1.4. Onaylanmış Örüntüler
Aşağıdaki örüntüler, M³TM projesinde standart uygulama olarak kabul edilmiştir:

#### 1.4.1. ConfigurationDataclass (PT-001)
- **Amaç:** Model parametrelerini yapılandırılabilir ve tip-güvenli bir şekilde yönetmek.
- **Kullanım:** Tüm yapılandırmalar için Python `dataclass` kullanılmalıdır. Alt yapılandırmalar ayrı sınıflar olarak tanımlanmalı, ana yapılandırma bu sınıfların bir kompozisyonu olmalıdır.
- **Gereklilikler:**
  - Varsayılan değerler, her zaman belirtilmelidir.
  - Yapılandırma tutarlılığını sağlamak için `__post_init__` metodu kullanılmalıdır.
  - Yaygın kullanım senaryoları için yardımcı factory fonksiyonları sağlanmalıdır (örn: `get_default_config()`).

#### 1.4.2. ModelCheckpointManager (PT-002)
- **Amaç:** Model durumunu (ağırlıklar, yapılandırma, meta veriler) düzenli bir şekilde kaydetmek ve geri yükleyebilmek.
- **Kullanım:** Tüm model sınıfları, `BaseModel` sınıfından türetilmeli ve onun `save_pretrained` ve `from_pretrained` metodlarını kullanmalıdır.
- **Gereklilikler:**
  - Model ve yapılandırması her zaman birlikte saklanmalıdır.
  - Sınıf metodları kullanılarak model yükleme sağlanmalıdır.
  - İleride: Versiyon bilgisi saklanmalıdır.

## 2. Python ve PyTorch Kodlama Standartları

### 2.1. Dosya Yapısı
- Her modül için ayrı bir Python paketi oluşturulmalıdır.
- Her sınıf için ayrı bir dosya kullanılmalıdır (istisnalar olabilir, örn: küçük yardımcı sınıflar).
- Dosya isimleri küçük harfle, snake_case formatında olmalıdır.
- Sınıf isimleri PascalCase formatında olmalıdır.

### 2.2. İçe Aktarmalar (Imports)
- İçe aktarmalar dosyanın üst kısmında, aşağıdaki sırayla gruplandırılmalıdır:
  1. Standart kütüphane içe aktarmaları
  2. Üçüncü taraf kütüphane içe aktarmaları (örn: torch, numpy)
  3. Uygulama/yerel modül içe aktarmaları
- Her grup arasında bir boş satır bırakılmalıdır.
- Wildcard içe aktarmalardan kaçınılmalıdır (`from module import *`).

### 2.3. PyTorch Kullanım Standartları
- `nn.Module` alt sınıflarında, her zaman `super().__init__()` çağrısı yapılmalıdır.
- `forward` metodu açıkça dokümante edilmeli, girdi ve çıktı şekilleri belirtilmelidir.
- Model bileşenleri `__init__` içinde oluşturulmalı, `forward` içinde dinamik olarak oluşturulmamalıdır.
- Bellek verimliliği için `inplace` operasyonlar tercih edilmelidir.
- Tensor şekil dönüşümleri açıkça belgelenmeli, "sihirli" indeks/boyut manipülasyonlarından kaçınılmalıdır.

### 2.4. Hiper-parametreler ve Yapılandırma
- Tüm hiper-parametreler, yapılandırma nesneleri aracılığıyla iletilmelidir.
- Hard-coded değerlerden kaçınılmalıdır.
- Yapılandırma sınıfları veya dataclass'lar, model bileşenlerini tanımlamak için kullanılmalıdır.

## 3. Dokümantasyon Standartları

### 3.1. Docstrings
- Tüm modüller, sınıflar ve fonksiyonlar Google docstring formatında dokümante edilmelidir.
- Dokümantasyon, işlevselliği, parametreleri ve dönüş değerlerini açıkça belirtmelidir.
- Kompleks algoritmaların dokümanları, matematiksel açıklamalar içermelidir.

### 3.2. Kod İçi Açıklamalar
- Karmaşık kod bloklarının amacını açıklayan yorumlar eklenmelidir.
- "Ne" yerine "neden" açıklanmalıdır (kod ne yaptığını gösterirken, yorumlar niçin yapıldığını açıklamalıdır).
- Kendi kendini dokümante eden kod tercih edilmelidir.

### 3.3. API Dokümantasyonu
- Tüm public API'ler kapsamlı bir şekilde dokümante edilmelidir.
- SDK'nın her örnek kullanımı için kullanım örnekleri sağlanmalıdır.

## 4. Test Standartları

### 4.1. Birim Testleri
- Her modül için birim testleri yazılmalıdır.
- Testler, edge case'leri içermelidir.
- Test senaryoları açıkça adlandırılmalı ve dokümante edilmelidir.

### 4.2. Entegrasyon Testleri
- Modül entegrasyonlarını doğrulayan testler yazılmalıdır.
- End-to-end iş akışları test edilmelidir.

### 4.3. Performans Testleri
- Bellek kullanımı, çıkarım süresi ve eğitim süresi ölçülmelidir.
- Performans değişiklikleri belgelenmeli ve izlenmelidir.

## 5. Mobil Spesifik Standartlar

### 5.1. Mobil Cihaz Kısıtlamaları
- Bellek ayak izi minimize edilmelidir.
- Batarya tüketimi göz önünde bulundurulmalıdır.
- Büyük matris çarpımları ve yüksek maliyetli operasyonlardan kaçınılmalıdır.

### 5.2. CPU/GPU Kullanımı
- Cihaz üzerinde eğitim, adaptif olarak CPU/GPU'yu kullanmalıdır.
- Farklı cihaz türleri için fallback mekanizmaları sağlanmalıdır.

### 5.3. SDK Geliştirme
- Platform spesifik kodlar, platformlar arası katmanlardan açıkça ayrılmalıdır.
- API'ler platform agnostik olmalı, platform detayları alt seviyelerde ele alınmalıdır.
- PyTorch Mobile C++ API'lerini saran temiz arayüzler sağlanmalıdır.

## 6. Kod Gözden Geçirme Kriterleri

Kod gözden geçirme sürecinde aşağıdaki kriterler kullanılacaktır:

1. Kod, belirtilen tüm kodlama standartlarına uyuyor mu?
2. Modül tanımlarına ve bağımlılık grafiğine uyuyor mu?
3. Tanımlanan kod örüntüleri uygun şekilde kullanılmış mı?
4. Dokümantasyon yeterli ve doğru mu?
5. Testler kapsamlı ve geçerli mi?
6. Performans ve bellek kullanımı optimize edilmiş mi?
7. Mobil kısıtlamaları dikkate alınmış mı?
8. Güvenlik ve gizlilik önlemleri uygulanmış mı?
9. İskelet yaklaşımına uygun mu, gereksiz karmaşıklık eklenmiş mi?
10. Ölçeklenebilirlik ve gelecekteki genişletmeler için uygun mu? 