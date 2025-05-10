# CodeFlow İş Akış Motoru

<p align="center">
  <img src="https://via.placeholder.com/800x200/0073CF/FFFFFF?text=CodeFlow+%C4%B0%C5%9F+Ak%C4%B1%C5%9F+Motoru" alt="CodeFlow Logo">
</p>

<p align="center">
  <a href="#temel-özellikler">Özellikler</a> •
  <a href="#mimari-yapı">Mimari</a> •
  <a href="#i̇ş-akışı">İş Akışı</a> •
  <a href="#kullanım-kılavuzu">Kullanım</a> •
  <a href="#metrikler">Metrikler</a> •
  <a href="#katkıda-bulunma">Katkıda Bulunma</a> •
  <a href="#lisans">Lisans</a>
</p>

## 📋 İçindekiler

- [Proje Tanımı](#proje-tanımı)
- [Temel Özellikler](#temel-özellikler)
- [Mimari Yapı](#mimari-yapı)
- [İş Akışı](#i̇ş-akışı)
- [Desen Yönetim Sistemi](#desen-yönetim-sistemi)
- [Mimari Yönetişim Sistemi](#mimari-yönetişim-sistemi)
- [Entegrasyon Sistemi](#entegrasyon-sistemi)
- [Yol Haritası Yönetimi](#yol-haritası-yönetimi)
- [Hata Yönetim Sistemi](#hata-yönetim-sistemi)
- [Kurulum ve Kullanım](#kurulum-ve-kullanım)
- [Kullanım Kılavuzu](#kullanım-kılavuzu)
- [Metrikler ve Analitik](#metrikler-ve-analitik)
- [Görselleştirme Araçları](#görselleştirme-araçları)
- [Gelişmiş Yapılandırma](#gelişmiş-yapılandırma)
- [Sık Sorulan Sorular](#sık-sorulan-sorular)
- [Çözümlenen Sorunlar](#çözümlenen-sorunlar)
- [Dökümantasyon](#dökümantasyon)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [İletişim](#i̇letişim)

## Proje Tanımı

CodeFlow, yapay zeka destekli proje geliştirme süreçlerinizi otomatize eden kapsamlı bir iş akış motorudur. Gelişmiş desen yönetimi, mimari yönetişim, entegrasyon mükemmelliği, yol haritası zekası ve hata yönetimi mekanizmaları sayesinde kod kalitesini ve sürdürülebilirliğini önemli ölçüde artırır.

**Neden CodeFlow?**

* **Tutarlılık:** Ekip genelinde kod kalitesini ve mimari bütünlüğü standartlaştırır
* **Verimlilik:** Rutin görevleri otomatikleştirir ve geliştirme süreçlerini hızlandırır
* **Öngörülebilirlik:** Proje planlamasını ve risk değerlendirmesini iyileştirir
* **Ölçeklenebilirlik:** Büyüyen kodtabanlarını ve ekipleri destekler
* **Sürdürülebilirlik:** Teknik borcun proaktif yönetimini sağlar

CodeFlow, iş akış motoru (codeflow.xml) ve iş akış talebi (prompt.xml) olmak üzere iki temel XML dosyasıyla çalışır. Bu dosyalar, proje geliştirme sürecinin otomatizasyonunu ve optimizasyonunu sağlayan kapsamlı bir çerçeve sunar.

### Ana Bileşenler

* **Desen Yönetim Motoru:** Kod desenlerini otomatik olarak tespit eder, sınıflandırır ve geliştirir
* **Mimari Doğrulama Sistemi:** Kod tabanının mimari ilkelere uyumunu sağlar
* **Entegrasyon Orkestratörü:** Çok seviyeli entegrasyon testlerini yönetir
* **Hikaye ve Bağımlılık Yöneticisi:** Proje planlamasını optimize eder
* **Gelişmiş Hata Analiz Sistemi:** Çok boyutlu hata sınıflandırması ve akıllı kurtarma

Bu sistemler birlikte çalışarak modern yazılım geliştirme süreçlerinde kalite, tutarlılık ve verimlilik sağlar.

## Temel Özellikler

### 🔄 Entegre İş Akışı

CodeFlow, doğrulanmış bir ürün gereksinimleri dokümanı (PRD) temelinde proje geliştirmeyi otomatize eden kapsamlı bir iş akış motoru sunar:

* **Otomatik PRD doğrulama ve analizi**
* **Yapay zeka destekli mimari tasarım ve değerlendirme**
* **Akıllı modüler yapı ve bağımlılık yönetimi**
* **İteratif ve döngüsel geliştirme süreç otomasyonu**
* **Aşamalı entegrasyon ve test otomasyonu**
* **Kesintisiz geri bildirim ve sürekli iyileştirme**
* **Dinamik, gerçek zamanlı dökümantasyon sistemi**

```xml
<!-- Örnek İş Akışı Yapılandırması (codeflow.xml'den özet) -->
<workflow>
  <step id="analyze_initial_architecture">
    <description>Establish comprehensive, robust initial architecture based on the validated PRD through multi-dimensional analysis.</description>
    <condition>After PRD is ready and validated.</condition>
    <action>
      Use `architecture_analyzer` to perform comprehensive architecture analysis:
        - Evaluate multiple architecture styles/patterns against PRD requirements
        - Perform quantitative analysis of quality attributes
        - Generate architecture quality scores for each candidate approach
      <!-- ... diğer eylemler ... -->
    </action>
  </step>
  <!-- ... diğer adımlar ... -->
</workflow>
```

### 📊 Gelişmiş Desen Yönetimi

Desen yönetim sistemi, kod kalitesini ve tutarlılığını artırmak için kapsamlı özellikler sunar:

* **Otomatik desen tanıma ve sınıflandırma**
* **Desen uygulamalarının tutarlılık kontrolü**
* **Anti-desen tespiti ve çözüm önerileri**
* **Desen kataloğu ve evrim yönetimi**
* **Desen metriklerinin ölçümü ve analizi**
* **Desen ilişkilerinin haritalanması ve görselleştirilmesi**

```xml
<!-- Örnek Desen Metrik Tanımlaması (prompt.xml'den) -->
<pattern_metrics>
  <metric name="pattern_adoption_rate" 
         description="Percentage of relevant coding opportunities where a cataloged pattern was applied." 
         target=">75%"/>
  <metric name="pattern_effectiveness_score" 
         description="Aggregated score based on pattern impact on complexity, testability, performance." 
         target="increasing_trend"/>
  <!-- ... diğer metrikler ... -->
</pattern_metrics>
```

### 🏗️ Mimari Yönetişim

Mimari yönetişim sistemi, kodunuzun tanımlanmış mimari ilkelere ve kısıtlamalara uyumunu sağlar:

* **Otomatik mimari uygunluk kontrolü**
* **Bileşen sınırlarının ve sorumluluklarının korunması**
* **Mimari kaymanın erken tespiti ve önlenmesi**
* **Mimari belgelendirme ve görselleştirme**
* **Mimari kararlar ve gerekçelerin kaydedilmesi**
* **Kalite özelliklerinin (performans, güvenlik, vb.) değerlendirilmesi**

```xml
<!-- Örnek Mimari Analiz Mekanizması (prompt.xml'den) -->
<architecture_analysis_mechanism>
  <process>Perform multi-dimensional architecture analysis against PRD requirements, architectural principles, and industry standards.</process>
  <process>Validate structural integrity, modularity, component interactions, and alignment with business goals.</process>
  <!-- ... diğer süreçler ... -->
  <feedback_loop>Continuously monitor architecture conformance and update metrics to detect drift early.</feedback_loop>
</architecture_analysis_mechanism>
```

### 🔌 Gelişmiş Entegrasyon

Entegrasyon sistemi, kodunuzun farklı bileşenlerinin sorunsuz bir şekilde birlikte çalışmasını sağlar:

* **Çok seviyeli entegrasyon test otomasyonu**
* **Arayüz sözleşme doğrulama ve sınır koşulu kontrolü**
* **Bileşen uyumluluk matrisi ve analizi**
* **Entegrasyon kararlılık metriklerinin izlenmesi**
* **Otomatik geri alma ve kurtarma mekanizmaları**
* **Entegrasyon hata modellerinin analizi**

### 🗺️ Akıllı Yol Haritası Yönetimi

Yol haritası yönetim sistemi, projenizin planlamasını ve izlenmesini geliştirir:

* **Gereksinimlerin hiyerarşik olarak yapılandırılmış hikayelere dönüştürülmesi**
* **Çok boyutlu önceliklendirme algoritmaları**
* **Detaylı bağımlılık analizi ve kritik yol hesaplaması**
* **Akıllı iş yükü dağılımı ve kapasite planlaması**
* **Tahmine dayalı analitik ve risk değerlendirmesi**
* **İnteraktif yol haritası görselleştirme ve izleme**

### 🛡️ İleri Düzey Hata Yönetimi

Hata yönetim sistemi, hatalarınızı etkili bir şekilde analiz eder, sınıflandırır ve çözer:

* **Çok boyutlu hata sınıflandırma taksonomisi**
* **Otomatik kök neden analizi ve etki değerlendirmesi**
* **Bağlam duyarlı kurtarma stratejisi seçimi**
* **İşlem tabanlı kurtarma ve veri bütünlüğü doğrulama**
* **Hata trendleri ve desenleri analizi**
* **Kademeli kurtarma işlemleri ve doğrulama kontrol noktaları**

```xml
<!-- Örnek Hata Analiz Mekanizması (prompt.xml'den) -->
<error_analysis_mechanism>
  <process>Perform multi-dimensional error classification using sophisticated taxonomies.</process>
  <process>Generate detailed fault trees with root cause determination and impact assessment.</process>
  <!-- ... diğer süreçler ... -->
  <feedback_loop>Continuously update error metrics to improve error handling capabilities over time.</feedback_loop>
</error_analysis_mechanism>
```

## Kurulum ve Kullanım

### Ön Koşullar

- Herhangi bir LLM (GPT-4, Claude, Llama vb.)
- codeflow.xml ve prompt.xml dosyaları

### Kurulum

CodeFlow, bir yazılım paketi olarak kurulmaz. XML dosyaları (codeflow.xml ve prompt.xml) doğrudan LLM'e sistem promptu olarak sunulur:

1. codeflow.xml ve prompt.xml dosyalarını indirin
2. Tercih ettiğiniz LLM platformuna sistem promptu olarak entegre edin
3. Çalışma akışınızı başlatmak için uygun talimatlar ile LLM'i çağırın

### Temel Kullanım

```
# LLM'e temel iş akışı talimatını gönderin
"[codeflow.xml içeriği] Lütfen bu iş akışını başlat ve PRD'yi analiz et."

# Belirli bir adıma odaklanmak için
"[codeflow.xml içeriği] Bu iş akışının 'analyze_initial_architecture' adımını çalıştır."

# Durum raporu almak için
"[codeflow.xml içeriği] Mevcut mimari durumu analiz et ve rapor oluştur."

# Hata analizi istemek için
"[codeflow.xml içeriği] Bu projede potansiyel hataları analiz et."

# Dökümantasyon durumunu kontrol etmek için
"[codeflow.xml içeriği] Projenin dökümantasyon kalitesini ve güncelliğini değerlendir."

# Bir hikayenin tamamlanma durumunu kontrol etmek için
"[codeflow.xml içeriği] 'story_123' kodlu hikayenin durumunu ve bağımlılıklarını kontrol et."

# Proje ilerlemesini özetlemek için
"[codeflow.xml içeriği] Mevcut iterasyonun ilerlemesini özet rapor olarak sun."

# Sonraki adımları planlamak için
"[codeflow.xml içeriği] Bir sonraki adımı planla ve önceliklendir."

# Çevik süreç için planlama toplantısı desteği
"[codeflow.xml içeriği] Bir sonraki sprint için hikaye önerilerini önceliklendir ve bağımlılıklarını analiz et."
```

### Gelişmiş Kullanım

```
# PRD'den yeni mimari analiz talep etme
"[codeflow.xml içeriği] [prompt.xml içeriği] [PRD içeriği] Bu PRD'yi kullanarak kapsamlı bir mimari analiz oluştur."

# Desen analizi talep etme
"[codeflow.xml içeriği] [kod içeriği] Bu kod tabanında kullanılan desenleri tespit et ve analiz et."

# Mimari doğrulama
"[codeflow.xml içeriği] [kod içeriği] Bu kod tabanının tanımlanan mimari prensiplerle uyumluluğunu doğrula."

# Entegrasyon analizi
"[codeflow.xml içeriği] [bileşen tanımları] Bu bileşenlerin entegrasyon uyumluluğunu analiz et."

# Desen önerisi
"[codeflow.xml içeriği] [kod içeriği] Bu kod için uygun tasarım desenleri öner ve uygulanabilirliğini açıkla."

# Dökümantasyon oluşturma ve güncelleme
"[codeflow.xml içeriği] [kod içeriği] Bu kod için otomatik API dökümantasyonu oluştur ve mevcut dökümantasyonu güncelle."

# Karmaşık mimari dönüşüm planlama
"[codeflow.xml içeriği] [kod içeriği] [hedef mimari] Bu kod tabanını hedef mimariye dönüştürmek için aşamalı bir plan oluştur."

# Teknik borç analizi ve refaktörleme planı
"[codeflow.xml içeriği] [kod içeriği] Bu kod tabanındaki teknik borcu analiz et ve önceliklendirilmiş refaktörleme önerileri sun."

# Güvenlik analizi ve iyileştirme
"[codeflow.xml içeriği] [kod içeriği] Bu kod tabanındaki güvenlik açıklarını analiz et ve iyileştirme önerileri sun."

# Performans optimizasyonu
"[codeflow.xml içeriği] [kod içeriği] [performans gereksinimleri] Bu kod tabanını belirtilen performans gereksinimlerini karşılayacak şekilde optimize etmek için öneriler sun."

# Birim testleri ve test kapsamı analizi
"[codeflow.xml içeriği] [kod içeriği] Bu kod tabanı için birim test kapsamını analiz et ve eksik alanlar için test önerileri sun."

# Anti-desen tespiti ve düzeltme
"[codeflow.xml içeriği] [kod içeriği] Bu kod tabanındaki anti-desenleri tespit et ve düzeltme önerileri sun."

# Bağımlılık optimizasyonu
"[codeflow.xml içeriği] [kod içeriği] Bu kod tabanındaki bağımlılıkları analiz et ve optimizasyon önerileri sun."

# Kod tutarlılığı ve standartlar doğrulama
"[codeflow.xml içeriği] [kod içeriği] [kodlama standartları] Bu kod tabanını belirtilen kodlama standartlarına göre doğrula ve iyileştirme önerileri sun."
```

## Metrikler ve Analitik

CodeFlow, LLM'in projenizdeki çeşitli metrikleri analiz etmesini sağlayan kavramsal bir çerçeve sunar:

### Metrik Panoları

* **Desen Metrikleri:** Desen benimseme oranı, etkinlik ve tutarlılık puanları
* **Mimari Metrikleri:** Uygunluk puanı, bağlaşım, uyum ve teknik borç göstergeleri
* **Entegrasyon Metrikleri:** Entegrasyon kararlılık endeksi, kapsam ve bileşen uyumluluk puanları
* **Yol Haritası Metrikleri:** Gereksinim kapsam oranı, hikaye kalite puanı, tahmin doğruluğu
* **Hata Metrikleri:** Hata çözüm oranı, sınıflandırma doğruluğu, kök neden belirleme oranı

Bu metrikler, LLM'in projenizin sağlığı ve gelişimi hakkında kapsamlı analiz yapmasını sağlar.

```
# LLM'den tüm metrikleri analiz etmesini isteyin
"[codeflow.xml içeriği] [kod içeriği] Bu projenin tüm metriklerini analiz et ve rapor oluştur."

# Belirli bir metrik grubunu analiz etmesini isteyin
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki desen metriklerini analiz et."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki mimari metrikleri analiz et."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki entegrasyon metriklerini analiz et."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki yol haritası metriklerini analiz et."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki hata metriklerini analiz et."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki dökümantasyon metriklerini analiz et."

# Metrik trendlerini analiz etme
"[codeflow.xml içeriği] [kod içeriği] [önceki metrikler] Bu projedeki metrik trendlerini son iki iterasyon için analiz et ve gelişim alanlarını belirle."

# Belirli bir kalite özelliği için metrik analizi
"[codeflow.xml içeriği] [kod içeriği] Bu projenin bakım yapılabilirlik metriklerini analiz et ve iyileştirme önerileri sun."
"[codeflow.xml içeriği] [kod içeriği] Bu projenin güvenlik metriklerini analiz et ve iyileştirme önerileri sun."
"[codeflow.xml içeriği] [kod içeriği] Bu projenin performans metriklerini analiz et ve optimizasyon önerileri sun."
"[codeflow.xml içeriği] [kod içeriği] Bu projenin ölçeklenebilirlik metriklerini analiz et ve darboğazları belirle."

# Bileşen seviyesinde metrik analizi
"[codeflow.xml içeriği] [kod içeriği] 'authentication' bileşeninin tüm metriklerini analiz et ve iyileştirme önerileri sun."

# Ekip performans metrikleri
"[codeflow.xml içeriği] [hikaye tamamlanma verileri] Ekip hızı, tahmin doğruluğu ve teslim kalitesi metriklerini analiz et."

# Dökümantasyon kalite metrikleri
"[codeflow.xml içeriği] [dökümantasyon içeriği] Dökümantasyon kalite puanı, kapsam oranı ve güncellik indeksi metriklerini hesapla."

# Entegrasyon kararlılık metrikleri
"[codeflow.xml içeriği] [entegrasyon test sonuçları] Entegrasyon kararlılık endeksi, başarısız test oranı ve entegrasyon kapsam metriklerini analiz et."
```

## Görselleştirme Araçları

CodeFlow, LLM'in projenizin durumunu çeşitli görselleştirmelerle açıklamasını sağlar:

### İnteraktif Açıklamalar

* **Desen Isı Haritası:** Modüllere göre desen kullanım frekansı dağılımı açıklamaları
* **Mimari Sağlık Analizi:** Bileşen düzeyinde mimari uygunluk değerlendirmeleri
* **Bağlaşım Ağı:** Bileşenler arası bağımlılık ilişkileri açıklamaları
* **Entegrasyon Uyumluluk Analizi:** Bileşenlerin birbirleriyle uyumluluk değerlendirmesi
* **Yol Haritası Analizi:** Proje dönüm noktaları ve iterasyon ilerleme durumu değerlendirmesi
* **Hata Trendi Analizi:** Hata türleri ve çözüm etkinliği trendleri açıklamaları

```
# LLM'den tüm görselleştirme açıklamalarını oluşturmasını isteyin
"[codeflow.xml içeriği] [kod içeriği] Bu projede tüm görselleştirme analizlerini metin olarak açıkla."

# Belirli görselleştirme açıklamaları oluşturmasını isteyin
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki desen dağılımını açıkla."
"[codeflow.xml içeriği] [kod içeriği] Bu projenin mimari sağlık analizini yap."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki bileşenler arası bağımlılıkları açıkla."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki yol haritasını analiz et."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki hata trendlerini analiz et."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki dökümantasyon kalite dağılımını görselleştir."

# Karmaşık mimari görselleştirmeler
"[codeflow.xml içeriği] [kod içeriği] Bu projenin bileşen etkileşim diyagramını açıkla."
"[codeflow.xml içeriği] [kod içeriği] Bu projenin veri akış diyagramını açıkla."
"[codeflow.xml içeriği] [kod içeriği] Bu projenin dağıtım mimarisi diyagramını açıkla."

# Desen ilişki görselleştirmeleri
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki desenlerin birbiriyle ilişkilerini ağ diyagramı şeklinde açıkla."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki desenlerden en sık birlikte kullanılanları gruplayarak açıkla."

# Teknik borç görselleştirmeleri
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki teknik borç dağılımını ısı haritası şeklinde açıkla."
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki teknik borç birikimini zaman içinde gösteren bir trend analizi sağla."

# Performans görselleştirmeleri
"[codeflow.xml içeriği] [kod içeriği] [performans test sonuçları] Bu projenin performans darboğazlarını gösteren bir analiz açıkla."

# Risk görselleştirmeleri
"[codeflow.xml içeriği] [kod içeriği] Bu projedeki risk faktörlerini etki ve olasılık matrisinde konumlandır."

# İlerleme görselleştirmeleri
"[codeflow.xml içeriği] [hikaye durumları] Bu projenin tamamlanma durumunu burndown grafik açıklaması şeklinde sun."
"[codeflow.xml içeriği] [hikaye durumları] Bu projenin hız metriklerini zaman içinde gösteren bir analiz sağla."

# Dökümantasyon kapsam görselleştirmeleri
"[codeflow.xml içeriği] [kod içeriği] [dökümantasyon içeriği] Kod tabanının dökümantasyon kapsamını gösteren ısı haritasını açıkla."
"[codeflow.xml içeriği] [dökümantasyon içeriği] Dökümantasyon bölümleri arasındaki ilişkileri ağ diyagramı olarak açıkla."

# Kullanıcı yolculuğu görselleştirmeleri
"[codeflow.xml içeriği] [API kullanım verisi] API kullanım akışını ve yaygın kullanıcı yolculuklarını görselleştir."
```

## Sık Sorulan Sorular

### Genel Sorular

**S: CodeFlow hangi LLM'ler ile kullanılabilir?**
C: CodeFlow, GPT-4, Claude, Llama gibi ileri düzey LLM'lerle kullanılabilir. Karmaşık XML yapıları ve talimatları anlayabilen herhangi bir LLM ile uyumludur.

**S: CodeFlow mevcut bir projeye nasıl entegre edilir?**
C: CodeFlow XML dosyalarını, kod tabanınızın içeriğiyle birlikte LLM'e sistem promptu olarak sunarak mevcut projelere entegre edebilirsiniz.

**S: XML dosyalarını manuel olarak düzenlemem gerekiyor mu?**
C: Evet, CodeFlow XML dosyalarını projenizin ihtiyaçlarına göre düzenleyebilirsiniz. XML şemalarını takip ederek dosyaları manuel olarak özelleştirebilirsiniz veya bir LLM'den bu konuda yardım alabilirsiniz.

### Teknik Sorular

**S: CodeFlow ne kadar ölçeklenebilir?**
C: CodeFlow'un ölçeklenebilirliği, kullandığınız LLM'in kapasitesine bağlıdır. Modern LLM'ler, büyük kod tabanlarında bile etkili analiz sağlayabilir, ancak çok büyük projelerde kodu parçalara bölmek gerekebilir.

**S: Özel desenler ve metrikleri nasıl tanımlarım?**
C: codeflow.xml ve prompt.xml dosyalarında özel desen tanımları ve metrikler ekleyebilirsiniz. XML şemasına uygun olarak yeni tanımlamalar yapabilirsiniz.

**S: CodeFlow LLM API'lerine nasıl entegre edilir?**
C: CodeFlow XML dosyalarını API çağrılarınızın sistem promptu bölümüne entegre edebilirsiniz. OpenAI API, Anthropic API veya Hugging Face API gibi LLM servislerini kullanarak entegrasyon sağlayabilirsiniz.

## Katkıda Bulunma

Projeye katkıda bulunmak için lütfen şu adımları izleyin:

1. Repoyu forklayın
2. Özellik dalınızı oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Dalınıza push yapın (`git push origin feature/amazing-feature`)
5. Bir Pull Request açın

Katkı sağlamadan önce lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## İletişim

Proje Ekibi - [proje@codeflow.io](mailto:proje@codeflow.io)

Proje Linki: [https://github.com/yunusgungor/codeflow](https://github.com/yunusgungor/codeflow)

---

<p align="center">
  <img src="https://via.placeholder.com/800x100/0073CF/FFFFFF?text=CodeFlow+%C4%B0%C5%9F+Ak%C4%B1%C5%9F+Motoru+v1.0" alt="CodeFlow Footer Logo">
</p>

<p align="center">
  <a href="#temel-özellikler">Özellikler</a> •
  <a href="#mimari-yapı">Mimari</a> •
  <a href="#i̇ş-akışı">İş Akışı</a> •
  <a href="#kurulum-ve-kullanım">Kurulum</a> •
  <a href="#katkıda-bulunma">Katkıda Bulunma</a> •
  <a href="#lisans">Lisans</a>
</p>

## Dökümantasyon

### 📚 Dinamik Dökümantasyon Sistemi

CodeFlow, gerçek zamanlı ve sürekli gelişen bir dökümantasyon sistemi sunar:

* **Kod değişikliklerini gerçek zamanlı izleme ve belgelendirme**
* **Otomatik API dökümantasyonu oluşturma ve güncelleme**
* **Kod-dökümantasyon senkronizasyonu ve tutarlılık kontrolü**
* **Akıllı çapraz referanslama ve semantik bağlantılar**
* **Dökümantasyon kalitesi ve güncelliği metrikleri**
* **İnteraktif API keşif araçları ve çalıştırılabilir örnekler**

```xml
<!-- Örnek Dökümantasyon Analiz Mekanizması (prompt.xml'den) -->
<documentation_analysis_mechanism>
  <process>Monitor code changes in real-time to identify affected documentation sections using semantic analysis.</process>
  <process>Generate and update documentation automatically based on code structure, comments, and semantic understanding.</process>
  <process>Create intelligent cross-references between code, documentation, and other artifacts using semantic analysis.</process>
  <process>Validate documentation for accuracy, completeness, consistency, and adherence to templates.</process>
  <!-- ... diğer süreçler ... -->
  <feedback_loop>Continuously update documentation quality metrics and usage analytics to drive improvement.</feedback_loop>
</documentation_analysis_mechanism>
```

### Dökümantasyon Mimarisi

CodeFlow'un dökümantasyon sistemi, `.project_meta/.docs/` dizini altında organizeli bir yapı sunar:

* **Ana Sayfa:** index.md - Dökümantasyon portali ve ana giriş noktası
* **API Dökümantasyonu:** api/ dizini - Otomatik oluşturulan API referansları, uç noktalar, modeller
* **Mimari Dökümantasyonu:** architecture/ dizini - Sistem mimarisi, bileşenler, veri akışı
* **Kullanım Kılavuzları:** guides/ dizini - Son kullanıcı, geliştirici ve dağıtım kılavuzları
* **Eğitimler:** tutorials/ dizini - Adım adım öğreticiler ve çalıştırılabilir örnekler
* **Bakım Belgeleri:** maintenance/ dizini - Sorun giderme, izleme, performans ayarlama kılavuzları
* **Metrikler:** metrics/ dizini - Dökümantasyon kalite metrikleri, kapsam raporları, kullanım analitiği
* **Sürümler:** versions/ dizini - Dökümantasyon sürüm geçmişi ve değişiklik günlükleri
* **İnteraktif İçerik:** interactive/ dizini - İnteraktif diyagramlar, API explorer, örnek kod çalıştırıcı
* **Doğrulama:** validation/ dizini - Dökümantasyon doğrulama raporları, tutarlılık kontrolü
* **Arama:** search/ dizini - Dökümantasyon arama indeksi ve anlamsal arama verisi
* **Şablonlar:** templates/ dizini - Standardize edilmiş dökümantasyon şablonları

### Gelişmiş Dökümantasyon Araçları

CodeFlow, kapsamlı dökümantasyon oluşturma ve yönetimi için gelişmiş araçlar sunar:

* **doc_generator:** Kod değişikliklerini analiz ederek otomatik dökümantasyon oluşturur
* **doc_watcher:** Gerçek zamanlı kod değişikliklerini izleyerek dökümantasyonun güncelliğini değerlendirir
* **doc_validator:** Dökümantasyonun doğruluğunu, tutarlılığını ve eksiksizliğini değerlendirir
* **doc_reference_analyzer:** Semantik anlama ile ilgili içeriği otomatik tespit eder ve bağlantılar oluşturur
* **doc_analytics:** Dökümantasyon kullanımını izler ve iyileştirme önerileri sunar

### Dökümantasyon Metrikleri

Dökümantasyon kalitesini ölçmek ve sürekli iyileştirmek için kapsamlı metrikler:

* **Dökümantasyon Kalite Puanı:** Dökümantasyonun doğruluk, eksiksizlik ve netlik ölçümü
* **Dökümantasyon Kapsam Oranı:** Kod/özellik/API'lerin uygun dökümantasyonla kapsanma yüzdesi
* **Dökümantasyon Güncellik İndeksi:** Dökümantasyonun kod değişikliklerine göre güncellik ölçüsü
* **Çapraz Referans Bütünlüğü:** Dökümantasyon referanslarının geçerlilik ve güncellik yüzdesi
* **Dökümantasyon Kullanım Oranı:** Dökümantasyon bölümleri arasındaki erişim/kullanım dağılımı
* **Dökümantasyon Arama Etkinliği:** İlgili sonuçları döndüren dökümantasyon aramalarının başarı oranı

```xml
<!-- Örnek Dökümantasyon Metrikleri (prompt.xml'den) -->
<documentation_metrics>
  <metric name="documentation_quality_score" 
         description="Dökümantasyon doğruluğu, eksiksizliği ve netliğini ölçen bileşik puan." 
         target=">85%"/>
  <metric name="documentation_freshness_index" 
         description="Dökümantasyonun kod değişikliklerine göre güncellik ölçüsü." 
         target=">80%"/>
  <!-- ... diğer metrikler ... -->
</documentation_metrics>
```

### İnteraktif Dökümantasyon

CodeFlow'un dinamik dökümantasyon sistemi, standart belgelerden daha fazlasını sunar:

* **İnteraktif API Keşif Aracı:** API'leri etkileşimli olarak test etme ve keşfetme
* **Çalıştırılabilir Kod Örnekleri:** Doğrudan dökümantasyon içinde çalıştırılabilen kod örnekleri
* **İnteraktif Diyagramlar:** Tıklanabilir bileşenler ve detay açılımları olan mimari diyagramlar
* **Sürüm Karşılaştırma:** Dökümantasyon sürümleri arasında görsel farklılık görüntüleme
* **Kontekst Duyarlı Arama:** Rol ve kullanım senaryolarına göre özelleştirilmiş arama sonuçları

### Dökümantasyon Sürekli İyileştirme

CodeFlow'un dökümantasyon sistemi sürekli olarak iyileştirilir:

* **Kalite Metrikleri İzleme:** Her iterasyonda dökümantasyon kalite puanını %5 artırma hedefi
* **Şablon ve Standart Geliştirme:** Kalite değerlendirmeleri ve kullanıcı geri bildirimleri temelli
* **Kullanım Analizi:** Yüksek değerli bölümleri ve az kullanılan dökümantasyonu tespit etme
* **Tarihsel Kalite Metrikleri:** Zaman içinde iyileştirme trendlerini izleme
* **Arama ve Çapraz Referanslama:** Arama analitiği ve kullanıcı geri bildirimi temelli geliştirme

### Dökümantasyon Sistemi Kullanımı

```
# Dökümantasyon kalite analizi
"[codeflow.xml içeriği] [kod içeriği] [mevcut dökümantasyon] Bu projede dökümantasyon kalitesini analiz et ve eksiklikleri tespit et."

# API dökümantasyonu oluşturma
"[codeflow.xml içeriği] [kod içeriği] Bu modül için otomatik API dökümantasyonu oluştur."

# Dökümantasyon çapraz referans analizi
"[codeflow.xml içeriği] [kod içeriği] [mevcut dökümantasyon] Kod ve dökümantasyon arasındaki çapraz referansları analiz et ve eksik bağlantıları tespit et."

# Dökümantasyon güncellik değerlendirmesi
"[codeflow.xml içeriği] [kod değişiklikleri] [mevcut dökümantasyon] Son kod değişikliklerine göre dökümantasyonun güncelliğini değerlendir ve güncellenmesi gereken bölümleri belirle."

# Kullanım kılavuzları oluşturma
"[codeflow.xml içeriği] [kod içeriği] Bu modül için son kullanıcı kılavuzu oluştur."

# Eğitim materyalleri oluşturma
"[codeflow.xml içeriği] [kod içeriği] Bu API için adım adım eğitim materyali ve kod örnekleri oluştur."

# İnteraktif API keşif aracı içeriği oluşturma
"[codeflow.xml içeriği] [API tanımları] Bu API'ler için interaktif keşif aracı içeriği oluştur."

# Mimari dökümantasyon oluşturma
"[codeflow.xml içeriği] [kod içeriği] Bu sistem için kapsamlı mimari dökümantasyon oluştur, bileşenler, bağlantılar ve veri akışı diyagramları dahil."

# Dökümantasyon şablonları oluşturma
"[codeflow.xml içeriği] [örnek dökümantasyon] Bu projenin stili ile uyumlu dökümantasyon şablonları oluştur."

# Dökümantasyon arama indeksi oluşturma
"[codeflow.xml içeriği] [mevcut dökümantasyon] Bu dökümantasyon için arama indeksi ve anahtar kelime analizi oluştur."