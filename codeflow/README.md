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

CodeFlow, yapay zeka destekli proje geliştirme süreçlerinizi otomatize eden kapsamlı bir iş akış motorudur. Sistemin kalbini oluşturan `codeflow.xml` (iş akış motoru) ve `prompt.xml` (iş akış yapılandırma ve yönlendirme) dosyaları aracılığıyla yönetilen CodeFlow, gelişmiş desen yönetimi, mimari yönetişim, entegrasyon mükemmelliği, yol haritası zekası ve hata yönetimi mekanizmaları sayesinde kod kalitesini ve sürdürülebilirliğini önemli ölçüde artırır. Tüm proje metrikleri, raporları, geçici dosyalar ve sistem tarafından üretilen diğer önemli çıktılar `.project_meta/` dizini altında yapılandırılmış bir şekilde saklanır.

**Neden CodeFlow?**

* **Tutarlılık:** Ekip genelinde kod kalitesini ve mimari bütünlüğü standartlaştırır
* **Verimlilik:** Rutin görevleri otomatikleştirir ve geliştirme süreçlerini hızlandırır
* **Öngörülebilirlik:** Proje planlamasını ve risk değerlendirmesini iyileştirir
* **Ölçeklenebilirlik:** Büyüyen kodtabanlarını ve ekipleri destekler
* **Sürdürülebilirlik:** Teknik borcun proaktif yönetimini sağlar

CodeFlow, proje geliştirme sürecinin otomasyonunu ve optimizasyonunu sağlayan kapsamlı bir çerçeve sunmak için birbirini tamamlayan iki temel XML dosyasıyla çalışır:
*   **`codeflow.xml`**: Ana iş akışlarını, adımlarını, kullanılacak araçları ve bu adımların sıralı mantığını tanımlar. Sistemin temel operasyonel motorudur.
*   **`prompt.xml`**: `codeflow.xml`'deki iş akışlarının belirli bir proje veya bağlam için nasıl çalışacağını detaylandıran, hedefleri, kalite standartlarını, beklenen çıktıları ve insan etkileşim noktalarını tanımlayan bir yapılandırma ve yönlendirme dosyasıdır. LLM'e verilen görevlerde `codeflow.xml` ile birlikte kullanılarak sürecin davranışını özelleştirir.

### Ana Bileşenler

CodeFlow'un temel yetenekleri, aşağıdaki gibi entegre çalışan ana bileşenler ve sistemler tarafından sağlanır:

* **İş Akışı Motoru (`codeflow.xml`)**: Proje yaşam döngüsündeki tüm adımları, koşulları ve eylemleri tanımlar. `initialize_project` gibi adımlarla proje için gerekli tüm başlangıç yapılandırmasını (örneğin, `.project_meta/` altındaki metrik ve rapor dizinlerini oluşturma) gerçekleştirir.
* **İş Akışı Yapılandırması (`prompt.xml`)**: Kalite beklentileri, hedeflenen iş akışı adımları, özel metrikler, raporlama formatları ve insan müdahalesi gerektiren durumlar gibi proje özelindeki girdileri tanımlar.
* **Desen Yönetim Motoru (`learn_patterns` adımı ve ilgili araçlar)**: Kod desenlerini otomatik olarak tespit eder, sınıflandırır, geliştirir, evrimini takip eder ve `.project_meta/.patterns/` altında metriklerini, geçmişini ve ilişkilerini yönetir.
* **Mimari Doğrulama Sistemi (`analyze_initial_architecture` adımı ve `architecture_analyzer` gibi araçlar)**: Kod tabanının mimari ilkelere uyumunu sağlar ve `.project_meta/.architecture/` altında ilgili metrik ve raporları üretir.
* **Entegrasyon Orkestratörü (`integration_phase` adımı ve ilgili araçlar)**: Çok seviyeli entegrasyon testlerini yönetir, uyumluluk analizleri yapar ve `.project_meta/.integration/` altında metrik ve raporları saklar.
* **Yol Haritası ve Hikaye Yöneticisi (`create_roadmap_and_stories` adımı ve `roadmap_manager` gibi araçlar)**: Proje planlamasını optimize eder, gereksinimleri hikayelere böler, bağımlılıkları yönetir ve `.project_meta/.stories/` altında metrikleri tutar.
* **Gelişmiş Hata Analiz ve Yönetim Sistemi (`handle_error` adımı ve `error_analyzer` gibi araçlar)**: Hataları çok boyutlu sınıflandırır, kök neden analizi yapar, akıllı kurtarma stratejileri sunar ve `.project_meta/.errors/` altında metrik ve raporları barındırır.
* **Dinamik Dökümantasyon Sistemi (çeşitli dokümantasyon araçları)**: Kod ve proje hakkında sürekli güncel kalan dokümanlar üretir, doğrular ve `.project_meta/.docs/` altında metriklerini, doğrulama sonuçlarını ve görselleştirmelerini yönetir.

Bu sistemler birlikte çalışarak modern yazılım geliştirme süreçlerinde kalite, tutarlılık ve verimlilik sağlar.

## Temel Özellikler

### 🔄 Entegre ve Yapılandırılabilir İş Akışı

CodeFlow, doğrulanmış bir ürün gereksinimleri dokümanı (PRD) ve `prompt.xml` içinde tanımlanan proje hedefleri temelinde proje geliştirmeyi otomatize eden kapsamlı bir iş akış motoru sunar. `codeflow.xml` ana süreci tanımlarken, `prompt.xml` bu sürecin özel ihtiyaçlara göre nasıl işleyeceğini (örneğin, hangi adımlara odaklanılacağı, kalite standartları, beklenen çıktılar) belirler.

* **Otomatik PRD doğrulama ve analizi (`validate_prd` adımı)**
* **`prompt.xml` ile yönlendirilen yapay zeka destekli mimari tasarım ve değerlendirme (`analyze_initial_architecture` adımı).** Çıktılar `.project_meta/.architecture/` altında saklanır.
* **Akıllı modüler yapı ve bağımlılık yönetimi (`create_roadmap_and_stories` adımı).** Çıktılar `.project_meta/.stories/` altında saklanır.
* **İteratif ve döngüsel geliştirme süreç otomasyonu.**
* **Aşamalı entegrasyon ve test otomasyonu (`integration_phase` adımı).** Çıktılar `.project_meta/.integration/` altında saklanır.
* **Hata yönetimi ve bilgi birikimi (`handle_error` adımı).** Çıktılar `.project_meta/.errors/` altında saklanır.
* **Kesintisiz geri bildirim ve sürekli iyileştirme.**
* **Dinamik, gerçek zamanlı dökümantasyon sistemi (dokümantasyon araçları).** Çıktılar `.project_meta/.docs/` altında saklanır.
* **Proje başlangıç yapılandırması (`initialize_project` adımı):** `.project_meta/` altında tüm gerekli metrik, rapor, ve çalışma dizinlerini oluşturur.

```xml
<!-- Örnek İş Akışı Adımı (codeflow.xml'den 'analyze_initial_architecture') -->
<step id="analyze_initial_architecture">
  <description>Establish comprehensive, robust initial architecture based on the validated PRD and prompt.xml specifications through multi-dimensional analysis. Relevant metrics and review files will be generated and validated under .project_meta/.architecture/.</description>
  <condition>After PRD is ready and validated, and project is initialized.</condition>
  <tools_setup>
    <tool name="architecture_analyzer"/>
    <tool name="prd_analyzer"/>
    <tool name="code_reviewer"/>
    <tool name="reporting_tool"/>
  </tools_setup>
  <action>
    <phase name="Architectural Analysis and Design">
      Use `architecture_analyzer` to perform comprehensive architecture analysis based on PRD and `prompt.xml` quality/performance targets:
        - Evaluate multiple architecture styles/patterns against PRD requirements.
        - Perform quantitative analysis of quality attributes (e.g., performance, scalability, security, maintainability).
        - Generate architecture quality scores for each candidate approach.
        - Identify potential risks and trade-offs for each option.
      Use `prd_analyzer` to ensure deep understanding of functional and non-functional requirements influencing architecture.
    </phase>
    <phase name="Documentation and Metric Generation">
      The `architecture_analyzer` and `reporting_tool` will generate and store detailed architectural documents, decision logs, and all relevant metrics (including those specified in `prompt.xml`) in `.project_meta/.architecture/architecture_metrics/` and `.project_meta/.architecture/reviews/`.
      All generated files will be validated for completeness and correctness.
    </phase>
  </action>
  <output>
    <item type="report" path=".project_meta/.architecture/main_architecture_document.json" description="Main architecture document detailing the chosen approach, rationale, and design."/>
    <item type="metrics" path=".project_meta/.architecture/architecture_metrics/" description="Directory containing all generated architectural metrics."/>
    <item type="reviews" path=".project_meta/.architecture/reviews/" description="Directory containing architectural review files and feedback."/>
    <item type="status" description="Architecture analysis complete. All metrics and reports are generated and validated in their respective .project_meta/architecture/ subdirectories."/>
  </output>
  <error_management>
    <handle error_type="prd_ambiguity" action="request_clarification_from_user"/>
    <handle error_type="tool_failure" action="retry_with_alternative_tool_or_log_error"/>
    <handle error_type="metric_generation_failure" action="log_error_and_proceed_with_available_data"/>
  </error_management>
</step>
```

### 📊 Gelişmiş Desen Yönetimi (`learn_patterns` Adımı)

`learn_patterns` adımı liderliğindeki desen yönetim sistemi, kod kalitesini ve tutarlılığını artırmak için kapsamlı ve iyileştirilmiş özellikler sunar. Bu adım, `pattern_learner` ve `pattern_analyzer` gibi daha spesifik araçlar kullanarak `.project_meta/.patterns/` altındaki tüm metrik, geçmiş, evrim ve ilişki dosyalarını günceller ve doğrular. Detaylı çıktılar, raporlama ve güçlendirilmiş izlenebilirlik sağlar.

* **Otomatik desen tanıma ve sınıflandırma (`pattern_analyzer`)**
* **Desen uygulamalarının tutarlılık kontrolü**
* **Anti-desen tespiti ve çözüm önerileri (`anti_pattern_detector`)**
* **Desen kataloğu ve evrim yönetimi (`pattern_learner`, çıktılar `.project_meta/.patterns/evolution/` altında)**
* **Kapsamlı desen metrikleri ölçümü ve analizi (çıktılar `.project_meta/.patterns/metrics/` altında)**
* **Desen ilişkilerinin haritalanması ve görselleştirilmesi (çıktılar `.project_meta/.patterns/relationships/` altında)**
* **Desen öğrenme ve uygulama geçmişinin takibi (çıktılar `.project_meta/.patterns/history/` altında)**

```xml
<!-- Örnek Desen Metrik Tanımlaması (prompt.xml'den) -->
<pattern_metrics>
  <metric name="pattern_adoption_rate"
          description="Percentage of relevant coding opportunities where a cataloged pattern was applied. Stored in .project_meta/.patterns/metrics/pattern_adoption_rate.json"
          target=">75%"/>
  <metric name="pattern_effectiveness_score"
          description="Aggregated score based on pattern impact on complexity, testability, performance. Stored in .project_meta/.patterns/metrics/pattern_effectiveness_score.json"
          target="increasing_trend"/>
  <metric name="new_patterns_identified_count"
          description="Number of new, potentially reusable patterns identified in the last analysis cycle. Stored in .project_meta/.patterns/metrics/new_patterns_identified_count.json"
          target="monitor"/>
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
  <process>Perform multi-dimensional architecture analysis against PRD requirements, architectural principles, defined quality standards (e.g., target_latency_ms < 100), and industry standards. Generate reports in .project_meta/.architecture/reviews/.</process>
  <process>Validate structural integrity, modularity, component interactions, and alignment with business goals. Document findings in .project_meta/.architecture/main_architecture_document.json.</process>
  <output_expectation format="detailed_report" location=".project_meta/.architecture/architecture_assessment_report.json"/>
  <output_expectation format="metrics_dashboard_data" location=".project_meta/.architecture/architecture_metrics/dashboard_data.json"/>
  <feedback_loop>Continuously monitor architecture conformance via metrics in .project_meta/.architecture/architecture_metrics/ and update them to detect drift early. Report deviations against prompt.xml targets.</feedback_loop>
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
  <process>Perform multi-dimensional error classification using sophisticated taxonomies. Store classifications in .project_meta/.errors/classified_errors.json.</process>
  <process>Generate detailed fault trees with root cause determination and impact assessment. Reports in .project_meta/.errors/reports/fault_trees/.</process>
  <output_expectation format="error_dashboard_data" location=".project_meta/.errors/metrics/dashboard_data.json"/>
  <output_expectation format="knowledge_base_update_summary" location=".project_meta/.errors/knowledge_base_updates.json"/>
  <feedback_loop>Continuously update error metrics in .project_meta/.errors/metrics/ and visualizations in .project_meta/.errors/reports/visualizations/ to improve error handling capabilities over time.</feedback_loop>
</error_analysis_mechanism>
```

## Kurulum ve Kullanım

### Ön Koşullar

-   Erişiminiz olan bir Büyük Dil Modeli (LLM) (örn: GPT-4, Claude, Llama vb.). LLM, XML formatındaki `codeflow.xml` ve `prompt.xml` dosyalarını işleyebilmeli ve talimatları anlayabilmelidir.
-   `codeflow.xml` (iş akışı motoru) ve `prompt.xml` (proje özelinde iş akışı yapılandırması) dosyaları.
-   Analiz edilecek veya geliştirilecek proje kodları ve ilgili dokümanlar (örn: PRD).

### Kurulum

CodeFlow, geleneksel bir yazılım paketi gibi kurulmaz. `codeflow.xml` ve `prompt.xml` dosyaları, kullanılacak LLM'e doğrudan bir sistem mesajı (system prompt) veya kullanıcı mesajının bir parçası olarak sunulur:

1.  CodeFlow deposundan (veya size sağlanan yerden) `codeflow.xml` ve `prompt.xml` dosyalarının en güncel sürümlerini edinin.
2.  Bu XML dosyalarının içeriğini kopyalayın.
3.  Kullandığınız LLM platformunda, bu içerikleri LLM'e talimatlarınızla birlikte verin. Genellikle, `codeflow.xml` ve `prompt.xml` içerikleri, asıl sorgunuzdan önce veya özel bir sistem mesajı alanına yerleştirilir.
    *   **Örnek:** "Aşağıdaki `codeflow.xml` ve `prompt.xml` içeriklerini kullanarak, sağladığım PRD dokümanını analiz et ve bir başlangıç mimari planı oluştur. Çıktıları `.project_meta/` altında belirtilen yapıya göre kaydet.
        
        `<codeflow_xml_icerigi_buraya_gelecek>`
        
        `<prompt_xml_icerigi_buraya_gelecek>`
        
        `<PRD_icerigi_veya_referansi_buraya_gelecek>`"
4.  LLM, bu XML'leri yorumlayarak tanımlanmış iş akışını ve yapılandırmayı anlayacak ve talimatlarınızı bu çerçevede yerine getirecektir.

CodeFlow, tüm çıktılarını ve çalışma dosyalarını projenizin kök dizininde oluşturacağı `.project_meta/` adlı bir klasör altında saklar. Bu nedenle, LLM'in bu dizine yazma ve okuma yetkisi olduğundan emin olun (eğer LLM yerel bir dosya sistemiyle etkileşimde bulunuyorsa).

### Temel Kullanım Senaryoları

LLM'e verilecek talimatlar genellikle `codeflow.xml` içeriği, `prompt.xml` içeriği (isteğe bağlı olarak hedeflenmiş kullanım için) ve spesifik bir görevi içerir.

```plaintext
# Projeyi başlatma ve PRD analizi (prompt.xml'deki varsayılan ayarlar kullanılır)
"[codeflow.xml] [prompt.xml] Bu PRD'yi analiz et ve projeyi ilklendir. Raporları .project_meta/architecture/ altına kaydet.
[PRD İçeriği]"

# Belirli bir adıma odaklanma (örneğin, desenleri öğrenme)
"[codeflow.xml] [prompt.xml] Bu kod tabanındaki desenleri öğren ve .project_meta/patterns/ altına kaydet.
[Kod Örnekleri veya Kod Tabanı Referansı]"

# Mevcut proje durumu ve metrikleri hakkında genel bir rapor alma
"[codeflow.xml] [prompt.xml] Projenin mevcut durumunu analiz et ve .project_meta/ altındaki metriklerden genel bir özet raporu oluştur."

# Spesifik bir metrik setini sorgulama
"[codeflow.xml] [prompt.xml] Entegrasyon metriklerini (.project_meta/integration/metrics/) ve hata trendlerini (.project_meta/errors/metrics/trend_analysis.json) raporla."

# Hata analizi istemek
"[codeflow.xml] [prompt.xml] Belirtilen log dosyalarındaki potansiyel hataları analiz et ve .project_meta/errors/ altına raporla.
[Log Dosyası İçeriği veya Referansı]"

# Dokümantasyon kalitesini ve güncelliğini kontrol etme
"[codeflow.xml] [prompt.xml] Projenin dokümantasyon kalitesini (.project_meta/docs/metrics/) değerlendir ve eksiklikleri raporla."

# Bir kullanıcı hikayesinin tamamlanma durumunu ve bağımlılıklarını kontrol etme
"[codeflow.xml] [prompt.xml] 'story_id_xyz' numaralı hikayenin durumunu, bağımlılıklarını ve .project_meta/stories/ altındaki ilgili metriklerini kontrol et."

# İnsan etkileşim noktası hakkında bilgi alma veya tetiklenme durumunda ne yapılacağını sorma
"[codeflow.xml] [prompt.xml] 'critical_anti_pattern_detected' olayı için tanımlanmış insan etkileşim noktasının detayları nelerdir? Tetiklenmesi durumunda ne gibi bildirimler alırım?"

# Bir sonraki iterasyon için planlama desteği
"[codeflow.xml] [prompt.xml] .project_meta/stories/product_backlog.json dosyasındaki hikayeleri önceliklendir, bağımlılıklarını analiz et ve bir sonraki sprint için öneriler sun."
```

### Gelişmiş Kullanım Senaryoları

```plaintext
# PRD'den yeni ve kapsamlı bir mimari analiz talep etme (prompt.xml'de özel hedefler belirtilmiş olabilir)
"[codeflow.xml] [prompt.xml] [PRD içeriği] Bu PRD'yi kullanarak, prompt.xml'deki performans ve güvenlik hedeflerini göz önünde bulundurarak kapsamlı bir mimari analiz oluştur. Çıktıları .project_meta/architecture/ altına detaylı olarak kaydet."

# Kod tabanında desen analizi ve yeni desen önerileri isteme
"[codeflow.xml] [prompt.xml] [kod içeriği veya referansı] Bu kod tabanında kullanılan desenleri tespit et, .project_meta/patterns/ altında analiz et ve kodun iyileştirilmesi için yeni desenler öner."

# Mimari doğrulama ve sapma analizi
"[codeflow.xml] [prompt.xml] [kod içeriği veya referansı] Bu kod tabanının .project_meta/architecture/main_architecture_document.json adresindeki tanımlanmış mimari prensiplerle uyumluluğunu doğrula. Sapmaları ve olası riskleri raporla."

# Entegrasyon uyumluluk analizi ve test senaryosu üretimi
"[codeflow.xml] [prompt.xml] [bileşen tanımları ve arayüzleri] Bu bileşenlerin entegrasyon uyumluluğunu analiz et, olası çakışmaları belirt ve .project_meta/integration/reports/ altına bir test senaryosu öneri listesi oluştur."

# Otomatik dokümantasyon oluşturma, güncelleme ve doğrulama
"[codeflow.xml] [prompt.xml] [kod içeriği veya referansı] Bu kod için API dokümantasyonunu .project_meta/docs/api/ altında oluştur, mevcut dokümantasyonla (.project_meta/docs/existing/) karşılaştırarak güncelle ve .project_meta/docs/validation/ altında doğrulama raporu üret."

# Performans testi sonuçlarını analiz etme ve darboğazları belirleme
"[codeflow.xml] [prompt.xml] [performans testi ham sonuçları] Bu performans testi sonuçlarını analiz et, darboğazları belirle ve .project_meta/.integration/reports/performance_tests/performance_analysis_report.json olarak kaydet."

# Güvenlik tarama raporunu inceleme ve kritik açıkları önceliklendirme
"[codeflow.xml] [prompt.xml] [güvenlik tarama raporu] Bu güvenlik raporunu incele, OWASP Top 10'a göre kritik açıkları .project_meta/.integration/reports/security_tests/critical_vulnerabilities.json altında önceliklendir."

# Belirli bir modül için desen evrimini ve teknik borcu analiz etme
"[codeflow.xml] [prompt.xml] 'user_management_module' için .project_meta/patterns/evolution/ ve .project_meta/patterns/metrics/ altındaki verileri kullanarak desen evrimini ve biriken teknik borcu analiz et."
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