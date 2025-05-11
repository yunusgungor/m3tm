# M3TM - Mobil Multi-Modal Modüler Transformer

M³TM, kullanıcının kişisel verileriyle tamamen cihaz üzerinde eğitilebilen, semantik arama yapabilen ve kullanıcıya verilerini indirme imkanı sunan gizlilik odaklı bir yapay zeka modelidir.

## SDK Dokümantasyonu

M³TM SDK, geliştiricilerin Android ve iOS uygulamalarında M³TM modelini kolayca entegre etmelerini ve kullanmalarını sağlar. SDK, modelin tüm özelliklerine erişim için tutarlı bir API sunar.

### SDK Özellikleri

- **Çoklu Platform Desteği**: Hem Android hem de iOS platformları için tam destek
- **Multi-Modal İşleme**: Metin ve görüntü verilerini işleme yeteneği
- **Cihaz Üzerinde Eğitim**: Model adaptörleri ve görev başlıklarını kullanıcının verilerine göre kişiselleştirme
- **Semantik Arama**: Kullanıcının verileri üzerinde anlamsal arama yapabilme
- **Veri Gizliliği**: Tüm işlemlerin tamamen cihaz üzerinde gerçekleştirilmesi
- **Veri İndirme**: Kullanıcının kişisel verilerini dışa aktarma imkanı

## Başlarken

- [Kurulum Kılavuzu](./guides/kurulum_kilavuzu.md)
- [Hızlı Başlangıç](./tutorials/quickstart.md)

## API Referansı

- [API Referansı](./api/api_reference.md)
  - [Android API Referansı](./api/endpoints/android_api_reference.md)
  - [iOS API Referansı](./api/endpoints/ios_api_reference.md)
  - [Hata Kodları ve Çözümleri](./api/endpoints/error_codes.md)

## Kullanım Kılavuzları

- [Modelin Yüklenmesi ve Çıkarım Yapma](./guides/model_yukleme_ve_cikarim.md)
- [Görev Başlıkları ve Adaptörler](./guides/gorev_basliklari_ve_adaptorler.md)
- [Semantik Arama Kullanımı](./guides/semantik_arama.md)
- [Veri Dışa Aktarma İşlemleri](./guides/veri_disa_aktarma.md)
- [Hata Yönetimi](./guides/hata_yonetimi.md)

## Örnekler

- [Örnek Kodlar](./tutorials/code_samples/README.md)
  - [Android (Java) Örnekleri](./api/usage_examples/M3TM_Java_Usage_Example.md)
  - [iOS (Swift) Örnekleri](./api/usage_examples/M3TM_Swift_Usage_Example.md)
- [Demo Uygulamalar](./tutorials/demo_uygulamalar.md)

## En İyi Uygulamalar

- [Performans Optimizasyonu](./maintenance/performance_tuning.md)
- [Bellek Yönetimi](./maintenance/memory_management.md)
- [Batarya Etkinliği](./maintenance/battery_efficiency.md)
- [Güvenlik Önerileri](./maintenance/security_recommendations.md)

## SSS

- [Sık Sorulan Sorular](./maintenance/troubleshooting.md#sss)

## Sorun Giderme

- [Yaygın Sorunlar ve Çözümleri](./maintenance/troubleshooting.md)
- [Destek Alma](./maintenance/troubleshooting.md#destek)

## Sürüm Notları

- [Sürüm Geçmişi](./versions/changelog.md)

## İletişim ve Topluluk

- [GitHub Repo](https://github.com/m3tm/mobile-sdk)
- [Sorun Bildirme](https://github.com/m3tm/mobile-sdk/issues)
- [Katkıda Bulunma Rehberi](./guides/contribution_guide.md)

# M³TM v2.3 Dokümantasyon

**Versiyon:** 1.0  
**Oluşturma Tarihi:** 23 Mayıs 2024

## Genel Bakış

M³TM v2.3 (Mobil Multi-Modal Modüler Transformer), tamamen cihaz üzerinde çalışan, gizlilik odaklı bir yapay zeka modelidir. Bu dokümantasyon, M³TM v2.3 projesinin tüm bileşenlerini, mimari kararlarını ve kullanım şekillerini kapsar.

## İçerik

### 1. Mimari
- [Mimari Genel Bakış](./architecture/overview.md)
- [Çekirdek Model Yapısı](./architecture/core_model.md)
- [Büyüme Modülleri](./architecture/growth_modules.md)
- [Mimari Karar Kayıtları (ADR)](./architecture/adrs.md)

### 2. Modül Dokümantasyonu
- [Metin İşleme Modülü](./modules/text_processing.md)
- [Görüntü İşleme Modülü](./modules/image_processing.md)
- [Transformer Blokları](./modules/transformer.md)
- [Füzyon Mekanizması](./modules/fusion.md)
- [Arama ve İndeksleme](./modules/search.md)
- [Adapter Mekanizması](./modules/adapters.md)
- [Görev Başlıkları](./modules/task_heads.md)
- [Eğitim Yöneticisi](./modules/training.md)
- [Veri İndirme](./modules/data_export.md)

### 3. SDK Dokümantasyonu
- [SDK Genel Bakış](./sdk/overview.md)
- [Android SDK](./sdk/android.md)
- [iOS SDK](./sdk/ios.md)
- [API Referansı](./sdk/api_reference.md)

### 4. Geliştirici Kılavuzları
- [Ortam Kurulumu](./guides/setup.md)
- [Katkıda Bulunma](./guides/contributing.md)
- [Kodlama Standartları](./guides/coding_standards.md)
- [Test Prosedürleri](./guides/testing.md)

### 5. Örnek Uygulamalar
- [Basit Resim Sınıflandırıcı](./examples/image_classifier.md)
- [Kişisel Not Arama](./examples/note_search.md)
- [Çoklu Modalite Entegrasyonu](./examples/multi_modal.md)

### 6. Örüntü Kataloğu
- [Mimari Örüntüler](./patterns/architectural.md)
- [Performans Örüntüleri](./patterns/performance.md)
- [Güvenlik Örüntüleri](./patterns/security.md)
- [Anti-Örüntüler ve Nasıl Kaçınılır](./patterns/anti_patterns.md)

## Katkıda Bulunanlar

- [Proje Ekibi ve Katkıda Bulunanlar](./contributors.md)

## Lisans ve Kullanım Şartları

- [Lisans Bilgileri](./license.md)
- [Kullanım Şartları](./terms.md)

---

**Not:** Bu dokümantasyon sürekli olarak geliştirilmektedir. Katkıda bulunmak için [Katkıda Bulunma](./guides/contributing.md) kılavuzunu inceleyebilirsiniz. 