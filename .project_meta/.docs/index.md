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