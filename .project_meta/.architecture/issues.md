# M³TM Mimari Sorunlar ve İyileştirme Önerileri

Bu belge, proje geliştirme sürecinde tespit edilen mimari sorunları, potansiyel iyileştirme alanlarını ve çözüm önerilerini içerir.

## Açık Sorunlar

### IS-001: Vektör niceleme (quantization) stratejisi
- **Durum**: Açık
- **Öncelik**: Orta
- **Tespit Tarihi**: 2024-05-09
- **Atanan**: Henüz atanmadı
- **Açıklama**: Mobil cihazlarda bellek verimliliği için vektör niceleme stratejisinin belirlenmesi gerekmektedir. INT8 vs FLOAT16 gibi farklı seçeneklerin performans ve doğruluk açısından değerlendirilmesi gerekiyor.
- **Çözüm Önerisi**: Transformer bloklarında dinamik niceleme desteği eklenebilir, böylece cihaz özelliklerine göre otomatik ayarlama yapılabilir.

### IS-002: Sekans uzunluğu optimizasyonu
- **Durum**: Açık
- **Öncelik**: Yüksek
- **Tespit Tarihi**: 2024-05-24
- **Atanan**: Henüz atanmadı
- **Açıklama**: Dikkat mekanizmaları araştırmasında, uzun sekansların standart dikkat mekanizmasında O(n²) karmaşıklık nedeniyle ciddi performans sorunlarına yol açtığı tespit edildi. Özellikle mobil cihazlarda bu sorun daha belirgin.
- **Çözüm Önerisi**: Linformer veya Performer gibi O(n) karmaşıklığa sahip dikkat mekanizmaları kullanılmalı. Alternatif olarak, maksimum sekans uzunluğu sınırlandırılabilir veya yerel dikkat (pencere tabanlı) mekanizmaları kullanılabilir.

### IS-003: Derinlik/genişlik dengesi
- **Durum**: Açık
- **Öncelik**: Orta
- **Tespit Tarihi**: 2024-05-24
- **Atanan**: Henüz atanmadı
- **Açıklama**: Mobil cihazlarda model mimarisinin derinlik ve genişlik dengesinin optimal olmayabileceği, özellikle konvolüsyon katmanlarında kaynak kullanımını etkileyebileceği görüldü. 
- **Çözüm Önerisi**: MobileNetV2 mimarisinde olduğu gibi ters darboğaz (inverted bottleneck) yapısı ile az sayıda kanal genişliği artırılıp, daha fazla derinlik eklenebilir. Derinlik yönlü ayrılabilir konvolüsyonlar kullanılmalı.

### IS-005: LongMethodAntiPattern tespit edildi
- **Durum**: Açık
- **Öncelik**: Orta
- **Tespit Tarihi**: 2024-05-31
- **Atanan**: Henüz atanmadı
- **Açıklama**: `m3tm/examples/text_classification.py` içerisindeki `main()` fonksiyonu 50+ satır ile birden fazla sorumluluğu (argüman işleme, model oluşturma, veri yükleme, eğitim) karıştırmaktadır. Bu Tek Sorumluluk İlkesi'ne (SRP) aykırıdır ve bakım zorluğu yaratmaktadır.
- **Çözüm Önerisi**: `main()` fonksiyonu daha küçük, odaklı fonksiyonlara bölünmelidir: `parse_arguments()`, `setup_environment()`, `prepare_dataset()`, `build_model()`, `train_and_evaluate()`. Özellikle örnek kodda, desen uygulaması daha net olmalıdır.
- **İlgili Desen/Anti-desen**: AP-001 (LongMethodAntiPattern), PT-013 (TrainingLoopTemplate)

### IS-006: DatasetFactory'de eksik soyutlama
- **Durum**: Açık
- **Öncelik**: Düşük
- **Tespit Tarihi**: 2024-05-31
- **Atanan**: Henüz atanmadı
- **Açıklama**: `DatasetFactory` sınıfı şu anda sadece metin sınıflandırma veri kümeleri oluşturmaktadır ve görüntü veri kümeleri için genişleme potansiyeli sınırlıdır. API, gelecekteki veri türleri için değişiklik gerektirecektir.
- **Çözüm Önerisi**: Genel bir `create_dataset()` metodu eklenmeli ve dataset türleri enum veya strateji deseni ile tanımlanmalıdır. Görüntü ve çoklu modalite veri kümeleri için hazırlık yapılmalıdır.
- **İlgili Desen/Anti-desen**: AP-002 (IncompleteAbstractionAntiPattern), PT-007 (DatasetFactory)

## Çözülen Sorunlar

### IS-004: Dikkat mekanizması standardizasyonu
- **Durum**: Çözüldü
- **Öncelik**: Yüksek
- **Tespit Tarihi**: 2024-05-09
- **Çözüm Tarihi**: 2024-05-24
- **Çözen**: codeflow_agent
- **Açıklama**: Farklı dikkat mekanizmalarının (self-attention, linformer, performer vb.) tutarlı bir arayüz ile kullanılabilmesi için standardizasyon gerekiyordu.
- **Çözüm**: MechanismRegistry örüntüsü (PT-007) uygulanarak tüm dikkat mekanizmaları için tutarlı bir arayüz ve merkezi kayıt sistemi oluşturuldu. Aynı sistem konvolüsyon mekanizmaları için de kullanıldı. 