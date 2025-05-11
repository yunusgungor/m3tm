# Mobil, Çoklu Modalite ve Eğitim Desenleri

Bu belge, kod tabanı analizi sonucunda tespit edilen ve kataloglanan Mobil, Çoklu Modalite ve Eğitim desenlerini açıklamaktadır.

## 1. Mobil Model Desenleri

Mobil cihazlara model dağıtımı ve optimizasyonu ile ilgili üç ana desen tespit edilmiştir:

### MobileModelConverter (PT-022)

**Kategori:** Mobile

**Açıklama:** PyTorch modellerini mobil platformlar için optimize edilmiş TorchScript formatına dönüştüren bir utility sınıfı deseni.

**Uygulama:** `src/m3tm/mobile/model_converter.py`

**Örnek Kullanım:**
```python
converter = ModelConverter()
torchscript_model = converter.to_torchscript(model, example_inputs, optimize=True)
model_path = converter.convert_for_android(model, example_inputs, "model.pt", optimize=True)
```

**Avantajları:**
- Platform bağımsız, ortak bir dönüşüm API'si sağlar
- Farklı optimizasyon seçeneklerini tek bir yerde encapsulate eder
- Her dönüşüm adımını izole ederek hata ayıklamayı kolaylaştırır
- Mobil platform özgü özelleştirmeleri destekler

**Uygulama Notları:**
- Statik metotlar kullanarak çeşitli dönüşüm ve optimizasyon işlemlerini gerçekleştirir
- TorchScript ve optimize_for_mobile kullanımını kolaylaştırır
- Android ve iOS gibi farklı platformlar için özel dönüşüm metotları sağlar

### BenchmarkStrategy (PT-023)

**Kategori:** Performance

**Açıklama:** Farklı modellerin mobil platformlardaki performansını tutarlı bir şekilde ölçen, karşılaştıran ve raporlayan strateji deseni.

**Uygulama:** `src/m3tm/mobile/benchmark.py`

**Örnek Kullanım:**
```python
benchmark = MobileBenchmark()
latency_results = benchmark.benchmark_inference_time(model, example_inputs, num_runs=100)
memory_results = benchmark.benchmark_memory_usage(model, example_inputs)
comparison = benchmark.compare_models(models_dict, example_inputs)
benchmark.generate_report(comparison, "benchmark_report.md")
```

**Avantajları:**
- Farklı optimizasyon stratejilerinin etkilerini ölçmek için tutarlı bir yöntem sağlar
- İstatistiksel anlamlı sonuçlar için tekrarlı ölçümler yapar
- Sonuçları okunabilir raporlara dönüştürür
- Çıkarım süresi, bellek kullanımı gibi farklı metrikleri standartlaştırır

**Uygulama Notları:**
- Isınma turları (warmup) ile daha doğru ölçümler sağlar
- İstatistiksel özet (ortalama, medyan, p95) değerleri hesaplar
- CPU ve CUDA cihazları için farklı ölçüm stratejileri uygular
- Karşılaştırmalı raporlar oluşturur

### MobileOptimizationPipeline (PT-024)

**Kategori:** Performance

**Açıklama:** Mobil modellerini optimize etmek için farklı teknikleri (quantization, pruning, vb.) bir pipeline şeklinde uygulayan bir desen.

**Uygulama:** `src/m3tm/mobile/optimization.py`

**Örnek Kullanım:**
```python
optimizer = MobileOptimizer()
quantized_model = optimizer.apply_dynamic_quantization(model)
pruned_model = optimizer.apply_pruning(model, amount=0.2)
optimized_model = optimizer.compress_model(model, example_inputs, methods=["quantize", "prune"])
```

**Avantajları:**
- Farklı optimizasyon tekniklerini birleştirme esnekliği sağlar
- Her tekniğin etkisini izole bir şekilde ölçmeyi mümkün kılar
- Yeni optimizasyon yöntemlerinin eklenmesine olanak tanır
- Modelin bellek kullanımını ve çalışma hızını optimize eder

**Uygulama Notları:**
- Her optimizasyon adımı modüler bir şekilde uygulanır ve birleştirilebilir
- Dinamik ve statik quantization, quantization aware training ve pruning gibi teknikleri uygular
- Optimize edilmiş modelleri kaydetme ve yükleme işlevleri sağlar

## 2. Çoklu Modalite Desenleri

### ModalityFusionStrategy (PT-025)

**Kategori:** Multimodal

**Açıklama:** Farklı modalitelerden (metin, görüntü) gelen gömmeleri çeşitli stratejilerle birleştiren strateji deseni.

**Uygulama:** `src/m3tm/fusion/fusion_strategies.py`

**Örnek Kullanım:**
```python
config = ConcatenationFusionConfig(text_dim=768, image_dim=512, output_dim=1024)
fusion = ConcatenationFusion(config)
fused_embeddings = fusion.fuse(text_embeddings, image_embeddings, text_mask, image_mask)
```

**Avantajları:**
- Farklı füzyon stratejilerini ortak bir arayüzle sunar
- Stratejiler arasında geçiş yapmayı kolaylaştırır
- Tek ve çoklu modalite durumlarını tutarlı şekilde ele alır
- Yeni füzyon stratejilerinin eklenmesini destekler

**Uygulama Notları:**
- Soyut bir temel sınıf (BaseFusion) tanımlayarak farklı füzyon stratejilerini standardize eder
- ConcatenationFusion, WeightedSumFusion, GatedFusion gibi farklı stratejiler sunar
- Eksik modalite durumları (_handle_text_only, _handle_image_only) için çözümler sağlar
- Global pooling, maskeleme ve normalizasyon gibi yardımcı işlevler içerir

**Strateji Çeşitleri:**
1. **ConcatenationFusion**: Farklı modalite gömmelerini basitçe birleştirir ve opsiyonel projeksiyon uygular
2. **WeightedSumFusion**: Modaliteleri ağırlıklı olarak toplar, ağırlıklar sabit veya öğrenilebilir olabilir
3. **GatedFusion**: Bir geçit mekanizması kullanarak modalitelerin dinamik ağırlıklandırmasını yapar

## 3. Eğitim Desenleri

### TrainingCallbackSystem (PT-026)

**Kategori:** Training

**Açıklama:** Eğitim sürecinin çeşitli aşamalarında özelleştirilebilir işlemler gerçekleştirmek için callback tabanlı bir sistem.

**Uygulama:** `src/m3tm/training/adapter_training.py`

**Örnek Kullanım:**
```python
# Callback oluşturma
early_stopping = EarlyStoppingCallback(patience=5, monitor="val_loss", mode="min")
lr_scheduler = LearningRateSchedulerCallback(scheduler)

# Callback'leri eğitim yöneticisine ekleyerek kullanma
training_manager = AdapterTrainingManager(
    model, adapter_manager, save_dir="checkpoints",
    callbacks=[early_stopping, lr_scheduler]
)
training_manager.train(train_loader, criterion, val_loader)
```

**Avantajları:**
- Eğitim sürecinin davranışını değiştirmeden genişletmeyi sağlar
- Farklı callback'lerin bir arada kullanılmasına olanak tanır
- Modüler ve test edilebilir kod yapısı oluşturur
- Erken durdurma, öğrenme oranı planlama gibi gelişmiş eğitim özellikleri ekler

**Uygulama Notları:**
- Temel `TrainingCallback` sınıfı, eğitim sürecinin farklı aşamalarını yakalar:
  - `on_training_start`, `on_training_end`
  - `on_epoch_start`, `on_epoch_end`
  - `on_batch_start`, `on_batch_end`
- Özel callback sınıfları bu temel sınıfı genişleterek belirli işlevler ekler
- Callback'ler eğitim yöneticisine kaydedilir ve ilgili noktalarda tetiklenir

**Mevcut Callback Türleri:**
1. **EarlyStoppingCallback**: Belirli sayıda epoch'ta iyileşme olmazsa eğitimi durdurur
2. **LearningRateSchedulerCallback**: Öğrenme oranı planlamasını yönetir
3. **ModelStatisticsCallback**: Model istatistiklerini toplar ve raporlar
4. **GradientCheckCallback**: Gradient değerlerini izler ve raporlar

## Örüntü İlişkileri

```
MobileModelConverter (PT-022) ------> BenchmarkStrategy (PT-023)
         |                                   |
         |                                   |
         v                                   v
MobileOptimizationPipeline (PT-024) <-- ölçer/değerlendirir

                 +----------------------------+
                 |                            |
ModelComposite (PT-003) <---- ModalityFusionStrategy (PT-025)
                 |              |
                 |              |
                 v              v
PluggableComponentStrategy (PT-015)

DecoratorPattern (PT-017) ---> TrainingCallbackSystem (PT-026)
                                   |
                                   |
                                   v
                           TrainingLoopTemplate (PT-013)
```

## Gelecekteki Gelişim Alanları

1. **MobileModelConverter** için:
   - Diğer edge platformlara (Web, Edge cihazlar) destek eklenmesi
   - CoreML dönüşümü için gelişmiş destek
   - Karma modellerin (PyTorch + TensorFlow) dönüşümü için destek

2. **ModalityFusionStrategy** için:
   - Ses, sensör verisi gibi ek modalitelerin eklenmesi
   - Daha gelişmiş dinamik ağırlıklandırma stratejileri
   - Attention-based füzyon mekanizmaları

3. **TrainingCallbackSystem** için:
   - Dağıtık eğitim için özel callback'ler
   - Görselleştirme ve raporlama callback'leri
   - Hiper-parametre otomatik ayarlama callback'leri

## Öneriler

1. `TrainingCallbackSystem` deseninin dokümantasyonu geliştirilebilir ve daha fazla örnek eklenebilir.
2. `ModalityFusionStrategy` deseni ses, video gibi diğer modaliteleri de içerecek şekilde genişletilebilir.
3. `MobileModelConverter` desenine CoreML ve Web platform desteği eklenebilir.
4. Mobil desenler için daha kapsamlı ölçüm ve raporlama araçları geliştirilebilir.

---

Hazırlayan: Proje Yöneticisi  
Tarih: 11 Haziran 2024 