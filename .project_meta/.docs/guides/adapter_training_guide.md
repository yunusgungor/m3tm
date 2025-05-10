# Adapter ve Görev Başlığı Eğitimi Kılavuzu

Bu kılavuz, M³TM modelinde Adapter'ları ve görev başlıklarını eğitmek için kullanılabilecek araçları ve teknikleri açıklar.

## İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Adapter ve Görev Başlığı Nedir?](#adapter-ve-görev-başlığı-nedir)
3. [Eğitim Mimarisi](#eğitim-mimarisi)
4. [Temel Kullanım](#temel-kullanım)
5. [İleri Düzey Kullanım](#i̇leri-düzey-kullanım)
6. [En İyi Uygulamalar](#en-i̇yi-uygulamalar)
7. [Sorun Giderme](#sorun-giderme)

## Genel Bakış

M³TM (Mobil Multi-Modal Modüler Transformer) modelinin önemli özelliklerinden biri, çekirdek modeli korurken yalnızca küçük adapter modüllerini ve görev başlıklarını eğiterek kişiselleştirilebilmesidir. Bu yaklaşım, mobil cihazlarda verimli bir şekilde eğitim yapılmasını sağlar ve daha az bellek/hesaplama gücü gerektirir.

Bu kılavuz, adapter'ları ve görev başlıklarını eğitmek için gereken adımları ve araçları açıklamaktadır.

## Adapter ve Görev Başlığı Nedir?

### Adapter

Adapter'lar, çekirdek modeli değiştirmeden modelin belirli bölümlerine eklenebilen küçük, eğitilebilir modüllerdir. Tipik bir adapter mimarisi şu bileşenlerden oluşur:

1. **Down-projeksiyon**: Girdiyi daha küçük bir boyuta indirir (darboğaz)
2. **Aktivasyon fonksiyonu**: Doğrusal olmayan bir dönüşüm uygular
3. **Up-projeksiyon**: Çıktıyı orijinal boyuta geri getirir
4. **Artık bağlantı**: Çıktı, orijinal girdiye eklenir

M³TM'de adapter'lar, genellikle transformer bloklarına eklenir ve çok daha az parametre içerir (genellikle çekirdek modelin %1'inden az).

### Görev Başlığı

Görev başlıkları, model çıktısını belirli bir görev için uygun formata dönüştüren modüllerdir. Örneğin:

- **ClassificationHead**: Sınıflandırma görevleri için lojistikler üretir
- **RegressionHead**: Regresyon görevleri için sayısal değerler üretir
- **MultiLabelHead**: Çoklu etiket sınıflandırması için kullanılır

Görev başlıkları genellikle küçüktür ve adapter'lar gibi, çekirdek modeli değiştirmeden eğitilebilirler.

## Eğitim Mimarisi

M³TM'de adapter ve görev başlığı eğitimi için aşağıdaki bileşenler bulunur:

- **AdapterTrainingConfig**: Eğitim parametrelerini ve hangi bileşenlerin eğitileceğini belirten yapılandırma sınıfı
- **AdapterTrainingManager**: Eğitim sürecini yöneten ana sınıf
- **Callback'ler**: Eğitim sırasında çeşitli aşamalarda çağrılan işlevler
  - EarlyStoppingCallback: Erken durdurma için
  - LearningRateSchedulerCallback: Öğrenme oranı ayarlaması için
  - ModelStatisticsCallback: Model istatistiklerini kaydetmek için
  - ve diğerleri...

Bu mimarinin ana avantajları şunlardır:

1. **Verimlilik**: Yalnızca küçük bir parametre alt kümesini eğitmek, hesaplama kaynaklarını önemli ölçüde azaltır
2. **Kişiselleştirme**: Farklı kullanıcılar veya görevler için ayrı adapter'lar eğitilebilir
3. **Modülerlik**: Adapter'lar ve görev başlıkları ayrı ayrı eğitilebilir ve birleştirilebilir

## Temel Kullanım

Adapter'ları ve görev başlıklarını eğitmek için temel adımlar:

### 1. Gerekli Modülleri İçe Aktarma

```python
from m3tm.training.adapter_training import AdapterTrainingConfig, AdapterTrainingManager
from m3tm.training.adapter_training_utils import create_adapter_training_callbacks
from m3tm.adapters.adapter_manager import AdapterManager
```

### 2. Adapter Yöneticisi ve Adapter'lar Oluşturma

```python
# Adapter yöneticisi oluştur
adapter_manager = AdapterManager(model)

# Adapter oluştur ve ekle
adapter_config = AdapterConfig(adapter_type=AdapterType.BOTTLENECK, bottleneck_dim=16)
adapter_manager.register_adapter(model.transformer, "my_adapter", "post_attention", adapter_config)
```

### 3. Eğitim Yapılandırması Oluşturma

```python
train_config = AdapterTrainingConfig(
    train_adapter_names=["my_adapter"],  # Hangi adapter'ların eğitileceği
    train_task_heads=True,               # Görev başlığını da eğit
    freeze_core_model=True,              # Çekirdek modeli dondur
    learning_rate=5e-4,                 
    epochs=5,
    batch_size=32
)
```

### 4. Callback'ler Oluşturma

```python
callbacks = create_adapter_training_callbacks(
    use_early_stopping=True,
    early_stopping_config={"patience": 3, "monitor": "val_loss"}
)
```

### 5. Eğitim Yöneticisi Oluşturma

```python
training_manager = AdapterTrainingManager(
    model=model,
    config=train_config,
    adapter_manager=adapter_manager,
    save_dir="./checkpoints",
    callbacks=callbacks
)
```

### 6. Modeli Eğitim İçin Hazırlama

```python
training_manager.prepare_model_for_training()
```

### 7. Kayıp Fonksiyonu Tanımlama

```python
def criterion(outputs, inputs):
    return outputs.get('loss', 0.0)  # 'loss' modelin forward geçişinde hesaplanıyorsa
```

### 8. Eğitimi Başlatma

```python
metrics = training_manager.train(
    train_loader=train_dataloader,
    criterion=criterion,
    val_loader=val_dataloader
)
```

### 9. Değerlendirme

```python
test_metrics = training_manager.evaluate(test_dataloader, criterion)
```

## İleri Düzey Kullanım

### Birden Fazla Adapter Eğitme

Farklı amaçlar için birden fazla adapter eğitebilirsiniz:

```python
# Farklı adapter'lar ekle
adapter_manager.add_adapter(model.transformer, "domain_adapter", domain_adapter)
adapter_manager.add_adapter(model.transformer, "task_adapter", task_adapter)

# Yapılandırmada her ikisini de belirt
train_config = AdapterTrainingConfig(
    train_adapter_names=["domain_adapter", "task_adapter"],
    ...
)
```

### Bellek Optimizasyonu

Mobil cihazlarda eğitim yaparken bellek sınırlı olabilir. Bellek kullanımını optimize etmek için:

```python
train_config = AdapterTrainingConfig(
    memory_optimization_level="aggressive",  # "low", "moderate" veya "aggressive"
    ...
)
```

Ayrıca, bellek kullanımını izlemek için:

```python
from m3tm.training.adapter_training_utils import MemoryTrackingCallback

memory_callback = MemoryTrackingCallback(log_frequency=10)
callbacks.append(memory_callback)
```

### Gradient Kontrolü

Eğitim sırasında gradientleri kontrol etmek için:

```python
from m3tm.training.adapter_training_utils import GradientCheckCallback

gradient_callback = GradientCheckCallback(
    check_frequency=50,
    gradient_threshold=10.0
)
callbacks.append(gradient_callback)
```

### Özelleştirilmiş Callback'ler

Kendi callback'lerinizi oluşturabilirsiniz:

```python
from m3tm.training.adapter_training import TrainingCallback

class MyCustomCallback(TrainingCallback):
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        # Özel işlemler
        pass
```

### Eğitime Devam Etme

Eğitimi kaldığı yerden devam ettirmek için:

```python
metrics = training_manager.train(
    train_loader=train_dataloader,
    criterion=criterion,
    resume=True  # Eğitime devam et
)
```

## En İyi Uygulamalar

### 1. Adapter Boyutunu Doğru Seçme

Adapter'ların darboğaz boyutu (`bottleneck_dim`) performansı etkiler:
- **Küçük boyut** (8-16): Daha az parametre, daha hızlı eğitim, ancak potansiyel olarak daha düşük performans
- **Orta boyut** (32-64): İyi bir denge
- **Büyük boyut** (128+): Daha fazla parametre, daha iyi performans potansiyeli, ancak daha yavaş eğitim

Uygulama gereksinimlerine göre dengeleme yapın.

### 2. Çekirdek Modeli Dondurun

Eğitim sırasında çekirdek modeli dondurmak, eğitim süresini önemli ölçüde azaltır ve aşırı uyumu önlemeye yardımcı olur:

```python
train_config = AdapterTrainingConfig(
    freeze_core_model=True,
    ...
)
```

### 3. Erken Durdurma Kullanın

Aşırı uyumu önlemek için erken durdurma kullanın:

```python
callbacks = create_adapter_training_callbacks(
    use_early_stopping=True,
    early_stopping_config={"patience": 3, "monitor": "val_loss"}
)
```

### 4. Checkpoint'leri Yönetin

Eğitim sırasında düzenli olarak checkpoint'ler oluşturun ve en iyilerini saklayın:

```python
train_config = AdapterTrainingConfig(
    checkpoint_frequency=1,  # Her epoch'ta bir checkpoint
    checkpoint_keep_best_n=3,  # En iyi 3 modeli sakla
    ...
)
```

### 5. TensorBoard Kullanın

Eğitimi izlemek için TensorBoard kullanın:

```python
training_manager = AdapterTrainingManager(
    use_tensorboard=True,
    ...
)
```

TensorBoard'ı başlatmak için:

```
tensorboard --logdir=./checkpoints/tensorboard
```

## Sorun Giderme

### Bellek Sorunları

**Sorun:** "CUDA bellek yetersiz" hatası alıyorsanız:

**Çözüm:**
1. Batch boyutunu azaltın
2. Bellek optimizasyonunu artırın
3. Daha küçük adapter'lar kullanın
4. Gradient biriktirme kullanın

```python
train_config = AdapterTrainingConfig(
    batch_size=16,  # Daha küçük batch
    memory_optimization_level="aggressive",
    ...
)
```

### Düşük Performans

**Sorun:** Eğitimden sonra model düşük performans gösteriyorsa:

**Çözüm:**
1. Daha büyük adapter boyutları deneyin
2. Öğrenme oranını ayarlayın
3. Daha uzun eğitim süresi kullanın
4. Regularizasyon ekleyin

```python
train_config = AdapterTrainingConfig(
    learning_rate=1e-4,  # Daha düşük öğrenme oranı
    epochs=10,  # Daha fazla epoch
    weight_decay=0.01,  # Regularizasyon ekle
    ...
)
```

### Eğitim İlerlemesi Yok

**Sorun:** Kayıp düşmüyorsa:

**Çözüm:**
1. Öğrenme oranını kontrol edin (çok küçük veya çok büyük olabilir)
2. Gradientleri kontrol edin
3. Modelin yapısını kontrol edin (eğitilebilir parametreler var mı?)

```python
from m3tm.training.adapter_training_utils import GradientCheckCallback, LearningRateMonitorCallback

# Gradientleri ve öğrenme oranını izle
callbacks = [
    GradientCheckCallback(check_frequency=10),
    LearningRateMonitorCallback(log_frequency=10)
]
```

## Özet

Adapter'lar ve görev başlıkları, M³TM modelini verimli bir şekilde kişiselleştirmenin anahtarıdır. Bu kılavuz, adapter eğitiminin temel ve ileri düzey kullanımını, en iyi uygulamaları ve sorun giderme ipuçlarını sunmaktadır.

Daha fazla bilgi için [API referans belgesine](../api/adapter_training.md) bakabilirsiniz.

## Kaynak Kodları

- [AdapterTrainingManager](../../src/m3tm/training/adapter_training.py)
- [AdapterTrainingUtils](../../src/m3tm/training/adapter_training_utils.py)
- [Adapter](../../src/m3tm/adapters/adapter.py)
- [AdapterManager](../../src/m3tm/adapters/adapter_manager.py)
- [TaskHead](../../src/m3tm/task_heads/base.py)

## Örnek Kodlar

- [Adapter Eğitimi Örneği](../tutorials/code_samples/adapter_training_tutorial.py)
- [Daha Kapsamlı Örnek](../../examples/adapter_training_example.py)