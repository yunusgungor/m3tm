# Adapter ve Görev Başlığı Eğitim API Referansı

Bu belge, M³TM modelinin Adapter'lar ve görev başlıklarını eğitmek için kullanılan API'yi açıklar.

## AdapterTrainingConfig

`AdapterTrainingConfig` sınıfı, adapter ve görev başlığı eğitimi için kullanılan yapılandırma seçeneklerini içerir.

```python
from m3tm.training.adapter_training import AdapterTrainingConfig

config = AdapterTrainingConfig(
    train_adapter_names=["domain_adapter", "task_adapter"],  # Eğitilecek adapter isimleri
    train_task_heads=True,                                   # Görev başlıklarını eğit
    freeze_core_model=True,                                  # Çekirdek modeli dondur
    learning_rate=5e-4,                                      # Öğrenme oranı
    epochs=5,                                                # Epoch sayısı
    batch_size=32,                                           # Batch boyutu
    optimizer="adamw",                                       # Optimizer tipi
    scheduler="cosine",                                      # Zamanlayıcı tipi
    checkpoint_frequency=1,                                  # Kaç epoch'ta bir checkpoint
    memory_optimization_level="moderate"                     # Bellek optimizasyon seviyesi
)
```

### Parametreler

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|-----------|
| `train_adapter_names` | List[str] | None | Eğitilecek adapter isimlerinin listesi. None ise tüm adapter'lar eğitilir. |
| `train_task_heads` | bool | True | Görev başlıklarını eğitme bayrağı. |
| `freeze_core_model` | bool | True | Çekirdek modeli dondurma bayrağı. |
| `checkpoint_frequency` | int | 1 | Kaç epoch'ta bir checkpoint oluşturulacağı. |
| `checkpoint_keep_best_n` | int | 3 | Saklanacak en iyi checkpoint sayısı. |
| `memory_optimization_level` | str | "moderate" | Bellek optimizasyon seviyesi ("low", "moderate", "aggressive"). |

Ayrıca, `TrainingConfig` sınıfından miras alınan tüm parametreler de kullanılabilir:

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|-----------|
| `learning_rate` | float | 1e-3 | Öğrenme oranı. |
| `epochs` | int | 10 | Epoch sayısı. |
| `batch_size` | int | 32 | Batch boyutu. |
| `optimizer` | str | "adam" | Optimizer tipi ("adam", "adamw", "sgd"). |
| `scheduler` | str | "none" | Zamanlayıcı tipi ("none", "cosine", "linear", "reduce_on_plateau"). |
| `warmup_steps` | int | 0 | Isınma adımı sayısı. |
| `weight_decay` | float | 0.0 | Ağırlık bozunumu. |
| `gradient_clip_val` | float | 0.0 | Gradient kesme değeri. |
| `device` | str | "cpu" | Eğitim cihazı ("cpu", "cuda", "mps"). |

## TrainingCallback

`TrainingCallback` sınıfı, eğitim süreci sırasında belirli olaylarda çağrılacak callback'leri tanımlamak için kullanılır.

```python
from m3tm.training.adapter_training import TrainingCallback

class MyCallback(TrainingCallback):
    def on_training_start(self, trainer, **kwargs):
        print("Eğitim başladı!")
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        print(f"Epoch {epoch+1} tamamlandı. Metrikler: {metrics}")
```

### Callback Metodları

| Metod | Parametreler | Açıklama |
|-------|-------------|-----------|
| `on_training_start` | trainer, **kwargs | Eğitim başladığında çağrılır. |
| `on_epoch_start` | trainer, epoch, **kwargs | Her epoch başında çağrılır. |
| `on_batch_start` | trainer, batch, **kwargs | Her batch başında çağrılır. |
| `on_batch_end` | trainer, batch_metrics, **kwargs | Her batch sonunda çağrılır. |
| `on_epoch_end` | trainer, epoch, metrics, **kwargs | Her epoch sonunda çağrılır. |
| `on_training_end` | trainer, **kwargs | Eğitim bittiğinde çağrılır. |

## EarlyStoppingCallback

`EarlyStoppingCallback` sınıfı, belirli bir metrik iyileşme göstermediğinde eğitimi durdurmak için kullanılır.

```python
from m3tm.training.adapter_training import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    patience=3,           # Kaç epoch'tan sonra durdurulacak
    monitor="val_loss",   # İzlenecek metrik
    mode="min"            # Optimize etme yönü (min veya max)
)
```

### Parametreler

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|-----------|
| `patience` | int | 3 | Kaç epoch boyunca iyileşme olmazsa durdurulacağı. |
| `monitor` | str | "val_loss" | İzlenecek metrik ismi. |
| `mode` | str | "min" | Metriğin optimize edilme yönü ("min" veya "max"). |

## LearningRateSchedulerCallback

`LearningRateSchedulerCallback` sınıfı, eğitim sırasında öğrenme oranını dinamik olarak ayarlamak için kullanılır.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
from m3tm.training.adapter_training import LearningRateSchedulerCallback

scheduler = CosineAnnealingLR(optimizer, T_max=10)
callback = LearningRateSchedulerCallback(
    scheduler=scheduler,         # Zamanlayıcı
    monitor="val_loss",          # İzlenecek metrik (ReduceLROnPlateau için)
    schedule_on_batch=False      # Batch sonunda mı epoch sonunda mı güncelleme
)
```

### Parametreler

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|-----------|
| `scheduler` | torch.optim.lr_scheduler._LRScheduler | - | Öğrenme oranı zamanlayıcısı. |
| `monitor` | str | "val_loss" | İzlenecek metrik (ReduceLROnPlateau için). |
| `schedule_on_batch` | bool | False | Batch sonunda mı epoch sonunda mı güncelleme yapılacağı. |

## AdapterTrainingManager

`AdapterTrainingManager` sınıfı, adapter ve görev başlığı eğitimini yönetir.

```python
from m3tm.training.adapter_training import AdapterTrainingManager

training_manager = AdapterTrainingManager(
    model=model,                      # Eğitilecek model
    config=config,                    # Eğitim yapılandırması
    adapter_manager=adapter_manager,  # Adapter yöneticisi
    save_dir="./checkpoints",         # Kaydetme dizini
    use_tensorboard=True,             # TensorBoard kullanımı
    callbacks=[early_stopping]        # Callback'ler
)

# Modeli eğitim için hazırla
training_manager.prepare_model_for_training()

# Eğitimi başlat
metrics = training_manager.train(
    train_loader=train_dataloader,
    criterion=criterion,
    val_loader=val_dataloader,
    resume=False                      # Eğitime devam et
)

# Test et
test_metrics = training_manager.evaluate(test_dataloader, criterion)
```

### Metotlar

| Metot | Parametreler | Dönüş Değeri | Açıklama |
|-------|-------------|--------------|-----------|
| `prepare_model_for_training` | - | None | Modeli eğitim için hazırlar. |
| `train` | train_loader, criterion, val_loader=None, resume=False | Dict[str, float] | Modeli eğitir ve metrikleri döndürür. |
| `evaluate` | test_loader, criterion | Dict[str, float] | Modeli değerlendirir ve metrikleri döndürür. |
| `get_trainable_parameter_count` | - | Dict[str, int] | Eğitilebilir parametre sayılarını döndürür. |

## Yardımcı Fonksiyonlar

### create_training_manager

```python
from m3tm.training.adapter_training import create_training_manager

training_manager = create_training_manager(
    model=model,
    config=config_dict,  # Dict veya AdapterTrainingConfig olabilir
    adapter_manager=adapter_manager,
    save_dir="./checkpoints"
)
```

### create_adapter_training_callbacks

```python
from m3tm.training.adapter_training_utils import create_adapter_training_callbacks

callbacks = create_adapter_training_callbacks(
    use_early_stopping=True,  # Erken durdurma kullanılsın mı?
    use_stats_logging=True,   # İstatistik loglama kullanılsın mı?
    use_grad_check=False,     # Gradient kontrolü kullanılsın mı?
    early_stopping_config={"patience": 3, "monitor": "val_loss"}
)
```

### create_adapter_criterion

```python
from m3tm.training.adapter_training_utils import create_adapter_criterion

# Sınıflandırma için kayıp fonksiyonu
criterion = create_adapter_criterion(
    task_type="classification",
    loss_config={"class_weights": [1.0, 2.0, 1.0]}  # Opsiyonel
)
```

### find_task_heads_and_adapters

```python
from m3tm.training.adapter_training_utils import find_task_heads_and_adapters

task_heads, adapters = find_task_heads_and_adapters(model)
```

## Callback Türleri

### ModelStatisticsCallback

Model istatistiklerini toplayan ve loglayan callback.

```python
from m3tm.training.adapter_training_utils import ModelStatisticsCallback

callback = ModelStatisticsCallback(log_frequency=1)  # Her epoch'ta logla
```

### GradientCheckCallback

Gradient değerlerini kontrol eden ve aşırı/sıfır gradient problemlerini tespit eden callback.

```python
from m3tm.training.adapter_training_utils import GradientCheckCallback

callback = GradientCheckCallback(
    check_frequency=10,                # Her 10 batch'te bir kontrol et
    log_histogram=False,               # Histogram loglamasını etkinleştir
    gradient_threshold=10.0,           # Aşırı gradient eşik değeri
    zero_fraction_threshold=0.9        # Sıfır gradient oranı eşik değeri
)
```

### MemoryTrackingCallback

Bellek kullanımını takip eden callback.

```python
from m3tm.training.adapter_training_utils import MemoryTrackingCallback

callback = MemoryTrackingCallback(log_frequency=10)  # Her 10 batch'te bir logla
```

### LearningRateMonitorCallback

Öğrenme oranını takip eden callback.

```python
from m3tm.training.adapter_training_utils import LearningRateMonitorCallback

callback = LearningRateMonitorCallback(log_frequency=10)  # Her 10 batch'te bir logla
```

## Örnek Kullanım

```python
from m3tm.training.adapter_training import AdapterTrainingConfig, AdapterTrainingManager
from m3tm.training.adapter_training_utils import create_adapter_training_callbacks, create_adapter_criterion

# Yapılandırma
config = AdapterTrainingConfig(
    train_adapter_names=["domain_adapter"],
    train_task_heads=True,
    freeze_core_model=True,
    learning_rate=5e-4,
    epochs=5
)

# Callback'ler
callbacks = create_adapter_training_callbacks(
    use_early_stopping=True,
    early_stopping_config={"patience": 3}
)

# Eğitim yöneticisi
training_manager = AdapterTrainingManager(
    model=model,
    config=config,
    adapter_manager=adapter_manager,
    save_dir="./checkpoints",
    callbacks=callbacks
)

# Kayıp fonksiyonu
criterion = create_adapter_criterion("classification")

# Eğitim
training_manager.prepare_model_for_training()
metrics = training_manager.train(train_loader, criterion, val_loader)
```

Daha detaylı bir örnek için `examples/adapter_training_example.py` dosyasına bakabilirsiniz. 