# TrainingManager API Dokümantasyonu

*Oluşturma Tarihi: 11 Haziran 2024*

## Genel Bakış

`TrainingManager` sınıfı, M³TM modeli için eğitim sürecini yöneten üst düzey bir API sağlar. Özellikle çekirdek modeli dondurarak sadece adapter'ları ve görev başlıklarını eğitmeye odaklanır. Bu, modelin hafif kalmasını ve verimli eğitim yapabilmesini sağlar.

## Kullanım Örneği

```python
from m3tm.training import TrainingManager

# Model ve eğitim yapılandırması oluştur
model = create_your_model()
training_config = {
    "learning_rate": 1e-3,
    "epochs": 5,
    "batch_size": 16,
    "optimizer": "adamw",
    "scheduler": "cosine",
    
    # Adapter eğitim özellikleri
    "train_adapter_names": ["adapter1", "adapter2"],  # Eğitilecek adapter'lar
    "train_task_heads": True,                         # Görev başlıklarını eğit
    "freeze_core_model": True                         # Çekirdek modeli dondur
}

# TrainingManager oluştur
trainer = TrainingManager.create(
    model=model,
    config=training_config,
    use_tensorboard=True,
    use_early_stopping=True,
    save_dir="./checkpoints"
)

# Eğitim için modeli hazırla
trainer.prepare_model_for_training()

# Modeli eğit
results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader
)

# Modeli değerlendir
eval_results = trainer.evaluate(test_loader)

# Checkpoint kaydet
trainer.save_checkpoint(is_best=True)
```

## Sınıf Detayları

### TrainingManager

```python
class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        adapter_manager: Optional[AdapterManager] = None,
        save_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = False,
        callbacks: Optional[List[TrainingCallback]] = None
    )
```

#### Parametreler

- **model** (`nn.Module`): Eğitilecek M³TM modeli
- **config** (`TrainingConfig`): Eğitim yapılandırması
- **adapter_manager** (`AdapterManager`, opsiyonel): AdapterManager örneği
- **save_dir** (`str` veya `Path`, opsiyonel): Model ve metriklerin kaydedileceği dizin
- **use_tensorboard** (`bool`, varsayılan=`False`): TensorBoard kullanılsın mı
- **callbacks** (`List[TrainingCallback]`, opsiyonel): Eğitim sürecindeki çeşitli aşamalarda çağrılacak callbacks

### Metotlar

#### prepare_model_for_training

```python
def prepare_model_for_training(self) -> None
```

Modeli eğitim için hazırlar. Çekirdek modeli dondurur (eğer belirtilmişse), sadece belirli adapter'ları eğitir (eğer belirtilmişse) ve görev başlıklarını eğitir/dondurur (yapılandırmaya göre).

#### train

```python
def train(
    self,
    train_loader: DataLoader,
    criterion: Optional[Callable] = None,
    val_loader: Optional[DataLoader] = None,
    resume: bool = False
) -> Dict[str, float]
```

Modeli eğitir.

##### Parametreler

- **train_loader** (`DataLoader`): Eğitim veri yükleyicisi
- **criterion** (`Callable`, opsiyonel): Kayıp fonksiyonu (belirtilmezse otomatik olarak oluşturulur)
- **val_loader** (`DataLoader`, opsiyonel): Doğrulama veri yükleyicisi
- **resume** (`bool`, varsayılan=`False`): Eğitimi bir checkpoint'ten devam ettir

##### Dönüş Değeri

- `Dict[str, float]`: Eğitim sonuçları içeren sözlük

#### evaluate

```python
def evaluate(
    self,
    test_loader: DataLoader,
    criterion: Optional[Callable] = None
) -> Dict[str, float]
```

Modeli değerlendirir.

##### Parametreler

- **test_loader** (`DataLoader`): Test veri yükleyicisi
- **criterion** (`Callable`, opsiyonel): Kayıp fonksiyonu (belirtilmezse otomatik olarak oluşturulur)

##### Dönüş Değeri

- `Dict[str, float]`: Değerlendirme sonuçları içeren sözlük

#### save_checkpoint

```python
def save_checkpoint(
    self,
    file_path: Optional[Union[str, Path]] = None,
    is_best: bool = False
) -> None
```

Model durumunu kaydeder.

##### Parametreler

- **file_path** (`str` veya `Path`, opsiyonel): Kaydedilecek dosya yolu (belirtilmezse varsayılan ad kullanılır)
- **is_best** (`bool`, varsayılan=`False`): Bu checkpoint en iyi sonucu veren model mi

#### load_checkpoint

```python
def load_checkpoint(
    self,
    file_path: Optional[Union[str, Path]] = None
) -> int
```

Model durumunu yükler.

##### Parametreler

- **file_path** (`str` veya `Path`, opsiyonel): Yüklenecek dosya yolu (belirtilmezse en son checkpoint kullanılır)

##### Dönüş Değeri

- `int`: Yüklenen checkpoint'in epoch numarası

#### create (Statik Metot)

```python
@staticmethod
def create(
    model: nn.Module,
    config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
    adapter_manager: Optional[AdapterManager] = None,
    save_dir: Optional[Union[str, Path]] = None,
    use_tensorboard: bool = False,
    use_early_stopping: bool = True,
    use_lr_scheduler: bool = True,
    memory_optimization_level: str = "moderate",
    additional_callbacks: Optional[List[TrainingCallback]] = None
) -> 'TrainingManager'
```

TrainingManager örneği oluşturur.

##### Parametreler

- **model** (`nn.Module`): Eğitilecek model
- **config** (`TrainingConfig` veya `Dict[str, Any]`, opsiyonel): Eğitim yapılandırması (sözlük olarak verilebilir)
- **adapter_manager** (`AdapterManager`, opsiyonel): AdapterManager örneği
- **save_dir** (`str` veya `Path`, opsiyonel): Model checkpoint'lerinin kaydedileceği dizin
- **use_tensorboard** (`bool`, varsayılan=`False`): TensorBoard kullanılsın mı
- **use_early_stopping** (`bool`, varsayılan=`True`): Erken durdurma kullanılsın mı
- **use_lr_scheduler** (`bool`, varsayılan=`True`): Öğrenme oranı zamanlayıcısı kullanılsın mı
- **memory_optimization_level** (`str`, varsayılan=`"moderate"`): Bellek optimizasyon seviyesi
- **additional_callbacks** (`List[TrainingCallback]`, opsiyonel): Ek callback'ler

##### Dönüş Değeri

- `TrainingManager`: Oluşturulan TrainingManager örneği

## AdapterTrainingConfig Yapılandırması

TrainingManager sınıfı, adapter eğitimi için özelleştirilmiş `AdapterTrainingConfig` yapılandırma sınıfını kullanır. Standart eğitim konfigürasyonuna ek olarak, şu parametreleri sunar:

- **train_adapter_names** (`List[str]`, opsiyonel): Eğitilecek adapter isimlerinin listesi. `None` ise tüm adapter'lar eğitilir.
- **train_task_heads** (`bool`, varsayılan=`True`): Görev başlıklarını eğit
- **freeze_core_model** (`bool`, varsayılan=`True`): Çekirdek modeli dondur
- **checkpoint_frequency** (`int`, varsayılan=`1`): Kaç epoch'ta bir checkpoint oluşturulacak
- **checkpoint_keep_best_n** (`int`, varsayılan=`3`): Saklanacak en iyi checkpoint sayısı
- **memory_optimization_level** (`str`, varsayılan=`"moderate"`): Bellek optimizasyon seviyesi ("low", "moderate", "aggressive")

## Kullanılan Örüntüler

TrainingManager ve ilgili modüller, aşağıdaki tasarım örüntülerini kullanır:

- **FactoryMethod (PT-002)**: Farklı eğitim yöneticileri oluşturma
- **DecoratorPattern (PT-017)**: İşlevselliği dinamik olarak genişletme  
- **ModelComposite (PT-003)**: Modüler model mimarisi
- **TrainingLoopTemplate (PT-013)**: Eğitim döngüsü şablonu
- **MetricsCollector (PT-008)**: Performans metriklerini toplama ve raporlama

## İlgili Sınıflar

- **AdapterTrainingManager**: Adapter eğitiminin temel uygulamasını sağlar
- **TrainingCallback**: Eğitim sürecindeki çeşitli aşamalarda çağrılabilecek callback arayüzü
- **ModelStatisticsCallback**: Model istatistiklerini toplayan ve loglayan callback
- **GradientCheckCallback**: Gradient değerlerini kontrol eden ve aşırı/sıfır gradient problemlerini tespit eden callback
- **MemoryTrackingCallback**: Bellek kullanımını takip eden callback
- **LearningRateMonitorCallback**: Öğrenme oranını takip eden callback 