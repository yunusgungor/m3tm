# M³TM Adapter ve Görev Başlığı Eğitim Pattern'leri

Bu belge, M³TM modelinde bulunan Adapter ve Görev Başlığı eğitim mekanizmasında kullanılan yazılım pattern'lerini açıklar. Bu pattern'ler, çekirdek modeli değiştirmeden modeli özelleştirmek için etkin bir eğitim yaklaşımı sunar.

## Pattern Özeti

M³TM Adapter ve Görev Başlığı eğitim mekanizması, aşağıdaki temel pattern'leri kullanır:

| Pattern ID | İsim | Kategori | Açıklama |
|------------|------|----------|----------|
| PT-018 | FrozenBackboneTraining | Eğitim | Çekirdek modeli dondurup sadece adapter'lar ve görev başlıklarını eğiten pattern |
| PT-019 | TrainingCallbackHook | Davranışsal | Eğitim sürecinin farklı aşamalarında özelleştirilebilir işlemler sunan pattern |
| PT-020 | MemoryOptimization | Performans | Mobil cihazlarda eğitim için bellek kullanımını optimize eden pattern |

## FrozenBackboneTraining (PT-018)

### Açıklama

Bu pattern, çekirdek modelin parametrelerini dondurarak (`requires_grad=False`), sadece adapter'ları ve görev başlıklarını eğitmeyi sağlar. Bu sayede, modelin büyük bir kısmını korurken, yalnızca küçük bir kısmını (genellikle %1-5) özelleştirerek eğitim yapmak mümkün olur.

### Faydaları

- Eğitim süresini önemli ölçüde azaltır
- Bellek kullanımını düşürür
- Aşırı uyumu (overfitting) azaltır
- Mobil cihazlarda bile model adaptasyonunu mümkün kılar

### Örnek Kullanım

```python
from m3tm.training.adapter_training import AdapterTrainingConfig, AdapterTrainingManager

# Eğitim yapılandırması
config = AdapterTrainingConfig(
    train_adapter_names=["domain_adapter"],  # Eğitilecek adapter isimleri
    train_task_heads=True,                   # Görev başlıklarını eğit
    freeze_core_model=True                   # Çekirdek modeli dondur
)

# Eğitim yöneticisi
manager = AdapterTrainingManager(model, config, adapter_manager)

# Modeli eğitim için hazırla (çekirdek modeli dondurur)
manager.prepare_model_for_training()

# Eğitimi başlat
manager.train(train_loader, criterion)
```

### İmplementasyon Detayları

Bu pattern, modelin çekirdek bileşenlerini tanımlayıp, bu bileşenlerin gradientlerini kapatarak çalışır:

```python
def prepare_model_for_training(self):
    # Tüm parametreleri dondur
    if self.config.freeze_core_model:
        for param in self.model.parameters():
            param.requires_grad = False
            
    # Adapter'ları eğitilebilir yap
    if self.adapter_manager is not None:
        self.adapter_manager.set_adapters_trainable(self.config.train_adapter_names)
        
    # Görev başlıklarını eğitilebilir yap
    if self.config.train_task_heads:
        # Görev başlıklarını bul ve eğitilebilir yap
        for name, submodule in self.model.named_modules():
            if isinstance(submodule, (ClassificationHead, RegressionHead)):
                for param in submodule.parameters():
                    param.requires_grad = True
```

## TrainingCallbackHook (PT-019)

### Açıklama

Bu pattern, eğitim sürecinin farklı aşamalarında (başlangıç, epoch başı/sonu, batch başı/sonu, eğitim sonu) çağrılan özelleştirilebilir hook'lar sunar. Bu sayede, eğitim süreci modüler ve genişletilebilir hale gelir.

### Faydaları

- Eğitim sürecini değiştirmeden özelleştirme imkanı
- Modüler eğitim süreçleri
- Daha iyi test edilebilirlik
- Farklı callback kombinasyonları ile esnek konfigürasyon

### Örnek Kullanım

```python
from m3tm.training.adapter_training import TrainingCallback, EarlyStoppingCallback
from m3tm.training.adapter_training_utils import ModelStatisticsCallback

# Erken durdurma callback'i
early_stopping = EarlyStoppingCallback(patience=3, monitor="val_loss")

# Model istatistikleri callback'i
stats_callback = ModelStatisticsCallback(log_frequency=1)

# Özel callback
class MyCallback(TrainingCallback):
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        print(f"Epoch {epoch} tamamlandı: {metrics}")

# Callback'leri eğitim yöneticisine ekle
callbacks = [early_stopping, stats_callback, MyCallback()]
manager = AdapterTrainingManager(model, config, adapter_manager, callbacks=callbacks)
```

### İmplementasyon Detayları

Ana callback sınıfı, eğitim sürecinin çeşitli aşamalarında çağrılan metotları tanımlar:

```python
class TrainingCallback:
    def on_training_start(self, trainer, **kwargs):
        pass
    
    def on_epoch_start(self, trainer, epoch, **kwargs):
        pass
    
    def on_batch_start(self, trainer, batch, **kwargs):
        pass
    
    def on_batch_end(self, trainer, batch_metrics, **kwargs):
        pass
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        pass
    
    def on_training_end(self, trainer, **kwargs):
        pass
```

## MemoryOptimization (PT-020)

### Açıklama

Bu pattern, özellikle mobil cihazlarda eğitim yaparken bellek kullanımını optimize etmek için çeşitli stratejiler sunar. Farklı optimizasyon seviyelerinde (düşük, orta, yüksek), farklı bellek optimizasyon teknikleri uygulanır.

### Faydaları

- Sınırlı bellek kaynağına sahip cihazlarda eğitimi mümkün kılar
- Bellek taşmalarını önler
- Daha büyük batch boyutlarına olanak tanır
- Eğitim sürecinin stabilitesini artırır

### Örnek Kullanım

```python
from m3tm.training.adapter_training import AdapterTrainingConfig

# Bellek optimizasyonu ile yapılandırma
config = AdapterTrainingConfig(
    # Diğer parametreler...
    memory_optimization_level="aggressive"  # "low", "moderate" veya "aggressive"
)

# Bellek izleme callback'i ekleyebilirsiniz
from m3tm.training.adapter_training_utils import MemoryTrackingCallback
memory_callback = MemoryTrackingCallback(log_frequency=10)
```

### İmplementasyon Detayları

Bellek optimizasyon seviyelerine göre farklı stratejiler uygulanır:

```python
def _optimize_memory(self, level: str = "moderate"):
    """Bellek kullanımını optimize eder."""
    if level == "low":
        # Sadece gradient checkpointing
        torch.utils.checkpoint.checkpoint_sequential = True
    
    elif level == "moderate":
        # Gradient checkpointing + model tamponu temizleme
        torch.utils.checkpoint.checkpoint_sequential = True
        torch.cuda.empty_cache()
        
    elif level == "aggressive":
        # Tüm optimizasyonlar
        torch.utils.checkpoint.checkpoint_sequential = True
        torch.cuda.empty_cache()
        
        # 16-bit hassasiyet kullanımı
        self.use_mixed_precision = True
        
        # Ara bellekleri temizle
        for module in self.model.modules():
            if hasattr(module, 'preserve_activations'):
                module.preserve_activations = False
```

## Pattern İlişkileri

Bu pattern'ler birbirleriyle ve diğer pattern'lerle şu şekilde ilişkilidir:

1. **FrozenBackboneTraining**, TrainingCallbackHook ve MemoryOptimization pattern'lerini kullanır.
2. **TrainingCallbackHook**, TrainingLoopTemplate (PT-013) pattern'ini özelleştirir.
3. **CompositeAdapter** (PT-014), FrozenBackboneTraining tarafından kullanılır ve adapter'ların modele eklenmesini sağlar.

## Anti-Pattern'ler ve Bunlardan Kaçınma

Adapter ve görev başlığı eğitiminde şu anti-pattern'lerden kaçınılmalıdır:

1. **MonolithicTraining (AP-001):** Tüm modeli eğitmek yerine, FrozenBackboneTraining kullanın.
2. **CallbackHell (AP-002):** Callback'leri basit ve tek görevli tutun, birbirleriyle aşırı etkileşimden kaçının.
3. **DeepAdapterStack (AP-003):** Çok fazla adapter eklemek yerine, stratejik olarak 1-2 adapter kullanın.

## En İyi Uygulamalar

1. **Adapter Boyutunu Dikkatli Seçin:**
   - Küçük (`bottleneck_dim=16`): Hızlı eğitim, az parametre
   - Orta (`bottleneck_dim=64`): İyi performans/hız dengesi
   - Büyük (`bottleneck_dim=128+`): Daha iyi performans, yavaş eğitim

2. **Gerekli Bellek Optimizasyonunu Kullanın:**
   - Yüksek RAM cihazlarda: "low"
   - Orta düzey cihazlarda: "moderate"
   - Düşük RAM cihazlarda: "aggressive"

3. **Callback Hiyerarşisini Düzenli Tutun:**
   - Her callback'i belirli bir göreve odaklayın
   - Callback'ler arası bağımlılıkları minimumda tutun
   - Callback çağrı sırasını göz önünde bulundurun

4. **Debug Seçeneklerini Kullanın:**
   - Eğitilebilir parametre sayısını kontrol edin
   - Bellek kullanımını izleyin
   - Performans metriklerini takip edin

## Örnek Uygulama

Tam bir uygulama örneği için şu dosyaya bakabilirsiniz:
- `.project_meta/.docs/tutorials/code_samples/adapter_training_tutorial.py`

## Daha Fazla Bilgi

- [Adapter ve Görev Başlığı Eğitimi Kılavuzu](../guides/adapter_training_guide.md)
- [Adapter Eğitim API Referansı](./adapter_training.md)
- [Pattern Görselleştirmeleri](../../.patterns/visualization/adapter_training_patterns.md)
- [Pattern İnceleme Raporu](../../.patterns/reviews/adapter_training_pattern_review.md) 