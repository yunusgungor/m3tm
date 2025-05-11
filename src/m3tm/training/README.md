# M³TM Eğitim Modülü

Bu modül, M³TM modelinin eğitim süreçlerini yönetmek için gerekli tüm bileşenleri içerir. Özellikle çekirdek modelin dondurularak sadece adapter'ların ve görev başlıklarının eğitimi için optimize edilmiş mekanizmalar sağlar.

## Modül Yapısı

```
training/
├── __init__.py              - Modül dışa aktarım tanımları
├── adapter_training.py      - Adapter eğitimi yöneticisi
├── adapter_training_utils.py - Adapter eğitimi yardımcı fonksiyonları
├── dataset.py               - Veri kümesi sınıfları
├── metrics.py               - Metrik hesaplama ve toplama
├── trainer.py               - Genel eğitim şablonu ve sınıflandırma eğiticisi
├── training_manager.py      - Üst düzey eğitim yönetim API'si
└── example.py               - Kullanım örnekleri
```

## Temel Bileşenler

### TrainingManager

Üst düzey API sağlayan sınıf. Hem çekirdek modelin hem de adapter'ların ve görev başlıklarının eğitimini koordine eder. Aşağıdaki özellikleri sağlar:

- Çekirdek modeli dondurma
- Belirli adapter'ları seçerek eğitme
- Eğitim metriklerini toplama ve raporlama
- Düzenli checkpoint oluşturma
- Callback ile eğitim sürecini takip etme
- Bellek optimizasyonu

Detaylı API dokümantasyonu: `.project_meta/.docs/api/training_manager.md`

### AdapterTrainingManager

`TrainingManager` tarafından kullanılan, adapter eğitiminin temel uygulamasını sağlayan sınıf. Aşağıdaki temel işlevleri gerçekleştirir:

- Eğitilecek parametrelerin seçimi
- Gradyan hesaplama ve geri yayılım
- Optimizasyon stratejileri
- Doğrulama ve değerlendirme

### TrainingCallback

Eğitim sürecinin farklı aşamalarında çağrılabilen callback arayüzü. Şu temel callback'ler sağlanır:

- `EarlyStoppingCallback` - Erken durdurma
- `LearningRateSchedulerCallback` - Öğrenme oranı ayarlama
- `ModelStatisticsCallback` - Model istatistiklerini toplama
- `GradientCheckCallback` - Gradient değerlerini kontrol etme
- `MemoryTrackingCallback` - Bellek kullanımını takip etme
- `LearningRateMonitorCallback` - Öğrenme oranını takip etme

## Bellek Optimizasyonu

Bu modül, mobil cihazlarda eğitim için bellek optimizasyonu stratejileri sağlar:

- **Düşük Seviye:** Temel torch.cuda.empty_cache() çağrıları
- **Orta Seviye:** Gradient biriktirme, mixed precision eğitim
- **Yüksek Seviye:** Gradient checkpointing, etkin önbellek yönetimi

## Kullanılan Örüntüler

- `TrainingLoopTemplate (PT-013)` - Eğitim döngüsü şablonu
- `ModelComposite (PT-003)` - Modüler model mimarisi
- `FactoryMethod (PT-002)` - Eğitim yöneticileri oluşturma
- `MetricsCollector (PT-008)` - Performans metriklerini toplama
- `DecoratorPattern (PT-017)` - İşlevselliği dinamik olarak genişletme

## Örnek Kullanım

Temel bir adapter eğitimi örneği:

```python
from m3tm.training import TrainingManager

# TrainingManager oluştur
trainer = TrainingManager.create(
    model=model,
    config={
        "learning_rate": 1e-3,
        "epochs": 5,
        "train_adapter_names": ["adapter1"],  # Sadece adapter1'i eğit
        "train_task_heads": True,             # Görev başlığını da eğit
        "freeze_core_model": True,            # Çekirdek modeli dondur
    },
    use_early_stopping=True,
    save_dir="./checkpoints"
)

# Eğitimi çalıştır
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader
)
```

Detaylı örnekler için `example.py` dosyasına bakabilirsiniz. 