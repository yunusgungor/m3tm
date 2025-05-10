# M³TM Adapter Modülü

Bu modül, M³TM modeli için etkili bir adapter implementasyonu sağlar. Adapter'lar, ana modeli değiştirmeden küçük, özelleştirilmiş modül eklemek için kullanılır ve parametre verimliliği sağlar.

## Adapter Nedir?

Adapter'lar, büyük dil modellerini verimli bir şekilde adapte etmek için 2019'da [Houlsby et al.](https://arxiv.org/abs/1902.00751) tarafından tanıtılan bir tekniktir. Temel fikir, tüm modeli fine-tune etmek yerine, küçük "bottleneck" katmanları ekleyip sadece bu katmanları eğitmektir. Bu yaklaşım şu avantajları sağlar:

- Parametre verimliliği (büyük modellerde %<1 ek parametre)
- Depolama verimliliği (her görev başına tek bir büyük model yerine küçük adapter'lar)
- Catastrophic forgetting (katastrofik unutma) olmadan çoklu görev adaptasyonu
- Modülerlik ve ölçeklenebilirlik

## Modül Yapısı

M³TM adapter modülü aşağıdaki bileşenlerden oluşur:

- `adapter.py`: Temel adapter sınıfları ve yapılandırma
  - `AdapterConfig`: Adapter yapılandırma sınıfı
  - `Adapter`: Temel abstract adapter sınıfı
  - `BottleneckAdapter`: Standart Houlsby-stil bottleneck adapter
  - `ParallelAdapter`: Alternatif paralel adapter mimarisi (MAD-X)
  - `create_adapter`: Fabrika fonksiyonu

- `adapter_manager.py`: Adapter'ları yönetmek için merkezi sınıf
  - `AdapterRegistration`: Adapter kayıt bilgilerini tutan veri sınıfı
  - `AdapterManager`: Model içindeki adapter'ları yöneten sınıf

- `adapter_utils.py`: Yardımcı fonksiyonlar
  - Adapter destekleyen modülleri bulma
  - Kolay adapter ekleme/kaldırma
  - Parametre hesaplama
  - Sadece adapter'ları eğitme

## Desteklenen Adapter Türleri

1. **Bottleneck Adapter (Varsayılan)**: Standart Houlsby adapter'ı. Down-projeksiyon, aktivasyon, up-projeksiyon ve artık bağlantı içerir.

   ```
   Input --> [Down Projection] --> [Activation] --> [Up Projection] --> [+ Input] --> Output
   ```

2. **Parallel Adapter**: MAD-X stilinde paralel adapter. Down-projeksiyon, aktivasyon, up-projeksiyon ve ölçeklendirilmiş bağlantı içerir.

   ```
   Input --> [Layer Norm] --> [Down Projection] --> [Activation] --> [Up Projection] --> [* α] --> [+ Input] --> Output
   ```

## Kullanım Örneği

```python
import torch
from m3tm.transformer.config import ProtoTransformerConfig
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.adapters.adapter import AdapterConfig, AdapterType
from m3tm.adapters.adapter_manager import AdapterManager

# Model oluştur
config = ProtoTransformerConfig(hidden_size=64)
transformer = ProtoTransformerBlock(config)

# AdapterManager oluştur
manager = AdapterManager(transformer)

# Adapter yapılandırması oluştur
adapter_config = AdapterConfig(
    adapter_type=AdapterType.BOTTLENECK,
    bottleneck_dim=16,
    use_layer_norm=True,
    activation="gelu"
)

# Adapter ekle
adapter_id = manager.register_adapter(
    transformer,
    "my_adapter",
    "post_attention",  # Desteklenen pozisyonlar: "pre_attention", "post_attention", "pre_ffn", "post_ffn"
    adapter_config
)

# Sadece adapter'ları eğit
from m3tm.adapters.adapter_utils import freeze_model_except_adapters
freeze_model_except_adapters(transformer)

# İleri geçiş
input_tensor = torch.randn(1, 10, 64)  # batch_size, seq_len, hidden_size
output, _ = transformer(input_tensor)

# Adapter'ı devre dışı bırak
manager.deactivate_adapter(adapter_id)

# Adapter'ı etkinleştir
manager.activate_adapter(adapter_id)

# Adapter'ı kaldır
manager.remove_adapter(adapter_id)

# Adapter'ları kaydet
manager.save_adapters("./adapters")

# Adapter'ları yükle
loaded_adapters = manager.load_adapters("./adapters")
```

## Gelişmiş Özellikler

- **Çoklu Adapter Konumları**: Transformer bloğunun farklı noktalarına adapter ekleyebilirsiniz
- **Parametre Verimliliği**: Adapter'lar, modelin %1'inden daha az ek parametre ekler
- **Merkezi Yönetim**: AdapterManager tüm adapter'ları merkezi olarak yönetir
- **Otomatik Kayıt ve Tanıma**: Adapter destekleyen modüller otomatik olarak keşfedilir
- **Esnek Yapılandırma**: Adapter özelliklerini ihtiyaçlarınıza göre yapılandırabilirsiniz
- **Kolay Serialization**: Adapter'ları kolayca kaydedin ve yükleyin
- **Otomatik Keşif**: Modeldeki tüm adapter-uyumlu modülleri otomatik olarak bulun

## Tasarım Kalıpları

Bu modül aşağıdaki tasarım kalıplarını kullanır:

- **ConfigurationDataclass (PT-001)**: Yapılandırma parametrelerini dataclass olarak modelleme
- **FactoryMethod (PT-002)**: Farklı adapter türlerini oluşturmak için fabrika metodu
- **ModelComposite (PT-003)**: Ana modelin içine küçük ve özelleştirilmiş modül ekleme
- **ConfigValidationPipeline (PT-009)**: Yapılandırma doğrulama adımlarını bir boru hattında birleştirme
- **MetricsCollector (PT-008)**: Adapter parametrelerini toplamak ve izlemek için

## Referanslar

- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) (Houlsby et al., 2019)
- [MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/abs/2005.00052) (Pfeiffer et al., 2020) 