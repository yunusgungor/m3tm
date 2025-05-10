# AdapterSlot Mekanizmasının ProtoTransformerBlock Entegrasyonu

Bu doküman, M³TM modelinde AdapterSlot mekanizmasının ProtoTransformerBlock ile entegrasyonunu açıklar.

**Versiyon:** 1.1  
**Tarih:** 2024-06-06  
**Güncelleme:** 2024-06-07 - Performans optimizasyonları ve çoklu adapter desteği  
**İlgili Hikaye:** [story_12 - AdapterSlot mekanizmasının ProtoTransformerBlock entegrasyonu](.project_meta/.stories/story_12.json)

## 1. Genel Bakış

AdapterSlot mekanizması, M³TM modelinin çekirdeğini değiştirmeden adapte edilebilir ve kişiselleştirilebilir olmasını sağlar. Bu mekanizma, ProtoTransformerBlock içerisinde belirli pozisyonlara Adapter modüllerinin dinamik olarak eklenmesine ve çıkarılmasına izin verir.

## 2. Temel Mimari

### 2.1. AdapterSlot Sınıfı

`AdapterSlot` sınıfı, Transformer bloğu içinde adaptör modüllerinin eklenip çıkarılabileceği bir "yuva" sağlar. 

Temel özellikleri:
- Birden fazla adaptör modülünü destekler
- Adaptör olmadığında kimlik (identity) fonksiyonu olarak davranır
- Eğitim ve çıkarım modları arasında geçiş yapabilir
- Adaptörleri isimle referans edebilir
- Adaptörleri sıralı olarak uygular

```python
class AdapterSlot(nn.Module):
    def __init__(self, config: AdapterConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.bottleneck_dim = config.bottleneck_dim
        
        # Birden fazla adaptörü desteklemek için liste kullanılıyor
        self.adapters = nn.ModuleList([])
        self.adapter_names = []
        
        # Eğitim modu bayrağı
        self.training_mode = True
        
        if config.enabled and config.initial_adapter_type is not None:
            adapter = self._create_adapter(config.initial_adapter_type)
            if adapter:
                self.adapters.append(adapter)
                self.adapter_names.append("default")
```

### 2.2. ProtoTransformerBlock Entegrasyonu

ProtoTransformerBlock, AdapterSlot'ları belirli pozisyonlarda içerir ve ileri geçiş sırasında bunları uygular:

```python
def __init__(self, config: ProtoTransformerConfig):
    # ... diğer başlatmalar ...
    
    # Adapter yuvaları
    if config.adapter_config.enabled:
        # Tüm pozisyonlar için adapter yuvaları oluştur
        self.adapter_slots = create_adapter_slots(
            config.adapter_config,
            config.hidden_size,
            ["pre_attention", "post_attention", "pre_ffn", "post_ffn"]
        )
    else:
        self.adapter_slots = {}
```

İleri geçiş sırasında adaptörlerin uygulanması:

```python
def _apply_adapters(self, x: torch.Tensor, position: str) -> torch.Tensor:
    """Belirli bir pozisyondaki adaptörleri uygular."""
    if position in self.adapter_slots:
        return self.adapter_slots[position](x)
    return x
```

## 3. Adaptör Pozisyonları

Transformer bloğu içinde dört adaptör pozisyonu desteklenir:

1. **pre_attention**: Dikkat mekanizmasından önce
2. **post_attention**: Dikkat mekanizmasından sonra
3. **pre_ffn**: Feed-forward ağdan önce
4. **post_ffn**: Feed-forward ağdan sonra

Bu pozisyonlar, adaptörlerin Transformer bloğunun farklı bölümlerini etkilemesine izin verir.

## 4. Adaptör Yönetimi

ProtoTransformerBlock şu adaptör yönetim fonksiyonlarını sağlar:

- `register_adapter(adapter, position, name)`: Belirtilen pozisyona bir adaptör ekler
- `remove_adapter(position, name)`: Belirtilen pozisyondaki adaptörü kaldırır
- `get_adapter(position, name)`: Belirtilen pozisyondaki adaptörü alır
- `get_adapter_positions()`: Desteklenen adaptör pozisyonlarını döndürür
- `get_adapter_names(position)`: Belirli bir pozisyondaki tüm adaptör isimlerini döndürür
- `set_training_adapters(training)`: Adaptörlerin eğitim modunu ayarlar

## 5. Çoklu Adaptör Desteği

Geliştirilmiş AdapterSlot mekanizması, aynı pozisyonda birden fazla adaptör modülünü destekler. İleri geçiş sırasında adaptörler sıralı olarak uygulanır:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Adaptör yoksa veya eğitim modu kapalıysa ve çıkarım modundaysak
    if not self.adapters or (not self.training_mode and not self.training):
        return x
    
    # Adaptörleri sıralı olarak uygula
    output = x
    for adapter in self.adapters:
        output = adapter(output)
    
    return output
```

## 6. Eğitim ve Çıkarım Modları

AdapterSlot mekanizması, eğitim ve çıkarım modları arasında geçiş yapabilir. Bu, çıkarım sırasında adaptörlerin devre dışı bırakılmasına olanak tanır:

```python
def set_training_adapters(self, training: bool) -> None:
    """Adaptörlerin eğitim modunu ayarlar."""
    self.training_adapters = training
    for pos, slot in self.adapter_slots.items():
        slot.set_training_mode(training)
```

## 7. Performans Etkileri

AdapterSlot mekanizması, çekirdeğe minimal ek yük getirmek üzere tasarlanmıştır:

- Adaptörler yoksa veya devre dışıysa, neredeyse hiç ek hesaplama gerektirmez (hızlı bir kontrol dışında)
- Adaptör parametrelerinin sayısı, genellikle temel modelin %1-5'i kadardır
- Çoklu adaptör kullanımı, doğrusal bir hesaplama artışına neden olur
- Tipik bir 256 boyutlu hidden size için 32 boyutlu bir darboğaz kullanıldığında, FLOP sayısında yaklaşık %2-3 civarında bir artış gözlenir
- Adaptörlerin erken kontrol optimizasyonu sayesinde çıkarım modunda neredeyse hiç performans etkisi yoktur

## 8. Örnek Kullanım

```python
# Adaptörü etkinleştiren bir konfigürasyon oluştur
adapter_config = AdapterConfig(enabled=True, bottleneck_dim=32)
config = ProtoTransformerConfig(hidden_size=768, adapter_config=adapter_config)

# Model oluştur
model = ProtoTransformerBlock(config)

# Yeni bir adaptör ekle
adapter = Adapter(
    config=AdapterConfig(bottleneck_dim=32),
    input_dim=config.hidden_size
)
model.register_adapter(adapter, "post_attention", "my_adapter")

# Çıkarım sırasında adaptörleri devre dışı bırak
model.set_training_adapters(False)
model.eval()
with torch.no_grad():
    output, _ = model(input_tensor)
```

## 9. İlişkili Örüntüler

- **ModelComposite (PT-003)**: Ana modelin içine küçük ve özelleştirilmiş modül ekleme 
- **CompositeAdapter (PT-014)**: Birden fazla adaptörü sıralı olarak uygulama
- **PluggableComponentStrategy**: Değiştirilebilir dikkat ve FFN mekanizmaları

## 10. Test ve Doğrulama

AdapterSlot mekanizmasının ProtoTransformerBlock entegrasyonu, kapsamlı test senaryoları içeren `adapter_integration_test.py` dosyası ile test edilmiştir. Bu testler:

- Tekli ve çoklu adaptör ekleme/kaldırma işlevselliğini
- Eğitim ve çıkarım modları arasında geçişi
- Adaptörlerin model çıktısı üzerindeki etkisini
- Performans etkilerini

doğrular.

## 11. İleri ve Geriye Transfer Yetenekleri

AdapterSlot mekanizması, modelin bilgi transferi yeteneklerini geliştirmek için tasarlanmıştır:

### 11.1. İleri Transfer (Forward Transfer)

Bu mekanizma, bir model özelleştirmesinin diğer görevlere aktarılmasını kolaylaştırır:

- **Görev Bağımsız Adaptörler**: Adaptörler, belirli veri veya görev türleri için eğitilebilir ve daha sonra başka görevlerde kullanılabilir.
- **Kademeli Öğrenme**: Yeni adaptörler, mevcut adaptörlerle birleştirilerek karmaşık görevlerde kademeli öğrenme sağlar.
- **Bilgi Destilasyonu**: Uzman adaptörlerdeki bilgi, diğer adaptörlere veya çekirdek modele aktarılabilir.

### 11.2 Geriye Transfer (Backward Transfer)

Yeni öğrenilen bilgilerin eski görevlere aktarılması sağlanır:

- **Adaptör Kompozisyonu**: Farklı adaptörlerin bir araya getirilmesiyle, çoklu görev bilgisi birleştirilebilir.
- **Adaptör Karışımı**: Farklı adaptörlerin çıktıları ağırlıklandırılarak karıştırılabilir (şu anki implemantasyonda sıralı uygulama kullanılmıştır).
- **Sürekli Öğrenme**: Yeni adaptörler eklendikçe, model bilgi unutmadan (catastrophic forgetting olmadan) yeni görevler öğrenebilir.

### 11.3. Performans Optimizasyonları

Adaptör mekanizması, çeşitli performans optimizasyonları içerir:

- **Erken Çıkış Kontrolü**: Adaptör olmadığında veya çıkarım modundayken hızlı bir kontrol ile minimal işlem yapılır.
- **Eğitim/Çıkarım Mod Kontrolü**: Çıkarım sırasında adaptörler tamamen devre dışı bırakılabilir.
- **Etkin Parametre Kullanımı**: Darboğaz mimarisi sayesinde, minimal parametre ile maksimum ifade gücü elde edilir.
- **Seçici Adaptör Etkinleştirme**: Sadece belirli pozisyonlardaki adaptörler etkinleştirilerek hesaplama verimliliği artırılır.

## 12. Gelecek Geliştirmeler

- Adaptör füzyon stratejileri (sıralı, paralel, ağırlıklı)
- Adaptör distilasyonu ve model küçültme
- Otomatik adaptör konfigürasyon optimizasyonu
- Adaptör paylaşımı ve yeniden kullanımı mekanizmaları
- Farklı darboğaz boyutlarına sahip adaptörlerin dinamik oluşturulması
- Eğitimli adaptör kitaplığı ve otomatik adaptör seçimi mekanizması 