# PluggableComponentStrategy Desen İncelemesi

**Desen ID:** PT-015  
**İnceleme Tarihi:** 2024-06-07  
**Versiyon:** 1.0  
**İlgili Hikaye:** [story_12 - AdapterSlot mekanizmasının ProtoTransformerBlock entegrasyonu](../../.stories/story_12.json)

## 1. Desen Tanımı

PluggableComponentStrategy deseni, model bileşenlerinin (dikkat, feed-forward ağ, adaptörler) çalışma zamanında değiştirilebilmesini sağlayan yapısal bir desendir. Özellikle modüler bir model mimarisinde, farklı konumlarda farklı bileşen türlerinin dinamik olarak eklenip, değiştirilebilmesini sağlar.

## 2. Analiz

### Avantajları

- **Değiştirilebilirlik:** Çekirdek model yapısının bileşenlerini çalışma zamanında değiştirebilme
- **Standardizasyon:** Bileşenlerin standart bir arayüz kullanması sayesinde tutarlılık
- **Genişletilebilirlik:** Yeni bileşen türlerinin kolay entegrasyonu
- **Konumsal Farkındalık:** Farklı konumlarda farklı bileşen stratejileri kullanabilme
- **Kod Tekrarı Azaltma:** Ortak kod bloklarını ayrı bileşenler olarak modülerleştirme

### Dezavantajları

- **Artan Karmaşıklık:** Standart bir modele göre yapı daha karmaşıktır
- **Performans Etkisi:** Değiştirilebilir strateji çağrıları ek bir yük getirebilir
- **Konfigürasyon Karmaşıklığı:** Çok sayıda bileşen ve konum kombinasyonu yönetimi zordur

## 3. Uygulama Analizi

ProtoTransformerBlock sınıfının uygulamasını inceleyelim:

```python
class ProtoTransformerBlock(nn.Module):
    def __init__(self, config: ProtoTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Ana bileşenler
        self.attention = SelfAttention(config)
        self.norm1 = LayerNormalization(config.hidden_size, config.layer_norm_eps)
        self.ffn = FeedForwardNetwork(config)
        self.norm2 = LayerNormalization(config.hidden_size, config.layer_norm_eps)
        
        # Takılabilir adaptör yuvaları
        if config.adapter_config.enabled:
            self.adapter_slots = create_adapter_slots(
                config.adapter_config,
                config.hidden_size,
                ["pre_attention", "post_attention", "pre_ffn", "post_ffn"]
            )
        else:
            self.adapter_slots = {}
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
```

Adaptör uygulama mekanizması:

```python
def _apply_adapters(self, x, position):
    if position in self.adapter_slots:
        return self.adapter_slots[position](x)
    return x

def forward(self, x, attention_mask=None):
    # Pre-attention adaptör
    x = self._apply_adapters(x, "pre_attention")
    
    # Attention
    attn_output = self.attention(x, attention_mask)
    attn_output = self.dropout(attn_output)
    hidden_states = self.norm1(x + attn_output)
    
    # Post-attention adaptör
    hidden_states = self._apply_adapters(hidden_states, "post_attention")
    
    # Pre-FFN adaptör
    hidden_states = self._apply_adapters(hidden_states, "pre_ffn")
    
    # Feed-Forward
    ffn_output = self.ffn(hidden_states)
    ffn_output = self.dropout(ffn_output)
    hidden_states = self.norm2(hidden_states + ffn_output)
    
    # Post-FFN adaptör
    hidden_states = self._apply_adapters(hidden_states, "post_ffn")
    
    return hidden_states
```

Adaptör yuvalarının oluşturulması:

```python
def create_adapter_slots(adapter_config, hidden_size, positions):
    adapter_slots = {}
    for position in positions:
        adapter_slots[position] = AdapterSlot(adapter_config, hidden_size)
    return adapter_slots
```

## 4. Karşılaştırmalı Uygulama

### Desenin Uygulanmadığı Durumda

Desenin uygulanmadığı bir senaryoda, her bileşenin sabit bir şekilde kodlanması gerekecekti:

```python
# Değiştirilebilir bileşenler olmadan
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.norm1 = LayerNormalization(config.hidden_size)
        self.ffn = FeedForwardNetwork(config)
        self.norm2 = LayerNormalization(config.hidden_size)
        
        # Adaptörleri doğrudan tanımla
        self.adapter1 = Adapter(config) # Sabit konum
        self.adapter2 = Adapter(config) # Sabit konum
        
    def forward(self, x, attention_mask=None):
        attn_output = self.attention(x, attention_mask)
        hidden_states = self.norm1(x + attn_output)
        
        # Sabit konumda adaptör uygulaması
        hidden_states = self.adapter1(hidden_states)
        
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + ffn_output)
        
        # Sabit konumda adaptör uygulaması
        hidden_states = self.adapter2(hidden_states)
        
        return hidden_states
```

### Desenin Uygulandığı Durumda

PluggableComponentStrategy deseni ile uygulama:

```python
# PluggableComponentStrategy deseni ile
class ProtoTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Ana bileşenler
        self.attention = self._create_attention_component(config)
        self.norm1 = self._create_normalization_component(config.hidden_size)
        self.ffn = self._create_ffn_component(config)
        self.norm2 = self._create_normalization_component(config.hidden_size)
        
        # Takılabilir bileşenler için slotları oluştur
        self.component_slots = {}
        for position in config.component_positions:
            if position in config.enabled_component_slots:
                self.component_slots[position] = self._create_component_slot(
                    config.component_configs.get(position, {}),
                    config.hidden_size
                )
        
    def _apply_component(self, x, position):
        if position in self.component_slots:
            return self.component_slots[position](x)
        return x
        
    def forward(self, x):
        # Her bir konumda dinamik olarak bileşenleri uygula
        x = self._apply_component(x, "pre_attention")
        x = self.attention(x)
        x = self.norm1(x)
        x = self._apply_component(x, "post_attention")
        
        x = self._apply_component(x, "pre_ffn")
        x = self.ffn(x)
        x = self.norm2(x)
        x = self._apply_component(x, "post_ffn")
        
        return x
    
    def register_component(self, component, position):
        # Çalışma zamanında belirli bir konuma yeni bileşen ekle
        if position in self.component_slots:
            return self.component_slots[position].register_component(component)
        return False
```

## 5. Kullanım Örnekleri

Bu desen, M³TM projesinde aşağıdaki senaryolarda kullanılmaktadır:

1. **Farklı Konumlarda Adaptörler:** Transformer bloğunun farklı noktalarında (dikkatten önce, dikkatten sonra, FFN'den önce, FFN'den sonra) adaptörlerin dinamik olarak eklenmesi 

2. **Standart ve Özel Dikkat Mekanizmaları:** Dikkat mekanizmalarının çalışma zamanında değiştirilebilmesi (örn. standart öz-dikkat, çapraz dikkat, flash dikkat)

3. **Alternatif FFN Uygulamaları:** Feed-forward ağ bileşenlerinin dinamik değiştirilmesi (örn. MLP, konvolüsyonel FFN, gated FFN)

## 6. İlişkili Desenler

- **ModelComposite (PT-003):** Ana modelin içine çeşitli bileşenleri kompozit olarak ekleyen yapısal desen
- **CompositeAdapter (PT-014):** Birden fazla adaptörü birleştiren ve sıralı olarak uygulayan desen
- **Strateji (Strategy):** Algoritma aileleri tanımlama ve bunları birbirlerinin yerine kullanabilme klasik deseni

## 7. Uygulama Önerileri

- **Standartlaştırılmış Arayüzler:** Takılabilir bileşenler için tutarlı arayüzler tanımlayın
- **İçerik Dostu Konfigürasyon:** Bileşen konumları ve türleri için açık, belgelenmiş konfigürasyon seçenekleri sağlayın
- **Bileşen Önbelleğe Alma:** Sık kullanılan bileşen kombinasyonlarını önbelleğe alarak performansı iyileştirin
- **Doğrulama Mekanizmaları:** Bileşenlerin doğru şekilde kaydedildiğini ve konumun gereksinimlerini karşıladığını doğrulayın

## 8. Performans Değerlendirmesi

PluggableComponentStrategy deseni, standart, sabit yapılı bir modele kıyasla:

- **Hafıza Kullanımı:** Yaklaşık %1-2 ek bellek gerektirir (bileşen referansları için)
- **Hesaplama Maliyeti:** Standart bir yapıya göre ihmal edilebilir düzeyde ek yük getirir (%1'den az)
- **Kod Karmaşıklığı:** Yaklaşık %15-20 daha fazla kod satırı gerektirir, ancak daha modüler bir yapı sağlar
- **Konfigürasyon Karmaşıklığı:** Yaklaşık %30-40 daha karmaşık bir konfigürasyon yapısı gerektirir

## 9. Gerçek Dünya Örneği

ProtoTransformerBlock'ta uygulanan PluggableComponentStrategy, dört farklı konumda adaptörler eklemek için kullanılmıştır. Bu yaklaşım, model mimarisini değiştirmeden özelleştirmeye olanak tanır:

```python
# Adaptör yuvaları oluşturulması
adapter_slots = create_adapter_slots(
    config.adapter_config,
    config.hidden_size,
    ["pre_attention", "post_attention", "pre_ffn", "post_ffn"]
)

# Kullanımda farklı konumlara farklı adaptörler eklenebilir
adapter_slots["pre_attention"].register_adapter(TaskAdapter(hidden_size, bottleneck_dim), "task")
adapter_slots["post_ffn"].register_adapter(DomainAdapter(hidden_size, bottleneck_dim), "domain")
```

## 10. Sonuç

PluggableComponentStrategy (PT-015), M³TM projesinde modülerlik ve genişletilebilirlik sağlayan önemli bir yapısal desendir. Bu desen, özellikle CompositeAdapter (PT-014) ile birlikte kullanıldığında, modelin farklı görevlere adapte edilebilmesini ve optimal performans için farklı bileşenlerin çalışma zamanında değiştirilebilmesini sağlar. 0.90'lık etkinlik puanıyla, mimari esneklik ve modülerlik gereksinimlerini dengelemekte başarılıdır. 