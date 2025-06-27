# Gemini-2.5-Flash ile M³TM Eğitimi

Bu dokümantasyon, Google'ın Gemini-2.5-Flash modelini M³TM eğitimi için kullanmanın farklı yollarını açıklar.

## 🚀 Özellikler

### 1. **Synthetic Data Generation**
- Gemini'den kaliteli SFT ve GRPO eğitim verisi üretimi
- Çoklu konu desteği
- Türkçe ve İngilizce dil desteği
- JSON Lines formatında otomatik kaydetme

### 2. **Knowledge Distillation**
- Gemini'yi teacher model olarak kullanma
- M³TM'yi student model olarak eğitme
- Temperature scaling ile soft target learning
- Cache sistemi ile API maliyeti optimizasyonu

### 3. **API-based Reward Model**
- GRPO için Gemini'yi reward model olarak kullanma
- Response kalite skorlaması (0-10 arası)
- Preference comparison (-1 ile 1 arası)
- Özelleştirilebilir değerlendirme kriterleri

## 📋 Gereksinimler

### API Key
Google AI Studio'dan Gemini API key alın:
1. [Google AI Studio](https://aistudio.google.com/app/apikey)'ya gidin
2. API key oluşturun
3. Environment variable olarak ayarlayın:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

### Python Paketleri
```bash
pip install google-generativeai
```

## 🎯 Kullanım

### Hızlı Başlangıç

```bash
# Gemini entegrasyonunu test et
PYTHONPATH=src python examples/gemini_training.py
```

### Seçenekler

1. **Synthetic Data Generation**: Gemini'den eğitim verisi üret
2. **Knowledge Distillation**: Gemini'yi teacher olarak kullan
3. **Gemini-Enhanced SFT**: Gemini verisi ile SFT eğitimi
4. **Gemini Reward Model GRPO**: Gemini ile preference learning
5. **Tümü**: Tüm yaklaşımları sıralı çalıştır

## 📊 Yaklaşımlar

### 1. Synthetic Data Generation

```python
from m3tm.training.gemini_integration import GeminiClient, SyntheticDataGenerator

# Gemini client oluştur
gemini_client = GeminiClient(GeminiConfig())
data_generator = SyntheticDataGenerator(gemini_client)

# SFT verisi üret
sft_data = data_generator.generate_sft_data(
    topics=["Python", "AI", "Data Science"],
    num_samples_per_topic=10,
    language="Turkish"
)

# GRPO verisi üret
grpo_data = data_generator.generate_grpo_data(
    prompts=["Python nedir?", "AI nasıl çalışır?"],
    language="Turkish"
)
```

### 2. Knowledge Distillation

```python
from m3tm.training.gemini_distillation import GeminiDistillationTrainer

# Distillation trainer oluştur
trainer = GeminiDistillationTrainer(
    model=m3tm_model,
    config=DistillationConfig(),
    gemini_client=gemini_client,
    tokenizer=tokenizer
)

# Eğitimi çalıştır
results = trainer.train(train_dataset)
```

### 3. API-based Reward Model

```python
from m3tm.training.gemini_integration import GeminiRewardModel

# Reward model oluştur
reward_model = GeminiRewardModel(gemini_client)

# Response'ları skorla
scores = reward_model.score_responses(
    prompt="Python nedir?",
    responses=["Python bir programlama dilidir.", "Bilmiyorum."],
    criteria="doğruluk, yararlılık, açıklık"
)

# İki response'u karşılaştır
comparison = reward_model.compare_responses(
    prompt="Python nedir?",
    response_a="Python bir programlama dilidir.",
    response_b="Python bir yılandır."
)
```

## ⚙️ Konfigürasyon

### GeminiConfig

```python
@dataclass
class GeminiConfig:
    api_key: Optional[str] = None  # GEMINI_API_KEY'den alınır
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_day: int = 1000
```

### DistillationConfig

```python
@dataclass
class DistillationConfig:
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 3
    
    # Distillation parametreleri
    temperature: float = 4.0  # Softmax temperature
    alpha: float = 0.7  # Hard vs soft target balance
    beta: float = 0.3   # Student vs distillation loss balance
```

## 💰 Maliyet Optimizasyonu

### Cache Sistemi
- Teacher outputs otomatik cache'lenir
- Tekrar eğitimlerde API çağrısı yapılmaz
- Cache dosyaları: `cache/gemini_teacher_outputs_*.jsonl`

### Rate Limiting
- Dakikalık ve günlük request limitleri
- Otomatik bekleme ve retry mekanizması
- Exponential backoff ile hata yönetimi

### Batch Processing
- Prompts batch'ler halinde işlenir
- API çağrıları optimize edilir
- Paralel işleme desteği

## 📈 Performans İpuçları

### 1. Temperature Ayarları
- **Data Generation**: 0.7-0.9 (çeşitlilik için)
- **Knowledge Distillation**: 0.1-0.3 (tutarlılık için)
- **Reward Model**: 0.1 (objektif skorlama için)

### 2. Batch Size
- API rate limit'e göre ayarlayın
- Küçük batch'ler daha stabil
- Önerilen: 2-5 örnek per batch

### 3. Sequence Length
- Gemini max_output_tokens: 2048
- M³TM max_seq_length ile uyumlu tutun
- Önerilen: 128-512 token

## 🔧 Troubleshooting

### API Key Hataları
```bash
# API key kontrolü
echo $GEMINI_API_KEY

# API key ayarlama
export GEMINI_API_KEY='your-key-here'
```

### Rate Limit Hataları
- `requests_per_minute` ve `requests_per_day` değerlerini azaltın
- Batch size'ı küçültün
- Cache kullanarak tekrar çağrıları önleyin

### Memory Hataları
- Batch size'ı azaltın
- Sequence length'i kısaltın
- Gradient accumulation kullanın

## 📝 Örnek Çıktılar

### SFT Data Örneği
```json
{
  "instruction": "Python'da liste nasıl oluşturulur?",
  "response": "Python'da liste oluşturmak için köşeli parantez kullanılır: my_list = [1, 2, 3, 'hello']"
}
```

### GRPO Data Örneği
```json
{
  "prompt": "Yapay zeka nedir?",
  "chosen": "Yapay zeka, makinelerin insan benzeri düşünme yeteneklerini simüle etmesini sağlayan teknoloji dalıdır.",
  "rejected": "Yapay zeka robotlar demek."
}
```

## 🎯 Sonuçlar

### Beklenen Performans
- **Synthetic Data**: 10-50 örnek/dakika
- **Knowledge Distillation**: %10-30 performans artışı
- **Reward Model**: %5-15 preference accuracy artışı

### Maliyet Tahmini
- **Data Generation**: ~$0.01-0.05 per örnek
- **Knowledge Distillation**: ~$0.10-0.50 per epoch
- **Reward Model**: ~$0.001-0.01 per comparison

## 🔗 İlgili Dosyalar

- `src/m3tm/training/gemini_integration.py`: Ana entegrasyon modülü
- `src/m3tm/training/gemini_distillation.py`: Knowledge distillation trainer
- `examples/gemini_training.py`: Örnek kullanım script'i
- `docs/gemini_integration.md`: Bu dokümantasyon

## 🤝 Katkıda Bulunma

Gemini entegrasyonunu geliştirmek için:
1. Yeni yaklaşımlar öner
2. Performance optimizasyonları ekle
3. Hata düzeltmeleri yap
4. Dokümantasyonu güncelle
