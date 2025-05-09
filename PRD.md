Ürün Gereksinimleri Dokümanı (PRD): M³TM v2.3

# M³TM v2.3 - Mobil Multi-Modal Modüler Transformer
**Versiyon:** 1.0  
**Tarih:** 23 Mayıs 2024  
**Proje Sahibi:** [Proje Sahibi Adı/Ekibi]  
**Belgeyi Hazırlayan:** Yapay Zeka Asistanı (Codeflow Agent)

## 1. Giriş ve Amaç

Bu belge, M³TM v2.3 (Mobil Multi-Modal Modüler Transformer) modelinin geliştirilmesi için ürün gereksinimlerini tanımlar. M³TM v2.3, kullanıcının kişisel verileriyle (resimler, notlar, belgeler vb.) tamamen cihaz üzerinde eğitilebilen, bu veriler üzerinde semantik arama yapabilen, kullanıcıya verilerini indirme imkanı sunan ve geliştiricilerin özel mobil uygulamalar oluşturabilmesi için bir SDK (Yazılım Geliştirme Kiti) sağlayan, gizlilik odaklı, gelişmiş bir multi-modal yapay zeka modelidir. Model, PyTorch kullanılarak sıfırdan geliştirilecektir.

**Temel Hedef:** Kullanıcılara, verilerinin tam kontrolünü ve gizliliğini koruyarak, cihazlarında kişiselleştirilmiş ve akıllı deneyimler sunan, geliştiriciler için ise yenilikçi mobil uygulamalar yaratma platformu sağlayan bir yapay zeka çözümü oluşturmak.

## 2. Kapsam

### İçinde Olanlar:

- Ultra-hafif, multi-modal çekirdek modelin (M³TM v2.3 İskeleti) PyTorch ile tasarımı ve implementasyonu.
- Kullanıcı verisiyle büyüyebilen modüler yapı (Adapter'lar, görev başlıkları).
- Cihaz üzerinde eğitim mekanizmalarının geliştirilmesi.
- Cihaz üzerinde semantik arama yeteneği (metin, görüntü).
- Kullanıcının işlenmiş/ham verilerini indirme özelliği.
- Geliştiriciler için temel SDK fonksiyonları (veri girişi, eğitim tetikleme, çıkarım, arama, veri indirme).
- Desteklenecek ilk modaliteler: Metin ve Görüntü.
- Gizlilik ve veri güvenliği önlemleri.

### Dışında Olanlar (İlk Versiyon İçin):

- Sunucu tabanlı eğitim veya bulut entegrasyonu.
- Ses, video gibi ek karmaşık modaliteler (gelecek versiyonlarda eklenebilir).
- Çok gelişmiş, büyük ölçekli modellerle rekabet edecek düzeyde genel dünya bilgisi (model, kullanıcının verileriyle kişiselleşmeye odaklanacaktır).
- Tam teşekküllü bir kullanıcı arayüzü (SDK, geliştiricilerin kendi arayüzlerini oluşturmasını sağlar).

## 3. Hedef Kullanıcılar

### Son Kullanıcılar (Mobil Uygulama Kullanıcıları):

- Veri gizliliğine önem veren.
- Cihazlarındaki kişisel verilerle (fotoğraflar, notlar) daha akıllı etkileşimler kurmak isteyen.
- Verileri üzerinde hızlı ve anlamlı aramalar yapabilmek isteyen.
- Verilerine kolayca erişip dışa aktarabilmek isteyen.

### Geliştiriciler (SDK Kullanıcıları):

- Mobil uygulamalarına kolayca entegre edilebilecek, cihaz üzerinde çalışan, kişiselleştirilebilir yapay zeka yetenekleri arayan.
- Kullanıcı gizliliğini ihlal etmeden yenilikçi özellikler sunmak isteyen.
- Multi-modal verilerle çalışabilen esnek bir modele ihtiyaç duyan.

## 4. Kullanıcı Hikayeleri ve Senaryoları

### Son Kullanıcı Olarak:

- Telefonumdaki tüm "plaj" temalı fotoğrafları anlamsal olarak arayıp bulabilmek istiyorum.
- Geçen ay yazdığım ve "proje fikirleri" içeren tüm notlarımı bulabilmek istiyorum.
- Modelin, sık kullandığım konularla ilgili notlarımı daha iyi anlamasını ve ilgili bilgileri daha hızlı sunmasını istiyorum (kişiselleştirme).
- Tüm notlarımı veya belirli bir zaman aralığındaki fotoğraflarımı JSON formatında dışa aktarabilmek istiyorum.
- Verilerimin telefonumdan asla ayrılmayacağını bilerek güvende hissetmek istiyorum.

### Geliştirici Olarak:

- Kullanıcının galerisindeki fotoğrafları otomatik olarak etiketleyebilen bir uygulama geliştirmek için M³TM SDK'sını kullanmak istiyorum.
- Kullanıcının notları arasında akıllı arama yapabilen bir not alma uygulaması için M³TM SDK'sını entegre etmek istiyorum.
- M³TM SDK'sını kullanarak, kullanıcının girdiği metin sorgusuna uygun görselleri cihazdaki galerisinden öneren bir özellik eklemek istiyorum.
- Modelin eğitilebilir kısımlarını (adapter'lar, görev başlıkları) kolayca tanımlayıp, kullanıcı verileriyle cihaz üzerinde eğitebilmek istiyorum.

## 5. Teknik Mimari ve Bileşenler (M³TM v2.3 Detayları)

Aşağıda, her bir ana bileşenin daha detaylı teknik açıklaması ve potansiyel PyTorch yapıları sunulmaktadır.

### 5.1. Ultra-Hafif Çoklu-Modal Çekirdek (İskelet Model)

**Amacı:** Temel veri akışını, minimal modalite etkileşimini sağlamak ve "büyüme modüllerinin" entegrasyonuna zemin hazırlamak.

**Genel PyTorch Yapısı (Kavramsal):**

```python
import torch
import torch.nn as nn

class M3TM_Core(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 5.1.1 Text Processing
        self.text_embedding = TextEmbedding(config.text_config)
        # 5.1.2 Image Processing
        self.image_patch_embedding = ImagePatchEmbedding(config.image_config)
        # 5.1.3 Proto-Transformer Blocks
        self.proto_transformer_blocks = nn.ModuleList(
            [ProtoTransformerBlock(config.transformer_config) for _ in range(config.num_core_blocks)]
        )
        # 5.1.4 Basic Fusion
        self.fusion_layer = nn.Linear(config.text_config.embed_dim + config.image_config.embed_dim, config.fused_embed_dim) # Örnek basit fusion
        # 5.1.5 Search Embedding Projection
        self.search_projection = nn.Linear(config.fused_embed_dim, config.search_embed_dim)

    def forward(self, text_input=None, image_input=None):
        text_features = None
        image_features = None

        if text_input is not None:
            text_features = self.text_embedding(text_input) # (batch, seq_len, embed_dim)
        if image_input is not None:
            image_features = self.image_patch_embedding(image_input) # (batch, num_patches, embed_dim)

        # Proto-Transformer İşleme (modalitelere ayrı ayrı veya birleştirilmiş olarak uygulanabilir)
        # Bu kısım tasarım kararlarına göre detaylandırılacak. Örnek:
        if text_features is not None:
            for block in self.proto_transformer_blocks:
                text_features = block(text_features) # Adapter yuvaları blok içinde
        if image_features is not None:
            for block in self.proto_transformer_blocks:
                image_features = block(image_features) # Adapter yuvaları blok içinde

        # Basit Füzyon (Örnek: Ortalama havuzlama ve birleştirme)
        if text_features is not None and image_features is not None:
            # Her modalitenin temsillerini global bir vektöre indirge (örn: [CLS] token'ı veya ortalama)
            text_pooled = text_features.mean(dim=1) # (batch, embed_dim)
            image_pooled = image_features.mean(dim=1) # (batch, embed_dim)
            fused_representation = torch.cat((text_pooled, image_pooled), dim=-1)
            fused_representation = self.fusion_layer(fused_representation) # (batch, fused_embed_dim)
        elif text_features is not None:
            fused_representation = text_features.mean(dim=1) # Basitleştirilmiş: Sadece metin varsa
            # fused_representation = self.some_projection_for_text_only(fused_representation)
        elif image_features is not None:
            fused_representation = image_features.mean(dim=1) # Basitleştirilmiş: Sadece görüntü varsa
            # fused_representation = self.some_projection_for_image_only(fused_representation)
        else:
            return None, None # Ya da hata döndür

        # Arama Gömme Uzayı
        search_embedding = self.search_projection(fused_representation)
        search_embedding = nn.functional.normalize(search_embedding, p=2, dim=-1)

        return fused_representation, search_embedding # Görev başlıkları için ve arama için
```

#### 5.1.1. Metin İşleme ve Gömme
* **Tokenizasyon:** Karakter seviyesinde veya SentencePiece Unigram/BPE (çok küçük dağarcık, örn: ~4000 token).
* **Gömme:** nn.Embedding(vocab_size, embed_dim). embed_dim çok küçük (örn: 32, 64).
* **PyTorch Yapısı (Kavramsal):**

```python
class TextEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = # Karakter veya SentencePiece tokenizer yükle/oluştur
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
        # Opsiyonel: Çok basit konumsal gömme
        self.positional_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

    def forward(self, text_input_ids): # text_input_ids: (batch, seq_len)
        embedded_text = self.embedding(text_input_ids)
        if hasattr(self, 'positional_embedding'):
            positions = torch.arange(0, text_input_ids.size(1), device=text_input_ids.device).unsqueeze(0)
            embedded_text += self.positional_embedding(positions)
        return embedded_text
```

#### 5.1.2. Görüntü İşleme ve Yama Gömme
* **Yama Çıkarımı:** Görüntüyü nn.Unfold veya özel bir fonksiyonla 3x3 veya 4x4 yamalara bölme.
* **Yama Gömme:** Her yamaya nn.Conv2d (az filtre, küçük kernel) uygulanır, ardından nn.Flatten ve bir nn.Linear katmanı ile embed_dim boyutuna getirilir.
* **PyTorch Yapısı (Kavramsal):**

```python
class ImagePatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels # Genellikle 3 (RGB)
        self.embed_dim = config.embed_dim
        # Basit Conv2d ile yama projeksiyonu
        self.projection = nn.Conv2d(self.in_channels, self.embed_dim,
                                   kernel_size=self.patch_size, stride=self.patch_size)
        # Opsiyonel: Yamalar için konumsal gömme
        # self.positional_embedding = nn.Parameter(torch.randn(1, config.num_patches + 1, self.embed_dim)) # ViT stili

    def forward(self, image_input): # image_input: (batch, channels, height, width)
        patches = self.projection(image_input) # (batch, embed_dim, num_patches_h, num_patches_w)
        patches = patches.flatten(2).transpose(1, 2) # (batch, num_patches_h * num_patches_w, embed_dim)
        # if hasattr(self, 'positional_embedding'):
        #     # [CLS] token'ı varsa ona göre ayarla
        #     patches += self.positional_embedding[:, 1:] # [CLS] token'ı yoksa
        return patches
```

#### 5.1.3. "Proto-Transformer" Blokları
* **Yapı:** LayerNorm -> Dikkat Alternatifi -> Artık Bağlantı -> LayerNorm -> FFN Alternatifi -> Artık Bağlantı.
* **Dikkat Alternatifi:** Pencereli/Seyreltilmiş Self-Attention (1-2 başlık, küçük başlık boyutu) veya Hafif Konvolüsyonel Blok.
* **FFN Alternatifi:** GLU veya 1 gizli katmanlı küçük MLP.
* **Adapter Yuvası:** FFN'den sonra, Adapter modülünün ekleneceği yer. Başlangıçta kimlik (identity) map'lemesi yapar.
* **PyTorch Yapısı (Kavramsal):**

```python
class ProtoTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        # self.attention = SimplifiedSelfAttention(config.embed_dim, config.num_heads)
        self.attention = LightweightConvModule(config.embed_dim, kernel_size=config.conv_kernel_size) # Alternatif
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = GatedLinearUnit(config.embed_dim, config.ffn_hidden_dim) # veya SmallMLP
        self.adapter_slot = AdapterSlot(config.embed_dim) # Başlangıçta identity

    def forward(self, x): # x: (batch, seq_len, embed_dim)
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        ffn_output = self.adapter_slot(ffn_output) # Adapter burada devreye girer
        x = x + ffn_output
        return x

class AdapterSlot(nn.Module): # Basit bir yuva
    def __init__(self, embed_dim):
        super().__init__()
        self.adapter_module = None # SDK veya eğitim yöneticisi tarafından atanacak

    def add_adapter(self, adapter_instance):
        self.adapter_module = adapter_instance

    def forward(self, x):
        if self.adapter_module:
            return x + self.adapter_module(x) # Adapter artık bağlantı ile
        return x
```

### 5.2. Büyüme Modülleri

#### 5.2.1. Adapter'lar
* **Yapı:** Genellikle daraltıcı bir nn.Linear, bir aktivasyon (örn: h-swish), ve genişletici bir nn.Linear. Çıktısı, ana akışa artık bağlantı ile eklenir.
* **PyTorch Yapısı (Kavramsal):**

```python
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, activation=nn.Hardswish()):
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.activation = activation
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        # Başlangıçta sıfıra yakın başlatma, böylece ilk başta kimlik gibi davranır
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return self.up_project(self.activation(self.down_project(x)))
```

#### 5.2.2. Arama İndeksleme Modülü
* **Gömme Alımı:** Çekirdek modelden gelen search_embedding vektörlerini alır.
* **İndeksleme:**
  * Mobil için çok hafif bir ANN (Approximate Nearest Neighbor) kütüphanesi veya basit bir brute-force k-NN (küçük veri setleri için).
  * LSH (Locality Sensitive Hashing) gibi tekniklerin basit bir implementasyonu.
  * İndeks, cihazda kalıcı bir dosyada saklanabilir (SQLite veya custom binary format).
* **PyTorch Entegrasyonu:** Bu modül, PyTorch'un kendisinden ziyade, PyTorch tensörlerini alıp harici bir C++/Swift kütüphanesiyle etkileşen bir yapı olabilir. SDK, bu indeksi yönetir.
* **Kavramsal İşlev:** `add_to_index(item_id, embedding_vector)`, `search_index(query_vector, top_k)`

#### 5.2.3. Veri Formatlama/İndirme Modülü
* Bu, daha çok bir yardımcı program (utility) olacaktır.
* Modelin işlediği (veya orijinal) verileri (metin, görüntü meta verileri, vb.) alır.
* İstenen filtrelemeyi (zaman aralığı, tip) uygular.
* JSON, CSV gibi formatlara dönüştürür.
* Cihazın dosya sistemine güvenli bir şekilde yazar.

### 5.3. Dinamik Görev Başlıkları
* Genellikle nn.Linear(input_dim, num_classes) veya birkaç katmanlı küçük bir MLP.
* input_dim, çekirdeğin fused_representation veya bir adapter'ın çıktısının boyutu olacaktır.
* **PyTorch Yapısı (Örnek - Sınıflandırma Başlığı):**

```python
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x): # x: (batch, input_dim)
        return self.linear(x)
```

### 5.4. Cihaz Üzerinde Eğitim ve Optimizasyon
* **Eğitim Döngüsü:** SDK üzerinden yönetilir. Kullanıcı verisi batch'ler halinde alınır, model ileri ve geri yayılım yapar.
* **Optimizer:** `torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay, eps=1e-8)`. Parametreler, sadece eğitilecek modüllere (adapter'lar, görev başlıkları) ait olanlar seçilerek verilir.
* **Öğrenme Oranı Zamanlayıcı:** `torch.optim.lr_scheduler.ReduceLROnPlateau` veya basit bir `CosineAnnealingLR`.
* **Kayıp Fonksiyonları:**
  * Sınıflandırma: `nn.CrossEntropyLoss()`
  * Arama (Contrastive Learning): Çekirdek modelin search_projection katmanını eğitmek için. Veri noktaları (anchor, positive, negative) gerektirir.
* **Gradyan Kırpma (Gradient Clipping):** `torch.nn.utils.clip_grad_norm_` kararlılık için.
* **Dondurulmuş Katmanlar:** Çekirdek modelin çoğu parametresi `.requires_grad = False` olarak ayarlanır.

## 6. SDK (Yazılım Geliştirme Kiti) Tasarımı

**Diller:** Android için Kotlin/Java, iOS için Swift/Objective-C. PyTorch Mobile C++ API'lerini sarmalayacak.

### Ana Sınıflar/Arayüzler (Kavramsal):

- **M3TMManager:** Modelin yüklenmesi, başlatılması, genel yapılandırma.

- **DataProcessor:** Farklı modalitelerden gelen verilerin modele uygun formata dönüştürülmesi.

- **TrainingController:** Eğitim döngüsünün başlatılması, durdurulması, ilerlemenin izlenmesi, hangi modüllerin eğitileceğinin belirlenmesi.
  - `add_adapter_to_core(block_index, adapter_config)`
  - `add_task_head(head_name, head_config, input_source_module_name)`
  - `train_module(module_name, data_provider, epochs, loss_function)`

- **InferenceEngine:** Modelden çıkarım yapma.
  - `predict(input_data, task_head_name)`

- **SearchService:** Semantik arama yapma.
  - `index_item(item_id, text_content, image_path)`
  - `search(query_text, query_image_path, top_k, filters) -> List<SearchResultItem>`

- **DataExporter:** Veri indirme.
  - `export(filter_criteria, format) -> FilePath`

### Veri Akışı (SDK ile):

1. Geliştirici uygulamada M3TMManager'ı başlatır.
2. Kullanıcı verisi (örn: yeni bir fotoğraf) geldiğinde, geliştirici `SearchService.index_item()` çağırır.
3. SDK, DataProcessor ile veriyi işler, M3TM_Core'dan search_embedding alır ve bunu arama indeksine ekler.
4. Bir görev için (örn: fotoğrafları etiketleme), geliştirici uygun Adapter'ı ve ClassificationHead'i tanımlar/ekler, ardından `TrainingController.train_module()` ile eğitimi başlatır.

## 7. Geliştirme Aşamaları ve Yol Haritası

### Aşama 0: Hazırlık ve Temel Araştırma (1-2 Hafta)

- Geliştirme ortamının kurulması (PyTorch, mobil geliştirme araçları).
- PyTorch Mobile ile temel deneyler.
- Mobil için optimize edilmiş dikkat ve konvolüsyon mekanizmaları üzerine derinlemesine literatür taraması ve küçük prototipler.
- Detaylı yapılandırma (config) dosyalarının formatının belirlenmesi.

### Aşama 1: Çekirdek Model MVP (Minimum Uygulanabilir Ürün) (6-8 Hafta)

- **1a:** Tek Modalite Çekirdek (Metin): TextEmbedding, ProtoTransformerBlock (1 blok), basit ClassificationHead. Cihaz üzerinde eğitim ve çıkarım.
- **1b:** İkinci Modalite Ekleme (Görüntü): ImagePatchEmbedding.
- **1c:** Temel Füzyon ve Arama Gömme Projeksiyonu: İki modaliteyi birleştiren ve arama gömmesi üreten M3TM_Core'un ilk versiyonu.
- **Kod Yapısı:** Bu aşamada 5.1'deki PyTorch modülleri oluşturulur.

### Aşama 2: Büyüme Modülleri (Adapter'lar) ve Görev Başlıkları (4-6 Hafta)

- Adapter sınıfının implementasyonu.
- AdapterSlot mekanizmasının ProtoTransformerBlock'a entegrasyonu.
- Farklı görevler için dinamik olarak eklenebilen TaskHead'ler.
- Çekirdek dondurularak sadece adapter ve görev başlıklarının eğitimi.
- **Kod Yapısı:** 5.2.1 ve 5.3'teki yapılar.

### Aşama 3: Semantik Arama ve Veri İndirme Temelleri (6-8 Hafta)

- **3a:** Arama İndeksleme Modülü: Basit bir mobil uyumlu ANN veya k-NN implementasyonu. Çekirdekten gelen search_embedding'lerin indekslenmesi.
- **3b:** Arama Servisi API'si: Temel arama sorgularını işleme.
- **3c:** Veri İndirme Modülü ve API'si: Basit filtreleme ve JSON export.
- **Kod Yapısı:** 5.2.2, 5.2.3'teki kavramsal yapılar ve SDK'daki ilgili servisler.

### Aşama 4: SDK Geliştirme ve Dokümantasyon (Sürekli, ~Aşama 2'den itibaren)

- Android ve iOS için temel SDK sarmalayıcılarının (wrapper) geliştirilmesi.
- API'lerin detaylı dokümantasyonu ve örnek kullanımlar.
- **Kod Yapısı:** 6. bölümdeki SDK sınıfları.

### Aşama 5: Test, Optimizasyon ve Entegrasyon (Sürekli)

- Birim testleri, entegrasyon testleri.
- Performans profillemesi (çıkarım hızı, eğitim hızı, bellek kullanımı).
- Model küçültme teknikleri (nicemleme - quantization, budama - pruning) araştırılması ve uygulanması.
- Örnek bir demo uygulama ile baştan sona test.

## 8. Veri Yönetimi ve Gizlilik

- Tüm kullanıcı verileri (ham ve işlenmiş) ve model ağırlıkları cihazda saklanır.
- SDK, veri erişimi için işletim sistemi izinlerini kullanır.
- Veri indirme, kullanıcının açık onayı ile yapılır.
- Arama indeksleri de cihazda şifrelenmiş olarak saklanabilir (platform yeteneklerine bağlı).

## 9. Başarı Metrikleri

### Çekirdek Model:
- Parametre sayısı (hedef: <5-10 Milyon).
- Çıkarım hızı (hedef: mobil CPU/GPU'da <50-100ms/örnek).
- Cihaz üzerinde eğitim hızı (kabul edilebilir bir sürede konverjans).

### Büyüme Modülleri:
- Adapter'ların öğrenme kapasitesi (az veriyle ne kadar iyileşme sağlıyor).

### Arama:
- Basit benchmark veri setlerinde arama doğruluğu (örn: Recall@K).
- Arama hızı.

### SDK:
- Geliştirici memnuniyeti (anketler, geri bildirimler).
- SDK entegrasyon kolaylığı.
- SDK ile geliştirilen örnek uygulama sayısı (uzun vadeli).

### Genel:
- Bellek kullanımı (RAM).
- Batarya tüketimi üzerindeki etki.

## 10. Riskler ve Zorluklar

- **Performans Kısıtlamaları:** Cihaz üzerinde hem eğitim hem de karmaşık çıkarım yapmak son derece zorlu.
- **Sıfırdan Eğitim:** Büyük, önceden eğitilmiş modellerin gücünden yoksun olmak, tatmin edici performans elde etmeyi zorlaştırabilir. Özellikle çekirdek modelin "eyleme geçebilmesi".
- **Veri Verimliliği:** Modelin az miktarda kullanıcı verisinden etkili bir şekilde öğrenebilmesi gerekiyor.
- **Multi-Modal Füzyon Karmaşıklığı:** Farklı modaliteleri anlamlı bir şekilde birleştirmek ve aralarında ilişki kurmak zor.
- **SDK Tasarımı ve Platformlar Arası Tutarlılık:** Android ve iOS için stabil ve kullanımı kolay bir SDK geliştirmek zaman alıcı.
- **Test Edilebilirlik:** Cihaz üzerinde eğitilen ve kişiselleşen bir modeli test etmek karmaşık.

## 11. Açık Sorular ve Gelecek Çalışmalar

- Çekirdek model için en uygun dikkat/konvolüsyon alternatifi hangisi olacak? (Prototipleme ile belirlenecek).
- Hangi mobil ANN çözümü en iyi performansı ve hafifliği sunar?
- Gelişmiş kendi kendine denetimli öğrenme teknikleri, çekirdeğin ilk başlatılmasında kullanılabilir mi?
- Federated Learning ile kullanıcılar arası (gizliliği koruyarak) model iyileştirmesi mümkün mü? (Gelecek versiyonlar için).
- Daha fazla modalite (ses, video) nasıl entegre edilebilir?

## 12. Codeflow Entegrasyonu

Bu PRD, Codeflow sisteminin parçası olarak aşağıdaki süreçleri destekleyecektir:

### 12.1. Mimari Prensipleri
- **Modülerlik:** Tüm bileşenler SRP (Single Responsibility Principle) prensibiyle tasarlanacak
- **Küçük Kod Birimleri:** Fonksiyonlar 5-10 satır (maksimum 20), sınıflar 100 satır limit hedefiyle
- **Örüntü Tanıma ve Kataloglama:** Kod içerisinde uygulanan tüm desenler belgelenecek ve .project_meta/.patterns altında kataloglanacak

### 12.2. Bağımlılık Yönetimi
- **Çevrimsel Bağımlılık Engelleme:** Modül ve bileşenler arasında çevrimsel bağımlılık olmayacak
- **Bağımlılık Grafiği:** Tüm modüller arası bağımlılıklar .project_meta/.dependencies içinde belgelenecek

### 12.3. Süreç Entegrasyonu
- **Yol Haritası-Modül Eşleştirmesi:** Her gelişim aşaması .project_meta/.stories altında izlenecek
- **Değişiklik Etki Analizi:** Yapılan değişikliklerin sistem geneline etkileri analiz edilecek
- **Örüntü Öğrenme:** Kod içinde tespit edilen desenler ve anti-desenler kataloglanacak

### 12.4. Dokümantasyon
- **Modül Dokümantasyonu:** Tüm modüller ve API'ler kapsamlı şekilde belgelenecek
- **Mimari Karar Kayıtları (ADR):** Önemli tasarım kararları .project_meta/.architecture/adr_log.json'da kayıt altına alınacak
- **Örüntü Dokümantasyonu:** Tespit edilen kod örüntüleri ve kullanım best practice'leri belgelenecek