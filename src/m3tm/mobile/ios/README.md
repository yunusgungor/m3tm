# M³TM iOS SDK

M³TM iOS SDK, Mobil Multi-Modal Modüler Transformer modelini iOS uygulamalarında kullanmak için gerekli araçları sağlar. Bu SDK, metin ve görüntü işleme, çoklu modlu (multimodal) işlemler, cihaz üzerinde eğitim ve semantik arama gibi yetenekleri iOS uygulamalarınıza entegre etmenizi sağlar.

## Özellikler

- **Metin İşleme**: Metin sınıflandırma, duygu analizi, özetleme, yanıt üretme
- **Görüntü İşleme**: Nesne tanıma, görüntü sınıflandırma, görüntü açıklama
- **Multimodal İşleme**: Görsel soru-cevap, resim açıklama, resimli metin anlama
- **Lokal Eğitim**: Cihaz üzerinde model ince ayarı (fine-tuning)
- **Semantik Arama**: Metin ve görüntü tabanlı içerik arama

## Sistem Gereksinimleri

- iOS 14.0+
- Xcode 13.0+
- Swift 5.5+ veya Objective-C
- Minimum 2GB RAM önerilen
- Metal destekli cihazlar için optimize edilmiş

## Kurulum

### CocoaPods ile

```ruby
pod 'M3TMSDK', '~> 0.1.0'
```

### Swift Package Manager ile

```swift
dependencies: [
    .package(url: "https://github.com/m3tm/ios-sdk.git", .upToNextMajor(from: "0.1.0"))
]
```

### Manuel Kurulum

1. `M3TMSDK.xcframework` dosyasını indirin
2. Projenize sürükleyin ve "Embed & Sign" seçeneğini işaretleyin
3. `import M3TMSDK` ile SDK'yı projenizde kullanmaya başlayın

## Hızlı Başlangıç

### Swift ile Kullanım

```swift
import M3TMSDK

// SDK'yı yapılandırın
M3TM.configure(["log_level": "debug", "use_gpu": true])

// Model yöneticisini alın
let modelManager = M3TM.modelManager()

// Modeli yükleyin
do {
    guard let model = try modelManager.loadModel(
        modelId: "m3tm_base",
        version: "1.0.0",
        error: nil
    ) else {
        print("Model yüklenemedi")
        return
    }
    
    // Metin işleme
    let text = "Bu bir örnek metindir."
    let result = try model.processText(
        text: text,
        task: "sentiment_analysis",
        options: ["language": "tr"],
        error: nil
    )
    
    print("Sonuç: \(result ?? [:])")
    
    // Modeli kapatın
    model.close(nil)
} catch {
    print("Hata: \(error)")
}
```

### Objective-C ile Kullanım

```objective-c
@import M3TMSDK;

// SDK'yı yapılandırın
[M3TM configure:@{@"log_level": @"debug", @"use_gpu": @YES}];

// Model yöneticisini alın
M3TMModelManager *modelManager = [M3TM modelManager];

// Modeli yükleyin
NSError *error = nil;
M3TMModel *model = [modelManager loadModel:@"m3tm_base" version:@"1.0.0" error:&error];

if (model) {
    // Metin işleme
    NSDictionary *result = [model processText:@"Bu bir örnek metindir."
                                         task:@"sentiment_analysis"
                                      options:@{@"language": @"tr"}
                                        error:&error];
    
    if (result) {
        NSLog(@"Sonuç: %@", result);
    } else {
        NSLog(@"İşleme hatası: %@", error.localizedDescription);
    }
    
    // Modeli kapatın
    [model close:&error];
} else {
    NSLog(@"Model yükleme hatası: %@", error.localizedDescription);
}
```

## Ana Bileşenler

### M3TM

SDK'nın ana giriş noktası. Yapılandırma, durum kontrolü ve yönetici sınıflarına erişim sağlar.

```swift
// SDK versiyonunu alın
let version = M3TM.version()

// SDK kullanılabilirliğini kontrol edin
let isAvailable = M3TM.isAvailable()

// Yöneticilere erişin
let modelManager = M3TM.modelManager()
let trainingManager = M3TM.trainingManager()
let searchManager = M3TM.searchManager()
```

### M3TMModelManager

Model yükleme, listeleme ve durum kontrolü için kullanılır.

```swift
// Mevcut modelleri listeleyin
let models = try modelManager.listAvailableModels(error: nil)

// Model mevcudiyetini kontrol edin
let isAvailable = modelManager.isModelAvailable("m3tm_base")

// Model bilgisini alın
let modelInfo = try modelManager.getModelInfo("m3tm_base", error: nil)
```

### M3TMModel

Model işlemleri için kullanılır. Metin, görüntü ve çoklu modlu işlemleri destekler.

```swift
// Metin işleme
let textResult = try model.processText(
    text: "Bu bir örnek metindir.",
    task: "classification",
    options: ["language": "tr"],
    error: nil
)

// Görüntü işleme
let imageResult = try model.processImage(
    image: UIImage(named: "sample")!,
    task: "object_detection",
    options: ["confidence": "0.7"],
    error: nil
)

// Multimodal işleme
let multimodalResult = try model.processMultimodal(
    text: "Bu resimde ne var?",
    image: UIImage(named: "sample")!,
    task: "visual_question_answering",
    options: nil,
    error: nil
)
```

### M3TMTrainingManager

Cihaz üzerinde eğitim için kullanılır.

```swift
// Eğitim oturumu oluşturma
let sessionId = try trainingManager.createTrainingSession(
    modelId: "m3tm_base",
    trainingConfig: ["learning_rate": "0.001"],
    error: nil
)

// Eğitim örneği ekleme
try trainingManager.addTextSample(
    sessionId: sessionId!,
    text: "Harika bir gün!",
    label: "positive",
    error: nil
)

// Eğitimi başlatma
try trainingManager.startTraining(
    sessionId: sessionId!,
    epochs: 5,
    callback: myCallback,
    error: nil
)
```

### M3TMSearchManager

Semantik arama işlemleri için kullanılır.

```swift
// Metin ile arama
let textResults = try searchManager.searchWithText(
    modelId: "m3tm_base",
    query: "yapay zeka",
    options: ["top_k": "5"],
    error: nil
)

// Görüntü ile arama
let imageResults = try searchManager.searchWithImage(
    modelId: "m3tm_base",
    image: UIImage(named: "sample")!,
    options: nil,
    error: nil
)

// İçerik indeksleme
try searchManager.indexText(
    modelId: "m3tm_base",
    text: "İndekslenecek örnek metin",
    metadata: ["source": "article"],
    error: nil
)
```

## Gelişmiş Kullanım

Daha detaylı örnekler için `Examples/` dizinine bakın. SwiftUI entegrasyonu, CoreML optimizasyonları ve özel model yapılandırmaları hakkında daha fazla bilgi edinebilirsiniz.

## Mimariye Genel Bakış

M³TM iOS SDK, çok katmanlı bir mimariye sahiptir:

1. **Swift/Objective-C API Katmanı**: Geliştiricinlerin ulaştığı yüksek seviyeli API
2. **C++ Köprüsü (Bridge)**: Native API ile PyTorch arasında iletişim
3. **PyTorch Mobile Çekirdeği**: Model çalıştırma motoru
4. **Cihaz Optimizasyon Katmanı**: Metal, CoreML ve diğer iOS yetenekleri ile performans artırma

## Performans İpuçları

- GPU hesaplama için `use_gpu: true` parametresini kullanın
- Düşük belleğe sahip cihazlarda `low_memory_mode: true` kullanmayı düşünün
- Büyük modeller için `streaming_mode: true` ayarını etkinleştirin
- Eğitim için bellek sınırlamalarına dikkat edin, `batch_size` parametresini düşük tutun

## Lisans

M³TM iOS SDK, [MIT Lisansı](LICENSE) altında dağıtılmaktadır.

## İletişim ve Destek

Sorunlar, öneriler ve katkılarınız için:
- GitHub: [github.com/m3tm/ios-sdk](https://github.com/m3tm/ios-sdk)
- E-posta: support@m3tm.com

---

© 2023 M³TM Team. Tüm hakları saklıdır. 