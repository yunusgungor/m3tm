# M3TM iOS SDK Kullanım Örneği (Swift)

Bu belge, iOS uygulamalarınızda M3TM SDK'yı Swift ile nasıl kullanacağınızı göstermektedir.

## İçindekiler

1. [Kurulum](#kurulum)
2. [SDK'yı Başlatma](#sdkyı-başlatma)
3. [Model Yükleme](#model-yükleme)
4. [Çıkarım Yapma](#çıkarım-yapma)
5. [Arama Özelliğini Kullanma](#arama-özelliğini-kullanma)
6. [Eğitim İşlemleri](#eğitim-i̇şlemleri)
7. [Veri Dışa Aktarma](#veri-dışa-aktarma)
8. [Hata Yönetimi](#hata-yönetimi)
9. [En İyi Uygulamalar](#en-i̇yi-uygulamalar)

## Kurulum

### CocoaPods ile

`Podfile` dosyanıza ekleyin:

```ruby
pod 'M3TM', '~> 2.3.0'
```

Sonra terminal üzerinden şu komutu çalıştırın:

```bash
pod install
```

### Swift Package Manager ile

1. Xcode'da projenizi açın.
2. File > Swift Packages > Add Package Dependency seçeneğine tıklayın.
3. URL alanına `https://github.com/m3tm/ios-sdk` yazın ve "Next" düğmesine tıklayın.
4. Version seçimi için "Up to Next Major" seçeneğini seçin ve "2.3.0" girin.
5. "Next" düğmesine tıklayarak kurulumu tamamlayın.

## SDK'yı Başlatma

Swift dosyanızın en üstüne import ifadesini ekleyin:

```swift
import M3TM
```

SDK'yı başlatmak için:

```swift
// SDK yapılandırması oluştur
let config = M3TMConfig(modelPath: "m3tm_model.pt", configPath: "m3tm_config.json")
config.computeDevice = .auto
config.optimizationLevel = .basic
config.maxMemoryUsageMB = 256

// SDK'yı başlat
M3TM.shared.initialize(config: config) { error in
    if let error = error {
        print("SDK başlatma hatası: \(error.localizedDescription)")
        return
    }
    
    print("M3TM SDK başarıyla başlatıldı!")
    // SDK hazır, diğer işlemler burada yapılabilir
}
```

## Model Yükleme

Model otomatik olarak başlatma sırasında yüklenir (yapılandırmada `.onInitialization` seçeneği belirlenmişse). Manuel olarak yüklemek için:

```swift
do {
    let modelManager = try M3TM.shared.getModelManager()
    try modelManager.loadModel()
    print("Model başarıyla yüklendi!")
} catch {
    print("Model yükleme hatası: \(error.localizedDescription)")
}
```

## Çıkarım Yapma

Çıkarım yapmak için:

```swift
do {
    let modelManager = try M3TM.shared.getModelManager()
    
    // Giriş verisi oluştur
    let inputData = M3TMModelManager.InputData(text: "Bu bir örnek metin verisidir")
    
    // Çıkarım seçenekleri oluştur
    let options = M3TMModelManager.InferenceOptions()
    options.outputFormat = .dictionary
    
    // Çıkarımı gerçekleştir
    modelManager.inference(inputs: [inputData], options: options) { results, error in
        if let error = error {
            print("Çıkarım hatası: \(error.localizedDescription)")
            return
        }
        
        if let results = results as? [[String: Any]] {
            for (index, result) in results.enumerated() {
                print("Sonuç \(index): \(result)")
            }
        }
    }
} catch {
    print("Çıkarım hatası: \(error.localizedDescription)")
}
```

Görüntü ile çıkarım yapmak için:

```swift
do {
    let modelManager = try M3TM.shared.getModelManager()
    
    // Görüntü verisi ile giriş oluştur
    let image = UIImage(named: "example_image")!
    let inputData = M3TMModelManager.InputData(image: image)
    
    // Çıkarım seçenekleri oluştur
    let options = M3TMModelManager.InferenceOptions()
    options.outputFormat = .dictionary
    
    // Çıkarımı gerçekleştir
    modelManager.inference(inputs: [inputData], options: options) { results, error in
        if let error = error {
            print("Görüntü çıkarım hatası: \(error.localizedDescription)")
            return
        }
        
        if let results = results as? [[String: Any]] {
            for (index, result) in results.enumerated() {
                print("Görüntü sonucu \(index): \(result)")
            }
        }
    }
} catch {
    print("Çıkarım hatası: \(error.localizedDescription)")
}
```

## Arama Özelliğini Kullanma

Arama özelliğini kullanmak için önce arama servisini başlatın:

```swift
do {
    let searchService = try M3TM.shared.getSearchService()
    
    // Arama servisi başlatma
    searchService.initialize { error in
        if let error = error {
            print("Arama servisi başlatma hatası: \(error.localizedDescription)")
            return
        }
        
        print("Arama servisi başarıyla başlatıldı!")
    }
} catch {
    print("Arama servisi hatası: \(error.localizedDescription)")
}
```

Öğe indeksleme:

```swift
do {
    let searchService = try M3TM.shared.getSearchService()
    
    // Bir metin öğesi indeksle
    searchService.indexItem(
        itemId: "doc_123", 
        textContent: "M3TM, gizlilik odaklı bir mobil yapay zeka modelidir.",
        metadata: ["category": "AI", "date": Date().timeIntervalSince1970]
    ) { error in
        if let error = error {
            print("İndeksleme hatası: \(error.localizedDescription)")
            return
        }
        
        print("Öğe başarıyla indekslendi!")
    }
} catch {
    print("Arama servisi hatası: \(error.localizedDescription)")
}
```

Metin araması yapma:

```swift
do {
    let searchService = try M3TM.shared.getSearchService()
    
    // Metin araması yap
    let filters = M3TMSearchService.SearchFilters()
    filters.contentTypes = ["text"]
    
    searchService.search(textQuery: "gizlilik yapay zeka", filters: filters, topK: 5) { results, error in
        if let error = error {
            print("Arama hatası: \(error.localizedDescription)")
            return
        }
        
        if let results = results {
            for (index, result) in results.enumerated() {
                print("Arama sonucu \(index):")
                print("  ID: \(result.itemId)")
                print("  Skor: \(result.score)")
                print("  İçerik: \(result.textContent ?? "N/A")")
                if let metadata = result.metadata {
                    print("  Metadata: \(metadata)")
                }
            }
        }
    }
} catch {
    print("Arama servisi hatası: \(error.localizedDescription)")
}
```

## Eğitim İşlemleri

Bir adapter oluşturmak ve eğitmek:

```swift
class MyDataProvider: M3TMTrainingManager.DataProvider {
    let textData = ["Bu bir örnek eğitim metnidir", "Bu başka bir örnektir", "M3TM SDK'yı öğreniyorum"]
    let labels = [0, 1, 0]
    
    func getDataCount() -> Int {
        return textData.count
    }
    
    func getData(forIndices indices: [Int]) -> [M3TMModelManager.InputData] {
        return indices.map { M3TMModelManager.InputData(text: textData[$0]) }
    }
    
    func getLabels(forIndices indices: [Int]) -> [Any] {
        return indices.map { labels[$0] }
    }
}

class MyTrainingDelegate: M3TMTrainingDelegate {
    func trainingManager(_ manager: M3TMTrainingManager, didUpdateProgress progress: Float, metrics: [String : Any]?, forEpoch epoch: Int) {
        print("Eğitim ilerlemesi: \(progress * 100)%, Epoch: \(epoch)")
    }
    
    func trainingManager(_ manager: M3TMTrainingManager, didCompleteEpoch epoch: Int, withMetrics metrics: [String : Any]?) {
        print("Epoch \(epoch) tamamlandı")
        if let metrics = metrics {
            print("Metrikler: \(metrics)")
        }
    }
    
    func trainingManager(_ manager: M3TMTrainingManager, didFinishWithSuccess success: Bool, finalMetrics: [String : Any]?) {
        print("Eğitim tamamlandı. Başarılı: \(success)")
        if let metrics = finalMetrics {
            print("Final metrikler: \(metrics)")
        }
    }
    
    func trainingManager(_ manager: M3TMTrainingManager, didEncounterError error: Error) {
        print("Eğitim hatası: \(error.localizedDescription)")
    }
}

// Eğitim işlemi
do {
    let trainingManager = try M3TM.shared.getTrainingManager()
    let trainingDelegate = MyTrainingDelegate()
    trainingManager.delegate = trainingDelegate
    
    // Veri sağlayıcısı
    let dataProvider = MyDataProvider()
    
    // Eğitim yapılandırması
    let trainingConfig = M3TMTrainingManager.TrainingConfig()
    trainingConfig.learningRate = 0.001
    trainingConfig.epochs = 5
    trainingConfig.batchSize = 2
    
    // Adapter oluşturma ve eğitim
    trainingManager.createAdapter(forBlockIndex: 0, config: ["dims": 768]) { adapterId, error in
        if let error = error {
            print("Adapter oluşturma hatası: \(error.localizedDescription)")
            return
        }
        
        guard let adapterId = adapterId else {
            print("Geçersiz adapter ID")
            return
        }
        
        print("Adapter oluşturuldu: \(adapterId)")
        
        // Adapter'ı eğit
        trainingManager.trainAdapter(adapterId: adapterId, dataProvider: dataProvider, config: trainingConfig) { error in
            if let error = error {
                print("Adapter eğitim hatası: \(error.localizedDescription)")
                return
            }
            
            print("Adapter eğitimi tamamlandı!")
        }
    }
} catch {
    print("Eğitim yöneticisi hatası: \(error.localizedDescription)")
}
```

## Veri Dışa Aktarma

Kullanıcı verilerini dışa aktarmak:

```swift
do {
    let dataExporter = try M3TM.shared.getDataExporter()
    
    // Dışa aktarma servisini başlat
    dataExporter.initialize { error in
        if let error = error {
            print("Dışa aktarma servisi başlatma hatası: \(error.localizedDescription)")
            return
        }
        
        // Filtreler oluştur
        let filters = M3TMDataExporter.ExportFilters()
        filters.dateStart = Calendar.current.date(byAdding: .month, value: -1, to: Date())
        filters.contentTypes = ["text", "image"]
        
        // Seçenekler oluştur
        let options = M3TMDataExporter.ExportOptions()
        options.format = .json
        options.includeImages = true
        options.imageQuality = 0.9
        
        // Dışa aktarma işlemi
        dataExporter.exportData(filters: filters, options: options) { fileURL, itemCount, error in
            if let error = error {
                print("Dışa aktarma hatası: \(error.localizedDescription)")
                return
            }
            
            guard let fileURL = fileURL else {
                print("Dışa aktarma başarılı, ancak dosya URL'si bulunamadı")
                return
            }
            
            print("Dışa aktarma başarılı!")
            print("Dosya: \(fileURL)")
            print("Aktarılan öğe sayısı: \(itemCount)")
            
            // Dışa aktarma tamamlandıktan sonra
            // Paylaşım için ActivityViewController kullanılabilir, örneğin:
            DispatchQueue.main.async {
                let activityVC = UIActivityViewController(activityItems: [fileURL], applicationActivities: nil)
                // activityVC'yi göstermek için uygun bir view controller kullanın
                // viewController.present(activityVC, animated: true, completion: nil)
            }
        }
    }
} catch {
    print("Dışa aktarma servisi hatası: \(error.localizedDescription)")
}
```

## Hata Yönetimi

M3TM SDK, hataları `M3TMError` enum'u üzerinden yönetir:

```swift
do {
    // SDK işlemleri...
} catch let error as M3TMError {
    switch error {
    case .notInitialized:
        print("SDK başlatılmamış")
    case .modelNotLoaded:
        print("Model yüklenmemiş")
    case .trainingInProgress:
        print("Zaten bir eğitim devam ediyor")
    case .searchIndexNotInitialized:
        print("Arama indeksi başlatılmamış")
    case .outOfMemory:
        print("Bellek yetersiz")
    default:
        print("Beklenmeyen hata: \(error.localizedDescription)")
    }
} catch {
    print("Genel hata: \(error.localizedDescription)")
}
```

## En İyi Uygulamalar

### Bellek Yönetimi

- Yüksek çözünürlüklü görüntülerle çalışırken resim boyutlarını kullanım öncesinde küçültün
- `maxMemoryUsageMB` sınırını uygulama ihtiyaçlarına göre ayarlayın
- Arka plana geçiş olaylarını izleyin ve bellek optimizasyonlarını yapın

### Performans

- Eğer yüksek performans gerekliyse, `optimizationLevel` değerini `.speed` olarak ayarlayın
- Toplu (batch) çıkarım işlemleri yapın
- Arama indekslerini düzenli olarak kaydedin
- Mümkünse Metal desteği etkinleştirin

### Gizlilik

- Verileri yalnızca cihaz üzerinde işleyin
- Kullanıcılara kişisel verilerini indirme seçeneği sunun
- Hassas verileri cihazdan göndermeyin

### Yaşam Döngüsü

- SDK'nın initialize ile başlatıldığından emin olun
- Uygulamanın sonlandırılması öncesinde SDK'nın shutdown ile kapatıldığından emin olun
- Yüksek bellek kullanımı gerektiren işlemleri arka planda yapmayın
- Önemli verileri arka plana geçiş öncesinde kaydedin