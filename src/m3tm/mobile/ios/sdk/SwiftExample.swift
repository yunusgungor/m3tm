import UIKit

/**
 * M3TM iOS SDK için Swift kullanım örneği.
 * Bu örnek kod, M3TM modelini Swift uygulamasında kullanma yöntemlerini gösterir.
 */
class M3TMSwiftExample {
    
    // MARK: - Model İşlemleri Örnekleri
    
    func modelExample() {
        // SDK'yı yapılandır
        M3TM.configure(["log_level": "debug", "use_gpu": true])
        
        // Modeli yükle
        do {
            guard let model = try M3TM.modelManager().loadModel(
                modelId: "m3tm_base",
                version: "1.0.0",
                error: nil
            ) else {
                print("Model yüklenemedi")
                return
            }
            
            print("Model yüklendi: \(model.modelId)")
            
            // Metin işleme
            textProcessingExample(with: model)
            
            // Görüntü işleme
            imageProcessingExample(with: model)
            
            // Multimodal işleme
            multimodalProcessingExample(with: model)
            
            // Modeli kapat
            var error: NSError?
            if !model.close(&error) {
                print("Model kapatılamadı: \(error?.localizedDescription ?? "bilinmeyen hata")")
            }
        } catch {
            print("Hata: \(error)")
        }
    }
    
    func textProcessingExample(with model: M3TMModel) {
        let text = "Bu bir örnek metin."
        let task = "sentiment_analysis"
        let options = ["language": "tr", "detailed": "true"]
        
        do {
            let result = try model.processText(
                text: text,
                task: task,
                options: options,
                error: nil
            )
            
            print("Metin işleme sonucu:")
            print(result ?? [:])
            
        } catch {
            print("Metin işleme hatası: \(error)")
        }
    }
    
    func imageProcessingExample(with model: M3TMModel) {
        guard let image = UIImage(named: "sample_image") else {
            print("Örnek görüntü bulunamadı")
            return
        }
        
        let task = "object_detection"
        let options = ["confidence_threshold": "0.5", "max_detections": "10"]
        
        do {
            let result = try model.processImage(
                image: image,
                task: task,
                options: options,
                error: nil
            )
            
            print("Görüntü işleme sonucu:")
            print(result ?? [:])
            
        } catch {
            print("Görüntü işleme hatası: \(error)")
        }
    }
    
    func multimodalProcessingExample(with model: M3TMModel) {
        let text = "Bu resimdeki nesneleri tanımla."
        guard let image = UIImage(named: "sample_image") else {
            print("Örnek görüntü bulunamadı")
            return
        }
        
        let task = "visual_question_answering"
        let options = ["max_length": "100", "language": "tr"]
        
        do {
            let result = try model.processMultimodal(
                text: text,
                image: image,
                task: task,
                options: options,
                error: nil
            )
            
            print("Multimodal işleme sonucu:")
            print(result ?? [:])
            
        } catch {
            print("Multimodal işleme hatası: \(error)")
        }
    }
    
    // MARK: - Eğitim İşlemleri Örnekleri
    
    func trainingExample() {
        class MyTrainingCallback: NSObject, M3TMTrainingCallback {
            func onBatchCompleted(_ sessionId: String, batch: Int, metrics: [AnyHashable : Any]) {
                print("Batch \(batch) tamamlandı: \(metrics)")
            }
            
            func onEpochCompleted(_ sessionId: String, epoch: Int, metrics: [AnyHashable : Any]) {
                print("Epoch \(epoch) tamamlandı: \(metrics)")
            }
            
            func onTrainingCompleted(_ sessionId: String, metrics: [AnyHashable : Any]) {
                print("Eğitim tamamlandı: \(metrics)")
            }
            
            func onTrainingError(_ sessionId: String, error: M3TMError) {
                print("Eğitim hatası: \(error.localizedDescription)")
            }
        }
        
        let trainingManager = M3TM.trainingManager()
        let modelId = "m3tm_base"
        let trainingConfig = [
            "learning_rate": "0.001",
            "batch_size": "16",
            "optimizer": "adam"
        ]
        
        do {
            // Eğitim oturumu oluştur
            guard let sessionId = try trainingManager.createTrainingSession(
                modelId: modelId,
                trainingConfig: trainingConfig,
                error: nil
            ) else {
                print("Eğitim oturumu oluşturulamadı")
                return
            }
            
            print("Eğitim oturumu oluşturuldu: \(sessionId)")
            
            // Metin örneği ekle
            let _ = try trainingManager.addTextSample(
                sessionId: sessionId,
                text: "Bu bir pozitif cümle.",
                label: "positive",
                error: nil
            )
            
            // Resim örneği ekle
            guard let image = UIImage(named: "sample_image") else {
                print("Örnek görüntü bulunamadı")
                return
            }
            
            let _ = try trainingManager.addImageSample(
                sessionId: sessionId,
                image: image,
                label: "cat",
                error: nil
            )
            
            // Eğitimi başlat
            let callback = MyTrainingCallback()
            let initialStatus = try trainingManager.startTraining(
                sessionId: sessionId,
                epochs: 10,
                callback: callback,
                error: nil
            )
            
            print("Eğitim başlatıldı: \(initialStatus ?? [:])")
            
            // Eğitim durumunu kontrol et
            DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                do {
                    let status = try trainingManager.getTrainingStatus(
                        sessionId: sessionId,
                        error: nil
                    )
                    print("Eğitim durumu: \(status ?? [:])")
                    
                    // Eğitimi iptal et
                    let _ = try trainingManager.cancelTraining(
                        sessionId: sessionId,
                        error: nil
                    )
                    
                    // Eğitim oturumunu sil
                    let _ = try trainingManager.deleteTrainingSession(
                        sessionId: sessionId,
                        error: nil
                    )
                } catch {
                    print("Hata: \(error)")
                }
            }
        } catch {
            print("Hata: \(error)")
        }
    }
    
    // MARK: - Arama İşlemleri Örnekleri
    
    func searchExample() {
        let searchManager = M3TM.searchManager()
        let modelId = "m3tm_base"
        
        do {
            // Metin ile arama
            let textResults = try searchManager.searchWithText(
                modelId: modelId,
                query: "örnek arama sorgusu",
                options: ["top_k": "5", "min_score": "0.7"],
                error: nil
            )
            
            print("Metin arama sonuçları:")
            for (index, result) in (textResults ?? []).enumerated() {
                print("Sonuç \(index + 1): \(result)")
            }
            
            // Görüntü ile arama
            guard let image = UIImage(named: "sample_image") else {
                print("Örnek görüntü bulunamadı")
                return
            }
            
            let imageResults = try searchManager.searchWithImage(
                modelId: modelId,
                image: image,
                options: ["top_k": "5", "min_score": "0.7"],
                error: nil
            )
            
            print("Görüntü arama sonuçları:")
            for (index, result) in (imageResults ?? []).enumerated() {
                print("Sonuç \(index + 1): \(result)")
            }
            
            // Metin indeksleme
            let textMetadata = ["source": "web", "category": "news", "date": "2023-05-01"]
            let _ = try searchManager.indexText(
                modelId: modelId,
                text: "Bu bir örnek metin. İndekse eklenecek.",
                metadata: textMetadata,
                error: nil
            )
            
            // Görüntü indeksleme
            let imageMetadata = ["source": "camera", "category": "person", "date": "2023-05-01"]
            let _ = try searchManager.indexImage(
                modelId: modelId,
                image: image,
                metadata: imageMetadata,
                error: nil
            )
            
            // İndeksi kaydet
            let _ = try searchManager.saveIndex(error: nil)
            
        } catch {
            print("Hata: \(error)")
        }
    }
    
    // MARK: - Genel SDK Kullanımı
    
    func generalSDKExample() {
        // SDK versiyonu
        let version = M3TM.version()
        print("M³TM SDK Versiyonu: \(version)")
        
        // SDK kullanılabilirliği
        let isAvailable = M3TM.isAvailable()
        print("M³TM SDK Kullanılabilir: \(isAvailable)")
        
        // Mevcut modelleri listele
        do {
            let models = try M3TM.modelManager().listAvailableModels(error: nil)
            
            print("Mevcut modeller:")
            for model in models ?? [] {
                print("- \(model.modelId) (v\(model.version)): \(model.name)")
                print("  \(model.description)")
                print("  Özellikler: \(model.properties)")
            }
        } catch {
            print("Modelleri listeleme hatası: \(error)")
        }
    }
    
    // MARK: - Ana Örnek
    
    func runAllExamples() {
        print("=== M³TM Swift API Örnekleri ===")
        
        // Genel SDK kullanımı
        generalSDKExample()
        
        // Model işlemleri
        modelExample()
        
        // Eğitim işlemleri
        trainingExample()
        
        // Arama işlemleri
        searchExample()
    }
}

// Örnek çalıştırma
let example = M3TMSwiftExample()
example.runAllExamples()

// ViewController içinde kullanım örneği
class ExampleViewController: UIViewController {
    
    private let m3tmExample = M3TMSwiftExample()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Örneği çalıştır
        DispatchQueue.global(qos: .userInitiated).async {
            self.m3tmExample.runAllExamples()
        }
    }
} 