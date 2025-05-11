import Foundation
import UIKit

/**
 * M3TM model yöneticisi sınıfı. Model yükleme, çıkarım yapmak ve görev başlıklarıyla çalışmak için gereken işlevleri sağlar.
 */
@objcMembers public class M3TMModelManager: NSObject {
    
    // MARK: - Enums & Types
    
    /**
     * Çıkarım sonucu formatı
     */
    @objc public enum OutputFormat: Int {
        /// JSON olarak çıktı
        case json = 0
        /// Tensor (NSData) olarak çıktı
        case tensor = 1
        /// Dictionary olarak çıktı
        case dictionary = 2
        /// Liste olarak çıktı
        case array = 3
    }
    
    /**
     * Bir tensör yığını (batch) için bir veri noktasının bilgilerini içerir
     */
    @objc public class InputData: NSObject {
        public var textData: String?
        public var imageData: UIImage?
        public var additionalFeatures: [String: Any]?
        
        public init(text: String? = nil, image: UIImage? = nil, features: [String: Any]? = nil) {
            self.textData = text
            self.imageData = image
            self.additionalFeatures = features
            super.init()
        }
    }
    
    /**
     * Çıkarım seçenekleri
     */
    @objc public class InferenceOptions: NSObject {
        public var outputFormat: OutputFormat = .dictionary
        public var taskHeadName: String?
        public var computeDevice: M3TMConfig.ComputeDevice?
        public var timeoutSeconds: TimeInterval = 30.0
        
        public override init() {
            super.init()
        }
    }
    
    /**
     * Arama sonucu
     */
    @objc public class SearchResult: NSObject {
        public var itemId: String
        public var score: Float
        public var metadata: [String: Any]?
        
        public init(itemId: String, score: Float, metadata: [String: Any]? = nil) {
            self.itemId = itemId
            self.score = score
            self.metadata = metadata
            super.init()
        }
    }
    
    // MARK: - Properties
    
    /// Model konfigürasyonu
    private let config: M3TMConfig
    
    /// Model durumu
    private var isModelLoaded = false
    
    // MARK: - Initialization
    
    /**
     * Model yöneticisi oluşturur
     *
     * @param config SDK konfigürasyonu
     */
    public init(config: M3TMConfig) {
        self.config = config
        super.init()
        
        // Başlatma stratejisine göre modeli yükle
        if config.loadingStrategy == .onInitialization {
            try? loadModel()
        }
        
        // Uygulama yaşam döngüsü dinleyicileri ekle
        setupApplicationLifecycleObservers()
    }
    
    deinit {
        // Yaşam döngüsü dinleyicilerini kaldır
        NotificationCenter.default.removeObserver(self)
    }
    
    // MARK: - Public Methods
    
    /**
     * Modeli yükler
     *
     * @throws Yükleme başarısız olursa hata döndürür
     */
    public func loadModel() throws {
        if isModelLoaded {
            return
        }
        
        // C++ köprüsü üzerinden model yükleme
        try M3TMBridge.shared.loadModel(configDict: config.asDictionary())
        isModelLoaded = true
    }
    
    /**
     * Model yüklü olup olmadığını kontrol eder
     *
     * @return Model yüklüyse true, değilse false
     */
    public func isLoaded() -> Bool {
        return isModelLoaded
    }
    
    /**
     * Modelden çıkarım yapar
     *
     * @param inputs Çıkarım için giriş verileri (batch)
     * @param options Çıkarım seçenekleri
     * @param completion Çıkarım tamamlandığında çağrılacak callback
     */
    public func inference(inputs: [InputData], options: InferenceOptions, completion: @escaping ([Any]?, Error?) -> Void) {
        // Model yüklü değilse ve otomatik yükleme seçilmişse yükle
        if !isModelLoaded && config.loadingStrategy == .onFirstUse {
            do {
                try loadModel()
            } catch {
                completion(nil, error)
                return
            }
        }
        
        // Model hala yüklü değilse hata döndür
        guard isModelLoaded else {
            completion(nil, M3TMError.modelNotLoaded)
            return
        }
        
        // Giriş verilerini dönüştür
        let processedInputs: [[String: Any]] = inputs.map { input in
            var inputDict: [String: Any] = [:]
            
            if let text = input.textData {
                inputDict["text"] = text
            }
            
            if let image = input.imageData {
                // UIImage -> NSData dönüşümü
                if let imageData = image.jpegData(compressionQuality: 0.9) {
                    inputDict["image"] = imageData
                }
            }
            
            if let features = input.additionalFeatures {
                inputDict["features"] = features
            }
            
            return inputDict
        }
        
        // Seçenekleri dönüştür
        var optionsDict: [String: Any] = [
            "outputFormat": options.outputFormat.rawValue,
            "timeoutSeconds": options.timeoutSeconds
        ]
        
        if let taskHead = options.taskHeadName {
            optionsDict["taskHeadName"] = taskHead
        }
        
        if let device = options.computeDevice {
            optionsDict["computeDevice"] = device.rawValue
        }
        
        // C++ köprüsü üzerinden inference yapma
        M3TMBridge.shared.performInference(inputs: processedInputs, options: optionsDict) { result, error in
            if let error = error {
                completion(nil, error)
                return
            }
            
            completion(result, nil)
        }
    }
    
    /**
     * Yeni bir görev başlığı ekler
     *
     * @param name Görev başlığı adı
     * @param config Görev başlığı yapılandırması
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func addTaskHead(name: String, config: [String: Any], completion: @escaping (Error?) -> Void) {
        guard isModelLoaded else {
            completion(M3TMError.modelNotLoaded)
            return
        }
        
        M3TMBridge.shared.addTaskHead(name: name, config: config) { error in
            completion(error)
        }
    }
    
    /**
     * Bir görev başlığını kaldırır
     *
     * @param name Görev başlığı adı
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func removeTaskHead(name: String, completion: @escaping (Error?) -> Void) {
        guard isModelLoaded else {
            completion(M3TMError.modelNotLoaded)
            return
        }
        
        M3TMBridge.shared.removeTaskHead(name: name) { error in
            completion(error)
        }
    }
    
    /**
     * Modeli boşaltır
     *
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func unloadModel(completion: @escaping (Error?) -> Void) {
        guard isModelLoaded else {
            completion(nil)
            return
        }
        
        M3TMBridge.shared.unloadModel { [weak self] error in
            guard let self = self else { return }
            
            if error == nil {
                self.isModelLoaded = false
            }
            
            completion(error)
        }
    }
    
    /**
     * Arama gömmelerini (embedding) oluşturur
     *
     * @param inputs Gömmeleri oluşturulacak giriş verileri (batch)
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func generateSearchEmbeddings(inputs: [InputData], completion: @escaping ([[Float]]?, Error?) -> Void) {
        guard isModelLoaded else {
            completion(nil, M3TMError.modelNotLoaded)
            return
        }
        
        // Giriş verilerini dönüştür
        let processedInputs: [[String: Any]] = inputs.map { input in
            var inputDict: [String: Any] = [:]
            
            if let text = input.textData {
                inputDict["text"] = text
            }
            
            if let image = input.imageData {
                if let imageData = image.jpegData(compressionQuality: 0.9) {
                    inputDict["image"] = imageData
                }
            }
            
            if let features = input.additionalFeatures {
                inputDict["features"] = features
            }
            
            return inputDict
        }
        
        M3TMBridge.shared.generateSearchEmbeddings(inputs: processedInputs) { embeddings, error in
            completion(embeddings as? [[Float]], error)
        }
    }
    
    // MARK: - Private Methods
    
    /**
     * Uygulama yaşam döngüsü olaylarını izlemek için gözlemciler ekler
     */
    private func setupApplicationLifecycleObservers() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleBackgroundNotification),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleForegroundNotification),
            name: UIApplication.willEnterForegroundNotification,
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleLowMemoryNotification),
            name: UIApplication.didReceiveMemoryWarningNotification,
            object: nil
        )
    }
    
    /**
     * Uygulama arka plana geçtiğinde çağrılır
     */
    @objc private func handleBackgroundNotification() {
        // Arka planda modeli boşaltma vs. gibi işlemler yapılabilir
        if config.autoSaveOnBackground {
            // İndeksleri kaydet
            M3TMBridge.shared.saveSearchIndexes(nil)
        }
    }
    
    /**
     * Uygulama ön plana döndüğünde çağrılır
     */
    @objc private func handleForegroundNotification() {
        // Gerekirse modeli tekrar yükleme gibi işlemler yapılabilir
    }
    
    /**
     * Düşük bellek uyarısı alındığında çağrılır
     */
    @objc private func handleLowMemoryNotification() {
        // Bellek optimizasyonu: Acil olmayan model bileşenlerini temizle
        if isModelLoaded {
            M3TMBridge.shared.optimizeMemoryUsage(nil)
        }
    }
} 