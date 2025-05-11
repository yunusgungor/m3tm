import Foundation
import UIKit

/**
 * M3TM eğitim yöneticisi protokolü. Eğitim sürecinde ilerleme, tamamlanma veya hata gibi olayları bildirmek için kullanılır.
 */
@objc public protocol M3TMTrainingDelegate: AnyObject {
    /**
     * Eğitim ilerlemesini bildirir
     *
     * @param manager Eğitim yöneticisi
     * @param progress İlerleme yüzdesi (0.0-1.0)
     * @param metrics İlerleme metrikleri
     * @param epoch Epoch numarası
     */
    @objc optional func trainingManager(_ manager: M3TMTrainingManager, didUpdateProgress progress: Float, metrics: [String: Any]?, forEpoch epoch: Int)
    
    /**
     * Bir epoch'un tamamlandığını bildirir
     *
     * @param manager Eğitim yöneticisi
     * @param metrics Tamamlanan epoch için metrikler
     * @param epoch Tamamlanan epoch numarası
     */
    @objc optional func trainingManager(_ manager: M3TMTrainingManager, didCompleteEpoch epoch: Int, withMetrics metrics: [String: Any]?)
    
    /**
     * Eğitimin tamamlandığını bildirir
     *
     * @param manager Eğitim yöneticisi
     * @param success Eğitim başarılı olduysa true, değilse false
     * @param finalMetrics Son metrikler
     */
    @objc optional func trainingManager(_ manager: M3TMTrainingManager, didFinishWithSuccess success: Bool, finalMetrics: [String: Any]?)
    
    /**
     * Eğitim sırasında bir hata oluştuğunu bildirir
     *
     * @param manager Eğitim yöneticisi
     * @param error Oluşan hata
     */
    @objc optional func trainingManager(_ manager: M3TMTrainingManager, didEncounterError error: Error)
}

/**
 * M3TM eğitim yöneticisi sınıfı. Model adaptörlerini ve görev başlıklarını cihaz üzerinde eğitmek için gereken işlevleri sağlar.
 */
@objcMembers public class M3TMTrainingManager: NSObject {
    
    // MARK: - Types
    
    /**
     * Eğitim yapılandırması
     */
    @objc public class TrainingConfig: NSObject {
        /// Öğrenme oranı
        public var learningRate: Float = 0.001
        
        /// Epoch sayısı
        public var epochs: Int = 10
        
        /// Batch büyüklüğü
        public var batchSize: Int = 8
        
        /// Ağırlık azaltma
        public var weightDecay: Float = 0.01
        
        /// Erken durdurma toleransı
        public var earlyStoppingPatience: Int = 5
        
        /// Öğrenme oranı düşürme faktörü
        public var learningRateDecay: Float = 0.5
        
        /// Optimizer tipi (0: Adam, 1: SGD, 2: AdamW)
        public var optimizerType: Int = 2
        
        /// Maksimum gradient norm
        public var maxGradNorm: Float = 1.0
        
        /// Eğitim sırasında kullanılacak compute device
        public var computeDevice: M3TMConfig.ComputeDevice?
        
        /// Eğitim verisi karıştırma
        public var shuffle: Bool = true
        
        /// Doğrulama oranı
        public var validationSplit: Float = 0.2
        
        /// Özel seçenekler
        public var customOptions: [String: Any]?
        
        public override init() {
            super.init()
        }
        
        /**
         * Yapılandırmayı sözlük olarak döndürür
         */
        public func asDictionary() -> [String: Any] {
            var dict: [String: Any] = [
                "learningRate": learningRate,
                "epochs": epochs,
                "batchSize": batchSize,
                "weightDecay": weightDecay,
                "earlyStoppingPatience": earlyStoppingPatience,
                "learningRateDecay": learningRateDecay,
                "optimizerType": optimizerType,
                "maxGradNorm": maxGradNorm,
                "shuffle": shuffle,
                "validationSplit": validationSplit
            ]
            
            if let computeDevice = computeDevice {
                dict["computeDevice"] = computeDevice.rawValue
            }
            
            if let customOptions = customOptions, !customOptions.isEmpty {
                dict["customOptions"] = customOptions
            }
            
            return dict
        }
    }
    
    /**
     * Veri toplayıcısı protokolü. Eğitim sırasında veri sunmak için kullanılır.
     */
    @objc public protocol DataProvider {
        /**
         * Toplam örnek sayısını döndürür
         */
        func getDataCount() -> Int
        
        /**
         * Batch veri noktalarını döndürür
         *
         * @param indices İndeksler
         */
        func getData(forIndices indices: [Int]) -> [M3TMModelManager.InputData]
        
        /**
         * Eğitim verilerinin etiketlerini döndürür
         *
         * @param indices İndeksler
         */
        func getLabels(forIndices indices: [Int]) -> [Any]
        
        /**
         * Veri sağlayıcı hazırlanıyor
         */
        @objc optional func prepareDataProvider()
        
        /**
         * Veri sağlayıcı kapatılıyor
         */
        @objc optional func closeDataProvider()
    }
    
    // MARK: - Properties
    
    /// SDK konfigürasyonu
    private let config: M3TMConfig
    
    /// Eğitim delegesi
    public weak var delegate: M3TMTrainingDelegate?
    
    /// Eğitim durumu
    private var isTrainingInProgress = false
    
    /// Eğitim iptal edildi mi
    private var isCancelled = false
    
    // MARK: - Initialization
    
    /**
     * Eğitim yöneticisi oluşturur
     *
     * @param config SDK konfigürasyonu
     */
    public init(config: M3TMConfig) {
        self.config = config
        super.init()
    }
    
    // MARK: - Public Methods
    
    /**
     * Yeni bir adaptör oluşturur
     *
     * @param blockIndex Adaptörün ekleneceği blok indeksi
     * @param config Adaptör yapılandırması
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func createAdapter(forBlockIndex blockIndex: Int, config: [String: Any], completion: @escaping (String?, Error?) -> Void) {
        M3TMBridge.shared.createAdapter(blockIndex: blockIndex, config: config) { adapterId, error in
            completion(adapterId, error)
        }
    }
    
    /**
     * Adaptör eğitir
     *
     * @param adapterId Eğitilecek adaptör ID'si
     * @param dataProvider Eğitim verisi sağlayıcısı
     * @param config Eğitim yapılandırması
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func trainAdapter(adapterId: String, dataProvider: DataProvider, config: TrainingConfig, completion: @escaping (Error?) -> Void) {
        // Zaten eğitim devam ediyorsa hata döndür
        guard !isTrainingInProgress else {
            completion(M3TMError.trainingInProgress)
            return
        }
        
        isTrainingInProgress = true
        isCancelled = false
        
        // Veri sağlayıcısını hazırla
        dataProvider.prepareDataProvider?()
        
        // Veri sayısını al
        let dataCount = dataProvider.getDataCount()
        guard dataCount > 0 else {
            isTrainingInProgress = false
            completion(M3TMError.trainingDataInvalid)
            dataProvider.closeDataProvider?()
            return
        }
        
        // Konfigürasyonu dict'e dönüştür
        let configDict = config.asDictionary()
        
        // Bridge üzerinden eğitimi başlat
        M3TMBridge.shared.trainAdapter(adapterId: adapterId, config: configDict) { [weak self] batchIndices, epoch, batchIndex, totalBatches in
            guard let self = self, !self.isCancelled else { return nil }
            
            // İlerleme güncellemesi için delegate'e bildir
            let progress = Float(batchIndex) / Float(totalBatches)
            self.delegate?.trainingManager?(self, didUpdateProgress: progress, metrics: nil, forEpoch: epoch)
            
            // İstenen indekslerdeki verileri ve etiketleri getir
            let batchData = dataProvider.getData(forIndices: batchIndices)
            let batchLabels = dataProvider.getLabels(forIndices: batchIndices)
            
            // Verileri dönüştür
            var processedData: [[String: Any]] = []
            for (index, input) in batchData.enumerated() {
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
                
                // Etiketi de inputDict içerisine ekle
                if index < batchLabels.count {
                    inputDict["label"] = batchLabels[index]
                }
                
                processedData.append(inputDict)
            }
            
            return processedData
        } progressCallback: { [weak self] epoch, metrics in
            guard let self = self, !self.isCancelled else { return }
            
            // Epoch tamamlanma için delegate'e bildir
            self.delegate?.trainingManager?(self, didCompleteEpoch: epoch, withMetrics: metrics)
        } completionCallback: { [weak self] success, finalMetrics, error in
            guard let self = self else { return }
            
            self.isTrainingInProgress = false
            
            // Veri sağlayıcısını kapat
            dataProvider.closeDataProvider?()
            
            // Hata varsa delegate'e bildir
            if let error = error {
                self.delegate?.trainingManager?(self, didEncounterError: error)
                completion(error)
                return
            }
            
            // Eğitim tamamlandı, delegate'e bildir
            self.delegate?.trainingManager?(self, didFinishWithSuccess: success, finalMetrics: finalMetrics)
            completion(nil)
        }
    }
    
    /**
     * Bir görev başlığını eğitir
     *
     * @param taskHeadName Eğitilecek görev başlığı adı
     * @param dataProvider Eğitim verisi sağlayıcısı
     * @param config Eğitim yapılandırması
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func trainTaskHead(taskHeadName: String, dataProvider: DataProvider, config: TrainingConfig, completion: @escaping (Error?) -> Void) {
        guard !isTrainingInProgress else {
            completion(M3TMError.trainingInProgress)
            return
        }
        
        isTrainingInProgress = true
        isCancelled = false
        
        // Veri sağlayıcısını hazırla
        dataProvider.prepareDataProvider?()
        
        // Veri sayısını al
        let dataCount = dataProvider.getDataCount()
        guard dataCount > 0 else {
            isTrainingInProgress = false
            completion(M3TMError.trainingDataInvalid)
            dataProvider.closeDataProvider?()
            return
        }
        
        // Konfigürasyonu dict'e dönüştür
        let configDict = config.asDictionary()
        
        // Bridge üzerinden eğitimi başlat
        M3TMBridge.shared.trainTaskHead(taskHeadName: taskHeadName, config: configDict) { [weak self] batchIndices, epoch, batchIndex, totalBatches in
            guard let self = self, !self.isCancelled else { return nil }
            
            // İlerleme güncellemesi için delegate'e bildir
            let progress = Float(batchIndex) / Float(totalBatches)
            self.delegate?.trainingManager?(self, didUpdateProgress: progress, metrics: nil, forEpoch: epoch)
            
            // İstenen indekslerdeki verileri ve etiketleri getir
            let batchData = dataProvider.getData(forIndices: batchIndices)
            let batchLabels = dataProvider.getLabels(forIndices: batchIndices)
            
            // Verileri dönüştür
            var processedData: [[String: Any]] = []
            for (index, input) in batchData.enumerated() {
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
                
                // Etiketi de inputDict içerisine ekle
                if index < batchLabels.count {
                    inputDict["label"] = batchLabels[index]
                }
                
                processedData.append(inputDict)
            }
            
            return processedData
        } progressCallback: { [weak self] epoch, metrics in
            guard let self = self, !self.isCancelled else { return }
            
            // Epoch tamamlanma için delegate'e bildir
            self.delegate?.trainingManager?(self, didCompleteEpoch: epoch, withMetrics: metrics)
        } completionCallback: { [weak self] success, finalMetrics, error in
            guard let self = self else { return }
            
            self.isTrainingInProgress = false
            
            // Veri sağlayıcısını kapat
            dataProvider.closeDataProvider?()
            
            // Hata varsa delegate'e bildir
            if let error = error {
                self.delegate?.trainingManager?(self, didEncounterError: error)
                completion(error)
                return
            }
            
            // Eğitim tamamlandı, delegate'e bildir
            self.delegate?.trainingManager?(self, didFinishWithSuccess: success, finalMetrics: finalMetrics)
            completion(nil)
        }
    }
    
    /**
     * Devam eden eğitimi iptal eder
     */
    public func cancelTraining() {
        if isTrainingInProgress {
            isCancelled = true
            M3TMBridge.shared.cancelTraining { [weak self] error in
                guard let self = self else { return }
                
                if let error = error {
                    self.delegate?.trainingManager?(self, didEncounterError: error)
                } else {
                    self.delegate?.trainingManager?(self, didFinishWithSuccess: false, finalMetrics: nil)
                }
                
                self.isTrainingInProgress = false
            }
        }
    }
    
    /**
     * Bir adaptörü kaldırır
     *
     * @param adapterId Adaptör ID'si
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func removeAdapter(adapterId: String, completion: @escaping (Error?) -> Void) {
        M3TMBridge.shared.removeAdapter(adapterId: adapterId) { error in
            completion(error)
        }
    }
    
    /**
     * Eğitim durumunu döndürür
     *
     * @return Eğitim devam ediyorsa true, değilse false
     */
    public func isTraining() -> Bool {
        return isTrainingInProgress
    }
} 