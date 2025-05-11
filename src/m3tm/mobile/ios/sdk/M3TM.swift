import Foundation

/**
 * M3TM - Mobil Multi-Modal Modüler Transformer iOS SDK
 *
 * M3TM, kullanıcının kişisel verileriyle tamamen cihaz üzerinde eğitilebilen,
 * bu veriler üzerinde semantik arama yapabilen, ve kullanıcıya verilerini indirme
 * imkanı sunan gizlilik odaklı bir yapay zeka modelidir.
 */
@objc public class M3TM: NSObject {
    
    // MARK: - Singleton
    
    /// Shared instance of the M3TM SDK
    @objc public static let shared = M3TM()
    
    // MARK: - Properties
    
    /// Model yöneticisi
    private var modelManager: M3TMModelManager?
    
    /// Eğitim yöneticisi
    private var trainingManager: M3TMTrainingManager?
    
    /// Arama servisi
    private var searchService: M3TMSearchService?
    
    /// Veri dışa aktarma servisi
    private var dataExporter: M3TMDataExporter?
    
    /// SDK sürüm bilgisi
    @objc public let version = "2.3.0"
    
    /// SDK'nın başlatılıp başlatılmadığını belirtir
    private var isInitialized = false
    
    // MARK: - Initialization
    
    private override init() {
        super.init()
    }
    
    /**
     * SDK'yı başlatır
     *
     * @param config SDK konfigürasyonu
     * @param completion Başlatma tamamlandığında çağrılacak callback
     */
    @objc public func initialize(config: M3TMConfig, completion: @escaping (Error?) -> Void) {
        if isInitialized {
            completion(M3TMError.alreadyInitialized)
            return
        }
        
        // C++ köprüsü üzerinden PyTorch Mobile başlatma
        M3TMBridge.shared.initializeRuntime(config: config) { [weak self] error in
            guard let self = self else { return }
            
            if let error = error {
                completion(error)
                return
            }
            
            // Yöneticileri oluştur
            self.modelManager = M3TMModelManager(config: config)
            self.trainingManager = M3TMTrainingManager(config: config)
            self.searchService = M3TMSearchService(indexDirectory: config.indexDirectory)
            self.dataExporter = M3TMDataExporter()
            
            // Arama servisini başlat
            self.searchService?.initialize { [weak self] error in
                guard let self = self else { return }
                
                if let error = error {
                    completion(error)
                    return
                }
                
                // Veri dışa aktarma servisini başlat
                self.dataExporter?.initialize { [weak self] error in
                    guard let self = self else { return }
                    
                    if let error = error {
                        completion(error)
                        return
                    }
                    
                    self.isInitialized = true
                    completion(nil)
                }
            }
        }
    }
    
    // MARK: - Public Methods
    
    /**
     * Model yöneticisine erişim sağlar
     *
     * @return Model yöneticisi
     */
    @objc public func getModelManager() throws -> M3TMModelManager {
        guard isInitialized else {
            throw M3TMError.notInitialized
        }
        
        guard let modelManager = modelManager else {
            throw M3TMError.internalError("Model manager not available")
        }
        
        return modelManager
    }
    
    /**
     * Eğitim yöneticisine erişim sağlar
     *
     * @return Eğitim yöneticisi
     */
    @objc public func getTrainingManager() throws -> M3TMTrainingManager {
        guard isInitialized else {
            throw M3TMError.notInitialized
        }
        
        guard let trainingManager = trainingManager else {
            throw M3TMError.internalError("Training manager not available")
        }
        
        return trainingManager
    }
    
    /**
     * Arama servisine erişim sağlar
     *
     * @return Arama servisi
     */
    @objc public func getSearchService() throws -> M3TMSearchService {
        guard isInitialized else {
            throw M3TMError.notInitialized
        }
        
        guard let searchService = searchService else {
            throw M3TMError.internalError("Search service not available")
        }
        
        return searchService
    }
    
    /**
     * Veri dışa aktarma servisine erişim sağlar
     *
     * @return Veri dışa aktarma servisi
     */
    @objc public func getDataExporter() throws -> M3TMDataExporter {
        guard isInitialized else {
            throw M3TMError.notInitialized
        }
        
        guard let dataExporter = dataExporter else {
            throw M3TMError.internalError("Data exporter not available")
        }
        
        return dataExporter
    }
    
    /**
     * SDK'yı temizler ve kapatır
     */
    @objc public func shutdown() {
        if isInitialized {
            // Arama indekslerini kaydet
            searchService?.saveIndex { _ in
                // İndeks kaydedildi (başarılı veya başarısız)
            }
            
            // C++ köprüsü üzerinden PyTorch Mobile kapatma
            M3TMBridge.shared.shutdownRuntime()
            
            // Yöneticileri temizle
            modelManager = nil
            trainingManager = nil
            searchService = nil
            dataExporter = nil
            
            isInitialized = false
        }
    }
} 