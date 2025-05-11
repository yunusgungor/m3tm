import Foundation

/**
 * M3TM SDK konfigürasyonu
 */
@objcMembers public class M3TMConfig: NSObject {
    
    // MARK: - Enum Types
    
    /**
     * Kullanılacak hesaplama cihazı
     */
    @objc public enum ComputeDevice: Int {
        /// Sadece CPU kullan
        case cpu = 0
        /// Metal kullan (eğer destekleniyorsa)
        case metal = 1
        /// Otomatik olarak en iyi olanı seç
        case auto = 2
    }
    
    /**
     * Model yükleme stratejisi
     */
    @objc public enum ModelLoadingStrategy: Int {
        /// Başlatma sırasında yükle
        case onInitialization = 0
        /// İlk kullanım sırasında yükle
        case onFirstUse = 1
        /// Manuel olarak yükle
        case manual = 2
    }
    
    /**
     * Yapılan optimizasyon seviyesi
     */
    @objc public enum OptimizationLevel: Int {
        /// Optimizasyon yok
        case none = 0
        /// Temel optimizasyonlar
        case basic = 1
        /// Hafıza odaklı optimizasyonlar
        case memory = 2
        /// Hız odaklı optimizasyonlar
        case speed = 3
        /// Tüm optimizasyonlar (hız ve hafıza)
        case full = 4
    }
    
    // MARK: - Properties
    
    /// Model dosyasının yolu. Bundle içindeki model dosyasına göre relatif olabilir.
    public var modelPath: String
    
    /// Yapılandırma dosyasının yolu. Bundle içindeki yapılandırma dosyasına göre relatif olabilir.
    public var configPath: String
    
    /// Compute device
    public var computeDevice: ComputeDevice
    
    /// Model yükleme stratejisi
    public var loadingStrategy: ModelLoadingStrategy
    
    /// Optimizasyon seviyesi
    public var optimizationLevel: OptimizationLevel
    
    /// İndeks verilerinin kaydedileceği dizin
    public var indexDirectory: String?
    
    /// Maksimum bellek kullanımı (MB)
    public var maxMemoryUsageMB: Int
    
    /// İndekslerin otomatik olarak kaydedilip kaydedilmeyeceği
    public var autoSaveIndexes: Bool
    
    /// Arka planda çalışma sırasında indekslerin otomatik olarak kaydedilip kaydedilmeyeceği
    public var autoSaveOnBackground: Bool
    
    /// CoreML entegrasyonunu kullanıp kullanmayacağı
    public var useCoreML: Bool
    
    /// Dil ayarları
    public var languageSettings: [String: Any]
    
    /// Kayıt seviyesi
    public var logLevel: Int
    
    // MARK: - Initialization
    
    /**
     * Varsayılan konfigürasyon oluştur
     *
     * @param modelPath Model dosyasının yolu
     * @param configPath Yapılandırma dosyasının yolu
     */
    public init(modelPath: String, configPath: String) {
        self.modelPath = modelPath
        self.configPath = configPath
        self.computeDevice = .auto
        self.loadingStrategy = .onFirstUse
        self.optimizationLevel = .basic
        self.maxMemoryUsageMB = 512
        self.autoSaveIndexes = true
        self.autoSaveOnBackground = true
        self.useCoreML = false
        self.languageSettings = [:]
        self.logLevel = 1
        super.init()
    }
    
    /**
     * Kopyalama metodu - mevcut konfigürasyonun bir kopyasını oluşturur
     */
    public func copy() -> M3TMConfig {
        let copy = M3TMConfig(modelPath: self.modelPath, configPath: self.configPath)
        copy.computeDevice = self.computeDevice
        copy.loadingStrategy = self.loadingStrategy
        copy.optimizationLevel = self.optimizationLevel
        copy.indexDirectory = self.indexDirectory
        copy.maxMemoryUsageMB = self.maxMemoryUsageMB
        copy.autoSaveIndexes = self.autoSaveIndexes
        copy.autoSaveOnBackground = self.autoSaveOnBackground
        copy.useCoreML = self.useCoreML
        copy.languageSettings = self.languageSettings
        copy.logLevel = self.logLevel
        return copy
    }
    
    /**
     * Yapılandırmayı JSON formatında sözlük olarak döndürür
     */
    public func asDictionary() -> [String: Any] {
        var dict: [String: Any] = [
            "modelPath": modelPath,
            "configPath": configPath,
            "computeDevice": computeDevice.rawValue,
            "loadingStrategy": loadingStrategy.rawValue,
            "optimizationLevel": optimizationLevel.rawValue,
            "maxMemoryUsageMB": maxMemoryUsageMB,
            "autoSaveIndexes": autoSaveIndexes,
            "autoSaveOnBackground": autoSaveOnBackground,
            "useCoreML": useCoreML,
            "logLevel": logLevel
        ]
        
        if let indexDirectory = indexDirectory {
            dict["indexDirectory"] = indexDirectory
        }
        
        if !languageSettings.isEmpty {
            dict["languageSettings"] = languageSettings
        }
        
        return dict
    }
} 