import Foundation
import UIKit

/**
 * M3TM veri dışa aktarma sınıfı. Kullanıcıların kişisel verilerini indirmelerini ve dışa aktarmalarını sağlar.
 */
@objcMembers public class M3TMDataExporter: NSObject {
    
    // MARK: - Types
    
    /**
     * Dışa aktarma formatı
     */
    @objc public enum ExportFormat: Int {
        /// JSON formatında dışa aktar
        case json = 0
        /// CSV formatında dışa aktar
        case csv = 1
        /// Özel formatında dışa aktar
        case custom = 2
    }
    
    /**
     * Dışa aktarma filtreleri
     */
    @objc public class ExportFilters: NSObject {
        /// İçerik türleri (text, image, both)
        public var contentTypes: [String]?
        
        /// Tarih aralığı (başlangıç)
        public var dateStart: Date?
        
        /// Tarih aralığı (bitiş)
        public var dateEnd: Date?
        
        /// Özel etiketler
        public var tags: [String]?
        
        /// Öğe ID'leri
        public var itemIds: [String]?
        
        /// Özel filtreler
        public var customFilters: [String: Any]?
        
        public override init() {
            super.init()
        }
        
        /**
         * Filtreleri sözlük olarak döndürür
         */
        public func asDictionary() -> [String: Any] {
            var dict: [String: Any] = [:]
            
            if let contentTypes = contentTypes {
                dict["contentTypes"] = contentTypes
            }
            
            if let dateStart = dateStart {
                dict["dateStart"] = Int(dateStart.timeIntervalSince1970)
            }
            
            if let dateEnd = dateEnd {
                dict["dateEnd"] = Int(dateEnd.timeIntervalSince1970)
            }
            
            if let tags = tags {
                dict["tags"] = tags
            }
            
            if let itemIds = itemIds {
                dict["itemIds"] = itemIds
            }
            
            if let customFilters = customFilters {
                dict["customFilters"] = customFilters
            }
            
            return dict
        }
    }
    
    /**
     * Dışa aktarma seçenekleri
     */
    @objc public class ExportOptions: NSObject {
        /// Dışa aktarma formatı
        public var format: ExportFormat = .json
        
        /// Ham verileri dahil et
        public var includeRawData: Bool = false
        
        /// Görüntü verilerini dahil et
        public var includeImages: Bool = true
        
        /// Görüntü kalitesi (0.0-1.0)
        public var imageQuality: Float = 0.9
        
        /// Görüntüleri yeniden boyutlandırma faktörü (1.0 = orijinal)
        public var imageResizeFactor: Float = 1.0
        
        /// Meta verileri dahil et
        public var includeMetadata: Bool = true
        
        /// Özel seçenekler
        public var customOptions: [String: Any]?
        
        public override init() {
            super.init()
        }
        
        /**
         * Seçenekleri sözlük olarak döndürür
         */
        public func asDictionary() -> [String: Any] {
            var dict: [String: Any] = [
                "format": format.rawValue,
                "includeRawData": includeRawData,
                "includeImages": includeImages,
                "imageQuality": imageQuality,
                "imageResizeFactor": imageResizeFactor,
                "includeMetadata": includeMetadata
            ]
            
            if let customOptions = customOptions {
                dict["customOptions"] = customOptions
            }
            
            return dict
        }
    }
    
    // MARK: - Properties
    
    /// Dışa aktarma servisi durumu
    private var isInitialized = false
    
    /// Dışa aktarılan dosyaların kaydedileceği dizin
    private let exportDirectoryURL: URL?
    
    // MARK: - Initialization
    
    /**
     * Veri dışa aktarma servisi oluşturur
     *
     * @param exportDirectory Dışa aktarılan dosyaların kaydedileceği dizin (nil ise varsayılan konum kullanılır)
     */
    public init(exportDirectory: String? = nil) {
        // Dışa aktarma dizinini belirle
        if let exportDirPath = exportDirectory {
            self.exportDirectoryURL = URL(fileURLWithPath: exportDirPath)
        } else {
            // Varsayılan konum: Belge dizini altında bir klasör
            let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            self.exportDirectoryURL = documentsURL.appendingPathComponent("m3tm_exports")
        }
        
        super.init()
    }
    
    // MARK: - Public Methods
    
    /**
     * Dışa aktarma servisini başlatır
     *
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func initialize(completion: @escaping (Error?) -> Void) {
        if isInitialized {
            completion(nil)
            return
        }
        
        // Dışa aktarma dizini yoksa oluştur
        if let exportDirURL = exportDirectoryURL {
            do {
                try FileManager.default.createDirectory(at: exportDirURL, withIntermediateDirectories: true, attributes: nil)
            } catch {
                completion(M3TMError.fileAccessDenied)
                return
            }
        }
        
        // C++ köprüsü üzerinden dışa aktarma servisini başlat
        let exportDirString = exportDirectoryURL?.path ?? ""
        
        M3TMBridge.shared.initializeExportService(exportDirectory: exportDirString) { [weak self] error in
            guard let self = self else { return }
            
            if let error = error {
                completion(error)
                return
            }
            
            self.isInitialized = true
            completion(nil)
        }
    }
    
    /**
     * Verileri dışa aktarır
     *
     * @param filters Dışa aktarma filtreleri
     * @param options Dışa aktarma seçenekleri
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func exportData(filters: ExportFilters, options: ExportOptions, completion: @escaping (URL?, Int, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, 0, M3TMError.notInitialized)
            return
        }
        
        // Filtreleri ve seçenekleri dict'e dönüştür
        let filtersDict = filters.asDictionary()
        let optionsDict = options.asDictionary()
        
        // Dosya adı oluştur
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        
        let formatExtension: String
        switch options.format {
        case .json:
            formatExtension = "json"
        case .csv:
            formatExtension = "csv"
        case .custom:
            formatExtension = "dat"
        }
        
        let filename = "m3tm_export_\(timestamp).\(formatExtension)"
        
        // Dışa aktarma işlemini başlat
        M3TMBridge.shared.exportData(filters: filtersDict, options: optionsDict, filename: filename) { [weak self] exportedFilePath, itemCount, error in
            guard let self = self else { return }
            
            if let error = error {
                completion(nil, 0, error)
                return
            }
            
            guard let path = exportedFilePath else {
                completion(nil, 0, M3TMError.exportFailed)
                return
            }
            
            let fileURL = URL(fileURLWithPath: path)
            completion(fileURL, itemCount ?? 0, nil)
        }
    }
    
    /**
     * Belirli öğeleri dışa aktarır
     *
     * @param itemIds Dışa aktarılacak öğe ID'leri
     * @param options Dışa aktarma seçenekleri
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func exportItems(itemIds: [String], options: ExportOptions, completion: @escaping (URL?, Int, Error?) -> Void) {
        let filters = ExportFilters()
        filters.itemIds = itemIds
        
        exportData(filters: filters, options: options, completion: completion)
    }
    
    /**
     * Son dışa aktarılan verilerin listesini döndürür
     *
     * @param maxCount En fazla kaç adet dışa aktarım dosyası listeleneceği
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func getExportedFiles(maxCount: Int, completion: @escaping ([URL]?, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, M3TMError.notInitialized)
            return
        }
        
        guard let exportDirURL = exportDirectoryURL else {
            completion(nil, M3TMError.internalError("Export directory not available"))
            return
        }
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(at: exportDirURL, includingPropertiesForKeys: [.creationDateKey], options: .skipsHiddenFiles)
            
            // Dosyaları oluşturulma tarihine göre sırala (en yeni en üstte)
            let sortedURLs = try fileURLs.sorted { url1, url2 in
                let values1 = try url1.resourceValues(forKeys: [.creationDateKey])
                let values2 = try url2.resourceValues(forKeys: [.creationDateKey])
                
                if let date1 = values1.creationDate, let date2 = values2.creationDate {
                    return date1 > date2
                }
                
                return false
            }
            
            // İstenen sayıda dosyayı döndür
            let limitedURLs: [URL]
            if maxCount > 0 && sortedURLs.count > maxCount {
                limitedURLs = Array(sortedURLs.prefix(maxCount))
            } else {
                limitedURLs = sortedURLs
            }
            
            completion(limitedURLs, nil)
        } catch {
            completion(nil, M3TMError.fileAccessDenied)
        }
    }
    
    /**
     * Bir dışa aktarma dosyasını siler
     *
     * @param fileURL Silinecek dosyanın URL'si
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func deleteExportedFile(fileURL: URL, completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.notInitialized)
            return
        }
        
        do {
            try FileManager.default.removeItem(at: fileURL)
            completion(nil)
        } catch {
            completion(M3TMError.fileAccessDenied)
        }
    }
    
    /**
     * Tüm dışa aktarma dosyalarını siler
     *
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func clearExportedFiles(completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.notInitialized)
            return
        }
        
        guard let exportDirURL = exportDirectoryURL else {
            completion(M3TMError.internalError("Export directory not available"))
            return
        }
        
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(at: exportDirURL, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
            
            for fileURL in fileURLs {
                try FileManager.default.removeItem(at: fileURL)
            }
            
            completion(nil)
        } catch {
            completion(M3TMError.fileAccessDenied)
        }
    }
    
    /**
     * Dışa aktarma servisinin başlatılıp başlatılmadığını döndürür
     *
     * @return Başlatıldıysa true, değilse false
     */
    public func isExportInitialized() -> Bool {
        return isInitialized
    }
} 