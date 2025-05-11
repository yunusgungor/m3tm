import Foundation
import UIKit

/**
 * M3TM arama servisi sınıfı. Model kullanarak semantik arama işlevleri sağlar.
 */
@objcMembers public class M3TMSearchService: NSObject {
    
    // MARK: - Types
    
    /**
     * Arama sonucu sınıfı
     */
    @objc public class SearchResult: NSObject {
        /// Öğe kimliği
        public let itemId: String
        
        /// Benzerlik skoru (0.0-1.0)
        public let score: Float
        
        /// Metin içeriği (varsa)
        public let textContent: String?
        
        /// Görüntü verisi (varsa)
        public let imageData: Data?
        
        /// Ek meta veriler
        public let metadata: [String: Any]?
        
        public init(itemId: String, score: Float, textContent: String? = nil, imageData: Data? = nil, metadata: [String: Any]? = nil) {
            self.itemId = itemId
            self.score = score
            self.textContent = textContent
            self.imageData = imageData
            self.metadata = metadata
            super.init()
        }
    }
    
    /**
     * Arama filtreleri
     */
    @objc public class SearchFilters: NSObject {
        /// İçerik türleri (text, image, both)
        public var contentTypes: [String]?
        
        /// Tarih aralığı (başlangıç)
        public var dateStart: Date?
        
        /// Tarih aralığı (bitiş)
        public var dateEnd: Date?
        
        /// Özel etiketler
        public var tags: [String]?
        
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
            
            if let customFilters = customFilters {
                dict["customFilters"] = customFilters
            }
            
            return dict
        }
    }
    
    // MARK: - Properties
    
    /// Arama servisi durumu
    private var isInitialized = false
    
    /// İndeks dizini
    private let indexDirectoryURL: URL?
    
    /// Kaydedilen indekslerin sayısı
    private var indexedItemCount: Int = 0
    
    // MARK: - Initialization
    
    /**
     * Arama servisi oluşturur
     *
     * @param indexDirectory İndekslerin saklanacağı dizin (nil ise varsayılan konum kullanılır)
     */
    public init(indexDirectory: String? = nil) {
        // İndeks dizinini belirle
        if let indexDirPath = indexDirectory {
            self.indexDirectoryURL = URL(fileURLWithPath: indexDirPath)
        } else {
            // Varsayılan konum: Belge dizini altında bir klasör
            let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            self.indexDirectoryURL = documentsURL.appendingPathComponent("m3tm_search_indexes")
        }
        
        super.init()
    }
    
    // MARK: - Public Methods
    
    /**
     * Arama servisini başlatır
     *
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func initialize(completion: @escaping (Error?) -> Void) {
        if isInitialized {
            completion(nil)
            return
        }
        
        // İndeks dizini yoksa oluştur
        if let indexDirURL = indexDirectoryURL {
            do {
                try FileManager.default.createDirectory(at: indexDirURL, withIntermediateDirectories: true, attributes: nil)
            } catch {
                completion(M3TMError.fileAccessDenied)
                return
            }
        }
        
        // Önceki indeksleri yükle
        let indexDirString = indexDirectoryURL?.path ?? ""
        
        M3TMBridge.shared.initializeSearchService(indexDirectory: indexDirString) { [weak self] itemCount, error in
            guard let self = self else { return }
            
            if let error = error {
                completion(error)
                return
            }
            
            self.isInitialized = true
            self.indexedItemCount = itemCount ?? 0
            completion(nil)
        }
    }
    
    /**
     * Bir öğeyi indekse ekler
     *
     * @param itemId Öğe kimliği
     * @param textContent Metin içeriği
     * @param image Görüntü verisi
     * @param metadata Ek meta veriler
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func indexItem(itemId: String, textContent: String? = nil, image: UIImage? = nil, metadata: [String: Any]? = nil, completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.searchIndexNotInitialized)
            return
        }
        
        var itemDict: [String: Any] = ["itemId": itemId]
        
        if let text = textContent {
            itemDict["text"] = text
        }
        
        if let image = image, let imageData = image.jpegData(compressionQuality: 0.9) {
            itemDict["image"] = imageData
        }
        
        if let metadata = metadata {
            itemDict["metadata"] = metadata
        }
        
        M3TMBridge.shared.indexItem(itemDict: itemDict) { [weak self] error in
            guard let self = self else { return }
            
            if error == nil {
                self.indexedItemCount += 1
            }
            
            completion(error)
        }
    }
    
    /**
     * Bir öğeyi indeksten kaldırır
     *
     * @param itemId Öğe kimliği
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func removeItem(itemId: String, completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.searchIndexNotInitialized)
            return
        }
        
        M3TMBridge.shared.removeItemFromIndex(itemId: itemId) { [weak self] error in
            guard let self = self else { return }
            
            if error == nil && self.indexedItemCount > 0 {
                self.indexedItemCount -= 1
            }
            
            completion(error)
        }
    }
    
    /**
     * Metin sorgusu kullanarak arama yapar
     *
     * @param textQuery Metin sorgusu
     * @param filters Arama filtreleri
     * @param topK Döndürülecek maksimum sonuç sayısı
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func search(textQuery: String, filters: SearchFilters? = nil, topK: Int = 10, completion: @escaping ([SearchResult]?, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, M3TMError.searchIndexNotInitialized)
            return
        }
        
        var searchOptions: [String: Any] = ["topK": topK]
        
        if let filters = filters {
            searchOptions["filters"] = filters.asDictionary()
        }
        
        M3TMBridge.shared.search(textQuery: textQuery, imageData: nil, options: searchOptions) { results, error in
            if let error = error {
                completion(nil, error)
                return
            }
            
            guard let resultArray = results as? [[String: Any]] else {
                completion(nil, M3TMError.internalError("Invalid search results format"))
                return
            }
            
            let searchResults = resultArray.compactMap { resultDict -> SearchResult? in
                guard let itemId = resultDict["itemId"] as? String,
                      let score = resultDict["score"] as? Float else {
                    return nil
                }
                
                let textContent = resultDict["text"] as? String
                let imageData = resultDict["image"] as? Data
                let metadata = resultDict["metadata"] as? [String: Any]
                
                return SearchResult(itemId: itemId, score: score, textContent: textContent, imageData: imageData, metadata: metadata)
            }
            
            completion(searchResults, nil)
        }
    }
    
    /**
     * Görüntü kullanarak arama yapar
     *
     * @param image Görüntü verisi
     * @param filters Arama filtreleri
     * @param topK Döndürülecek maksimum sonuç sayısı
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func search(image: UIImage, filters: SearchFilters? = nil, topK: Int = 10, completion: @escaping ([SearchResult]?, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, M3TMError.searchIndexNotInitialized)
            return
        }
        
        guard let imageData = image.jpegData(compressionQuality: 0.9) else {
            completion(nil, M3TMError.inputDataInvalid)
            return
        }
        
        var searchOptions: [String: Any] = ["topK": topK]
        
        if let filters = filters {
            searchOptions["filters"] = filters.asDictionary()
        }
        
        M3TMBridge.shared.search(textQuery: nil, imageData: imageData, options: searchOptions) { results, error in
            if let error = error {
                completion(nil, error)
                return
            }
            
            guard let resultArray = results as? [[String: Any]] else {
                completion(nil, M3TMError.internalError("Invalid search results format"))
                return
            }
            
            let searchResults = resultArray.compactMap { resultDict -> SearchResult? in
                guard let itemId = resultDict["itemId"] as? String,
                      let score = resultDict["score"] as? Float else {
                    return nil
                }
                
                let textContent = resultDict["text"] as? String
                let imageData = resultDict["image"] as? Data
                let metadata = resultDict["metadata"] as? [String: Any]
                
                return SearchResult(itemId: itemId, score: score, textContent: textContent, imageData: imageData, metadata: metadata)
            }
            
            completion(searchResults, nil)
        }
    }
    
    /**
     * Metin ve görüntü birlikte kullanarak multimodal arama yapar
     *
     * @param textQuery Metin sorgusu
     * @param image Görüntü verisi
     * @param filters Arama filtreleri
     * @param topK Döndürülecek maksimum sonuç sayısı
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func search(textQuery: String, image: UIImage, filters: SearchFilters? = nil, topK: Int = 10, completion: @escaping ([SearchResult]?, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, M3TMError.searchIndexNotInitialized)
            return
        }
        
        guard let imageData = image.jpegData(compressionQuality: 0.9) else {
            completion(nil, M3TMError.inputDataInvalid)
            return
        }
        
        var searchOptions: [String: Any] = ["topK": topK]
        
        if let filters = filters {
            searchOptions["filters"] = filters.asDictionary()
        }
        
        M3TMBridge.shared.search(textQuery: textQuery, imageData: imageData, options: searchOptions) { results, error in
            if let error = error {
                completion(nil, error)
                return
            }
            
            guard let resultArray = results as? [[String: Any]] else {
                completion(nil, M3TMError.internalError("Invalid search results format"))
                return
            }
            
            let searchResults = resultArray.compactMap { resultDict -> SearchResult? in
                guard let itemId = resultDict["itemId"] as? String,
                      let score = resultDict["score"] as? Float else {
                    return nil
                }
                
                let textContent = resultDict["text"] as? String
                let imageData = resultDict["image"] as? Data
                let metadata = resultDict["metadata"] as? [String: Any]
                
                return SearchResult(itemId: itemId, score: score, textContent: textContent, imageData: imageData, metadata: metadata)
            }
            
            completion(searchResults, nil)
        }
    }
    
    /**
     * İndeksi kaydeder
     *
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func saveIndex(completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.searchIndexNotInitialized)
            return
        }
        
        M3TMBridge.shared.saveSearchIndexes { error in
            completion(error)
        }
    }
    
    /**
     * İndeksi temizler
     *
     * @param completion İşlem tamamlandığında çağrılacak callback
     */
    public func clearIndex(completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.searchIndexNotInitialized)
            return
        }
        
        M3TMBridge.shared.clearSearchIndex { [weak self] error in
            guard let self = self else { return }
            
            if error == nil {
                self.indexedItemCount = 0
            }
            
            completion(error)
        }
    }
    
    /**
     * İndekste bulunan öğe sayısını döndürür
     *
     * @return Öğe sayısı
     */
    public func getIndexedItemCount() -> Int {
        return indexedItemCount
    }
    
    /**
     * Arama servisinin başlatılıp başlatılmadığını döndürür
     *
     * @return Başlatıldıysa true, değilse false
     */
    public func isIndexInitialized() -> Bool {
        return isInitialized
    }
} 