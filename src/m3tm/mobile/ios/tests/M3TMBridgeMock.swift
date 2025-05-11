import Foundation
@testable import M3TM

/**
 * Test amaçlı M3TMBridge mock sınıfı
 */
class M3TMBridgeMock: M3TMBridge {
    // Mock için kullanılacak beklenen sonuçlar ve hatalar
    var shouldSucceedInitialize = true
    var shouldSucceedLoadModel = true
    var shouldSucceedSearch = true
    var shouldSucceedExport = true
    
    // Mock için kullanılacak test verileri
    var testSearchResults: [[String: Any]] = [
        ["itemId": "test1", "score": 0.95, "text": "Test sonucu 1"],
        ["itemId": "test2", "score": 0.85, "text": "Test sonucu 2"]
    ]
    
    var testModelInfo: [String: Any] = [
        "name": "TestModel",
        "version": "1.0.0",
        "parameters": 10000000,
        "size": "25MB"
    ]
    
    var testExportFilePath = "/test/export/path/file.json"
    var testExportedItemCount = 5
    
    // Çağrı izleme
    var initializeRuntimeCalled = false
    var loadModelCalled = false
    var performInferenceCalled = false
    var searchCalled = false
    var exportDataCalled = false
    
    // Mock metod implementasyonları
    override func initializeRuntime(config: M3TMConfig, completion: @escaping (Error?) -> Void) {
        initializeRuntimeCalled = true
        
        // Async çağrıyı simüle et
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            completion(self.shouldSucceedInitialize ? nil : M3TMError.internalError("Test Error"))
        }
    }
    
    override func loadModel(configDict: [String: Any]) throws {
        loadModelCalled = true
        
        if !shouldSucceedLoadModel {
            throw M3TMError.modelLoadFailed
        }
    }
    
    override func performInference(inputs: [[String: Any]], options: [String: Any], completion: @escaping ([Any]?, Error?) -> Void) {
        performInferenceCalled = true
        
        // Test sonuçları oluştur
        let results: [Any] = [
            ["probability": 0.9, "label": "test_label"],
            ["probability": 0.8, "label": "test_label2"]
        ]
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            completion(results, nil)
        }
    }
    
    override func generateSearchEmbeddings(inputs: [[String: Any]], completion: @escaping ([Any]?, Error?) -> Void) {
        // Test gömme vektörleri
        let embeddings: [[Float]] = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ]
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            completion(embeddings, nil)
        }
    }
    
    override func search(textQuery: String?, imageData: Data?, options: [String: Any], completion: @escaping ([Any]?, Error?) -> Void) {
        searchCalled = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            if self.shouldSucceedSearch {
                completion(self.testSearchResults, nil)
            } else {
                completion(nil, M3TMError.searchFailed)
            }
        }
    }
    
    override func exportData(filters: [String: Any], options: [String: Any], filename: String, completion: @escaping (String?, Int?, Error?) -> Void) {
        exportDataCalled = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            if self.shouldSucceedExport {
                completion(self.testExportFilePath, self.testExportedItemCount, nil)
            } else {
                completion(nil, nil, M3TMError.exportFailed)
            }
        }
    }
    
    // Diğer metodlar da benzer şekilde override edilebilir
} 