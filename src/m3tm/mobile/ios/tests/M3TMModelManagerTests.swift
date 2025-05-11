import XCTest
@testable import M3TM

class M3TMModelManagerTests: XCTestCase {
    
    var bridgeMock: M3TMBridgeMock!
    var originalBridge: M3TMBridge!
    var modelManager: M3TMModelManager!
    
    override func setUp() {
        super.setUp()
        // Test öncesi hazırlık
        originalBridge = M3TMBridge.shared
        bridgeMock = M3TMBridgeMock()
        M3TMBridge.shared = bridgeMock
        
        // SDK başlatma ve model yöneticisi oluşturma
        let config = M3TMConfig(modelPath: "test_model.pt", configPath: "test_config.json")
        modelManager = M3TMModelManager(config: config)
    }
    
    override func tearDown() {
        // Test sonrası temizlik
        M3TMBridge.shared = originalBridge
        modelManager = nil
        super.tearDown()
    }
    
    // MARK: - Model Yükleme Testleri
    
    func testLoadModelSuccess() {
        // Given
        bridgeMock.shouldSucceedLoadModel = true
        
        // When & Then
        XCTAssertNoThrow(try modelManager.loadModel(), "Model yükleme hatası vermemeli")
        XCTAssertTrue(bridgeMock.loadModelCalled, "loadModel çağrılmalı")
        XCTAssertTrue(modelManager.isModelLoaded(), "Model yüklenmiş olmalı")
    }
    
    func testLoadModelFailure() {
        // Given
        bridgeMock.shouldSucceedLoadModel = false
        
        // When & Then
        XCTAssertThrowsError(try modelManager.loadModel()) { error in
            XCTAssertEqual(error as? M3TMError, M3TMError.modelLoadFailed, "Model yükleme hatası vermeli")
        }
        XCTAssertTrue(bridgeMock.loadModelCalled, "loadModel çağrılmalı")
        XCTAssertFalse(modelManager.isModelLoaded(), "Model yüklenmemiş olmalı")
    }
    
    func testUnloadModel() {
        // Given - başarılı model yükleme
        bridgeMock.shouldSucceedLoadModel = true
        try? modelManager.loadModel()
        XCTAssertTrue(modelManager.isModelLoaded(), "Model yüklenmiş olmalı")
        
        // Unload için beklenti oluştur
        let expectation = XCTestExpectation(description: "Model boşaltma tamamlanmalı")
        
        // When
        modelManager.unloadModel { error in
            // Then
            XCTAssertNil(error, "Model boşaltma hatası olmamalı")
            XCTAssertFalse(self.modelManager.isModelLoaded(), "Model boşaltılmış olmalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Çıkarım Testleri
    
    func testInference() {
        // Given - başarılı model yükleme
        bridgeMock.shouldSucceedLoadModel = true
        try? modelManager.loadModel()
        XCTAssertTrue(modelManager.isModelLoaded(), "Model yüklenmiş olmalı")
        
        let inputData = M3TMModelManager.InputData(text: "Test giriş metni")
        let options = M3TMModelManager.InferenceOptions()
        let expectation = XCTestExpectation(description: "Çıkarım tamamlanmalı")
        
        // When
        modelManager.inference(inputs: [inputData], options: options) { results, error in
            // Then
            XCTAssertNil(error, "Çıkarım hatası olmamalı")
            XCTAssertNotNil(results, "Sonuç nil olmamalı")
            XCTAssertTrue(self.bridgeMock.performInferenceCalled, "performInference çağrılmalı")
            
            if let results = results as? [[String: Any]] {
                XCTAssertEqual(results.count, 2, "2 sonuç dönmeli")
                if results.count > 0 {
                    XCTAssertNotNil(results[0]["probability"], "Sonuçta probability alanı olmalı")
                    XCTAssertNotNil(results[0]["label"], "Sonuçta label alanı olmalı")
                }
            } else {
                XCTFail("Sonuçlar [[String: Any]] formatında olmalı")
            }
            
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testInferenceWithoutModelLoad() {
        // Given - Model yüklenmemiş olmalı
        XCTAssertFalse(modelManager.isModelLoaded(), "Model yüklenmemiş olmalı")
        
        let inputData = M3TMModelManager.InputData(text: "Test giriş metni")
        let options = M3TMModelManager.InferenceOptions()
        let expectation = XCTestExpectation(description: "Çıkarım hata vermeli")
        
        // When
        modelManager.inference(inputs: [inputData], options: options) { results, error in
            // Then
            XCTAssertNotNil(error, "Çıkarım hatası dönmeli")
            XCTAssertEqual(error as? M3TMError, M3TMError.modelNotLoaded, "Model yüklenmemiş hatası dönmeli")
            XCTAssertNil(results, "Sonuç nil olmalı")
            XCTAssertFalse(self.bridgeMock.performInferenceCalled, "performInference çağrılmamalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testImageInference() {
        // Given - başarılı model yükleme
        bridgeMock.shouldSucceedLoadModel = true
        try? modelManager.loadModel()
        
        // Test görüntüsü oluştur
        let rect = CGRect(x: 0, y: 0, width: 100, height: 100)
        UIGraphicsBeginImageContext(rect.size)
        let context = UIGraphicsGetCurrentContext()!
        context.setFillColor(UIColor.red.cgColor)
        context.fill(rect)
        let testImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        let inputData = M3TMModelManager.InputData(image: testImage)
        let options = M3TMModelManager.InferenceOptions()
        let expectation = XCTestExpectation(description: "Görüntü çıkarımı tamamlanmalı")
        
        // When
        modelManager.inference(inputs: [inputData], options: options) { results, error in
            // Then
            XCTAssertNil(error, "Çıkarım hatası olmamalı")
            XCTAssertNotNil(results, "Sonuç nil olmamalı")
            XCTAssertTrue(self.bridgeMock.performInferenceCalled, "performInference çağrılmalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Model Bilgisi Testleri
    
    func testGetModelInfo() {
        // Given - başarılı model yükleme
        bridgeMock.shouldSucceedLoadModel = true
        try? modelManager.loadModel()
        
        let expectation = XCTestExpectation(description: "Model bilgisi alınmalı")
        
        // When
        modelManager.getModelInfo { info, error in
            // Then
            XCTAssertNil(error, "Model bilgisi alma hatası olmamalı")
            XCTAssertNotNil(info, "Model bilgisi nil olmamalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
} 