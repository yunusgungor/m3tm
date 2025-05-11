import XCTest
@testable import M3TM

class M3TMTests: XCTestCase {
    
    var bridgeMock: M3TMBridgeMock!
    var originalBridge: M3TMBridge!
    
    override func setUp() {
        super.setUp()
        // Test öncesi hazırlık
        originalBridge = M3TMBridge.shared
        bridgeMock = M3TMBridgeMock()
        M3TMBridge.shared = bridgeMock
    }
    
    override func tearDown() {
        // Test sonrası temizlik
        M3TMBridge.shared = originalBridge
        super.tearDown()
    }
    
    // MARK: - Başlatma Testleri
    
    func testInitializeSuccess() {
        // Given
        bridgeMock.shouldSucceedInitialize = true
        let config = M3TMConfig(modelPath: "test_model.pt", configPath: "test_config.json")
        let expectation = XCTestExpectation(description: "SDK başarıyla başlatılmalı")
        
        // When
        M3TM.shared.initialize(config: config) { error in
            // Then
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            XCTAssertTrue(self.bridgeMock.initializeRuntimeCalled, "Runtime başlatma çağrılmalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testInitializeFailure() {
        // Given
        bridgeMock.shouldSucceedInitialize = false
        let config = M3TMConfig(modelPath: "test_model.pt", configPath: "test_config.json")
        let expectation = XCTestExpectation(description: "SDK başlatılamamalı")
        
        // When
        M3TM.shared.initialize(config: config) { error in
            // Then
            XCTAssertNotNil(error, "Bir hata dönmeli")
            XCTAssertTrue(self.bridgeMock.initializeRuntimeCalled, "Runtime başlatma çağrılmalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testDoubleInitialization() {
        // Given
        bridgeMock.shouldSucceedInitialize = true
        let config = M3TMConfig(modelPath: "test_model.pt", configPath: "test_config.json")
        let expectation1 = XCTestExpectation(description: "İlk başlatma başarılı olmalı")
        let expectation2 = XCTestExpectation(description: "İkinci başlatma hata vermeli")
        
        // When - First initialization
        M3TM.shared.initialize(config: config) { error in
            XCTAssertNil(error, "İlk başlatmada hata olmamalı")
            expectation1.fulfill()
            
            // Then - Second initialization
            M3TM.shared.initialize(config: config) { error in
                XCTAssertNotNil(error, "İkinci başlatma hatası olmalı")
                XCTAssertEqual((error as? M3TMError), M3TMError.alreadyInitialized, "Hata türü alreadyInitialized olmalı")
                expectation2.fulfill()
            }
        }
        
        wait(for: [expectation1, expectation2], timeout: 1.0)
    }
    
    // MARK: - Yönetici Erişim Testleri
    
    func testGetModelManagerBeforeInitialize() {
        // Shutdown to ensure not initialized
        M3TM.shared.shutdown()
        
        // Then
        XCTAssertThrowsError(try M3TM.shared.getModelManager()) { error in
            XCTAssertEqual(error as? M3TMError, M3TMError.notInitialized)
        }
    }
    
    func testGetTrainingManagerBeforeInitialize() {
        // Shutdown to ensure not initialized
        M3TM.shared.shutdown()
        
        // Then
        XCTAssertThrowsError(try M3TM.shared.getTrainingManager()) { error in
            XCTAssertEqual(error as? M3TMError, M3TMError.notInitialized)
        }
    }
    
    func testGetSearchServiceBeforeInitialize() {
        // Shutdown to ensure not initialized
        M3TM.shared.shutdown()
        
        // Then
        XCTAssertThrowsError(try M3TM.shared.getSearchService()) { error in
            XCTAssertEqual(error as? M3TMError, M3TMError.notInitialized)
        }
    }
    
    func testGetDataExporterBeforeInitialize() {
        // Shutdown to ensure not initialized
        M3TM.shared.shutdown()
        
        // Then
        XCTAssertThrowsError(try M3TM.shared.getDataExporter()) { error in
            XCTAssertEqual(error as? M3TMError, M3TMError.notInitialized)
        }
    }
    
    func testGetManagersAfterInitialize() {
        // Given
        let initExpectation = XCTestExpectation(description: "SDK başarıyla başlatılmalı")
        let config = M3TMConfig(modelPath: "test_model.pt", configPath: "test_config.json")
        
        // When
        M3TM.shared.initialize(config: config) { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // Then
            do {
                let modelManager = try M3TM.shared.getModelManager()
                XCTAssertNotNil(modelManager, "Model yöneticisi nil olmamalı")
                
                let trainingManager = try M3TM.shared.getTrainingManager()
                XCTAssertNotNil(trainingManager, "Eğitim yöneticisi nil olmamalı")
                
                let searchService = try M3TM.shared.getSearchService()
                XCTAssertNotNil(searchService, "Arama servisi nil olmamalı")
                
                let dataExporter = try M3TM.shared.getDataExporter()
                XCTAssertNotNil(dataExporter, "Veri dışa aktarma servisi nil olmamalı")
            } catch {
                XCTFail("Yöneticilere erişirken hata oluştu: \(error)")
            }
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    // MARK: - Shutdown Testleri
    
    func testShutdown() {
        // Given
        let initExpectation = XCTestExpectation(description: "SDK başarıyla başlatılmalı")
        let config = M3TMConfig(modelPath: "test_model.pt", configPath: "test_config.json")
        
        // When
        M3TM.shared.initialize(config: config) { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            XCTAssertTrue(M3TM.shared.isInitialized, "SDK başlatılmış olmalı")
            initExpectation.fulfill()
            
            // Then
            M3TM.shared.shutdown()
            XCTAssertFalse(M3TM.shared.isInitialized, "SDK kapatılmış olmalı")
            
            // Verify can't access managers
            XCTAssertThrowsError(try M3TM.shared.getModelManager()) { error in
                XCTAssertEqual(error as? M3TMError, M3TMError.notInitialized)
            }
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    // MARK: - Versiyon Testleri
    
    func testVersion() {
        XCTAssertFalse(M3TM.shared.version.isEmpty, "Sürüm bilgisi boş olmamalı")
    }
} 