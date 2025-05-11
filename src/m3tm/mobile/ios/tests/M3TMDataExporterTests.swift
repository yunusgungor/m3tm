import XCTest
@testable import M3TM

class M3TMDataExporterTests: XCTestCase {
    
    var bridgeMock: M3TMBridgeMock!
    var originalBridge: M3TMBridge!
    var dataExporter: M3TMDataExporter!
    
    override func setUp() {
        super.setUp()
        // Test öncesi hazırlık
        originalBridge = M3TMBridge.shared
        bridgeMock = M3TMBridgeMock()
        M3TMBridge.shared = bridgeMock
        
        // Veri dışa aktarma servisi oluşturma
        dataExporter = M3TMDataExporter(exportDirectory: "test_export_directory")
    }
    
    override func tearDown() {
        // Test sonrası temizlik
        M3TMBridge.shared = originalBridge
        dataExporter = nil
        super.tearDown()
    }
    
    // MARK: - Başlatma Testleri
    
    func testInitializeSuccess() {
        // Given
        let expectation = XCTestExpectation(description: "Dışa aktarma servisi başarıyla başlatılmalı")
        
        // When
        dataExporter.initialize { error in
            // Then
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            XCTAssertTrue(self.dataExporter.isExportInitialized(), "Dışa aktarma servisi başlatılmış olmalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Dışa Aktarma Testleri
    
    func testExportDataSuccess() {
        // Given
        bridgeMock.shouldSucceedExport = true
        let initExpectation = XCTestExpectation(description: "Dışa aktarma servisi başarıyla başlatılmalı")
        
        dataExporter.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "Veri dışa aktarma tamamlanmalı")
            
            let filters = M3TMDataExporter.ExportFilters()
            filters.contentTypes = ["text", "image"]
            filters.dateStart = Date(timeIntervalSince1970: 1640995200) // 2022-01-01
            
            let options = M3TMDataExporter.ExportOptions()
            options.format = .json
            options.includeImages = true
            
            self.dataExporter.exportData(filters: filters, options: options) { fileURL, itemCount, error in
                // Then
                XCTAssertNil(error, "Dışa aktarma hatası olmamalı")
                XCTAssertNotNil(fileURL, "Dosya URL'si nil olmamalı")
                XCTAssertEqual(itemCount, 5, "Öğe sayısı doğru olmalı")
                XCTAssertTrue(self.bridgeMock.exportDataCalled, "exportData çağrılmalı")
                
                if let fileURL = fileURL {
                    XCTAssertEqual(fileURL.path, self.bridgeMock.testExportFilePath, "Dosya yolu doğru olmalı")
                }
                
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    func testExportDataFailure() {
        // Given
        bridgeMock.shouldSucceedExport = false
        let initExpectation = XCTestExpectation(description: "Dışa aktarma servisi başarıyla başlatılmalı")
        
        dataExporter.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "Veri dışa aktarma hatası vermeli")
            
            let filters = M3TMDataExporter.ExportFilters()
            let options = M3TMDataExporter.ExportOptions()
            
            self.dataExporter.exportData(filters: filters, options: options) { fileURL, itemCount, error in
                // Then
                XCTAssertNotNil(error, "Dışa aktarma hatası dönmeli")
                XCTAssertNil(fileURL, "Dosya URL'si nil olmalı")
                XCTAssertEqual(itemCount, 0, "Öğe sayısı 0 olmalı")
                XCTAssertEqual(error as? M3TMError, M3TMError.exportFailed, "Hata kodu doğru olmalı")
                XCTAssertTrue(self.bridgeMock.exportDataCalled, "exportData çağrılmalı")
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    func testExportDataWithoutInitialize() {
        // Given
        let filters = M3TMDataExporter.ExportFilters()
        let options = M3TMDataExporter.ExportOptions()
        let expectation = XCTestExpectation(description: "Dışa aktarma hatası vermeli")
        
        // When
        dataExporter.exportData(filters: filters, options: options) { fileURL, itemCount, error in
            // Then
            XCTAssertNotNil(error, "Hata dönmeli")
            XCTAssertNil(fileURL, "Dosya URL'si nil olmalı")
            XCTAssertEqual(itemCount, 0, "Öğe sayısı 0 olmalı")
            XCTAssertEqual(error as? M3TMError, M3TMError.notInitialized, "Başlatılmamış hata kodu dönmeli")
            XCTAssertFalse(self.bridgeMock.exportDataCalled, "exportData çağrılmamalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    func testExportItems() {
        // Given
        bridgeMock.shouldSucceedExport = true
        let initExpectation = XCTestExpectation(description: "Dışa aktarma servisi başarıyla başlatılmalı")
        
        dataExporter.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "Öğe dışa aktarma tamamlanmalı")
            
            let itemIds = ["item1", "item2", "item3"]
            let options = M3TMDataExporter.ExportOptions()
            
            self.dataExporter.exportItems(itemIds: itemIds, options: options) { fileURL, itemCount, error in
                // Then
                XCTAssertNil(error, "Dışa aktarma hatası olmamalı")
                XCTAssertNotNil(fileURL, "Dosya URL'si nil olmamalı")
                XCTAssertEqual(itemCount, 5, "Öğe sayısı doğru olmalı")
                XCTAssertTrue(self.bridgeMock.exportDataCalled, "exportData çağrılmalı")
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    // MARK: - Dışa Aktarılan Dosya Yönetimi Testleri
    
    func testGetExportedFiles() {
        // Given
        // FileManager mock'u gerekir, burada örnek olarak basit bir varsayım yapıyoruz
        let initExpectation = XCTestExpectation(description: "Dışa aktarma servisi başarıyla başlatılmalı")
        
        dataExporter.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // Burada FileManager'ı mock etmek gerekiyor, gerçek testlerde bu yapılmalıdır
            // Bu test sadece API çağrısının yapılıp yapılmadığını kontrol ediyor
            let expectation = XCTestExpectation(description: "Dışa aktarılan dosyalar listelenmeli")
            
            self.dataExporter.getExportedFiles(maxCount: 5) { fileURLs, error in
                // Bu noktada gerçek implementasyon farklı davranabilir
                // Burada sadece API çağrısının çalışıp çalışmadığını test ediyoruz
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    func testDeleteExportedFileAndClearExportedFiles() {
        // Given
        let initExpectation = XCTestExpectation(description: "Dışa aktarma servisi başarıyla başlatılmalı")
        
        dataExporter.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // Burada FileManager'ı mock etmek gerekiyor, gerçek testlerde bu yapılmalıdır
            // Bu test sadece API çağrısının yapılıp yapılmadığını kontrol ediyor
            
            // Dosya silme testi
            let deleteExpectation = XCTestExpectation(description: "Dosya silinmeli")
            let testFileURL = URL(fileURLWithPath: "/test/file/path.json")
            
            self.dataExporter.deleteExportedFile(fileURL: testFileURL) { error in
                // Bu noktada gerçek implementasyon farklı davranabilir
                deleteExpectation.fulfill()
            }
            
            // Tüm dosyaları temizleme testi
            let clearExpectation = XCTestExpectation(description: "Tüm dosyalar temizlenmeli")
            
            self.dataExporter.clearExportedFiles { error in
                // Bu noktada gerçek implementasyon farklı davranabilir
                clearExpectation.fulfill()
            }
            
            self.wait(for: [deleteExpectation, clearExpectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    // MARK: - Filtre ve Seçenek Testleri
    
    func testExportFilters() {
        // Given
        let filters = M3TMDataExporter.ExportFilters()
        filters.contentTypes = ["text", "image"]
        filters.dateStart = Date(timeIntervalSince1970: 1640995200) // 2022-01-01
        filters.dateEnd = Date(timeIntervalSince1970: 1672531200) // 2023-01-01
        filters.tags = ["tag1", "tag2"]
        filters.itemIds = ["item1", "item2"]
        filters.customFilters = ["custom": true]
        
        // When
        let dict = filters.asDictionary()
        
        // Then
        XCTAssertEqual(dict["contentTypes"] as? [String], ["text", "image"])
        XCTAssertEqual(dict["dateStart"] as? Int, 1640995200)
        XCTAssertEqual(dict["dateEnd"] as? Int, 1672531200)
        XCTAssertEqual(dict["tags"] as? [String], ["tag1", "tag2"])
        XCTAssertEqual(dict["itemIds"] as? [String], ["item1", "item2"])
        XCTAssertEqual((dict["customFilters"] as? [String: Bool])?["custom"], true)
    }
    
    func testExportOptions() {
        // Given
        let options = M3TMDataExporter.ExportOptions()
        options.format = .csv
        options.includeRawData = true
        options.includeImages = false
        options.imageQuality = 0.75
        options.imageResizeFactor = 0.5
        options.includeMetadata = true
        options.customOptions = ["custom": true]
        
        // When
        let dict = options.asDictionary()
        
        // Then
        XCTAssertEqual(dict["format"] as? Int, M3TMDataExporter.ExportFormat.csv.rawValue)
        XCTAssertEqual(dict["includeRawData"] as? Bool, true)
        XCTAssertEqual(dict["includeImages"] as? Bool, false)
        XCTAssertEqual(dict["imageQuality"] as? Float, 0.75)
        XCTAssertEqual(dict["imageResizeFactor"] as? Float, 0.5)
        XCTAssertEqual(dict["includeMetadata"] as? Bool, true)
        XCTAssertEqual((dict["customOptions"] as? [String: Bool])?["custom"], true)
    }
} 