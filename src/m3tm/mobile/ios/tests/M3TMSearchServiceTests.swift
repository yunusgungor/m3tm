import XCTest
@testable import M3TM

class M3TMSearchServiceTests: XCTestCase {
    
    var bridgeMock: M3TMBridgeMock!
    var originalBridge: M3TMBridge!
    var searchService: M3TMSearchService!
    
    override func setUp() {
        super.setUp()
        // Test öncesi hazırlık
        originalBridge = M3TMBridge.shared
        bridgeMock = M3TMBridgeMock()
        M3TMBridge.shared = bridgeMock
        
        // Arama servisi oluşturma
        searchService = M3TMSearchService(indexDirectory: "test_index_directory")
    }
    
    override func tearDown() {
        // Test sonrası temizlik
        M3TMBridge.shared = originalBridge
        searchService = nil
        super.tearDown()
    }
    
    // MARK: - Başlatma Testleri
    
    func testInitializeSuccess() {
        // Given
        let expectation = XCTestExpectation(description: "Arama servisi başarıyla başlatılmalı")
        
        // When
        searchService.initialize { error in
            // Then
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            XCTAssertTrue(self.searchService.isInitialized, "Arama servisi başlatılmış olmalı")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - İndeksleme Testleri
    
    func testIndexItem() {
        // Given
        let initExpectation = XCTestExpectation(description: "Arama servisi başarıyla başlatılmalı")
        
        searchService.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "Öğe indeksleme tamamlanmalı")
            
            self.searchService.indexItem(
                itemId: "test_item_1", 
                textContent: "Bu bir test metnidir", 
                imageData: nil, 
                metadata: ["category": "test"]
            ) { error in
                // Then
                XCTAssertNil(error, "İndeksleme hatası olmamalı")
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    func testIndexItemWithoutInitialize() {
        // When & Then
        let expectation = XCTestExpectation(description: "İndeksleme hata vermeli")
        
        searchService.indexItem(
            itemId: "test_item_1", 
            textContent: "Bu bir test metnidir", 
            imageData: nil, 
            metadata: nil
        ) { error in
            XCTAssertNotNil(error, "Hata dönmeli")
            XCTAssertEqual(error as? M3TMError, M3TMError.searchIndexNotInitialized, "Başlatılmamış hata kodu dönmeli")
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 1.0)
    }
    
    // MARK: - Arama Testleri
    
    func testSearchByText() {
        // Given
        bridgeMock.shouldSucceedSearch = true
        let initExpectation = XCTestExpectation(description: "Arama servisi başarıyla başlatılmalı")
        
        searchService.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "Arama tamamlanmalı")
            let filters = M3TMSearchService.SearchFilters()
            
            self.searchService.search(textQuery: "test query", filters: filters, topK: 5) { results, error in
                // Then
                XCTAssertNil(error, "Arama hatası olmamalı")
                XCTAssertNotNil(results, "Sonuçlar nil olmamalı")
                XCTAssertTrue(self.bridgeMock.searchCalled, "search çağrılmalı")
                
                if let results = results {
                    XCTAssertEqual(results.count, 2, "2 sonuç dönmeli")
                    if results.count > 0 {
                        XCTAssertEqual(results[0].itemId, "test1", "İlk sonuç ID'si doğru olmalı")
                        XCTAssertEqual(results[0].score, 0.95, "İlk sonuç skoru doğru olmalı")
                        XCTAssertEqual(results[0].textContent, "Test sonucu 1", "İlk sonuç metni doğru olmalı")
                    }
                }
                
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    func testSearchFailure() {
        // Given
        bridgeMock.shouldSucceedSearch = false
        let initExpectation = XCTestExpectation(description: "Arama servisi başarıyla başlatılmalı")
        
        searchService.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "Arama hatası vermeli")
            let filters = M3TMSearchService.SearchFilters()
            
            self.searchService.search(textQuery: "test query", filters: filters, topK: 5) { results, error in
                // Then
                XCTAssertNotNil(error, "Arama hatası dönmeli")
                XCTAssertNil(results, "Sonuçlar nil olmalı")
                XCTAssertEqual(error as? M3TMError, M3TMError.searchFailed, "Arama hatası kodu doğru olmalı")
                XCTAssertTrue(self.bridgeMock.searchCalled, "search çağrılmalı")
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    func testSearchByImage() {
        // Given
        bridgeMock.shouldSucceedSearch = true
        let initExpectation = XCTestExpectation(description: "Arama servisi başarıyla başlatılmalı")
        
        searchService.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // Test görüntüsü oluştur
            let rect = CGRect(x: 0, y: 0, width: 100, height: 100)
            UIGraphicsBeginImageContext(rect.size)
            let context = UIGraphicsGetCurrentContext()!
            context.setFillColor(UIColor.red.cgColor)
            context.fill(rect)
            let testImage = UIGraphicsGetImageFromCurrentImageContext()!
            UIGraphicsEndImageContext()
            
            // When
            let expectation = XCTestExpectation(description: "Görüntü araması tamamlanmalı")
            let filters = M3TMSearchService.SearchFilters()
            
            self.searchService.search(imageQuery: testImage, filters: filters, topK: 5) { results, error in
                // Then
                XCTAssertNil(error, "Arama hatası olmamalı")
                XCTAssertNotNil(results, "Sonuçlar nil olmamalı")
                XCTAssertTrue(self.bridgeMock.searchCalled, "search çağrılmalı")
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    // MARK: - İndeks Yönetimi Testleri
    
    func testSaveIndex() {
        // Given
        let initExpectation = XCTestExpectation(description: "Arama servisi başarıyla başlatılmalı")
        
        searchService.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "İndeks kaydedilmeli")
            
            self.searchService.saveIndex { error in
                // Then
                XCTAssertNil(error, "İndeks kaydetme hatası olmamalı")
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    func testClearIndex() {
        // Given
        let initExpectation = XCTestExpectation(description: "Arama servisi başarıyla başlatılmalı")
        
        searchService.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "İndeks temizlenmeli")
            
            self.searchService.clearIndex { error in
                // Then
                XCTAssertNil(error, "İndeks temizleme hatası olmamalı")
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
    
    func testGetIndexInfo() {
        // Given
        let initExpectation = XCTestExpectation(description: "Arama servisi başarıyla başlatılmalı")
        
        searchService.initialize { error in
            XCTAssertNil(error, "Başlatma hatası olmamalı")
            initExpectation.fulfill()
            
            // When
            let expectation = XCTestExpectation(description: "İndeks bilgisi alınmalı")
            
            self.searchService.getIndexInfo { info, error in
                // Then
                XCTAssertNil(error, "İndeks bilgisi alma hatası olmamalı")
                XCTAssertNotNil(info, "İndeks bilgisi nil olmamalı")
                expectation.fulfill()
            }
            
            self.wait(for: [expectation], timeout: 1.0)
        }
        
        wait(for: [initExpectation], timeout: 1.0)
    }
} 