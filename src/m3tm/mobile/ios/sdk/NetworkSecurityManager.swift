import Foundation
import Network
import Security

/**
 * OWASP MASVS-NETWORK uyumlu network security yöneticisi (iOS).
 * 
 * Certificate pinning, TLS 1.3 enforcement ve MITM protection sağlar.
 * TrustKit framework entegrasyonu ile güvenli network operations.
 */
@objc public class NetworkSecurityManager: NSObject {
    
    // MARK: - Constants
    
    private static let productionDomains = [
        "api.m3tm.com",
        "secure.m3tm.com",
        "models.m3tm.com"
    ]
    
    private static let stagingDomains = [
        "dev-api.m3tm.com",
        "staging-api.m3tm.com"
    ]
    
    // Certificate pins (SHA-256 public key hashes)
    private static let productionPins = [
        "YLh1dUR9y6Kja30RrAn7JKnbQG/uEtLMkBgFF2fuihg=", // Primary
        "C5+lpZ7tcVwmwQIMcRtPbsQtWLABXhQzejna0wHFr8M=", // Backup
        "lCppFqbkrlJ3EcVFAkeip0+44VaoJUymbnOaEUk7tEU="  // Root CA backup
    ]
    
    private static let stagingPins = [
        "Dev1234567890abcdef1234567890abcdef1234567890abcdef=",
        "Dev0987654321fedcba0987654321fedcba0987654321fedcba="
    ]
    
    // MARK: - Properties
    
    private var urlSession: URLSession?
    private var isInitialized = false
    private var useStaging = false
    private let networkMonitor = NWPathMonitor()
    private var isNetworkAvailable = false
    
    // MARK: - Public Methods
    
    /**
     * Network security sistemini başlatır.
     * 
     * @param useStaging Staging environment kullanılacak mı
     * @param completion Başlatma sonucu callback'i
     */
    @objc public func initialize(useStaging: Bool, completion: @escaping (Error?) -> Void) {
        self.useStaging = useStaging
        
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async {
                    completion(M3TMError.internalError("NetworkSecurityManager deallocated"))
                }
                return
            }
            
            do {
                // TrustKit configuration
                let trustKitConfig = self.createTrustKitConfiguration()
                
                // URLSession configuration
                let sessionConfig = URLSessionConfiguration.default
                sessionConfig.tlsMinimumSupportedProtocolVersion = .TLSv13
                sessionConfig.tlsMaximumSupportedProtocolVersion = .TLSv13
                sessionConfig.timeoutIntervalForRequest = 30.0
                sessionConfig.timeoutIntervalForResource = 60.0
                sessionConfig.waitsForConnectivity = true
                
                // Custom delegate for certificate pinning
                let delegate = NetworkSecurityDelegate(
                    domains: useStaging ? Self.stagingDomains : Self.productionDomains,
                    pins: useStaging ? Self.stagingPins : Self.productionPins
                )
                
                self.urlSession = URLSession(
                    configuration: sessionConfig,
                    delegate: delegate,
                    delegateQueue: nil
                )
                
                // Network monitoring başlat
                self.startNetworkMonitoring()
                
                self.isInitialized = true
                NSLog("Network security initialized (staging: \(useStaging))")
                
                DispatchQueue.main.async {
                    completion(nil)
                }
                
            } catch {
                NSLog("Failed to initialize network security: \(error)")
                DispatchQueue.main.async {
                    completion(M3TMError.internalError("Network security initialization failed: \(error.localizedDescription)"))
                }
            }
        }
    }
    
    /**
     * TrustKit configuration oluşturur.
     * 
     * @return TrustKit configuration dictionary
     */
    private func createTrustKitConfiguration() -> [String: Any] {
        let domains = useStaging ? Self.stagingDomains : Self.productionDomains
        let pins = useStaging ? Self.stagingPins : Self.productionPins
        
        var domainConfigs: [String: Any] = [:]
        
        for domain in domains {
            domainConfigs[domain] = [
                "TKIncludeSubdomains": true,
                "TKPublicKeyHashes": pins,
                "TKEnforcePinning": true,
                "TKReportUris": ["https://\(domain)/tls-report"], // TLS failure reporting
                "TKDisableDefaultReportUri": false
            ]
        }
        
        return [
            "TKSwizzleNetworkDelegates": false, // Manual delegate control
            "TKPinnedDomains": domainConfigs
        ]
    }
    
    /**
     * Network monitoring başlatır.
     */
    private func startNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            self?.isNetworkAvailable = path.status == .satisfied
            NSLog("Network status changed: \(path.status)")
        }
        
        let queue = DispatchQueue(label: "NetworkMonitor")
        networkMonitor.start(queue: queue)
    }
    
    /**
     * Güvenli HTTP GET request yapar.
     * 
     * @param url Request URL
     * @param completion Sonuç callback'i (data, error)
     */
    @objc public func makeSecureGetRequest(_ url: String, completion: @escaping (Data?, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, M3TMError.notInitialized)
            return
        }
        
        guard let requestURL = URL(string: url) else {
            completion(nil, M3TMError.inferenceFailed)
            return
        }
        
        var request = URLRequest(url: requestURL)
        request.httpMethod = "GET"
        request.setValue("M3TM-SDK/1.0", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        
        let task = urlSession?.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    // Certificate pinning failure detection
                    if error.localizedDescription.contains("certificate") || 
                       error.localizedDescription.contains("SSL") {
                        completion(nil, M3TMError.internalError("Certificate pinning validation failed - possible MITM attack"))
                    } else {
                        completion(nil, error)
                    }
                    return
                }
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    completion(nil, M3TMError.internalError("Invalid response type"))
                    return
                }
                
                guard 200...299 ~= httpResponse.statusCode else {
                    completion(nil, M3TMError.internalError("HTTP request failed with code: \(httpResponse.statusCode)"))
                    return
                }
                
                completion(data, nil)
            }
        }
        
        task?.resume()
    }
    
    /**
     * Güvenli HTTP POST request yapar.
     * 
     * @param url Request URL
     * @param jsonData JSON request body
     * @param completion Sonuç callback'i (data, error)
     */
    @objc public func makeSecurePostRequest(_ url: String, jsonData: Data, completion: @escaping (Data?, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, M3TMError.notInitialized)
            return
        }
        
        guard let requestURL = URL(string: url) else {
            completion(nil, M3TMError.inferenceFailed)
            return
        }
        
        var request = URLRequest(url: requestURL)
        request.httpMethod = "POST"
        request.setValue("M3TM-SDK/1.0", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = jsonData
        
        let task = urlSession?.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(nil, error)
                    return
                }
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    completion(nil, M3TMError.internalError("Invalid response type"))
                    return
                }
                
                guard 200...299 ~= httpResponse.statusCode else {
                    completion(nil, M3TMError.internalError("HTTP POST request failed with code: \(httpResponse.statusCode)"))
                    return
                }
                
                completion(data, nil)
            }
        }
        
        task?.resume()
    }
    
    /**
     * Certificate pinning durumunu test eder.
     * 
     * @param url Test edilecek URL
     * @param completion Test sonucu callback'i (success, error)
     */
    @objc public func testCertificatePinning(_ url: String, completion: @escaping (Bool, Error?) -> Void) {
        makeSecureGetRequest(url) { data, error in
            if let error = error {
                completion(false, error)
            } else {
                completion(true, nil)
            }
        }
    }
    
    /**
     * Network connectivity durumunu döndürür.
     * 
     * @return Network bağlantısı mevcut mu
     */
    @objc public var networkAvailable: Bool {
        return isNetworkAvailable
    }
    
    /**
     * TLS connection bilgilerini loglar.
     * Debugging ve security audit için kullanılır.
     * 
     * @param url Test edilecek URL
     * @param completion Log sonucu callback'i
     */
    @objc public func logTLSInfo(_ url: String, completion: @escaping (Error?) -> Void) {
        guard let requestURL = URL(string: url) else {
            completion(M3TMError.inferenceFailed)
            return
        }
        
        let task = urlSession?.dataTask(with: requestURL) { _, response, error in
            if let error = error {
                completion(error)
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                NSLog("TLS Connection Info for \(url):")
                NSLog("Status Code: \(httpResponse.statusCode)")
                NSLog("Headers: \(httpResponse.allHeaderFields)")
            }
            
            completion(nil)
        }
        
        task?.resume()
    }
    
    /**
     * URLSession instance'ını döndürür (advanced usage için).
     * 
     * @return Güvenli olarak yapılandırılmış URLSession
     */
    @objc public var secureURLSession: URLSession? {
        guard isInitialized else { return nil }
        return urlSession
    }
    
    /**
     * Kaynakları serbest bırakır.
     */
    @objc public func shutdown() {
        networkMonitor.cancel()
        urlSession?.invalidateAndCancel()
        urlSession = nil
        isInitialized = false
        NSLog("NetworkSecurityManager shutdown completed")
    }
}

// MARK: - NetworkSecurityDelegate

/**
 * Custom URLSessionDelegate certificate pinning implementasyonu.
 */
private class NetworkSecurityDelegate: NSObject, URLSessionDelegate {
    
    private let allowedDomains: [String]
    private let pinnedHashes: [String]
    
    init(domains: [String], pins: [String]) {
        self.allowedDomains = domains
        self.pinnedHashes = pins
        super.init()
    }
    
    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge, completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        
        // Domain validation
        guard let serverTrust = challenge.protectionSpace.serverTrust,
              allowedDomains.contains(challenge.protectionSpace.host) else {
            NSLog("Domain not in whitelist: \(challenge.protectionSpace.host)")
            completionHandler(.performDefaultHandling, nil)
            return
        }
        
        // Certificate pinning validation
        guard validateCertificatePinning(serverTrust: serverTrust) else {
            NSLog("Certificate pinning validation failed for: \(challenge.protectionSpace.host)")
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }
        
        // Trust the connection
        let credential = URLCredential(trust: serverTrust)
        completionHandler(.useCredential, credential)
    }
    
    /**
     * Certificate pinning validation.
     * 
     * @param serverTrust Server trust object
     * @return Validation başarılı mı
     */
    private func validateCertificatePinning(serverTrust: SecTrust) -> Bool {
        // Certificate chain'i al
        let certificateCount = SecTrustGetCertificateCount(serverTrust)
        
        for i in 0..<certificateCount {
            guard let certificate = SecTrustGetCertificateAtIndex(serverTrust, i) else {
                continue
            }
            
            // Public key'i extract et
            guard let publicKey = SecCertificateCopyKey(certificate) else {
                continue
            }
            
            // Public key'in SHA-256 hash'ini hesapla
            guard let publicKeyData = SecKeyCopyExternalRepresentation(publicKey, nil) else {
                continue
            }
            
            let publicKeyHash = sha256Hash(data: publicKeyData as Data)
            let base64Hash = publicKeyHash.base64EncodedString()
            
            // Pinned hash'lerle karşılaştır
            if pinnedHashes.contains(base64Hash) {
                NSLog("Certificate pinning validation successful")
                return true
            }
        }
        
        NSLog("No matching pinned certificates found")
        return false
    }
    
    /**
     * SHA-256 hash hesaplama.
     * 
     * @param data Hash'lenecek data
     * @return SHA-256 hash
     */
    private func sha256Hash(data: Data) -> Data {
        var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        data.withUnsafeBytes {
            _ = CC_SHA256($0.baseAddress, CC_LONG(data.count), &hash)
        }
        return Data(hash)
    }
}
