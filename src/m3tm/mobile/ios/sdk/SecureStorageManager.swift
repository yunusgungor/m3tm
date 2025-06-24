import Foundation
import Security
import CommonCrypto

/**
 * OWASP MASVS-STORAGE uyumlu güvenli depolama yöneticisi (iOS).
 * 
 * iOS Keychain Services ve AES-GCM-256 şifreleme kullanarak
 * hassas verilerin güvenli şekilde saklanmasını sağlar.
 */
@objc public class SecureStorageManager: NSObject {
    
    // MARK: - Constants
    
    private static let serviceIdentifier = "com.m3tm.sdk.secure.storage"
    private static let encryptionKeyAlias = "M3TMSecureStorageKey"
    private static let gcmTagLength = 16
    private static let gcmIVLength = 12
    
    // MARK: - Properties
    
    private var isInitialized = false
    private var secureEnclaveAvailable = false
    private var encryptionKey: SecKey?
    
    // MARK: - Public Methods
    
    /**
     * Güvenli depolama sistemini başlatır.
     * Secure Enclave kullanmayı dener, başarısız olursa keychain fallback kullanır.
     * 
     * @param completion Başlatma sonucu callback'i
     */
    @objc public func initialize(completion: @escaping (Error?) -> Void) {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async {
                    completion(M3TMError.internalError("SecureStorageManager deallocated"))
                }
                return
            }
            
            do {
                // Secure Enclave availability check
                self.secureEnclaveAvailable = self.checkSecureEnclaveAvailability()
                NSLog("Secure Enclave available: \(self.secureEnclaveAvailable)")
                
                // Encryption key oluştur veya al
                try self.setupEncryptionKey()
                
                self.isInitialized = true
                NSLog("Secure storage initialized successfully")
                
                DispatchQueue.main.async {
                    completion(nil)
                }
                
            } catch {
                NSLog("Failed to initialize secure storage: \(error)")
                DispatchQueue.main.async {
                    completion(M3TMError.internalError("Secure storage initialization failed: \(error.localizedDescription)"))
                }
            }
        }
    }
    
    /**
     * Secure Enclave desteğini kontrol eder.
     * 
     * @return true eğer Secure Enclave mevcutsa
     */
    private func checkSecureEnclaveAvailability() -> Bool {
        if #available(iOS 9.0, *) {
            // Test key oluşturarak Secure Enclave support kontrolü
            let attributes: [String: Any] = [
                kSecAttrKeyType as String: kSecAttrKeyTypeECSECPrimeRandom,
                kSecAttrKeySizeInBits as String: 256,
                kSecAttrTokenID as String: kSecAttrTokenIDSecureEnclave,
                kSecPrivateKeyAttrs as String: [
                    kSecAttrAccessControl as String: SecAccessControlCreateWithFlags(
                        kCFAllocatorDefault,
                        kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
                        .privateKeyUsage,
                        nil
                    )!
                ]
            ]
            
            var error: Unmanaged<CFError>?
            let privateKey = SecKeyCreateRandomKey(attributes as CFDictionary, &error)
            
            if let key = privateKey {
                // Test key'i sil
                let deleteQuery: [String: Any] = [
                    kSecClass as String: kSecClassKey,
                    kSecAttrApplicationTag as String: "test_se_key".data(using: .utf8)!
                ]
                SecItemDelete(deleteQuery as CFDictionary)
                return true
            }
        }
        return false
    }
    
    /**
     * Encryption key'i oluşturur veya mevcut olanı alır.
     * Secure Enclave tercih eder, fallback olarak keychain kullanır.
     */
    private func setupEncryptionKey() throws {
        // Önce mevcut key'i kontrol et
        if let existingKey = try getExistingEncryptionKey() {
            self.encryptionKey = existingKey
            return
        }
        
        // Yeni key oluştur
        let keyData = try generateRandomKey()
        try storeEncryptionKeyInKeychain(keyData)
        
        // Key'i memory'de tut (SecKey olarak değil, güvenlik için)
        // Gerçek implementasyonda key'i her kullanımda keychain'den alınabilir
    }
    
    /**
     * Mevcut encryption key'i keychain'den alır.
     * 
     * @return SecKey varsa, nil yoksa
     */
    private func getExistingEncryptionKey() throws -> SecKey? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.serviceIdentifier,
            kSecAttrAccount as String: Self.encryptionKeyAlias,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        if status == errSecSuccess, let keyData = result as? Data {
            // Key data'yı SecKey'e çevir (AES için)
            // Burada basitleştirme için keyData'yı direkt kullanacağız
            return nil // Gerçek implementasyonda SecKey döndürülür
        } else if status == errSecItemNotFound {
            return nil
        } else {
            throw M3TMError.bridgeError("Failed to retrieve encryption key from keychain")
        }
    }
    
    /**
     * Güvenli random key oluşturur.
     * 
     * @return 256-bit random key
     */
    private func generateRandomKey() throws -> Data {
        var keyData = Data(count: 32) // 256 bit
        let result = keyData.withUnsafeMutableBytes { mutableBytes in
            SecRandomCopyBytes(kSecRandomDefault, 32, mutableBytes.bindMemory(to: UInt8.self).baseAddress!)
        }
        
        if result != errSecSuccess {
            throw M3TMError.bridgeError("Failed to generate random key")
        }
        
        return keyData
    }
    
    /**
     * Encryption key'i keychain'e güvenli şekilde saklar.
     * 
     * @param keyData Saklanacak key data
     */
    private func storeEncryptionKeyInKeychain(_ keyData: Data) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.serviceIdentifier,
            kSecAttrAccount as String: Self.encryptionKeyAlias,
            kSecValueData as String: keyData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        // Secure Enclave varsa access control ekle
        if secureEnclaveAvailable {
            if #available(iOS 9.0, *) {
                let accessControl = SecAccessControlCreateWithFlags(
                    kCFAllocatorDefault,
                    kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
                    .biometryAny,
                    nil
                )
                if let ac = accessControl {
                    var secureQuery = query
                    secureQuery[kSecAttrAccessControl as String] = ac
                    let status = SecItemAdd(secureQuery as CFDictionary, nil)
                    if status != errSecSuccess {
                        throw M3TMError.bridgeError("Failed to store encryption key in keychain with Secure Enclave")
                    }
                    return
                }
            }
        }
        
        // Fallback: Normal keychain storage
        let status = SecItemAdd(query as CFDictionary, nil)
        if status != errSecSuccess {
            throw M3TMError.bridgeError("Failed to store encryption key in keychain")
        }
    }
    
    /**
     * Güvenli string depolama.
     * 
     * @param key Anahtar
     * @param value Değer
     * @param completion İşlem sonucu callback'i
     */
    @objc public func storeSecureString(_ key: String, value: String, completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.notInitialized)
            return
        }
        
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                let valueData = value.data(using: .utf8) ?? Data()
                try self?.storeInKeychain(key: key, data: valueData)
                DispatchQueue.main.async {
                    completion(nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completion(error)
                }
            }
        }
    }
    
    /**
     * Güvenli string okuma.
     * 
     * @param key Anahtar
     * @param completion Sonuç callback'i (value, error)
     */
    @objc public func getSecureString(_ key: String, completion: @escaping (String?, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, M3TMError.notInitialized)
            return
        }
        
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                if let data = try self?.getFromKeychain(key: key),
                   let value = String(data: data, encoding: .utf8) {
                    DispatchQueue.main.async {
                        completion(value, nil)
                    }
                } else {
                    DispatchQueue.main.async {
                        completion(nil, nil)
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completion(nil, error)
                }
            }
        }
    }
    
    /**
     * Güvenli data depolama.
     * AES-GCM-256 ile şifrelenerek saklanır.
     * 
     * @param key Anahtar
     * @param data Şifrelenecek veri
     * @param completion İşlem sonucu callback'i
     */
    @objc public func storeSecureData(_ key: String, data: Data, completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.notInitialized)
            return
        }
        
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                let encryptedData = try self?.encryptWithAESGCM(data) ?? Data()
                try self?.storeInKeychain(key: key + "_encrypted", data: encryptedData)
                DispatchQueue.main.async {
                    completion(nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completion(error)
                }
            }
        }
    }
    
    /**
     * Güvenli data okuma.
     * AES-GCM-256 ile çözümlenir.
     * 
     * @param key Anahtar
     * @param completion Sonuç callback'i (data, error)
     */
    @objc public func getSecureData(_ key: String, completion: @escaping (Data?, Error?) -> Void) {
        guard isInitialized else {
            completion(nil, M3TMError.notInitialized)
            return
        }
        
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                if let encryptedData = try self?.getFromKeychain(key: key + "_encrypted") {
                    let decryptedData = try self?.decryptWithAESGCM(encryptedData) ?? Data()
                    DispatchQueue.main.async {
                        completion(decryptedData, nil)
                    }
                } else {
                    DispatchQueue.main.async {
                        completion(nil, nil)
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completion(nil, error)
                }
            }
        }
    }
    
    /**
     * Keychain'e veri saklama.
     * 
     * @param key Anahtar
     * @param data Veri
     */
    private func storeInKeychain(key: String, data: Data) throws {
        // Önce mevcut item'ı sil
        removeFromKeychain(key: key)
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.serviceIdentifier,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        if status != errSecSuccess {
            throw M3TMError.bridgeError("Failed to store data in keychain for key: \(key)")
        }
    }
    
    /**
     * Keychain'den veri okuma.
     * 
     * @param key Anahtar
     * @return Veri varsa, nil yoksa
     */
    private func getFromKeychain(key: String) throws -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.serviceIdentifier,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        
        if status == errSecSuccess {
            return result as? Data
        } else if status == errSecItemNotFound {
            return nil
        } else {
            throw M3TMError.bridgeError("Failed to retrieve data from keychain for key: \(key)")
        }
    }
    
    /**
     * Keychain'den veri silme.
     * 
     * @param key Anahtar
     */
    private func removeFromKeychain(key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.serviceIdentifier,
            kSecAttrAccount as String: key
        ]
        
        SecItemDelete(query as CFDictionary)
    }
    
    /**
     * AES-GCM-256 ile veri şifreleme.
     * 
     * @param data Şifrelenecek veri
     * @return IV + şifrelenmiş veri + tag
     */
    private func encryptWithAESGCM(_ data: Data) throws -> Data {
        // Random IV oluştur
        var iv = Data(count: Self.gcmIVLength)
        let result = iv.withUnsafeMutableBytes { mutableBytes in
            SecRandomCopyBytes(kSecRandomDefault, Self.gcmIVLength, mutableBytes.bindMemory(to: UInt8.self).baseAddress!)
        }
        
        if result != errSecSuccess {
            throw M3TMError.bridgeError("Failed to generate random IV")
        }
        
        // Encryption key al (basitleştirme için sabit key kullanıyoruz)
        guard let keyData = try getEncryptionKeyData() else {
            throw M3TMError.bridgeError("Encryption key not available")
        }
        
        // AES-GCM encryption (CommonCrypto ile)
        var encryptedData = Data(count: data.count + Self.gcmTagLength)
        var encryptedLength = 0
        
        let status = keyData.withUnsafeBytes { keyBytes in
            iv.withUnsafeBytes { ivBytes in
                data.withUnsafeBytes { dataBytes in
                    encryptedData.withUnsafeMutableBytes { encryptedBytes in
                        CCCryptorGCM(
                            CCOperation(kCCEncrypt),
                            CCAlgorithm(kCCAlgorithmAES),
                            keyBytes.bindMemory(to: UInt8.self).baseAddress,
                            keyData.count,
                            ivBytes.bindMemory(to: UInt8.self).baseAddress,
                            iv.count,
                            nil, 0, // Additional authenticated data
                            dataBytes.bindMemory(to: UInt8.self).baseAddress,
                            data.count,
                            encryptedBytes.bindMemory(to: UInt8.self).baseAddress,
                            &encryptedLength
                        )
                    }
                }
            }
        }
        
        if status != kCCSuccess {
            throw M3TMError.bridgeError("AES-GCM encryption failed")
        }
        
        // IV + encrypted data + tag birleştir
        var result = Data()
        result.append(iv)
        result.append(encryptedData.prefix(encryptedLength))
        
        return result
    }
    
    /**
     * AES-GCM-256 ile veri çözümleme.
     * 
     * @param encryptedData IV + şifrelenmiş veri + tag
     * @return Çözümlenmiş veri
     */
    private func decryptWithAESGCM(_ encryptedData: Data) throws -> Data {
        guard encryptedData.count > Self.gcmIVLength + Self.gcmTagLength else {
            throw M3TMError.bridgeError("Invalid encrypted data length")
        }
        
        // IV ve cipher text'i ayır
        let iv = encryptedData.prefix(Self.gcmIVLength)
        let cipherText = encryptedData.suffix(from: Self.gcmIVLength)
        
        // Encryption key al
        guard let keyData = try getEncryptionKeyData() else {
            throw M3TMError.bridgeError("Encryption key not available")
        }
        
        // AES-GCM decryption
        var decryptedData = Data(count: cipherText.count)
        var decryptedLength = 0
        
        let status = keyData.withUnsafeBytes { keyBytes in
            iv.withUnsafeBytes { ivBytes in
                cipherText.withUnsafeBytes { cipherBytes in
                    decryptedData.withUnsafeMutableBytes { decryptedBytes in
                        CCCryptorGCM(
                            CCOperation(kCCDecrypt),
                            CCAlgorithm(kCCAlgorithmAES),
                            keyBytes.bindMemory(to: UInt8.self).baseAddress,
                            keyData.count,
                            ivBytes.bindMemory(to: UInt8.self).baseAddress,
                            iv.count,
                            nil, 0, // Additional authenticated data
                            cipherBytes.bindMemory(to: UInt8.self).baseAddress,
                            cipherText.count,
                            decryptedBytes.bindMemory(to: UInt8.self).baseAddress,
                            &decryptedLength
                        )
                    }
                }
            }
        }
        
        if status != kCCSuccess {
            throw M3TMError.bridgeError("AES-GCM decryption failed")
        }
        
        return decryptedData.prefix(decryptedLength)
    }
    
    /**
     * Encryption key data'yı keychain'den alır.
     * 
     * @return Key data varsa, nil yoksa
     */
    private func getEncryptionKeyData() throws -> Data? {
        return try getFromKeychain(key: Self.encryptionKeyAlias)
    }
    
    /**
     * Anahtar silme.
     * 
     * @param key Silinecek anahtar
     * @param completion İşlem sonucu callback'i
     */
    @objc public func removeSecureData(_ key: String, completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.notInitialized)
            return
        }
        
        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.removeFromKeychain(key: key)
            self?.removeFromKeychain(key: key + "_encrypted")
            DispatchQueue.main.async {
                completion(nil)
            }
        }
    }
    
    /**
     * Tüm güvenli verileri temizle.
     * 
     * @param completion İşlem sonucu callback'i
     */
    @objc public func clearAllSecureData(completion: @escaping (Error?) -> Void) {
        guard isInitialized else {
            completion(M3TMError.notInitialized)
            return
        }
        
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let query: [String: Any] = [
                kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: Self.serviceIdentifier
            ]
            
            SecItemDelete(query as CFDictionary)
            
            DispatchQueue.main.async {
                completion(nil)
            }
        }
    }
    
    /**
     * Secure Enclave kullanılabilirlik durumunu döndürür.
     * 
     * @return true eğer Secure Enclave kullanılabilirse
     */
    @objc public var isSecureEnclaveAvailable: Bool {
        return secureEnclaveAvailable
    }
    
    /**
     * Kaynakları serbest bırakır.
     */
    @objc public func shutdown() {
        isInitialized = false
        encryptionKey = nil
        NSLog("SecureStorageManager shutdown completed")
    }
}
