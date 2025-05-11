import Foundation

/**
 * M3TM SDK hata tanımlamaları
 */
@objc public enum M3TMError: Int, Error {
    // SDK Yaşam Döngüsü Hataları
    case notInitialized = 1001
    case alreadyInitialized = 1002
    
    // Model Hataları
    case modelLoadFailed = 2001
    case modelNotLoaded = 2002
    case modelVersionMismatch = 2003
    case moduleNotSupported = 2004
    case modelConfigurationError = 2005
    
    // Eğitim Hataları
    case trainingFailed = 3001
    case trainingInProgress = 3002
    case trainingCancelled = 3003
    case trainingDataInvalid = 3004
    case adapterCreationFailed = 3005
    case taskHeadCreationFailed = 3006
    
    // Çıkarım Hataları
    case inferenceFailed = 4001
    case inputDataInvalid = 4002
    case outputFormatError = 4003
    
    // Arama Hataları
    case searchIndexNotInitialized = 5001
    case searchFailed = 5002
    case searchQueryInvalid = 5003
    
    // Veri İndirme Hataları
    case exportFailed = 6001
    case exportPermissionDenied = 6002
    
    // Kaynak Hataları
    case outOfMemory = 7001
    case deviceNotSupported = 7002
    case gpuNotAvailable = 7003
    
    // İO Hataları
    case fileNotFound = 8001
    case fileAccessDenied = 8002
    case fileSaveFailed = 8003
    
    // İçsel Hataları
    case internalError = 9001
    case bridgeError = 9002
    case unexpectedError = 9999
    
    /// Hata mesajları
    private static let errorMessages: [M3TMError: String] = [
        .notInitialized: "SDK is not initialized",
        .alreadyInitialized: "SDK is already initialized",
        
        .modelLoadFailed: "Failed to load model",
        .modelNotLoaded: "Model is not loaded",
        .modelVersionMismatch: "Model version mismatch",
        .moduleNotSupported: "Module is not supported on this platform",
        .modelConfigurationError: "Model configuration error",
        
        .trainingFailed: "Training failed",
        .trainingInProgress: "Training is already in progress",
        .trainingCancelled: "Training was cancelled",
        .trainingDataInvalid: "Training data is invalid",
        .adapterCreationFailed: "Failed to create adapter",
        .taskHeadCreationFailed: "Failed to create task head",
        
        .inferenceFailed: "Inference failed",
        .inputDataInvalid: "Input data is invalid",
        .outputFormatError: "Output format error",
        
        .searchIndexNotInitialized: "Search index is not initialized",
        .searchFailed: "Search failed",
        .searchQueryInvalid: "Search query is invalid",
        
        .exportFailed: "Export failed",
        .exportPermissionDenied: "Export permission denied",
        
        .outOfMemory: "Out of memory",
        .deviceNotSupported: "Device not supported",
        .gpuNotAvailable: "GPU not available",
        
        .fileNotFound: "File not found",
        .fileAccessDenied: "File access denied",
        .fileSaveFailed: "File save failed",
        
        .internalError: "Internal error",
        .bridgeError: "Bridge error",
        .unexpectedError: "Unexpected error occurred"
    ]
    
    /// NSError domain
    public static let domain = "com.m3tm.sdk.error"
    
    /// Özel hata mesajlı internal error durumu
    public static func internalError(_ message: String) -> Error {
        return NSError(domain: M3TMError.domain,
                      code: M3TMError.internalError.rawValue,
                      userInfo: [NSLocalizedDescriptionKey: message])
    }
    
    /// Özel hata mesajlı bridge error durumu
    public static func bridgeError(_ message: String) -> Error {
        return NSError(domain: M3TMError.domain,
                      code: M3TMError.bridgeError.rawValue,
                      userInfo: [NSLocalizedDescriptionKey: message])
    }
}

/// Error protokolü uyumluluğu için eklenti
extension M3TMError: LocalizedError {
    public var errorDescription: String? {
        return M3TMError.errorMessages[self] ?? "Unknown error"
    }
} 