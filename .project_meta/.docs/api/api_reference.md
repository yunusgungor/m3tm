# M3TM SDK API Referansı

Bu belge, M³TM SDK'sı için API referans dokümantasyonudur. SDK, hem Android hem de iOS platformlarında benzer işlevsellik sunar, ancak her platformun kendine özgü özellikleri ve arayüzleri vardır.

## API Yapısı

M³TM SDK, aşağıdaki ana bileşenlerden oluşur:

1. **Ana SDK Sınıfı** - SDK'nın ana giriş noktası (`M3TM`)
2. **Yapılandırma** - SDK yapılandırma seçenekleri (`M3TMConfig`)
3. **Model Yöneticisi** - Model yükleme ve çıkarım işlemleri (`M3TMModelManager`)
4. **Eğitim Yöneticisi** - Model adaptörleri ve görev başlıkları için eğitim işlemleri (`M3TMTrainingManager`)
5. **Arama Servisi** - Semantik arama yetenekleri (`M3TMSearchService`)
6. **Veri Dışa Aktarma** - Kullanıcı verilerini indirme ve dışa aktarma işlemleri (`M3TMDataExporter`)
7. **Hata Yönetimi** - Hata durumları ve hata kodları (`M3TMError` / `M3TMException`)

## Platform Karşılaştırması

| Özellik | Android | iOS |
|---------|---------|-----|
| Ana SDK Sınıfı | `com.m3tm.sdk.M3TM` (Singleton) | `M3TM` (Singleton) |
| Yapılandırma | `M3TMConfig` (Builder modeli) | `M3TMConfig` (nesne) |
| Model Yönetimi | `M3TMModelManager` | `M3TMModelManager` |
| Eğitim İşlemleri | `M3TMTrainingManager` | `M3TMTrainingManager` |
| Arama | `M3TMSearchService` | `M3TMSearchService` |
| Veri Dışa Aktarma | `M3TMDataExporter` | `M3TMDataExporter` |
| Hata Yönetimi | `M3TMException` | `M3TMError` (enum) |
| Paralel İşlemler | Callback arayüzü | Swift completion blokları |

## Ana SDK Sınıfı (M3TM)

M3TM, SDK'nın ana giriş noktasıdır ve diğer yönetici sınıflarına erişim sağlar.

### Android

```java
public class M3TM {
    // Singleton erişimi
    public static M3TM getInstance()
    
    // SDK'yı başlatma
    public void initialize(M3TMConfig config, M3TMCallback<Void> callback)
    
    // SDK'yı kapatma
    public void shutdown(M3TMCallback<Void> callback)
    
    // Model yöneticisine erişim
    public M3TMModelManager getModelManager() throws M3TMException
    
    // Eğitim yöneticisine erişim
    public M3TMTrainingManager getTrainingManager() throws M3TMException
    
    // Arama servisine erişim
    public M3TMSearchService getSearchService() throws M3TMException
    
    // Veri dışa aktarma servisine erişim
    public M3TMDataExporter getDataExporter() throws M3TMException
    
    // SDK sürüm bilgisi
    public String getVersion()
    
    // SDK durum bilgisi
    public boolean isInitialized()
}
```

### iOS (Swift)

```swift
@objc public class M3TM: NSObject {
    // Singleton erişimi
    @objc public static let shared = M3TM()
    
    // SDK'yı başlatma
    @objc public func initialize(config: M3TMConfig, completion: @escaping (Error?) -> Void)
    
    // SDK'yı kapatma
    @objc public func shutdown(completion: @escaping (Error?) -> Void)
    
    // Model yöneticisine erişim
    @objc public func getModelManager() throws -> M3TMModelManager
    
    // Eğitim yöneticisine erişim
    @objc public func getTrainingManager() throws -> M3TMTrainingManager
    
    // Arama servisine erişim
    @objc public func getSearchService() throws -> M3TMSearchService
    
    // Veri dışa aktarma servisine erişim
    @objc public func getDataExporter() throws -> M3TMDataExporter
    
    // SDK sürüm bilgisi
    @objc public let version: String
    
    // SDK durum bilgisi
    @objc public var isInitialized: Bool { get }
}
```

## Yapılandırma (M3TMConfig)

M3TMConfig, SDK'nın davranışını özelleştirmek için kullanılır.

### Android

```java
public class M3TMConfig {
    // Enum sınıfları
    public enum ComputeDevice { CPU, GPU, AUTO }
    public enum OptimizationLevel { NONE, BASIC, BALANCED, SPEED, SIZE }
    public enum ModelLoadingStrategy { ON_INITIALIZATION, ON_FIRST_USE, MANUAL }
    
    // Builder sınıfı
    public static class Builder {
        public Builder(Context context)
        public Builder setModelPath(String modelPath)
        public Builder setConfigPath(String configPath)
        public Builder setComputeDevice(ComputeDevice device)
        public Builder setOptimizationLevel(OptimizationLevel level)
        public Builder setMaxMemoryUsageMB(int maxMemoryMB)
        public Builder setModelLoadingStrategy(ModelLoadingStrategy strategy)
        public Builder setLogLevel(int logLevel)
        public Builder setCustomOption(String key, Object value)
        public M3TMConfig build()
    }
    
    // Getter metodları
    public Context getContext()
    public String getModelPath()
    public String getConfigPath()
    public ComputeDevice getComputeDevice()
    public OptimizationLevel getOptimizationLevel()
    public int getMaxMemoryUsageMB()
    public ModelLoadingStrategy getModelLoadingStrategy()
    public int getLogLevel()
    public Map<String, Object> getCustomOptions()
}
```

### iOS (Swift)

```swift
@objcMembers public class M3TMConfig: NSObject {
    // Enum türleri
    @objc public enum ComputeDevice: Int {
        case cpu = 0
        case metal = 1
        case auto = 2
    }
    
    @objc public enum OptimizationLevel: Int {
        case none = 0
        case basic = 1
        case balanced = 2
        case speed = 3
        case size = 4
    }
    
    @objc public enum ModelLoadingStrategy: Int {
        case onInitialization = 0
        case onFirstUse = 1
        case manual = 2
    }
    
    // Ana constructor
    public init(modelPath: String, configPath: String)
    
    // Özellikler
    public var modelPath: String
    public var configPath: String
    public var computeDevice: ComputeDevice
    public var optimizationLevel: OptimizationLevel
    public var maxMemoryUsageMB: Int
    public var modelLoadingStrategy: ModelLoadingStrategy
    public var logLevel: Int
    public var customOptions: [String: Any]
}
```

## Model Yöneticisi (M3TMModelManager)

M3TMModelManager, model yönetimini ve çıkarım (inference) işlemlerini gerçekleştirir.

### Android

```java
public class M3TMModelManager {
    // İç içe sınıflar
    public enum OutputFormat { JSON, TENSOR, DICTIONARY, ARRAY }
    
    public static class InputData {
        public static class Builder {
            public Builder setText(String text)
            public Builder setImage(Bitmap image)
            public Builder setTensor(float[] tensorData, int[] shape)
            public Builder setEmbedding(float[] embedding)
            public Builder setOptions(Map<String, Object> options)
            public InputData build()
        }
    }
    
    public static class InferenceOptions {
        public static class Builder {
            public Builder setOutputFormat(OutputFormat format)
            public Builder setCustomOptions(Map<String, Object> options)
            public InferenceOptions build()
        }
    }
    
    // Ana metodlar
    public void loadModel() throws M3TMException
    public void unloadModel() throws M3TMException
    public boolean isModelLoaded()
    
    // Çıkarım işlemleri
    public void inference(List<InputData> inputs, InferenceOptions options, M3TMCallback<List<Object>> callback)
    public void getModelInfo(M3TMCallback<Map<String, Object>> callback)
}
```

### iOS (Swift)

```swift
@objcMembers public class M3TMModelManager: NSObject {
    // Enum türleri
    @objc public enum OutputFormat: Int {
        case json = 0
        case tensor = 1
        case dictionary = 2
        case array = 3
    }
    
    // İç içe sınıflar
    @objc public class InputData: NSObject {
        public init(text: String)
        public init(image: UIImage)
        public init(tensorData: Data, shape: [Int])
        public init(embedding: [Float])
        
        public var text: String?
        public var image: UIImage?
        public var tensorData: Data?
        public var tensorShape: [Int]?
        public var embedding: [Float]?
        public var options: [String: Any]?
    }
    
    @objc public class InferenceOptions: NSObject {
        public var outputFormat: OutputFormat
        public var customOptions: [String: Any]?
    }
    
    // Ana metodlar
    public func loadModel() throws
    public func unloadModel() throws
    public var isModelLoaded: Bool { get }
    
    // Çıkarım işlemleri
    public func inference(inputs: [InputData], options: InferenceOptions, completion: @escaping (Any?, Error?) -> Void)
    public func getModelInfo(completion: @escaping ([String: Any]?, Error?) -> Void)
}
```

## Eğitim Yöneticisi (M3TMTrainingManager)

M3TMTrainingManager, model adaptörlerini ve görev başlıklarını cihaz üzerinde eğitmek için kullanılır.

### Android

```java
public class M3TMTrainingManager {
    // Arayüzler
    public interface DataProvider {
        int getDataCount();
        List<M3TMModelManager.InputData> getData(int[] indices);
        List<Object> getLabels(int[] indices);
    }
    
    public interface TrainingDelegate {
        void onTrainingProgress(float progress, Map<String, Object> metrics, int epoch);
        void onEpochComplete(int epoch, Map<String, Object> metrics);
        void onTrainingComplete(boolean success, Map<String, Object> finalMetrics);
        void onTrainingError(M3TMException error);
    }
    
    // İç içe sınıflar
    public static class TrainingConfig {
        public static class Builder {
            public Builder setLearningRate(float learningRate)
            public Builder setEpochs(int epochs)
            public Builder setBatchSize(int batchSize)
            public Builder setCustomOptions(Map<String, Object> options)
            public TrainingConfig build()
        }
    }
    
    // Yönetim metodları
    public void setTrainingDelegate(TrainingDelegate delegate)
    
    // Adapter işlemleri
    public void createAdapter(int blockIndex, Map<String, Object> config, M3TMCallback<String> callback)
    public void trainAdapter(String adapterId, DataProvider dataProvider, TrainingConfig config, M3TMCallback<Void> callback)
    public void getAdapters(M3TMCallback<List<String>> callback)
    public void getAdapterInfo(String adapterId, M3TMCallback<Map<String, Object>> callback)
    public void removeAdapter(String adapterId, M3TMCallback<Void> callback)
    
    // Görev başlığı işlemleri
    public void createTaskHead(String taskType, Map<String, Object> config, M3TMCallback<String> callback)
    public void trainTaskHead(String taskHeadId, DataProvider dataProvider, TrainingConfig config, M3TMCallback<Void> callback)
    public void getTaskHeads(M3TMCallback<List<String>> callback)
    public void getTaskHeadInfo(String taskHeadId, M3TMCallback<Map<String, Object>> callback)
    public void removeTaskHead(String taskHeadId, M3TMCallback<Void> callback)
    
    // Eğitim durumu
    public boolean isTrainingInProgress()
    public void cancelTraining(M3TMCallback<Void> callback)
}
```

### iOS (Swift)

```swift
@objcMembers public class M3TMTrainingManager: NSObject {
    // Protokoller
    @objc public protocol DataProvider {
        func getDataCount() -> Int
        func getData(forIndices indices: [Int]) -> [M3TMModelManager.InputData]
        func getLabels(forIndices indices: [Int]) -> [Any]
    }
    
    @objc public protocol M3TMTrainingDelegate: AnyObject {
        @objc optional func trainingManager(_ manager: M3TMTrainingManager, didUpdateProgress progress: Float, metrics: [String: Any]?, forEpoch epoch: Int)
        @objc optional func trainingManager(_ manager: M3TMTrainingManager, didCompleteEpoch epoch: Int, withMetrics metrics: [String: Any]?)
        @objc optional func trainingManager(_ manager: M3TMTrainingManager, didFinishWithSuccess success: Bool, finalMetrics: [String: Any]?)
        @objc optional func trainingManager(_ manager: M3TMTrainingManager, didEncounterError error: Error)
    }
    
    // İç içe sınıflar
    @objc public class TrainingConfig: NSObject {
        public var learningRate: Float
        public var epochs: Int
        public var batchSize: Int
        public var customOptions: [String: Any]?
    }
    
    // Yönetim özellikleri
    public weak var delegate: M3TMTrainingDelegate?
    
    // Adapter işlemleri
    public func createAdapter(forBlockIndex blockIndex: Int, config: [String: Any], completion: @escaping (String?, Error?) -> Void)
    public func trainAdapter(adapterId: String, dataProvider: DataProvider, config: TrainingConfig, completion: @escaping (Error?) -> Void)
    public func getAdapters(completion: @escaping ([String]?, Error?) -> Void)
    public func getAdapterInfo(adapterId: String, completion: @escaping ([String: Any]?, Error?) -> Void)
    public func removeAdapter(adapterId: String, completion: @escaping (Error?) -> Void)
    
    // Görev başlığı işlemleri
    public func createTaskHead(taskType: String, config: [String: Any], completion: @escaping (String?, Error?) -> Void)
    public func trainTaskHead(taskHeadId: String, dataProvider: DataProvider, config: TrainingConfig, completion: @escaping (Error?) -> Void)
    public func getTaskHeads(completion: @escaping ([String]?, Error?) -> Void)
    public func getTaskHeadInfo(taskHeadId: String, completion: @escaping ([String: Any]?, Error?) -> Void)
    public func removeTaskHead(taskHeadId: String, completion: @escaping (Error?) -> Void)
    
    // Eğitim durumu
    public var isTrainingInProgress: Bool { get }
    public func cancelTraining(completion: @escaping (Error?) -> Void)
}
```

## Arama Servisi (M3TMSearchService)

M3TMSearchService, model kullanarak semantik arama yetenekleri sağlar.

### Android

```java
public class M3TMSearchService {
    // İç içe sınıflar
    public static class SearchResult {
        public String getItemId()
        public float getScore()
        public String getTextContent()
        public Bitmap getImageData()
        public Map<String, Object> getMetadata()
    }
    
    public static class SearchFilters {
        public void setContentTypes(List<String> contentTypes)
        public void setDateStart(Date dateStart)
        public void setDateEnd(Date dateEnd)
        public void setCustomFilters(Map<String, Object> customFilters)
    }
    
    // Ana metodlar
    public void initialize(M3TMCallback<Void> callback)
    public boolean isInitialized()
    
    // İndeksleme işlemleri
    public void indexItem(String itemId, String textContent, Bitmap imageData, Map<String, Object> metadata, M3TMCallback<Void> callback)
    public void removeItem(String itemId, M3TMCallback<Void> callback)
    public void getIndexedItems(M3TMCallback<List<String>> callback)
    public void clearIndex(M3TMCallback<Void> callback)
    
    // Arama işlemleri
    public void search(String textQuery, SearchFilters filters, int topK, M3TMCallback<List<SearchResult>> callback)
    public void searchByImage(Bitmap imageQuery, SearchFilters filters, int topK, M3TMCallback<List<SearchResult>> callback)
    public void searchByEmbedding(float[] embeddingQuery, SearchFilters filters, int topK, M3TMCallback<List<SearchResult>> callback)
    
    // İndeks yönetimi
    public void saveIndex(String path, M3TMCallback<Void> callback)
    public void loadIndex(String path, M3TMCallback<Void> callback)
    public void getIndexInfo(M3TMCallback<Map<String, Object>> callback)
}
```

### iOS (Swift)

```swift
@objcMembers public class M3TMSearchService: NSObject {
    // İç içe sınıflar
    @objc public class SearchResult: NSObject {
        public let itemId: String
        public let score: Float
        public let textContent: String?
        public let imageData: Data?
        public let metadata: [String: Any]?
    }
    
    @objc public class SearchFilters: NSObject {
        public var contentTypes: [String]?
        public var dateStart: Date?
        public var dateEnd: Date?
        public var customFilters: [String: Any]?
    }
    
    // Ana metodlar
    public func initialize(completion: @escaping (Error?) -> Void)
    public var isInitialized: Bool { get }
    
    // İndeksleme işlemleri
    public func indexItem(itemId: String, textContent: String?, imageData: UIImage?, metadata: [String: Any]?, completion: @escaping (Error?) -> Void)
    public func removeItem(itemId: String, completion: @escaping (Error?) -> Void)
    public func getIndexedItems(completion: @escaping ([String]?, Error?) -> Void)
    public func clearIndex(completion: @escaping (Error?) -> Void)
    
    // Arama işlemleri
    public func search(textQuery: String, filters: SearchFilters?, topK: Int, completion: @escaping ([SearchResult]?, Error?) -> Void)
    public func search(imageQuery: UIImage, filters: SearchFilters?, topK: Int, completion: @escaping ([SearchResult]?, Error?) -> Void)
    public func search(embeddingQuery: [Float], filters: SearchFilters?, topK: Int, completion: @escaping ([SearchResult]?, Error?) -> Void)
    
    // İndeks yönetimi
    public func saveIndex(toPath path: String, completion: @escaping (Error?) -> Void)
    public func loadIndex(fromPath path: String, completion: @escaping (Error?) -> Void)
    public func getIndexInfo(completion: @escaping ([String: Any]?, Error?) -> Void)
}
```

## Veri Dışa Aktarma (M3TMDataExporter)

M3TMDataExporter, kullanıcının kişisel verilerini dışa aktarmasını sağlar.

### Android

```java
public class M3TMDataExporter {
    // Enum sınıfları
    public enum ExportFormat { JSON, CSV, CUSTOM }
    
    // İç içe sınıflar
    public static class ExportFilters {
        public void setContentTypes(List<String> contentTypes)
        public void setDateStart(Date dateStart)
        public void setDateEnd(Date dateEnd)
        public void setCustomFilters(Map<String, Object> customFilters)
    }
    
    public static class ExportOptions {
        public static class Builder {
            public Builder setFormat(ExportFormat format)
            public Builder setIncludeImages(boolean includeImages)
            public Builder setImageQuality(float quality)
            public Builder setCustomOptions(Map<String, Object> options)
            public ExportOptions build()
        }
    }
    
    public static class ExportResult {
        public File getFile()
        public int getItemCount()
    }
    
    // Ana metodlar
    public void initialize(M3TMCallback<Void> callback)
    public boolean isInitialized()
    public void exportData(ExportFilters filters, ExportOptions options, M3TMCallback<ExportResult> callback)
    public void getAvailableDataInfo(M3TMCallback<Map<String, Object>> callback)
}
```

### iOS (Swift)

```swift
@objcMembers public class M3TMDataExporter: NSObject {
    // Enum türleri
    @objc public enum ExportFormat: Int {
        case json = 0
        case csv = 1
        case custom = 2
    }
    
    // İç içe sınıflar
    @objc public class ExportFilters: NSObject {
        public var contentTypes: [String]?
        public var dateStart: Date?
        public var dateEnd: Date?
        public var customFilters: [String: Any]?
    }
    
    @objc public class ExportOptions: NSObject {
        public var format: ExportFormat
        public var includeImages: Bool
        public var imageQuality: Float
        public var customOptions: [String: Any]?
    }
    
    // Ana metodlar
    public func initialize(completion: @escaping (Error?) -> Void)
    public var isInitialized: Bool { get }
    public func exportData(filters: ExportFilters?, options: ExportOptions, completion: @escaping (URL?, Int, Error?) -> Void)
    public func getAvailableDataInfo(completion: @escaping ([String: Any]?, Error?) -> Void)
}
```

## Hata Yönetimi

### Android - M3TMException

```java
public class M3TMException extends Exception {
    // Hata kodları
    public static final int NOT_INITIALIZED = 1001;
    public static final int ALREADY_INITIALIZED = 1002;
    public static final int MODEL_LOAD_FAILED = 2001;
    public static final int MODEL_NOT_LOADED = 2002;
    public static final int MODEL_VERSION_MISMATCH = 2003;
    public static final int MODULE_NOT_SUPPORTED = 2004;
    public static final int MODEL_CONFIGURATION_ERROR = 2005;
    public static final int TRAINING_FAILED = 3001;
    public static final int TRAINING_IN_PROGRESS = 3002;
    public static final int TRAINING_CANCELLED = 3003;
    public static final int TRAINING_DATA_INVALID = 3004;
    public static final int ADAPTER_CREATION_FAILED = 3005;
    public static final int TASK_HEAD_CREATION_FAILED = 3006;
    public static final int INFERENCE_FAILED = 4001;
    public static final int INPUT_DATA_INVALID = 4002;
    public static final int OUTPUT_FORMAT_ERROR = 4003;
    public static final int SEARCH_INDEX_NOT_INITIALIZED = 5001;
    public static final int SEARCH_FAILED = 5002;
    public static final int ITEM_NOT_FOUND = 5003;
    public static final int INDEX_OPERATION_FAILED = 5004;
    public static final int EXPORTER_NOT_INITIALIZED = 6001;
    public static final int EXPORT_FAILED = 6002;
    public static final int FILE_IO_ERROR = 7001;
    public static final int PERMISSION_DENIED = 7002;
    public static final int OUT_OF_MEMORY = 8001;
    public static final int TIMEOUT = 8002;
    public static final int INTERNAL_ERROR = 9001;
    public static final int UNSUPPORTED_OPERATION = 9002;
    
    // Metodlar
    public int getErrorCode()
    public Map<String, Object> getErrorDetails()
}
```

### iOS - M3TMError

```swift
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
    case itemNotFound = 5003
    case indexOperationFailed = 5004
    
    // Dışa Aktarma Hataları
    case exporterNotInitialized = 6001
    case exportFailed = 6002
    
    // Dosya ve İzin Hataları
    case fileIOError = 7001
    case permissionDenied = 7002
    
    // Sistem Hataları
    case outOfMemory = 8001
    case timeout = 8002
    
    // Genel Hatalar
    case internalError = 9001
    case unsupportedOperation = 9002
}
```

## Ayrıntılı Dokümantasyon

Daha ayrıntılı platform özel dokümantasyon için:

- [Android API Referansı](./endpoints/android_api_reference.md)
- [iOS API Referansı](./endpoints/ios_api_reference.md)
- [API Kullanım Örnekleri](./usage_examples/)
  - [Android (Java)](./usage_examples/M3TM_Java_Usage_Example.md)
  - [iOS (Swift)](./usage_examples/M3TM_Swift_Usage_Example.md)
- [Hata Kodları ve Çözümleri](./endpoints/error_codes.md)
- [Veri Modelleri](./models/) 