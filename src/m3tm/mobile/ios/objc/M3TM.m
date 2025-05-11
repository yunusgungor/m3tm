#import "M3TM.h"
#import <CoreGraphics/CoreGraphics.h>

// Özel C++ köprüsü bağlantısı
#ifdef __cplusplus
extern "C" {
#endif

// Bu fonksiyonlar C++ bridge'ine bağlantı sağlar
void* ModelManager_Create(const char* modelBasePath, const char* configPath);
void ModelManager_Destroy(void* manager);
int64_t ModelManager_LoadModel(void* manager, const char* modelId, const char* version);
void* ModelManager_ProcessText(void* manager, int64_t modelHandle, const char* text, const char* task, void* options);
void* ModelManager_ProcessImage(void* manager, int64_t modelHandle, const unsigned char* imageData, int width, int height, int channels, const char* task, void* options);
bool ModelManager_CloseModel(void* manager, int64_t modelHandle);

void* TrainingManager_Create(const char* modelBasePath, const char* configPath);
void TrainingManager_Destroy(void* manager);
const char* TrainingManager_CreateSession(void* manager, int64_t modelHandle, const char* task, void* options);
bool TrainingManager_AddTextSample(void* manager, const char* sessionId, const char* text, const char* label);
bool TrainingManager_AddImageSample(void* manager, const char* sessionId, const unsigned char* imageData, int width, int height, int channels, const char* label);
bool TrainingManager_StartTraining(void* manager, const char* sessionId, int epochs, int callbackInterval);
bool TrainingManager_CancelTraining(void* manager, const char* sessionId);
void* TrainingManager_GetStatus(void* manager, const char* sessionId);
bool TrainingManager_CloseSession(void* manager, const char* sessionId);

void* SearchManager_Create(const char* modelBasePath, const char* indexPath);
void SearchManager_Destroy(void* manager);
void* SearchManager_SearchText(void* manager, int64_t modelHandle, const char* query, void* options);
void* SearchManager_SearchImage(void* manager, int64_t modelHandle, const unsigned char* imageData, int width, int height, int channels, void* options);
bool SearchManager_IndexText(void* manager, int64_t modelHandle, const char* text, void* metadata);
bool SearchManager_IndexImage(void* manager, int64_t modelHandle, const unsigned char* imageData, int width, int height, int channels, void* metadata);
bool SearchManager_SaveIndex(void* manager);
bool SearchManager_LoadIndex(void* manager);

// Callback fonksiyonları için typedef'ler
typedef void (*BatchCompletedCallback)(const char* sessionId, int batch, void* metrics);
typedef void (*EpochCompletedCallback)(const char* sessionId, int epoch, void* metrics);
typedef void (*TrainingCompletedCallback)(const char* sessionId, void* metrics);
typedef void (*TrainingErrorCallback)(const char* sessionId, int errorCode, const char* errorMessage);

// Callback fonksiyonlarını kaydetme
void TrainingManager_SetCallbacks(void* manager, const char* sessionId, 
                                 BatchCompletedCallback batchCallback,
                                 EpochCompletedCallback epochCallback,
                                 TrainingCompletedCallback completedCallback,
                                 TrainingErrorCallback errorCallback);

#ifdef __cplusplus
}
#endif

// Yardımcı fonksiyonlar
@interface M3TMUtils : NSObject

+ (NSData *)dataFromUIImage:(UIImage *)image;
+ (NSDictionary<NSString *, NSObject *> *)dictionaryFromNativePtr:(void *)ptr;
+ (void *)nativePtrFromDictionary:(NSDictionary<NSString *, NSObject *> *)dict;
+ (NSArray<M3TMSearchResult *> *)searchResultsFromNativePtr:(void *)ptr;
+ (NSError *)errorWithCode:(NSInteger)code message:(NSString *)message;

@end

@implementation M3TMUtils

+ (NSData *)dataFromUIImage:(UIImage *)image {
    CGImageRef cgImage = image.CGImage;
    NSUInteger width = CGImageGetWidth(cgImage);
    NSUInteger height = CGImageGetHeight(cgImage);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    
    unsigned char *rawData = (unsigned char *)calloc(height * width * bytesPerPixel, sizeof(unsigned char));
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                               bitsPerComponent, bytesPerRow, colorSpace,
                                               kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
    CGContextRelease(context);
    
    NSData *data = [NSData dataWithBytes:rawData length:height * width * bytesPerPixel];
    free(rawData);
    
    return data;
}

+ (NSDictionary<NSString *, NSObject *> *)dictionaryFromNativePtr:(void *)ptr {
    // Bu fonksiyon, native C++ köprüsünden gelen verileri NSDictionary'ye dönüştürür
    // Gerçek implementasyonda C++ tarafından oluşturulan veri yapısı parse edilmelidir
    
    // Örnek implementasyon:
    return @{@"status": @"success"};
}

+ (void *)nativePtrFromDictionary:(NSDictionary<NSString *, NSObject *> *)dict {
    // Bu fonksiyon, NSDictionary'yi native C++ köprüsünün anlayacağı bir veri yapısına dönüştürür
    // Gerçek implementasyonda Objective-C dictionary'si C++ map'ine dönüştürülmelidir
    
    // Örnek implementasyon (sadece yer tutucu):
    return NULL;
}

+ (NSArray<M3TMSearchResult *> *)searchResultsFromNativePtr:(void *)ptr {
    // Bu fonksiyon, native C++ köprüsünden gelen arama sonuçlarını NSArray<M3TMSearchResult *> tipine dönüştürür
    // Gerçek implementasyonda C++ tarafından oluşturulan sonuç listesi parse edilmelidir
    
    // Örnek implementasyon:
    NSMutableArray<M3TMSearchResult *> *results = [NSMutableArray array];
    // Parse edilmiş sonuçları ekle
    return results;
}

+ (NSError *)errorWithCode:(NSInteger)code message:(NSString *)message {
    return [NSError errorWithDomain:@"com.m3tm.sdk" 
                               code:code 
                           userInfo:@{NSLocalizedDescriptionKey: message}];
}

@end

// Model bilgisi implementasyonu
@implementation M3TMModelInfo

- (instancetype)initWithModelId:(NSString *)modelId 
                        version:(NSString *)version 
                 supportedTasks:(NSArray<NSString *> *)supportedTasks 
            supportedModalities:(NSArray<NSString *> *)supportedModalities {
    self = [super init];
    if (self) {
        _modelId = [modelId copy];
        _version = [version copy];
        _supportedTasks = [supportedTasks copy];
        _supportedModalities = [supportedModalities copy];
    }
    return self;
}

@end

// M3TMException
@implementation M3TMException

- (instancetype)initWithCode:(NSInteger)code message:(NSString *)message {
    self = [super initWithName:@"M3TMException" reason:message userInfo:nil];
    if (self) {
        _errorCode = code;
    }
    return self;
}

@end

// Model sınıfı implementasyonu
@interface M3TMModel () {
    void *_managerPtr;
    int64_t _modelHandle;
    BOOL _closed;
}

@end

@implementation M3TMModel

- (instancetype)initWithManager:(void *)managerPtr 
                    modelHandle:(int64_t)modelHandle 
                      modelInfo:(M3TMModelInfo *)modelInfo {
    self = [super init];
    if (self) {
        _managerPtr = managerPtr;
        _modelHandle = modelHandle;
        _modelInfo = modelInfo;
        _closed = NO;
    }
    return self;
}

- (void)dealloc {
    [self close];
}

- (NSDictionary<NSString *, NSObject *> *)processText:(NSString *)text 
                                                 task:(NSString *)task 
                                              options:(NSDictionary<NSString *, NSObject *> *)options 
                                                error:(NSError **)error {
    if (_closed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:1001 message:@"Model kapatılmış"];
        }
        return nil;
    }
    
    void *optionsPtr = [M3TMUtils nativePtrFromDictionary:options];
    void *resultPtr = ModelManager_ProcessText(_managerPtr, _modelHandle, [text UTF8String], [task UTF8String], optionsPtr);
    
    if (!resultPtr) {
        if (error) {
            *error = [M3TMUtils errorWithCode:1002 message:@"Metin işlenemedi"];
        }
        return nil;
    }
    
    return [M3TMUtils dictionaryFromNativePtr:resultPtr];
}

- (NSDictionary<NSString *, NSObject *> *)processImage:(UIImage *)image 
                                                  task:(NSString *)task 
                                               options:(NSDictionary<NSString *, NSObject *> *)options 
                                                 error:(NSError **)error {
    if (_closed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:1001 message:@"Model kapatılmış"];
        }
        return nil;
    }
    
    NSData *imageData = [M3TMUtils dataFromUIImage:image];
    CGImageRef cgImage = image.CGImage;
    NSUInteger width = CGImageGetWidth(cgImage);
    NSUInteger height = CGImageGetHeight(cgImage);
    NSUInteger channels = 4; // RGBA
    
    void *optionsPtr = [M3TMUtils nativePtrFromDictionary:options];
    void *resultPtr = ModelManager_ProcessImage(_managerPtr, _modelHandle, [imageData bytes], (int)width, (int)height, (int)channels, [task UTF8String], optionsPtr);
    
    if (!resultPtr) {
        if (error) {
            *error = [M3TMUtils errorWithCode:1003 message:@"Görüntü işlenemedi"];
        }
        return nil;
    }
    
    return [M3TMUtils dictionaryFromNativePtr:resultPtr];
}

- (void)close {
    if (!_closed && _managerPtr != NULL && _modelHandle != 0) {
        ModelManager_CloseModel(_managerPtr, _modelHandle);
        _closed = YES;
    }
}

- (BOOL)isClosed {
    return _closed;
}

@end

// Training session sınıfı implementasyonu
@interface M3TMTrainingSession () {
    void *_managerPtr;
    NSString *_sessionId;
    BOOL _closed;
}

@end

@implementation M3TMTrainingSession

- (instancetype)initWithManager:(void *)managerPtr sessionId:(NSString *)sessionId {
    self = [super init];
    if (self) {
        _managerPtr = managerPtr;
        _sessionId = [sessionId copy];
        _closed = NO;
        
        // Callback'leri ayarla
        TrainingManager_SetCallbacks(_managerPtr, [_sessionId UTF8String], 
                                   &batchCompletedCallback,
                                   &epochCompletedCallback,
                                   &trainingCompletedCallback,
                                   &trainingErrorCallback);
    }
    return self;
}

// Callback fonksiyonları
static void batchCompletedCallback(const char* sessionId, int batch, void* metricsPtr) {
    NSString *sid = [NSString stringWithUTF8String:sessionId];
    M3TMTrainingSession *session = nil; // Burada session'ı bulmak için bir mekanizma gerekli
    
    if (session && session.delegate && [session.delegate respondsToSelector:@selector(onBatchComplete:metrics:)]) {
        NSDictionary *metrics = [M3TMUtils dictionaryFromNativePtr:metricsPtr];
        [session.delegate onBatchComplete:batch metrics:metrics];
    }
}

static void epochCompletedCallback(const char* sessionId, int epoch, void* metricsPtr) {
    NSString *sid = [NSString stringWithUTF8String:sessionId];
    M3TMTrainingSession *session = nil; // Burada session'ı bulmak için bir mekanizma gerekli
    
    if (session && session.delegate && [session.delegate respondsToSelector:@selector(onEpochComplete:metrics:)]) {
        NSDictionary *metrics = [M3TMUtils dictionaryFromNativePtr:metricsPtr];
        [session.delegate onEpochComplete:epoch metrics:metrics];
    }
}

static void trainingCompletedCallback(const char* sessionId, void* metricsPtr) {
    NSString *sid = [NSString stringWithUTF8String:sessionId];
    M3TMTrainingSession *session = nil; // Burada session'ı bulmak için bir mekanizma gerekli
    
    if (session && session.delegate && [session.delegate respondsToSelector:@selector(onTrainingComplete:)]) {
        NSDictionary *metrics = [M3TMUtils dictionaryFromNativePtr:metricsPtr];
        [session.delegate onTrainingComplete:metrics];
    }
}

static void trainingErrorCallback(const char* sessionId, int errorCode, const char* errorMessage) {
    NSString *sid = [NSString stringWithUTF8String:sessionId];
    M3TMTrainingSession *session = nil; // Burada session'ı bulmak için bir mekanizma gerekli
    
    if (session && session.delegate && [session.delegate respondsToSelector:@selector(onTrainingError:)]) {
        NSString *message = [NSString stringWithUTF8String:errorMessage];
        NSError *error = [M3TMUtils errorWithCode:errorCode message:message];
        [session.delegate onTrainingError:error];
    }
}

- (BOOL)addTextSample:(NSString *)text label:(NSString *)label error:(NSError **)error {
    if (_closed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:2001 message:@"Eğitim oturumu kapatılmış"];
        }
        return NO;
    }
    
    BOOL result = TrainingManager_AddTextSample(_managerPtr, [_sessionId UTF8String], [text UTF8String], [label UTF8String]);
    
    if (!result && error) {
        *error = [M3TMUtils errorWithCode:2002 message:@"Metin örneği eklenemedi"];
    }
    
    return result;
}

- (BOOL)addImageSample:(UIImage *)image label:(NSString *)label error:(NSError **)error {
    if (_closed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:2001 message:@"Eğitim oturumu kapatılmış"];
        }
        return NO;
    }
    
    NSData *imageData = [M3TMUtils dataFromUIImage:image];
    CGImageRef cgImage = image.CGImage;
    NSUInteger width = CGImageGetWidth(cgImage);
    NSUInteger height = CGImageGetHeight(cgImage);
    NSUInteger channels = 4; // RGBA
    
    BOOL result = TrainingManager_AddImageSample(_managerPtr, [_sessionId UTF8String], [imageData bytes], (int)width, (int)height, (int)channels, [label UTF8String]);
    
    if (!result && error) {
        *error = [M3TMUtils errorWithCode:2003 message:@"Görüntü örneği eklenemedi"];
    }
    
    return result;
}

- (BOOL)startTrainingWithEpochs:(NSInteger)epochs error:(NSError **)error {
    if (_closed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:2001 message:@"Eğitim oturumu kapatılmış"];
        }
        return NO;
    }
    
    BOOL result = TrainingManager_StartTraining(_managerPtr, [_sessionId UTF8String], (int)epochs, 10); // 10 batch'de bir callback
    
    if (!result && error) {
        *error = [M3TMUtils errorWithCode:2004 message:@"Eğitim başlatılamadı"];
    }
    
    return result;
}

- (BOOL)cancelTrainingWithError:(NSError **)error {
    if (_closed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:2001 message:@"Eğitim oturumu kapatılmış"];
        }
        return NO;
    }
    
    BOOL result = TrainingManager_CancelTraining(_managerPtr, [_sessionId UTF8String]);
    
    if (!result && error) {
        *error = [M3TMUtils errorWithCode:2005 message:@"Eğitim iptal edilemedi"];
    }
    
    return result;
}

- (nullable NSDictionary<NSString *, NSObject *> *)getTrainingStatusWithError:(NSError **)error {
    if (_closed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:2001 message:@"Eğitim oturumu kapatılmış"];
        }
        return nil;
    }
    
    void *statusPtr = TrainingManager_GetStatus(_managerPtr, [_sessionId UTF8String]);
    
    if (!statusPtr) {
        if (error) {
            *error = [M3TMUtils errorWithCode:2006 message:@"Eğitim durumu alınamadı"];
        }
        return nil;
    }
    
    return [M3TMUtils dictionaryFromNativePtr:statusPtr];
}

- (void)close {
    if (!_closed && _managerPtr != NULL) {
        TrainingManager_CloseSession(_managerPtr, [_sessionId UTF8String]);
        _closed = YES;
    }
}

- (BOOL)isClosed {
    return _closed;
}

- (void)dealloc {
    [self close];
}

@end

// Search result implementasyonu
@implementation M3TMSearchResult

- (instancetype)initWithItemId:(NSString *)itemId score:(float)score metadata:(NSDictionary<NSString *, NSString *> *)metadata {
    self = [super init];
    if (self) {
        _itemId = [itemId copy];
        _score = score;
        _metadata = [metadata copy];
    }
    return self;
}

@end

// M3TM ana sınıf implementasyonu
@interface M3TM () {
    void *_modelManagerPtr;
    void *_trainingManagerPtr;
    void *_searchManagerPtr;
    BOOL _initialized;
    
    // Yüklenen modelleri izlemek için sözlük
    NSMutableDictionary<NSString *, M3TMModel *> *_loadedModels;
    
    // Eğitim oturumlarını izlemek için sözlük
    NSMutableDictionary<NSString *, M3TMTrainingSession *> *_trainingSessions;
}

@end

@implementation M3TM

+ (instancetype)sharedInstance {
    static M3TM *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[self alloc] init];
    });
    return instance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _modelManagerPtr = NULL;
        _trainingManagerPtr = NULL;
        _searchManagerPtr = NULL;
        _initialized = NO;
        _loadedModels = [NSMutableDictionary dictionary];
        _trainingSessions = [NSMutableDictionary dictionary];
    }
    return self;
}

- (BOOL)initializeWithOptions:(NSDictionary<NSString *, NSObject *> *)options error:(NSError **)error {
    if (_initialized) {
        return YES; // Zaten başlatıldı
    }
    
    // Dizin yollarını al
    NSString *modelBasePath = options[@"modelBasePath"];
    NSString *configPath = options[@"configPath"];
    NSString *indexPath = options[@"indexPath"];
    
    if (!modelBasePath) {
        modelBasePath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:@"m3tm/models"];
    }
    
    if (!configPath) {
        configPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:@"m3tm/config.json"];
    }
    
    if (!indexPath) {
        indexPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject stringByAppendingPathComponent:@"m3tm/search_index"];
    }
    
    // Dizinlerin varlığını kontrol et
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSError *fileError = nil;
    
    // Model dizini oluştur
    if (![fileManager fileExistsAtPath:modelBasePath]) {
        if (![fileManager createDirectoryAtPath:modelBasePath withIntermediateDirectories:YES attributes:nil error:&fileError]) {
            if (error) {
                *error = [M3TMUtils errorWithCode:3001 message:[NSString stringWithFormat:@"Model dizini oluşturulamadı: %@", fileError.localizedDescription]];
            }
            return NO;
        }
    }
    
    // Manager'ları oluştur
    _modelManagerPtr = ModelManager_Create([modelBasePath UTF8String], [configPath UTF8String]);
    if (!_modelManagerPtr) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3002 message:@"ModelManager oluşturulamadı"];
        }
        return NO;
    }
    
    _trainingManagerPtr = TrainingManager_Create([modelBasePath UTF8String], [configPath UTF8String]);
    if (!_trainingManagerPtr) {
        ModelManager_Destroy(_modelManagerPtr);
        _modelManagerPtr = NULL;
        
        if (error) {
            *error = [M3TMUtils errorWithCode:3003 message:@"TrainingManager oluşturulamadı"];
        }
        return NO;
    }
    
    _searchManagerPtr = SearchManager_Create([modelBasePath UTF8String], [indexPath UTF8String]);
    if (!_searchManagerPtr) {
        ModelManager_Destroy(_modelManagerPtr);
        TrainingManager_Destroy(_trainingManagerPtr);
        _modelManagerPtr = NULL;
        _trainingManagerPtr = NULL;
        
        if (error) {
            *error = [M3TMUtils errorWithCode:3004 message:@"SearchManager oluşturulamadı"];
        }
        return NO;
    }
    
    _initialized = YES;
    return YES;
}

- (nullable M3TMModel *)loadModel:(NSString *)modelId version:(nullable NSString *)version error:(NSError **)error {
    if (!_initialized) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3005 message:@"SDK başlatılmamış"];
        }
        return nil;
    }
    
    // Model zaten yüklenmişse onu döndür
    NSString *key = version ? [NSString stringWithFormat:@"%@_%@", modelId, version] : modelId;
    M3TMModel *existingModel = _loadedModels[key];
    if (existingModel && !existingModel.isClosed) {
        return existingModel;
    }
    
    int64_t modelHandle = ModelManager_LoadModel(_modelManagerPtr, [modelId UTF8String], [version UTF8String]);
    if (modelHandle == 0) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3006 message:@"Model yüklenemedi"];
        }
        return nil;
    }
    
    // Model bilgilerini oluştur
    M3TMModelInfo *modelInfo = [[M3TMModelInfo alloc] initWithModelId:modelId 
                                                              version:version ?: @"latest" 
                                                       supportedTasks:@[@"classification", @"embedding"] 
                                                  supportedModalities:@[@"text", @"image"]];
    
    M3TMModel *model = [[M3TMModel alloc] initWithManager:_modelManagerPtr modelHandle:modelHandle modelInfo:modelInfo];
    _loadedModels[key] = model;
    
    return model;
}

- (nullable M3TMTrainingSession *)createTrainingSession:(M3TMModel *)model 
                                                   task:(NSString *)task 
                                                options:(nullable NSDictionary<NSString *, NSObject *> *)options 
                                                  error:(NSError **)error {
    if (!_initialized) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3005 message:@"SDK başlatılmamış"];
        }
        return nil;
    }
    
    if (!model || model.isClosed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3007 message:@"Geçersiz model"];
        }
        return nil;
    }
    
    // Model üzerine reflection ile modelHandle'ı almak gerekiyor
    // Bu örnek için basit bir yaklaşım kullanıyoruz
    int64_t modelHandle = 0; // model._modelHandle değerini al
    
    void *optionsPtr = [M3TMUtils nativePtrFromDictionary:options];
    const char *sessionIdCStr = TrainingManager_CreateSession(_trainingManagerPtr, modelHandle, [task UTF8String], optionsPtr);
    
    if (!sessionIdCStr) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3008 message:@"Eğitim oturumu oluşturulamadı"];
        }
        return nil;
    }
    
    NSString *sessionId = [NSString stringWithUTF8String:sessionIdCStr];
    M3TMTrainingSession *session = [[M3TMTrainingSession alloc] initWithManager:_trainingManagerPtr sessionId:sessionId];
    _trainingSessions[sessionId] = session;
    
    return session;
}

- (nullable NSArray<M3TMSearchResult *> *)searchWithText:(M3TMModel *)model 
                                                   query:(NSString *)query 
                                                 options:(nullable NSDictionary<NSString *, NSObject *> *)options 
                                                   error:(NSError **)error {
    if (!_initialized) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3005 message:@"SDK başlatılmamış"];
        }
        return nil;
    }
    
    if (!model || model.isClosed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3007 message:@"Geçersiz model"];
        }
        return nil;
    }
    
    // Model üzerine reflection ile modelHandle'ı almak gerekiyor
    int64_t modelHandle = 0; // model._modelHandle değerini al
    
    void *optionsPtr = [M3TMUtils nativePtrFromDictionary:options];
    void *resultsPtr = SearchManager_SearchText(_searchManagerPtr, modelHandle, [query UTF8String], optionsPtr);
    
    if (!resultsPtr) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3009 message:@"Arama yapılamadı"];
        }
        return nil;
    }
    
    return [M3TMUtils searchResultsFromNativePtr:resultsPtr];
}

- (nullable NSArray<M3TMSearchResult *> *)searchWithImage:(M3TMModel *)model 
                                                    image:(UIImage *)image 
                                                  options:(nullable NSDictionary<NSString *, NSObject *> *)options 
                                                    error:(NSError **)error {
    if (!_initialized) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3005 message:@"SDK başlatılmamış"];
        }
        return nil;
    }
    
    if (!model || model.isClosed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3007 message:@"Geçersiz model"];
        }
        return nil;
    }
    
    // Model üzerine reflection ile modelHandle'ı almak gerekiyor
    int64_t modelHandle = 0; // model._modelHandle değerini al
    
    NSData *imageData = [M3TMUtils dataFromUIImage:image];
    CGImageRef cgImage = image.CGImage;
    NSUInteger width = CGImageGetWidth(cgImage);
    NSUInteger height = CGImageGetHeight(cgImage);
    NSUInteger channels = 4; // RGBA
    
    void *optionsPtr = [M3TMUtils nativePtrFromDictionary:options];
    void *resultsPtr = SearchManager_SearchImage(_searchManagerPtr, modelHandle, [imageData bytes], (int)width, (int)height, (int)channels, optionsPtr);
    
    if (!resultsPtr) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3010 message:@"Görüntü araması yapılamadı"];
        }
        return nil;
    }
    
    return [M3TMUtils searchResultsFromNativePtr:resultsPtr];
}

- (BOOL)indexText:(M3TMModel *)model 
             text:(NSString *)text 
         metadata:(nullable NSDictionary<NSString *, NSString *> *)metadata 
            error:(NSError **)error {
    if (!_initialized) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3005 message:@"SDK başlatılmamış"];
        }
        return NO;
    }
    
    if (!model || model.isClosed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3007 message:@"Geçersiz model"];
        }
        return NO;
    }
    
    // Model üzerine reflection ile modelHandle'ı almak gerekiyor
    int64_t modelHandle = 0; // model._modelHandle değerini al
    
    void *metadataPtr = [M3TMUtils nativePtrFromDictionary:metadata];
    BOOL result = SearchManager_IndexText(_searchManagerPtr, modelHandle, [text UTF8String], metadataPtr);
    
    if (!result && error) {
        *error = [M3TMUtils errorWithCode:3011 message:@"Metin indekslenemedi"];
    }
    
    return result;
}

- (BOOL)indexImage:(M3TMModel *)model 
            image:(UIImage *)image 
         metadata:(nullable NSDictionary<NSString *, NSString *> *)metadata 
            error:(NSError **)error {
    if (!_initialized) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3005 message:@"SDK başlatılmamış"];
        }
        return NO;
    }
    
    if (!model || model.isClosed) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3007 message:@"Geçersiz model"];
        }
        return NO;
    }
    
    // Model üzerine reflection ile modelHandle'ı almak gerekiyor
    int64_t modelHandle = 0; // model._modelHandle değerini al
    
    NSData *imageData = [M3TMUtils dataFromUIImage:image];
    CGImageRef cgImage = image.CGImage;
    NSUInteger width = CGImageGetWidth(cgImage);
    NSUInteger height = CGImageGetHeight(cgImage);
    NSUInteger channels = 4; // RGBA
    
    void *metadataPtr = [M3TMUtils nativePtrFromDictionary:metadata];
    BOOL result = SearchManager_IndexImage(_searchManagerPtr, modelHandle, [imageData bytes], (int)width, (int)height, (int)channels, metadataPtr);
    
    if (!result && error) {
        *error = [M3TMUtils errorWithCode:3012 message:@"Görüntü indekslenemedi"];
    }
    
    return result;
}

- (BOOL)saveSearchIndexWithError:(NSError **)error {
    if (!_initialized) {
        if (error) {
            *error = [M3TMUtils errorWithCode:3005 message:@"SDK başlatılmamış"];
        }
        return NO;
    }
    
    BOOL result = SearchManager_SaveIndex(_searchManagerPtr);
    
    if (!result && error) {
        *error = [M3TMUtils errorWithCode:3013 message:@"Arama indeksi kaydedilemedi"];
    }
    
    return result;
}

+ (NSString *)version {
    return @"0.1.0";
}

- (void)shutdown {
    if (_initialized) {
        // Tüm oturumları kapat
        for (M3TMTrainingSession *session in _trainingSessions.allValues) {
            [session close];
        }
        [_trainingSessions removeAllObjects];
        
        // Tüm modelleri kapat
        for (M3TMModel *model in _loadedModels.allValues) {
            [model close];
        }
        [_loadedModels removeAllObjects];
        
        // Manager'ları yok et
        if (_searchManagerPtr) {
            SearchManager_Destroy(_searchManagerPtr);
            _searchManagerPtr = NULL;
        }
        
        if (_trainingManagerPtr) {
            TrainingManager_Destroy(_trainingManagerPtr);
            _trainingManagerPtr = NULL;
        }
        
        if (_modelManagerPtr) {
            ModelManager_Destroy(_modelManagerPtr);
            _modelManagerPtr = NULL;
        }
        
        _initialized = NO;
    }
}

- (void)dealloc {
    [self shutdown];
}

@end 