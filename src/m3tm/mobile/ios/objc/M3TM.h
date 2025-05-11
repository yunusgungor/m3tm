#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * M3TM SDK hata sınıfı
 */
@interface M3TMError : NSError

/**
 * M3TM hata kodu alanı
 */
extern NSErrorDomain const M3TMErrorDomain;

/**
 * M3TM hata kodları
 */
typedef NS_ENUM(NSInteger, M3TMErrorCode) {
    M3TMErrorCodeModelNotFound = 1000,
    M3TMErrorCodeModelLoadFailed = 1001,
    M3TMErrorCodeInvalidInput = 1002,
    M3TMErrorCodeProcessingFailed = 1003,
    M3TMErrorCodeTrainingFailed = 1004,
    M3TMErrorCodeInvalidSession = 1005,
    M3TMErrorCodeInvalidOperation = 1006,
    M3TMErrorCodeSearchFailed = 1007,
    M3TMErrorCodeInternalError = 9999
};

/**
 * Belirtilen hata kodu ve açıklama ile bir hata oluşturur
 */
+ (instancetype)errorWithCode:(M3TMErrorCode)code description:(NSString *)description;

@end

/**
 * Model meta bilgilerini içeren sınıf
 */
@interface M3TMModelInfo : NSObject

/**
 * Model benzersiz tanımlayıcısı
 */
@property (nonatomic, readonly) NSString *modelId;

/**
 * Model versiyonu
 */
@property (nonatomic, readonly) NSString *version;

/**
 * Model ismi
 */
@property (nonatomic, readonly) NSString *name;

/**
 * Model açıklaması
 */
@property (nonatomic, readonly) NSString *description;

/**
 * Ek model özellikleri
 */
@property (nonatomic, readonly) NSDictionary<NSString *, id> *properties;

/**
 * JSON veriden model bilgisi oluşturur
 */
+ (nullable instancetype)modelInfoFromJSON:(NSString *)jsonString;

/**
 * Model bilgisini JSON formatına dönüştürür
 */
- (NSString *)toJSON;

@end

/**
 * M3TM Model işlemlerini gerçekleştiren ana sınıf
 */
@interface M3TMModel : NSObject

/**
 * Model bilgisini döndürür
 */
@property (nonatomic, readonly) M3TMModelInfo *modelInfo;

/**
 * Model ID'sini döndürür
 */
@property (nonatomic, readonly) NSString *modelId;

/**
 * Yeni içerik oluşturma yok, fabrika metodları kullanılmalı
 */
- (instancetype)init NS_UNAVAILABLE;

/**
 * Metin işler ve sonuçları döndürür
 *
 * @param text İşlenecek metin
 * @param task Çalıştırılacak görev (opsiyonel)
 * @param options Ek görev opsiyonları (opsiyonel)
 * @param error Oluşabilecek hata
 * @return İşleme sonuçlarını içeren sözlük
 */
- (nullable NSDictionary *)processText:(NSString *)text
                                  task:(nullable NSString *)task
                               options:(nullable NSDictionary *)options
                                 error:(NSError **)error;

/**
 * Görüntü işler ve sonuçları döndürür
 *
 * @param image İşlenecek görüntü
 * @param task Çalıştırılacak görev (opsiyonel)
 * @param options Ek görev opsiyonları (opsiyonel)
 * @param error Oluşabilecek hata
 * @return İşleme sonuçlarını içeren sözlük
 */
- (nullable NSDictionary *)processImage:(UIImage *)image
                                   task:(nullable NSString *)task
                                options:(nullable NSDictionary *)options
                                  error:(NSError **)error;

/**
 * Metin ve görüntüyü birlikte işler (multimodal)
 *
 * @param text İşlenecek metin (opsiyonel)
 * @param image İşlenecek görüntü (opsiyonel)
 * @param task Çalıştırılacak görev (opsiyonel)
 * @param options Ek görev opsiyonları (opsiyonel)
 * @param error Oluşabilecek hata
 * @return İşleme sonuçlarını içeren sözlük
 */
- (nullable NSDictionary *)processMultimodal:(nullable NSString *)text
                                       image:(nullable UIImage *)image
                                        task:(nullable NSString *)task
                                     options:(nullable NSDictionary *)options
                                       error:(NSError **)error;

/**
 * Modeli kapatır ve kaynakları serbest bırakır
 *
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)close:(NSError **)error;

@end

/**
 * Model yönetimini sağlayan sınıf
 */
@interface M3TMModelManager : NSObject

/**
 * Varsayılan yapılandırma ile model yöneticisi oluşturur
 */
+ (instancetype)defaultManager;

/**
 * Belirtilen model önbellek dizini ile model yöneticisi oluşturur
 *
 * @param modelCacheDirectory Model önbellek dizini
 */
- (instancetype)initWithModelCacheDirectory:(nullable NSString *)modelCacheDirectory;

/**
 * Belirtilen modeli yükler
 *
 * @param modelId Yüklenecek model ID'si
 * @param version Model versiyonu (opsiyonel, varsayılan en son versiyon)
 * @param error Oluşabilecek hata
 * @return Yüklenen model
 */
- (nullable M3TMModel *)loadModel:(NSString *)modelId
                          version:(nullable NSString *)version
                            error:(NSError **)error;

/**
 * Mevcut tüm modelleri listeler
 *
 * @param error Oluşabilecek hata
 * @return Model bilgilerini içeren dizi
 */
- (nullable NSArray<M3TMModelInfo *> *)listAvailableModels:(NSError **)error;

/**
 * Belirtilen modelin bilgilerini döndürür
 *
 * @param modelId Bilgileri istenilen model ID'si
 * @param error Oluşabilecek hata
 * @return Model bilgisi
 */
- (nullable M3TMModelInfo *)getModelInfo:(NSString *)modelId error:(NSError **)error;

/**
 * Belirtilen modelin mevcut olup olmadığını kontrol eder
 *
 * @param modelId Kontrol edilecek model ID'si
 * @return Modelin mevcut olup olmadığı
 */
- (BOOL)isModelAvailable:(NSString *)modelId;

@end

/**
 * Eğitim ilerlemesini izleme protokolü
 */
@protocol M3TMTrainingCallback <NSObject>

@optional
/**
 * Her batch tamamlandığında çağrılır
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param batch Tamamlanan batch numarası
 * @param metrics Batch metrikleri
 */
- (void)onBatchCompleted:(NSString *)sessionId batch:(NSInteger)batch metrics:(NSDictionary *)metrics;

/**
 * Her epoch tamamlandığında çağrılır
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param epoch Tamamlanan epoch numarası
 * @param metrics Epoch metrikleri
 */
- (void)onEpochCompleted:(NSString *)sessionId epoch:(NSInteger)epoch metrics:(NSDictionary *)metrics;

/**
 * Eğitim tamamlandığında çağrılır
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param metrics Eğitim sonuç metrikleri
 */
- (void)onTrainingCompleted:(NSString *)sessionId metrics:(NSDictionary *)metrics;

/**
 * Eğitim sırasında hata oluştuğunda çağrılır
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param error Oluşan hata
 */
- (void)onTrainingError:(NSString *)sessionId error:(M3TMError *)error;

@end

/**
 * Eğitim yönetimini sağlayan sınıf
 */
@interface M3TMTrainingManager : NSObject

/**
 * Varsayılan yapılandırma ile eğitim yöneticisi oluşturur
 */
+ (instancetype)defaultManager;

/**
 * Belirtilen model önbellek dizini ile eğitim yöneticisi oluşturur
 *
 * @param modelCacheDirectory Model önbellek dizini
 */
- (instancetype)initWithModelCacheDirectory:(nullable NSString *)modelCacheDirectory;

/**
 * Yeni bir eğitim oturumu oluşturur
 *
 * @param modelId Eğitilecek model ID'si
 * @param trainingConfig Eğitim yapılandırması
 * @param error Oluşabilecek hata
 * @return Eğitim oturumu ID'si
 */
- (nullable NSString *)createTrainingSession:(NSString *)modelId
                             trainingConfig:(NSDictionary *)trainingConfig
                                     error:(NSError **)error;

/**
 * Eğitim oturumuna metin örneği ekler
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param text Metin içeriği
 * @param label Etiket (sınıflandırma için)
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)addTextSample:(NSString *)sessionId
                 text:(NSString *)text
                label:(NSString *)label
                error:(NSError **)error;

/**
 * Eğitim oturumuna görüntü örneği ekler
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param image Görüntü
 * @param label Etiket (sınıflandırma için)
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)addImageSample:(NSString *)sessionId
                 image:(UIImage *)image
                 label:(NSString *)label
                 error:(NSError **)error;

/**
 * Eğitimi başlatır
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param epochs Epoch sayısı
 * @param callback Eğitim geri çağrı protokolü
 * @param error Oluşabilecek hata
 * @return İlk durum bilgisi
 */
- (nullable NSDictionary *)startTraining:(NSString *)sessionId
                                  epochs:(NSInteger)epochs
                                callback:(nullable id<M3TMTrainingCallback>)callback
                                   error:(NSError **)error;

/**
 * Eğitimi iptal eder
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)cancelTraining:(NSString *)sessionId error:(NSError **)error;

/**
 * Eğitim durumunu alır
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param error Oluşabilecek hata
 * @return Durum bilgisi
 */
- (nullable NSDictionary *)getTrainingStatus:(NSString *)sessionId error:(NSError **)error;

/**
 * Eğitilmiş modeli kaydeder
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param outputPath Çıktı dizini (opsiyonel)
 * @param modelName Model ismi (opsiyonel)
 * @param error Oluşabilecek hata
 * @return Kayıt sonuç bilgisi
 */
- (nullable NSDictionary *)saveTrainedModel:(NSString *)sessionId
                                 outputPath:(nullable NSString *)outputPath
                                 modelName:(nullable NSString *)modelName
                                     error:(NSError **)error;

/**
 * Eğitim oturumunu siler
 *
 * @param sessionId Eğitim oturumu ID'si
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)deleteTrainingSession:(NSString *)sessionId error:(NSError **)error;

@end

/**
 * Arama yönetimi için sınıf
 */
@interface M3TMSearchManager : NSObject

/**
 * Varsayılan yapılandırma ile arama yöneticisi oluşturur
 */
+ (instancetype)defaultManager;

/**
 * Belirtilen indeks dizini ile arama yöneticisi oluşturur
 *
 * @param indexDirectory İndeks dizini
 */
- (instancetype)initWithIndexDirectory:(nullable NSString *)indexDirectory;

/**
 * Metin sorgusu ile arama yapar
 *
 * @param modelId Kullanılacak model ID'si
 * @param query Metin sorgusu
 * @param options Arama opsiyonları
 * @param error Oluşabilecek hata
 * @return Arama sonuçları
 */
- (nullable NSArray<NSDictionary *> *)searchWithText:(NSString *)modelId
                                              query:(NSString *)query
                                            options:(nullable NSDictionary *)options
                                              error:(NSError **)error;

/**
 * Görüntü sorgusu ile arama yapar
 *
 * @param modelId Kullanılacak model ID'si
 * @param image Görüntü sorgusu
 * @param options Arama opsiyonları
 * @param error Oluşabilecek hata
 * @return Arama sonuçları
 */
- (nullable NSArray<NSDictionary *> *)searchWithImage:(NSString *)modelId
                                              image:(UIImage *)image
                                            options:(nullable NSDictionary *)options
                                              error:(NSError **)error;

/**
 * Metin içeriği indeksler
 *
 * @param modelId Kullanılacak model ID'si
 * @param text Metin içeriği
 * @param metadata İçerik meta bilgileri
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)indexText:(NSString *)modelId
             text:(NSString *)text
         metadata:(nullable NSDictionary *)metadata
            error:(NSError **)error;

/**
 * Görüntü içeriği indeksler
 *
 * @param modelId Kullanılacak model ID'si
 * @param image Görüntü içeriği
 * @param metadata İçerik meta bilgileri
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)indexImage:(NSString *)modelId
             image:(UIImage *)image
          metadata:(nullable NSDictionary *)metadata
             error:(NSError **)error;

/**
 * İndeksi kaydeder
 *
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)saveIndex:(NSError **)error;

/**
 * İndeksi yükler
 *
 * @param error Oluşabilecek hata
 * @return İşlemin başarılı olup olmadığı
 */
- (BOOL)loadIndex:(NSError **)error;

@end

/**
 * Ana SDK sınıfı
 */
@interface M3TM : NSObject

/**
 * SDK versiyonunu döndürür
 */
+ (NSString *)version;

/**
 * SDK durumunu kontrol eder
 */
+ (BOOL)isAvailable;

/**
 * Varsayılan model yöneticisini döndürür
 */
+ (M3TMModelManager *)modelManager;

/**
 * Varsayılan eğitim yöneticisini döndürür
 */
+ (M3TMTrainingManager *)trainingManager;

/**
 * Varsayılan arama yöneticisini döndürür
 */
+ (M3TMSearchManager *)searchManager;

/**
 * SDK'yı yapılandırır
 *
 * @param configuration Yapılandırma opsiyonları
 */
+ (void)configure:(NSDictionary *)configuration;

@end

NS_ASSUME_NONNULL_END 