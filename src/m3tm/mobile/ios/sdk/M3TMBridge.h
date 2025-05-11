#ifndef M3TM_BRIDGE_H
#define M3TM_BRIDGE_H

#include <string>
#include <vector>
#include <map>
#include <memory>

/**
 * M3TM iOS C++ köprüsü.
 * Bu köprü, iOS Swift/Objective-C ile C++ arasında M³TM modelinin
 * yeteneklerine erişim sağlar.
 */
namespace m3tm {
namespace ios {

/**
 * Model yönetimi için ana sınıf.
 */
class ModelManager {
public:
    /**
     * ModelManager örneği oluşturur.
     * @param modelBasePath Model dosyalarının bulunduğu temel dizin
     * @param configPath Model yapılandırma dosyası yolu
     * @return Başarı durumu
     */
    ModelManager(const std::string& modelBasePath, const std::string& configPath);
    
    /**
     * Belirtilen modeli yükler.
     * @param modelId Yüklenecek model ID'si
     * @param version Model versiyonu
     * @return Model handle veya hata durumunda 0
     */
    int64_t loadModel(const std::string& modelId, const std::string& version);
    
    /**
     * Metin işleme yapar.
     * @param modelHandle Yüklenen model handle'ı
     * @param text İşlenecek metin
     * @param task Çalıştırılacak görev adı
     * @param options Görev için opsiyonlar
     * @return Sonuç değerleri içeren map
     */
    std::map<std::string, std::string> processText(
        int64_t modelHandle, 
        const std::string& text, 
        const std::string& task, 
        const std::map<std::string, std::string>& options
    );
    
    /**
     * Görüntü işleme yapar.
     * @param modelHandle Yüklenen model handle'ı
     * @param imageData Görüntü verileri
     * @param width Görüntü genişliği
     * @param height Görüntü yüksekliği
     * @param channels Kanal sayısı (genellikle 3 - RGB)
     * @param task Çalıştırılacak görev adı
     * @param options Görev için opsiyonlar
     * @return Sonuç değerleri içeren map
     */
    std::map<std::string, std::string> processImage(
        int64_t modelHandle,
        const unsigned char* imageData,
        int width,
        int height,
        int channels,
        const std::string& task,
        const std::map<std::string, std::string>& options
    );
    
    /**
     * Modeli kapatır ve kaynakları serbest bırakır.
     * @param modelHandle Kapatılacak model handle'ı
     * @return Başarı durumu
     */
    bool closeModel(int64_t modelHandle);
    
    /**
     * ModelManager'ı kapatır ve tüm kaynakları serbest bırakır.
     */
    ~ModelManager();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Eğitim yönetimi için sınıf.
 */
class TrainingManager {
public:
    /**
     * TrainingManager örneği oluşturur.
     * @param modelBasePath Model dosyalarının bulunduğu temel dizin
     * @param configPath Model yapılandırma dosyası yolu
     */
    TrainingManager(const std::string& modelBasePath, const std::string& configPath);
    
    /**
     * Yeni bir eğitim oturumu başlatır.
     * @param modelHandle Eğitilecek model handle'ı
     * @param task Eğitim görevi
     * @param options Eğitim opsiyonları
     * @return Eğitim oturumu ID'si veya hata durumunda boş string
     */
    std::string createTrainingSession(
        int64_t modelHandle,
        const std::string& task,
        const std::map<std::string, std::string>& options
    );
    
    /**
     * Eğitim oturumuna metin verisi ekler.
     * @param sessionId Eğitim oturumu ID'si
     * @param text Metin verisi
     * @param label Etiket (sınıflandırma için)
     * @return Başarı durumu
     */
    bool addTextSample(
        const std::string& sessionId,
        const std::string& text,
        const std::string& label
    );
    
    /**
     * Eğitim oturumuna görüntü verisi ekler.
     * @param sessionId Eğitim oturumu ID'si
     * @param imageData Görüntü verileri
     * @param width Görüntü genişliği
     * @param height Görüntü yüksekliği
     * @param channels Kanal sayısı
     * @param label Etiket (sınıflandırma için)
     * @return Başarı durumu
     */
    bool addImageSample(
        const std::string& sessionId,
        const unsigned char* imageData,
        int width,
        int height,
        int channels,
        const std::string& label
    );
    
    /**
     * Eğitimi başlatır.
     * @param sessionId Eğitim oturumu ID'si
     * @param epochs Epoch sayısı
     * @param callbackInterval Callback çağırma aralığı (batch sayısı)
     * @return Başarı durumu
     */
    bool startTraining(
        const std::string& sessionId,
        int epochs,
        int callbackInterval
    );
    
    /**
     * Eğitimi iptal eder.
     * @param sessionId Eğitim oturumu ID'si
     * @return Başarı durumu
     */
    bool cancelTraining(const std::string& sessionId);
    
    /**
     * Eğitim durumunu alır.
     * @param sessionId Eğitim oturumu ID'si
     * @return Durum bilgisi içeren map
     */
    std::map<std::string, std::string> getTrainingStatus(const std::string& sessionId);
    
    /**
     * Eğitim oturumunu kapatır.
     * @param sessionId Eğitim oturumu ID'si
     * @return Başarı durumu
     */
    bool closeTrainingSession(const std::string& sessionId);
    
    /**
     * TrainingManager'ı kapatır ve tüm kaynakları serbest bırakır.
     */
    ~TrainingManager();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Arama fonksiyonları için sınıf.
 */
class SearchManager {
public:
    /**
     * SearchManager örneği oluşturur.
     * @param modelBasePath Model dizini
     * @param indexPath İndeks dizini
     */
    SearchManager(const std::string& modelBasePath, const std::string& indexPath);
    
    /**
     * Metin sorgusu ile arama yapar.
     * @param modelHandle Model handle
     * @param query Sorgu metni
     * @param options Arama opsiyonları
     * @return Arama sonuçları
     */
    std::vector<std::map<std::string, std::string>> searchText(
        int64_t modelHandle,
        const std::string& query,
        const std::map<std::string, std::string>& options
    );
    
    /**
     * Görüntü sorgusu ile arama yapar.
     * @param modelHandle Model handle
     * @param imageData Görüntü verileri
     * @param width Görüntü genişliği
     * @param height Görüntü yüksekliği
     * @param channels Kanal sayısı
     * @param options Arama opsiyonları
     * @return Arama sonuçları
     */
    std::vector<std::map<std::string, std::string>> searchImage(
        int64_t modelHandle,
        const unsigned char* imageData,
        int width,
        int height,
        int channels,
        const std::map<std::string, std::string>& options
    );
    
    /**
     * Metin veriyi indeksler.
     * @param modelHandle Model handle
     * @param text Metin içeriği
     * @param metadata Metin hakkında ek bilgiler
     * @return Başarı durumu
     */
    bool indexText(
        int64_t modelHandle,
        const std::string& text,
        const std::map<std::string, std::string>& metadata
    );
    
    /**
     * Görüntü veriyi indeksler.
     * @param modelHandle Model handle
     * @param imageData Görüntü verileri
     * @param width Görüntü genişliği
     * @param height Görüntü yüksekliği
     * @param channels Kanal sayısı
     * @param metadata Görüntü hakkında ek bilgiler
     * @return Başarı durumu
     */
    bool indexImage(
        int64_t modelHandle,
        const unsigned char* imageData,
        int width,
        int height,
        int channels,
        const std::map<std::string, std::string>& metadata
    );
    
    /**
     * İndeksi diske kaydeder.
     * @return Başarı durumu
     */
    bool saveIndex();
    
    /**
     * İndeksi diskten yükler.
     * @return Başarı durumu
     */
    bool loadIndex();
    
    /**
     * SearchManager'ı kapatır ve tüm kaynakları serbest bırakır.
     */
    ~SearchManager();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace ios
} // namespace m3tm

#endif // M3TM_BRIDGE_H 