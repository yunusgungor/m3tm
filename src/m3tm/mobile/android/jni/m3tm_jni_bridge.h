/**
 * M³TM v2.3 - Android SDK JNI Köprüsü
 * 
 * Java Native Interface köprüsü için C++ header dosyası.
 * Bu dosya, Java/Kotlin API'leri ile PyTorch/Python implementasyonu 
 * arasındaki köprüyü tanımlar.
 */

#ifndef M3TM_JNI_BRIDGE_H
#define M3TM_JNI_BRIDGE_H

#include <jni.h>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

// Python erişimi için gerekli başlıklar
#include <Python.h>
#include <torch/script.h>

namespace m3tm {
namespace jni {

/**
 * Python ve JNI arasında köprü görevi gören sınıf.
 * Singleton tasarım desenini kullanır.
 */
class M3TMJniBridge {
public:
    /**
     * Singleton örneği döndürür, gerekirse oluşturur.
     */
    static M3TMJniBridge& getInstance();
    
    /**
     * JNI çevresi ile köprüyü başlatır.
     * 
     * @param env JNI çevresi
     * @param javaVM Java VM referansı
     * @return başlatma başarılı mı
     */
    bool initialize(JNIEnv* env, JavaVM* javaVM);
    
    /**
     * Köprü kaynakları serbest bırakır.
     */
    void shutdown();
    
    /**
     * Python modülünü yükler ve başlatır.
     * 
     * @param modulePath Python modülünün yolu
     * @return yükleme başarılı mı
     */
    bool loadPythonModule(const std::string& modulePath);
    
    /**
     * JNIEnv referansını döndürür.
     */
    JNIEnv* getEnv();
    
    /**
     * Java VM referansını döndürür.
     */
    JavaVM* getJavaVM();
    
    /**
     * Python modülü yüklü mü kontrol eder.
     */
    bool isPythonModuleLoaded() const;
    
    /**
     * Python nesne referansını döndürür.
     */
    PyObject* getPythonModule() const;
    
    /**
     * Python metodunu çağırır ve sonucu jobject olarak döndürür.
     * 
     * @param methodName çağrılacak Python metod adı
     * @param args JNI argümanları
     * @return JNI cevabı
     */
    jobject callPythonMethod(const std::string& methodName, const std::vector<jobject>& args);
    
    /**
     * Java istisnası fırlatır.
     * 
     * @param exceptionClass istisna sınıfı
     * @param message hata mesajı
     */
    void throwJavaException(const std::string& exceptionClass, const std::string& message);

private:
    // Singleton için private constructor
    M3TMJniBridge();
    ~M3TMJniBridge();
    
    // Copy constructor ve atama operatörünü devre dışı bırak
    M3TMJniBridge(const M3TMJniBridge&) = delete;
    M3TMJniBridge& operator=(const M3TMJniBridge&) = delete;
    
    // Python ve JNI çevresi referansları
    PyObject* m_pythonModule;
    JavaVM* m_javaVM;
    
    // Python thread durumu
    PyGILState_STATE m_gilState;
    
    // Python ve JNI arasında veri dönüşümü için yardımcı metotlar
    jobject pythonToJava(JNIEnv* env, PyObject* pyObj);
    PyObject* javaToPython(JNIEnv* env, jobject jObj);
    
    // Başlatma ve temizleme yardımcı metotları
    bool initializePython();
    void cleanupPython();
};

// JNI fonksiyon deklarasyonları

#ifdef __cplusplus
extern "C" {
#endif

// Sistem fonksiyonları
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved);
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved);

// ModelManager API
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModelManager_nativeLoadModel(JNIEnv* env, jobject thiz, jstring modelId, jstring version);
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModelManager_nativeListAvailableModels(JNIEnv* env, jobject thiz);
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModelManager_nativeGetModelInfo(JNIEnv* env, jobject thiz, jstring modelId);
JNIEXPORT jboolean JNICALL Java_com_m3tm_sdk_M3TMModelManager_nativeIsModelAvailable(JNIEnv* env, jobject thiz, jstring modelId);

// Model API
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModel_nativeProcessText(JNIEnv* env, jobject thiz, jstring modelId, jstring text, jstring task, jobject options);
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModel_nativeProcessImage(JNIEnv* env, jobject thiz, jstring modelId, jobject imageData, jstring task, jobject options);
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModel_nativeProcessMultimodal(JNIEnv* env, jobject thiz, jstring modelId, jstring text, jobject imageData, jstring task, jobject options);
JNIEXPORT jboolean JNICALL Java_com_m3tm_sdk_M3TMModel_nativeCloseModel(JNIEnv* env, jobject thiz, jstring modelId);

// TrainingManager API
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeCreateTrainingSession(JNIEnv* env, jobject thiz, jstring modelId, jobject trainingConfig);
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeStartTraining(JNIEnv* env, jobject thiz, jstring sessionId, jobject trainingData, jobject callback);
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeGetTrainingStatus(JNIEnv* env, jobject thiz, jstring sessionId);
JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeSaveTrainedModel(JNIEnv* env, jobject thiz, jstring sessionId, jstring outputPath, jstring modelName);
JNIEXPORT jboolean JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeDeleteTrainingSession(JNIEnv* env, jobject thiz, jstring sessionId);
JNIEXPORT jint JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeCleanExpiredSessions(JNIEnv* env, jobject thiz);

#ifdef __cplusplus
}
#endif

} // namespace jni
} // namespace m3tm

#endif // M3TM_JNI_BRIDGE_H 