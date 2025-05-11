package com.m3tm.sdk;

import java.util.HashMap;
import java.util.Map;

/**
 * Model eğitimi ve ince ayar işlemleri için işlevler sağlayan sınıf.
 */
public class M3TMTrainingManager {
    private static final String DEFAULT_CACHE_DIR = "m3tm_models";
    private final String modelCacheDir;
    
    // JNI tarafından kullanılan native işaretçi (pointer)
    private long nativeHandle;
    
    static {
        System.loadLibrary("m3tm_sdk_jni");
    }

    /**
     * M3TMTrainingManager sınıfını başlatır.
     */
    public M3TMTrainingManager() {
        this(null);
    }

    /**
     * M3TMTrainingManager sınıfını başlatır.
     *
     * @param modelCacheDir Model önbellek dizini (opsiyonel)
     */
    public M3TMTrainingManager(String modelCacheDir) {
        this.modelCacheDir = modelCacheDir != null ? modelCacheDir : DEFAULT_CACHE_DIR;
        
        this.nativeHandle = nativeInitialize(this.modelCacheDir);
        if (this.nativeHandle == 0) {
            throw new RuntimeException("TrainingManager başlatılamadı");
        }
    }

    /**
     * Eğitim oturumu oluşturur.
     *
     * @param modelId         Eğitilecek modelin benzersiz tanımlayıcısı
     * @param trainingConfig  Eğitim yapılandırması
     * @return Oluşturulan oturum kimliği
     * @throws TrainingException Oturum oluşturulamadığında
     */
    public String createTrainingSession(String modelId, Map<String, Object> trainingConfig) throws TrainingException {
        if (modelId == null || modelId.isEmpty()) {
            throw new TrainingException("Model ID boş olamaz");
        }
        
        if (trainingConfig == null) {
            trainingConfig = new HashMap<>();
        }
        
        try {
            Map<String, Object> result = nativeCreateTrainingSession(nativeHandle, modelId, trainingConfig);
            
            if (result == null || !result.containsKey("session_id")) {
                throw new TrainingException("Eğitim oturumu oluşturulamadı");
            }
            
            return (String) result.get("session_id");
        } catch (Exception e) {
            throw new TrainingException("Eğitim oturumu oluşturulamadı: " + e.getMessage(), e);
        }
    }

    /**
     * Eğitim oturumunu başlatır.
     *
     * @param sessionId    Eğitim oturumu kimliği
     * @param trainingData Eğitim veri setleri
     * @return Eğitim sonuçları
     * @throws TrainingException Eğitim sırasında hata oluştuğunda
     */
    public Map<String, Object> startTraining(String sessionId, Map<String, Object> trainingData) throws TrainingException {
        return startTraining(sessionId, trainingData, null);
    }

    /**
     * Eğitim oturumunu başlatır.
     *
     * @param sessionId    Eğitim oturumu kimliği
     * @param trainingData Eğitim veri setleri
     * @param callback     Eğitim ilerlemesini bildirmek için isteğe bağlı callback
     * @return Eğitim sonuçları
     * @throws TrainingException Eğitim sırasında hata oluştuğunda
     */
    public Map<String, Object> startTraining(String sessionId, Map<String, Object> trainingData, TrainingCallback callback) throws TrainingException {
        if (sessionId == null || sessionId.isEmpty()) {
            throw new TrainingException("Oturum ID boş olamaz");
        }
        
        if (trainingData == null) {
            throw new TrainingException("Eğitim verileri boş olamaz");
        }
        
        try {
            return nativeStartTraining(nativeHandle, sessionId, trainingData, callback);
        } catch (Exception e) {
            throw new TrainingException("Eğitim başlatılamadı: " + e.getMessage(), e);
        }
    }

    /**
     * Eğitim oturumunun durumunu alır.
     *
     * @param sessionId Eğitim oturumu kimliği
     * @return Oturum durumu
     * @throws TrainingException Durum alınamadığında
     */
    public Map<String, Object> getTrainingStatus(String sessionId) throws TrainingException {
        if (sessionId == null || sessionId.isEmpty()) {
            throw new TrainingException("Oturum ID boş olamaz");
        }
        
        try {
            Map<String, Object> status = nativeGetTrainingStatus(nativeHandle, sessionId);
            
            if (status == null) {
                throw new TrainingException("Eğitim durumu alınamadı");
            }
            
            return status;
        } catch (Exception e) {
            throw new TrainingException("Eğitim durumu alınamadı: " + e.getMessage(), e);
        }
    }

    /**
     * Eğitilmiş modeli kaydeder.
     *
     * @param sessionId  Eğitim oturumu kimliği
     * @param outputPath Kaydedilecek dizin (opsiyonel)
     * @param modelName  Kaydedilecek model adı (opsiyonel)
     * @return Kaydetme işlemi bilgisi
     * @throws TrainingException Model kaydedilemediğinde
     */
    public Map<String, Object> saveTrainedModel(String sessionId, String outputPath, String modelName) throws TrainingException {
        if (sessionId == null || sessionId.isEmpty()) {
            throw new TrainingException("Oturum ID boş olamaz");
        }
        
        try {
            Map<String, Object> saveInfo = nativeSaveTrainedModel(nativeHandle, sessionId, outputPath, modelName);
            
            if (saveInfo == null) {
                throw new TrainingException("Eğitilmiş model kaydedilemedi");
            }
            
            return saveInfo;
        } catch (Exception e) {
            throw new TrainingException("Eğitilmiş model kaydedilemedi: " + e.getMessage(), e);
        }
    }

    /**
     * Eğitim oturumunu siler.
     *
     * @param sessionId Eğitim oturumu kimliği
     * @return İşlemin başarı durumu
     */
    public boolean deleteTrainingSession(String sessionId) {
        if (sessionId == null || sessionId.isEmpty()) {
            return false;
        }
        
        try {
            return nativeDeleteTrainingSession(nativeHandle, sessionId);
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Süresi dolmuş oturumları temizler.
     *
     * @return Temizlenen oturum sayısı
     */
    public int cleanExpiredSessions() {
        try {
            Map<String, Object> result = nativeCleanExpiredSessions(nativeHandle);
            
            if (result != null && result.containsKey("cleaned_count")) {
                return ((Number) result.get("cleaned_count")).intValue();
            }
            
            return 0;
        } catch (Exception e) {
            return 0;
        }
    }

    /**
     * Kaynakları serbest bırakır.
     */
    public synchronized void close() {
        if (nativeHandle != 0) {
            nativeClose(nativeHandle);
            nativeHandle = 0;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    // Native metotlar
    private native long nativeInitialize(String modelCacheDir);
    private native Map<String, Object> nativeCreateTrainingSession(long handle, String modelId, Map<String, Object> trainingConfig);
    private native Map<String, Object> nativeStartTraining(long handle, String sessionId, Map<String, Object> trainingData, TrainingCallback callback);
    private native Map<String, Object> nativeGetTrainingStatus(long handle, String sessionId);
    private native Map<String, Object> nativeSaveTrainedModel(long handle, String sessionId, String outputPath, String modelName);
    private native boolean nativeDeleteTrainingSession(long handle, String sessionId);
    private native Map<String, Object> nativeCleanExpiredSessions(long handle);
    private native void nativeClose(long handle);
} 