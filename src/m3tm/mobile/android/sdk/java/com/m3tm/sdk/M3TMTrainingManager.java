package com.m3tm.sdk;

import android.util.Log;

import java.util.HashMap;
import java.util.Map;

/**
 * M³TM eğitim yönetim sınıfı.
 * 
 * Bu sınıf, modellerin eğitim/ince ayar işlevselliğini sağlar.
 */
public class M3TMTrainingManager {
    private static final String TAG = "M3TMTrainingManager";
    
    private final M3TM sdk;
    private String modelDirectory;
    private String cacheDirectory;
    private boolean isInitialized = false;
    
    /**
     * M3TMTrainingManager oluşturur.
     * 
     * @param sdk SDK referansı
     */
    protected M3TMTrainingManager(M3TM sdk) {
        this.sdk = sdk;
    }
    
    /**
     * Eğitim yöneticisini başlatır.
     * 
     * @param modelDirectory Model dizini
     * @param cacheDirectory Önbellek dizini
     * @return Başlatma başarılı oldu mu
     */
    protected boolean initialize(String modelDirectory, String cacheDirectory) {
        this.modelDirectory = modelDirectory;
        this.cacheDirectory = cacheDirectory;
        
        try {
            // Dizinlerin var olduğundan emin ol
            createDirectories();
            isInitialized = true;
            Log.i(TAG, "Eğitim yöneticisi başlatıldı");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Eğitim yöneticisi başlatma hatası: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Eğitim yöneticisini kapatır ve kaynakları serbest bırakır.
     */
    protected void shutdown() {
        try {
            // Gerekirse kaynakları temizle
            cleanExpiredSessions();
            Log.i(TAG, "Eğitim yöneticisi kapatıldı");
        } catch (Exception e) {
            Log.e(TAG, "Eğitim yöneticisi kapatma hatası: " + e.getMessage(), e);
        }
    }
    
    /**
     * Yeni bir eğitim oturumu oluşturur.
     * 
     * @param modelId Eğitilecek model ID'si
     * @param config Eğitim yapılandırması
     * @return Oturum bilgisi
     * @throws M3TMException Oturum oluşturulamazsa
     */
    public Map<String, Object> createTrainingSession(String modelId, Map<String, Object> config) throws M3TMException {
        checkInitialized();
        
        if (modelId == null || modelId.isEmpty()) {
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_ERROR, "Model ID belirtilmelidir");
        }
        
        if (config == null) {
            config = new HashMap<>();
        }
        
        try {
            Log.i(TAG, "Eğitim oturumu oluşturuluyor: " + modelId);
            
            Map<String, Object> result = nativeCreateTrainingSession(modelId, config);
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Eğitim oturumu oluşturma hatası: " + errorMessage);
                
                if (errorMessage.contains("not found")) {
                    throw new M3TMException(M3TMException.ErrorCode.MODEL_NOT_FOUND, "Model bulunamadı: " + modelId);
                }
                
                throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> sessionInfo = (Map<String, Object>) result.get("data");
            
            Log.i(TAG, "Eğitim oturumu oluşturuldu: " + sessionInfo.get("session_id"));
            return sessionInfo;
            
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Eğitim oturumu oluşturma hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_ERROR, "Eğitim oturumu oluşturulamadı", e);
        }
    }
    
    /**
     * Eğitim oturumunu başlatır.
     * 
     * @param sessionId Oturum ID'si
     * @param trainingData Eğitim verileri
     * @return Eğitim sonuçları
     * @throws M3TMException Eğitim başlatılamazsa
     */
    public Map<String, Object> startTraining(String sessionId, Map<String, Object> trainingData) throws M3TMException {
        return startTraining(sessionId, trainingData, null);
    }
    
    /**
     * Eğitim oturumunu başlatır.
     * 
     * @param sessionId Oturum ID'si
     * @param trainingData Eğitim verileri
     * @param callback İlerleme geri çağrısı
     * @return Eğitim sonuçları
     * @throws M3TMException Eğitim başlatılamazsa
     */
    public Map<String, Object> startTraining(String sessionId, Map<String, Object> trainingData, M3TMTrainingCallback callback) throws M3TMException {
        checkInitialized();
        
        if (sessionId == null || sessionId.isEmpty()) {
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_ERROR, "Oturum ID belirtilmelidir");
        }
        
        if (trainingData == null) {
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_ERROR, "Eğitim verileri belirtilmelidir");
        }
        
        try {
            Log.i(TAG, "Eğitim başlatılıyor: " + sessionId);
            
            // Callback adaptörünü oluştur
            TrainingCallbackAdapter callbackAdapter = callback != null ? new TrainingCallbackAdapter(callback) : null;
            
            Map<String, Object> result = nativeStartTraining(sessionId, trainingData, callbackAdapter);
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Eğitim başlatma hatası: " + errorMessage);
                
                if (errorMessage.contains("not found")) {
                    throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_NOT_FOUND, "Eğitim oturumu bulunamadı: " + sessionId);
                } else if (errorMessage.contains("expired")) {
                    throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_EXPIRED, "Eğitim oturumunun süresi doldu: " + sessionId);
                }
                
                throw new M3TMException(M3TMException.ErrorCode.TRAINING_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> trainingResult = (Map<String, Object>) result.get("data");
            
            Log.i(TAG, "Eğitim tamamlandı: " + sessionId);
            return trainingResult;
            
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Eğitim başlatma hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_ERROR, "Eğitim başlatılamadı", e);
        }
    }
    
    /**
     * Eğitim oturumunun durumunu alır.
     * 
     * @param sessionId Oturum ID'si
     * @return Oturum durumu
     * @throws M3TMException Durum alınamazsa
     */
    public Map<String, Object> getTrainingStatus(String sessionId) throws M3TMException {
        checkInitialized();
        
        if (sessionId == null || sessionId.isEmpty()) {
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_ERROR, "Oturum ID belirtilmelidir");
        }
        
        try {
            Map<String, Object> result = nativeGetTrainingStatus(sessionId);
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Eğitim durumu alma hatası: " + errorMessage);
                
                if (errorMessage.contains("not found")) {
                    throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_NOT_FOUND, "Eğitim oturumu bulunamadı: " + sessionId);
                }
                
                throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> statusInfo = (Map<String, Object>) result.get("data");
            
            return statusInfo;
            
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Eğitim durumu alma hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_ERROR, "Eğitim durumu alınamadı", e);
        }
    }
    
    /**
     * Eğitilmiş modeli kaydeder.
     * 
     * @param sessionId Oturum ID'si
     * @return Kaydetme bilgisi
     * @throws M3TMException Model kaydedilemezse
     */
    public Map<String, Object> saveTrainedModel(String sessionId) throws M3TMException {
        return saveTrainedModel(sessionId, null, null);
    }
    
    /**
     * Eğitilmiş modeli belirtilen yola kaydeder.
     * 
     * @param sessionId Oturum ID'si
     * @param outputPath Çıktı yolu
     * @param modelName Model adı
     * @return Kaydetme bilgisi
     * @throws M3TMException Model kaydedilemezse
     */
    public Map<String, Object> saveTrainedModel(String sessionId, String outputPath, String modelName) throws M3TMException {
        checkInitialized();
        
        if (sessionId == null || sessionId.isEmpty()) {
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_ERROR, "Oturum ID belirtilmelidir");
        }
        
        try {
            Log.i(TAG, "Eğitilmiş model kaydediliyor: " + sessionId);
            
            Map<String, Object> result = nativeSaveTrainedModel(sessionId, outputPath, modelName);
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Model kaydetme hatası: " + errorMessage);
                
                if (errorMessage.contains("not found")) {
                    throw new M3TMException(M3TMException.ErrorCode.TRAINING_SESSION_NOT_FOUND, "Eğitim oturumu bulunamadı: " + sessionId);
                } else if (errorMessage.contains("not completed")) {
                    throw new M3TMException(M3TMException.ErrorCode.TRAINING_ERROR, "Eğitim tamamlanmadı: " + sessionId);
                }
                
                throw new M3TMException(M3TMException.ErrorCode.TRAINING_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> saveInfo = (Map<String, Object>) result.get("data");
            
            Log.i(TAG, "Eğitilmiş model kaydedildi: " + saveInfo.get("model_id") + " v" + saveInfo.get("version"));
            return saveInfo;
            
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Model kaydetme hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.TRAINING_ERROR, "Eğitilmiş model kaydedilemedi", e);
        }
    }
    
    /**
     * Eğitim oturumunu siler.
     * 
     * @param sessionId Oturum ID'si
     * @return İşlem başarılı oldu mu
     */
    public boolean deleteTrainingSession(String sessionId) {
        checkInitialized();
        
        if (sessionId == null || sessionId.isEmpty()) {
            return false;
        }
        
        try {
            return nativeDeleteTrainingSession(sessionId);
        } catch (Exception e) {
            Log.e(TAG, "Oturum silme hatası: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Süresi dolmuş oturumları temizler.
     * 
     * @return Temizlenen oturum sayısı
     */
    public int cleanExpiredSessions() {
        checkInitialized();
        
        try {
            int count = nativeCleanExpiredSessions();
            Log.i(TAG, count + " süresi dolmuş oturum temizlendi");
            return count;
        } catch (Exception e) {
            Log.e(TAG, "Oturum temizleme hatası: " + e.getMessage(), e);
            return 0;
        }
    }
    
    /**
     * Gerekli dizinleri oluşturur.
     */
    private void createDirectories() {
        // Bu metot Java tarafında dizinleri oluşturacak
        // İlerde implementasyon eklenecek
    }
    
    /**
     * Başlatılmış olup olmadığını kontrol eder.
     * 
     * @throws M3TMException Başlatılmamışsa
     */
    private void checkInitialized() {
        if (!isInitialized) {
            throw new M3TMException(M3TMException.ErrorCode.NOT_INITIALIZED, "Eğitim yöneticisi henüz başlatılmadı");
        }
    }
    
    /**
     * Java callback'ini JNI'ya geçirmek için adaptör sınıfı.
     */
    private static class TrainingCallbackAdapter {
        private final M3TMTrainingCallback callback;
        
        public TrainingCallbackAdapter(M3TMTrainingCallback callback) {
            this.callback = callback;
        }
        
        public void onBatchComplete(int batch, Map<String, Object> metrics) {
            callback.onBatchComplete(batch, metrics);
        }
        
        public void onEpochComplete(int epoch, Map<String, Object> metrics) {
            callback.onEpochComplete(epoch, metrics);
        }
        
        public void onTrainingComplete(Map<String, Object> metrics) {
            callback.onTrainingComplete(metrics);
        }
    }
    
    // Native metot çağrıları
    
    private native Map<String, Object> nativeCreateTrainingSession(String modelId, Map<String, Object> config);
    
    private native Map<String, Object> nativeStartTraining(String sessionId, Map<String, Object> trainingData, TrainingCallbackAdapter callback);
    
    private native Map<String, Object> nativeGetTrainingStatus(String sessionId);
    
    private native Map<String, Object> nativeSaveTrainedModel(String sessionId, String outputPath, String modelName);
    
    private native boolean nativeDeleteTrainingSession(String sessionId);
    
    private native int nativeCleanExpiredSessions();
} 