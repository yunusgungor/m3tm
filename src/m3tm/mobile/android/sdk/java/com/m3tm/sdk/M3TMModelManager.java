package com.m3tm.sdk;

import android.util.Log;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * M³TM model yönetim sınıfı.
 * 
 * Bu sınıf, SDK'da kullanılacak modelleri yükleme, listeleme ve yönetme işlevselliğini sağlar.
 */
public class M3TMModelManager {
    private static final String TAG = "M3TMModelManager";
    
    private final M3TM sdk;
    private final Map<String, M3TMModel> loadedModels;
    private String modelDirectory;
    private String cacheDirectory;
    private boolean isInitialized = false;
    
    /**
     * M3TMModelManager oluşturur.
     * 
     * @param sdk SDK referansı
     */
    protected M3TMModelManager(M3TM sdk) {
        this.sdk = sdk;
        this.loadedModels = new HashMap<>();
    }
    
    /**
     * Model yöneticisini başlatır.
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
            Log.i(TAG, "Model yöneticisi başlatıldı");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Model yöneticisi başlatma hatası: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Model yöneticisini kapatır ve kaynakları serbest bırakır.
     */
    protected void shutdown() {
        try {
            // Tüm modelleri kapat
            for (M3TMModel model : loadedModels.values()) {
                try {
                    model.close();
                } catch (Exception e) {
                    Log.w(TAG, "Model kapatma hatası: " + e.getMessage(), e);
                }
            }
            
            loadedModels.clear();
            Log.i(TAG, "Model yöneticisi kapatıldı");
        } catch (Exception e) {
            Log.e(TAG, "Model yöneticisi kapatma hatası: " + e.getMessage(), e);
        }
    }
    
    /**
     * Belirtilen ID'ye sahip modeli yükler ve döndürür.
     * 
     * @param modelId Model ID'si
     * @return Yüklenen model
     * @throws M3TMException Model yüklenemezse
     */
    public M3TMModel loadModel(String modelId) throws M3TMException {
        return loadModel(modelId, null);
    }
    
    /**
     * Belirtilen ID ve versiyona sahip modeli yükler ve döndürür.
     * 
     * @param modelId Model ID'si
     * @param version Model versiyonu (null ise en son versiyon)
     * @return Yüklenen model
     * @throws M3TMException Model yüklenemezse
     */
    public M3TMModel loadModel(String modelId, String version) throws M3TMException {
        checkInitialized();
        
        // Model zaten yüklü mü kontrol et
        String key = modelId + (version != null ? "_" + version : "");
        if (loadedModels.containsKey(key)) {
            return loadedModels.get(key);
        }
        
        try {
            Log.i(TAG, "Model yükleniyor: " + modelId + (version != null ? " v" + version : ""));
            
            // JNI üzerinden modeli yükle
            Map<String, Object> result = nativeLoadModel(modelId, version);
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Model yükleme hatası: " + errorMessage);
                throw new M3TMException(M3TMException.ErrorCode.MODEL_LOAD_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> data = (Map<String, Object>) result.get("data");
            String actualModelId = (String) data.get("model_id");
            String actualVersion = (String) data.get("version");
            
            // Model nesnesini oluştur
            M3TMModel model = new M3TMModel(actualModelId, actualVersion, this);
            
            // Yüklenen modellere ekle
            loadedModels.put(key, model);
            
            Log.i(TAG, "Model yüklendi: " + actualModelId + " v" + actualVersion);
            return model;
            
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Model yükleme hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.MODEL_LOAD_ERROR, "Model yüklenemedi: " + modelId, e);
        }
    }
    
    /**
     * Modeli bellekten kaldırır.
     * 
     * @param modelId Model ID'si
     * @return İşlem başarılı oldu mu
     */
    public boolean unloadModel(String modelId) {
        return unloadModel(modelId, null);
    }
    
    /**
     * Belirtilen ID ve versiyona sahip modeli bellekten kaldırır.
     * 
     * @param modelId Model ID'si
     * @param version Model versiyonu (null ise en son versiyon)
     * @return İşlem başarılı oldu mu
     */
    public boolean unloadModel(String modelId, String version) {
        checkInitialized();
        
        String key = modelId + (version != null ? "_" + version : "");
        M3TMModel model = loadedModels.get(key);
        
        if (model == null) {
            return false;
        }
        
        try {
            boolean success = model.close();
            if (success) {
                loadedModels.remove(key);
                Log.i(TAG, "Model bellekten kaldırıldı: " + modelId + (version != null ? " v" + version : ""));
            }
            return success;
        } catch (Exception e) {
            Log.e(TAG, "Model kaldırma hatası: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Kullanılabilir modellerin listesini döndürür.
     * 
     * @return Model listesi
     * @throws M3TMException Liste alınamazsa
     */
    public List<Map<String, Object>> listAvailableModels() throws M3TMException {
        checkInitialized();
        
        try {
            Map<String, Object> result = nativeListAvailableModels();
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Model listesi alınamadı: " + errorMessage);
                throw new M3TMException(M3TMException.ErrorCode.MODEL_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> models = (List<Map<String, Object>>) result.get("data");
            return models != null ? models : new ArrayList<>();
            
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Model listesi alınamadı: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.MODEL_ERROR, "Model listesi alınamadı", e);
        }
    }
    
    /**
     * Belirtilen modelin bilgilerini döndürür.
     * 
     * @param modelId Model ID'si
     * @return Model bilgileri
     * @throws M3TMException Bilgiler alınamazsa
     */
    public Map<String, Object> getModelInfo(String modelId) throws M3TMException {
        checkInitialized();
        
        try {
            Map<String, Object> result = nativeGetModelInfo(modelId);
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Model bilgisi alınamadı: " + errorMessage);
                
                if (errorMessage.contains("not found")) {
                    throw new M3TMException(M3TMException.ErrorCode.MODEL_NOT_FOUND, "Model bulunamadı: " + modelId);
                }
                
                throw new M3TMException(M3TMException.ErrorCode.MODEL_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> modelInfo = (Map<String, Object>) result.get("data");
            return modelInfo;
            
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Model bilgisi alınamadı: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.MODEL_ERROR, "Model bilgisi alınamadı: " + modelId, e);
        }
    }
    
    /**
     * Belirtilen modelin kullanılabilir olup olmadığını kontrol eder.
     * 
     * @param modelId Model ID'si
     * @return Model kullanılabilir mi
     */
    public boolean isModelAvailable(String modelId) {
        checkInitialized();
        
        try {
            return nativeIsModelAvailable(modelId);
        } catch (Exception e) {
            Log.e(TAG, "Model kullanılabilirlik kontrolü hatası: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Model dizinini döndürür.
     * 
     * @return Model dizini
     */
    protected String getModelDirectory() {
        return modelDirectory;
    }
    
    /**
     * Önbellek dizinini döndürür.
     * 
     * @return Önbellek dizini
     */
    protected String getCacheDirectory() {
        return cacheDirectory;
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
            throw new M3TMException(M3TMException.ErrorCode.NOT_INITIALIZED, "Model yöneticisi henüz başlatılmadı");
        }
    }
    
    // Native metot çağrıları
    
    private native Map<String, Object> nativeLoadModel(String modelId, String version);
    
    private native Map<String, Object> nativeListAvailableModels();
    
    private native Map<String, Object> nativeGetModelInfo(String modelId);
    
    private native boolean nativeIsModelAvailable(String modelId);
} 