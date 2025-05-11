package com.m3tm.sdk;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Model yükleme, listeleme ve yönetimi için işlevler sağlayan sınıf.
 */
public class M3TMModelManager {
    private static final String DEFAULT_CACHE_DIR = "m3tm_models";
    private final String modelCacheDir;
    private final Map<String, M3TMModel> loadedModels;
    
    // JNI tarafından kullanılan native işaretçi (pointer)
    private long nativeHandle;
    
    static {
        System.loadLibrary("m3tm_sdk_jni");
    }

    /**
     * M3TMModelManager sınıfını başlatır.
     */
    public M3TMModelManager() {
        this(null);
    }

    /**
     * M3TMModelManager sınıfını başlatır.
     *
     * @param modelCacheDir Model önbellek dizini (opsiyonel)
     */
    public M3TMModelManager(String modelCacheDir) {
        this.modelCacheDir = modelCacheDir != null ? modelCacheDir : DEFAULT_CACHE_DIR;
        this.loadedModels = new HashMap<>();
        
        this.nativeHandle = nativeInitialize(this.modelCacheDir);
        if (this.nativeHandle == 0) {
            throw new RuntimeException("ModelManager başlatılamadı");
        }
    }

    /**
     * Belirtilen modeli yükler.
     *
     * @param modelId Model benzersiz tanımlayıcısı
     * @return Yüklenen model
     * @throws ModelException Model yüklenemediğinde
     */
    public M3TMModel loadModel(String modelId) throws ModelException {
        return loadModel(modelId, null);
    }

    /**
     * Belirtilen modeli yükler.
     *
     * @param modelId Model benzersiz tanımlayıcısı
     * @param version İsteğe bağlı model versiyonu
     * @return Yüklenen model
     * @throws ModelException Model yüklenemediğinde
     */
    public synchronized M3TMModel loadModel(String modelId, String version) throws ModelException {
        // Daha önce yüklendi mi kontrol et
        if (loadedModels.containsKey(modelId)) {
            M3TMModel model = loadedModels.get(modelId);
            if (!model.isClosed()) {
                return model;
            }
            // Model kapatılmışsa, haritadan kaldır
            loadedModels.remove(modelId);
        }
        
        try {
            // Native tarafta modeli yükle
            long modelHandle = nativeLoadModel(nativeHandle, modelId, version);
            if (modelHandle == 0) {
                throw new ModelException("Model yüklenemedi: " + modelId);
            }
            
            // Model meta verilerini al
            ModelInfo modelInfo = getModelInfo(modelId);
            
            // Model nesnesini oluştur
            M3TMModel model = new M3TMModel(modelHandle, modelId, modelInfo);
            loadedModels.put(modelId, model);
            
            return model;
        } catch (Exception e) {
            throw new ModelException("Model yüklenemedi: " + e.getMessage(), e);
        }
    }

    /**
     * Kullanılabilir modellerin listesini döndürür.
     *
     * @return Kullanılabilir modellerin listesi
     */
    public List<ModelInfo> listAvailableModels() {
        try {
            List<Map<String, Object>> modelsData = nativeListAvailableModels(nativeHandle);
            List<ModelInfo> modelInfos = new ArrayList<>();
            
            if (modelsData != null) {
                for (Map<String, Object> modelData : modelsData) {
                    String modelId = (String) modelData.get("model_id");
                    String version = (String) modelData.getOrDefault("latest_version", "unknown");
                    String name = (String) modelData.getOrDefault("name", modelId);
                    String description = (String) modelData.getOrDefault("description", "");
                    
                    ModelInfo modelInfo = new ModelInfo(modelId, version, name, description, modelData);
                    modelInfos.add(modelInfo);
                }
            }
            
            return modelInfos;
        } catch (Exception e) {
            // Hata durumunda boş liste döndür
            return new ArrayList<>();
        }
    }

    /**
     * Belirtilen model hakkında bilgi alır.
     *
     * @param modelId Model benzersiz tanımlayıcısı
     * @return Model bilgisi
     * @throws ModelException Model bilgisi alınamadığında
     */
    public ModelInfo getModelInfo(String modelId) throws ModelException {
        try {
            Map<String, Object> metadata = nativeGetModelMetadata(nativeHandle, modelId);
            
            if (metadata == null) {
                throw new ModelException("Model bilgisi alınamadı: " + modelId);
            }
            
            String version = (String) metadata.getOrDefault("version", "unknown");
            Map<String, Object> info = (Map<String, Object>) metadata.getOrDefault("info", new HashMap<>());
            
            String name = (String) info.getOrDefault("name", modelId);
            String description = (String) info.getOrDefault("description", "");
            
            return new ModelInfo(modelId, version, name, description, info);
        } catch (Exception e) {
            throw new ModelException("Model bilgisi alınamadı: " + e.getMessage(), e);
        }
    }

    /**
     * Belirtilen modelin mevcut olup olmadığını kontrol eder.
     *
     * @param modelId Model benzersiz tanımlayıcısı
     * @return Modelin mevcut olup olmadığı
     */
    public boolean isModelAvailable(String modelId) {
        try {
            getModelInfo(modelId);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Kaynakları serbest bırakır.
     */
    public synchronized void close() {
        if (nativeHandle != 0) {
            // Tüm yüklü modelleri kapat
            for (M3TMModel model : loadedModels.values()) {
                if (!model.isClosed()) {
                    model.close();
                }
            }
            loadedModels.clear();
            
            // Native ModelManager'ı kapat
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
    private native long nativeLoadModel(long handle, String modelId, String version);
    private native List<Map<String, Object>> nativeListAvailableModels(long handle);
    private native Map<String, Object> nativeGetModelMetadata(long handle, String modelId);
    private native void nativeClose(long handle);
} 