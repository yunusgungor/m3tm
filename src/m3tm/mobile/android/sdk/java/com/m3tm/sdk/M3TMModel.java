package com.m3tm.sdk;

import android.graphics.Bitmap;
import android.util.Log;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * M³TM model sınıfı.
 * 
 * Bu sınıf, yüklenmiş bir M³TM modeli üzerinde çıkarım işlemleri
 * gerçekleştirmek için kullanılır.
 */
public class M3TMModel {
    private static final String TAG = "M3TMModel";
    
    private final String modelId;
    private final String version;
    private final M3TMModelManager modelManager;
    private boolean isClosed = false;
    
    /**
     * M3TMModel oluşturur.
     * 
     * @param modelId Model ID'si
     * @param version Model versiyonu
     * @param modelManager Model yöneticisi
     */
    protected M3TMModel(String modelId, String version, M3TMModelManager modelManager) {
        this.modelId = modelId;
        this.version = version;
        this.modelManager = modelManager;
    }
    
    /**
     * Metin verisi üzerinde çıkarım yapar.
     * 
     * @param text İşlenecek metin
     * @param task Görev tipi
     * @return Çıkarım sonuçları
     * @throws M3TMException Çıkarım başarısız olursa
     */
    public Map<String, Object> processText(String text, String task) throws M3TMException {
        return processText(text, task, null);
    }
    
    /**
     * Metin verisi üzerinde çıkarım yapar.
     * 
     * @param text İşlenecek metin
     * @param task Görev tipi
     * @param options Çıkarım seçenekleri
     * @return Çıkarım sonuçları
     * @throws M3TMException Çıkarım başarısız olursa
     */
    public Map<String, Object> processText(String text, String task, Map<String, Object> options) throws M3TMException {
        checkClosed();
        
        if (text == null || text.isEmpty()) {
            throw new M3TMException(M3TMException.ErrorCode.INPUT_FORMAT_ERROR, "Metin boş olamaz");
        }
        
        if (task == null || task.isEmpty()) {
            throw new M3TMException(M3TMException.ErrorCode.INPUT_FORMAT_ERROR, "Görev tipi belirtilmelidir");
        }
        
        try {
            Log.i(TAG, "Metin çıkarımı başlıyor: " + task);
            
            Map<String, Object> result = nativeProcessText(modelId, text, task, options != null ? options : new HashMap<>());
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Metin çıkarımı hatası: " + errorMessage);
                throw new M3TMException(M3TMException.ErrorCode.INFERENCE_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> data = (Map<String, Object>) result.get("data");
            
            Log.i(TAG, "Metin çıkarımı tamamlandı");
            return data;
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Metin çıkarımı hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.INFERENCE_ERROR, "Metin çıkarımı sırasında hata oluştu", e);
        }
    }
    
    /**
     * Görüntü verisi üzerinde çıkarım yapar.
     * 
     * @param image İşlenecek görüntü
     * @param task Görev tipi
     * @return Çıkarım sonuçları
     * @throws M3TMException Çıkarım başarısız olursa
     */
    public Map<String, Object> processImage(Bitmap image, String task) throws M3TMException {
        return processImage(image, task, null);
    }
    
    /**
     * Görüntü verisi üzerinde çıkarım yapar.
     * 
     * @param image İşlenecek görüntü
     * @param task Görev tipi
     * @param options Çıkarım seçenekleri
     * @return Çıkarım sonuçları
     * @throws M3TMException Çıkarım başarısız olursa
     */
    public Map<String, Object> processImage(Bitmap image, String task, Map<String, Object> options) throws M3TMException {
        checkClosed();
        
        if (image == null || image.isRecycled()) {
            throw new M3TMException(M3TMException.ErrorCode.INPUT_FORMAT_ERROR, "Geçerli bir görüntü sağlanmalıdır");
        }
        
        if (task == null || task.isEmpty()) {
            throw new M3TMException(M3TMException.ErrorCode.INPUT_FORMAT_ERROR, "Görev tipi belirtilmelidir");
        }
        
        try {
            Log.i(TAG, "Görüntü çıkarımı başlıyor: " + task);
            
            // Bitmap'i byte dizisine dönüştür
            byte[] imageData = bitmapToByteArray(image);
            
            Map<String, Object> result = nativeProcessImage(modelId, imageData, task, options != null ? options : new HashMap<>());
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Görüntü çıkarımı hatası: " + errorMessage);
                throw new M3TMException(M3TMException.ErrorCode.INFERENCE_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> data = (Map<String, Object>) result.get("data");
            
            Log.i(TAG, "Görüntü çıkarımı tamamlandı");
            return data;
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Görüntü çıkarımı hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.INFERENCE_ERROR, "Görüntü çıkarımı sırasında hata oluştu", e);
        }
    }
    
    /**
     * Metin ve görüntü üzerinde çoklu-modalite çıkarımı yapar.
     * 
     * @param text İşlenecek metin
     * @param image İşlenecek görüntü
     * @param task Görev tipi
     * @return Çıkarım sonuçları
     * @throws M3TMException Çıkarım başarısız olursa
     */
    public Map<String, Object> processMultimodal(String text, Bitmap image, String task) throws M3TMException {
        return processMultimodal(text, image, task, null);
    }
    
    /**
     * Metin ve görüntü üzerinde çoklu-modalite çıkarımı yapar.
     * 
     * @param text İşlenecek metin
     * @param image İşlenecek görüntü
     * @param task Görev tipi
     * @param options Çıkarım seçenekleri
     * @return Çıkarım sonuçları
     * @throws M3TMException Çıkarım başarısız olursa
     */
    public Map<String, Object> processMultimodal(String text, Bitmap image, String task, Map<String, Object> options) throws M3TMException {
        checkClosed();
        
        if (image == null || image.isRecycled()) {
            throw new M3TMException(M3TMException.ErrorCode.INPUT_FORMAT_ERROR, "Geçerli bir görüntü sağlanmalıdır");
        }
        
        if (text == null) {
            text = ""; // Boş metin kabul edilebilir
        }
        
        if (task == null || task.isEmpty()) {
            throw new M3TMException(M3TMException.ErrorCode.INPUT_FORMAT_ERROR, "Görev tipi belirtilmelidir");
        }
        
        try {
            Log.i(TAG, "Çoklu-modalite çıkarımı başlıyor: " + task);
            
            // Bitmap'i byte dizisine dönüştür
            byte[] imageData = bitmapToByteArray(image);
            
            Map<String, Object> result = nativeProcessMultimodal(modelId, text, imageData, task, options != null ? options : new HashMap<>());
            
            if (!"success".equals(result.get("status"))) {
                String errorMessage = (String) result.get("error");
                Log.e(TAG, "Çoklu-modalite çıkarımı hatası: " + errorMessage);
                throw new M3TMException(M3TMException.ErrorCode.INFERENCE_ERROR, errorMessage);
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> data = (Map<String, Object>) result.get("data");
            
            Log.i(TAG, "Çoklu-modalite çıkarımı tamamlandı");
            return data;
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Çoklu-modalite çıkarımı hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.INFERENCE_ERROR, "Çoklu-modalite çıkarımı sırasında hata oluştu", e);
        }
    }
    
    /**
     * Modelin desteklediği görevleri döndürür.
     * 
     * @return Desteklenen görevler listesi
     * @throws M3TMException Görevler alınamazsa
     */
    public String[] getSupportedTasks() throws M3TMException {
        checkClosed();
        
        try {
            Map<String, Object> modelInfo = modelManager.getModelInfo(modelId);
            
            @SuppressWarnings("unchecked")
            String[] tasks = (String[]) modelInfo.get("supported_tasks");
            return tasks != null ? tasks : new String[0];
        } catch (M3TMException e) {
            throw e;
        } catch (Exception e) {
            Log.e(TAG, "Desteklenen görevler alınamadı: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.MODEL_ERROR, "Desteklenen görevler alınamadı", e);
        }
    }
    
    /**
     * Modeli kapatır ve kaynakları serbest bırakır.
     * 
     * @return İşlem başarılı oldu mu
     * @throws M3TMException Modeli kapatırken hata oluşursa
     */
    public boolean close() throws M3TMException {
        if (isClosed) {
            return true;
        }
        
        try {
            boolean result = nativeCloseModel(modelId);
            if (result) {
                isClosed = true;
                Log.i(TAG, "Model kapatıldı: " + modelId);
            }
            return result;
        } catch (Exception e) {
            Log.e(TAG, "Model kapatma hatası: " + e.getMessage(), e);
            throw new M3TMException(M3TMException.ErrorCode.MODEL_ERROR, "Model kapatılamadı", e);
        }
    }
    
    /**
     * Model ID'sini döndürür.
     * 
     * @return Model ID'si
     */
    public String getModelId() {
        return modelId;
    }
    
    /**
     * Model versiyonunu döndürür.
     * 
     * @return Model versiyonu
     */
    public String getVersion() {
        return version;
    }
    
    /**
     * Modelin kapatılıp kapatılmadığını kontrol eder.
     * 
     * @return Model kapatıldı mı
     */
    public boolean isClosed() {
        return isClosed;
    }
    
    /**
     * Modelin kapatılıp kapatılmadığını kontrol eder, kapatılmışsa hata fırlatır.
     * 
     * @throws M3TMException Model kapatılmışsa
     */
    private void checkClosed() throws M3TMException {
        if (isClosed) {
            throw new M3TMException(M3TMException.ErrorCode.MODEL_ERROR, "Model zaten kapatıldı: " + modelId);
        }
    }
    
    /**
     * Bitmap'i byte dizisine dönüştürür.
     * 
     * @param bitmap Dönüştürülecek bitmap
     * @return Byte dizisi
     */
    private byte[] bitmapToByteArray(Bitmap bitmap) {
        // Bu metot bitmap'i byte dizisine dönüştürecek
        // İlerde implementasyon eklenecek
        return new byte[0];
    }
    
    // Native metot çağrıları
    
    private native Map<String, Object> nativeProcessText(String modelId, String text, String task, Map<String, Object> options);
    
    private native Map<String, Object> nativeProcessImage(String modelId, byte[] imageData, String task, Map<String, Object> options);
    
    private native Map<String, Object> nativeProcessMultimodal(String modelId, String text, byte[] imageData, String task, Map<String, Object> options);
    
    private native boolean nativeCloseModel(String modelId);
} 