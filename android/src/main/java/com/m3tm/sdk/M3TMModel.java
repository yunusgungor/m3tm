package com.m3tm.sdk;

import java.util.HashMap;
import java.util.Map;

/**
 * Yüklenen M3TM modelini temsil eden ve çıkarım işlevleri sağlayan sınıf.
 */
public class M3TMModel {
    // JNI tarafından kullanılan native işaretçi (pointer)
    private long nativeHandle;
    private final String modelId;
    private final ModelInfo modelInfo;
    private boolean isClosed = false;

    /**
     * Native tarafından çağrılacak, JNI tarafından kullanılacak yapıcı.
     * Bu yapıcı doğrudan çağrılmamalı, M3TMModelManager.loadModel() kullanılmalıdır.
     *
     * @param nativeHandle Native taraftaki model işaretçisi
     * @param modelId      Model benzersiz tanımlayıcısı
     * @param modelInfo    Model meta verileri
     */
    M3TMModel(long nativeHandle, String modelId, ModelInfo modelInfo) {
        this.nativeHandle = nativeHandle;
        this.modelId = modelId;
        this.modelInfo = modelInfo;
    }

    /**
     * Model bilgisini döndürür.
     *
     * @return Model bilgisi
     */
    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    /**
     * Metin verisi üzerinde model çıkarımı yapar.
     *
     * @param text    İşlenecek metin
     * @param task    İsteğe bağlı görev adı
     * @param options Çıkarım seçenekleri
     * @return Çıkarım sonuçları
     * @throws InferenceException      Çıkarım sırasında hata oluştuğunda
     * @throws IllegalStateException   Model kapatıldığında
     */
    public Map<String, Object> processText(String text, String task, Map<String, Object> options) throws InferenceException {
        checkNotClosed();
        
        if (text == null || text.isEmpty()) {
            throw new InferenceException("İşlenecek metin boş olamaz");
        }
        
        Map<String, Object> optionsMap = options != null ? options : new HashMap<>();
        
        try {
            return nativeProcessText(nativeHandle, text, task, optionsMap);
        } catch (Exception e) {
            throw new InferenceException("Metin işleme hatası: " + e.getMessage(), e);
        }
    }
    
    /**
     * Metin verisi üzerinde model çıkarımı yapar (basitleştirilmiş versiyon).
     *
     * @param text İşlenecek metin
     * @return Çıkarım sonuçları
     * @throws InferenceException    Çıkarım sırasında hata oluştuğunda
     * @throws IllegalStateException Model kapatıldığında
     */
    public Map<String, Object> processText(String text) throws InferenceException {
        return processText(text, null, null);
    }

    /**
     * Görüntü verisi üzerinde model çıkarımı yapar.
     *
     * @param imageData İşlenecek görüntü (byte[] veya Bitmap olarak geçirilebilir)
     * @param task      İsteğe bağlı görev adı
     * @param options   Çıkarım seçenekleri
     * @return Çıkarım sonuçları
     * @throws InferenceException    Çıkarım sırasında hata oluştuğunda
     * @throws IllegalStateException Model kapatıldığında
     */
    public Map<String, Object> processImage(Object imageData, String task, Map<String, Object> options) throws InferenceException {
        checkNotClosed();
        
        if (imageData == null) {
            throw new InferenceException("İşlenecek görüntü verisi boş olamaz");
        }
        
        Map<String, Object> optionsMap = options != null ? options : new HashMap<>();
        
        try {
            return nativeProcessImage(nativeHandle, imageData, task, optionsMap);
        } catch (Exception e) {
            throw new InferenceException("Görüntü işleme hatası: " + e.getMessage(), e);
        }
    }
    
    /**
     * Görüntü verisi üzerinde model çıkarımı yapar (basitleştirilmiş versiyon).
     *
     * @param imageData İşlenecek görüntü
     * @return Çıkarım sonuçları
     * @throws InferenceException    Çıkarım sırasında hata oluştuğunda
     * @throws IllegalStateException Model kapatıldığında
     */
    public Map<String, Object> processImage(Object imageData) throws InferenceException {
        return processImage(imageData, null, null);
    }

    /**
     * Metin ve görüntü verilerini birleştirerek çoklu-modalite çıkarımı yapar.
     *
     * @param text      İşlenecek metin (opsiyonel)
     * @param imageData İşlenecek görüntü (opsiyonel)
     * @param task      İsteğe bağlı görev adı
     * @param options   Çıkarım seçenekleri
     * @return Çıkarım sonuçları
     * @throws InferenceException    Çıkarım sırasında hata oluştuğunda
     * @throws IllegalStateException Model kapatıldığında
     */
    public Map<String, Object> processMultimodal(String text, Object imageData, String task, Map<String, Object> options) throws InferenceException {
        checkNotClosed();
        
        if (text == null && imageData == null) {
            throw new InferenceException("En az bir modalite (metin veya görüntü) gereklidir");
        }
        
        Map<String, Object> optionsMap = options != null ? options : new HashMap<>();
        
        try {
            return nativeProcessMultimodal(nativeHandle, text, imageData, task, optionsMap);
        } catch (Exception e) {
            throw new InferenceException("Çoklu-modalite işleme hatası: " + e.getMessage(), e);
        }
    }
    
    /**
     * Metin ve görüntü verilerini birleştirerek çoklu-modalite çıkarımı yapar (basitleştirilmiş versiyon).
     *
     * @param text      İşlenecek metin
     * @param imageData İşlenecek görüntü
     * @return Çıkarım sonuçları
     * @throws InferenceException    Çıkarım sırasında hata oluştuğunda
     * @throws IllegalStateException Model kapatıldığında
     */
    public Map<String, Object> processMultimodal(String text, Object imageData) throws InferenceException {
        return processMultimodal(text, imageData, null, null);
    }

    /**
     * Modeli kapatır ve kaynakları serbest bırakır.
     */
    public void close() {
        if (!isClosed && nativeHandle != 0) {
            nativeClose(nativeHandle);
            nativeHandle = 0;
            isClosed = true;
        }
    }

    /**
     * Model kapatıldı mı kontrol eder.
     *
     * @return Model kapatıldıysa true
     */
    public boolean isClosed() {
        return isClosed;
    }

    /**
     * Modelin kapatılmadığını kontrol eder.
     *
     * @throws IllegalStateException Model kapatıldığında
     */
    private void checkNotClosed() {
        if (isClosed) {
            throw new IllegalStateException("Model kapatıldı ve artık kullanılamaz");
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
    private native Map<String, Object> nativeProcessText(long handle, String text, String task, Map<String, Object> options);
    private native Map<String, Object> nativeProcessImage(long handle, Object imageData, String task, Map<String, Object> options);
    private native Map<String, Object> nativeProcessMultimodal(long handle, String text, Object imageData, String task, Map<String, Object> options);
    private native void nativeClose(long handle);
} 