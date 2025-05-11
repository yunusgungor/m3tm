package com.m3tm.sdk;

import java.util.Map;

/**
 * M³TM eğitim ilerleme geri çağrı arabirimi.
 * 
 * Bu arabirim, eğitim işlemi sırasında ilerleme bildirimleri
 * almak için kullanılır.
 */
public interface M3TMTrainingCallback {
    
    /**
     * Her batch işlemi tamamlandığında çağrılır.
     * 
     * @param batch Tamamlanan batch numarası
     * @param metrics Batch metrikleri
     */
    void onBatchComplete(int batch, Map<String, Object> metrics);
    
    /**
     * Her epoch işlemi tamamlandığında çağrılır.
     * 
     * @param epoch Tamamlanan epoch numarası
     * @param metrics Epoch metrikleri
     */
    void onEpochComplete(int epoch, Map<String, Object> metrics);
    
    /**
     * Eğitim işlemi tamamlandığında çağrılır.
     * 
     * @param metrics Final eğitim metrikleri
     */
    void onTrainingComplete(Map<String, Object> metrics);
} 