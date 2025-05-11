package com.m3tm.sdk;

import java.util.Map;

/**
 * Eğitim ilerleme geri çağrıları için arayüz.
 * Bu arayüz, eğitim sürecinin durumu hakkında bilgi sağlamak için kullanılır.
 */
public interface TrainingCallback {
    /**
     * Her batch tamamlandığında çağrılır.
     *
     * @param batch   Tamamlanan batch numarası
     * @param metrics Batch metrikleri
     */
    void onBatchComplete(int batch, Map<String, Object> metrics);

    /**
     * Her epoch tamamlandığında çağrılır.
     *
     * @param epoch   Tamamlanan epoch numarası
     * @param metrics Epoch metrikleri
     */
    void onEpochComplete(int epoch, Map<String, Object> metrics);

    /**
     * Eğitim tamamlandığında çağrılır.
     *
     * @param metrics Final eğitim metrikleri
     */
    void onTrainingComplete(Map<String, Object> metrics);
} 