package com.m3tm.sdk;

/**
 * Eğitim işlemleri sırasında oluşan hatalar için istisna sınıfı.
 */
public class TrainingException extends M3TMException {
    /**
     * Yeni bir TrainingException örneği oluşturur.
     *
     * @param message Hata mesajı
     */
    public TrainingException(String message) {
        super(message);
    }

    /**
     * Yeni bir TrainingException örneği oluşturur.
     *
     * @param message Hata mesajı
     * @param cause   Hatanın nedeni
     */
    public TrainingException(String message, Throwable cause) {
        super(message, cause);
    }
} 