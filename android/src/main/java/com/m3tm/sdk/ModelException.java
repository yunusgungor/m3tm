package com.m3tm.sdk;

/**
 * Model yükleme ve işleme sırasında oluşan hatalar için istisna sınıfı.
 */
public class ModelException extends M3TMException {
    /**
     * Yeni bir ModelException örneği oluşturur.
     *
     * @param message Hata mesajı
     */
    public ModelException(String message) {
        super(message);
    }

    /**
     * Yeni bir ModelException örneği oluşturur.
     *
     * @param message Hata mesajı
     * @param cause   Hatanın nedeni
     */
    public ModelException(String message, Throwable cause) {
        super(message, cause);
    }
} 