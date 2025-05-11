package com.m3tm.sdk;

/**
 * Çıkarım işlemleri sırasında oluşan hatalar için istisna sınıfı.
 */
public class InferenceException extends M3TMException {
    /**
     * Yeni bir InferenceException örneği oluşturur.
     *
     * @param message Hata mesajı
     */
    public InferenceException(String message) {
        super(message);
    }

    /**
     * Yeni bir InferenceException örneği oluşturur.
     *
     * @param message Hata mesajı
     * @param cause   Hatanın nedeni
     */
    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
} 