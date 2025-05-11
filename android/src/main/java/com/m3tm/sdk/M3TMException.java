package com.m3tm.sdk;

/**
 * M3TM SDK'sıyla ilgili tüm istisnaların temel sınıfı.
 */
public class M3TMException extends Exception {
    private Throwable cause;

    /**
     * Yeni bir M3TMException örneği oluşturur.
     *
     * @param message Hata mesajı
     */
    public M3TMException(String message) {
        super(message);
    }

    /**
     * Yeni bir M3TMException örneği oluşturur.
     *
     * @param message Hata mesajı
     * @param cause   Hatanın nedeni
     */
    public M3TMException(String message, Throwable cause) {
        super(message);
        this.cause = cause;
    }

    /**
     * Hatanın nedenini döndürür.
     *
     * @return Hatanın nedeni
     */
    @Override
    public Throwable getCause() {
        return cause;
    }
} 