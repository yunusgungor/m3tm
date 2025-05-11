package com.m3tm.sdk;

/**
 * M³TM SDK istisna sınıfı.
 * 
 * SDK kullanımı sırasında oluşabilecek hataları temsil eder.
 */
public class M3TMException extends RuntimeException {
    private final ErrorCode errorCode;
    
    /**
     * M³TM ile ilgili hata kodları.
     */
    public enum ErrorCode {
        GENERAL_ERROR(1000, "Genel hata"),
        INITIALIZATION_ERROR(1001, "Başlatma hatası"),
        NOT_INITIALIZED(1002, "SDK başlatılmadı"),
        
        MODEL_ERROR(2000, "Model hatası"),
        MODEL_NOT_FOUND(2001, "Model bulunamadı"),
        MODEL_LOAD_ERROR(2002, "Model yükleme hatası"),
        MODEL_VERSION_ERROR(2003, "Model versiyon hatası"),
        MODEL_EXECUTION_ERROR(2004, "Model çalıştırma hatası"),
        
        INFERENCE_ERROR(3000, "Çıkarım hatası"),
        INFERENCE_TIMEOUT(3001, "Çıkarım zaman aşımı"),
        INPUT_FORMAT_ERROR(3002, "Girdi format hatası"),
        
        TRAINING_ERROR(4000, "Eğitim hatası"),
        TRAINING_SESSION_ERROR(4001, "Eğitim oturumu hatası"),
        TRAINING_SESSION_NOT_FOUND(4002, "Eğitim oturumu bulunamadı"),
        TRAINING_SESSION_EXPIRED(4003, "Eğitim oturumu süresi doldu"),
        
        NATIVE_ERROR(5000, "Native kod hatası"),
        JNI_ERROR(5001, "JNI hatası"),
        PYTHON_ERROR(5002, "Python hatası"),
        
        IO_ERROR(6000, "I/O hatası"),
        NETWORK_ERROR(6001, "Ağ hatası"),
        PERMISSION_ERROR(6002, "İzin hatası"),
        
        UNKNOWN_ERROR(9999, "Bilinmeyen hata");
        
        private final int code;
        private final String description;
        
        ErrorCode(int code, String description) {
            this.code = code;
            this.description = description;
        }
        
        public int getCode() {
            return code;
        }
        
        public String getDescription() {
            return description;
        }
    }
    
    /**
     * Varsayılan hata koduyla (GENERAL_ERROR) yeni bir istisna oluşturur.
     * 
     * @param message Hata mesajı
     */
    public M3TMException(String message) {
        this(ErrorCode.GENERAL_ERROR, message);
    }
    
    /**
     * Belirtilen hata koduyla yeni bir istisna oluşturur.
     * 
     * @param errorCode Hata kodu
     * @param message Hata mesajı
     */
    public M3TMException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    /**
     * Belirtilen neden istisnasıyla yeni bir istisna oluşturur.
     * 
     * @param message Hata mesajı
     * @param cause Neden istisnası
     */
    public M3TMException(String message, Throwable cause) {
        this(ErrorCode.GENERAL_ERROR, message, cause);
    }
    
    /**
     * Belirtilen hata kodu ve neden istisnasıyla yeni bir istisna oluşturur.
     * 
     * @param errorCode Hata kodu
     * @param message Hata mesajı
     * @param cause Neden istisnası
     */
    public M3TMException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }
    
    /**
     * Hata kodunu döndürür.
     * 
     * @return Hata kodu
     */
    public ErrorCode getErrorCode() {
        return errorCode;
    }
    
    @Override
    public String toString() {
        return "M3TMException: " + errorCode.getCode() + " - " + errorCode.getDescription() + ": " + getMessage();
    }
} 