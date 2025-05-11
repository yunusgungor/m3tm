package com.m3tm.sdk;

import android.content.Context;
import android.os.Build;

import java.io.File;

/**
 * M³TM SDK yapılandırma sınıfı.
 * 
 * SDK'nın davranışını ve çalışma ortamını yapılandırmak için kullanılır.
 */
public class M3TMConfig {
    private String modelDirectory;
    private String cacheDirectory;
    private int maxCacheSizeMB;
    private LogLevel logLevel;
    private boolean useGpu;
    private boolean enableTelemetry;
    
    /**
     * Log seviyesi.
     */
    public enum LogLevel {
        VERBOSE, DEBUG, INFO, WARN, ERROR, NONE
    }
    
    private M3TMConfig(Builder builder) {
        this.modelDirectory = builder.modelDirectory;
        this.cacheDirectory = builder.cacheDirectory;
        this.maxCacheSizeMB = builder.maxCacheSizeMB;
        this.logLevel = builder.logLevel;
        this.useGpu = builder.useGpu;
        this.enableTelemetry = builder.enableTelemetry;
    }
    
    /**
     * Model dizinini döndürür.
     * 
     * @return Model dizini
     */
    public String getModelDirectory() {
        return modelDirectory;
    }
    
    /**
     * Önbellek dizinini döndürür.
     * 
     * @return Önbellek dizini
     */
    public String getCacheDirectory() {
        return cacheDirectory;
    }
    
    /**
     * Maksimum önbellek boyutunu döndürür (MB).
     * 
     * @return Maksimum önbellek boyutu
     */
    public int getMaxCacheSizeMB() {
        return maxCacheSizeMB;
    }
    
    /**
     * Log seviyesini döndürür.
     * 
     * @return Log seviyesi
     */
    public LogLevel getLogLevel() {
        return logLevel;
    }
    
    /**
     * GPU kullanımının etkin olup olmadığını döndürür.
     * 
     * @return GPU kullanımı etkin mi
     */
    public boolean isUseGpu() {
        return useGpu;
    }
    
    /**
     * Telemetri'nin etkin olup olmadığını döndürür.
     * 
     * @return Telemetri etkin mi
     */
    public boolean isEnableTelemetry() {
        return enableTelemetry;
    }
    
    /**
     * M3TMConfig oluşturucu sınıfı.
     */
    public static class Builder {
        private String modelDirectory;
        private String cacheDirectory;
        private int maxCacheSizeMB = 100;  // Varsayılan 100MB
        private LogLevel logLevel = LogLevel.INFO;  // Varsayılan INFO
        private boolean useGpu = false;  // Varsayılan olarak CPU
        private boolean enableTelemetry = true;  // Varsayılan olarak etkin
        
        /**
         * Verilen Context ile yeni builder oluşturur ve varsayılan yolları ayarlar.
         * 
         * @param context Uygulama bağlamı
         */
        public Builder(Context context) {
            // Varsayılan model ve önbellek dizinlerini ayarla
            File filesDir = context.getFilesDir();
            this.modelDirectory = new File(filesDir, "models").getAbsolutePath();
            this.cacheDirectory = new File(context.getCacheDir(), "m3tm").getAbsolutePath();
            
            // GPU desteğini kontrol et
            this.useGpu = checkGpuSupport();
        }
        
        /**
         * Model dizinini ayarlar.
         * 
         * @param modelDirectory Model dizini
         * @return Builder instance
         */
        public Builder setModelDirectory(String modelDirectory) {
            this.modelDirectory = modelDirectory;
            return this;
        }
        
        /**
         * Önbellek dizinini ayarlar.
         * 
         * @param cacheDirectory Önbellek dizini
         * @return Builder instance
         */
        public Builder setCacheDirectory(String cacheDirectory) {
            this.cacheDirectory = cacheDirectory;
            return this;
        }
        
        /**
         * Maksimum önbellek boyutunu ayarlar (MB).
         * 
         * @param maxCacheSizeMB Maksimum önbellek boyutu
         * @return Builder instance
         */
        public Builder setMaxCacheSizeMB(int maxCacheSizeMB) {
            this.maxCacheSizeMB = maxCacheSizeMB > 0 ? maxCacheSizeMB : 100;
            return this;
        }
        
        /**
         * Log seviyesini ayarlar.
         * 
         * @param logLevel Log seviyesi
         * @return Builder instance
         */
        public Builder setLogLevel(LogLevel logLevel) {
            this.logLevel = logLevel != null ? logLevel : LogLevel.INFO;
            return this;
        }
        
        /**
         * GPU kullanımını etkinleştirir veya devre dışı bırakır.
         * 
         * @param useGpu GPU kullanımı etkin mi
         * @return Builder instance
         */
        public Builder setUseGpu(boolean useGpu) {
            this.useGpu = useGpu;
            return this;
        }
        
        /**
         * Telemetri'yi etkinleştirir veya devre dışı bırakır.
         * 
         * @param enableTelemetry Telemetri etkin mi
         * @return Builder instance
         */
        public Builder setEnableTelemetry(boolean enableTelemetry) {
            this.enableTelemetry = enableTelemetry;
            return this;
        }
        
        /**
         * Yapılandırılmış M3TMConfig nesnesini oluşturur.
         * 
         * @return M3TMConfig instance
         */
        public M3TMConfig build() {
            return new M3TMConfig(this);
        }
        
        /**
         * Cihazın GPU desteğini kontrol eder.
         * 
         * @return GPU destekleniyor mu
         */
        private boolean checkGpuSupport() {
            // Bu basit bir örnek, gerçekte daha kapsamlı kontroller yapılabilir
            return Build.VERSION.SDK_INT >= Build.VERSION_CODES.N;
        }
    }
} 