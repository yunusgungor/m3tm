package com.m3tm.sdk;

import android.content.Context;
import android.util.Log;

/**
 * M³TM SDK ana sınıfı.
 * 
 * Bu sınıf, SDK'nın ana giriş noktasıdır ve model yönetimi, 
 * çıkarım ve eğitim için gereken tüm işlevlere erişim sağlar.
 */
public class M3TM {
    private static final String TAG = "M3TM";
    private static volatile M3TM instance;
    
    private final Context context;
    private final M3TMModelManager modelManager;
    private final M3TMTrainingManager trainingManager;
    private boolean isInitialized = false;
    
    /**
     * Singleton instance döndürür.
     * 
     * @param context Uygulama bağlamı
     * @return M3TM instance
     */
    public static synchronized M3TM getInstance(Context context) {
        if (instance == null) {
            instance = new M3TM(context.getApplicationContext());
        }
        return instance;
    }
    
    private M3TM(Context context) {
        this.context = context.getApplicationContext();
        this.modelManager = new M3TMModelManager(this);
        this.trainingManager = new M3TMTrainingManager(this);
    }
    
    /**
     * SDK'yı belirtilen yapılandırma ile başlatır.
     * 
     * @param config SDK yapılandırması
     * @return Başlatma işlemi başarılı oldu mu
     */
    public boolean initialize(M3TMConfig config) {
        if (isInitialized) {
            Log.w(TAG, "M3TM SDK zaten başlatılmış durumda.");
            return true;
        }
        
        try {
            // Native kütüphaneleri yükle
            System.loadLibrary("m3tm_jni");
            
            // Yolları ayarla
            String modelDir = config.getModelDirectory();
            String cacheDir = config.getCacheDirectory();
            
            // Alt bileşenleri başlat
            boolean modelManagerInit = modelManager.initialize(modelDir, cacheDir);
            boolean trainingManagerInit = trainingManager.initialize(modelDir, cacheDir);
            
            isInitialized = modelManagerInit && trainingManagerInit;
            
            if (isInitialized) {
                Log.i(TAG, "M3TM SDK başarıyla başlatıldı.");
            } else {
                Log.e(TAG, "M3TM SDK başlatma hatası.");
            }
            
            return isInitialized;
        } catch (Exception e) {
            Log.e(TAG, "M3TM SDK başlatma hatası: " + e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * SDK'yı kapatır ve kaynakları serbest bırakır.
     */
    public void shutdown() {
        if (!isInitialized) {
            return;
        }
        
        try {
            modelManager.shutdown();
            trainingManager.shutdown();
            isInitialized = false;
            Log.i(TAG, "M3TM SDK kapatıldı.");
        } catch (Exception e) {
            Log.e(TAG, "M3TM SDK kapatma hatası: " + e.getMessage(), e);
        }
    }
    
    /**
     * SDK'nın başlatılıp başlatılmadığını kontrol eder.
     * 
     * @return SDK başlatılmış mı
     */
    public boolean isInitialized() {
        return isInitialized;
    }
    
    /**
     * Model yöneticisini döndürür.
     * 
     * @return Model yöneticisi
     */
    public M3TMModelManager getModelManager() {
        checkInitialized();
        return modelManager;
    }
    
    /**
     * Eğitim yöneticisini döndürür.
     * 
     * @return Eğitim yöneticisi
     */
    public M3TMTrainingManager getTrainingManager() {
        checkInitialized();
        return trainingManager;
    }
    
    /**
     * SDK başlatılmamışsa hata fırlatır.
     * 
     * @throws M3TMException SDK başlatılmamışsa
     */
    private void checkInitialized() {
        if (!isInitialized) {
            throw new M3TMException("M3TM SDK henüz başlatılmadı. Lütfen önce initialize() metodunu çağırın.");
        }
    }
    
    /**
     * SDK'nın uygulama bağlamını döndürür.
     * 
     * @return Uygulama bağlamı
     */
    protected Context getContext() {
        return context;
    }
    
    /**
     * SDK versiyonunu döndürür.
     * 
     * @return SDK versiyonu
     */
    public static String getVersion() {
        return "2.3.0";
    }
} 