package com.m3tm.sdk.security;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Log;

import androidx.annotation.RequiresApi;
import androidx.security.crypto.EncryptedSharedPreferences;
import androidx.security.crypto.MasterKey;

import com.m3tm.sdk.M3TMException;

import java.io.IOException;
import java.security.GeneralSecurityException;
import java.security.KeyStore;
import java.security.SecureRandom;
import java.util.Arrays;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/**
 * OWASP MASVS-STORAGE uyumlu güvenli depolama yöneticisi.
 * 
 * Android KeyStore ve EncryptedSharedPreferences kullanarak
 * hassas verilerin güvenli şekilde saklanmasını sağlar.
 */
public class SecureStorageManager {
    private static final String TAG = "SecureStorageManager";
    private static final String PREFS_NAME = "m3tm_secure_prefs";
    private static final String KEYSTORE_ALIAS = "M3TMSecureStorageKey";
    private static final String ANDROID_KEYSTORE = "AndroidKeyStore";
    private static final int GCM_IV_LENGTH = 12;
    private static final int GCM_TAG_LENGTH = 16;
    
    private final Context context;
    private SharedPreferences encryptedPrefs;
    private MasterKey masterKey;
    private boolean isInitialized = false;
    private boolean hardwareBackedSecurity = false;
    
    public SecureStorageManager(Context context) {
        this.context = context.getApplicationContext();
    }
    
    /**
     * Güvenli depolama sistemini başlatır.
     * Hardware-backed keystore kullanmayı dener, başarısız olursa software fallback kullanır.
     * 
     * @throws M3TMException Başlatma başarısız olursa
     */
    public void initialize() throws M3TMException {
        try {
            // Hardware-backed keystore kontrolü
            hardwareBackedSecurity = checkHardwareBackedSecurity();
            Log.i(TAG, "Hardware-backed security: " + hardwareBackedSecurity);
            
            // MasterKey oluştur (hardware-backed preferred)
            masterKey = createMasterKey();
            
            // EncryptedSharedPreferences oluştur
            encryptedPrefs = EncryptedSharedPreferences.create(
                context,
                PREFS_NAME,
                masterKey,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            );
            
            isInitialized = true;
            Log.i(TAG, "Secure storage initialized successfully");
            
        } catch (GeneralSecurityException | IOException e) {
            Log.e(TAG, "Failed to initialize secure storage", e);
            throw new M3TMException(M3TMException.ErrorCode.INITIALIZATION_ERROR, 
                "Secure storage initialization failed", e);
        }
    }
    
    /**
     * Hardware-backed security desteğini kontrol eder.
     * 
     * @return true eğer hardware-backed keystore mevcutsa
     */
    private boolean checkHardwareBackedSecurity() {
        try {
            KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
            keyStore.load(null);
            
            // Test key oluşturarak hardware-backed support kontrolü
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                KeyGenerator keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE);
                KeyGenParameterSpec.Builder builder = new KeyGenParameterSpec.Builder(
                    "test_hw_key",
                    KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                    .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE);
                
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    builder.setIsStrongBoxBacked(true); // Hardware security module
                }
                
                keyGenerator.init(builder.build());
                SecretKey testKey = keyGenerator.generateKey();
                
                // Test key'i sil
                keyStore.deleteEntry("test_hw_key");
                
                return testKey != null;
            }
            return false;
        } catch (Exception e) {
            Log.w(TAG, "Hardware-backed security check failed", e);
            return false;
        }
    }
    
    /**
     * MasterKey oluşturur, hardware-backed keystore tercih eder.
     * 
     * @return MasterKey instance
     * @throws GeneralSecurityException
     */
    private MasterKey createMasterKey() throws GeneralSecurityException {
        MasterKey.Builder builder = new MasterKey.Builder(context, KEYSTORE_ALIAS)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM);
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P && hardwareBackedSecurity) {
            builder.setRequestStrongBoxBacked(true);
        }
        
        try {
            return builder.build();
        } catch (Exception e) {
            Log.w(TAG, "Failed to create hardware-backed master key, falling back to software", e);
            // Software fallback
            return new MasterKey.Builder(context, KEYSTORE_ALIAS + "_sw")
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .setRequestStrongBoxBacked(false)
                .build();
        }
    }
    
    /**
     * Güvenli string depolama.
     * 
     * @param key Anahtar
     * @param value Değer
     * @throws M3TMException Depolama başarısız olursa
     */
    public void storeSecureString(String key, String value) throws M3TMException {
        checkInitialized();
        
        try {
            encryptedPrefs.edit()
                .putString(key, value)
                .apply();
            
            Log.d(TAG, "Secure string stored for key: " + key);
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to store secure string", e);
            throw new M3TMException(M3TMException.ErrorCode.STORAGE_ERROR, 
                "Failed to store secure data", e);
        }
    }
    
    /**
     * Güvenli string okuma.
     * 
     * @param key Anahtar
     * @param defaultValue Varsayılan değer
     * @return Saklanan değer veya varsayılan değer
     * @throws M3TMException Okuma başarısız olursa
     */
    public String getSecureString(String key, String defaultValue) throws M3TMException {
        checkInitialized();
        
        try {
            return encryptedPrefs.getString(key, defaultValue);
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to get secure string", e);
            throw new M3TMException(M3TMException.ErrorCode.STORAGE_ERROR, 
                "Failed to retrieve secure data", e);
        }
    }
    
    /**
     * Güvenli byte array depolama.
     * AES-GCM-256 ile şifrelenerek saklanır.
     * 
     * @param key Anahtar
     * @param data Şifrelenecek veri
     * @throws M3TMException Şifreleme veya depolama başarısız olursa
     */
    public void storeSecureData(String key, byte[] data) throws M3TMException {
        checkInitialized();
        
        try {
            byte[] encryptedData = encryptWithAESGCM(data);
            String base64Data = android.util.Base64.encodeToString(encryptedData, android.util.Base64.DEFAULT);
            
            encryptedPrefs.edit()
                .putString(key + "_encrypted", base64Data)
                .apply();
            
            Log.d(TAG, "Secure data stored for key: " + key);
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to store secure data", e);
            throw new M3TMException(M3TMException.ErrorCode.STORAGE_ERROR, 
                "Failed to store encrypted data", e);
        }
    }
    
    /**
     * Güvenli byte array okuma.
     * AES-GCM-256 ile çözümlenir.
     * 
     * @param key Anahtar
     * @return Çözümlenmiş veri, null eğer bulunamazsa
     * @throws M3TMException Çözümleme başarısız olursa
     */
    public byte[] getSecureData(String key) throws M3TMException {
        checkInitialized();
        
        try {
            String base64Data = encryptedPrefs.getString(key + "_encrypted", null);
            if (base64Data == null) {
                return null;
            }
            
            byte[] encryptedData = android.util.Base64.decode(base64Data, android.util.Base64.DEFAULT);
            return decryptWithAESGCM(encryptedData);
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to get secure data", e);
            throw new M3TMException(M3TMException.ErrorCode.STORAGE_ERROR, 
                "Failed to retrieve encrypted data", e);
        }
    }
    
    /**
     * AES-GCM-256 ile veri şifreleme.
     * 
     * @param data Şifrelenecek veri
     * @return IV + şifrelenmiş veri + tag
     * @throws Exception Şifreleme başarısız olursa
     */
    private byte[] encryptWithAESGCM(byte[] data) throws Exception {
        // Secure random IV oluştur
        byte[] iv = new byte[GCM_IV_LENGTH];
        new SecureRandom().nextBytes(iv);
        
        // Cipher oluştur ve başlat
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        GCMParameterSpec gcmParameterSpec = new GCMParameterSpec(GCM_TAG_LENGTH * 8, iv);
        
        // KeyStore'dan key al
        KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
        keyStore.load(null);
        SecretKey secretKey = (SecretKey) keyStore.getKey(KEYSTORE_ALIAS, null);
        
        cipher.init(Cipher.ENCRYPT_MODE, secretKey, gcmParameterSpec);
        
        // Şifrele
        byte[] encryptedData = cipher.doFinal(data);
        
        // IV + encrypted data + tag birleştir
        byte[] result = new byte[iv.length + encryptedData.length];
        System.arraycopy(iv, 0, result, 0, iv.length);
        System.arraycopy(encryptedData, 0, result, iv.length, encryptedData.length);
        
        return result;
    }
    
    /**
     * AES-GCM-256 ile veri çözümleme.
     * 
     * @param encryptedData IV + şifrelenmiş veri + tag
     * @return Çözümlenmiş veri
     * @throws Exception Çözümleme başarısız olursa
     */
    private byte[] decryptWithAESGCM(byte[] encryptedData) throws Exception {
        // IV'yi ayır
        byte[] iv = Arrays.copyOfRange(encryptedData, 0, GCM_IV_LENGTH);
        byte[] cipherText = Arrays.copyOfRange(encryptedData, GCM_IV_LENGTH, encryptedData.length);
        
        // Cipher oluştur ve başlat
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        GCMParameterSpec gcmParameterSpec = new GCMParameterSpec(GCM_TAG_LENGTH * 8, iv);
        
        // KeyStore'dan key al
        KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
        keyStore.load(null);
        SecretKey secretKey = (SecretKey) keyStore.getKey(KEYSTORE_ALIAS, null);
        
        cipher.init(Cipher.DECRYPT_MODE, secretKey, gcmParameterSpec);
        
        // Çözümle
        return cipher.doFinal(cipherText);
    }
    
    /**
     * Anahtar silme.
     * 
     * @param key Silinecek anahtar
     */
    public void removeSecureData(String key) {
        if (!isInitialized) {
            return;
        }
        
        encryptedPrefs.edit()
            .remove(key)
            .remove(key + "_encrypted")
            .apply();
        
        Log.d(TAG, "Secure data removed for key: " + key);
    }
    
    /**
     * Tüm güvenli verileri temizle.
     */
    public void clearAllSecureData() {
        if (!isInitialized) {
            return;
        }
        
        encryptedPrefs.edit().clear().apply();
        Log.i(TAG, "All secure data cleared");
    }
    
    /**
     * Hardware-backed security durumunu döndürür.
     * 
     * @return true eğer hardware-backed keystore kullanılıyorsa
     */
    public boolean isHardwareBackedSecurity() {
        return hardwareBackedSecurity;
    }
    
    /**
     * Başlatılma durumunu kontrol eder.
     * 
     * @throws M3TMException Başlatılmamışsa
     */
    private void checkInitialized() throws M3TMException {
        if (!isInitialized) {
            throw new M3TMException(M3TMException.ErrorCode.NOT_INITIALIZED, 
                "SecureStorageManager not initialized");
        }
    }
    
    /**
     * Kaynakları serbest bırakır.
     */
    public void shutdown() {
        isInitialized = false;
        encryptedPrefs = null;
        masterKey = null;
        Log.i(TAG, "SecureStorageManager shutdown completed");
    }
}
