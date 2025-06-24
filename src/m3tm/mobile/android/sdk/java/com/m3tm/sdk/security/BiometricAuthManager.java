package com.m3tm.sdk.security;

import android.content.Context;
import android.os.Build;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyPermanentlyInvalidatedException;
import android.security.keystore.KeyProperties;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.biometric.BiometricManager;
import androidx.biometric.BiometricPrompt;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;

import com.m3tm.sdk.M3TMException;

import java.security.KeyStore;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.spec.InvalidKeySpecException;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/**
 * OWASP MASVS-AUTH uyumlu biometric authentication yöneticisi.
 * 
 * BiometricPrompt API kullanarak güvenli biometric authentication sağlar.
 * Hardware-backed authentication ile cryptographic operations destekler.
 */
public class BiometricAuthManager {
    private static final String TAG = "BiometricAuthManager";
    private static final String BIOMETRIC_KEY_ALIAS = "M3TMBiometricKey";
    private static final String ANDROID_KEYSTORE = "AndroidKeyStore";
    private static final int GCM_IV_LENGTH = 12;
    private static final int GCM_TAG_LENGTH = 16;
    
    private final Context context;
    private boolean isInitialized = false;
    private BiometricManager biometricManager;
    private SecretKey biometricKey;
    
    /**
     * Biometric authentication callback interface.
     */
    public interface BiometricAuthCallback {
        void onAuthenticationSucceeded(BiometricPrompt.AuthenticationResult result);
        void onAuthenticationError(int errorCode, CharSequence errString);
        void onAuthenticationFailed();
    }
    
    /**
     * Biometric encryption callback interface.
     */
    public interface BiometricEncryptionCallback {
        void onEncryptionSucceeded(byte[] encryptedData);
        void onDecryptionSucceeded(byte[] decryptedData);
        void onError(int errorCode, CharSequence errString);
    }
    
    public BiometricAuthManager(Context context) {
        this.context = context.getApplicationContext();
    }
    
    /**
     * Biometric authentication sistemini başlatır.
     * 
     * @throws M3TMException Başlatma başarısız olursa
     */
    public void initialize() throws M3TMException {
        biometricManager = BiometricManager.from(context);
        
        // Biometric support kontrolü
        int biometricStatus = checkBiometricSupport();
        if (biometricStatus != BiometricManager.BIOMETRIC_SUCCESS) {
            throw new M3TMException(M3TMException.ErrorCode.INITIALIZATION_ERROR, 
                "Biometric authentication not supported: " + getBiometricStatusMessage(biometricStatus));
        }
        
        // Biometric key oluştur/al
        try {
            biometricKey = getOrCreateBiometricKey();
            isInitialized = true;
            Log.i(TAG, "Biometric authentication initialized successfully");
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize biometric authentication", e);
            throw new M3TMException(M3TMException.ErrorCode.INITIALIZATION_ERROR, 
                "Biometric key setup failed", e);
        }
    }
    
    /**
     * Biometric support durumunu kontrol eder.
     * 
     * @return BiometricManager status code
     */
    public int checkBiometricSupport() {
        if (biometricManager == null) {
            biometricManager = BiometricManager.from(context);
        }
        
        return biometricManager.canAuthenticate(
            BiometricManager.Authenticators.BIOMETRIC_STRONG | 
            BiometricManager.Authenticators.DEVICE_CREDENTIAL
        );
    }
    
    /**
     * Biometric status mesajını döndürür.
     * 
     * @param status BiometricManager status code
     * @return Human-readable status message
     */
    private String getBiometricStatusMessage(int status) {
        switch (status) {
            case BiometricManager.BIOMETRIC_SUCCESS:
                return "Biometric authentication is available";
            case BiometricManager.BIOMETRIC_ERROR_NO_HARDWARE:
                return "No biometric hardware available";
            case BiometricManager.BIOMETRIC_ERROR_HW_UNAVAILABLE:
                return "Biometric hardware is currently unavailable";
            case BiometricManager.BIOMETRIC_ERROR_NONE_ENROLLED:
                return "No biometric credentials enrolled";
            case BiometricManager.BIOMETRIC_ERROR_SECURITY_UPDATE_REQUIRED:
                return "Security update required for biometric authentication";
            case BiometricManager.BIOMETRIC_ERROR_UNSUPPORTED:
                return "Biometric authentication is not supported";
            case BiometricManager.BIOMETRIC_STATUS_UNKNOWN:
                return "Biometric status unknown";
            default:
                return "Unknown biometric status";
        }
    }
    
    /**
     * Biometric key oluşturur veya mevcut olanı alır.
     * 
     * @return Biometric operations için SecretKey
     */
    @RequiresApi(api = Build.VERSION_CODES.M)
    private SecretKey getOrCreateBiometricKey() throws Exception {
        KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
        keyStore.load(null);
        
        // Mevcut key'i kontrol et
        if (keyStore.containsAlias(BIOMETRIC_KEY_ALIAS)) {
            try {
                return (SecretKey) keyStore.getKey(BIOMETRIC_KEY_ALIAS, null);
            } catch (KeyPermanentlyInvalidatedException e) {
                // Key geçersiz hale gelmiş, yeniden oluştur
                Log.w(TAG, "Biometric key invalidated, recreating");
                keyStore.deleteEntry(BIOMETRIC_KEY_ALIAS);
            }
        }
        
        // Yeni key oluştur
        KeyGenerator keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE);
        
        KeyGenParameterSpec.Builder builder = new KeyGenParameterSpec.Builder(
            BIOMETRIC_KEY_ALIAS,
            KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setUserAuthenticationRequired(true)
            .setInvalidatedByBiometricEnrollment(true); // Yeni biometric enrollment key'i geçersiz kılsın
        
        // Biometric authentication requirement
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            builder.setUnlockedDeviceRequired(true)
                   .setUserAuthenticationValidityDurationSeconds(-1); // Her kullanımda auth gerekli
        } else {
            builder.setUserAuthenticationValidityDurationSeconds(0); // Her kullanımda auth gerekli
        }
        
        // Hardware-backed security tercih et
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            builder.setIsStrongBoxBacked(true);
        }
        
        try {
            keyGenerator.init(builder.build());
            SecretKey secretKey = keyGenerator.generateKey();
            Log.i(TAG, "Biometric key created successfully");
            return secretKey;
            
        } catch (Exception e) {
            // StrongBox desteklenmiyorsa fallback
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                Log.w(TAG, "StrongBox not supported, falling back to TEE");
                builder.setIsStrongBoxBacked(false);
                keyGenerator.init(builder.build());
                return keyGenerator.generateKey();
            }
            throw e;
        }
    }
    
    /**
     * Basit biometric authentication gerçekleştirir.
     * 
     * @param activity Fragment activity reference
     * @param title Prompt title
     * @param subtitle Prompt subtitle
     * @param callback Authentication callback
     */
    public void authenticateUser(FragmentActivity activity, String title, String subtitle, BiometricAuthCallback callback) {
        checkInitialized();
        
        BiometricPrompt.PromptInfo promptInfo = new BiometricPrompt.PromptInfo.Builder()
            .setTitle(title)
            .setSubtitle(subtitle)
            .setNegativeButtonText("İptal")
            .setAllowedAuthenticators(
                BiometricManager.Authenticators.BIOMETRIC_STRONG | 
                BiometricManager.Authenticators.DEVICE_CREDENTIAL
            )
            .build();
        
        BiometricPrompt biometricPrompt = new BiometricPrompt(activity, 
            ContextCompat.getMainExecutor(context), 
            new BiometricPrompt.AuthenticationCallback() {
                @Override
                public void onAuthenticationSucceeded(@NonNull BiometricPrompt.AuthenticationResult result) {
                    super.onAuthenticationSucceeded(result);
                    Log.i(TAG, "Biometric authentication succeeded");
                    callback.onAuthenticationSucceeded(result);
                }
                
                @Override
                public void onAuthenticationError(int errorCode, @NonNull CharSequence errString) {
                    super.onAuthenticationError(errorCode, errString);
                    Log.w(TAG, "Biometric authentication error: " + errString);
                    callback.onAuthenticationError(errorCode, errString);
                }
                
                @Override
                public void onAuthenticationFailed() {
                    super.onAuthenticationFailed();
                    Log.w(TAG, "Biometric authentication failed");
                    callback.onAuthenticationFailed();
                }
            });
        
        biometricPrompt.authenticate(promptInfo);
    }
    
    /**
     * Cryptographic object ile biometric authentication.
     * Veri şifreleme/çözümleme için kullanılır.
     * 
     * @param activity Fragment activity reference
     * @param data Şifrelenecek/çözümlenecek veri
     * @param isEncryption true ise şifreleme, false ise çözümleme
     * @param callback Encryption/decryption callback
     */
    @RequiresApi(api = Build.VERSION_CODES.M)
    public void authenticateWithCrypto(FragmentActivity activity, byte[] data, boolean isEncryption, BiometricEncryptionCallback callback) {
        checkInitialized();
        
        try {
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
            
            if (isEncryption) {
                cipher.init(Cipher.ENCRYPT_MODE, biometricKey);
            } else {
                // Decryption için IV gerekli (data'nın başında saklanmış olmalı)
                if (data.length < GCM_IV_LENGTH) {
                    callback.onError(-1, "Invalid encrypted data");
                    return;
                }
                
                byte[] iv = new byte[GCM_IV_LENGTH];
                System.arraycopy(data, 0, iv, 0, GCM_IV_LENGTH);
                
                GCMParameterSpec gcmParameterSpec = new GCMParameterSpec(GCM_TAG_LENGTH * 8, iv);
                cipher.init(Cipher.DECRYPT_MODE, biometricKey, gcmParameterSpec);
            }
            
            BiometricPrompt.CryptoObject cryptoObject = new BiometricPrompt.CryptoObject(cipher);
            
            BiometricPrompt.PromptInfo promptInfo = new BiometricPrompt.PromptInfo.Builder()
                .setTitle("Biometric Authentication Required")
                .setSubtitle(isEncryption ? "Encrypt data with biometric" : "Decrypt data with biometric")
                .setNegativeButtonText("İptal")
                .setAllowedAuthenticators(BiometricManager.Authenticators.BIOMETRIC_STRONG)
                .build();
            
            BiometricPrompt biometricPrompt = new BiometricPrompt(activity, 
                ContextCompat.getMainExecutor(context), 
                new BiometricPrompt.AuthenticationCallback() {
                    @Override
                    public void onAuthenticationSucceeded(@NonNull BiometricPrompt.AuthenticationResult result) {
                        super.onAuthenticationSucceeded(result);
                        
                        try {
                            Cipher authenticatedCipher = result.getCryptoObject().getCipher();
                            
                            if (isEncryption) {
                                // Encrypt
                                byte[] encryptedData = authenticatedCipher.doFinal(data);
                                byte[] iv = authenticatedCipher.getIV();
                                
                                // IV + encrypted data birleştir
                                byte[] result_data = new byte[iv.length + encryptedData.length];
                                System.arraycopy(iv, 0, result_data, 0, iv.length);
                                System.arraycopy(encryptedData, 0, result_data, iv.length, encryptedData.length);
                                
                                callback.onEncryptionSucceeded(result_data);
                                
                            } else {
                                // Decrypt
                                byte[] cipherText = new byte[data.length - GCM_IV_LENGTH];
                                System.arraycopy(data, GCM_IV_LENGTH, cipherText, 0, cipherText.length);
                                
                                byte[] decryptedData = authenticatedCipher.doFinal(cipherText);
                                callback.onDecryptionSucceeded(decryptedData);
                            }
                            
                        } catch (Exception e) {
                            Log.e(TAG, "Cryptographic operation failed", e);
                            callback.onError(-1, "Cryptographic operation failed: " + e.getMessage());
                        }
                    }
                    
                    @Override
                    public void onAuthenticationError(int errorCode, @NonNull CharSequence errString) {
                        super.onAuthenticationError(errorCode, errString);
                        callback.onError(errorCode, errString);
                    }
                    
                    @Override
                    public void onAuthenticationFailed() {
                        super.onAuthenticationFailed();
                        callback.onError(-1, "Biometric authentication failed");
                    }
                });
            
            biometricPrompt.authenticate(promptInfo, cryptoObject);
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to setup biometric crypto authentication", e);
            callback.onError(-1, "Crypto setup failed: " + e.getMessage());
        }
    }
    
    /**
     * Biometric authentication availability durumunu döndürür.
     * 
     * @return Authentication mevcut mu
     */
    public boolean isBiometricAuthAvailable() {
        return checkBiometricSupport() == BiometricManager.BIOMETRIC_SUCCESS;
    }
    
    /**
     * Hardware-backed biometric security durumunu kontrol eder.
     * 
     * @return Hardware-backed security mevcut mu
     */
    public boolean isHardwareBackedSecurity() {
        try {
            KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
            keyStore.load(null);
            
            if (keyStore.containsAlias(BIOMETRIC_KEY_ALIAS)) {
                SecretKey key = (SecretKey) keyStore.getKey(BIOMETRIC_KEY_ALIAS, null);
                return key != null;
            }
            
        } catch (Exception e) {
            Log.w(TAG, "Failed to check hardware-backed security", e);
        }
        
        return false;
    }
    
    /**
     * Başlatılma durumunu kontrol eder.
     * 
     * @throws M3TMException Başlatılmamışsa
     */
    private void checkInitialized() throws M3TMException {
        if (!isInitialized) {
            throw new M3TMException(M3TMException.ErrorCode.NOT_INITIALIZED, 
                "BiometricAuthManager not initialized");
        }
    }
    
    /**
     * Biometric key'i siler ve kaynakları serbest bırakır.
     */
    public void shutdown() {
        try {
            KeyStore keyStore = KeyStore.getInstance(ANDROID_KEYSTORE);
            keyStore.load(null);
            
            if (keyStore.containsAlias(BIOMETRIC_KEY_ALIAS)) {
                keyStore.deleteEntry(BIOMETRIC_KEY_ALIAS);
                Log.i(TAG, "Biometric key deleted");
            }
            
        } catch (Exception e) {
            Log.w(TAG, "Failed to delete biometric key", e);
        }
        
        isInitialized = false;
        biometricKey = null;
        Log.i(TAG, "BiometricAuthManager shutdown completed");
    }
}
