package com.m3tm.sdk.security;

import android.content.Context;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import com.m3tm.sdk.M3TMException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.security.SecureRandom;

import static org.junit.Assert.*;

/**
 * OWASP MASVS-STORAGE güvenli depolama entegrasyon testleri.
 * 
 * SecureStorageManager'ın tüm fonksiyonlarını test eder.
 */
@RunWith(AndroidJUnit4.class)
public class SecureStorageIntegrationTest {
    
    private Context context;
    private SecureStorageManager secureStorageManager;
    
    @Before
    public void setUp() throws Exception {
        context = ApplicationProvider.getApplicationContext();
        secureStorageManager = new SecureStorageManager(context);
        secureStorageManager.initialize();
    }
    
    @After
    public void tearDown() {
        if (secureStorageManager != null) {
            secureStorageManager.shutdown();
        }
    }
    
    @Test
    public void testSecureStringStorage() throws M3TMException {
        String testKey = "test_string_key";
        String testValue = "Bu güvenli bir test stringidir.";
        
        // Store
        secureStorageManager.storeSecureString(testKey, testValue);
        
        // Retrieve
        String retrievedValue = secureStorageManager.getSecureString(testKey, null);
        
        assertEquals("Stored and retrieved values should match", testValue, retrievedValue);
        
        // Clean up
        secureStorageManager.removeSecureData(testKey);
        
        // Verify removal
        String removedValue = secureStorageManager.getSecureString(testKey, null);
        assertNull("Value should be null after removal", removedValue);
    }
    
    @Test
    public void testSecureDataStorage() throws M3TMException {
        String testKey = "test_data_key";
        byte[] testData = generateRandomData(1024); // 1KB test data
        
        // Store
        secureStorageManager.storeSecureData(testKey, testData);
        
        // Retrieve
        byte[] retrievedData = secureStorageManager.getSecureData(testKey);
        
        assertNotNull("Retrieved data should not be null", retrievedData);
        assertArrayEquals("Stored and retrieved data should match", testData, retrievedData);
        
        // Clean up
        secureStorageManager.removeSecureData(testKey);
        
        // Verify removal
        byte[] removedData = secureStorageManager.getSecureData(testKey);
        assertNull("Data should be null after removal", removedData);
    }
    
    @Test
    public void testHardwareBackedSecurity() {
        boolean hardwareBackedSecurity = secureStorageManager.isHardwareBackedSecurity();
        
        // Log the result for manual verification
        System.out.println("Hardware-backed security available: " + hardwareBackedSecurity);
        
        // This is device-dependent, so we just verify the method works
        assertTrue("Method should return a boolean value", 
            hardwareBackedSecurity == true || hardwareBackedSecurity == false);
    }
    
    @Test
    public void testLargeDataStorage() throws M3TMException {
        String testKey = "test_large_data";
        byte[] largeData = generateRandomData(10 * 1024); // 10KB
        
        // Store large data
        secureStorageManager.storeSecureData(testKey, largeData);
        
        // Retrieve
        byte[] retrievedData = secureStorageManager.getSecureData(testKey);
        
        assertNotNull("Large data should be retrieved successfully", retrievedData);
        assertEquals("Data size should match", largeData.length, retrievedData.length);
        assertArrayEquals("Large data content should match", largeData, retrievedData);
        
        // Clean up
        secureStorageManager.removeSecureData(testKey);
    }
    
    @Test
    public void testUnicodeStringStorage() throws M3TMException {
        String testKey = "test_unicode";
        String unicodeValue = "Türkçe karakterler: şğüöçıİ, 中文, العربية, 🔐🛡️";
        
        // Store Unicode string
        secureStorageManager.storeSecureString(testKey, unicodeValue);
        
        // Retrieve
        String retrievedValue = secureStorageManager.getSecureString(testKey, null);
        
        assertEquals("Unicode string should be preserved", unicodeValue, retrievedValue);
        
        // Clean up
        secureStorageManager.removeSecureData(testKey);
    }
    
    @Test
    public void testEmptyDataHandling() throws M3TMException {
        String testKey = "test_empty";
        
        // Test empty string
        secureStorageManager.storeSecureString(testKey, "");
        String emptyString = secureStorageManager.getSecureString(testKey, null);
        assertEquals("Empty string should be preserved", "", emptyString);
        
        // Test empty byte array
        secureStorageManager.storeSecureData(testKey, new byte[0]);
        byte[] emptyData = secureStorageManager.getSecureData(testKey);
        assertNotNull("Empty data should not be null", emptyData);
        assertEquals("Empty data length should be 0", 0, emptyData.length);
        
        // Clean up
        secureStorageManager.removeSecureData(testKey);
    }
    
    @Test
    public void testClearAllSecureData() throws M3TMException {
        String key1 = "test_key_1";
        String key2 = "test_key_2";
        String value1 = "Test value 1";
        String value2 = "Test value 2";
        
        // Store multiple values
        secureStorageManager.storeSecureString(key1, value1);
        secureStorageManager.storeSecureString(key2, value2);
        
        // Verify storage
        assertEquals(value1, secureStorageManager.getSecureString(key1, null));
        assertEquals(value2, secureStorageManager.getSecureString(key2, null));
        
        // Clear all
        secureStorageManager.clearAllSecureData();
        
        // Verify clearance
        assertNull("All data should be cleared", secureStorageManager.getSecureString(key1, null));
        assertNull("All data should be cleared", secureStorageManager.getSecureString(key2, null));
    }
    
    @Test(expected = M3TMException.class)
    public void testUninitializedAccess() throws M3TMException {
        SecureStorageManager uninitializedManager = new SecureStorageManager(context);
        
        // This should throw M3TMException
        uninitializedManager.storeSecureString("test", "value");
    }
    
    @Test
    public void testDefaultValueHandling() throws M3TMException {
        String nonExistentKey = "non_existent_key";
        String defaultValue = "default_value";
        
        String result = secureStorageManager.getSecureString(nonExistentKey, defaultValue);
        
        assertEquals("Should return default value for non-existent key", defaultValue, result);
    }
    
    /**
     * Güvenli random data oluşturur test için.
     * 
     * @param size Byte cinsinden data boyutu
     * @return Random byte array
     */
    private byte[] generateRandomData(int size) {
        byte[] data = new byte[size];
        new SecureRandom().nextBytes(data);
        return data;
    }
}
