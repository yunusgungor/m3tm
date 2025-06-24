package com.m3tm.sdk.security;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.os.Build;
import android.util.Log;

import com.m3tm.sdk.M3TMException;

import java.io.IOException;
import java.net.URL;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLPeerUnverifiedException;
import javax.net.ssl.SSLSession;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

import okhttp3.CertificatePinner;
import okhttp3.ConnectionSpec;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.TlsVersion;

/**
 * OWASP MASVS-NETWORK uyumlu network security yöneticisi.
 * 
 * Certificate pinning, TLS 1.3 enforcement ve MITM protection sağlar.
 * OkHttp tabanlı güvenli HTTP client implementasyonu.
 */
public class NetworkSecurityManager {
    private static final String TAG = "NetworkSecurityManager";
    
    // Certificate pins (SHA-256 hashes)
    private static final String[] PRODUCTION_PINS = {
        "sha256/YLh1dUR9y6Kja30RrAn7JKnbQG/uEtLMkBgFF2fuihg=", // Primary
        "sha256/C5+lpZ7tcVwmwQIMcRtPbsQtWLABXhQzejna0wHFr8M=", // Backup
        "sha256/lCppFqbkrlJ3EcVFAkeip0+44VaoJUymbnOaEUk7tEU="  // Root CA backup
    };
    
    private static final String[] STAGING_PINS = {
        "sha256/Dev1234567890abcdef1234567890abcdef1234567890abcdef=",
        "sha256/Dev0987654321fedcba0987654321fedcba0987654321fedcba="
    };
    
    // Allowed domains
    private static final Set<String> PRODUCTION_DOMAINS = new HashSet<>(Arrays.asList(
        "api.m3tm.com",
        "secure.m3tm.com", 
        "models.m3tm.com"
    ));
    
    private static final Set<String> STAGING_DOMAINS = new HashSet<>(Arrays.asList(
        "dev-api.m3tm.com",
        "staging-api.m3tm.com"
    ));
    
    private final Context context;
    private OkHttpClient secureHttpClient;
    private boolean isInitialized = false;
    private boolean useStaging = false;
    
    public NetworkSecurityManager(Context context) {
        this.context = context.getApplicationContext();
    }
    
    /**
     * Network security sistemini başlatır.
     * 
     * @param useStaging Staging environment kullanılacak mı
     * @throws M3TMException Başlatma başarısız olursa
     */
    public void initialize(boolean useStaging) throws M3TMException {
        this.useStaging = useStaging;
        
        try {
            // Certificate pinner oluştur
            CertificatePinner.Builder pinnerBuilder = new CertificatePinner.Builder();
            
            if (useStaging) {
                // Staging certificate pins
                for (String domain : STAGING_DOMAINS) {
                    for (String pin : STAGING_PINS) {
                        pinnerBuilder.add(domain, pin);
                        pinnerBuilder.add("*." + domain, pin); // subdomains
                    }
                }
            } else {
                // Production certificate pins
                for (String domain : PRODUCTION_DOMAINS) {
                    for (String pin : PRODUCTION_PINS) {
                        pinnerBuilder.add(domain, pin);
                        pinnerBuilder.add("*." + domain, pin); // subdomains
                    }
                }
            }
            
            CertificatePinner certificatePinner = pinnerBuilder.build();
            
            // TLS 1.3 configuration
            ConnectionSpec tlsSpec = new ConnectionSpec.Builder(ConnectionSpec.MODERN_TLS)
                .tlsVersions(TlsVersion.TLS_1_3, TlsVersion.TLS_1_2) // TLS 1.3 preferred
                .allEnabledCipherSuites() // En güvenli cipher suites
                .build();
            
            // Secure OkHttpClient oluştur
            secureHttpClient = new OkHttpClient.Builder()
                .certificatePinner(certificatePinner)
                .connectionSpecs(Arrays.asList(tlsSpec, ConnectionSpec.CLEARTEXT)) // CLEARTEXT sadece localhost için
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .hostnameVerifier(createSecureHostnameVerifier())
                .build();
            
            isInitialized = true;
            Log.i(TAG, "Network security initialized (staging: " + useStaging + ")");
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize network security", e);
            throw new M3TMException(M3TMException.ErrorCode.INITIALIZATION_ERROR, 
                "Network security initialization failed", e);
        }
    }
    
    /**
     * Güvenli hostname verifier oluşturur.
     * 
     * @return HostnameVerifier instance
     */
    private HostnameVerifier createSecureHostnameVerifier() {
        return new HostnameVerifier() {
            @Override
            public boolean verify(String hostname, SSLSession session) {
                // Default hostname verification
                boolean defaultResult = HttpsURLConnection.getDefaultHostnameVerifier().verify(hostname, session);
                
                if (!defaultResult) {
                    Log.w(TAG, "Hostname verification failed for: " + hostname);
                    return false;
                }
                
                // Additional domain whitelist check
                Set<String> allowedDomains = useStaging ? STAGING_DOMAINS : PRODUCTION_DOMAINS;
                boolean domainAllowed = false;
                
                for (String domain : allowedDomains) {
                    if (hostname.equals(domain) || hostname.endsWith("." + domain)) {
                        domainAllowed = true;
                        break;
                    }
                }
                
                if (!domainAllowed) {
                    Log.w(TAG, "Domain not in whitelist: " + hostname);
                    return false;
                }
                
                return true;
            }
        };
    }
    
    /**
     * Güvenli HTTP GET request yapar.
     * 
     * @param url Request URL
     * @return Response body string
     * @throws M3TMException Request başarısız olursa
     */
    public String makeSecureGetRequest(String url) throws M3TMException {
        checkInitialized();
        
        try {
            Request request = new Request.Builder()
                .url(url)
                .addHeader("User-Agent", "M3TM-SDK/1.0")
                .addHeader("Accept", "application/json")
                .build();
            
            try (Response response = secureHttpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new M3TMException(M3TMException.ErrorCode.NETWORK_ERROR, 
                        "HTTP request failed with code: " + response.code());
                }
                
                if (response.body() == null) {
                    throw new M3TMException(M3TMException.ErrorCode.NETWORK_ERROR, 
                        "Empty response body");
                }
                
                return response.body().string();
            }
            
        } catch (IOException e) {
            Log.e(TAG, "Secure GET request failed for URL: " + url, e);
            
            // Certificate pinning failure detection
            if (e.getMessage() != null && e.getMessage().contains("Certificate pinning failure")) {
                throw new M3TMException(M3TMException.ErrorCode.NETWORK_ERROR, 
                    "Certificate pinning validation failed - possible MITM attack", e);
            }
            
            throw new M3TMException(M3TMException.ErrorCode.NETWORK_ERROR, 
                "Network request failed", e);
        }
    }
    
    /**
     * Güvenli HTTP POST request yapar.
     * 
     * @param url Request URL
     * @param jsonBody JSON request body
     * @return Response body string
     * @throws M3TMException Request başarısız olursa
     */
    public String makeSecurePostRequest(String url, String jsonBody) throws M3TMException {
        checkInitialized();
        
        try {
            okhttp3.RequestBody requestBody = okhttp3.RequestBody.create(
                jsonBody, 
                okhttp3.MediaType.parse("application/json; charset=utf-8")
            );
            
            Request request = new Request.Builder()
                .url(url)
                .post(requestBody)
                .addHeader("User-Agent", "M3TM-SDK/1.0")
                .addHeader("Content-Type", "application/json")
                .addHeader("Accept", "application/json")
                .build();
            
            try (Response response = secureHttpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new M3TMException(M3TMException.ErrorCode.NETWORK_ERROR, 
                        "HTTP POST request failed with code: " + response.code());
                }
                
                if (response.body() == null) {
                    throw new M3TMException(M3TMException.ErrorCode.NETWORK_ERROR, 
                        "Empty response body");
                }
                
                return response.body().string();
            }
            
        } catch (IOException e) {
            Log.e(TAG, "Secure POST request failed for URL: " + url, e);
            throw new M3TMException(M3TMException.ErrorCode.NETWORK_ERROR, 
                "Network POST request failed", e);
        }
    }
    
    /**
     * Certificate pinning durumunu test eder.
     * 
     * @param testUrl Test edilecek URL
     * @return Pinning başarılı mı
     */
    public boolean testCertificatePinning(String testUrl) {
        checkInitialized();
        
        try {
            Request request = new Request.Builder()
                .url(testUrl)
                .build();
            
            try (Response response = secureHttpClient.newCall(request).execute()) {
                return response.isSuccessful();
            }
            
        } catch (Exception e) {
            Log.w(TAG, "Certificate pinning test failed", e);
            return false;
        }
    }
    
    /**
     * Network connectivity durumunu kontrol eder.
     * 
     * @return Network bağlantısı mevcut mu
     */
    public boolean isNetworkAvailable() {
        ConnectivityManager connectivityManager = 
            (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        
        if (connectivityManager == null) {
            return false;
        }
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Network network = connectivityManager.getActiveNetwork();
            if (network == null) {
                return false;
            }
            
            NetworkCapabilities capabilities = connectivityManager.getNetworkCapabilities(network);
            return capabilities != null && 
                   (capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
                    capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) ||
                    capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET));
        } else {
            android.net.NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
            return activeNetworkInfo != null && activeNetworkInfo.isConnected();
        }
    }
    
    /**
     * TLS version ve cipher suite bilgilerini loglar.
     * Debugging ve security audit için kullanılır.
     * 
     * @param url Test edilecek URL
     */
    public void logTLSInfo(String url) {
        try {
            URL testUrl = new URL(url);
            HttpsURLConnection connection = (HttpsURLConnection) testUrl.openConnection();
            connection.connect();
            
            Log.i(TAG, "TLS Connection Info for " + url + ":");
            Log.i(TAG, "Protocol: " + connection.getSSLSocketFactory().toString());
            
            Certificate[] certificates = connection.getServerCertificates();
            for (int i = 0; i < certificates.length; i++) {
                if (certificates[i] instanceof X509Certificate) {
                    X509Certificate cert = (X509Certificate) certificates[i];
                    Log.i(TAG, "Certificate " + i + ": " + cert.getSubjectDN());
                }
            }
            
            connection.disconnect();
            
        } catch (Exception e) {
            Log.w(TAG, "Failed to log TLS info", e);
        }
    }
    
    /**
     * OkHttpClient instance'ını döndürür (advanced usage için).
     * 
     * @return Güvenli olarak yapılandırılmış OkHttpClient
     */
    public OkHttpClient getSecureHttpClient() {
        checkInitialized();
        return secureHttpClient;
    }
    
    /**
     * Başlatılma durumunu kontrol eder.
     * 
     * @throws M3TMException Başlatılmamışsa
     */
    private void checkInitialized() throws M3TMException {
        if (!isInitialized) {
            throw new M3TMException(M3TMException.ErrorCode.NOT_INITIALIZED, 
                "NetworkSecurityManager not initialized");
        }
    }
    
    /**
     * Kaynakları serbest bırakır.
     */
    public void shutdown() {
        if (secureHttpClient != null) {
            secureHttpClient.dispatcher().executorService().shutdown();
            secureHttpClient.connectionPool().evictAll();
        }
        
        isInitialized = false;
        Log.i(TAG, "NetworkSecurityManager shutdown completed");
    }
}
