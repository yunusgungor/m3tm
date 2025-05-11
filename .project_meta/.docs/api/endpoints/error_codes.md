# M3TM SDK Hata Kodları ve Çözümleri

Bu belge, M³TM SDK'sı tarafından döndürülen hata kodlarını ve bunların olası çözümlerini açıklar. Hata durumları hem Android hem de iOS platformları için benzer şekilde yapılandırılmıştır, ancak hata sınıflarının adları ve kullanım şekilleri farklıdır.

## Platform Farklılıkları

### Android

Android'de, hatalar `M3TMException` sınıfı ile temsil edilir, bu da `Exception` sınıfından türetilmiştir. Hata kodları, `M3TMException` sınıfında statik sabitler olarak tanımlanmıştır ve hata nesnesi üzerindeki `getErrorCode()` metodu ile erişilebilir.

```java
try {
    // SDK işlemleri...
} catch (M3TMException e) {
    int errorCode = e.getErrorCode();
    String errorMessage = e.getMessage();
    Map<String, Object> errorDetails = e.getErrorDetails();
    
    // Hata koduna göre işlem
    switch (errorCode) {
        case M3TMException.NOT_INITIALIZED:
            // İşlem...
            break;
        // Diğer durumlar...
    }
}
```

### iOS

iOS'ta, hatalar `M3TMError` enum'u ile temsil edilir, bu da Swift'in `Error` protokolünü uygular. Hata kodları, enum durumları olarak tanımlanır.

```swift
do {
    // SDK işlemleri...
} catch let error as M3TMError {
    // Hata durumuna göre işlem
    switch error {
        case .notInitialized:
            // İşlem...
            break;
        // Diğer durumlar...
    }
} catch {
    // Diğer hatalar
}
```

## Hata Kategorileri ve Kodları

### 1000 - SDK Yaşam Döngüsü Hataları

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 1001 | `NOT_INITIALIZED` | `notInitialized` | SDK başlatılmadan önce bir işlem çağrıldı | SDK'yı `M3TM.initialize()` ile başlatın ve başarılı olduğundan emin olun |
| 1002 | `ALREADY_INITIALIZED` | `alreadyInitialized` | SDK zaten başlatılmışken tekrar başlatılmaya çalışıldı | `M3TM.isInitialized()` ile başlatma durumunu kontrol edin veya önce `shutdown()` çağırın |

### 2000 - Model Hataları

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 2001 | `MODEL_LOAD_FAILED` | `modelLoadFailed` | Model yüklenirken bir hata oluştu | Model dosyasının doğru yolda olduğunu ve erişilebilir olduğunu doğrulayın |
| 2002 | `MODEL_NOT_LOADED` | `modelNotLoaded` | Model yüklenmeden bir işlem çağrıldı | Model yöneticisi aracılığıyla `loadModel()` çağırın |
| 2003 | `MODEL_VERSION_MISMATCH` | `modelVersionMismatch` | Model sürümü SDK sürümüyle uyumlu değil | SDK ve modelin uyumlu sürümlerini kullanın |
| 2004 | `MODULE_NOT_SUPPORTED` | `moduleNotSupported` | İstenen modül, yüklenen modelde mevcut değil | Tam model paketini yükleyin veya eksik modül dışındaki işlemleri kullanın |
| 2005 | `MODEL_CONFIGURATION_ERROR` | `modelConfigurationError` | Model yapılandırması geçersiz | Yapılandırma dosyasını doğrulayın ve geçerli bir yapılandırma olduğundan emin olun |

### 3000 - Eğitim Hataları

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 3001 | `TRAINING_FAILED` | `trainingFailed` | Eğitim işlemi sırasında bir hata oluştu | Eğitim verilerini kontrol edin ve yeterli bellek olduğundan emin olun |
| 3002 | `TRAINING_IN_PROGRESS` | `trainingInProgress` | Zaten bir eğitim işlemi devam ederken yeni bir eğitim başlatılmaya çalışıldı | Mevcut eğitimin tamamlanmasını bekleyin veya `cancelTraining()` ile iptal edin |
| 3003 | `TRAINING_CANCELLED` | `trainingCancelled` | Eğitim kullanıcı veya sistem tarafından iptal edildi | Eğitimi tekrar başlatın veya daha düşük bir bellek ayarıyla çalıştırın |
| 3004 | `TRAINING_DATA_INVALID` | `trainingDataInvalid` | Eğitim verileri geçersiz veya eksik | Veri sağlayıcıdan gelen verilerin doğru formatta olduğunu kontrol edin |
| 3005 | `ADAPTER_CREATION_FAILED` | `adapterCreationFailed` | Adapter oluşturulurken bir hata oluştu | Adapter yapılandırmasını kontrol edin ve model ile uyumlu olduğundan emin olun |
| 3006 | `TASK_HEAD_CREATION_FAILED` | `taskHeadCreationFailed` | Görev başlığı oluşturulurken bir hata oluştu | Görev başlığı yapılandırmasını kontrol edin ve model ile uyumlu olduğundan emin olun |

### 4000 - Çıkarım Hataları

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 4001 | `INFERENCE_FAILED` | `inferenceFailed` | Çıkarım işlemi sırasında bir hata oluştu | Giriş verilerini kontrol edin ve modelin doğru yüklendiğinden emin olun |
| 4002 | `INPUT_DATA_INVALID` | `inputDataInvalid` | Giriş verileri geçersiz veya eksik | Giriş verilerinin doğru biçimde oluşturulduğunu kontrol edin |
| 4003 | `OUTPUT_FORMAT_ERROR` | `outputFormatError` | Çıkış formatı ile ilgili bir hata oluştu | İstenilen çıkış formatının model tarafından desteklenip desteklenmediğini kontrol edin |

### 5000 - Arama Hataları

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 5001 | `SEARCH_INDEX_NOT_INITIALIZED` | `searchIndexNotInitialized` | Arama indeksi başlatılmadan bir arama işlemi çağrıldı | Arama servisini `initialize()` ile başlatın |
| 5002 | `SEARCH_FAILED` | `searchFailed` | Arama işlemi sırasında bir hata oluştu | Arama sorgusunu kontrol edin ve indeksin doğru yüklendiğinden emin olun |
| 5003 | `ITEM_NOT_FOUND` | `itemNotFound` | Belirtilen öğe indekste bulunamadı | Öğe kimliğinin doğru olduğunu kontrol edin veya öğeyi indekse ekleyin |
| 5004 | `INDEX_OPERATION_FAILED` | `indexOperationFailed` | İndeksleme işlemi başarısız oldu | Bellek durumunu kontrol edin ve indeksin desteklenen boyutları aşmadığını doğrulayın |

### 6000 - Dışa Aktarma Hataları

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 6001 | `EXPORTER_NOT_INITIALIZED` | `exporterNotInitialized` | Dışa aktarma servisi başlatılmadan bir dışa aktarma işlemi çağrıldı | Dışa aktarma servisini `initialize()` ile başlatın |
| 6002 | `EXPORT_FAILED` | `exportFailed` | Veri dışa aktarma işlemi başarısız oldu | Depolama izninin verildiğini ve yeteri kadar boş alan olduğunu kontrol edin |

### 7000 - Dosya ve İzin Hataları

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 7001 | `FILE_IO_ERROR` | `fileIOError` | Dosya okuma/yazma hatası oluştu | Dosya izinlerini ve depolama alanını kontrol edin |
| 7002 | `PERMISSION_DENIED` | `permissionDenied` | İzin verilmedi (örn. depolama erişimi) | Gerekli izinlerin uygulama tarafından istendiğini ve kullanıcı tarafından verildiğini kontrol edin |

### 8000 - Sistem Hataları

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 8001 | `OUT_OF_MEMORY` | `outOfMemory` | Bellek yetersiz | Bellek kullanımını azaltın, `maxMemoryUsageMB` ayarını düşürün veya uygulamayı yeniden başlatın |
| 8002 | `TIMEOUT` | `timeout` | İşlem zaman aşımına uğradı | İşlemi tekrar deneyin veya daha küçük veri kümeleriyle çalışın |

### 9000 - Genel Hatalar

| Kod | Android Sabiti | iOS Enum Değeri | Açıklama | Çözüm |
|-----|---------------|-----------------|-----------|-------|
| 9001 | `INTERNAL_ERROR` | `internalError` | SDK içinde beklenmeyen bir hata oluştu | Sorunu bildirin ve uygulamayı yeniden başlatın |
| 9002 | `UNSUPPORTED_OPERATION` | `unsupportedOperation` | Çağrılan işlem mevcut SDK sürümünde desteklenmiyor | SDK'nın daha yeni bir sürümünü kullanın veya alternatif bir yaklaşım bulun |

## Yaygın Hata Senaryoları ve Çözümleri

### "SDK başlatılmamış" Hatası (1001)

**Belirtiler:** `NOT_INITIALIZED` / `notInitialized` hatası alıyorsunuz.

**Çözüm:**
1. SDK'yı uygulama başlangıcında başlattığınızdan emin olun:
   ```java
   // Android
   M3TM.getInstance().initialize(config, callback);
   ```
   ```swift
   // iOS
   M3TM.shared.initialize(config: config, completion: completion)
   ```
2. Başlatma işleminin başarıyla tamamlandığını doğrulayın.
3. SDK metodlarını çağırmadan önce `isInitialized` durumunu kontrol edin.

### Bellek Yetersizliği (8001)

**Belirtiler:** `OUT_OF_MEMORY` / `outOfMemory` hatası alıyorsunuz.

**Çözüm:**
1. Yapılandırmada `maxMemoryUsageMB` değerini azaltın.
2. Daha küçük toplu işlemlerle çalışın.
3. Kullanılmayan kaynakları serbest bırakın.
4. Özellikle yüksek çözünürlüklü görüntüleri işlemeden önce küçültün.
5. Kullanılmayan model bileşenlerini `unloadModel()` ile boşaltın.

### Model Yükleme Hatası (2001)

**Belirtiler:** `MODEL_LOAD_FAILED` / `modelLoadFailed` hatası alıyorsunuz.

**Çözüm:**
1. Model dosyasının doğru yolda olduğunu kontrol edin.
2. Model dosyasının doğru formatta olduğunu doğrulayın.
3. Model ve SDK sürüm uyumluluğunu kontrol edin.
4. Yeterli depolama alanı ve bellek olduğundan emin olun.
5. Asset klasöründeki veya bundle içindeki model dosyalarına erişim izinlerini kontrol edin.

### İndeksleme İşlemi Başarısız (5004)

**Belirtiler:** `INDEX_OPERATION_FAILED` / `indexOperationFailed` hatası alıyorsunuz.

**Çözüm:**
1. Arama servisinin başlatıldığını doğrulayın.
2. İndekslenen verilerin geçerli olduğunu kontrol edin.
3. İndeks boyutunu kontrol edin, çok büyük olabilir.
4. Depolama izinlerini kontrol edin.
5. Eskimiş indeksleri temizleyin ve yeniden oluşturun.

## Hata Bildirim ve Günlük Kaydı

Gidermeniz mümkün olmayan hatalarla karşılaşırsanız, şu ayrıntıları içeren bir hata raporu gönderin:

1. SDK Sürümü (`M3TM.getInstance().getVersion()` / `M3TM.shared.version`)
2. Hata Kodu
3. Hata Mesajı
4. Hata Detayları (varsa)
5. Adım Adım Yeniden Oluşturma
6. Cihaz Modeli ve İşletim Sistemi Sürümü

Hataları daha iyi tespit edebilmek için günlük kaydını etkinleştirebilirsiniz:

### Android
```java
M3TMConfig config = new M3TMConfig.Builder(context)
    // Diğer ayarlar...
    .setLogLevel(M3TMConfig.LOG_LEVEL_VERBOSE)
    .build();
```

### iOS
```swift
let config = M3TMConfig(modelPath: "m3tm_model.pt", configPath: "m3tm_config.json")
config.logLevel = M3TMConfig.LogLevel.verbose.rawValue
``` 