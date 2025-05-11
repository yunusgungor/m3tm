# M³TM Android SDK

M³TM (Mobil Multi-Modal Modüler Transformer) modelini Android uygulamalarında kullanmak için geliştirilmiş bir SDK.

## Özellikler

- M³TM modellerini cihaz üzerinde çalıştırma
- Metin, görüntü ve çoklu-modalite çıkarım işlemleri
- Kullanıcı verisiyle modeli cihaz üzerinde eğitme ve ince ayar yapma
- PyTorch mobile optimizasyonu
- Hızlı ve verimli API

## Kurulum

### Gradle ile

```gradle
dependencies {
    implementation 'com.m3tm:sdk:1.0.0'
}
```

### El ile

1. `m3tm-sdk.aar` dosyasını indirin
2. Android Studio'da `File -> New -> New Module -> Import .JAR/.AAR Package` seçeneğini tıklayın
3. `m3tm-sdk.aar` dosyasını seçin
4. `app/build.gradle` dosyasına aşağıdaki bağımlılığı ekleyin:

```gradle
dependencies {
    implementation project(':m3tm-sdk')
}
```

## Kullanım

### Başlatma

```java
import com.m3tm.sdk.M3TMModelManager;
import com.m3tm.sdk.M3TMModel;
import com.m3tm.sdk.ModelException;

// Model yöneticisini başlat
M3TMModelManager modelManager = new M3TMModelManager();
```

### Model Yükleme

```java
try {
    // Modeli yükle
    M3TMModel model = modelManager.loadModel("m3tm_base");
} catch (ModelException e) {
    // Hata yönetimi
    e.printStackTrace();
}
```

### Metin İşleme

```java
try {
    Map<String, Object> result = model.processText("Örnek metin");
    // Sonuçları işle
} catch (InferenceException e) {
    // Hata yönetimi
    e.printStackTrace();
}
```

### Görüntü İşleme

```java
try {
    Bitmap image = BitmapFactory.decodeFile(imagePath);
    Map<String, Object> result = model.processImage(image);
    // Sonuçları işle
} catch (InferenceException e) {
    // Hata yönetimi
    e.printStackTrace();
}
```

### Çoklu-Modalite İşleme

```java
try {
    Bitmap image = BitmapFactory.decodeFile(imagePath);
    String text = "Görüntüdeki nesne nedir?";
    
    Map<String, Object> result = model.processMultimodal(text, image);
    // Sonuçları işle
} catch (InferenceException e) {
    // Hata yönetimi
    e.printStackTrace();
}
```

### Model Eğitimi

```java
import com.m3tm.sdk.M3TMTrainingManager;
import com.m3tm.sdk.TrainingCallback;
import com.m3tm.sdk.TrainingException;

// Eğitim yöneticisini başlat
M3TMTrainingManager trainingManager = new M3TMTrainingManager();

try {
    // Eğitim yapılandırması
    Map<String, Object> trainingConfig = new HashMap<>();
    trainingConfig.put("learning_rate", 0.0001);
    trainingConfig.put("epochs", 3);
    
    // Eğitim oturumu oluştur
    String sessionId = trainingManager.createTrainingSession("m3tm_base", trainingConfig);
    
    // Eğitim verileri
    Map<String, Object> trainingData = new HashMap<>();
    List<Map<String, Object>> samples = new ArrayList<>();
    
    // Eğitim örnekleri ekle
    Map<String, Object> sample1 = new HashMap<>();
    sample1.put("text", "Örnek metin 1");
    sample1.put("label", "Kategori A");
    samples.add(sample1);
    
    Map<String, Object> sample2 = new HashMap<>();
    sample2.put("text", "Örnek metin 2");
    sample2.put("label", "Kategori B");
    samples.add(sample2);
    
    trainingData.put("samples", samples);
    
    // Eğitim ilerleme callback'i
    TrainingCallback callback = new TrainingCallback() {
        @Override
        public void onBatchComplete(int batch, Map<String, Object> metrics) {
            // Batch ilerleme güncellemesi
        }
        
        @Override
        public void onEpochComplete(int epoch, Map<String, Object> metrics) {
            // Epoch ilerleme güncellemesi
        }
        
        @Override
        public void onTrainingComplete(Map<String, Object> metrics) {
            // Eğitim tamamlandı bildirimi
        }
    };
    
    // Eğitimi başlat
    Map<String, Object> trainingResults = trainingManager.startTraining(sessionId, trainingData, callback);
    
    // Eğitilmiş modeli kaydet
    Map<String, Object> saveResults = trainingManager.saveTrainedModel(sessionId, null, "m3tm_custom");
    
} catch (TrainingException e) {
    // Hata yönetimi
    e.printStackTrace();
}
```

### Kaynakları Serbest Bırakma

```java
// Kullanılan modelleri kapat
model.close();

// Model yöneticisini kapat
modelManager.close();

// Eğitim yöneticisini kapat
trainingManager.close();
```

## Gereksinin<mler

- Android API level 24+ (Android 7.0 Nougat veya üzeri)
- ARMv8 (AArch64) işlemci mimarisi (64-bit)
- Yeterli RAM ve depolama alanı (model boyutuna bağlı)

## Lisans

Bu SDK, M³TM modeli için özel lisans altında dağıtılmaktadır. Detaylı bilgi için lisans sözleşmesine bakınız.

## İletişim

Herhangi bir soru veya destek ihtiyacınız için: 
- E-posta: support@m3tm.com
- Web: https://m3tm.com 