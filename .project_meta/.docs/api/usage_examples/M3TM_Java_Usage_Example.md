# M3TM Android SDK Kullanım Örneği (Java)

Bu belge, Android uygulamalarınızda M3TM SDK'yı Java ile nasıl kullanacağınızı göstermektedir.

## İçindekiler

1. [Kurulum](#kurulum)
2. [SDK'yı Başlatma](#sdkyı-başlatma)
3. [Model Yükleme](#model-yükleme)
4. [Çıkarım Yapma](#çıkarım-yapma)
5. [Arama Özelliğini Kullanma](#arama-özelliğini-kullanma)
6. [Eğitim İşlemleri](#eğitim-işlemleri)
7. [Veri Dışa Aktarma](#veri-dışa-aktarma)
8. [Hata Yönetimi](#hata-yönetimi)
9. [En İyi Uygulamalar](#en-işyi-uygulamalar)

## Kurulum

### Gradle ile

`build.gradle` (uygulama modülü) dosyanıza ekleyin:

```gradle
dependencies {
    implementation 'com.m3tm:android-sdk:2.3.0'
}
```

### Maven ile

```xml
<dependency>
    <groupId>com.m3tm</groupId>
    <artifactId>android-sdk</artifactId>
    <version>2.3.0</version>
</dependency>
```

## SDK'yı Başlatma

```java
import com.m3tm.sdk.M3TM;
import com.m3tm.sdk.M3TMConfig;
import com.m3tm.sdk.M3TMCallback;

public class MyApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        
        // SDK Yapılandırması
        M3TMConfig config = new M3TMConfig.Builder(getApplicationContext())
            .setModelPath("m3tm_model.pt")
            .setConfigPath("m3tm_config.json")
            .setComputeDevice(M3TMConfig.ComputeDevice.AUTO)
            .setOptimizationLevel(M3TMConfig.OptimizationLevel.BASIC)
            .setMaxMemoryUsageMB(256)
            .build();
        
        // SDK'yı başlat
        M3TM.getInstance().initialize(config, new M3TMCallback<Void>() {
            @Override
            public void onSuccess(Void result) {
                Log.d("M3TM", "SDK başarıyla başlatıldı!");
            }
            
            @Override
            public void onError(M3TMException error) {
                Log.e("M3TM", "SDK başlatma hatası: " + error.getMessage());
            }
        });
    }
}
```

## Model Yükleme

Model otomatik olarak başlatma sırasında yüklenir (yapılandırmada `ON_INITIALIZATION` seçeneği belirlenmişse). Manuel olarak yüklemek için:

```java
try {
    M3TMModelManager modelManager = M3TM.getInstance().getModelManager();
    modelManager.loadModel();
    Log.d("M3TM", "Model başarıyla yüklendi!");
} catch (M3TMException e) {
    Log.e("M3TM", "Model yükleme hatası: " + e.getMessage());
}
```

## Çıkarım Yapma

Çıkarım yapmak için:

```java
try {
    M3TMModelManager modelManager = M3TM.getInstance().getModelManager();
    
    // Giriş verisi oluştur
    M3TMModelManager.InputData inputData = new M3TMModelManager.InputData.Builder()
        .setText("Bu bir örnek metin verisidir")
        .build();
    
    // Çıkarım seçenekleri oluştur
    M3TMModelManager.InferenceOptions options = new M3TMModelManager.InferenceOptions.Builder()
        .setOutputFormat(M3TMModelManager.OutputFormat.DICTIONARY)
        .build();
    
    // Çıkarımı gerçekleştir
    modelManager.inference(Collections.singletonList(inputData), options, new M3TMCallback<List<Map<String, Object>>>() {
        @Override
        public void onSuccess(List<Map<String, Object>> results) {
            for (int i = 0; i < results.size(); i++) {
                Log.d("M3TM", "Sonuç " + i + ": " + results.get(i));
            }
        }
        
        @Override
        public void onError(M3TMException error) {
            Log.e("M3TM", "Çıkarım hatası: " + error.getMessage());
        }
    });
} catch (M3TMException e) {
    Log.e("M3TM", "Çıkarım hatası: " + e.getMessage());
}
```

Görüntü ile çıkarım yapmak için:

```java
try {
    M3TMModelManager modelManager = M3TM.getInstance().getModelManager();
    
    // Görüntü verisi ile giriş oluştur
    Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.example_image);
    M3TMModelManager.InputData inputData = new M3TMModelManager.InputData.Builder()
        .setImage(bitmap)
        .build();
    
    // Çıkarım seçenekleri oluştur
    M3TMModelManager.InferenceOptions options = new M3TMModelManager.InferenceOptions.Builder()
        .setOutputFormat(M3TMModelManager.OutputFormat.DICTIONARY)
        .build();
    
    // Çıkarımı gerçekleştir
    modelManager.inference(Collections.singletonList(inputData), options, new M3TMCallback<List<Map<String, Object>>>() {
        @Override
        public void onSuccess(List<Map<String, Object>> results) {
            for (int i = 0; i < results.size(); i++) {
                Log.d("M3TM", "Görüntü sonucu " + i + ": " + results.get(i));
            }
        }
        
        @Override
        public void onError(M3TMException error) {
            Log.e("M3TM", "Görüntü çıkarım hatası: " + error.getMessage());
        }
    });
} catch (M3TMException e) {
    Log.e("M3TM", "Çıkarım hatası: " + e.getMessage());
}
```

## Arama Özelliğini Kullanma

Arama özelliğini kullanmak için önce arama servisini başlatın:

```java
try {
    M3TMSearchService searchService = M3TM.getInstance().getSearchService();
    
    // Arama servisi başlatma
    searchService.initialize(new M3TMCallback<Void>() {
        @Override
        public void onSuccess(Void result) {
            Log.d("M3TM", "Arama servisi başarıyla başlatıldı!");
        }
        
        @Override
        public void onError(M3TMException error) {
            Log.e("M3TM", "Arama servisi başlatma hatası: " + error.getMessage());
        }
    });
} catch (M3TMException e) {
    Log.e("M3TM", "Arama servisi hatası: " + e.getMessage());
}
```

Öğe indeksleme:

```java
try {
    M3TMSearchService searchService = M3TM.getInstance().getSearchService();
    
    // Metadata oluştur
    Map<String, Object> metadata = new HashMap<>();
    metadata.put("category", "AI");
    metadata.put("date", System.currentTimeMillis() / 1000);
    
    // Bir metin öğesi indeksle
    searchService.indexItem(
        "doc_123", 
        "M3TM, gizlilik odaklı bir mobil yapay zeka modelidir.", 
        null,  // bitmap null
        metadata, 
        new M3TMCallback<Void>() {
            @Override
            public void onSuccess(Void result) {
                Log.d("M3TM", "Öğe başarıyla indekslendi!");
            }
            
            @Override
            public void onError(M3TMException error) {
                Log.e("M3TM", "İndeksleme hatası: " + error.getMessage());
            }
        }
    );
} catch (M3TMException e) {
    Log.e("M3TM", "Arama servisi hatası: " + e.getMessage());
}
```

Metin araması yapma:

```java
try {
    M3TMSearchService searchService = M3TM.getInstance().getSearchService();
    
    // Metin araması yap
    M3TMSearchService.SearchFilters filters = new M3TMSearchService.SearchFilters();
    filters.setContentTypes(Arrays.asList("text"));
    
    searchService.search("gizlilik yapay zeka", filters, 5, new M3TMCallback<List<M3TMSearchService.SearchResult>>() {
        @Override
        public void onSuccess(List<M3TMSearchService.SearchResult> results) {
            for (int i = 0; i < results.size(); i++) {
                M3TMSearchService.SearchResult result = results.get(i);
                Log.d("M3TM", "Arama sonucu " + i + ":");
                Log.d("M3TM", "  ID: " + result.getItemId());
                Log.d("M3TM", "  Skor: " + result.getScore());
                Log.d("M3TM", "  İçerik: " + (result.getTextContent() != null ? result.getTextContent() : "N/A"));
                
                if (result.getMetadata() != null) {
                    Log.d("M3TM", "  Metadata: " + result.getMetadata());
                }
            }
        }
        
        @Override
        public void onError(M3TMException error) {
            Log.e("M3TM", "Arama hatası: " + error.getMessage());
        }
    });
} catch (M3TMException e) {
    Log.e("M3TM", "Arama servisi hatası: " + e.getMessage());
}
```

## Eğitim İşlemleri

Bir adapter oluşturmak ve eğitmek:

```java
// Veri sağlayıcı tanımı
class MyDataProvider implements M3TMTrainingManager.DataProvider {
    private final String[] textData = {
        "Bu bir örnek eğitim metnidir", 
        "Bu başka bir örnektir", 
        "M3TM SDK'yı öğreniyorum"
    };
    private final int[] labels = {0, 1, 0};
    
    @Override
    public int getDataCount() {
        return textData.length;
    }
    
    @Override
    public List<M3TMModelManager.InputData> getData(int[] indices) {
        List<M3TMModelManager.InputData> data = new ArrayList<>();
        for (int index : indices) {
            M3TMModelManager.InputData inputData = new M3TMModelManager.InputData.Builder()
                .setText(textData[index])
                .build();
            data.add(inputData);
        }
        return data;
    }
    
    @Override
    public List<Object> getLabels(int[] indices) {
        List<Object> result = new ArrayList<>();
        for (int index : indices) {
            result.add(labels[index]);
        }
        return result;
    }
}

// Eğitim delegesi tanımı
class MyTrainingDelegate implements M3TMTrainingManager.TrainingDelegate {
    @Override
    public void onTrainingProgress(float progress, Map<String, Object> metrics, int epoch) {
        Log.d("M3TM", "Eğitim ilerlemesi: " + (progress * 100) + "%, Epoch: " + epoch);
    }
    
    @Override
    public void onEpochComplete(int epoch, Map<String, Object> metrics) {
        Log.d("M3TM", "Epoch " + epoch + " tamamlandı");
        if (metrics != null) {
            Log.d("M3TM", "Metrikler: " + metrics);
        }
    }
    
    @Override
    public void onTrainingComplete(boolean success, Map<String, Object> finalMetrics) {
        Log.d("M3TM", "Eğitim tamamlandı. Başarılı: " + success);
        if (finalMetrics != null) {
            Log.d("M3TM", "Final metrikler: " + finalMetrics);
        }
    }
    
    @Override
    public void onTrainingError(M3TMException error) {
        Log.e("M3TM", "Eğitim hatası: " + error.getMessage());
    }
}

// Eğitim işlemi
try {
    M3TMTrainingManager trainingManager = M3TM.getInstance().getTrainingManager();
    MyTrainingDelegate trainingDelegate = new MyTrainingDelegate();
    trainingManager.setTrainingDelegate(trainingDelegate);
    
    // Veri sağlayıcısı
    MyDataProvider dataProvider = new MyDataProvider();
    
    // Eğitim yapılandırması
    M3TMTrainingManager.TrainingConfig trainingConfig = new M3TMTrainingManager.TrainingConfig.Builder()
        .setLearningRate(0.001f)
        .setEpochs(5)
        .setBatchSize(2)
        .build();
    
    // Adapter oluşturma
    Map<String, Object> adapterConfig = new HashMap<>();
    adapterConfig.put("dims", 768);
    
    trainingManager.createAdapter(0, adapterConfig, new M3TMCallback<String>() {
        @Override
        public void onSuccess(String adapterId) {
            Log.d("M3TM", "Adapter oluşturuldu: " + adapterId);
            
            // Adapter'ı eğit
            trainingManager.trainAdapter(adapterId, dataProvider, trainingConfig, new M3TMCallback<Void>() {
                @Override
                public void onSuccess(Void result) {
                    Log.d("M3TM", "Adapter eğitimi tamamlandı!");
                }
                
                @Override
                public void onError(M3TMException error) {
                    Log.e("M3TM", "Adapter eğitim hatası: " + error.getMessage());
                }
            });
        }
        
        @Override
        public void onError(M3TMException error) {
            Log.e("M3TM", "Adapter oluşturma hatası: " + error.getMessage());
        }
    });
} catch (M3TMException e) {
    Log.e("M3TM", "Eğitim yöneticisi hatası: " + e.getMessage());
}
```

## Veri Dışa Aktarma

Kullanıcı verilerini dışa aktarmak:

```java
try {
    M3TMDataExporter dataExporter = M3TM.getInstance().getDataExporter();
    
    // Dışa aktarma servisini başlat
    dataExporter.initialize(new M3TMCallback<Void>() {
        @Override
        public void onSuccess(Void result) {
            Log.d("M3TM", "Dışa aktarma servisi başarıyla başlatıldı!");
            
            // Filtreler oluştur
            M3TMDataExporter.ExportFilters filters = new M3TMDataExporter.ExportFilters();
            
            Calendar calendar = Calendar.getInstance();
            calendar.add(Calendar.MONTH, -1);
            filters.setDateStart(calendar.getTime());
            filters.setContentTypes(Arrays.asList("text", "image"));
            
            // Seçenekler oluştur
            M3TMDataExporter.ExportOptions options = new M3TMDataExporter.ExportOptions.Builder()
                .setFormat(M3TMDataExporter.ExportFormat.JSON)
                .setIncludeImages(true)
                .setImageQuality(0.9f)
                .build();
            
            // Dışa aktarma işlemi
            dataExporter.exportData(filters, options, new M3TMCallback<M3TMDataExporter.ExportResult>() {
                @Override
                public void onSuccess(M3TMDataExporter.ExportResult result) {
                    Log.d("M3TM", "Dışa aktarma başarılı!");
                    Log.d("M3TM", "Dosya: " + result.getFile().getAbsolutePath());
                    Log.d("M3TM", "Aktarılan öğe sayısı: " + result.getItemCount());
                    
                    // Dışa aktarma tamamlandıktan sonra
                    // Dosyayı paylaşmak için intent kullanılabilir, örneğin:
                    Uri contentUri = FileProvider.getUriForFile(
                        context,
                        "com.your.package.fileprovider",
                        result.getFile()
                    );
                    
                    Intent shareIntent = new Intent();
                    shareIntent.setAction(Intent.ACTION_SEND);
                    shareIntent.setType("application/json");
                    shareIntent.putExtra(Intent.EXTRA_STREAM, contentUri);
                    shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                    
                    // Intent'i başlatın
                    // startActivity(Intent.createChooser(shareIntent, "Verilerinizi Paylaşın"));
                }
                
                @Override
                public void onError(M3TMException error) {
                    Log.e("M3TM", "Dışa aktarma hatası: " + error.getMessage());
                }
            });
        }
        
        @Override
        public void onError(M3TMException error) {
            Log.e("M3TM", "Dışa aktarma servisi başlatma hatası: " + error.getMessage());
        }
    });
} catch (M3TMException e) {
    Log.e("M3TM", "Dışa aktarma servisi hatası: " + e.getMessage());
}
```

## Hata Yönetimi

M3TM SDK, hataları `M3TMException` sınıfı üzerinden yönetir:

```java
try {
    // SDK işlemleri...
} catch (M3TMException e) {
    switch (e.getErrorCode()) {
        case M3TMException.NOT_INITIALIZED:
            Log.e("M3TM", "SDK başlatılmamış");
            break;
        case M3TMException.MODEL_NOT_LOADED:
            Log.e("M3TM", "Model yüklenmemiş");
            break;
        case M3TMException.TRAINING_IN_PROGRESS:
            Log.e("M3TM", "Zaten bir eğitim devam ediyor");
            break;
        case M3TMException.SEARCH_INDEX_NOT_INITIALIZED:
            Log.e("M3TM", "Arama indeksi başlatılmamış");
            break;
        case M3TMException.OUT_OF_MEMORY:
            Log.e("M3TM", "Bellek yetersiz");
            break;
        default:
            Log.e("M3TM", "Beklenmeyen hata: " + e.getMessage());
            break;
    }
}
```

## En İyi Uygulamalar

### Bellek Yönetimi

- Yüksek çözünürlüklü görüntülerle çalışırken resim boyutlarını kullanım öncesinde küçültün
- `maxMemoryUsageMB` sınırını uygulama ihtiyaçlarına göre ayarlayın
- Yaşam döngüsü olaylarını izleyin (örneğin `onLowMemory()` içinde `optimizeMemoryUsage()` çağırın)

### Performans

- Eğer yüksek performans gerekliyse, `optimizationLevel` değerini `SPEED` olarak ayarlayın
- Toplu (batch) çıkarım işlemleri yapın
- Arama indekslerini düzenli olarak kaydedin
- Mümkünse GPU desteği etkinleştirin

### Gizlilik

- Verileri yalnızca cihaz üzerinde işleyin
- Kullanıcılara kişisel verilerini indirme seçeneği sunun
- Hassas verileri cihazdan göndermeyin

### Yaşam Döngüsü

- SDK'yı Application sınıfının onCreate metodunda başlatın
- Uygulamanın sonlandırılması öncesinde SDK'nın shutdown ile kapatıldığından emin olun
- Arka plan servislerinde ExecutorService kullanarak ağır işlemleri yönetin
- onPause/onStop gibi durumlar için bellek temizleme stratejileri belirleyin 