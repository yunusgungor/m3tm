# PyTorch Mobile Kullanım Rehberi

Bu dokümantasyon, M³TM projesinde PyTorch Mobile kullanımı için kapsamlı bir rehberdir.

## İçindekiler

1. [Giriş](#giriş)
2. [PyTorch Mobile Kurulumu](#pytorch-mobile-kurulumu)
3. [Model Dönüşümü](#model-dönüşümü)
4. [Android Entegrasyonu](#android-entegrasyonu)
5. [iOS Entegrasyonu](#ios-entegrasyonu)
6. [Optimizasyon Teknikleri](#optimizasyon-teknikleri)
7. [Performans Değerlendirmesi](#performans-değerlendirmesi)
8. [Bilinen Sorunlar ve Çözümler](#bilinen-sorunlar-ve-çözümler)

## Giriş

PyTorch Mobile, PyTorch modellerini mobil platformlarda (Android ve iOS) çalıştırmak için kullanılan hafif bir çerçevedir. Bu rehber, M³TM modelinin mobil cihazlarda çalıştırılması için gereken adımları içerir.

## PyTorch Mobile Kurulumu

### Gereksinimler

- PyTorch 1.12.0 veya daha yeni
- Android Studio 4.0+ (Android için)
- Xcode 12.0+ (iOS için)
- CMake 3.13+ (Kaynak koddan derleme için)

### Python Geliştirme Ortamı

```bash
# Gerekli Python paketlerini yükleyin
pip install torch torchvision

# Mobil entegrasyon için gerekli araçlar
pip install numpy pillow
```

## Model Dönüşümü

PyTorch modelinizi mobil uyumlu formata dönüştürmek için iki temel yaklaşım vardır: TorchScript ve ONNX. Bu rehber, projenin mevcut yaklaşımı olan TorchScript yöntemini açıklamaktadır.

### TorchScript'e Dönüştürme

```python
import torch
from m3tm.core.base_model import BaseModel
from m3tm.mobile.model_converter import ModelConverter

# Model yüklenir
model = BaseModel.from_pretrained("model/path")

# Örnek girdi oluşturulur (modelin beklediği şekle göre)
example_input = torch.rand(1, 32)  # Örnek: batch_size=1, embed_dim=32

# TorchScript modeline dönüştürülür
torchscript_model = ModelConverter.to_torchscript(
    model, 
    example_input,
    save_path="mobile_model.pt",
    optimize=True
)
```

### Mobil İçin İyileştirme

```python
from m3tm.mobile.optimization import MobileOptimizer

# Quantization ve diğer optimizasyonlar
optimized_model = MobileOptimizer.compress_model(
    model,
    example_input,
    methods=["quantize", "optimize"],
    save_path="optimized_mobile_model.pt"
)
```

## Android Entegrasyonu

### Android Projenizde PyTorch Mobile Kurulumu

Android uygulamanızın `build.gradle` dosyasına şu bağımlılıkları ekleyin:

```gradle
dependencies {
    implementation 'org.pytorch:pytorch_android:1.12.2'
    implementation 'org.pytorch:pytorch_android_torchvision:1.12.2'
}
```

### Python'dan Android'e Model Aktarımı

1. TorchScript'e dönüştürülmüş modeli Android projenizin `app/src/main/assets` dizinine kopyalayın.

```bash
cp optimized_mobile_model.pt android_app/app/src/main/assets/
```

### Android'de Model Kullanımı

```java
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

// Model yüklenir
Module module = Module.load(assetFilePath(this, "optimized_mobile_model.pt"));

// Girdi tensörü oluşturulur
float[] inputData = new float[32]; // Modelin beklediği girdi boyutu
// ... girdi verileri doldurulur
Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 32});

// Çıkarım yapılır
Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
float[] outputData = outputTensor.getDataAsFloatArray();
```

## iOS Entegrasyonu

### CocoaPods ile PyTorch Mobile Kurulumu

`Podfile` dosyanıza şu satırları ekleyin:

```ruby
target 'YourApp' do
  pod 'LibTorch', '~> 1.12.0'
end
```

Ardından pod'ları yükleyin:

```bash
pod install
```

### iOS'ta Model Kullanımı

```swift
import TorchModule

// Model yüklenir
let filePath = Bundle.main.path(forResource: "optimized_mobile_model", ofType: "pt")!
let module = TorchModule(fileAtPath: filePath)

// Girdi tensörü oluşturulur
var inputData = [Float](repeating: 0.0, count: 32)
// ... girdi verileri doldurulur

// Çıkarım yapılır
guard let outputTensor = module.forward(withInputs: [inputData]) else {
    fatalError("Model çıkarımı başarısız oldu!")
}
let outputData = outputTensor.data(as: Float.self)
```

## Optimizasyon Teknikleri

M³TM modelinin mobil cihazlarda verimli çalışması için birkaç önemli optimizasyon tekniği bulunmaktadır:

### 1. Quantization

Model ağırlıklarını ve aktivasyonlarını daha düşük bit derinliklerine (örn. int8) dönüştürerek model boyutunu küçültür ve çıkarım hızını artırır.

```python
quantized_model = MobileOptimizer.apply_dynamic_quantization(
    model,
    save_path="quantized_model.pt"
)
```

### 2. Pruning (Budama)

Modeldeki önemsiz ağırlıkları kaldırarak seyreklik oluşturur.

```python
pruned_model = MobileOptimizer.apply_pruning(
    model,
    amount=0.2,  # %20 parametreyi buda
    save_path="pruned_model.pt"
)
```

### 3. Karma Optimizasyon

En iyi sonuçlar için farklı teknikleri birleştirmek:

```python
optimized_model = MobileOptimizer.compress_model(
    model,
    example_input,
    methods=["quantize", "prune", "optimize"],
    save_path="fully_optimized_model.pt"
)
```

## Performans Değerlendirmesi

Model performansını ölçmek için `MobileBenchmark` sınıfını kullanabilirsiniz:

```python
from m3tm.mobile.benchmark import MobileBenchmark

# Farklı modelleri karşılaştırma
results = MobileBenchmark.compare_models(
    {
        "Original": original_model,
        "Quantized": quantized_model,
        "Fully Optimized": optimized_model
    },
    example_input,
    num_runs=100
)

# Rapor oluşturma
MobileBenchmark.generate_report(results, "benchmark_report.md")
```

## Bilinen Sorunlar ve Çözümler

### 1. Tip Dönüşümü Hatası

**Sorun:** Android veya iOS'ta model çalıştırılırken tip uyumsuzluğu hataları.

**Çözüm:** Model dönüşümü sırasında kullanılan örnek girdilerin, mobil uygulamada kullanılan girdilerle aynı şekil ve veri tipine sahip olduğundan emin olun.

### 2. Bellek Sınırlamaları

**Sorun:** Büyük modeller, mobil cihazlarda bellek sınırlamalarına çarpar.

**Çözüm:** 
- Quantization uygulayın
- Model parametrelerini budayın
- Modeli daha küçük alt modellere bölün

### 3. Desteklenmeyen Operatörler

**Sorun:** Bazı PyTorch operatörleri PyTorch Mobile'da desteklenmeyebilir.

**Çözüm:** TorchScript'e dönüştürmeden önce model mimarisini basitleştirin veya desteklenmeyen katmanları desteklenen alternatiflerle değiştirin.

### 4. Performans Sorunları

**Sorun:** Model mobil cihazda beklenenden yavaş çalışıyor.

**Çözüm:**
- `optimize_for_mobile` fonksiyonunu kullanın
- Daha agresif quantization uygulayın
- Girdi boyutlarını küçültün
- Batch inference yerine tekli çıkarım yapın

## Ek Kaynaklar

- [PyTorch Mobile Resmi Dokümantasyonu](https://pytorch.org/mobile/home/)
- [PyTorch Android Örnekleri](https://github.com/pytorch/android-demo-app)
- [PyTorch iOS Örnekleri](https://github.com/pytorch/ios-demo-app) 