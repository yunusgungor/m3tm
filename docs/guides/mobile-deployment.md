# Mobile Deployment Guide

This guide covers end-to-end mobile deployment of M³TM models for Android and iOS applications.

## Overview

M³TM provides comprehensive mobile deployment capabilities that enable you to:

- Optimize models for mobile hardware constraints
- Convert models to mobile-friendly formats (ONNX, TorchScript, TensorFlow Lite)
- Integrate models into native Android and iOS applications
- Monitor and profile model performance on devices

## Quick Start

### 1. Model Optimization

```python
from m3tm.core import M3TMModel
from m3tm.mobile import MobileOptimizer

# Load your trained model
model = M3TMModel.from_pretrained("your-model-name")

# Create optimizer for target platform
optimizer = MobileOptimizer(
    target_platform="android",  # or "ios"
    optimization_level="balanced"
)

# Optimize the model
optimized_model = optimizer.optimize(model)
```

### 2. Model Export

```python
from m3tm.mobile import export_to_onnx, export_to_tflite

# Export to ONNX (recommended for most cases)
onnx_path = export_to_onnx(optimized_model, "model.onnx")

# Or export to TensorFlow Lite for specific use cases
tflite_path = export_to_tflite(optimized_model, "model.tflite")
```

### 3. Validation

```python
from m3tm.mobile import validate_mobile_model

# Validate model for target platform
result = validate_mobile_model(onnx_path, platform="android")
print(f"Model valid: {result['is_valid']}")
```

## Platform-Specific Deployment

### Android Deployment

#### Prerequisites
- Android Studio 4.0+
- Android SDK API level 21+
- NDK (for native performance)

#### Step 1: Optimize for Android

```python
from m3tm.mobile.android import AndroidOptimizer

android_optimizer = AndroidOptimizer(
    target_api_level=28,
    use_nnapi=True,          # Use Android Neural Networks API
    gpu_acceleration=True,    # Enable GPU acceleration
    quantization="int8"      # Quantize to int8 for smaller size
)

android_model = android_optimizer.optimize(model)
```

#### Step 2: Export Model

```python
# Export to ONNX with Android-specific optimizations
onnx_path = export_to_onnx(
    android_model,
    "android_model.onnx",
    optimize_for_mobile=True,
    use_external_data_format=False  # Keep model self-contained
)
```

#### Step 3: Integrate into Android App

Add to your `app/build.gradle`:

```gradle
dependencies {
    implementation 'org.pytorch:pytorch_android_lite:1.12.2'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.12.2'
    // OR for ONNX Runtime
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.12.0'
}
```

Java/Kotlin integration:

```kotlin
// Using ONNX Runtime
import ai.onnxruntime.*

class M3TMInference {
    private lateinit var ortSession: OrtSession
    
    fun loadModel(modelPath: String) {
        val ortEnvironment = OrtEnvironment.getEnvironment()
        ortSession = ortEnvironment.createSession(modelPath)
    }
    
    fun runInference(textInput: String, imageInput: FloatArray): FloatArray {
        // Prepare inputs
        val textTensor = OnnxTensor.createTensor(
            ortEnvironment, 
            tokenizeText(textInput)
        )
        val imageTensor = OnnxTensor.createTensor(
            ortEnvironment,
            imageInput,
            longArrayOf(1, 3, 224, 224)
        )
        
        // Run inference
        val inputs = mapOf(
            "text_input" to textTensor,
            "image_input" to imageTensor
        )
        
        val results = ortSession.run(inputs)
        return results.get("output").get().value as FloatArray
    }
}
```

### iOS Deployment

#### Prerequisites
- Xcode 12.0+
- iOS 13.0+ (for Core ML 3 support)
- macOS 10.15+ for development

#### Step 1: Optimize for iOS

```python
from m3tm.mobile.ios import IOSOptimizer

ios_optimizer = IOSOptimizer(
    target_ios_version="13.0",
    use_coreml=True,         # Use Core ML for optimal performance
    neural_engine=True,      # Enable Neural Engine acceleration
    quantization="fp16"      # Use half-precision for better performance
)

ios_model = ios_optimizer.optimize(model)
```

#### Step 2: Export to Core ML

```python
from m3tm.mobile import export_to_coreml

coreml_path = export_to_coreml(
    ios_model,
    "M3TMModel.mlmodel",
    input_descriptions={
        "text_input": "Tokenized text input",
        "image_input": "Preprocessed image tensor"
    },
    output_descriptions={
        "embeddings": "Multimodal embeddings"
    }
)
```

#### Step 3: Integrate into iOS App

Add the Core ML model to your Xcode project, then use it:

```swift
import CoreML
import Vision

class M3TMInference {
    private var model: M3TMModel?
    
    init() {
        do {
            model = try M3TMModel(configuration: MLModelConfiguration())
        } catch {
            print("Failed to load model: \(error)")
        }
    }
    
    func runInference(text: String, image: CGImage) -> [Float]? {
        guard let model = model else { return nil }
        
        do {
            // Prepare text input (tokenization would be done here)
            let textInput = try MLMultiArray(shape: [1, 512], dataType: .int32)
            
            // Prepare image input
            let imageInput = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
            // ... preprocess image into imageInput
            
            // Create input
            let input = M3TMModelInput(
                text_input: textInput,
                image_input: imageInput
            )
            
            // Run prediction
            let output = try model.prediction(input: input)
            
            // Convert output to array
            let embeddings = output.embeddings
            return Array(UnsafeBufferPointer(start: embeddings.dataPointer.bindMemory(to: Float.self, capacity: embeddings.count), count: embeddings.count))
            
        } catch {
            print("Inference error: \(error)")
            return nil
        }
    }
}
```

## Optimization Strategies

### Size Optimization

For apps with strict size constraints:

```python
# Aggressive size optimization
size_optimizer = MobileOptimizer(
    target_platform="android",
    optimization_level="size"
)

# Apply multiple techniques
optimized = size_optimizer.optimize(model)
pruned = PruningOptimizer(pruning_ratio=0.5).apply(optimized)
quantized = QuantizationOptimizer(method="int8").apply(pruned)
```

### Speed Optimization

For real-time applications:

```python
# Speed-focused optimization
speed_optimizer = MobileOptimizer(
    target_platform="ios",
    optimization_level="speed"
)

# Use specialized optimizations
optimized = speed_optimizer.optimize(model)
compiled = compile_for_neural_engine(optimized)  # iOS-specific
```

### Accuracy Preservation

For applications requiring high accuracy:

```python
# Accuracy-preserving optimization
accuracy_optimizer = MobileOptimizer(
    target_platform="android",
    optimization_level="accuracy"
)

# Use knowledge distillation instead of pruning
teacher_model = model
student_model = create_mobile_architecture("efficient")
distilled = DistillationOptimizer().distill(teacher_model, student_model, training_data)
```

## Performance Monitoring

### Benchmarking

```python
from m3tm.mobile import MobileProfiler

# Create profiler for target device
profiler = MobileProfiler(device_type="android")

# Profile your model
results = profiler.profile(
    model=optimized_model,
    inputs=test_inputs,
    num_runs=100
)

print(f"Average latency: {results['avg_latency']:.2f}ms")
print(f"Memory usage: {results['peak_memory']:.1f}MB")
print(f"CPU usage: {results['cpu_usage']:.1f}%")
```

### Real-time Monitoring

For production apps, implement monitoring:

```python
# Android monitoring example
from m3tm.mobile.monitoring import AndroidMonitor

monitor = AndroidMonitor()
monitor.start_monitoring()

# Run inference
result = model.inference(inputs)

# Get performance metrics
metrics = monitor.get_metrics()
```

## Common Issues and Solutions

### Memory Issues

**Problem**: Out of memory errors on device
**Solution**: 
- Reduce batch size to 1
- Use int8 quantization
- Enable memory mapping for large models

```python
# Memory-efficient configuration
optimizer = MobileOptimizer(
    target_platform="android",
    batch_size=1,
    use_memory_mapping=True,
    quantization="int8"
)
```

### Slow Inference

**Problem**: Inference takes too long
**Solution**:
- Enable hardware acceleration
- Optimize model architecture
- Use appropriate threading

```python
# Speed optimization
android_optimizer = AndroidOptimizer(
    use_nnapi=True,
    gpu_acceleration=True,
    num_threads=4
)
```

### Model Compatibility

**Problem**: Model doesn't work on target device
**Solution**:
- Check supported operations
- Use compatible export formats
- Validate before deployment

```python
# Comprehensive validation
validation = validate_mobile_model(
    model_path="model.onnx",
    platform="android",
    check_operations=True,
    check_performance=True
)
```

## Best Practices

### 1. Model Design
- Design models with mobile constraints in mind
- Use efficient architectures (MobileNet, EfficientNet)
- Minimize model complexity

### 2. Optimization Pipeline
- Always validate optimized models
- Test on real devices, not just emulators
- Use appropriate optimization levels

### 3. Integration
- Implement proper error handling
- Use asynchronous inference for UI responsiveness
- Cache models to avoid repeated loading

### 4. Testing
- Test on multiple device types and OS versions
- Monitor performance in production
- Have fallback strategies for unsupported devices

## Example: Complete Deployment Workflow

```python
import os
from m3tm.core import M3TMModel
from m3tm.mobile import MobileOptimizer, export_to_onnx, validate_mobile_model

def deploy_to_mobile(model_name: str, target_platform: str):
    """Complete mobile deployment workflow"""
    
    # 1. Load model
    print("Loading model...")
    model = M3TMModel.from_pretrained(model_name)
    
    # 2. Optimize for mobile
    print(f"Optimizing for {target_platform}...")
    optimizer = MobileOptimizer(
        target_platform=target_platform,
        optimization_level="balanced"
    )
    optimized = optimizer.optimize(model)
    
    # 3. Export model
    print("Exporting model...")
    output_path = f"{model_name}_{target_platform}.onnx"
    export_to_onnx(optimized, output_path)
    
    # 4. Validate
    print("Validating model...")
    validation = validate_mobile_model(output_path, target_platform)
    
    if validation['is_valid']:
        print(f"✅ Model successfully deployed to {output_path}")
        return output_path
    else:
        print(f"❌ Validation failed: {validation['issues']}")
        return None

# Deploy to both platforms
android_model = deploy_to_mobile("m3tm-base", "android")
ios_model = deploy_to_mobile("m3tm-base", "ios")
```

## Next Steps

- [Android Integration Tutorial](../tutorials/android-integration.md)
- [iOS Integration Tutorial](../tutorials/ios-integration.md)
- [Performance Optimization Guide](./performance.md)
- [Troubleshooting Guide](./troubleshooting.md)
