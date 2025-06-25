# Mobile Optimization API

The mobile optimization module provides tools for converting and optimizing M³TM models for mobile deployment across Android and iOS platforms.

## Classes

### `MobileOptimizer`

Main class for mobile model optimization.

```python
from m3tm.mobile import MobileOptimizer

optimizer = MobileOptimizer(
    target_platform="android",
    optimization_level="balanced"
)

# Optimize a model
optimized_model = optimizer.optimize(model)
```

#### Methods

##### `__init__(target_platform: str, optimization_level: str = "balanced", **kwargs)`

Initialize the mobile optimizer.

**Parameters:**
- `target_platform` (str): Target platform ("android", "ios", "both")
- `optimization_level` (str): Optimization level ("speed", "size", "balanced", "accuracy")
- `**kwargs`: Platform-specific optimization parameters

**Optimization Levels:**
- `"speed"`: Prioritize inference speed
- `"size"`: Minimize model size
- `"balanced"`: Balance between speed and accuracy
- `"accuracy"`: Preserve maximum accuracy

##### `optimize(model: M3TMModel) -> OptimizedModel`

Optimize a model for mobile deployment.

**Parameters:**
- `model` (M3TMModel): Model to optimize

**Returns:**
- `OptimizedModel`: Optimized model wrapper

**Example:**
```python
from m3tm.core import M3TMModel
from m3tm.mobile import MobileOptimizer

# Load original model
model = M3TMModel.from_pretrained("m3tm-base")

# Optimize for Android
optimizer = MobileOptimizer("android", "balanced")
optimized = optimizer.optimize(model)

# Save optimized model
optimized.save("./optimized_model")
```

##### `benchmark(model: M3TMModel, test_data: dict = None) -> dict`

Benchmark model performance before and after optimization.

**Parameters:**
- `model` (M3TMModel): Model to benchmark
- `test_data` (dict, optional): Custom test data

**Returns:**
- `dict`: Benchmark results including latency, memory usage, and accuracy

## Optimization Strategies

### `QuantizationOptimizer`

Specialized quantization optimization.

```python
from m3tm.mobile.optimization import QuantizationOptimizer

quantizer = QuantizationOptimizer(
    method="int8",
    calibration_data=sample_inputs
)

quantized_model = quantizer.apply(model)
```

#### Methods

##### `__init__(method: str = "int8", calibration_data: np.ndarray = None)`

Initialize quantization optimizer.

**Parameters:**
- `method` (str): Quantization method ("int8", "fp16", "dynamic")
- `calibration_data` (np.ndarray, optional): Calibration data for static quantization

### `PruningOptimizer`

Model pruning for size reduction.

```python
from m3tm.mobile.optimization import PruningOptimizer

pruner = PruningOptimizer(
    pruning_ratio=0.3,
    structured=True
)

pruned_model = pruner.apply(model)
```

### `DistillationOptimizer`

Knowledge distillation for creating smaller student models.

```python
from m3tm.mobile.optimization import DistillationOptimizer

distiller = DistillationOptimizer(
    student_architecture="mobile-small",
    temperature=4.0
)

student_model = distiller.distill(teacher_model, training_data)
```

## Model Export

### `export_to_onnx(model: M3TMModel, output_path: str, **kwargs) -> str`

Export model to ONNX format.

```python
from m3tm.mobile import export_to_onnx

onnx_path = export_to_onnx(
    model,
    "model.onnx",
    input_shapes=[(1, 3, 224, 224)],
    opset_version=11
)
```

### `export_to_torchscript(model: M3TMModel, output_path: str) -> str`

Export model to TorchScript format.

```python
from m3tm.mobile import export_to_torchscript

ts_path = export_to_torchscript(model, "model.pt")
```

### `export_to_tflite(model: M3TMModel, output_path: str, **kwargs) -> str`

Export model to TensorFlow Lite format.

```python
from m3tm.mobile import export_to_tflite

tflite_path = export_to_tflite(
    model,
    "model.tflite",
    quantization="int8",
    representative_dataset=calibration_data
)
```

## Performance Monitoring

### `MobileProfiler`

Profile model performance on mobile devices.

```python
from m3tm.mobile import MobileProfiler

profiler = MobileProfiler(device_type="android")

# Profile model
profile_results = profiler.profile(
    model,
    inputs=test_inputs,
    num_runs=100
)

print(f"Average latency: {profile_results['avg_latency']:.2f}ms")
print(f"Memory usage: {profile_results['peak_memory']:.1f}MB")
```

#### Methods

##### `profile(model: M3TMModel, inputs: dict, num_runs: int = 10) -> dict`

Profile model performance.

**Parameters:**
- `model` (M3TMModel): Model to profile
- `inputs` (dict): Input data for profiling
- `num_runs` (int): Number of profiling runs

**Returns:**
- `dict`: Profiling results

## Platform-Specific Optimization

### Android Optimization

```python
from m3tm.mobile.android import AndroidOptimizer

android_optimizer = AndroidOptimizer(
    target_api_level=28,
    use_nnapi=True,
    gpu_acceleration=True
)

android_model = android_optimizer.optimize(model)
```

### iOS Optimization

```python
from m3tm.mobile.ios import IOSOptimizer

ios_optimizer = IOSOptimizer(
    target_ios_version="13.0",
    use_coreml=True,
    neural_engine=True
)

ios_model = ios_optimizer.optimize(model)
```

## Deployment Utilities

### `create_deployment_package(optimized_model: OptimizedModel, platform: str) -> str`

Create deployment package for mobile app integration.

```python
from m3tm.mobile import create_deployment_package

package_path = create_deployment_package(
    optimized_model,
    platform="android",
    include_samples=True,
    include_docs=True
)
```

### `validate_mobile_model(model_path: str, platform: str) -> dict`

Validate mobile model compatibility.

```python
from m3tm.mobile import validate_mobile_model

validation_result = validate_mobile_model(
    "optimized_model.onnx",
    platform="android"
)

if validation_result['is_valid']:
    print("Model is ready for deployment!")
else:
    print(f"Issues found: {validation_result['issues']}")
```

## Example: Complete Mobile Optimization Pipeline

```python
from m3tm.core import M3TMModel
from m3tm.mobile import MobileOptimizer, export_to_onnx, validate_mobile_model

# Load and optimize model
model = M3TMModel.from_pretrained("m3tm-base")
optimizer = MobileOptimizer("android", "balanced")

# Apply optimizations
optimized = optimizer.optimize(model)

# Export to mobile format
onnx_path = export_to_onnx(optimized, "mobile_model.onnx")

# Validate for deployment
validation = validate_mobile_model(onnx_path, "android")

if validation['is_valid']:
    print("Model ready for Android deployment!")
    # Create deployment package
    package = create_deployment_package(optimized, "android")
    print(f"Deployment package: {package}")
```

See also:
- [Mobile Deployment Guide](../../guides/mobile-deployment.md)
- [Android Integration Tutorial](../../tutorials/android-integration.md)
- [iOS Integration Tutorial](../../tutorials/ios-integration.md)
