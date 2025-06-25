# Troubleshooting Guide

This comprehensive troubleshooting guide helps you resolve common issues when working with M³TM models.

## Quick Diagnosis

### 🚨 Emergency Checklist

If your M³TM model isn't working, check these first:

1. **Python Environment**: `python --version` (3.8+ required)
2. **PyTorch Installation**: `python -c "import torch; print(torch.__version__)"`
3. **M³TM Installation**: `python -c "import m3tm; print('OK')"`
4. **GPU Availability**: `python -c "import torch; print(torch.cuda.is_available())"`
5. **Memory Available**: Check system memory usage

## Common Issues & Solutions

### 🔧 Installation Issues

#### Problem: ImportError when importing M³TM
```python
ImportError: No module named 'm3tm'
```

**Solutions:**
1. **Virtual Environment Check**:
   ```bash
   # Verify you're in the correct environment
   which python
   pip list | grep m3tm
   ```

2. **Fresh Installation**:
   ```bash
   pip uninstall m3tm
   pip install --upgrade m3tm
   ```

3. **Development Installation**:
   ```bash
   git clone https://github.com/your-org/m3tm.git
   cd m3tm
   pip install -e .
   ```

#### Problem: CUDA/PyTorch compatibility issues
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Check CUDA Version**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **Install Compatible PyTorch**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

### 🧠 Model Loading Issues

#### Problem: Model download fails
```
ConnectionError: Failed to download model
```

**Solutions:**
1. **Check Internet Connection**:
   ```python
   import requests
   response = requests.get('https://huggingface.co')
   print(response.status_code)  # Should be 200
   ```

2. **Manual Download**:
   ```python
   from m3tm.core import M3TMModel
   
   # Try with cache_dir specified
   model = M3TMModel.from_pretrained(
       "m3tm-base",
       cache_dir="./model_cache",
       force_download=True
   )
   ```

3. **Offline Mode**:
   ```python
   # Use local model files
   model = M3TMModel.from_pretrained("./local_model_path")
   ```

#### Problem: Model loading is slow
```
# Takes >5 minutes to load
```

**Solutions:**
1. **Enable Model Caching**:
   ```python
   import m3tm
   m3tm.config.set_cache_dir("/fast/ssd/cache")
   ```

2. **Use Smaller Models**:
   ```python
   # Instead of m3tm-large, use:
   model = M3TMModel.from_pretrained("m3tm-small")
   ```

3. **Memory Mapping**:
   ```python
   model = M3TMModel.from_pretrained(
       "m3tm-base",
       use_memory_mapping=True,
       low_cpu_mem_usage=True
   )
   ```

### 💾 Memory Issues

#### Problem: Out of memory during inference
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. **Reduce Batch Size**:
   ```python
   # Instead of batch_size=32
   results = model.inference_batch(texts, images, batch_size=4)
   ```

2. **Clear Cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   
   # Or use context manager
   with torch.cuda.device(0):
       torch.cuda.empty_cache()
   ```

3. **Use CPU for Large Inputs**:
   ```python
   model = model.to('cpu')
   result = model.inference(text, image)
   ```

4. **Enable Gradient Checkpointing**:
   ```python
   model.enable_gradient_checkpointing()
   ```

#### Problem: Memory leak during training
```
# Memory usage keeps increasing
```

**Solutions:**
1. **Explicit Cleanup**:
   ```python
   for batch in dataloader:
       outputs = model(batch)
       loss = compute_loss(outputs)
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
       
       # Explicit cleanup
       del outputs, loss
       torch.cuda.empty_cache()
   ```

2. **Use Context Managers**:
   ```python
   with torch.no_grad():
       for batch in validation_loader:
           outputs = model(batch)
           # Memory automatically freed
   ```

### ⚡ Performance Issues

#### Problem: Slow inference speed
```
# >1 second per inference
```

**Solutions:**
1. **Enable Optimizations**:
   ```python
   # Compile model (PyTorch 2.0+)
   model = torch.compile(model)
   
   # Use TorchScript
   traced_model = torch.jit.trace(model, example_inputs)
   ```

2. **Optimize Data Loading**:
   ```python
   from torch.utils.data import DataLoader
   
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=4,  # Parallel loading
       pin_memory=True,  # Faster GPU transfer
       prefetch_factor=2
   )
   ```

3. **Use Mixed Precision**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       outputs = model(inputs)
       loss = compute_loss(outputs)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

#### Problem: High CPU usage
```
# 100% CPU usage during inference
```

**Solutions:**
1. **Limit Thread Count**:
   ```python
   import torch
   torch.set_num_threads(4)  # Limit to 4 threads
   ```

2. **Use GPU**:
   ```python
   if torch.cuda.is_available():
       model = model.to('cuda')
       inputs = inputs.to('cuda')
   ```

### 📱 Mobile Deployment Issues

#### Problem: Model export fails
```
RuntimeError: Failed to export model to ONNX
```

**Solutions:**
1. **Check Model Compatibility**:
   ```python
   from m3tm.mobile import validate_mobile_model
   
   validation = validate_mobile_model(model)
   print(validation['issues'])
   ```

2. **Simplify Model**:
   ```python
   # Remove dynamic shapes
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(
       model,
       dummy_input,
       "model.onnx",
       input_names=['input'],
       output_names=['output'],
       dynamic_axes=None  # Remove dynamic axes
   )
   ```

3. **Use Mobile-Specific Export**:
   ```python
   from m3tm.mobile import export_for_mobile
   
   mobile_model = export_for_mobile(
       model,
       example_inputs,
       platform="android"
   )
   ```

#### Problem: Mobile model too large
```
# Model > 100MB, too large for mobile
```

**Solutions:**
1. **Apply Quantization**:
   ```python
   from m3tm.mobile import quantize_model
   
   quantized = quantize_model(model, method="int8")
   ```

2. **Use Knowledge Distillation**:
   ```python
   from m3tm.training import create_student_model, distill_knowledge
   
   student = create_student_model(teacher=model, compression=0.5)
   distilled = distill_knowledge(teacher, student, train_data)
   ```

3. **Prune Model**:
   ```python
   from m3tm.optimization import prune_model
   
   pruned = prune_model(model, sparsity=0.3)
   ```

### 🔐 Security & Privacy Issues

#### Problem: Data privacy concerns
```
# Need to ensure user data isn't stored
```

**Solutions:**
1. **Enable Privacy Mode**:
   ```python
   m3tm.config.enable_privacy_mode(True)
   m3tm.config.disable_telemetry(True)
   ```

2. **Local Processing Only**:
   ```python
   # Disable cloud features
   model = M3TMModel.from_pretrained(
       "m3tm-base",
       use_cloud_features=False,
       local_only=True
   )
   ```

3. **Data Anonymization**:
   ```python
   from m3tm.privacy import anonymize_embeddings
   
   embeddings = model.encode(text, image)
   anonymous_embeddings = anonymize_embeddings(embeddings)
   ```

## Advanced Debugging

### 🔍 Debug Mode

Enable detailed logging for debugging:

```python
import logging
import m3tm

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
m3tm.config.set_log_level("DEBUG")

# Enable model debugging
model = M3TMModel.from_pretrained("m3tm-base", debug=True)
```

### 📊 Performance Profiling

Profile your model to identify bottlenecks:

```python
from m3tm.utils import ProfilerContext

with ProfilerContext() as profiler:
    result = model.inference(text, image)

print(profiler.get_report())
```

### 🧪 Testing Framework

Use the built-in testing framework:

```python
from m3tm.testing import ModelTester

tester = ModelTester(model)
test_results = tester.run_comprehensive_tests()

if not test_results['all_passed']:
    print("Issues found:", test_results['failures'])
```

## Platform-Specific Issues

### 🐧 Linux Issues

#### Problem: Permission denied errors
```bash
# Create proper permissions
sudo chown -R $(whoami) ~/.cache/m3tm
chmod -R 755 ~/.cache/m3tm
```

#### Problem: Missing system dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++
```

### 🍎 macOS Issues

#### Problem: Apple Silicon compatibility
```bash
# Install ARM64 compatible packages
conda install pytorch torchvision -c pytorch-nightly

# Or use Rosetta for Intel packages
arch -x86_64 pip install m3tm
```

#### Problem: Xcode command line tools missing
```bash
xcode-select --install
```

### 🪟 Windows Issues

#### Problem: Visual Studio Build Tools missing
- Download and install Microsoft C++ Build Tools
- Or install Visual Studio Community with C++ development tools

#### Problem: Long path issues
```cmd
# Enable long paths in Windows
git config --global core.longpaths true
```

## Environment-Specific Solutions

### 🐳 Docker Issues

#### Problem: CUDA not available in container
```dockerfile
# Use NVIDIA base image
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install PyTorch with CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Problem: Container out of memory
```yaml
# docker-compose.yml
services:
  m3tm:
    image: m3tm:latest
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### ☁️ Cloud Platform Issues

#### Problem: AWS SageMaker deployment fails
```python
# Use SageMaker-compatible export
from m3tm.cloud import prepare_sagemaker_model

sagemaker_model = prepare_sagemaker_model(
    model,
    instance_type="ml.g4dn.xlarge"
)
```

#### Problem: Google Colab memory limits
```python
# Monitor memory usage in Colab
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    print(f"Available: {memory.available / 1024**3:.1f} GB")

check_memory()
```

### 📱 Mobile Platform Issues

#### Problem: Android app crashes
```java
// Add to AndroidManifest.xml
<application android:largeHeap="true">

// Check device capabilities
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
    // Use NNAPI
}
```

#### Problem: iOS app rejected
```swift
// Ensure privacy compliance
NSCameraUsageDescription = "App uses camera for image analysis"
NSPhotoLibraryUsageDescription = "App accesses photos for processing"
```

## Getting Help

### 📞 Support Channels

1. **GitHub Issues**: Report bugs and feature requests
2. **Discussion Forum**: Community Q&A and best practices
3. **Documentation**: Comprehensive guides and API reference
4. **Stack Overflow**: Tag questions with `m3tm`

### 🐛 Bug Report Template

When reporting issues, include:

```markdown
**Environment**:
- OS: [Windows 10 / macOS 12.0 / Ubuntu 20.04]
- Python version: [3.8.10]
- PyTorch version: [1.12.0]
- M³TM version: [1.0.0]
- CUDA version: [11.8] (if applicable)

**Issue Description**:
Brief description of the problem

**Steps to Reproduce**:
1. Load model with `M3TMModel.from_pretrained(...)`
2. Run inference with `model.inference(...)`
3. Error occurs

**Expected Behavior**:
What should happen

**Actual Behavior**:
What actually happens

**Error Message**:
```
Full error traceback here
```

**Additional Context**:
Any other relevant information
```

### 💡 Feature Request Template

```markdown
**Feature Description**:
Clear description of the proposed feature

**Use Case**:
Why this feature would be useful

**Proposed Implementation**:
How you think it could be implemented

**Alternatives Considered**:
Other approaches you've considered
```

## Prevention Best Practices

### ✅ Development Best Practices

1. **Virtual Environments**: Always use isolated environments
2. **Version Pinning**: Pin dependency versions in production
3. **Testing**: Write tests for your M³TM integrations
4. **Monitoring**: Set up performance monitoring
5. **Documentation**: Document your model configurations

### 🔒 Security Best Practices

1. **API Keys**: Never commit API keys to version control
2. **Data Validation**: Validate all inputs before processing
3. **Access Control**: Implement proper authentication
4. **Audit Logging**: Log all model access and usage
5. **Privacy**: Follow data protection regulations

### 🚀 Performance Best Practices

1. **Batch Processing**: Process multiple items together
2. **Caching**: Cache embeddings and results
3. **Resource Management**: Monitor and limit resource usage
4. **Model Optimization**: Use appropriate optimization techniques
5. **Load Testing**: Test under expected production loads

Remember: Most issues can be resolved by checking the basics first (environment, dependencies, inputs) before diving into complex debugging. When in doubt, start with a minimal reproducible example!
