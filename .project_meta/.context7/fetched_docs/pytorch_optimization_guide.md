# PyTorch Model Optimization Documentation (Context7 Cache)

## 1. Quantization Techniques

### Post Training Dynamic Quantization (PTDQ)
- **Kullanım Durumu**: LSTM ve Transformer modelleri, küçük batch boyutları
- **Avantajlar**: Ağırlıklar önceden quantize edilir, aktivasyonlar dinamik
- **API**: `torch.ao.quantization.quantize_dynamic()`

```python
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,  # original model
    {torch.nn.Linear},  # layers to quantize
    dtype=torch.qint8)  # target dtype
```

### Post Training Static Quantization (PTSQ)
- **Kullanım Durumu**: CNN modelleri, hem memory hem compute tasarruf
- **Gereksinimler**: Representative dataset ile kalibrasyon
- **Backend**: 'qnnpack' (mobile), 'x86' (server)

```python
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
# Calibration step
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
```

### Quantization Aware Training (QAT)
- **Kullanım Durumu**: En yüksek accuracy gerektiren durumlar
- **Süreç**: Training sırasında fake quantization kullanır

```python
model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())
# Training loop
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
```

## 2. Pruning Techniques

### Structured Pruning
- **Faydalar**: Gerçek hızlanma sağlar
- **Kullanım**: Channel/filter düzeyinde pruning

### Unstructured Pruning 
- **API**: `torch.nn.utils.prune`
- **Yöntemler**: L1, L2, random

```python
import torch.nn.utils.prune as prune
prune.l1_unstructured(module.conv1, name='weight', amount=0.2)
```

## 3. Mobile Optimization Best Practices

### Mobile Backend Configuration
```python
torch.backends.quantized.engine = 'qnnpack'  # Mobile ARM CPUs için
```

### Model Fusion
```python
model_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
```

### Performance Targets (Story 22 Hedefleri)
- Model boyutu: %60+ azalma
- Inference hızı: 2x+ hızlanma  
- Memory kullanımı: %50+ azalma
- Accuracy korunması: %95+ korunma

## 4. Efficient Mobile Architectures

### MobileNetV2/V3 Patterns
- **Inverted Residuals**: Expansion -> Depthwise -> Projection
- **Depthwise Separable Convolutions**: Spatial + channel separation
- **Squeeze-and-Excitation**: Channel attention mechanism

### Key Implementation Insights
- Use `torch.compile` for additional optimization
- Apply proper QuantStub/DeQuantStub for QAT
- Test with representative mobile hardware targets
