# API Reference

M³TM provides a comprehensive Python API for multimodal machine learning and mobile deployment. This section covers all public APIs, classes, and functions.

## Core Modules

### `m3tm.core`
Core functionality for model management and inference.

- **[Embedding Module](./core/embedding.md)** - Text and image embedding generation
- **[Fusion Module](./core/fusion.md)** - Multimodal feature fusion
- **[Model Management](./core/models.md)** - Model loading and management
- **[Cache System](./core/cache.md)** - Intelligent caching for performance

### `m3tm.mobile`
Mobile-specific optimization and deployment utilities.

- **[Optimization](./mobile/optimization.md)** - Model optimization for mobile
- **[Export Utilities](./mobile/export.md)** - Model export and conversion
- **[Performance Monitoring](./mobile/monitoring.md)** - Runtime performance tracking

### `m3tm.adapters`
Adapter pattern implementation for flexible model architectures.

- **[Base Adapters](./adapters/base.md)** - Abstract adapter interfaces
- **[Text Adapters](./adapters/text.md)** - Text processing adapters
- **[Vision Adapters](./adapters/vision.md)** - Image processing adapters
- **[Fusion Adapters](./adapters/fusion.md)** - Multimodal fusion adapters

### `m3tm.enterprise`
Enterprise features and integrations.

- **[Security](./enterprise/security.md)** - Enterprise security features
- **[Monitoring](./enterprise/monitoring.md)** - Advanced monitoring and analytics
- **[Integration](./enterprise/integration.md)** - Enterprise system integration

## Quick Reference

### Common Usage Patterns

```python
from m3tm.core import M3TMModel
from m3tm.mobile import optimize_for_mobile

# Load and optimize model
model = M3TMModel.from_pretrained("m3tm-base")
mobile_model = optimize_for_mobile(model, target_platform="android")
```

### Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `M3TMModel` | `m3tm.core` | Main model class for inference |
| `TextEmbedding` | `m3tm.core.embedding` | Text embedding generation |
| `ImageEmbedding` | `m3tm.core.embedding` | Image embedding generation |
| `FusionProcessor` | `m3tm.core.fusion` | Multimodal feature fusion |
| `MobileOptimizer` | `m3tm.mobile` | Mobile optimization utilities |

## Error Handling

All M³TM APIs use consistent error handling patterns:

```python
from m3tm.exceptions import M3TMError, ModelLoadError, OptimizationError

try:
    model = M3TMModel.from_pretrained("invalid-model")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except M3TMError as e:
    print(f"General M³TM error: {e}")
```

## Configuration

M³TM can be configured globally or per-operation:

```python
import m3tm

# Global configuration
m3tm.config.set_cache_dir("/path/to/cache")
m3tm.config.set_log_level("INFO")
m3tm.config.enable_mobile_optimizations(True)

# Per-operation configuration
model = M3TMModel.from_pretrained(
    "m3tm-base",
    cache_dir="/custom/cache",
    device="cuda"
)
```

## Advanced Usage

For advanced usage patterns, see:

- [Custom Adapters Guide](../guides/custom-adapters.md)
- [Performance Optimization](../guides/performance.md)
- [Enterprise Integration](../guides/enterprise.md)
- [Mobile Deployment](../guides/mobile-deployment.md)
