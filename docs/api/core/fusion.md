# Fusion Module API

The fusion module provides multimodal feature fusion capabilities, allowing combination of text and image embeddings for enhanced understanding.

## Classes

### `FusionProcessor`

Core class for multimodal feature fusion.

```python
from m3tm.core.fusion import FusionProcessor

fusion = FusionProcessor(
    fusion_method="attention",
    output_dim=512
)

# Fuse text and image embeddings
text_emb = text_embedder.encode("A beautiful sunset")
image_emb = image_embedder.encode(sunset_image)
fused_emb = fusion.fuse(text_emb, image_emb)
```

#### Methods

##### `__init__(fusion_method: str = "concat", output_dim: int = None, **kwargs)`

Initialize the fusion processor.

**Parameters:**
- `fusion_method` (str): Fusion strategy ("concat", "attention", "gated", "bilinear")
- `output_dim` (int, optional): Output embedding dimension
- `**kwargs`: Method-specific parameters

**Fusion Methods:**
- `"concat"`: Simple concatenation of embeddings
- `"attention"`: Cross-attention fusion mechanism
- `"gated"`: Gated fusion with learned weights
- `"bilinear"`: Bilinear pooling fusion

##### `fuse(text_embedding: np.ndarray, image_embedding: np.ndarray) -> np.ndarray`

Fuse text and image embeddings.

**Parameters:**
- `text_embedding` (np.ndarray): Text embedding vector(s)
- `image_embedding` (np.ndarray): Image embedding vector(s)

**Returns:**
- `np.ndarray`: Fused multimodal embedding

**Example:**
```python
# Single pair fusion
fused = fusion.fuse(text_emb, image_emb)

# Batch fusion
text_batch = text_embedder.encode(["Text 1", "Text 2"])
image_batch = image_embedder.encode([img1, img2])
fused_batch = fusion.fuse(text_batch, image_batch)
```

##### `train_fusion_weights(text_embeddings: np.ndarray, image_embeddings: np.ndarray, labels: np.ndarray)`

Train fusion parameters on labeled data.

**Parameters:**
- `text_embeddings` (np.ndarray): Training text embeddings
- `image_embeddings` (np.ndarray): Training image embeddings  
- `labels` (np.ndarray): Training labels for supervised learning

### `AttentionFusion`

Specialized fusion using cross-attention mechanisms.

```python
from m3tm.core.fusion import AttentionFusion

attention_fusion = AttentionFusion(
    num_heads=8,
    hidden_dim=256,
    dropout=0.1
)
```

#### Methods

##### `__init__(num_heads: int = 8, hidden_dim: int = 256, dropout: float = 0.1)`

Initialize attention-based fusion.

**Parameters:**
- `num_heads` (int): Number of attention heads
- `hidden_dim` (int): Hidden dimension for attention computation
- `dropout` (float): Dropout rate for regularization

##### `fuse_with_attention(text_emb: np.ndarray, image_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Perform fusion with attention weights.

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (fused_embedding, attention_weights)

## Utility Functions

### `compute_fusion_similarity(fused_emb1: np.ndarray, fused_emb2: np.ndarray) -> float`

Compute similarity between fused embeddings.

```python
from m3tm.core.fusion import compute_fusion_similarity

sim = compute_fusion_similarity(fused_emb1, fused_emb2)
```

### `evaluate_fusion_quality(fusion_processor: FusionProcessor, test_data: dict) -> dict`

Evaluate fusion quality on test data.

```python
from m3tm.core.fusion import evaluate_fusion_quality

metrics = evaluate_fusion_quality(
    fusion_processor=fusion,
    test_data={
        'text_embeddings': test_text_embs,
        'image_embeddings': test_image_embs,
        'labels': test_labels
    }
)
```

## Advanced Configuration

### Custom Fusion Architectures

```python
from m3tm.core.fusion import CustomFusion

# Define custom fusion architecture
custom_fusion = CustomFusion(
    architecture_config={
        'layers': [
            {'type': 'linear', 'in_dim': 768, 'out_dim': 512},
            {'type': 'relu'},
            {'type': 'dropout', 'p': 0.2},
            {'type': 'linear', 'in_dim': 512, 'out_dim': 256}
        ]
    }
)
```

### Adaptive Fusion

```python
from m3tm.core.fusion import AdaptiveFusion

# Automatically select best fusion method
adaptive_fusion = AdaptiveFusion(
    methods=['concat', 'attention', 'gated'],
    selection_criterion='validation_accuracy'
)

# Train and select best method
adaptive_fusion.fit(train_text_embs, train_image_embs, train_labels)
best_method = adaptive_fusion.get_best_method()
```

## Mobile Optimization

```python
from m3tm.mobile import optimize_fusion_model

# Optimize fusion for mobile deployment
mobile_fusion = optimize_fusion_model(
    fusion_processor,
    quantization="int8",
    pruning_ratio=0.3
)
```

## Example: Multi-stage Fusion Pipeline

```python
from m3tm.core.fusion import FusionPipeline

# Create multi-stage fusion pipeline
pipeline = FusionPipeline([
    ('early_fusion', FusionProcessor(fusion_method='concat')),
    ('attention', AttentionFusion(num_heads=4)),
    ('late_fusion', FusionProcessor(fusion_method='gated'))
])

# Process through pipeline
result = pipeline.process(text_emb, image_emb)
```

See also:
- [Fusion Architecture Guide](../../guides/fusion-architectures.md)
- [Multimodal Training Tutorial](../../tutorials/multimodal-training.md)
