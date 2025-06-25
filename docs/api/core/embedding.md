# Core Embedding API

The embedding module provides high-performance text and image embedding generation with caching and mobile optimization support.

## Classes

### `TextEmbedding`

Generates embeddings for text inputs using transformer models.

```python
from m3tm.core.embedding import TextEmbedding

embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding = embedder.encode("Hello world")
```

#### Methods

##### `__init__(model_name: str, device: str = "auto", cache_enabled: bool = True)`

Initialize the text embedding model.

**Parameters:**
- `model_name` (str): Name or path of the transformer model
- `device` (str, optional): Device to run on ("cpu", "cuda", "auto"). Defaults to "auto"
- `cache_enabled` (bool, optional): Enable embedding caching. Defaults to True

**Example:**
```python
# Basic initialization
embedder = TextEmbedding("all-MiniLM-L6-v2")

# Custom device and caching
embedder = TextEmbedding(
    "all-MiniLM-L6-v2", 
    device="cuda:0", 
    cache_enabled=False
)
```

##### `encode(texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray`

Generate embeddings for text inputs.

**Parameters:**
- `texts` (str or List[str]): Text(s) to encode
- `normalize` (bool, optional): Normalize embeddings to unit vectors. Defaults to True

**Returns:**
- `np.ndarray`: Embedding vectors with shape (num_texts, embedding_dim)

**Example:**
```python
# Single text
embedding = embedder.encode("Machine learning is fascinating")
print(embedding.shape)  # (384,)

# Multiple texts
embeddings = embedder.encode([
    "First document",
    "Second document",
    "Third document"
])
print(embeddings.shape)  # (3, 384)
```

##### `encode_batch(texts: List[str], batch_size: int = 32) -> np.ndarray`

Encode large batches of text efficiently.

**Parameters:**
- `texts` (List[str]): List of texts to encode
- `batch_size` (int, optional): Batch size for processing. Defaults to 32

**Returns:**
- `np.ndarray`: Embedding vectors

**Example:**
```python
large_corpus = ["Document " + str(i) for i in range(1000)]
embeddings = embedder.encode_batch(large_corpus, batch_size=64)
```

### `ImageEmbedding`

Generates embeddings for images using vision transformer models.

```python
from m3tm.core.embedding import ImageEmbedding
from PIL import Image

embedder = ImageEmbedding(model_name="openai/clip-vit-base-patch32")
image = Image.open("photo.jpg")
embedding = embedder.encode(image)
```

#### Methods

##### `__init__(model_name: str, device: str = "auto", preprocess_config: dict = None)`

Initialize the image embedding model.

**Parameters:**
- `model_name` (str): Name or path of the vision model
- `device` (str, optional): Device to run on. Defaults to "auto"
- `preprocess_config` (dict, optional): Custom preprocessing configuration

##### `encode(images: Union[Image.Image, List[Image.Image]]) -> np.ndarray`

Generate embeddings for image inputs.

**Parameters:**
- `images` (PIL.Image or List[PIL.Image]): Image(s) to encode

**Returns:**
- `np.ndarray`: Embedding vectors

**Example:**
```python
from PIL import Image

# Single image
image = Image.open("photo.jpg")
embedding = embedder.encode(image)

# Multiple images
images = [Image.open(f"photo_{i}.jpg") for i in range(5)]
embeddings = embedder.encode(images)
```

## Utility Functions

### `similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray`

Compute cosine similarity between embeddings.

```python
from m3tm.core.embedding import similarity

text_emb = text_embedder.encode("Machine learning")
image_emb = image_embedder.encode(image)
sim_score = similarity(text_emb, image_emb)
```

### `search(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, k: int = 5) -> List[int]`

Find k most similar embeddings in a corpus.

```python
from m3tm.core.embedding import search

query = "AI and robotics"
query_emb = embedder.encode(query)
top_indices = search(query_emb, corpus_embeddings, k=5)
```

## Configuration

### Cache Settings

```python
# Configure embedding cache
from m3tm.core.embedding import configure_cache

configure_cache(
    cache_dir="/path/to/cache",
    max_size_gb=10,
    ttl_hours=24
)
```

### Performance Tuning

```python
# Enable mixed precision for faster inference
embedder = TextEmbedding(
    "all-MiniLM-L6-v2",
    mixed_precision=True,
    batch_size=64
)
```

## Mobile Optimization

```python
from m3tm.mobile import optimize_embedding_model

# Optimize for mobile deployment
mobile_embedder = optimize_embedding_model(
    embedder,
    quantization="int8",
    target_platform="android"
)
```

See also:
- [Mobile Optimization Guide](../../guides/mobile-deployment.md)
- [Performance Best Practices](../../guides/performance.md)
