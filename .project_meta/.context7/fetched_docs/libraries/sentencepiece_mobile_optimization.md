# SentencePiece Mobile Optimization Documentation

## Overview
SentencePiece is Google's unsupervised text tokenizer for Neural Network-based text generation, crucial for our mobile model's text processing capabilities.

## Mobile Optimization Best Practices

### Memory Efficiency
- **TCMalloc Integration**: Use TCMalloc for optimized memory allocation
  ```cmake
  if (SPM_ENABLE_TCMALLOC)
    find_library(TCMALLOC_LIB NAMES libtcmalloc_minimal.a)
    list(APPEND SPM_LIBS ${TCMALLOC_LIB})
    add_definitions(-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free)
  endif()
  ```

- **Vocabulary Restriction**: Limit vocabulary to reduce memory footprint
  ```python
  # Only use tokens appearing more than threshold times
  vocabs = list(filter(lambda x: x in freq and freq[x] > 1000, vocabs))
  sp.set_vocabulary(vocabs)
  ```

### Performance Optimizations

#### Model Loading
- **Load from Serialized Proto**: Avoid file I/O overhead
  ```python
  serialized_model_proto = tf.io.gfile.GFile('m.model', 'rb').read()
  sp = spm.SentencePieceProcessor()
  sp.load_from_serialized_proto(serialized_model_proto)
  ```

- **In-Memory Model**: Train and store models in memory
  ```python
  model = io.BytesIO()
  spm.SentencePieceTrainer.train(sentence_iterator=response, model_writer=model, vocab_size=1000)
  sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
  ```

#### Encoding Strategies
- **Batch Processing**: Process multiple texts together
  ```python
  sp.encode(['This is a test', 'Hello world'], out_type=str)
  ```

- **Sampling for Data Augmentation**: Use stochastic segmentation
  ```python
  sp.encode('This is a test', out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
  ```

### Model Type Selection for Mobile

#### Unigram Model (Recommended for Mobile)
```python
spm.SentencePieceTrainer.train('--input=data.txt --model_prefix=m_unigram --vocab_size=2000 --model_type=unigram')
```
- **Advantages**: Supports probabilistic segmentation, n-best results
- **Use Case**: When model robustness and uncertainty estimation are important

#### BPE Model (Lightweight Alternative)
```python
spm.SentencePieceTrainer.train('--input=data.txt --model_prefix=m_bpe --vocab_size=2000 --model_type=bpe')
```
- **Advantages**: Deterministic, faster training
- **Use Case**: When consistency and speed are prioritized

#### Character Model (Minimal Footprint)
```python
spm.SentencePieceTrainer.train('--input=data.txt --model_prefix=m_char --model_type=char --vocab_size=2000')
```
- **Advantages**: Smallest vocabulary size, handles any character
- **Use Case**: Resource-constrained environments

### Mobile-Specific Configuration

#### Optimized Training Parameters
```bash
spm_train \
  --input=training_data.txt \
  --model_prefix=mobile_model \
  --vocab_size=8000 \
  --model_type=unigram \
  --character_coverage=0.9995 \
  --input_sentence_size=1000000 \
  --shuffle_input_sentence=true \
  --bos_id=1 \
  --eos_id=2 \
  --unk_id=0
```

#### Control Symbols for Mobile
```python
# Use control symbols instead of user-defined for production
spm.SentencePieceTrainer.train(
    '--input=data.txt --model_prefix=m_ctrl --control_symbols=<sep>,<cls> --vocab_size=2000'
)
```

### C++ Integration for Android/iOS

#### Basic Usage Pattern
```cpp
#include <sentencepiece_processor.h>

sentencepiece::SentencePieceProcessor processor;
const auto status = processor.Load("model.model");

// Encoding
std::vector<int> ids;
processor.Encode("Input text", &ids);

// Decoding
std::string text;
processor.Decode(ids, &text);
```

#### Advanced Features
```cpp
// Sampling for robustness
std::vector<std::string> pieces;
processor.SampleEncode("Input text", &pieces, -1, 0.2);

// Access detailed information
sentencepiece::ImmutableSentencePieceText spt;
processor.Encode("Input text", spt.mutable_proto());
for (const auto &piece : spt.pieces()) {
    std::cout << piece.piece() << " " << piece.id() << std::endl;
}
```

### Error Handling and Validation

#### Model Validation
```cpp
const auto status = processor.Load("model.model");
if (!status.ok()) {
    std::cerr << "Failed to load model: " << status.ToString() << std::endl;
    return false;
}
```

#### Vocabulary Checks
```python
# Check vocabulary size and special tokens
print(f"Vocabulary size: {sp.get_piece_size()}")
print(f"UNK ID: {sp.unk_id()}")
print(f"BOS ID: {sp.bos_id()}")
print(f"EOS ID: {sp.eos_id()}")
```

### Build Configuration for Mobile

#### CMake Configuration
```cmake
# Enable optimizations
set(SPM_ENABLE_TCMALLOC ON)
set(SPM_TCMALLOC_STATIC ON)

# Core executables
add_executable(spm_normalize spm_normalize_main.cc)
add_executable(spm_train spm_train_main.cc)
add_executable(spm_export_vocab spm_export_vocab_main.cc)
```

#### Dependencies
```bash
# Ubuntu/Linux
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev

# For mobile cross-compilation, ensure proper toolchain setup
```

### Integration Patterns

#### Subword Regularization for Robustness
```python
# Training-time augmentation
for n in range(10):
    pieces = sp.sample_encode_as_pieces('hello world', -1, 0.1)
    # Use different segmentations for training
```

#### Vocabulary Alignment
```bash
# Ensure vocabulary consistency across languages
spm_encode --model=model.model --vocabulary=vocab.L1 --vocabulary_threshold=50 < test.L1 > test.seg.L1
```

### Performance Monitoring

#### Encoding Speed
- Monitor tokens per second for different model types
- Profile memory usage during batch processing
- Test on target mobile hardware

#### Model Size Optimization
- Balance vocabulary size vs. segmentation quality
- Consider model compression techniques
- Validate inference speed on mobile devices

### Key Constraints for Mobile Development

1. **Memory Usage**: Keep vocabulary size reasonable (8K-32K tokens)
2. **Model Size**: Prefer smaller model files for faster loading
3. **Threading**: SentencePiece is thread-safe for read operations
4. **Batch Size**: Optimize batch sizes for mobile memory constraints
5. **Error Recovery**: Implement robust fallback mechanisms
6. **Platform Support**: Ensure compatibility across Android/iOS architectures

### Testing and Validation

#### Round-trip Testing
```python
original = "Test input text"
pieces = sp.encode_as_pieces(original)
reconstructed = sp.decode_pieces(pieces)
assert original == reconstructed
```

#### Performance Benchmarking
```python
import time
text_batch = ["Sample text"] * 1000
start = time.time()
results = sp.encode(text_batch, out_type=int)
end = time.time()
print(f"Encoding speed: {len(text_batch)/(end-start)} texts/sec")
```

This documentation provides comprehensive guidance for integrating SentencePiece efficiently in mobile environments while maintaining performance and reliability.
