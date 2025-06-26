# M³TM Model Eğitim Rehberi

Bu rehber, M³TM modelini eğitmek için gerekli tüm bilgileri içerir.

## 🚀 Hızlı Başlangıç

### 1. Basit Eğitim

```bash
# Hızlı test eğitimi
python train.py --config configs/quick_test.yaml --mode full

# Tam model eğitimi
python train.py --config configs/training_config.yaml --mode full

# Adapter eğitimi
python train.py --config configs/adapter_training.yaml --mode adapter
```

### 2. Eğitim Testleri

```bash
# Hızlı test
python test_training_quick.py

# Unit testler
python -m pytest tests/test_training_pipeline.py -v
```

## 📁 Dosya Yapısı

```
├── train.py                           # Ana eğitim scripti
├── configs/                           # Konfigürasyon dosyaları
│   ├── training_config.yaml          # Tam model eğitimi
│   ├── adapter_training.yaml         # Adapter eğitimi
│   ├── multimodal_training.yaml      # Multimodal eğitim
│   └── quick_test.yaml               # Hızlı test
├── src/m3tm/training/                 # Eğitim modülleri
│   ├── data_pipeline.py              # Veri yükleme ve önişleme
│   ├── checkpoint_manager.py         # Checkpoint yönetimi
│   ├── monitoring.py                 # Metrik takibi
│   └── training_manager.py           # Eğitim yöneticisi
└── tests/                            # Test dosyaları
    └── test_training_pipeline.py     # Eğitim testleri
```

## ⚙️ Konfigürasyon

### Temel Konfigürasyon Yapısı

```yaml
# Model konfigürasyonu
model:
  size: "base"  # tiny, small, base, large
  use_text_modality: true
  use_image_modality: true

# Eğitim konfigürasyonu
training:
  batch_size: 32
  learning_rate: 2e-4
  epochs: 50
  optimizer: "adamw"
  scheduler: "cosine"
  device: "auto"  # auto, cpu, cuda, mps

# Veri konfigürasyonu
data:
  data_dir: "./data"
  train_file: "train.jsonl"
  eval_file: "val.jsonl"
  max_seq_length: 512
  text_augmentation: true
  image_augmentation: true

# Logging
logging:
  tensorboard: true
  wandb: false
  log_every_n_steps: 100
```

### Veri Formatı

#### JSONL Format (Önerilen)
```json
{"text": "Örnek metin", "label": 1}
{"text": "Başka örnek", "label": 0}
```

#### Multimodal Format
```json
{"text": "Resim açıklaması", "image": "path/to/image.jpg", "label": 1}
```

## 🔧 Gelişmiş Özellikler

### 1. Veri Pipeline

```python
from src.m3tm.training.data_pipeline import DataPipeline, DataPipelineConfig

# Konfigürasyon
config = DataPipelineConfig(
    data_dir="./data",
    train_file="train.jsonl",
    val_file="val.jsonl",
    text_augmentation=True,
    image_augmentation=True,
    max_seq_length=512
)

# Pipeline oluştur
pipeline = DataPipeline(config)
train_dataset, val_dataset, test_dataset = pipeline.create_datasets()
```

### 2. Checkpoint Yönetimi

```python
from src.m3tm.training.checkpoint_manager import CheckpointManager

# Checkpoint manager
manager = CheckpointManager(
    save_dir="./checkpoints",
    max_checkpoints=5,
    monitor_metric="val_loss",
    mode="min"
)

# Checkpoint kaydet
manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    step=step,
    metrics=metrics,
    model_config=model_config,
    training_config=training_config
)
```

### 3. Monitoring ve Metrikler

```python
from src.m3tm.training.monitoring import TrainingMonitor

# Monitor oluştur
monitor = TrainingMonitor(
    log_dir="./logs",
    use_tensorboard=True,
    use_wandb=True,
    wandb_project="m3tm-training"
)

# Metrik logla
monitor.log_metrics(
    metrics={"loss": 0.5, "accuracy": 0.8},
    step=step,
    epoch=epoch,
    phase="train"
)
```

## 📊 Eğitim Modları

### 1. Tam Model Eğitimi (`--mode full`)
- Tüm model parametrelerini eğitir
- En yüksek performans
- En fazla kaynak gereksinimi

### 2. Adapter Eğitimi (`--mode adapter`)
- Sadece adapter katmanlarını eğitir
- Hızlı ve verimli
- Transfer learning için ideal

### 3. Task Head Eğitimi (`--mode task_head`)
- Sadece görev başlıklarını eğitir
- Çok hızlı fine-tuning
- Spesifik görevler için

## 🎯 Eğitim Stratejileri

### 1. Progressive Training
```yaml
training:
  progressive_stages:
    - stage: "text_only"
      epochs: 10
      freeze_image: true
    - stage: "multimodal"
      epochs: 40
      freeze_text: false
      freeze_image: false
```

### 2. Curriculum Learning
```yaml
training:
  curriculum_learning: true
  difficulty_schedule: "linear"  # linear, exponential
```

### 3. Mixed Precision Training
```yaml
training:
  mixed_precision: true
  gradient_clip: 1.0
```

## 🔍 Monitoring ve Debugging

### TensorBoard
```bash
# TensorBoard başlat
tensorboard --logdir ./outputs/tensorboard
```

### Weights & Biases
```yaml
logging:
  wandb: true
  wandb_project: "m3tm-training"
  wandb_name: "experiment-1"
```

### Metrik Dosyaları
- `metrics.jsonl`: Adım bazında metrikler
- `final_metrics_summary.json`: Final özet
- `training_summary.yaml`: Eğitim özeti

## 🚨 Sorun Giderme

### Yaygın Hatalar

1. **CUDA Out of Memory**
   ```yaml
   training:
     batch_size: 16  # Daha küçük batch size
     mixed_precision: true
     gradient_checkpointing: true
   ```

2. **Yavaş Eğitim**
   ```yaml
   data:
     num_workers: 4  # Daha fazla worker
     pin_memory: true
   ```

3. **Overfitting**
   ```yaml
   training:
     dropout: 0.1
     weight_decay: 0.01
     early_stopping_patience: 10
   ```

### Debug Modu
```yaml
debug:
  profile_memory: true
  detect_anomaly: true
  deterministic: true
```

## 📈 Performans Optimizasyonu

### 1. Veri Yükleme
- `num_workers` artırın
- `pin_memory=True` kullanın
- Veri önişlemeyi optimize edin

### 2. Model Optimizasyonu
- Mixed precision training
- Gradient checkpointing
- Model parallelism

### 3. Distributed Training
```yaml
distributed:
  enabled: true
  backend: "nccl"  # nccl (GPU), gloo (CPU), mpi
  world_size: 4
  master_addr: "localhost"
  master_port: "12355"

  # Multi-node ayarları
  node_rank: 0
  num_nodes: 1
  gpus_per_node: 4

  # Performance optimizations
  find_unused_parameters: false
  gradient_as_bucket_view: true
  static_graph: false
```

## 📝 Örnek Komutlar

### Temel Eğitim
```bash
# CPU'da hızlı test
python train.py --config configs/quick_test.yaml --output-dir ./test_output

# GPU'da tam eğitim
python train.py --config configs/training_config.yaml --output-dir ./full_training

# Checkpoint'ten devam etme
python train.py --config configs/training_config.yaml --resume ./checkpoints/best_model.pt

# Adapter eğitimi
python train.py --config configs/adapter_training.yaml --mode adapter --output-dir ./adapter_training

# Multimodal eğitim
python train.py --config configs/multimodal_training.yaml --output-dir ./multimodal_training
```

### Distributed Training
```bash
# Single-node multi-GPU (4 GPU)
python scripts/run_distributed.py --config configs/distributed_training.yaml --gpus 4 --output-dir ./distributed_output

# Manuel distributed training
python train.py --config configs/distributed_training.yaml --distributed --world-size 4 --rank 0 --local-rank 0

# torchrun ile distributed training
torchrun --nproc_per_node=4 train.py --config configs/distributed_training.yaml --distributed

# Multi-node training (Master node)
python scripts/run_distributed.py --config configs/distributed_training.yaml --num-nodes 2 --gpus-per-node 4 --node-rank 0 --master-addr localhost

# Multi-node training (Worker node)
python scripts/run_distributed.py --config configs/distributed_training.yaml --num-nodes 2 --gpus-per-node 4 --node-rank 1 --master-addr <MASTER_IP>

# Bash script ile kolay kullanım
./scripts/run_distributed.sh multi-gpu configs/distributed_training.yaml ./outputs 4
./scripts/run_distributed.sh torchrun configs/distributed_training.yaml ./outputs 4
```

## 🎓 İleri Düzey Konular

### Distributed Training Setup

#### Environment Variables
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE="4"
export RANK="0"
export LOCAL_RANK="0"
```

#### SLURM ile Distributed Training
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

srun python train.py --config configs/distributed_training.yaml --distributed
```

#### Docker ile Distributed Training
```dockerfile
FROM pytorch/pytorch:latest
COPY . /workspace
WORKDIR /workspace
RUN pip install -r requirements.txt
CMD ["python", "train.py", "--config", "configs/distributed_training.yaml", "--distributed"]
```

### Custom Loss Functions
```python
def custom_loss_function(outputs, targets):
    # Özel loss implementasyonu
    pass
```

### Custom Metrics
```python
def custom_metric(predictions, targets):
    # Özel metrik implementasyonu
    pass
```

### Model Hooks
```python
def register_hooks(model):
    # Model hook'ları
    pass
```

### Distributed Training Best Practices

1. **Batch Size Scaling**: Total batch size = per_gpu_batch_size × world_size
2. **Learning Rate Scaling**: Linear scaling rule veya sqrt scaling
3. **Gradient Synchronization**: Automatic with DDP
4. **Memory Management**: Gradient checkpointing, mixed precision
5. **Communication Optimization**: Bucket size, compression

## 📚 Ek Kaynaklar

- [Model Architecture Guide](docs/model_architecture.md)
- [Data Preparation Guide](docs/data_preparation.md)
- [Mobile Optimization Guide](docs/mobile_optimization.md)
- [API Reference](docs/api_reference.md)
