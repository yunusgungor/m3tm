"""
Adapter ve Görev Başlığı Eğitimi Eğitimi

Bu eğitsel kod örneği, M³TM modelinde Adapter'ları ve görev başlıklarını eğitmek için
gereken temel adımları gösterir.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# M³TM modüllerini içe aktar
from m3tm.config.model_config import TrainingConfig
from m3tm.adapters.adapter import BottleneckAdapter, AdapterConfig, AdapterType
from m3tm.adapters.adapter_manager import AdapterManager
from m3tm.task_heads.classification import ClassificationHead, ClassificationHeadConfig
from m3tm.transformer.proto_transformer import ProtoTransformerBlock, ProtoTransformerConfig
from m3tm.embedding.text_embedding import TextEmbedding, TextEmbeddingConfig
from m3tm.training.adapter_training import (
    AdapterTrainingConfig, 
    AdapterTrainingManager, 
    EarlyStoppingCallback
)
from m3tm.training.adapter_training_utils import create_adapter_training_callbacks

#
# 1. Modeli oluşturma
#

# Text embedding modülü
text_config = TextEmbeddingConfig(vocab_size=10000, embed_dim=64)
text_embedding = TextEmbedding(text_config)

# Transformer blok
transformer_config = ProtoTransformerConfig(hidden_size=64)
transformer_block = ProtoTransformerBlock(transformer_config)

# Sınıflandırma başlığı
head_config = ClassificationHeadConfig(input_dim=64, num_classes=5)
classification_head = ClassificationHead(head_config)

# Model sınıfı
class TextModel(nn.Module):
    def __init__(self, embedding, transformer, head):
        super().__init__()
        self.embedding = embedding
        self.transformer = transformer
        self.head = head
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        outputs = self.head(x)
        
        # Eğer etiketler varsa, kayıp hesapla
        if labels is not None:
            outputs['loss'] = nn.CrossEntropyLoss()(outputs['logits'], labels)
        
        return outputs

# Modeli oluştur
model = TextModel(text_embedding, transformer_block, classification_head)

#
# 2. Adapter'lar ekleme
#

# Adapter yöneticisi oluştur
adapter_manager = AdapterManager(model)

# Birinci adapter: Domain adaptasyonu için
adapter_config1 = AdapterConfig(
    adapter_type=AdapterType.BOTTLENECK,
    bottleneck_dim=16
)
adapter_manager.register_adapter(transformer_block, "domain_adapter", "post_attention", adapter_config1)

# İkinci adapter: Görev adaptasyonu için
adapter_config2 = AdapterConfig(
    adapter_type=AdapterType.BOTTLENECK,
    bottleneck_dim=8
)
adapter_manager.register_adapter(transformer_block, "task_adapter", "post_attention", adapter_config2)

#
# 3. Eğitim verisi oluşturma
#

# Rastgele veri oluştur (gerçek uygulamada kendi verilerinizi kullanın)
num_samples = 500
seq_length = 20
input_ids = torch.randint(0, 10000, (num_samples, seq_length))
labels = torch.randint(0, 5, (num_samples,))

# Eğitim ve doğrulama setlerine böl
train_size = int(0.8 * num_samples)
train_inputs = input_ids[:train_size]
train_labels = labels[:train_size]
val_inputs = input_ids[train_size:]
val_labels = labels[train_size:]

# DataLoader'lar oluştur
train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

#
# 4. Eğitim yapılandırması
#

train_config = AdapterTrainingConfig(
    # Hangi bileşenlerin eğitileceği
    train_adapter_names=["domain_adapter", "task_adapter"],
    train_task_heads=True,
    freeze_core_model=True,
    
    # Eğitim parametreleri
    learning_rate=5e-4,
    epochs=5,
    batch_size=32,
    optimizer="adamw",
    scheduler="cosine",
    warmup_steps=50,
    
    # Checkpoint ayarları
    checkpoint_frequency=1,
    checkpoint_keep_best_n=2,
    
    # Bellek optimizasyonu
    memory_optimization_level="moderate"
)

#
# 5. Callback'ler
#

callbacks = create_adapter_training_callbacks(
    use_early_stopping=True,
    early_stopping_config={
        "patience": 3, 
        "monitor": "val_loss"
    },
    use_stats_logging=True
)

#
# 6. Eğitim yöneticisi
#

# Kaydetme dizini oluştur
save_dir = Path("./model_checkpoints")
save_dir.mkdir(exist_ok=True)

# Eğitim yöneticisi
training_manager = AdapterTrainingManager(
    model=model,
    config=train_config,
    adapter_manager=adapter_manager,
    save_dir=save_dir,
    use_tensorboard=True,
    callbacks=callbacks
)

#
# 7. Kayıp fonksiyonu
#

def criterion(outputs, inputs):
    # `forward` metodunda hesaplanan kaybı kullan
    return outputs.get('loss', 0.0)

#
# 8. Eğitimi gerçekleştirme
#

# Modeli hazırla (sadece adapter'lar ve görev başlığı eğitilebilir)
training_manager.prepare_model_for_training()

# Eğitilebilir parametre sayısını kontrol et
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Eğitilebilir parametre sayısı: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

# Eğitimi başlat
metrics = training_manager.train(
    train_loader=train_loader,
    criterion=criterion,
    val_loader=val_loader,
    resume=False  # Eğitime kaldığı yerden devam etme
)

#
# 9. Değerlendirme
#

# Test veri kümesi (bu örnekte doğrulama kümesini kullanıyoruz)
test_loader = val_loader

# Değerlendirme
test_metrics = training_manager.evaluate(test_loader, criterion)
print(f"Test metrikleri: {test_metrics}")

#
# 10. Eğitilen adapter'ların parametre sayılarını göster
#
param_counts = training_manager.get_trainable_parameter_count()
print("\nEğitilen bileşenlerin parametre sayıları:")
for name, count in param_counts.items():
    print(f"  {name}: {count:,}") 