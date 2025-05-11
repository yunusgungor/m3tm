"""
TrainingManager kullanımını gösteren örnek.

Bu örnek, M³TM modelinde Adapter'lar ve görev başlıklarını eğitimi için
TrainingManager ve ilgili bileşenlerin nasıl kullanılacağını gösterir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from m3tm.core.base_model import BaseModel
from m3tm.task_heads.classification import ClassificationHead
from m3tm.task_heads.config import ClassificationHeadConfig, TaskHeadFactory
from m3tm.adapters.adapter import BottleneckAdapter, AdapterConfig
from m3tm.adapters.adapter_manager import AdapterManager
from m3tm.config.model_config import TrainingConfig
from m3tm.training.training_manager import TrainingManager


class DummyClassificationDataset(Dataset):
    """
    Basit bir sınıflandırma veri kümesi.
    """
    def __init__(self, num_samples=100, seq_len=20, num_classes=3):
        self.input_ids = torch.randint(0, 1000, (num_samples, seq_len))
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }


def create_dummy_data():
    """
    Örnek için yapay veri oluşturur.
    """
    # Özel veri kümesi sınıfını kullan
    train_dataset = DummyClassificationDataset(num_samples=100)
    val_dataset = DummyClassificationDataset(num_samples=20)
    
    # DataLoader oluştur - dict formatında verileri döndürür
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    return train_loader, val_loader


class SimplifiedM3TM(nn.Module):
    """
    Basitleştirilmiş M3TM modeli.
    """
    def __init__(self):
        super().__init__()
        # Giriş gömmeleri
        self.embedding = nn.Embedding(1000, 64)
        
        # Proto-transformer blokları (burada basit bir LSTM)
        self.transformer = nn.LSTM(64, 64, batch_first=True)
        
        # AdapterSlot içeren bir katman
        self.adapter_slot = nn.ModuleDict({
            "adapter1": nn.Identity(),  # Adapter yerine geçici olarak Identity
            "adapter2": nn.Identity()
        })
        
        # Görev başlığı - TaskHeadFactory kullanarak yapılandırma oluştur
        task_head_config = TaskHeadFactory.create_multiclass_classification_config(
            input_dim=64,
            hidden_dim=32,
            num_classes=3
        )
        self.task_head = ClassificationHead(task_head_config)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward geçişi gerçekleştirir.
        
        Args:
            input_ids: Girdi token ID'leri, şekil (batch_size, seq_len)
            attention_mask: Dikkat maskesi (opsiyonel)
            labels: Etiketler (opsiyonel)
            
        Returns:
            Çıktı sözlüğü
        """
        # Embedding
        x = self.embedding(input_ids)
        
        # Transformer (LSTM)
        x, _ = self.transformer(x)
        
        # Son token'ı al (basitleştirme)
        x = x[:, -1, :]
        
        # Adapter uygulaması (normalde adapter_manager tarafından yönetilir)
        for adapter_name, adapter in self.adapter_slot.items():
            x = adapter(x) + x  # Artık bağlantı
        
        # Görev başlığı
        task_output = self.task_head(x)
        
        # Çıktı sözlüğü
        logits = task_output["logits"]
        output = {"logits": logits}
        
        # Eğitim modunda kayıp hesapla
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            output["loss"] = loss_fn(logits, labels)
        
        return output


# Özel kayıp fonksiyonu
def classification_criterion(model_output, batch):
    """
    Özel kayıp fonksiyonu.
    
    Args:
        model_output: Model çıktısı
        batch: Girdi batch
        
    Returns:
        Kayıp değeri
    """
    if isinstance(model_output, dict) and "loss" in model_output:
        return model_output["loss"]
    
    if isinstance(model_output, dict) and "logits" in model_output:
        logits = model_output["logits"]
    else:
        logits = model_output
    
    labels = batch["labels"] if isinstance(batch, dict) else batch[1]
    return F.cross_entropy(logits, labels)


def adapter_training_example():
    """
    Adapter eğitimi örneği.
    """
    print("=== M³TM Adapter Eğitimi Örneği ===")
    
    # Model ve veri oluştur
    model = SimplifiedM3TM()
    train_loader, val_loader = create_dummy_data()
    
    # Adapter'ları tanımla
    adapter_config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=16,
        activation="gelu"
    )
    
    # Adapter'ları oluştur ve ekle (gerçek uygulamada AdapterManager kullanılır)
    adapter1 = BottleneckAdapter(adapter_config, 64)
    adapter2 = BottleneckAdapter(adapter_config, 64)
    
    # Adapter'ları modele ekle (basitleştirilmiş)
    model.adapter_slot["adapter1"] = adapter1
    model.adapter_slot["adapter2"] = adapter2
    
    # TrainingManager oluştur
    training_config = {
        "learning_rate": 1e-3,
        "epochs": 2,  # Hızlı test için az sayıda epoch
        "batch_size": 16,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_steps": 10,
        "gradient_clip": 1.0,
        "device": "cpu",
        
        # Adapter eğitim özellikleri
        "train_adapter_names": ["adapter1"],  # Sadece adapter1'i eğit
        "train_task_heads": True,            # Görev başlığını da eğit
        "freeze_core_model": True,           # Çekirdek modeli dondur
        "memory_optimization_level": "moderate"
    }
    
    # TrainingManager oluştur
    trainer = TrainingManager.create(
        model=model,
        config=training_config,
        use_tensorboard=False,
        use_early_stopping=True,
        save_dir="./checkpoints"
    )
    
    # Eğitim için modeli hazırla
    trainer.prepare_model_for_training()
    
    # Modeli eğit
    print("Eğitim başlıyor...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=classification_criterion
    )
    
    print(f"Eğitim tamamlandı. Son metrikler: {results}")
    
    # Model değerlendirme
    eval_results = trainer.evaluate(
        val_loader,
        criterion=classification_criterion
    )
    print(f"Değerlendirme sonuçları: {eval_results}")
    
    # Not: save_checkpoint TrainingManager API'sinin bu sürümünde doğrudan 
    # kullanılamıyor, optimizer erişimi sorunu var
    print("Örnek tamamlandı")
    
    return model, trainer


if __name__ == "__main__":
    adapter_training_example() 