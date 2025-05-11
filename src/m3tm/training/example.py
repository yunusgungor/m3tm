"""
TrainingManager kullanımını gösteren örnek.

Bu örnek, M³TM modelinde Adapter'lar ve görev başlıklarını eğitimi için
TrainingManager ve ilgili bileşenlerin nasıl kullanılacağını gösterir.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from m3tm.core.m3tm_core import M3TMCore
from m3tm.task_heads.classification import ClassificationHead
from m3tm.adapters.adapter import Adapter, AdapterConfig
from m3tm.adapters.adapter_manager import AdapterManager
from m3tm.config.model_config import TrainingConfig
from m3tm.training.training_manager import TrainingManager


def create_dummy_data():
    """
    Örnek için yapay veri oluşturur.
    """
    # Metin girdisi temsil eden yapay veriler (burada embeddingler)
    input_ids = torch.randint(0, 1000, (100, 20))  # 100 örnek, 20 token
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 3, (100,))  # 3 sınıflı sınıflandırma
    
    # DataLoader oluştur
    dataset = TensorDataset(input_ids, attention_mask, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset[:20], batch_size=16)
    
    return train_loader, val_loader


def create_dummy_model():
    """
    Örnek model oluşturur.
    """
    # Basitleştirilmiş M3TM modeli
    class SimplifiedM3TM(nn.Module):
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
            
            # Görev başlığı
            self.task_head = ClassificationHead(64, 3)
        
        def forward(self, input_ids, attention_mask=None, labels=None):
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
            logits = self.task_head(x)
            
            # Çıktı sözlüğü
            output = {"logits": logits}
            
            # Eğitim modunda kayıp hesapla
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                output["loss"] = loss_fn(logits, labels)
            
            return output
    
    return SimplifiedM3TM()


def adapter_training_example():
    """
    Adapter eğitimi örneği.
    """
    print("=== M³TM Adapter Eğitimi Örneği ===")
    
    # Model ve veri oluştur
    model = create_dummy_model()
    train_loader, val_loader = create_dummy_data()
    
    # Adapter'ları tanımla
    adapter_config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=16,
        activation="gelu"
    )
    
    # Adapter'ları oluştur ve ekle (gerçek uygulamada AdapterManager kullanılır)
    adapter1 = Adapter(adapter_config, 64)
    adapter2 = Adapter(adapter_config, 64)
    
    # Adapter'ları modele ekle (basitleştirilmiş)
    model.adapter_slot["adapter1"] = adapter1
    model.adapter_slot["adapter2"] = adapter2
    
    # TrainingManager oluştur
    training_config = {
        "learning_rate": 1e-3,
        "epochs": 5,
        "batch_size": 16,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "warmup_steps": 10,
        "gradient_clip_val": 1.0,
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
        val_loader=val_loader
    )
    
    print(f"Eğitim tamamlandı. Son metrikler: {results}")
    
    # Model değerlendirme
    eval_results = trainer.evaluate(val_loader)
    print(f"Değerlendirme sonuçları: {eval_results}")
    
    # Checkpoint kaydet
    trainer.save_checkpoint(is_best=True)
    print("Model kaydedildi")
    
    return model, trainer


if __name__ == "__main__":
    adapter_training_example() 