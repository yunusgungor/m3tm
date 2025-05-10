"""
Adapter'lar ve Görev Başlıkları Eğitimi Örneği

Bu örnek, M³TM modeli için Adapter'lar ve görev başlıklarını eğitme mekanizmasını gösterir.
Bu mekanizma, kişiselleştirme için gereken mobil cihaz üzerinde eğitim sürecini verimli hale getirir.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

from m3tm.config.model_config import TrainingConfig
from m3tm.adapters.adapter import BottleneckAdapter, AdapterConfig, AdapterType
from m3tm.adapters.adapter_manager import AdapterManager
from m3tm.task_heads.classification import ClassificationHead, ClassificationHeadConfig
from m3tm.transformer.proto_transformer import ProtoTransformerBlock, ProtoTransformerConfig
from m3tm.embedding.text_embedding import TextEmbedding, TextEmbeddingConfig
from m3tm.training.adapter_training import (
    AdapterTrainingConfig, 
    AdapterTrainingManager,
    TrainingCallback, 
    EarlyStoppingCallback
)
from m3tm.training.adapter_training_utils import (
    create_adapter_training_callbacks,
    create_adapter_criterion,
    find_task_heads_and_adapters,
    ModelStatisticsCallback
)
from m3tm.embedding.config import TokenizerConfig


def create_dummy_data(num_samples=1000, seq_length=20, vocab_size=10000, num_classes=5, train_ratio=0.8):
    """
    Örnek eğitim ve test veri kümeleri oluşturur.
    
    Args:
        num_samples: Toplam örnek sayısı
        seq_length: Her örneğin sekans uzunluğu
        vocab_size: Kelime dağarcığı boyutu
        num_classes: Sınıf sayısı
        train_ratio: Eğitim setinin oranı
        
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    # Rastgele giriş ve etiketler oluştur
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Eğitim ve test setlerini ayır
    train_size = int(num_samples * train_ratio)
    
    train_inputs = input_ids[:train_size]
    train_labels = labels[:train_size]
    
    test_inputs = input_ids[train_size:]
    test_labels = labels[train_size:]
    
    # TensorDataset ve DataLoader oluştur
    train_dataset = TensorDataset(train_inputs, train_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    return train_dataloader, test_dataloader


def create_model(vocab_size=10000, embed_dim=64, num_classes=5):
    """
    Örnek bir model oluşturur.
    
    Args:
        vocab_size: Kelime dağarcığı boyutu
        embed_dim: Gömme boyutu
        num_classes: Sınıf sayısı
        
    Returns:
        tuple: (model, adapter_manager)
    """
    # Text embedding modülü
    # Önce TokenizerConfig oluştur
    tokenizer_config = TokenizerConfig(
        vocab_size=vocab_size,
        max_seq_length=128
    )
    
    # TextEmbeddingConfig oluştur
    text_config = TextEmbeddingConfig(
        embed_dim=embed_dim,
        tokenizer_config=tokenizer_config
    )
    
    text_embedding = TextEmbedding(text_config)
    
    # Transformer blok
    transformer_config = ProtoTransformerConfig(
        hidden_size=embed_dim,
        intermediate_size=embed_dim*4
    )
    transformer_block = ProtoTransformerBlock(transformer_config)
    
    # Sınıflandırma başlığı
    head_config = ClassificationHeadConfig(input_dim=embed_dim, num_classes=num_classes)
    classification_head = ClassificationHead(head_config)
    
    # Model
    class TextClassificationModel(nn.Module):
        def __init__(self, embedding, transformer, head):
            super().__init__()
            self.embedding = embedding
            self.transformer = transformer
            self.head = head
        
        def forward(self, input_ids, labels=None):
            # Embedding
            x = self.embedding(input_ids)
            if isinstance(x, dict):
                x = x.get("embeddings", x)
            
            # Transformer (çıktı bir tuple olabilir)
            transformer_output = self.transformer(x)
            if isinstance(transformer_output, tuple):
                x = transformer_output[0]  # İlk öğe genellikle ana çıktıdır
            else:
                x = transformer_output
            
            # Head
            outputs = self.head(x)
            
            # Eğer etiketler varsa, kayıp hesapla
            if labels is not None:
                outputs['loss'] = nn.CrossEntropyLoss()(outputs['logits'], labels)
            
            return outputs
    
    model = TextClassificationModel(text_embedding, transformer_block, classification_head)
    
    # Adapter yöneticisi oluştur (model ile)
    adapter_manager = AdapterManager(model)
    
    # Adapter yapılandırmaları
    from m3tm.adapters.adapter import AdapterConfig as AdapterConfigClass, AdapterType
    
    # Adapters oluştur
    adapter_config1 = AdapterConfigClass(adapter_type=AdapterType.BOTTLENECK, bottleneck_dim=16)
    adapter_config2 = AdapterConfigClass(adapter_type=AdapterType.BOTTLENECK, bottleneck_dim=8)
    
    # Adapter'ları kaydet
    adapter_manager.register_adapter(transformer_block, "adapter_domain", "post_attention", adapter_config1)
    adapter_manager.register_adapter(transformer_block, "adapter_task", "post_ffn", adapter_config2)
    
    return model, adapter_manager


def train_adapters():
    """
    Adapter'ları ve görev başlıklarını eğitir.
    """
    print("Adapter eğitimi örneği başlatılıyor...")
    
    # Cihazı belirle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Veri oluştur
    print("Örnek veri oluşturuluyor...")
    train_dataloader, test_dataloader = create_dummy_data()
    
    # Model oluştur
    print("Model oluşturuluyor...")
    model, adapter_manager = create_model()
    
    # Modelin özetini yazdır
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model toplam parametre sayısı: {total_params:,}")
    
    # Kaydetme dizini
    save_dir = Path("./model_checkpoints/adapter_example")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Eğitim yapılandırması
    print("Eğitim yapılandırması ayarlanıyor...")
    train_config = AdapterTrainingConfig(
        train_adapter_names=["adapter_domain", "adapter_task"],  # Hangi adapter'ların eğitileceği
        train_task_heads=True,  # Görev başlığı da eğitilsin mi?
        freeze_core_model=True,  # Çekirdek model dondurulsun mu?
        
        # Eğitim hiperparametreleri
        learning_rate=5e-4,
        epochs=5,
        batch_size=32,
        optimizer="adamw",
        scheduler="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        gradient_clip=1.0,
        
        # Checkpoint ayarları
        checkpoint_frequency=1,
        checkpoint_keep_best_n=2,
        
        # Bellek optimizasyonu
        memory_optimization_level="moderate"
    )
    
    # Callback'ler oluştur
    print("Callback'ler oluşturuluyor...")
    callbacks = create_adapter_training_callbacks(
        use_early_stopping=True,
        early_stopping_config={"patience": 2, "monitor": "val_loss"},
        
        use_stats_logging=True,
        stats_config={"log_frequency": 1},
        
        use_grad_check=True,
        grad_check_config={"check_frequency": 50},
        
        use_memory_tracking=True,
        memory_config={"log_frequency": 20},
        
        use_lr_monitor=True,
        lr_monitor_config={"log_frequency": 20}
    )
    
    # Eğitim yöneticisi oluştur
    print("Eğitim yöneticisi oluşturuluyor...")
    training_manager = AdapterTrainingManager(
        model=model,
        config=train_config,
        adapter_manager=adapter_manager,
        save_dir=save_dir,
        use_tensorboard=True,  # TensorBoard kullanılsın mı?
        callbacks=callbacks
    )
    
    # Kayıp fonksiyonu
    def criterion(outputs, inputs):
        # loss doğrudan forward geçişi sırasında hesaplanıyor
        return outputs.get('loss', 0.0)
    
    # Modeli eğitim için hazırla
    print("Model eğitim için hazırlanıyor...")
    training_manager.prepare_model_for_training()
    
    # Eğitilebilir parametre sayısını göster
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Eğitimi başlat
    print("Eğitim başlatılıyor...")
    metrics = training_manager.train(
        train_loader=train_dataloader,
        criterion=criterion,
        val_loader=test_dataloader,
        resume=False  # Eğitime kaldığı yerden devam etsin mi?
    )
    
    print(f"Eğitim tamamlandı. Son metrikler: {metrics}")
    
    # Test et
    print("Test ediliyor...")
    test_metrics = training_manager.evaluate(test_dataloader, criterion)
    print(f"Test metrikleri: {test_metrics}")
    
    # Eğitilen adapter'ların parametre sayılarını göster
    param_counts = training_manager.get_trainable_parameter_count()
    print("\nEğitilen bileşenlerin parametre sayıları:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    print(f"\nModel başarıyla kaydedildi: {save_dir}")
    
    return model, adapter_manager, training_manager


def main():
    train_adapters()


if __name__ == "__main__":
    main() 