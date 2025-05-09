#!/usr/bin/env python3
"""
Metin Sınıflandırma Örneği

Bu betik, M³TM modelinin metin sınıflandırma yeteneklerini göstermek için
basit bir örnek sağlar. TextEmbedding, ProtoTransformerBlock ve ClassificationHead
bileşenlerini birleştirerek bir model oluşturur ve eğitir.

Kullanım:
    python -m src.m3tm.examples.text_classification --data_path <veri_yolu> --save_dir <kayıt_dizini>

Örüntüler:
- ModelComposite (PT-003): Alt modülleri birleştiren kompozit model yapısı
- TrainingLoopTemplate: Standart eğitim döngüsü şablonu
- MetricsCollector (PT-008): Performans metriklerini toplama ve raporlama
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer

from m3tm.config.model_config import ModelConfig, TrainingConfig, DataConfig
from m3tm.embedding import TextEmbedding, TextEmbeddingConfig
from m3tm.transformer import ProtoTransformerBlock, ProtoTransformerConfig
from m3tm.task_heads import ClassificationHead, ClassificationHeadConfig
from m3tm.training import (
    TextClassificationDataset,
    DatasetFactory,
    TextClassificationTrainer
)


def parse_args():
    """
    Komut satırı argümanlarını ayrıştırır.
    
    Returns:
        argparse.Namespace: Ayrıştırılmış argümanlar
    """
    parser = argparse.ArgumentParser(description="M³TM Metin Sınıflandırma Örneği")
    
    # Veri argümanları
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Veri dosyası yolu (JSON formatında)"
    )
    parser.add_argument(
        "--sample_data",
        action="store_true",
        help="Örnek veri kullanılsın mı?"
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="JSON dosyasındaki metin alanının adı"
    )
    parser.add_argument(
        "--label_field",
        type=str,
        default="label",
        help="JSON dosyasındaki etiket alanının adı"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maksimum token uzunluğu"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch boyutu"
    )
    
    # Model argümanları
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="distilbert-base-uncased",
        help="Kullanılacak gömme modeli"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="Gömme boyutu"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Gizli katman boyutu"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Dikkat başlığı sayısı"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Transformer katmanı sayısı"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Sınıf sayısı"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout oranı"
    )
    
    # Eğitim argümanları
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epoch sayısı"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Öğrenme oranı"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Ağırlık azaltma"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "reduce_on_plateau", "none"],
        help="Öğrenme oranı zamanlayıcısı"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Isınma adımı sayısı"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradyan kırpma değeri"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Erken durdurma sabır değeri"
    )
    
    # Diğer argümanlar
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs",
        help="Modelin ve metriklerin kaydedileceği dizin"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Kullanılacak cihaz (belirtilmezse otomatik seçilir)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgele tohum"
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="TensorBoard kullanılsın mı?"
    )
    
    return parser.parse_args()


def create_sample_data() -> Tuple[List[str], List[int]]:
    """
    Örnek veri oluşturur.
    
    Returns:
        Tuple[List[str], List[int]]: Metin ve etiket listeleri
    """
    texts = [
        "Bu film harika, kesinlikle tavsiye ederim.",
        "Çok kötü bir deneyimdi, zamanıma yazık.",
        "Ortalama bir film, ne iyi ne kötü.",
        "Harika bir performans, oyuncular muhteşemdi.",
        "Tamamen zaman kaybı, berbat bir senaryo.",
        "Film beklentilerimi aştı, çok etkileyiciydi.",
        "Sıkıcı ve tahmin edilebilir bir hikaye.",
        "Görsel efektler mükemmeldi ama hikaye zayıftı.",
        "Karakterler iyi geliştirilmiş, hikaye akıcıydı.",
        "Müzik ve görüntüler güzel ama senaryo kötüydü.",
        "Kesinlikle izlemeye değer, çok duygu yüklü.",
        "Vasat bir film, hiçbir şey katmıyor.",
        "Çok güldüm, harika bir komedi.",
        "Çok sıkıcı, filmi bitiremedim bile.",
        "Oyunculuk berbat, diyaloglar yapay.",
        "Yönetmen harika iş çıkarmış, her sahne düşünülmüş.",
        "Teknik olarak iyi ama duygusal bağ kuramadım.",
        "Hikaye karmaşık ama sonunda her şey yerine oturuyor.",
        "Çok uzun ve gereksiz sahneler var.",
        "Başyapıt, kesinlikle tekrar izleyeceğim."
    ]
    
    # 0: Olumsuz, 1: Nötr, 2: Olumlu
    labels = [
        2, 0, 1, 2, 0, 2, 0, 1, 2, 1,
        2, 1, 2, 0, 0, 2, 1, 2, 0, 2
    ]
    
    return texts, labels


def create_configs(args) -> Tuple[ModelConfig, TrainingConfig, DataConfig]:
    """
    Yapılandırma nesnelerini oluşturur.
    
    Args:
        args: Komut satırı argümanları
        
    Returns:
        Tuple[ModelConfig, TrainingConfig, DataConfig]: Model, eğitim ve veri yapılandırmaları
    """
    # Cihazı belirle
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    # Model yapılandırması
    model_config = ModelConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Eğitim yapılandırması
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        seed=args.seed
    )
    
    # Veri yapılandırması
    data_config = DataConfig(
        batch_size=args.batch_size,
        max_length=args.max_length,
        text_field=args.text_field,
        label_field=args.label_field
    )
    
    return model_config, training_config, data_config


def create_model(
    model_config: ModelConfig,
    num_classes: int,
    embedding_model: str
) -> torch.nn.Module:
    """
    Model bileşenlerini oluşturur ve birleştirir.
    
    Args:
        model_config: Model yapılandırması
        num_classes: Sınıf sayısı
        embedding_model: Gömme modeli adı
        
    Returns:
        torch.nn.Module: Birleştirilmiş model
    """
    # TextEmbedding yapılandırması ve modülü
    embedding_config = TextEmbeddingConfig(
        model_name=embedding_model,
        embedding_dim=model_config.embedding_dim,
        max_length=model_config.max_length,
        trainable=False
    )
    text_embedding = TextEmbedding(embedding_config)
    
    # ProtoTransformerBlock yapılandırması ve modülü
    transformer_config = ProtoTransformerConfig(
        embedding_dim=model_config.embedding_dim,
        hidden_dim=model_config.hidden_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        activation="gelu"
    )
    transformer_block = ProtoTransformerBlock(transformer_config)
    
    # ClassificationHead yapılandırması ve modülü
    head_config = ClassificationHeadConfig(
        embedding_dim=model_config.embedding_dim,
        hidden_dim=model_config.hidden_dim,
        num_classes=num_classes,
        dropout=model_config.dropout,
        pooling_type="mean"
    )
    classification_head = ClassificationHead(head_config)
    
    # Kompozit model oluştur
    model = TextClassificationTrainer.create_composite_model(
        text_embedding=text_embedding,
        transformer_block=transformer_block,
        classification_head=classification_head
    )
    
    return model


def main():
    """
    Ana fonksiyon.
    """
    # Argümanları ayrıştır
    args = parse_args()
    
    # Rastgele tohum ayarla
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Yapılandırmaları oluştur
    model_config, training_config, data_config = create_configs(args)
    
    # Kaydetme dizinini oluştur
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Tokenizer oluştur
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    
    # Veri kümesi oluştur
    if args.sample_data:
        texts, labels = create_sample_data()
        dataset = DatasetFactory.create_text_classification_dataset(
            config=data_config,
            tokenizer=tokenizer,
            texts=texts,
            labels=labels
        )
    elif args.data_path:
        dataset = DatasetFactory.create_text_classification_dataset(
            config=data_config,
            tokenizer=tokenizer,
            data_path=args.data_path
        )
    else:
        print("Hata: --data_path veya --sample_data belirtilmelidir.")
        sys.exit(1)
    
    # Veri yükleyicileri oluştur
    train_loader, val_loader, test_loader = DatasetFactory.create_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        num_workers=0,  # Cihaz içi eğitim için 0 kullanılabilir
        pin_memory=True
    )
    
    # Sınıf sayısını belirle
    if args.num_classes is None:
        if hasattr(dataset, "label_map") and dataset.label_map:
            num_classes = len(dataset.label_map)
        else:
            # Etiketlerden benzersiz değerleri bul
            labels = []
            for batch in train_loader:
                labels.extend(batch["labels"].tolist())
            num_classes = len(set(labels))
    else:
        num_classes = args.num_classes
    
    # Model oluştur
    model = create_model(
        model_config=model_config,
        num_classes=num_classes,
        embedding_model=args.embedding_model
    )
    
    # Parametre sayısını yazdır
    param_counts = model.count_parameters()
    print(f"Toplam parametre sayısı: {param_counts['total']:,}")
    print(f"  - Embedding: {param_counts['embedding']:,}")
    print(f"  - Transformer: {param_counts['transformer']:,}")
    print(f"  - Classification Head: {param_counts['head']:,}")
    
    # Eğitim döngüsü oluştur
    trainer = TextClassificationTrainer(
        config=training_config,
        save_dir=save_dir,
        use_tensorboard=args.use_tensorboard
    )
    
    # Modeli eğit
    print(f"Eğitim başlıyor... (Cihaz: {training_config.device})")
    model = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Test et
    print("\nTest değerlendirmesi yapılıyor...")
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = trainer._validate(model, test_loader, criterion)
    
    print(f"Test Kayıp: {test_metrics['loss']:.4f}")
    print(f"Test Doğruluk: {test_metrics.get('accuracy', 0):.4f}")
    if 'precision' in test_metrics:
        print(f"Test Hassasiyet: {test_metrics['precision']:.4f}")
    if 'recall' in test_metrics:
        print(f"Test Duyarlılık: {test_metrics['recall']:.4f}")
    if 'f1' in test_metrics:
        print(f"Test F1 Skoru: {test_metrics['f1']:.4f}")
    
    # Yapılandırmaları kaydet
    config_path = save_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "data_config": data_config.to_dict(),
            "num_classes": num_classes,
            "embedding_model": args.embedding_model
        }, f, indent=2)
    
    print(f"\nModel ve yapılandırmalar {save_dir} dizinine kaydedildi.")


if __name__ == "__main__":
    main() 