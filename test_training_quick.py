#!/usr/bin/env python3
"""
Hızlı Eğitim Testi

Bu script, eğitim pipeline'ının doğru çalıştığını hızlıca test eder.
"""

import os
import sys
import tempfile
import shutil
import json
import logging
from pathlib import Path

# M3TM imports
from src.m3tm.config.model_config import M3TMConfig
from src.m3tm.core.base_model import M3TMBaseModel
from src.m3tm.training.data_pipeline import DataPipeline, DataPipelineConfig
from src.m3tm.training.checkpoint_manager import CheckpointManager
from src.m3tm.training.monitoring import TrainingMonitor
from train import create_training_config, run_advanced_training


def setup_logging():
    """Logging'i kur."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_dummy_data(data_dir: str):
    """Test için dummy veri oluştur."""
    # Train data
    train_data = [
        {"text": "Bu pozitif bir örnek cümle", "label": 1},
        {"text": "Bu negatif bir örnek cümle", "label": 0},
        {"text": "Harika bir gün", "label": 1},
        {"text": "Kötü bir deneyim", "label": 0},
        {"text": "Mükemmel sonuç", "label": 1},
        {"text": "Berbat performans", "label": 0},
        {"text": "Çok iyi çalışıyor", "label": 1},
        {"text": "Hiç beğenmedim", "label": 0}
    ]
    
    # Val data
    val_data = [
        {"text": "Doğrulama pozitif örnek", "label": 1},
        {"text": "Doğrulama negatif örnek", "label": 0},
        {"text": "İyi bir test", "label": 1},
        {"text": "Kötü bir test", "label": 0}
    ]
    
    # JSONL dosyalarını kaydet
    with open(os.path.join(data_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(os.path.join(data_dir, "val.jsonl"), 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logging.info(f"Dummy veri oluşturuldu: {data_dir}")


def test_data_pipeline():
    """Veri pipeline testleri."""
    logging.info("=== Veri Pipeline Testi ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Dummy veri oluştur
        create_dummy_data(temp_dir)
        
        # DataPipelineConfig
        config = DataPipelineConfig(
            data_dir=temp_dir,
            train_file="train.jsonl",
            val_file="val.jsonl",
            max_seq_length=32,
            vocab_size=1000,
            text_augmentation=False
        )
        
        # Pipeline oluştur
        pipeline = DataPipeline(config)
        
        # Veri setlerini oluştur
        train_dataset, val_dataset, test_dataset = pipeline.create_datasets()
        
        assert train_dataset is not None
        assert val_dataset is not None
        assert len(train_dataset) == 8
        assert len(val_dataset) == 4
        
        # DataLoader'ları oluştur
        train_loader, val_loader, test_loader = pipeline.create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=2, num_workers=0
        )
        
        assert train_loader is not None
        assert val_loader is not None
        
        # Bir batch test et
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 2  # batch_size
        assert batch["input_ids"].shape[1] == 32  # max_seq_length
        
        logging.info("✓ Veri pipeline testi başarılı")


def test_model_creation():
    """Model oluşturma testleri."""
    logging.info("=== Model Oluşturma Testi ===")
    
    # Tiny config kullan
    config = M3TMConfig.get_tiny_config()
    
    # Model oluştur
    model = M3TMBaseModel(config)
    
    # Parametre sayısını kontrol et
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Toplam parametreler: {total_params:,}")
    logging.info(f"Eğitilebilir parametreler: {trainable_params:,}")
    
    assert total_params > 0
    assert trainable_params > 0
    
    logging.info("✓ Model oluşturma testi başarılı")


def test_checkpoint_manager():
    """Checkpoint manager testleri."""
    logging.info("=== Checkpoint Manager Testi ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # CheckpointManager oluştur
        manager = CheckpointManager(
            save_dir=temp_dir,
            max_checkpoints=2,
            monitor_metric="val_loss",
            mode="min"
        )
        
        # Dummy model ve optimizer
        import torch
        import torch.nn as nn
        
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Checkpoint kaydet
        metrics = {"train_loss": 0.5, "val_loss": 0.4}
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            step=100,
            metrics=metrics,
            model_config={},
            training_config={}
        )
        
        assert checkpoint_path != ""
        assert os.path.exists(checkpoint_path)
        
        # Checkpoint listele
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 1
        
        logging.info("✓ Checkpoint manager testi başarılı")


def test_training_monitor():
    """Training monitor testleri."""
    logging.info("=== Training Monitor Testi ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # TrainingMonitor oluştur
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            use_tensorboard=False,  # Test için devre dışı
            use_wandb=False,
            log_every_n_steps=1
        )
        
        # Metrik logla
        metrics = {"loss": 0.5, "accuracy": 0.8}
        monitor.log_metrics(metrics, step=1, epoch=1, phase="train")
        
        # Özet al
        summary = monitor.get_metrics_summary()
        assert "loss" in summary
        assert "accuracy" in summary
        
        monitor.close()
        
        logging.info("✓ Training monitor testi başarılı")


def test_mini_training():
    """Mini eğitim testi."""
    logging.info("=== Mini Eğitim Testi ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Dummy veri oluştur
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir)
        create_dummy_data(data_dir)
        
        # Output dizini
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir)
        
        # Model config
        model_config = M3TMConfig.get_tiny_config()
        model = M3TMBaseModel(model_config)
        
        # Training config
        training_config_dict = {
            "training": {
                "batch_size": 2,
                "learning_rate": 1e-3,
                "epochs": 2,
                "optimizer": "adamw",
                "scheduler": "constant",
                "device": "cpu",
                "mixed_precision": False,
                "gradient_clip": 1.0,
                "early_stopping_patience": 10
            }
        }
        training_config = create_training_config(training_config_dict)
        
        # Veri pipeline
        pipeline_config = DataPipelineConfig(
            data_dir=data_dir,
            train_file="train.jsonl",
            val_file="val.jsonl",
            max_seq_length=32,
            vocab_size=1000,
            text_augmentation=False
        )
        
        pipeline = DataPipeline(pipeline_config)
        train_dataset, val_dataset, _ = pipeline.create_datasets()
        train_loader, val_loader, _ = pipeline.create_dataloaders(
            train_dataset, val_dataset, None,
            batch_size=2, num_workers=0
        )
        
        # Checkpoint manager
        checkpoint_manager = CheckpointManager(
            save_dir=os.path.join(output_dir, "checkpoints"),
            max_checkpoints=2,
            monitor_metric="val_loss",
            mode="min"
        )
        
        # Training monitor
        training_monitor = TrainingMonitor(
            log_dir=output_dir,
            use_tensorboard=False,
            use_wandb=False,
            log_every_n_steps=1
        )
        
        try:
            # Mini eğitim çalıştır
            results = run_advanced_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                training_config=training_config,
                checkpoint_manager=checkpoint_manager,
                training_monitor=training_monitor,
                start_epoch=0,
                start_step=0,
                config_dict=training_config_dict
            )
            
            # Sonuçları kontrol et
            assert "train_loss" in results
            assert "val_loss" in results
            assert results["total_steps"] > 0
            
            logging.info(f"Eğitim sonuçları: {results}")
            logging.info("✓ Mini eğitim testi başarılı")
            
        finally:
            training_monitor.close()


def main():
    """Ana test fonksiyonu."""
    setup_logging()
    
    logging.info("M³TM Eğitim Pipeline Hızlı Testleri Başlıyor...")
    
    try:
        test_data_pipeline()
        test_model_creation()
        test_checkpoint_manager()
        test_training_monitor()
        test_mini_training()
        
        logging.info("🎉 Tüm testler başarılı!")
        
    except Exception as e:
        logging.error(f"Test hatası: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
