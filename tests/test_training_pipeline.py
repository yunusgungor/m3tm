"""
Eğitim Pipeline Testleri

Bu modül, M³TM eğitim pipeline'ının doğru çalıştığını test eden unit testleri içerir.
"""

import os
import tempfile
import shutil
import json
import yaml
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Test edilecek modüller
from src.m3tm.config.model_config import M3TMConfig, TrainingConfig, DataConfig
from src.m3tm.core.base_model import M3TMBaseModel
from src.m3tm.training.data_pipeline import DataPipeline, DataPipelineConfig, MultimodalDataset
from src.m3tm.training.checkpoint_manager import CheckpointManager, CheckpointMetadata
from src.m3tm.training.monitoring import TrainingMonitor, MetricsTracker
from src.m3tm.training.distributed import DistributedManager, DistributedConfig, auto_detect_distributed_config
from src.m3tm.training.dataset import DatasetFactory


class TestDataPipeline:
    """Veri pipeline testleri."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Geçici veri dizini oluşturur."""
        temp_dir = tempfile.mkdtemp()
        
        # Örnek veri dosyaları oluştur
        train_data = [
            {"text": "Bu pozitif bir örnek", "label": 1},
            {"text": "Bu negatif bir örnek", "label": 0},
            {"text": "Başka bir pozitif örnek", "label": 1},
            {"text": "Başka bir negatif örnek", "label": 0}
        ]
        
        val_data = [
            {"text": "Doğrulama pozitif", "label": 1},
            {"text": "Doğrulama negatif", "label": 0}
        ]
        
        # JSONL dosyalarını kaydet
        with open(os.path.join(temp_dir, "train.jsonl"), 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        with open(os.path.join(temp_dir, "val.jsonl"), 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
        
        yield temp_dir
        
        # Temizlik
        shutil.rmtree(temp_dir)
    
    def test_data_pipeline_config(self):
        """DataPipelineConfig testleri."""
        config = DataPipelineConfig(
            data_dir="/test/dir",
            train_file="train.jsonl",
            val_file="val.jsonl",
            max_seq_length=128,
            text_augmentation=True
        )
        
        assert config.data_dir == "/test/dir"
        assert config.train_file == "train.jsonl"
        assert config.max_seq_length == 128
        assert config.text_augmentation is True
    
    def test_data_loading(self, temp_data_dir):
        """Veri yükleme testleri."""
        config = DataPipelineConfig(
            data_dir=temp_data_dir,
            train_file="train.jsonl",
            val_file="val.jsonl"
        )
        
        pipeline = DataPipeline(config)
        
        # Train veri yükleme
        train_data = pipeline.load_data(os.path.join(temp_data_dir, "train.jsonl"))
        assert len(train_data) == 4
        assert train_data[0]["text"] == "Bu pozitif bir örnek"
        assert train_data[0]["label"] == 1
        
        # Validation veri yükleme
        val_data = pipeline.load_data(os.path.join(temp_data_dir, "val.jsonl"))
        assert len(val_data) == 2
    
    def test_multimodal_dataset(self, temp_data_dir):
        """MultimodalDataset testleri."""
        data = [
            {"text": "Test metni", "label": 1},
            {"text": "Başka test metni", "label": 0}
        ]
        
        config = DataPipelineConfig(
            data_dir=temp_data_dir,
            max_seq_length=64,
            vocab_size=1000
        )
        
        dataset = MultimodalDataset(data, config, split="train")
        
        assert len(dataset) == 2
        
        # İlk örneği test et
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape[0] == 64  # max_seq_length
        assert item["labels"].item() == 1


class TestCheckpointManager:
    """Checkpoint manager testleri."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Geçici checkpoint dizini."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def dummy_model(self):
        """Test için basit model."""
        return nn.Linear(10, 2)
    
    @pytest.fixture
    def dummy_optimizer(self, dummy_model):
        """Test için optimizer."""
        return torch.optim.Adam(dummy_model.parameters())
    
    def test_checkpoint_metadata(self):
        """CheckpointMetadata testleri."""
        metadata = CheckpointMetadata(
            epoch=5,
            step=1000,
            best_metric=0.85,
            best_metric_name="accuracy",
            train_loss=0.2,
            val_loss=0.3,
            learning_rate=1e-4,
            timestamp="2024-01-01T00:00:00",
            model_config={},
            training_config={},
            total_params=1000,
            trainable_params=1000
        )
        
        # Dict'e çevirme
        metadata_dict = metadata.to_dict()
        assert metadata_dict["epoch"] == 5
        assert metadata_dict["best_metric"] == 0.85
        
        # Dict'ten oluşturma
        new_metadata = CheckpointMetadata.from_dict(metadata_dict)
        assert new_metadata.epoch == 5
        assert new_metadata.best_metric == 0.85
    
    def test_checkpoint_save_load(self, temp_checkpoint_dir, dummy_model, dummy_optimizer):
        """Checkpoint kaydetme ve yükleme testleri."""
        manager = CheckpointManager(
            save_dir=temp_checkpoint_dir,
            max_checkpoints=3,
            monitor_metric="val_loss",
            mode="min"
        )
        
        # Checkpoint kaydet
        metrics = {"train_loss": 0.5, "val_loss": 0.4, "accuracy": 0.8}
        checkpoint_path = manager.save_checkpoint(
            model=dummy_model,
            optimizer=dummy_optimizer,
            scheduler=None,
            epoch=1,
            step=100,
            metrics=metrics,
            model_config={},
            training_config={}
        )
        
        assert checkpoint_path != ""
        assert os.path.exists(checkpoint_path)
        
        # Yeni model oluştur ve checkpoint yükle
        new_model = nn.Linear(10, 2)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        checkpoint_info = manager.load_checkpoint(
            checkpoint_path, new_model, new_optimizer
        )
        
        assert checkpoint_info["epoch"] == 1
        assert checkpoint_info["step"] == 100
    
    def test_best_checkpoint_tracking(self, temp_checkpoint_dir, dummy_model, dummy_optimizer):
        """En iyi checkpoint takibi testleri."""
        manager = CheckpointManager(
            save_dir=temp_checkpoint_dir,
            monitor_metric="val_loss",
            mode="min"
        )
        
        # İlk checkpoint (kötü)
        metrics1 = {"val_loss": 0.8}
        manager.save_checkpoint(
            dummy_model, dummy_optimizer, None, 1, 100, metrics1, {}, {}
        )
        
        # İkinci checkpoint (daha iyi)
        metrics2 = {"val_loss": 0.5}
        manager.save_checkpoint(
            dummy_model, dummy_optimizer, None, 2, 200, metrics2, {}, {}
        )
        
        # Üçüncü checkpoint (daha da iyi)
        metrics3 = {"val_loss": 0.3}
        manager.save_checkpoint(
            dummy_model, dummy_optimizer, None, 3, 300, metrics3, {}, {}
        )
        
        assert manager.best_metric == 0.3
        
        # En iyi checkpoint yolu
        best_path = manager.get_best_checkpoint_path()
        assert best_path is not None
        assert "best_model.pt" in best_path


class TestTrainingMonitor:
    """Training monitor testleri."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Geçici log dizini."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_metrics_tracker(self):
        """MetricsTracker testleri."""
        tracker = MetricsTracker(window_size=3)
        
        # Metrik güncelleme
        tracker.update({"loss": 0.5, "accuracy": 0.8}, step=1)
        tracker.update({"loss": 0.4, "accuracy": 0.85}, step=2)
        tracker.update({"loss": 0.3, "accuracy": 0.9}, step=3)
        
        # En son değerler
        assert tracker.get_latest("loss") == 0.3
        assert tracker.get_latest("accuracy") == 0.9
        
        # Hareketli ortalama
        avg_loss = tracker.get_moving_average("loss")
        assert abs(avg_loss - 0.4) < 1e-6  # (0.5 + 0.4 + 0.3) / 3
        
        # Geçmiş
        history = tracker.get_history("loss")
        assert len(history) == 3
        assert history[0] == (1, 0.5)
        assert history[-1] == (3, 0.3)
    
    def test_training_monitor_initialization(self, temp_log_dir):
        """TrainingMonitor başlatma testleri."""
        monitor = TrainingMonitor(
            log_dir=temp_log_dir,
            use_tensorboard=True,
            use_wandb=False,
            log_every_n_steps=10
        )
        
        assert monitor.log_dir == Path(temp_log_dir)
        assert monitor.log_every_n_steps == 10
        assert monitor.tensorboard_logger is not None
        assert monitor.wandb_logger is None
        
        monitor.close()
    
    def test_metrics_logging(self, temp_log_dir):
        """Metrik loglama testleri."""
        monitor = TrainingMonitor(
            log_dir=temp_log_dir,
            use_tensorboard=False,  # TensorBoard'u devre dışı bırak
            use_wandb=False,
            log_every_n_steps=1
        )
        
        # Metrik logla
        metrics = {"loss": 0.5, "accuracy": 0.8}
        monitor.log_metrics(metrics, step=1, epoch=1, phase="train")
        
        # Metrik tracker'da olup olmadığını kontrol et
        assert monitor.metrics_tracker.get_latest("loss") == 0.5
        assert monitor.metrics_tracker.get_latest("accuracy") == 0.8
        
        monitor.close()


class TestDistributedTraining:
    """Distributed training testleri."""

    def test_distributed_config_creation(self):
        """DistributedConfig oluşturma testleri."""
        config = DistributedConfig(
            enabled=True,
            backend="nccl",
            world_size=4,
            rank=0,
            local_rank=0
        )

        assert config.enabled is True
        assert config.backend == "nccl"
        assert config.world_size == 4
        assert config.rank == 0

    def test_auto_detect_distributed_config(self):
        """Otomatik distributed config algılama testleri."""
        # Environment variables'ı temizle
        env_vars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
        original_values = {}
        for var in env_vars:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

        try:
            # Otomatik algılama
            config = auto_detect_distributed_config()

            # Single GPU/CPU durumunda disabled olmalı
            assert config.enabled is False or config.world_size == 1

        finally:
            # Environment variables'ı geri yükle
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_distributed_manager_initialization(self):
        """DistributedManager başlatma testleri."""
        config = DistributedConfig(enabled=False)  # Disabled mode
        manager = DistributedManager(config)

        assert manager.config.enabled is False
        assert manager.is_initialized is False
        assert manager.is_main_process() is True  # Single process durumunda

        # Setup çağrısı (disabled durumunda hiçbir şey yapmamalı)
        manager.setup()
        assert manager.is_initialized is False

        # Cleanup
        manager.cleanup()

    def test_distributed_sampler_creation(self):
        """Distributed sampler oluşturma testleri."""
        config = DistributedConfig(enabled=False)
        manager = DistributedManager(config)

        # Dummy dataset
        dataset = torch.utils.data.TensorDataset(torch.randn(100, 10))

        # Disabled durumunda None dönmeli
        sampler = manager.create_distributed_sampler(dataset)
        assert sampler is None

    def test_metrics_reduction(self):
        """Metrik reduction testleri."""
        config = DistributedConfig(enabled=False)
        manager = DistributedManager(config)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        reduced_metrics = manager.reduce_metrics(metrics)

        # Single process durumunda değişmemeli
        assert reduced_metrics == metrics


class TestTrainingIntegration:
    """Entegrasyon testleri."""

    @pytest.fixture
    def temp_output_dir(self):
        """Geçici çıktı dizini."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_training_config_creation(self):
        """Eğitim konfigürasyonu oluşturma testleri."""
        config_dict = {
            "training": {
                "batch_size": 16,
                "learning_rate": 1e-3,
                "epochs": 5,
                "optimizer": "adamw"
            }
        }

        # train.py'den import et
        from train import create_training_config

        training_config = create_training_config(config_dict)

        assert training_config.batch_size == 16
        assert training_config.learning_rate == 1e-3
        assert training_config.epochs == 5
        assert training_config.optimizer == "adamw"

    def test_distributed_config_creation_from_dict(self):
        """Konfigürasyon sözlüğünden distributed config oluşturma testleri."""
        config_dict = {
            "distributed": {
                "enabled": True,
                "backend": "gloo",
                "world_size": 2,
                "find_unused_parameters": True
            }
        }

        from train import create_distributed_config

        distributed_config = create_distributed_config(config_dict)

        assert distributed_config.enabled is True
        assert distributed_config.backend == "gloo"
        assert distributed_config.world_size == 2
        assert distributed_config.find_unused_parameters is True

    def test_model_creation(self):
        """Model oluşturma testleri."""
        from train import create_model

        # Tiny config kullan
        model_config = M3TMConfig.get_tiny_config()

        # Full mode
        model = create_model(model_config, mode="full")
        assert isinstance(model, M3TMBaseModel)

        # Parametre sayısını kontrol et
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
