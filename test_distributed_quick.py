#!/usr/bin/env python3
"""
Hızlı Distributed Training Testi

Bu script, distributed training bileşenlerinin doğru çalıştığını hızlıca test eder.
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
from src.m3tm.training.distributed import DistributedManager, DistributedConfig, auto_detect_distributed_config
from src.m3tm.training.data_pipeline import DataPipeline, DataPipelineConfig


def setup_logging():
    """Logging'i kur."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def test_distributed_config():
    """DistributedConfig testleri."""
    logging.info("=== DistributedConfig Testi ===")
    
    # Manuel config
    config = DistributedConfig(
        enabled=True,
        backend="gloo",  # CPU için gloo
        world_size=2,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port="12355"
    )
    
    assert config.enabled is True
    assert config.backend == "gloo"
    assert config.world_size == 2
    
    logging.info("✓ DistributedConfig testi başarılı")


def test_auto_detect_config():
    """Otomatik config algılama testleri."""
    logging.info("=== Otomatik Config Algılama Testi ===")
    
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
        
        # Single process durumunda disabled olmalı
        assert config.enabled is False or config.world_size == 1
        
        logging.info(f"Algılanan config: enabled={config.enabled}, world_size={config.world_size}")
        logging.info("✓ Otomatik config algılama testi başarılı")
        
    finally:
        # Environment variables'ı geri yükle
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value


def test_distributed_manager():
    """DistributedManager testleri."""
    logging.info("=== DistributedManager Testi ===")
    
    # Disabled mode test
    config = DistributedConfig(enabled=False)
    manager = DistributedManager(config)
    
    assert manager.config.enabled is False
    assert manager.is_initialized is False
    assert manager.is_main_process() is True
    
    # Setup (disabled durumunda hiçbir şey yapmamalı)
    manager.setup()
    assert manager.is_initialized is False
    
    # Device test
    device = manager.get_device()
    assert device.type in ["cpu", "cuda"]
    
    # Cleanup
    manager.cleanup()
    
    logging.info("✓ DistributedManager testi başarılı")


def test_model_wrapping():
    """Model wrapping testleri."""
    logging.info("=== Model Wrapping Testi ===")
    
    # Tiny model oluştur
    config = M3TMConfig.get_tiny_config()
    model = M3TMBaseModel(config)
    
    # Distributed manager (disabled)
    dist_config = DistributedConfig(enabled=False)
    manager = DistributedManager(dist_config)
    manager.setup()
    
    # Model'i wrap et
    wrapped_model = manager.wrap_model(model)
    
    # Disabled durumunda aynı model dönmeli
    assert wrapped_model is model
    
    # Parametre sayısını kontrol et
    total_params = sum(p.numel() for p in wrapped_model.parameters())
    assert total_params > 0
    
    manager.cleanup()
    
    logging.info("✓ Model wrapping testi başarılı")


def test_distributed_dataloader():
    """Distributed DataLoader testleri."""
    logging.info("=== Distributed DataLoader Testi ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Dummy veri oluştur
        data = [
            {"text": f"Test metni {i}", "label": i % 2}
            for i in range(20)
        ]
        
        data_file = os.path.join(temp_dir, "test.jsonl")
        with open(data_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        # DataPipeline config
        pipeline_config = DataPipelineConfig(
            data_dir=temp_dir,
            train_file="test.jsonl",
            max_seq_length=32,
            vocab_size=1000
        )
        
        # Pipeline oluştur
        pipeline = DataPipeline(pipeline_config)
        train_dataset, _, _ = pipeline.create_datasets()
        
        # Distributed manager (disabled)
        dist_config = DistributedConfig(enabled=False)
        manager = DistributedManager(dist_config)
        manager.setup()
        
        # Distributed DataLoader oluştur
        dataloader = manager.create_distributed_dataloader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # DataLoader'ı test et
        batch = next(iter(dataloader))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 4  # batch_size
        
        manager.cleanup()
        
        logging.info("✓ Distributed DataLoader testi başarılı")


def test_metrics_reduction():
    """Metrik reduction testleri."""
    logging.info("=== Metrik Reduction Testi ===")
    
    # Distributed manager (disabled)
    config = DistributedConfig(enabled=False)
    manager = DistributedManager(config)
    manager.setup()
    
    # Test metrikleri
    metrics = {
        "loss": 0.5,
        "accuracy": 0.8,
        "f1_score": 0.75
    }
    
    # Reduce metrikleri
    reduced_metrics = manager.reduce_metrics(metrics)
    
    # Single process durumunda değişmemeli
    assert reduced_metrics == metrics
    
    manager.cleanup()
    
    logging.info("✓ Metrik reduction testi başarılı")


def test_distributed_sampler():
    """Distributed sampler testleri."""
    logging.info("=== Distributed Sampler Testi ===")
    
    import torch
    
    # Dummy dataset
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 10))
    
    # Distributed manager (disabled)
    config = DistributedConfig(enabled=False)
    manager = DistributedManager(config)
    manager.setup()
    
    # Sampler oluştur
    sampler = manager.create_distributed_sampler(dataset, shuffle=True)
    
    # Disabled durumunda None dönmeli
    assert sampler is None
    
    manager.cleanup()
    
    logging.info("✓ Distributed sampler testi başarılı")


def test_environment_setup():
    """Environment setup testleri."""
    logging.info("=== Environment Setup Testi ===")
    
    from src.m3tm.training.distributed import setup_distributed_environment, find_free_port
    
    # Free port bulma
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535
    
    # Environment setup
    setup_distributed_environment(
        world_size=2,
        rank=0,
        master_addr="localhost",
        master_port=port
    )
    
    # Environment variables kontrol
    assert os.environ.get('WORLD_SIZE') == '2'
    assert os.environ.get('RANK') == '0'
    assert os.environ.get('MASTER_ADDR') == 'localhost'
    assert os.environ.get('MASTER_PORT') == str(port)
    
    logging.info("✓ Environment setup testi başarılı")


def main():
    """Ana test fonksiyonu."""
    setup_logging()
    
    logging.info("M³TM Distributed Training Hızlı Testleri Başlıyor...")
    
    try:
        test_distributed_config()
        test_auto_detect_config()
        test_distributed_manager()
        test_model_wrapping()
        test_distributed_dataloader()
        test_metrics_reduction()
        test_distributed_sampler()
        test_environment_setup()
        
        logging.info("🎉 Tüm distributed training testleri başarılı!")
        
    except Exception as e:
        logging.error(f"Test hatası: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
