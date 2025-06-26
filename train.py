#!/usr/bin/env python3
"""
M³TM Model Eğitim Scripti

Bu script, M³TM modelini farklı görevler için eğitmek üzere tasarlanmıştır.
Hem tam model eğitimi hem de adapter tabanlı eğitimi destekler.

Kullanım:
    python train.py --config configs/training_config.yaml
    python train.py --config configs/adapter_training.yaml --mode adapter
    python train.py --config configs/multimodal_training.yaml --mode full
"""

import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# M3TM imports
from src.m3tm.config.model_config import M3TMConfig, TrainingConfig, DataConfig
from src.m3tm.core.base_model import M3TMBaseModel
from src.m3tm.training.training_manager import TrainingManager
from src.m3tm.training.dataset import DatasetFactory, TextClassificationDataset
from src.m3tm.training.metrics import MetricsCollector
from src.m3tm.training.data_pipeline import DataPipeline, DataPipelineConfig
from src.m3tm.training.checkpoint_manager import CheckpointManager
from src.m3tm.training.monitoring import TrainingMonitor
from src.m3tm.training.distributed import DistributedManager, DistributedConfig, auto_detect_distributed_config
from src.m3tm.adapters.adapter_manager import AdapterManager
from src.m3tm.task_heads.config import TaskHeadFactory


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Logging sistemini kurar.
    
    Args:
        log_level: Log seviyesi (DEBUG, INFO, WARNING, ERROR)
        log_file: Log dosyası yolu (opsiyonel)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML konfigürasyon dosyasını yükler.
    
    Args:
        config_path: Konfigürasyon dosyası yolu
        
    Returns:
        Konfigürasyon sözlüğü
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model_config(config_dict: Dict[str, Any]) -> M3TMConfig:
    """
    Konfigürasyon sözlüğünden M3TMConfig oluşturur.
    
    Args:
        config_dict: Konfigürasyon sözlüğü
        
    Returns:
        M3TMConfig örneği
    """
    model_config_dict = config_dict.get('model', {})
    
    # Eğer özel konfigürasyon belirtilmemişse varsayılan kullan
    if not model_config_dict:
        return M3TMConfig.get_default_config()
    
    # Konfigürasyon boyutuna göre uygun config seç
    config_size = model_config_dict.get('size', 'default')
    
    if config_size == 'tiny':
        return M3TMConfig.get_tiny_config()
    elif config_size == 'small':
        return M3TMConfig.get_small_config()
    elif config_size == 'base':
        return M3TMConfig.get_base_config()
    else:
        return M3TMConfig.get_default_config()


def create_training_config(config_dict: Dict[str, Any]) -> TrainingConfig:
    """
    Konfigürasyon sözlüğünden TrainingConfig oluşturur.
    
    Args:
        config_dict: Konfigürasyon sözlüğü
        
    Returns:
        TrainingConfig örneği
    """
    training_dict = config_dict.get('training', {})
    
    return TrainingConfig(
        batch_size=training_dict.get('batch_size', 16),
        learning_rate=training_dict.get('learning_rate', 1e-3),
        weight_decay=training_dict.get('weight_decay', 0.01),
        epochs=training_dict.get('epochs', 10),
        optimizer=training_dict.get('optimizer', 'adamw'),
        scheduler=training_dict.get('scheduler', 'cosine'),
        warmup_steps=training_dict.get('warmup_steps', 100),
        gradient_clip=training_dict.get('gradient_clip', 1.0),
        early_stopping_patience=training_dict.get('early_stopping_patience', 5),
        device=training_dict.get('device', 'cpu'),
        mixed_precision=training_dict.get('mixed_precision', False)
    )


def create_data_config(config_dict: Dict[str, Any]) -> DataConfig:
    """
    Konfigürasyon sözlüğünden DataConfig oluşturur.
    
    Args:
        config_dict: Konfigürasyon sözlüğü
        
    Returns:
        DataConfig örneği
    """
    data_dict = config_dict.get('data', {})
    
    return DataConfig(
        max_seq_length=data_dict.get('max_seq_length', 128),
        train_batch_size=data_dict.get('train_batch_size', 32),
        eval_batch_size=data_dict.get('eval_batch_size', 64),
        num_workers=data_dict.get('num_workers', 4),
        tokenizer_name=data_dict.get('tokenizer_name', 'default'),
        dataset_name=data_dict.get('dataset_name'),
        dataset_path=data_dict.get('dataset_path'),
        train_file=data_dict.get('train_file'),
        eval_file=data_dict.get('eval_file'),
        test_file=data_dict.get('test_file'),
        preprocessing_config=data_dict.get('preprocessing', {})
    )


def create_model(model_config: M3TMConfig, mode: str = 'full') -> nn.Module:
    """
    Belirtilen konfigürasyona göre model oluşturur.
    
    Args:
        model_config: Model konfigürasyonu
        mode: Eğitim modu ('full', 'adapter', 'task_head')
        
    Returns:
        Oluşturulan model
    """
    logging.info(f"Model oluşturuluyor - Mod: {mode}")
    
    # Temel modeli oluştur
    model = M3TMBaseModel(model_config)
    
    if mode == 'adapter':
        # Adapter yöneticisini ekle
        adapter_manager = AdapterManager(model_config.adapter_config)
        model.adapter_manager = adapter_manager
        logging.info("Adapter yöneticisi eklendi")
    
    elif mode == 'task_head':
        # Görev başlıklarını ekle
        task_head_factory = TaskHeadFactory()
        # Bu kısım görev tipine göre özelleştirilebilir
        logging.info("Görev başlıkları eklendi")
    
    logging.info(f"Model oluşturuldu - Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def create_distributed_config(config_dict: Dict[str, Any]) -> DistributedConfig:
    """
    Konfigürasyon sözlüğünden DistributedConfig oluşturur.

    Args:
        config_dict: Konfigürasyon sözlüğü

    Returns:
        DistributedConfig örneği
    """
    distributed_dict = config_dict.get('distributed', {})

    # Otomatik algılama ile başla
    config = auto_detect_distributed_config()

    # Konfigürasyon dosyasından override et
    if 'enabled' in distributed_dict:
        config.enabled = distributed_dict['enabled']

    if 'backend' in distributed_dict:
        config.backend = distributed_dict['backend']

    if 'world_size' in distributed_dict:
        config.world_size = distributed_dict['world_size']

    if 'master_addr' in distributed_dict:
        config.master_addr = distributed_dict['master_addr']

    if 'master_port' in distributed_dict:
        config.master_port = str(distributed_dict['master_port'])

    # Performance ayarları
    config.find_unused_parameters = distributed_dict.get('find_unused_parameters', False)
    config.gradient_as_bucket_view = distributed_dict.get('gradient_as_bucket_view', True)
    config.static_graph = distributed_dict.get('static_graph', False)

    return config


def create_datasets(
    config_dict: Dict[str, Any],
    distributed_manager: Optional[DistributedManager] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Gelişmiş veri pipeline kullanarak veri setlerini oluşturur.

    Args:
        config_dict: Konfigürasyon sözlüğü
        distributed_manager: Distributed manager (opsiyonel)

    Returns:
        (train_loader, val_loader, test_loader) tuple'ı
    """
    logging.info("Gelişmiş veri pipeline ile veri setleri oluşturuluyor...")

    data_config_dict = config_dict.get('data', {})

    # Eğer gelişmiş veri pipeline konfigürasyonu varsa kullan
    if 'data_dir' in data_config_dict:
        # DataPipelineConfig oluştur
        pipeline_config = DataPipelineConfig(
            data_dir=data_config_dict['data_dir'],
            train_file=data_config_dict.get('train_file'),
            val_file=data_config_dict.get('eval_file'),
            test_file=data_config_dict.get('test_file'),
            data_format=data_config_dict.get('data_format', 'jsonl'),
            text_column=data_config_dict.get('text_column', 'text'),
            label_column=data_config_dict.get('label_column', 'label'),
            image_column=data_config_dict.get('image_column'),
            max_seq_length=data_config_dict.get('max_seq_length', 512),
            image_size=tuple(data_config_dict.get('image_size', [224, 224])),
            text_augmentation=data_config_dict.get('text_augmentation', False),
            image_augmentation=data_config_dict.get('image_augmentation', False),
            augmentation_prob=data_config_dict.get('augmentation_prob', 0.5),
            use_cache=data_config_dict.get('use_cache', True),
            cache_dir=data_config_dict.get('cache_dir')
        )

        # DataPipeline oluştur
        pipeline = DataPipeline(pipeline_config)

        # Veri setlerini oluştur
        train_dataset, val_dataset, test_dataset = pipeline.create_datasets()

        # DataLoader'ları oluştur
        batch_size = data_config_dict.get('train_batch_size', 32)
        num_workers = data_config_dict.get('num_workers', 4)

        # Distributed training için DataLoader'ları oluştur
        if distributed_manager and distributed_manager.config.enabled:
            train_loader = distributed_manager.create_distributed_dataloader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = distributed_manager.create_distributed_dataloader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            test_loader = None
            if test_dataset:
                test_loader = distributed_manager.create_distributed_dataloader(
                    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
                )
        else:
            train_loader, val_loader, test_loader = pipeline.create_dataloaders(
                train_dataset, val_dataset, test_dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )

        logging.info("Gelişmiş veri pipeline ile veri setleri oluşturuldu")

    else:
        # Fallback: Eski yöntem
        logging.info("Basit veri yükleme yöntemi kullanılıyor...")
        data_config = create_data_config(config_dict)

        # DatasetFactory kullanarak veri setlerini oluştur
        factory = DatasetFactory()

        # Eğer belirli dosyalar belirtilmişse onları kullan
        if data_config.train_file and data_config.eval_file:
            train_dataset = factory.create_text_classification_dataset(
                config=data_config,
                data_path=data_config.train_file
            )
            val_dataset = factory.create_text_classification_dataset(
                config=data_config,
                data_path=data_config.eval_file
            )
            test_dataset = None
            if data_config.test_file:
                test_dataset = factory.create_text_classification_dataset(
                    config=data_config,
                    data_path=data_config.test_file
                )
        else:
            # Varsayılan örnek veri seti oluştur
            logging.warning("Veri dosyaları belirtilmemiş, örnek veri seti oluşturuluyor")
            from src.m3tm.training.example import create_dummy_data
            train_dataset, val_dataset = create_dummy_data()
            test_dataset = None

        # DataLoader'ları oluştur
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config.train_batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.eval_batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=data_config.eval_batch_size,
                shuffle=False,
                num_workers=data_config.num_workers,
                pin_memory=True if torch.cuda.is_available() else False
            )

    return train_loader, val_loader, test_loader


def main():
    """Ana eğitim fonksiyonu."""
    parser = argparse.ArgumentParser(description="M³TM Model Eğitim Scripti")
    parser.add_argument("--config", type=str, required=True, help="Konfigürasyon dosyası yolu")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "adapter", "task_head"],
                       help="Eğitim modu")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Çıktı dizini")
    parser.add_argument("--resume", type=str, help="Checkpoint'ten devam et")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Distributed training argümanları
    parser.add_argument("--distributed", action="store_true", help="Distributed training'i etkinleştir")
    parser.add_argument("--world-size", type=int, default=1, help="Toplam process sayısı")
    parser.add_argument("--rank", type=int, default=0, help="Process rank'i")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master adres")
    parser.add_argument("--master-port", type=str, default="12355", help="Master port")

    args = parser.parse_args()
    
    # Çıktı dizinini oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging'i kur
    log_file = os.path.join(args.output_dir, "training.log")
    setup_logging(args.log_level, log_file)
    
    logging.info("M³TM Model Eğitimi Başlıyor...")
    logging.info(f"Konfigürasyon: {args.config}")
    logging.info(f"Mod: {args.mode}")
    logging.info(f"Çıktı dizini: {args.output_dir}")
    
    # Random seed ayarla
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    try:
        # Konfigürasyonu yükle
        config_dict = load_config(args.config)

        # Distributed konfigürasyonu oluştur
        distributed_config = create_distributed_config(config_dict)

        # Command line argümanlarından override et
        if args.distributed:
            distributed_config.enabled = True
        if args.world_size > 1:
            distributed_config.world_size = args.world_size
        if args.rank != 0:
            distributed_config.rank = args.rank
        if args.local_rank != 0:
            distributed_config.local_rank = args.local_rank
        if args.master_addr != "localhost":
            distributed_config.master_addr = args.master_addr
        if args.master_port != "12355":
            distributed_config.master_port = args.master_port

        # Distributed manager oluştur
        distributed_manager = DistributedManager(distributed_config)

        # Distributed training'i kur
        distributed_manager.setup()

        # Alt konfigürasyonları oluştur
        model_config = create_model_config(config_dict)
        training_config = create_training_config(config_dict)
        data_config = create_data_config(config_dict)

        # Modeli oluştur
        model = create_model(model_config, args.mode)

        # Modeli distributed wrapper ile sar
        model = distributed_manager.wrap_model(model)

        # Veri setlerini oluştur
        train_loader, val_loader, test_loader = create_datasets(config_dict, distributed_manager)

        logging.info("Eğitim başlatılıyor...")

        # Checkpoint manager oluştur (sadece main process'te)
        checkpoint_manager = None
        if distributed_manager.is_main_process():
            checkpoint_manager = CheckpointManager(
                save_dir=os.path.join(args.output_dir, "checkpoints"),
                max_checkpoints=5,
                save_best_only=False,
                monitor_metric="val_loss",
                mode="min"
            )

        # Training monitor oluştur (sadece main process'te)
        training_monitor = None
        if distributed_manager.is_main_process():
            monitor_config = config_dict.get('logging', {})
            training_monitor = TrainingMonitor(
                log_dir=args.output_dir,
                use_tensorboard=monitor_config.get('tensorboard', True),
                use_wandb=monitor_config.get('wandb', False),
                wandb_project=monitor_config.get('wandb_project'),
                wandb_name=monitor_config.get('wandb_name'),
                config=config_dict,
                log_every_n_steps=monitor_config.get('log_every_n_steps', 100),
                save_metrics_every_n_steps=monitor_config.get('save_metrics_every_n_steps', 1000)
            )

        # TrainingManager oluştur
        trainer = TrainingManager.create(
            model=model,
            config=training_config.__dict__,  # TrainingConfig'i dict'e çevir
            use_tensorboard=False,  # TrainingMonitor kullanacağız
            use_early_stopping=True,
            save_dir=args.output_dir
        )

        logging.info(f"Çıktı dizini: {args.output_dir}")
        logging.info(f"Checkpoint dizini: {os.path.join(args.output_dir, 'checkpoints')}")

        # Eğitimi başlat
        start_time = time.time()

        try:
            # Resume checkpoint varsa yükle
            start_epoch = 0
            start_step = 0

            if args.resume:
                if os.path.exists(args.resume):
                    checkpoint_info = checkpoint_manager.load_checkpoint(
                        args.resume, model,
                        load_optimizer=True,
                        load_scheduler=True
                    )
                    start_epoch = checkpoint_info['epoch']
                    start_step = checkpoint_info['step']
                    logging.info(f"Checkpoint yüklendi - Epoch: {start_epoch}, Step: {start_step}")
                else:
                    logging.warning(f"Checkpoint dosyası bulunamadı: {args.resume}")

            # Gelişmiş eğitim döngüsü
            results = run_advanced_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                training_config=training_config,
                checkpoint_manager=checkpoint_manager,
                training_monitor=training_monitor,
                distributed_manager=distributed_manager,
                start_epoch=start_epoch,
                start_step=start_step,
                config_dict=config_dict
            )

            # Eğitim süresini hesapla
            training_time = time.time() - start_time

            # Sonuçları logla
            logging.info("Eğitim tamamlandı!")
            logging.info(f"Toplam süre: {training_time:.2f} saniye")
            logging.info(f"Son eğitim kaybı: {results.get('train_loss', 'N/A')}")
            logging.info(f"Son doğrulama kaybı: {results.get('val_loss', 'N/A')}")

            if 'val_accuracy' in results:
                logging.info(f"Son doğrulama doğruluğu: {results['val_accuracy']:.4f}")

            # Test değerlendirmesi (eğer test seti varsa)
            if test_loader:
                logging.info("Test değerlendirmesi başlatılıyor...")
                test_results = evaluate_model(model, test_loader, training_config.device)
                logging.info(f"Test sonuçları: {test_results}")

                # Test sonuçlarını kaydet
                test_results_path = os.path.join(args.output_dir, "test_results.yaml")
                with open(test_results_path, 'w') as f:
                    yaml.dump(test_results, f)

            # Final model'i kaydet
            final_model_path = os.path.join(args.output_dir, "final_model")
            model.save_pretrained(final_model_path)
            logging.info(f"Final model kaydedildi: {final_model_path}")

            # Eğitim özetini kaydet
            training_summary = {
                'config': config_dict,
                'results': results,
                'training_time': training_time,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'best_checkpoint': checkpoint_manager.get_best_checkpoint_path(),
                'metrics_summary': training_monitor.get_metrics_summary()
            }

            summary_path = os.path.join(args.output_dir, "training_summary.yaml")
            with open(summary_path, 'w') as f:
                yaml.dump(training_summary, f)

            logging.info(f"Eğitim özeti kaydedildi: {summary_path}")

        finally:
            # Monitoring'i kapat (sadece main process'te)
            if training_monitor:
                training_monitor.close()

            # Distributed training'i temizle
            distributed_manager.cleanup()

    except Exception as e:
        logging.error(f"Eğitim sırasında hata oluştu: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise


def run_advanced_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    training_config: TrainingConfig,
    checkpoint_manager: Optional[CheckpointManager],
    training_monitor: Optional[TrainingMonitor],
    distributed_manager: DistributedManager,
    start_epoch: int = 0,
    start_step: int = 0,
    config_dict: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Gelişmiş eğitim döngüsü.

    Args:
        model: Eğitilecek model
        train_loader: Eğitim veri yükleyicisi
        val_loader: Doğrulama veri yükleyicisi
        training_config: Eğitim konfigürasyonu
        checkpoint_manager: Checkpoint yöneticisi (opsiyonel)
        training_monitor: Eğitim monitörü (opsiyonel)
        distributed_manager: Distributed manager
        start_epoch: Başlangıç epoch'u
        start_step: Başlangıç step'i
        config_dict: Konfigürasyon sözlüğü

    Returns:
        Eğitim sonuçları
    """
    # Cihazı belirle
    device = distributed_manager.get_device()

    # Model zaten distributed_manager.wrap_model() ile doğru device'a taşınmış

    # Optimizer oluştur
    if training_config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
    elif training_config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            momentum=0.9
        )

    # Scheduler oluştur
    total_steps = len(train_loader) * training_config.epochs
    if training_config.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
    elif training_config.scheduler.lower() == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    else:
        scheduler = None

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if training_config.mixed_precision else None

    # Eğitim döngüsü
    global_step = start_step
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, training_config.epochs):
        # Eğitim fazı
        model.train()
        train_metrics = {'train_loss': 0.0, 'train_accuracy': 0.0}
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Batch'i cihaza taşı
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            if training_config.mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    loss = criterion(logits, labels)

                # Backward pass
                scaler.scale(loss).backward()
                if training_config.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs
                loss = criterion(logits, labels)

                # Backward pass
                loss.backward()
                if training_config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
                optimizer.step()

            if scheduler:
                scheduler.step()

            # Metrikleri hesapla
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean().item()

                train_metrics['train_loss'] += loss.item()
                train_metrics['train_accuracy'] += accuracy
                num_train_batches += 1

            global_step += 1

            # Monitoring (sadece main process'te)
            if training_monitor and global_step % training_monitor.log_every_n_steps == 0:
                step_metrics = {
                    'loss': loss.item(),
                    'accuracy': accuracy
                }
                training_monitor.log_metrics(step_metrics, global_step, epoch, "train")
                training_monitor.log_learning_rate(optimizer, global_step)
                training_monitor.log_system_metrics(global_step)

        # Epoch metrikleri
        train_metrics['train_loss'] /= num_train_batches
        train_metrics['train_accuracy'] /= num_train_batches

        # Distributed training için metrikleri ortala
        train_metrics = distributed_manager.reduce_metrics(train_metrics)

        # Validation
        val_metrics = {}
        if val_loader:
            val_metrics = evaluate_model_detailed(model, val_loader, device, criterion)
            val_metrics = distributed_manager.reduce_metrics(val_metrics)

        # Epoch metrikleri logla (sadece main process'te)
        all_metrics = {**train_metrics, **val_metrics}
        if training_monitor:
            training_monitor.log_metrics(all_metrics, global_step, epoch, "epoch")

        # Checkpoint kaydet (sadece main process'te)
        if checkpoint_manager:
            # DDP model'den gerçek model'i al
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_manager.save_checkpoint(
                model=model_to_save,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                metrics=all_metrics,
                model_config=config_dict.get('model', {}),
                training_config=config_dict.get('training', {})
            )

        # Process'ler arası senkronizasyon
        distributed_manager.barrier()

        # Early stopping
        current_val_loss = val_metrics.get('val_loss', float('inf'))
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= training_config.early_stopping_patience:
            logging.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break

    return {
        'train_loss': train_metrics['train_loss'],
        'train_accuracy': train_metrics['train_accuracy'],
        'val_loss': val_metrics.get('val_loss', 0.0),
        'val_accuracy': val_metrics.get('val_accuracy', 0.0),
        'best_val_loss': best_val_loss,
        'total_steps': global_step
    }


def evaluate_model_detailed(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    criterion: nn.Module
) -> Dict[str, float]:
    """Detaylı model değerlendirmesi."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs

            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    return {
        'val_loss': total_loss / len(data_loader),
        'val_accuracy': correct_predictions / total_predictions
    }


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Modeli test seti üzerinde değerlendirir.

    Args:
        model: Değerlendirilecek model
        test_loader: Test veri yükleyicisi
        device: Hesaplama cihazı

    Returns:
        Test sonuçları sözlüğü
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            # Batch'i cihaza taşı
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Loss hesapla
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs

            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Doğruluk hesapla
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    # Ortalama metrikleri hesapla
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions

    return {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        'total_samples': total_predictions
    }


if __name__ == "__main__":
    main()
