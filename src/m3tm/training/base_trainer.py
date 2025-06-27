"""
Base Trainer Class for M³TM

Bu modül, tüm trainer sınıfları için temel interface ve ortak fonksiyonları sağlar.
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class BaseTrainingConfig:
    """Temel eğitim konfigürasyonu"""
    learning_rate: float = 1e-4
    batch_size: int = 8
    epochs: int = 3
    max_seq_length: int = 512
    
    # Regularization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True


class BaseTrainer(ABC):
    """Tüm trainer sınıfları için temel sınıf"""
    
    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainingConfig,
        save_dir: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path("./checkpoints")
        
        # Logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Device setup
        self.device = self._setup_device()
        
        # Model'i device'a taşı
        self.model.to(self.device)
        
        # Checkpoint dizinini oluştur
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        self.logger.info(f"{self.__class__.__name__} oluşturuldu - Device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Device setup"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"CUDA kullanılıyor: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("CPU kullanılıyor")
        else:
            device = torch.device(self.config.device)
            self.logger.info(f"Manuel device: {device}")
        
        return device
    
    @abstractmethod
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """Eğitim metodunu implement et"""
        pass
    
    @abstractmethod
    def create_dataset(
        self,
        data_path: str,
        mode: str = "train"
    ) -> Dataset:
        """Dataset oluşturma metodunu implement et"""
        pass
    
    def save_model(self, path: str):
        """Modeli kaydet"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Model state dict kaydet
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch
        }, save_path)
        
        self.logger.info(f"Model kaydedildi: {save_path}")
    
    def load_model(self, path: str):
        """Modeli yükle"""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Model state dict yükle
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Training state yükle
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        
        self.logger.info(f"Model yüklendi: {load_path}")
    
    def get_model_parameters_count(self) -> Dict[str, int]:
        """Model parametre sayısını hesapla"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params
        }
    
    def log_model_info(self):
        """Model bilgilerini logla"""
        param_info = self.get_model_parameters_count()
        
        self.logger.info("=== Model Bilgileri ===")
        self.logger.info(f"Toplam parametre: {param_info['total_parameters']:,}")
        self.logger.info(f"Eğitilebilir parametre: {param_info['trainable_parameters']:,}")
        self.logger.info(f"Dondurulmuş parametre: {param_info['non_trainable_parameters']:,}")
        self.logger.info(f"Model boyutu: {param_info['total_parameters'] * 4 / 1024 / 1024:.2f} MB")
    
    def freeze_parameters(self, module_names: List[str]):
        """Belirtilen modüllerin parametrelerini dondur"""
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            for module_name in module_names:
                if module_name in name:
                    param.requires_grad = False
                    frozen_count += 1
                    break
        
        self.logger.info(f"{frozen_count} parametre donduruldu")
    
    def unfreeze_parameters(self, module_names: List[str]):
        """Belirtilen modüllerin parametrelerini çöz"""
        unfrozen_count = 0
        
        for name, param in self.model.named_parameters():
            for module_name in module_names:
                if module_name in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    break
        
        self.logger.info(f"{unfrozen_count} parametre çözüldü")
    
    def get_learning_rate(self, optimizer: torch.optim.Optimizer) -> float:
        """Mevcut learning rate'i al"""
        return optimizer.param_groups[0]['lr']
    
    def cleanup_checkpoints(self):
        """Eski checkpoint'leri temizle"""
        if not self.save_dir.exists():
            return
        
        # Checkpoint dosyalarını bul
        checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pt"))
        
        # Epoch numarasına göre sırala
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Fazla olanları sil
        if len(checkpoint_files) > self.config.save_total_limit:
            files_to_delete = checkpoint_files[:-self.config.save_total_limit]
            for file_path in files_to_delete:
                file_path.unlink()
                self.logger.info(f"Eski checkpoint silindi: {file_path}")
    
    def create_optimizer(self, learning_rate: Optional[float] = None) -> torch.optim.Optimizer:
        """Optimizer oluştur"""
        lr = learning_rate or self.config.learning_rate
        
        # Sadece eğitilebilir parametreleri al
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config.weight_decay
        )
        
        self.logger.info(f"Optimizer oluşturuldu - LR: {lr}, Trainable params: {len(trainable_params)}")
        
        return optimizer
    
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int
    ):
        """Learning rate scheduler oluştur"""
        from transformers import get_linear_schedule_with_warmup
        
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.info(
            f"Scheduler oluşturuldu - "
            f"Warmup steps: {warmup_steps}, "
            f"Total steps: {num_training_steps}"
        )
        
        return scheduler
    
    def log_training_progress(
        self,
        step: int,
        total_steps: int,
        loss: float,
        learning_rate: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Eğitim ilerlemesini logla"""
        progress = (step / total_steps) * 100
        
        log_msg = (
            f"Step {step}/{total_steps} ({progress:.1f}%) - "
            f"Loss: {loss:.4f}, LR: {learning_rate:.2e}"
        )
        
        if additional_metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in additional_metrics.items()])
            log_msg += f", {metrics_str}"
        
        self.logger.info(log_msg)
    
    def validate_config(self):
        """Konfigürasyonu doğrula"""
        if self.config.learning_rate <= 0:
            raise ValueError("Learning rate pozitif olmalı")
        
        if self.config.batch_size <= 0:
            raise ValueError("Batch size pozitif olmalı")
        
        if self.config.epochs <= 0:
            raise ValueError("Epoch sayısı pozitif olmalı")
        
        if self.config.max_seq_length <= 0:
            raise ValueError("Max sequence length pozitif olmalı")
        
        self.logger.info("Konfigürasyon doğrulandı")


class TrainingCallback:
    """Eğitim callback'leri için temel sınıf"""
    
    def on_train_start(self, trainer: BaseTrainer):
        """Eğitim başlangıcında çağrılır"""
        pass
    
    def on_train_end(self, trainer: BaseTrainer):
        """Eğitim sonunda çağrılır"""
        pass
    
    def on_epoch_start(self, trainer: BaseTrainer, epoch: int):
        """Epoch başlangıcında çağrılır"""
        pass
    
    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Dict[str, float]):
        """Epoch sonunda çağrılır"""
        pass
    
    def on_batch_start(self, trainer: BaseTrainer, batch: Any):
        """Batch başlangıcında çağrılır"""
        pass
    
    def on_batch_end(self, trainer: BaseTrainer, batch_metrics: Dict[str, float]):
        """Batch sonunda çağrılır"""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback"""
    
    def __init__(self, patience: int = 3, monitor: str = "val_loss", min_delta: float = 0.001):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_score = None
        self.patience_counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Dict[str, float]):
        """Epoch sonunda early stopping kontrolü"""
        if self.monitor not in metrics:
            return
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            trainer.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            self.should_stop = True
