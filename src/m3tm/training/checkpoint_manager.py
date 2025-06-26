"""
Model Checkpoint ve Resume Sistemi

Bu modül, eğitim sırasında model durumunu kaydetme ve yükleme işlemlerini
yöneten sınıfları içerir.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class CheckpointMetadata:
    """Checkpoint metadata bilgileri."""
    epoch: int
    step: int
    best_metric: float
    best_metric_name: str
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    timestamp: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    total_params: int
    trainable_params: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Metadata'yı dictionary'e çevirir."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Dictionary'den metadata oluşturur."""
        return cls(**data)


class CheckpointManager:
    """
    Model checkpoint yönetimi için ana sınıf.
    
    Bu sınıf aşağıdaki işlemleri gerçekleştirir:
    - Model, optimizer ve scheduler durumlarını kaydetme
    - Checkpoint'leri yükleme ve resume etme
    - En iyi modelleri takip etme
    - Checkpoint geçmişini yönetme
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor_metric: str = "val_loss",
        mode: str = "min"  # "min" veya "max"
    ):
        """
        CheckpointManager'ı başlatır.
        
        Args:
            save_dir: Checkpoint'lerin kaydedileceği dizin
            max_checkpoints: Maksimum checkpoint sayısı
            save_best_only: Sadece en iyi modeli kaydet
            monitor_metric: İzlenecek metrik
            mode: Metrik optimizasyon yönü ("min" veya "max")
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.logger = logging.getLogger(__name__)
        
        # En iyi metrik değeri
        self.best_metric = float('inf') if mode == "min" else float('-inf')
        
        # Checkpoint geçmişi
        self.checkpoint_history: List[Dict[str, Any]] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Checkpoint geçmişini yükler."""
        history_file = self.save_dir / "checkpoint_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.checkpoint_history = json.load(f)
                
                # En iyi metrik değerini güncelle
                if self.checkpoint_history:
                    best_checkpoint = min(self.checkpoint_history, 
                                        key=lambda x: x['metadata']['best_metric'])
                    self.best_metric = best_checkpoint['metadata']['best_metric']
                    
            except Exception as e:
                self.logger.warning(f"Checkpoint geçmişi yüklenemedi: {e}")
                self.checkpoint_history = []
    
    def _save_history(self) -> None:
        """Checkpoint geçmişini kaydeder."""
        history_file = self.save_dir / "checkpoint_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.checkpoint_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Checkpoint geçmişi kaydedilemedi: {e}")
    
    def _is_better_metric(self, current_metric: float) -> bool:
        """Mevcut metrik değerinin daha iyi olup olmadığını kontrol eder."""
        if self.mode == "min":
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        is_best: bool = False
    ) -> str:
        """
        Checkpoint kaydeder.
        
        Args:
            model: Model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Mevcut epoch
            step: Mevcut step
            metrics: Eğitim metrikleri
            model_config: Model konfigürasyonu
            training_config: Eğitim konfigürasyonu
            is_best: En iyi model mi?
            
        Returns:
            Kaydedilen checkpoint dosyasının yolu
        """
        # Metrik değerini al
        current_metric = metrics.get(self.monitor_metric, float('inf'))
        
        # En iyi model kontrolü
        if not is_best:
            is_best = self._is_better_metric(current_metric)
            if is_best:
                self.best_metric = current_metric
        
        # Sadece en iyi modeli kaydet seçeneği
        if self.save_best_only and not is_best:
            return ""
        
        # Checkpoint metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            step=step,
            best_metric=self.best_metric,
            best_metric_name=self.monitor_metric,
            train_loss=metrics.get('train_loss', 0.0),
            val_loss=metrics.get('val_loss'),
            learning_rate=optimizer.param_groups[0]['lr'],
            timestamp=datetime.now().isoformat(),
            model_config=model_config,
            training_config=training_config,
            total_params=sum(p.numel() for p in model.parameters()),
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
        
        # Checkpoint dosya adı
        if is_best:
            checkpoint_name = "best_model.pt"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}.pt"
        
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Checkpoint verilerini hazırla
        checkpoint_data = {
            'metadata': metadata.to_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'epoch': epoch,
            'step': step
        }
        
        try:
            # Checkpoint'i kaydet
            torch.save(checkpoint_data, checkpoint_path)
            
            # Geçmişe ekle
            history_entry = {
                'checkpoint_path': str(checkpoint_path),
                'metadata': metadata.to_dict(),
                'is_best': is_best
            }
            self.checkpoint_history.append(history_entry)
            
            # Eski checkpoint'leri temizle
            self._cleanup_old_checkpoints()
            
            # Geçmişi kaydet
            self._save_history()
            
            self.logger.info(f"Checkpoint kaydedildi: {checkpoint_path}")
            if is_best:
                self.logger.info(f"Yeni en iyi model! {self.monitor_metric}: {current_metric:.6f}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Checkpoint kaydedilemedi: {e}")
            return ""
    
    def _cleanup_old_checkpoints(self) -> None:
        """Eski checkpoint'leri temizler."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # En iyi olmayan checkpoint'leri bul
        non_best_checkpoints = [
            cp for cp in self.checkpoint_history 
            if not cp.get('is_best', False)
        ]
        
        # Fazla checkpoint'leri sil
        if len(non_best_checkpoints) > self.max_checkpoints - 1:  # -1 for best model
            # Epoch'a göre sırala ve eski olanları sil
            non_best_checkpoints.sort(key=lambda x: x['metadata']['epoch'])
            
            to_remove = len(non_best_checkpoints) - (self.max_checkpoints - 1)
            for i in range(to_remove):
                checkpoint_to_remove = non_best_checkpoints[i]
                checkpoint_path = Path(checkpoint_to_remove['checkpoint_path'])
                
                if checkpoint_path.exists():
                    try:
                        checkpoint_path.unlink()
                        self.logger.info(f"Eski checkpoint silindi: {checkpoint_path}")
                    except Exception as e:
                        self.logger.warning(f"Checkpoint silinemedi {checkpoint_path}: {e}")
                
                # Geçmişten kaldır
                self.checkpoint_history.remove(checkpoint_to_remove)
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        Checkpoint yükler.
        
        Args:
            checkpoint_path: Checkpoint dosya yolu
            model: Model
            optimizer: Optimizer (opsiyonel)
            scheduler: Scheduler (opsiyonel)
            load_optimizer: Optimizer durumunu yükle
            load_scheduler: Scheduler durumunu yükle
            
        Returns:
            Checkpoint metadata ve metrikleri
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
        
        try:
            self.logger.info(f"Checkpoint yükleniyor: {checkpoint_path}")
            
            # Checkpoint'i yükle
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Model durumunu yükle
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Optimizer durumunu yükle
            if load_optimizer and optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Scheduler durumunu yükle
            if load_scheduler and scheduler and 'scheduler_state_dict' in checkpoint:
                if checkpoint['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Metadata'yı güncelle
            if 'metadata' in checkpoint:
                metadata = CheckpointMetadata.from_dict(checkpoint['metadata'])
                self.best_metric = metadata.best_metric
            
            self.logger.info(f"Checkpoint başarıyla yüklendi - Epoch: {checkpoint.get('epoch', 'N/A')}")
            
            return {
                'epoch': checkpoint.get('epoch', 0),
                'step': checkpoint.get('step', 0),
                'metrics': checkpoint.get('metrics', {}),
                'metadata': checkpoint.get('metadata', {})
            }
            
        except Exception as e:
            self.logger.error(f"Checkpoint yüklenemedi: {e}")
            raise
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """En iyi checkpoint'in yolunu döndürür."""
        best_checkpoint_path = self.save_dir / "best_model.pt"
        if best_checkpoint_path.exists():
            return str(best_checkpoint_path)
        
        # Geçmişten en iyi olanı bul
        best_checkpoints = [cp for cp in self.checkpoint_history if cp.get('is_best', False)]
        if best_checkpoints:
            return best_checkpoints[-1]['checkpoint_path']
        
        return None
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """En son checkpoint'in yolunu döndürür."""
        if not self.checkpoint_history:
            return None
        
        # En son epoch'a göre sırala
        latest_checkpoint = max(
            self.checkpoint_history,
            key=lambda x: x['metadata']['epoch']
        )
        
        checkpoint_path = Path(latest_checkpoint['checkpoint_path'])
        if checkpoint_path.exists():
            return str(checkpoint_path)
        
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Mevcut checkpoint'leri listeler."""
        return [
            {
                'path': cp['checkpoint_path'],
                'epoch': cp['metadata']['epoch'],
                'step': cp['metadata']['step'],
                'metric': cp['metadata']['best_metric'],
                'is_best': cp.get('is_best', False),
                'timestamp': cp['metadata']['timestamp']
            }
            for cp in self.checkpoint_history
            if Path(cp['checkpoint_path']).exists()
        ]
