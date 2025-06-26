"""
Eğitim Monitoring ve Metrik Takibi

Bu modül, eğitim sırasında metrikleri takip etmek, TensorBoard entegrasyonu
ve detaylı monitoring işlemleri için sınıfları içerir.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict, deque
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MetricsTracker:
    """Eğitim metriklerini takip eden sınıf."""
    
    def __init__(self, window_size: int = 100):
        """
        MetricsTracker'ı başlatır.
        
        Args:
            window_size: Hareketli ortalama için pencere boyutu
        """
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.moving_averages = defaultdict(lambda: deque(maxlen=window_size))
        self.step_count = 0
        self.epoch_count = 0
        
    def update(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Metrikleri günceller.
        
        Args:
            metrics: Metrik sözlüğü
            step: Adım sayısı (opsiyonel)
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
            
        for name, value in metrics.items():
            self.metrics[name].append((self.step_count, value))
            self.moving_averages[name].append(value)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """En son metrik değerini döndürür."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1][1]
        return None
    
    def get_moving_average(self, metric_name: str) -> Optional[float]:
        """Hareketli ortalamayı döndürür."""
        if metric_name in self.moving_averages and self.moving_averages[metric_name]:
            return sum(self.moving_averages[metric_name]) / len(self.moving_averages[metric_name])
        return None
    
    def get_history(self, metric_name: str) -> List[tuple]:
        """Metrik geçmişini döndürür."""
        return self.metrics.get(metric_name, [])
    
    def reset_epoch(self) -> None:
        """Epoch sayacını sıfırlar."""
        self.epoch_count += 1
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Metrik özetini döndürür."""
        summary = {}
        for name in self.metrics:
            if self.metrics[name]:
                values = [v for _, v in self.metrics[name]]
                summary[name] = {
                    'latest': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'moving_avg': self.get_moving_average(name)
                }
        return summary


class TensorBoardLogger:
    """TensorBoard logging sınıfı."""
    
    def __init__(self, log_dir: Union[str, Path], flush_secs: int = 30):
        """
        TensorBoardLogger'ı başlatır.
        
        Args:
            log_dir: Log dizini
            flush_secs: Flush aralığı (saniye)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)
        self.logger = logging.getLogger(__name__)
        
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Skaler değer loglar."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Birden fazla skaler değer loglar."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Histogram loglar."""
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int) -> None:
        """Görüntü loglar."""
        self.writer.add_image(tag, img_tensor, step)
    
    def log_text(self, tag: str, text_string: str, step: int) -> None:
        """Metin loglar."""
        self.writer.add_text(tag, text_string, step)
    
    def log_model_graph(self, model: nn.Module, input_to_model: torch.Tensor) -> None:
        """Model grafiğini loglar."""
        try:
            self.writer.add_graph(model, input_to_model)
        except Exception as e:
            self.logger.warning(f"Model grafiği loglanamadı: {e}")
    
    def log_hyperparameters(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]) -> None:
        """Hiperparametreleri loglar."""
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self) -> None:
        """Writer'ı kapatır."""
        self.writer.close()


class WandBLogger:
    """Weights & Biases logging sınıfı."""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        WandBLogger'ı başlatır.
        
        Args:
            project: Proje adı
            name: Run adı
            config: Konfigürasyon
            tags: Etiketler
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb paketi yüklü değil. 'pip install wandb' ile yükleyin.")
        
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags
        )
        self.logger = logging.getLogger(__name__)
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Metrikleri loglar."""
        wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str, name: str) -> None:
        """Model artifact'ini loglar."""
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def finish(self) -> None:
        """Run'ı bitirir."""
        wandb.finish()


class TrainingMonitor:
    """Kapsamlı eğitim monitoring sınıfı."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_every_n_steps: int = 100,
        save_metrics_every_n_steps: int = 1000
    ):
        """
        TrainingMonitor'ı başlatır.
        
        Args:
            log_dir: Log dizini
            use_tensorboard: TensorBoard kullan
            use_wandb: WandB kullan
            wandb_project: WandB proje adı
            wandb_name: WandB run adı
            config: Konfigürasyon
            log_every_n_steps: Log aralığı
            save_metrics_every_n_steps: Metrik kaydetme aralığı
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_every_n_steps = log_every_n_steps
        self.save_metrics_every_n_steps = save_metrics_every_n_steps
        
        # Metrik tracker
        self.metrics_tracker = MetricsTracker()
        
        # TensorBoard logger
        self.tensorboard_logger = None
        if use_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            self.tensorboard_logger = TensorBoardLogger(tb_dir)
        
        # WandB logger
        self.wandb_logger = None
        if use_wandb and wandb_project:
            self.wandb_logger = WandBLogger(
                project=wandb_project,
                name=wandb_name,
                config=config
            )
        
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
        # Metrik dosyası
        self.metrics_file = self.log_dir / "metrics.jsonl"
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        epoch: Optional[int] = None,
        phase: str = "train"
    ) -> None:
        """
        Metrikleri loglar.
        
        Args:
            metrics: Metrik sözlüğü
            step: Adım sayısı
            epoch: Epoch sayısı
            phase: Eğitim fazı (train, val, test)
        """
        # Metrik tracker'ı güncelle
        self.metrics_tracker.update(metrics, step)
        
        # TensorBoard'a logla
        if self.tensorboard_logger:
            for name, value in metrics.items():
                tag = f"{phase}/{name}"
                self.tensorboard_logger.log_scalar(tag, value, step)
        
        # WandB'a logla
        if self.wandb_logger:
            wandb_metrics = {f"{phase}_{name}": value for name, value in metrics.items()}
            if epoch is not None:
                wandb_metrics["epoch"] = epoch
            self.wandb_logger.log(wandb_metrics, step=step)
        
        # Konsola logla (belirli aralıklarla)
        if step % self.log_every_n_steps == 0:
            elapsed_time = time.time() - self.start_time
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            self.logger.info(
                f"Step {step} | Epoch {epoch} | {phase.upper()} | "
                f"{metrics_str} | Time: {elapsed_time:.2f}s"
            )
        
        # Metrikleri dosyaya kaydet
        if step % self.save_metrics_every_n_steps == 0:
            self._save_metrics_to_file(metrics, step, epoch, phase)
    
    def log_model_parameters(self, model: nn.Module, step: int) -> None:
        """Model parametrelerini loglar."""
        if self.tensorboard_logger:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Parametre değerleri
                    self.tensorboard_logger.log_histogram(f"parameters/{name}", param, step)
                    # Gradyan değerleri
                    self.tensorboard_logger.log_histogram(f"gradients/{name}", param.grad, step)
                    # Gradyan norm
                    grad_norm = param.grad.norm().item()
                    self.tensorboard_logger.log_scalar(f"gradient_norms/{name}", grad_norm, step)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        """Learning rate'i loglar."""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            tag = f"learning_rate/group_{i}" if len(optimizer.param_groups) > 1 else "learning_rate"
            
            if self.tensorboard_logger:
                self.tensorboard_logger.log_scalar(tag, lr, step)
            
            if self.wandb_logger:
                self.wandb_logger.log({tag: lr}, step=step)
    
    def log_system_metrics(self, step: int) -> None:
        """Sistem metriklerini loglar."""
        try:
            import psutil
            
            # CPU ve RAM kullanımı
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # GPU kullanımı (eğer varsa)
            gpu_metrics = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    gpu_memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    gpu_metrics[f"gpu_{i}_memory_allocated"] = gpu_memory
                    gpu_metrics[f"gpu_{i}_memory_cached"] = gpu_memory_cached
            
            system_metrics = {
                "system/cpu_percent": cpu_percent,
                "system/memory_percent": memory_percent,
                **gpu_metrics
            }
            
            # Log
            if self.tensorboard_logger:
                for name, value in system_metrics.items():
                    self.tensorboard_logger.log_scalar(name, value, step)
            
            if self.wandb_logger:
                self.wandb_logger.log(system_metrics, step=step)
                
        except ImportError:
            pass  # psutil yüklü değilse sistem metriklerini atla
    
    def _save_metrics_to_file(
        self,
        metrics: Dict[str, float],
        step: int,
        epoch: Optional[int],
        phase: str
    ) -> None:
        """Metrikleri dosyaya kaydeder."""
        try:
            metric_entry = {
                "step": step,
                "epoch": epoch,
                "phase": phase,
                "timestamp": time.time(),
                "metrics": metrics
            }
            
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metric_entry) + '\n')
                
        except Exception as e:
            self.logger.warning(f"Metrikler dosyaya kaydedilemedi: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Metrik özetini döndürür."""
        return self.metrics_tracker.get_summary()
    
    def close(self) -> None:
        """Logger'ları kapatır."""
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
        
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        # Final özet
        summary = self.get_metrics_summary()
        summary_file = self.log_dir / "final_metrics_summary.json"
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Final özet kaydedilemedi: {e}")
        
        total_time = time.time() - self.start_time
        self.logger.info(f"Monitoring tamamlandı. Toplam süre: {total_time:.2f}s")
