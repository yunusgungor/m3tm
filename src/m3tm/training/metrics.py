"""
Eğitim ve Değerlendirme Metrikleri Modülü

Bu modül, model eğitimi ve değerlendirmesi sırasında çeşitli metrikleri
hesaplamak, toplamak ve raporlamak için sınıfları içerir.

Örüntüler:
- MetricsCollector (PT-008): Performans metriklerini tutarlı şekilde toplayan, 
  işleyen ve raporlayan koleksiyon deseni.
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainingMetrics:
    """
    Eğitim esnasında toplanan metrikler.
    
    Örüntü: MetricsCollector (PT-008)
    """
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    train_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    parameter_count: int = 0
    
    # İsteğe bağlı metrikler
    train_f1: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    train_precision: List[float] = field(default_factory=list)
    val_precision: List[float] = field(default_factory=list)
    train_recall: List[float] = field(default_factory=list)
    val_recall: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Metrikleri sözlük olarak döndürür.
        
        Returns:
            Dict[str, Any]: Metrik değerlerini içeren sözlük
        """
        return asdict(self)
    
    def save_json(self, file_path: Union[str, Path]) -> None:
        """
        Metrikleri JSON formatında kaydeder.
        
        Args:
            file_path: Metriklerin kaydedileceği dosya yolu
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> 'TrainingMetrics':
        """
        JSON dosyasından metrikleri yükler.
        
        Args:
            file_path: Metriklerin yükleneceği dosya yolu
            
        Returns:
            TrainingMetrics: Yüklenen metrikler
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return cls(**data)


class MetricsCollector:
    """
    Eğitim ve değerlendirme metriklerini toplayan, işleyen ve raporlayan sınıf.
    
    Örüntü: MetricsCollector (PT-008)
    """
    
    def __init__(
        self,
        save_dir: Optional[Union[str, Path]] = None,
        use_tensorboard: bool = False
    ):
        """
        MetricsCollector sınıfını başlatır.
        
        Args:
            save_dir: Metriklerin ve logların kaydedileceği dizin
            use_tensorboard: TensorBoard kullanılsın mı?
        """
        self.metrics = TrainingMetrics()
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.start_time = time.time()
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        # Kaydetme dizini
        if save_dir is not None:
            if isinstance(save_dir, str):
                save_dir = Path(save_dir)
                
            self.save_dir = save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            # TensorBoard writer'ı başlat
            if self.use_tensorboard:
                self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))
        else:
            self.save_dir = None
    
    def update_train_metrics(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        train_time: Optional[float] = None,
        memory_usage: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1: Optional[float] = None
    ) -> None:
        """
        Eğitim metriklerini günceller.
        
        Args:
            loss: Eğitim kaybı
            accuracy: Eğitim doğruluğu
            learning_rate: Öğrenme oranı
            train_time: Eğitim süresi (saniye)
            memory_usage: Bellek kullanımı (MB)
            precision: Precision metriği
            recall: Recall metriği
            f1: F1 skoru
        """
        self.metrics.train_loss.append(loss)
        
        if accuracy is not None:
            self.metrics.train_accuracy.append(accuracy)
            
        if learning_rate is not None:
            self.metrics.learning_rates.append(learning_rate)
            
        if train_time is not None:
            self.metrics.train_times.append(train_time)
            
        if memory_usage is not None:
            self.metrics.memory_usage.append(memory_usage)
            
        if precision is not None:
            self.metrics.train_precision.append(precision)
            
        if recall is not None:
            self.metrics.train_recall.append(recall)
            
        if f1 is not None:
            self.metrics.train_f1.append(f1)
        
        # TensorBoard'a yaz
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('Loss/train', loss, self.global_step)
            
            if accuracy is not None:
                self.writer.add_scalar('Accuracy/train', accuracy, self.global_step)
                
            if learning_rate is not None:
                self.writer.add_scalar('Learning_rate', learning_rate, self.global_step)
                
            if train_time is not None:
                self.writer.add_scalar('Time/train', train_time, self.global_step)
                
            if memory_usage is not None:
                self.writer.add_scalar('Memory_usage', memory_usage, self.global_step)
                
            if precision is not None:
                self.writer.add_scalar('Precision/train', precision, self.global_step)
                
            if recall is not None:
                self.writer.add_scalar('Recall/train', recall, self.global_step)
                
            if f1 is not None:
                self.writer.add_scalar('F1/train', f1, self.global_step)
        
        self.global_step += 1
    
    def update_val_metrics(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1: Optional[float] = None
    ) -> bool:
        """
        Doğrulama metriklerini günceller.
        
        Args:
            loss: Doğrulama kaybı
            accuracy: Doğrulama doğruluğu
            precision: Precision metriği
            recall: Recall metriği
            f1: F1 skoru
            
        Returns:
            bool: Yeni en iyi model elde edildi mi?
        """
        self.metrics.val_loss.append(loss)
        improved = False
        
        if accuracy is not None:
            self.metrics.val_accuracy.append(accuracy)
            
            # En iyi doğruluk değerini güncelle
            if accuracy > self.best_val_accuracy:
                self.best_val_accuracy = accuracy
                improved = True
                
        if precision is not None:
            self.metrics.val_precision.append(precision)
            
        if recall is not None:
            self.metrics.val_recall.append(recall)
            
        if f1 is not None:
            self.metrics.val_f1.append(f1)
        
        # En iyi kayıp değerini güncelle
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            improved = True
        
        # TensorBoard'a yaz
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('Loss/val', loss, self.global_step)
            
            if accuracy is not None:
                self.writer.add_scalar('Accuracy/val', accuracy, self.global_step)
                
            if precision is not None:
                self.writer.add_scalar('Precision/val', precision, self.global_step)
                
            if recall is not None:
                self.writer.add_scalar('Recall/val', recall, self.global_step)
                
            if f1 is not None:
                self.writer.add_scalar('F1/val', f1, self.global_step)
        
        self.epoch += 1
        
        return improved
    
    def set_parameter_count(self, count: int) -> None:
        """
        Eğitilebilir parametre sayısını ayarlar.
        
        Args:
            count: Parametre sayısı
        """
        self.metrics.parameter_count = count
        
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('Stats/parameter_count', count, 0)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Metriklerin özetini döndürür.
        
        Returns:
            Dict[str, Any]: Metrik özetini içeren sözlük
        """
        summary = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "parameter_count": self.metrics.parameter_count,
            "total_training_time": time.time() - self.start_time
        }
        
        # Son metrik değerlerini ekle
        if self.metrics.train_loss:
            summary["last_train_loss"] = self.metrics.train_loss[-1]
            
        if self.metrics.val_loss:
            summary["last_val_loss"] = self.metrics.val_loss[-1]
            
        if self.metrics.train_accuracy:
            summary["last_train_accuracy"] = self.metrics.train_accuracy[-1]
            
        if self.metrics.val_accuracy:
            summary["last_val_accuracy"] = self.metrics.val_accuracy[-1]
            
        if self.metrics.train_f1:
            summary["last_train_f1"] = self.metrics.train_f1[-1]
            
        if self.metrics.val_f1:
            summary["last_val_f1"] = self.metrics.val_f1[-1]
        
        # Ortalama hesapla
        if self.metrics.train_times:
            summary["avg_train_time"] = np.mean(self.metrics.train_times)
            
        if self.metrics.memory_usage:
            summary["avg_memory_usage"] = np.mean(self.metrics.memory_usage)
        
        return summary
    
    def save_metrics(self, file_name: str = "metrics.json") -> None:
        """
        Metrikleri dosyaya kaydeder.
        
        Args:
            file_name: Kaydedilecek dosya adı
        """
        if self.save_dir is not None:
            file_path = self.save_dir / file_name
            self.metrics.save_json(file_path)
    
    def save_summary(self, file_name: str = "summary.json") -> None:
        """
        Metrik özetini dosyaya kaydeder.
        
        Args:
            file_name: Kaydedilecek dosya adı
        """
        if self.save_dir is not None:
            file_path = self.save_dir / file_name
            summary = self.get_summary()
            
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=2)
    
    def plot_metrics(self, file_name: str = "metrics.png", figsize: tuple = (12, 8)) -> None:
        """
        Metrikleri çizer ve kaydeder.
        
        Args:
            file_name: Kaydedilecek dosya adı
            figsize: Figür boyutu
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.save_dir is not None:
                file_path = self.save_dir / file_name
                
                # Figür oluştur
                fig, axes = plt.subplots(2, 2, figsize=figsize)
                
                # Kayıp grafiği
                if self.metrics.train_loss and self.metrics.val_loss:
                    axes[0, 0].plot(self.metrics.train_loss, label='Eğitim')
                    axes[0, 0].plot(self.metrics.val_loss, label='Doğrulama')
                    axes[0, 0].set_title('Kayıp')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Kayıp')
                    axes[0, 0].legend()
                
                # Doğruluk grafiği
                if self.metrics.train_accuracy and self.metrics.val_accuracy:
                    axes[0, 1].plot(self.metrics.train_accuracy, label='Eğitim')
                    axes[0, 1].plot(self.metrics.val_accuracy, label='Doğrulama')
                    axes[0, 1].set_title('Doğruluk')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Doğruluk')
                    axes[0, 1].legend()
                
                # Öğrenme oranı grafiği
                if self.metrics.learning_rates:
                    axes[1, 0].plot(self.metrics.learning_rates)
                    axes[1, 0].set_title('Öğrenme Oranı')
                    axes[1, 0].set_xlabel('Adım')
                    axes[1, 0].set_ylabel('Öğrenme Oranı')
                
                # Eğitim süresi grafiği
                if self.metrics.train_times:
                    axes[1, 1].plot(self.metrics.train_times)
                    axes[1, 1].set_title('Eğitim Süresi (Saniye)')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Süre (s)')
                
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
        except ImportError:
            print("Matplotlib bulunamadı. Grafik çizilemedi.")
    
    def close(self) -> None:
        """TensorBoard writer'ı kapatır."""
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Doğruluk metriğini hesaplar.
    
    Args:
        predictions: Model tahminleri [batch_size, num_classes] veya [batch_size] (indeksler)
        targets: Hedef etiketler [batch_size] (indeksler)
        
    Returns:
        float: Doğruluk değeri (0-1 arası)
    """
    if predictions.dim() > 1 and predictions.size(1) > 1:
        # Softmax logit'leri: En yüksek olasılığa sahip sınıfı seç
        predictions = predictions.argmax(dim=1)
    
    # İkili sınıflandırma için (tek sınıf olasılığı)
    if predictions.dim() > 1 and predictions.size(1) == 1:
        predictions = (predictions > 0.5).long().squeeze()
    
    # Doğru tahmin sayısı
    correct = (predictions == targets).float().sum().item()
    
    # Toplam örnek sayısı
    total = targets.size(0)
    
    return correct / total


def calculate_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Sınıflandırma metriklerini (doğruluk, precision, recall, F1) hesaplar.
    
    Args:
        predictions: Model tahminleri [batch_size, num_classes] veya [batch_size] (indeksler)
        targets: Hedef etiketler [batch_size] (indeksler)
        
    Returns:
        Dict[str, float]: Metrik değerlerini içeren sözlük
    """
    if predictions.dim() > 1 and predictions.size(1) > 1:
        # Softmax logit'leri: En yüksek olasılığa sahip sınıfı seç
        preds = predictions.argmax(dim=1)
    elif predictions.dim() > 1 and predictions.size(1) == 1:
        # İkili sınıflandırma için (tek sınıf olasılığı)
        preds = (predictions > 0.5).long().squeeze()
    else:
        preds = predictions
    
    # NumPy'a dönüştür
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Metrikler
    metrics = {}
    
    # Doğruluk
    metrics["accuracy"] = (preds_np == targets_np).mean()
    
    # Sınıf sayısını belirle
    num_classes = max(preds_np.max(), targets_np.max()) + 1
    
    # Binary sınıflandırma için precision, recall, F1
    if num_classes == 2:
        # True positives, false positives, false negatives
        tp = ((preds_np == 1) & (targets_np == 1)).sum()
        fp = ((preds_np == 1) & (targets_np == 0)).sum()
        fn = ((preds_np == 0) & (targets_np == 1)).sum()
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["precision"] = precision
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["recall"] = recall
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics["f1"] = f1
    
    return metrics 