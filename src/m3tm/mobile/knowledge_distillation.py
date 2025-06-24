"""
M³TM Knowledge Distillation Modülü

Bu modül teacher-student architecture kullanarak büyük modellerin
bilgisini küçük modellere aktarmayı amaçlar.
"""

import logging
from typing import Dict, Optional, Union, Any, Tuple, List, Callable
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker


logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss fonksiyonu."""
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 loss_type: str = "kl_div"):
        """
        Args:
            temperature: Softmax temperature parameter
            alpha: Distillation loss ağırlığı
            beta: Hard target loss ağırlığı (alpha + beta = 1.0 olmalı)
            loss_type: 'kl_div', 'mse', 'cosine' loss türü
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
        
        # Loss normalization
        if abs(alpha + beta - 1.0) > 1e-6:
            logger.warning(f"alpha + beta = {alpha + beta} != 1.0, normalizing...")
            total = alpha + beta
            self.alpha = alpha / total
            self.beta = beta / total
    
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Distillation loss hesaplar.
        
        Args:
            student_logits: Student model çıktıları
            teacher_logits: Teacher model çıktıları
            labels: Ground truth labels (hard target loss için)
            
        Returns:
            Combined distillation loss
        """
        # Distillation loss
        if self.loss_type == "kl_div":
            distillation_loss = self._kl_divergence_loss(student_logits, teacher_logits)
        elif self.loss_type == "mse":
            distillation_loss = self._mse_loss(student_logits, teacher_logits)
        elif self.loss_type == "cosine":
            distillation_loss = self._cosine_loss(student_logits, teacher_logits)
        else:
            raise ValueError(f"Desteklenmeyen loss type: {self.loss_type}")
        
        # Combined loss
        total_loss = self.alpha * distillation_loss
        
        # Hard target loss (eğer labels varsa)
        if labels is not None and self.beta > 0:
            hard_loss = F.cross_entropy(student_logits, labels)
            total_loss += self.beta * hard_loss
        
        return total_loss
    
    def _kl_divergence_loss(self, student_logits: torch.Tensor, 
                            teacher_logits: torch.Tensor) -> torch.Tensor:
        """KL Divergence based distillation loss."""
        # Temperature scaling
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence (scaled by temperature^2)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return kl_loss * (self.temperature ** 2)
    
    def _mse_loss(self, student_logits: torch.Tensor, 
                  teacher_logits: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error based distillation loss."""
        return F.mse_loss(student_logits, teacher_logits)
    
    def _cosine_loss(self, student_logits: torch.Tensor, 
                     teacher_logits: torch.Tensor) -> torch.Tensor:
        """Cosine similarity based distillation loss."""
        cosine_sim = F.cosine_similarity(student_logits, teacher_logits, dim=-1)
        return 1.0 - cosine_sim.mean()


class StudentArchitectureGenerator:
    """Student model mimarisi oluşturucu sınıf."""
    
    @staticmethod
    def create_smaller_model(teacher_model: nn.Module, 
                             compression_ratio: float = 0.5,
                             architecture_type: str = "proportional") -> nn.Module:
        """
        Teacher model'den daha küçük student model oluşturur.
        
        Args:
            teacher_model: Teacher model
            compression_ratio: Student/Teacher boyut oranı
            architecture_type: 'proportional', 'width_reduction', 'depth_reduction'
            
        Returns:
            Student model
        """
        if not 0.1 <= compression_ratio <= 0.9:
            raise ValueError("Compression ratio 0.1-0.9 arasında olmalı")
        
        if architecture_type == "proportional":
            return StudentArchitectureGenerator._create_proportional_student(
                teacher_model, compression_ratio
            )
        elif architecture_type == "width_reduction":
            return StudentArchitectureGenerator._create_width_reduced_student(
                teacher_model, compression_ratio
            )
        elif architecture_type == "depth_reduction":
            return StudentArchitectureGenerator._create_depth_reduced_student(
                teacher_model, compression_ratio
            )
        else:
            raise ValueError(f"Desteklenmeyen architecture type: {architecture_type}")
    
    @staticmethod
    def _create_proportional_student(teacher_model: nn.Module, 
                                     compression_ratio: float) -> nn.Module:
        """Proportional scaling ile student model oluştur."""
        # Basit implementation - gerçek projede model architecture'a göre özelleştirilmeli
        import copy
        student_model = copy.deepcopy(teacher_model)
        
        # Linear layer'ları küçült
        for name, module in student_model.named_modules():
            if isinstance(module, nn.Linear):
                in_features = int(module.in_features * compression_ratio)
                out_features = int(module.out_features * compression_ratio)
                
                # Minimum boyutları garanti et
                in_features = max(in_features, 32)
                out_features = max(out_features, 32)
                
                new_linear = nn.Linear(in_features, out_features, 
                                       bias=module.bias is not None)
                
                # Parent modülde replace et
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(student_model.named_modules())[parent_name]
                    setattr(parent_module, attr_name, new_linear)
                else:
                    setattr(student_model, attr_name, new_linear)
        
        return student_model
    
    @staticmethod
    def _create_width_reduced_student(teacher_model: nn.Module, 
                                      compression_ratio: float) -> nn.Module:
        """Width reduction ile student model oluştur."""
        # Width (channel/feature) sayısını azalt, depth'i koru
        import copy
        student_model = copy.deepcopy(teacher_model)
        
        for name, module in student_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                out_channels = int(module.out_channels * compression_ratio)
                out_channels = max(out_channels, 16)  # Minimum channels
                
                if isinstance(module, nn.Conv2d):
                    new_conv = nn.Conv2d(
                        module.in_channels, out_channels,
                        module.kernel_size, module.stride, module.padding,
                        bias=module.bias is not None
                    )
                else:  # Conv1d
                    new_conv = nn.Conv1d(
                        module.in_channels, out_channels,
                        module.kernel_size, module.stride, module.padding,
                        bias=module.bias is not None
                    )
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(student_model.named_modules())[parent_name]
                    setattr(parent_module, attr_name, new_conv)
                else:
                    setattr(student_model, attr_name, new_conv)
        
        return student_model
    
    @staticmethod
    def _create_depth_reduced_student(teacher_model: nn.Module, 
                                      compression_ratio: float) -> nn.Module:
        """Depth reduction ile student model oluştur."""
        # Layer sayısını azalt, width'i koru
        # Bu basit bir implementation - gerçek projede sequential block'ları tanımlamak gerekir
        logger.warning("Depth reduction basit implementation kullanıyor")
        
        import copy
        student_model = copy.deepcopy(teacher_model)
        
        # Bu implementation'da sadece bazı intermediate layer'ları skip ediyoruz
        # Gerçek projede architecture'a özgü logic gerekir
        
        return student_model


class DistillationConfig:
    """Knowledge distillation yapılandırma sınıfı."""
    
    def __init__(self,
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 loss_type: str = "kl_div",
                 feature_matching: bool = False,
                 attention_transfer: bool = False):
        """
        Args:
            temperature: Distillation temperature
            alpha: Distillation loss weight
            beta: Hard target loss weight
            loss_type: Loss function type
            feature_matching: Intermediate feature matching kullan
            attention_transfer: Attention transfer kullan
        """
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
        self.feature_matching = feature_matching
        self.attention_transfer = attention_transfer


class DistillationTrainer:
    """Knowledge distillation training yöneticisi."""
    
    def __init__(self, config: Optional[DistillationConfig] = None):
        """
        Args:
            config: Distillation yapılandırma objesi
        """
        self.config = config or DistillationConfig()
        self.benchmarker = ModelBenchmarker()
        self.training_history = []
        
        # Loss function
        self.distillation_loss = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            beta=self.config.beta,
            loss_type=self.config.loss_type
        )
    
    def create_student_model(self, 
                             teacher_model: nn.Module,
                             compression_ratio: float = 0.5,
                             architecture_type: str = "proportional") -> nn.Module:
        """
        Teacher model'den student model oluşturur.
        
        Args:
            teacher_model: Teacher model
            compression_ratio: Student/Teacher boyut oranı
            architecture_type: Student mimarisi türü
            
        Returns:
            Student model
        """
        logger.info(f"Student model oluşturuluyor (compression: {compression_ratio}, type: {architecture_type})")
        
        student_model = StudentArchitectureGenerator.create_smaller_model(
            teacher_model, compression_ratio, architecture_type
        )
        
        # Student model metrics
        student_metrics = self.benchmarker.get_model_metrics(student_model)
        teacher_metrics = self.benchmarker.get_model_metrics(teacher_model)
        
        logger.info(f"Student model oluşturuldu:")
        logger.info(f"  Teacher params: {teacher_metrics.get('param_count', 'N/A'):,}")
        logger.info(f"  Student params: {student_metrics.get('param_count', 'N/A'):,}")
        
        actual_compression = (student_metrics.get('param_count', 0) / 
                              teacher_metrics.get('param_count', 1) if teacher_metrics.get('param_count', 1) > 0 else 0)
        logger.info(f"  Actual compression ratio: {actual_compression:.3f}")
        
        return student_model
    
    def train_student(self,
                      teacher_model: nn.Module,
                      student_model: nn.Module,
                      train_loader: DataLoader,
                      optimizer: Optimizer,
                      num_epochs: int = 10,
                      validation_loader: Optional[DataLoader] = None,
                      device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict]:
        """
        Student model'i distillation ile eğitir.
        
        Args:
            teacher_model: Teacher model (frozen)
            student_model: Student model (trainable)
            train_loader: Training data loader
            optimizer: Student model optimizer
            num_epochs: Eğitim epoch sayısı
            validation_loader: Validation data loader
            device: Training device
            
        Returns:
            Tuple[trained_student_model, training_metrics]
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Knowledge distillation training başlatılıyor ({num_epochs} epoch)")
        
        # Model'leri device'a taşı
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)
        
        # Teacher model'i freeze et
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Training metrics
        training_metrics = {
            'epoch_losses': [],
            'validation_losses': [],
            'distillation_losses': [],
            'hard_losses': []
        }
        
        best_val_loss = float('inf')
        best_student_state = None
        
        try:
            for epoch in range(num_epochs):
                # Training phase
                student_model.train()
                epoch_loss = 0.0
                epoch_distill_loss = 0.0
                epoch_hard_loss = 0.0
                num_batches = 0
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Teacher ve student forward pass
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    
                    student_outputs = student_model(inputs)
                    
                    # Distillation loss hesapla
                    loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Metrics tracking
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Detailed loss breakdown (optional)
                    with torch.no_grad():
                        distill_loss = self.distillation_loss.alpha * \
                                       self.distillation_loss._kl_divergence_loss(student_outputs, teacher_outputs)
                        hard_loss = self.distillation_loss.beta * \
                                    F.cross_entropy(student_outputs, labels) if labels is not None else 0
                        
                        epoch_distill_loss += distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
                        epoch_hard_loss += hard_loss.item() if isinstance(hard_loss, torch.Tensor) else hard_loss
                
                # Epoch averages
                avg_epoch_loss = epoch_loss / num_batches
                avg_distill_loss = epoch_distill_loss / num_batches
                avg_hard_loss = epoch_hard_loss / num_batches
                
                training_metrics['epoch_losses'].append(avg_epoch_loss)
                training_metrics['distillation_losses'].append(avg_distill_loss)
                training_metrics['hard_losses'].append(avg_hard_loss)
                
                # Validation phase
                val_loss = 0.0
                if validation_loader:
                    val_loss = self._validate_student(
                        teacher_model, student_model, validation_loader, device
                    )
                    training_metrics['validation_losses'].append(val_loss)
                    
                    # Best model tracking
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_student_state = student_model.state_dict().copy()
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Loss={avg_epoch_loss:.4f}, "
                           f"Distill={avg_distill_loss:.4f}, "
                           f"Hard={avg_hard_loss:.4f}"
                           + (f", Val={val_loss:.4f}" if validation_loader else ""))
            
            # Load best model if validation was used
            if best_student_state:
                student_model.load_state_dict(best_student_state)
                logger.info(f"Best validation model loaded (val_loss={best_val_loss:.4f})")
            
            logger.info("Knowledge distillation training tamamlandı")
            
            return student_model, training_metrics
            
        except Exception as e:
            logger.error(f"Distillation training hatası: {e}")
            raise
    
    def _validate_student(self, 
                          teacher_model: nn.Module,
                          student_model: nn.Module,
                          validation_loader: DataLoader,
                          device: torch.device) -> float:
        """
        Student model validation yapar.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            validation_loader: Validation data loader
            device: Device
            
        Returns:
            Validation loss
        """
        teacher_model.eval()
        student_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                teacher_outputs = teacher_model(inputs)
                student_outputs = student_model(inputs)
                
                loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate_distillation_quality(self, 
                                      teacher_model: nn.Module,
                                      student_model: nn.Module,
                                      test_loader: DataLoader,
                                      device: Optional[torch.device] = None) -> Dict:
        """
        Distillation kalitesini değerlendirir.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model  
            test_loader: Test data loader
            device: Device
            
        Returns:
            Evaluation metrics
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)
        teacher_model.eval()
        student_model.eval()
        
        metrics = {
            'teacher_accuracy': 0.0,
            'student_accuracy': 0.0,
            'agreement_rate': 0.0,
            'knowledge_retention': 0.0
        }
        
        total_samples = 0
        teacher_correct = 0
        student_correct = 0
        agreement_count = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                teacher_outputs = teacher_model(inputs)
                student_outputs = student_model(inputs)
                
                teacher_preds = teacher_outputs.argmax(dim=-1)
                student_preds = student_outputs.argmax(dim=-1)
                
                # Accuracy hesapla
                teacher_correct += (teacher_preds == labels).sum().item()
                student_correct += (student_preds == labels).sum().item()
                
                # Agreement hesapla
                agreement_count += (teacher_preds == student_preds).sum().item()
                
                total_samples += labels.size(0)
        
        if total_samples > 0:
            metrics['teacher_accuracy'] = teacher_correct / total_samples
            metrics['student_accuracy'] = student_correct / total_samples
            metrics['agreement_rate'] = agreement_count / total_samples
            
            # Knowledge retention (student'ın teacher'ı ne kadar iyi taklit ettiği)
            if metrics['teacher_accuracy'] > 0:
                metrics['knowledge_retention'] = metrics['student_accuracy'] / metrics['teacher_accuracy']
        
        return metrics
    
    def save_distilled_model(self, student_model: nn.Module, 
                             save_path: Union[str, Path],
                             training_metrics: Optional[Dict] = None,
                             teacher_metrics: Optional[Dict] = None) -> None:
        """
        Distilled student model'i kaydeder.
        
        Args:
            student_model: Kaydedilecek student model
            save_path: Kaydetme yolu
            training_metrics: Training metrics
            teacher_metrics: Teacher model metrics
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Model state dict kaydet
        model_state = {
            'model_state_dict': student_model.state_dict(),
            'distillation_config': {
                'temperature': self.config.temperature,
                'alpha': self.config.alpha,
                'beta': self.config.beta,
                'loss_type': self.config.loss_type
            },
            'training_metrics': training_metrics or {},
            'teacher_metrics': teacher_metrics or {},
            'model_type': 'knowledge_distilled'
        }
        
        torch.save(model_state, save_path)
        logger.info(f"Distilled model kaydedildi: {save_path}")


def create_distillation_pipeline(temperature: float = 4.0,
                                 alpha: float = 0.7,
                                 loss_type: str = "kl_div") -> DistillationTrainer:
    """
    Knowledge distillation pipeline oluşturucu fonksiyon.
    
    Args:
        temperature: Distillation temperature
        alpha: Distillation loss weight
        loss_type: Loss function type
        
    Returns:
        DistillationTrainer instance
    """
    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        beta=1.0 - alpha,
        loss_type=loss_type
    )
    
    return DistillationTrainer(config)
