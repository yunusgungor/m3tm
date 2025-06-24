"""
M³TM Model Pruning Modülü

Bu modül PyTorch'un pruning API'sini kullanarak model parametrelerini 
azaltmayı ve model boyutunu küçültmeyi amaçlar.
"""

import logging
from typing import Dict, Optional, Union, Any, Tuple, List, Callable, Set
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import BasePruningMethod, L1Unstructured, L2Unstructured, RandomUnstructured
import numpy as np

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker


logger = logging.getLogger(__name__)


class StructuredChannelPruning(BasePruningMethod):
    """Kanal bazlı structured pruning implementation."""
    
    PRUNING_TYPE = "structured"
    
    def __init__(self, amount: Union[int, float]):
        """
        Args:
            amount: Prune edilecek kanal sayısı veya yüzdesi
        """
        self.amount = amount
        
    def compute_mask(self, tensor: torch.Tensor, default_mask: torch.Tensor) -> torch.Tensor:
        """
        Structured pruning mask hesaplar.
        
        Args:
            tensor: Prune edilecek weight tensor
            default_mask: Varsayılan mask
            
        Returns:
            Computed pruning mask
        """
        # Channel importance hesapla (L1-norm bazlı)
        if tensor.dim() == 4:  # Conv2d weight: [out_channels, in_channels, H, W]
            channel_importance = tensor.abs().sum(dim=(1, 2, 3))
            num_channels = tensor.size(0)
        elif tensor.dim() == 2:  # Linear weight: [out_features, in_features]  
            channel_importance = tensor.abs().sum(dim=1)
            num_channels = tensor.size(0)
        else:
            # Unsupported tensor dimension
            return default_mask
            
        # Prune edilecek kanal sayısını belirle
        if isinstance(self.amount, float):
            num_to_prune = int(self.amount * num_channels)
        else:
            num_to_prune = min(self.amount, num_channels)
            
        # En az önemli kanalları bul
        _, prune_indices = torch.topk(channel_importance, num_to_prune, largest=False)
        
        # Mask oluştur
        mask = default_mask.clone()
        if tensor.dim() == 4:
            mask[prune_indices] = 0
        elif tensor.dim() == 2:
            mask[prune_indices] = 0
            
        return mask
    
    @classmethod
    def apply(cls, module: nn.Module, name: str, amount: Union[int, float]) -> 'StructuredChannelPruning':
        """
        Modüle structured pruning uygular.
        
        Args:
            module: Prune edilecek modül
            name: Parameter adı
            amount: Prune miktarı
            
        Returns:
            StructuredChannelPruning instance
        """
        return super().apply(module, name, amount=amount)


class PruningConfig:
    """Pruning yapılandırma sınıfı."""
    
    def __init__(self,
                 pruning_type: str = "unstructured",
                 pruning_method: str = "magnitude",
                 amount: float = 0.2,
                 global_pruning: bool = False,
                 preserve_ratio: float = 0.1,
                 structured_dimensions: Optional[List[int]] = None):
        """
        Args:
            pruning_type: 'structured' veya 'unstructured'
            pruning_method: 'magnitude', 'random', 'gradient' vs.
            amount: Prune edilecek parametrelerin yüzdesi (0.0-1.0)
            global_pruning: Global pruning kullanılsın mı
            preserve_ratio: Korunacak minimum parametre oranı
            structured_dimensions: Structured pruning için boyutlar
        """
        self.pruning_type = pruning_type
        self.pruning_method = pruning_method
        self.amount = amount
        self.global_pruning = global_pruning
        self.preserve_ratio = preserve_ratio
        self.structured_dimensions = structured_dimensions or []
        
        # Validation
        if not 0.0 <= amount <= 1.0:
            raise ValueError("Pruning amount 0.0-1.0 arasında olmalı")
        if not 0.0 <= preserve_ratio <= 1.0:
            raise ValueError("Preserve ratio 0.0-1.0 arasında olmalı")


class PruningManager:
    """Gelişmiş model pruning yöneticisi."""
    
    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Args:
            config: Pruning yapılandırma objesi
        """
        self.config = config or PruningConfig()
        self.benchmarker = ModelBenchmarker()
        self.pruning_history = []
        
    def apply_unstructured_pruning(self, 
                                   model: Union[nn.Module, BaseModel],
                                   modules_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
                                   benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Unstructured pruning uygular.
        
        Args:
            model: Prune edilecek model
            modules_to_prune: Prune edilecek (module, parameter_name) listesi
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[pruned_model, metrics]
        """
        logger.info("Unstructured pruning başlatılıyor...")
        
        # Orijinal model metrics
        original_metrics = {}
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)
        
        try:
            # Prune edilecek modülleri belirle
            if modules_to_prune is None:
                modules_to_prune = self._get_default_modules_to_prune(model)
            
            # Pruning method seç
            if self.config.pruning_method == "magnitude":
                pruning_fn = L1Unstructured
            elif self.config.pruning_method == "l2":
                pruning_fn = L2Unstructured
            elif self.config.pruning_method == "random":
                pruning_fn = RandomUnstructured
            else:
                raise ValueError(f"Desteklenmeyen pruning method: {self.config.pruning_method}")
            
            # Global pruning kullan
            if self.config.global_pruning:
                prune.global_unstructured(
                    modules_to_prune,
                    pruning_method=pruning_fn,
                    amount=self.config.amount
                )
            else:
                # Her modül için ayrı pruning
                for module, parameter_name in modules_to_prune:
                    pruning_fn.apply(module, parameter_name, amount=self.config.amount)
            
            # Sparsity hesapla
            sparsity_info = self._calculate_sparsity(model, modules_to_prune)
            
            # Pruned model metrics
            pruned_metrics = {}
            if benchmark:
                pruned_metrics = self.benchmarker.get_model_metrics(model)
                
            # Compression metrics hesapla
            compression_metrics = self._calculate_pruning_metrics(
                original_metrics, pruned_metrics, sparsity_info
            )
            
            logger.info(f"Unstructured pruning tamamlandı. "
                       f"Sparsity: {sparsity_info.get('global_sparsity', 0):.1f}%")
            
            return model, {
                'original_metrics': original_metrics,
                'pruned_metrics': pruned_metrics,
                'compression_metrics': compression_metrics,
                'sparsity_info': sparsity_info,
                'pruning_type': 'unstructured',
                'pruning_method': self.config.pruning_method,
                'amount': self.config.amount
            }
            
        except Exception as e:
            logger.error(f"Unstructured pruning hatası: {e}")
            raise
    
    def apply_structured_pruning(self,
                                 model: Union[nn.Module, BaseModel],
                                 modules_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
                                 benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Structured pruning uygular.
        
        Args:
            model: Prune edilecek model
            modules_to_prune: Prune edilecek (module, parameter_name) listesi
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[pruned_model, metrics]
        """
        logger.info("Structured pruning başlatılıyor...")
        
        # Orijinal model metrics
        original_metrics = {}
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)
        
        try:
            # Prune edilecek modülleri belirle
            if modules_to_prune is None:
                modules_to_prune = self._get_default_modules_to_prune(model)
            
            # Structured pruning uygula
            for module, parameter_name in modules_to_prune:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    StructuredChannelPruning.apply(
                        module, 
                        parameter_name, 
                        amount=self.config.amount
                    )
            
            # Sparsity hesapla
            sparsity_info = self._calculate_sparsity(model, modules_to_prune)
            
            # Pruned model metrics
            pruned_metrics = {}
            if benchmark:
                pruned_metrics = self.benchmarker.get_model_metrics(model)
                
            # Compression metrics hesapla
            compression_metrics = self._calculate_pruning_metrics(
                original_metrics, pruned_metrics, sparsity_info
            )
            
            logger.info(f"Structured pruning tamamlandı. "
                       f"Sparsity: {sparsity_info.get('global_sparsity', 0):.1f}%")
            
            return model, {
                'original_metrics': original_metrics,
                'pruned_metrics': pruned_metrics,
                'compression_metrics': compression_metrics,
                'sparsity_info': sparsity_info,
                'pruning_type': 'structured',
                'amount': self.config.amount
            }
            
        except Exception as e:
            logger.error(f"Structured pruning hatası: {e}")
            raise
    
    def apply_progressive_pruning(self,
                                  model: Union[nn.Module, BaseModel],
                                  target_sparsity: float,
                                  num_steps: int = 5,
                                  fine_tune_callback: Optional[Callable] = None,
                                  benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Progressive (aşamalı) pruning uygular.
        
        Args:
            model: Prune edilecek model
            target_sparsity: Hedef sparsity oranı
            num_steps: Aşama sayısı
            fine_tune_callback: Her aşama sonrası fine-tuning fonksiyonu
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[final_pruned_model, metrics]
        """
        logger.info(f"Progressive pruning başlatılıyor (target: {target_sparsity:.1f}%, {num_steps} aşama)")
        
        # Orijinal model metrics
        original_metrics = {}
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)
        
        try:
            current_sparsity = 0.0
            step_size = target_sparsity / num_steps
            step_metrics = []
            
            for step in range(num_steps):
                current_target = min(current_sparsity + step_size, target_sparsity)
                step_amount = step_size / (1.0 - current_sparsity) if current_sparsity < 1.0 else 0.0
                
                logger.info(f"Progressive pruning step {step + 1}/{num_steps} "
                           f"(target sparsity: {current_target:.1f}%)")
                
                # Bu adım için geçici config oluştur
                temp_config = PruningConfig(
                    pruning_type=self.config.pruning_type,
                    pruning_method=self.config.pruning_method,
                    amount=step_amount,
                    global_pruning=self.config.global_pruning
                )
                
                # Temp manager oluştur ve pruning uygula
                temp_manager = PruningManager(temp_config)
                
                if self.config.pruning_type == "structured":
                    model, step_result = temp_manager.apply_structured_pruning(model, benchmark=False)
                else:
                    model, step_result = temp_manager.apply_unstructured_pruning(model, benchmark=False)
                
                current_sparsity = step_result['sparsity_info'].get('global_sparsity', current_sparsity)
                step_metrics.append(step_result)
                
                # Fine-tuning callback çağır
                if fine_tune_callback:
                    logger.info(f"Fine-tuning step {step + 1}")
                    model = fine_tune_callback(model)
            
            # Final metrics
            final_metrics = {}
            if benchmark:
                final_metrics = self.benchmarker.get_model_metrics(model)
            
            # Final sparsity hesapla
            modules_to_prune = self._get_default_modules_to_prune(model)
            final_sparsity_info = self._calculate_sparsity(model, modules_to_prune)
            
            # Compression metrics hesapla
            compression_metrics = self._calculate_pruning_metrics(
                original_metrics, final_metrics, final_sparsity_info
            )
            
            logger.info(f"Progressive pruning tamamlandı. "
                       f"Final sparsity: {final_sparsity_info.get('global_sparsity', 0):.1f}%")
            
            return model, {
                'original_metrics': original_metrics,
                'final_metrics': final_metrics,
                'compression_metrics': compression_metrics,
                'final_sparsity_info': final_sparsity_info,
                'step_metrics': step_metrics,
                'pruning_type': 'progressive',
                'target_sparsity': target_sparsity,
                'num_steps': num_steps
            }
            
        except Exception as e:
            logger.error(f"Progressive pruning hatası: {e}")
            raise
    
    def remove_pruning_masks(self, model: nn.Module) -> nn.Module:
        """
        Pruning mask'lerini kaldırır ve sparse parametreleri kalıcı hale getirir.
        
        Args:
            model: Mask'leri kaldırılacak model
            
        Returns:
            Permanent pruned model
        """
        logger.info("Pruning mask'leri kaldırılıyor...")
        
        try:
            for module in model.modules():
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
                if hasattr(module, 'bias_mask'):
                    prune.remove(module, 'bias')
            
            logger.info("Pruning mask'leri başarıyla kaldırıldı")
            return model
            
        except Exception as e:
            logger.error(f"Mask kaldırma hatası: {e}")
            raise
    
    def _get_default_modules_to_prune(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        """
        Varsayılan olarak prune edilecek modülleri belirler.
        
        Args:
            model: Analiz edilecek model
            
        Returns:
            (module, parameter_name) tuple'larının listesi
        """
        modules_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                modules_to_prune.append((module, 'weight'))
                # Bias varsa onu da ekle
                if hasattr(module, 'bias') and module.bias is not None:
                    modules_to_prune.append((module, 'bias'))
        
        return modules_to_prune
    
    def _calculate_sparsity(self, model: nn.Module, 
                            modules_to_prune: List[Tuple[nn.Module, str]]) -> Dict:
        """
        Model sparsity'sini hesaplar.
        
        Args:
            model: Sparsity hesaplanacak model
            modules_to_prune: Prune edilmiş modüller
            
        Returns:
            Sparsity information dict
        """
        total_params = 0
        total_zeros = 0
        layer_sparsities = {}
        
        for module, param_name in modules_to_prune:
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if param is not None:
                    param_total = param.numel()
                    param_zeros = (param == 0).sum().item()
                    
                    total_params += param_total
                    total_zeros += param_zeros
                    
                    layer_name = f"{module.__class__.__name__}_{param_name}"
                    layer_sparsity = param_zeros / param_total * 100 if param_total > 0 else 0
                    layer_sparsities[layer_name] = layer_sparsity
        
        global_sparsity = total_zeros / total_params * 100 if total_params > 0 else 0
        
        return {
            'global_sparsity': global_sparsity,
            'total_params': total_params,
            'total_zeros': total_zeros,
            'layer_sparsities': layer_sparsities
        }
    
    def _calculate_pruning_metrics(self, original_metrics: Dict, 
                                   pruned_metrics: Dict, 
                                   sparsity_info: Dict) -> Dict:
        """
        Pruning compression metrics hesaplar.
        
        Args:
            original_metrics: Orijinal model metrikleri
            pruned_metrics: Pruned model metrikleri
            sparsity_info: Sparsity bilgileri
            
        Returns:
            Compression metrics dict
        """
        compression_metrics = sparsity_info.copy()
        
        if not original_metrics or not pruned_metrics:
            return compression_metrics
        
        # Effective compression (sparsity'yi dikkate alarak)
        sparsity_ratio = sparsity_info.get('global_sparsity', 0) / 100
        effective_compression = sparsity_ratio
        
        compression_metrics.update({
            'effective_compression_ratio': effective_compression,
            'theoretical_size_reduction': sparsity_ratio * 100
        })
        
        # Actual metrics comparison
        if 'model_size_mb' in original_metrics and 'model_size_mb' in pruned_metrics:
            original_size = original_metrics['model_size_mb']
            pruned_size = pruned_metrics['model_size_mb']
            
            if original_size > 0:
                actual_size_reduction = (original_size - pruned_size) / original_size * 100
                compression_metrics.update({
                    'actual_size_reduction_percent': actual_size_reduction,
                    'original_size_mb': original_size,
                    'pruned_size_mb': pruned_size
                })
        
        return compression_metrics
    
    def save_pruned_model(self, model: nn.Module, save_path: Union[str, Path],
                          metadata: Optional[Dict] = None, 
                          remove_masks: bool = True) -> None:
        """
        Pruned modeli kaydeder.
        
        Args:
            model: Kaydedilecek pruned model
            save_path: Kaydetme yolu
            metadata: Ek metadata bilgileri
            remove_masks: Mask'leri kaldırıp kalıcı hale getir
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Mask'leri kaldır (istenirse)
        if remove_masks:
            model = self.remove_pruning_masks(model)
        
        # Model state dict kaydet
        model_state = {
            'model_state_dict': model.state_dict(),
            'pruning_metadata': metadata or {},
            'model_type': 'pruned',
            'config': {
                'pruning_type': self.config.pruning_type,
                'pruning_method': self.config.pruning_method,
                'amount': self.config.amount
            }
        }
        
        torch.save(model_state, save_path)
        logger.info(f"Pruned model kaydedildi: {save_path}")


def create_pruning_pipeline(pruning_type: str = "unstructured",
                            pruning_method: str = "magnitude", 
                            amount: float = 0.2,
                            global_pruning: bool = False) -> PruningManager:
    """
    Pruning pipeline oluşturucu fonksiyon.
    
    Args:
        pruning_type: 'structured' veya 'unstructured'
        pruning_method: 'magnitude', 'random', vs.
        amount: Prune edilecek parametre yüzdesi
        global_pruning: Global pruning kullanılsın mı
        
    Returns:
        PruningManager instance
    """
    config = PruningConfig(
        pruning_type=pruning_type,
        pruning_method=pruning_method,
        amount=amount,
        global_pruning=global_pruning
    )
    
    return PruningManager(config)
