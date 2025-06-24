"""
M³TM Model Pruning Modülü - Context7 Enhanced

Bu modül PyTorch'un latest pruning API'sini kullanarak model parametrelerini
azaltmayı ve model boyutunu küçültmeyi amaçlar. Context7 documentation'dan
alınan current best practices uygulanmıştır.

Desteklenen Teknikler (Context7 Based):
- Structured Pruning: Gerçek hızlanma sağlar (channel/filter level)
- Unstructured Pruning: L1, L2, random weight pruning
- Global Unstructured Pruning: Model-wide pruning optimization
- Magnitude-based Pruning: Weight importance based pruning

Best Practices (Context7):
- FX Graph Mode compatibility için symbolic_trace kontrolü
- Structured pruning for real speedup gains
- Proper mask management and removal after pruning
- Progressive pruning with validation checkpoints
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Context7: Check for experimental structured pruning
try:
    from torch.ao.pruning._experimental.pruner import SaliencyPruner
    from torch.fx import symbolic_trace
    EXPERIMENTAL_PRUNING_AVAILABLE = True
except ImportError:
    EXPERIMENTAL_PRUNING_AVAILABLE = False

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker

logger = logging.getLogger(__name__)


class ContextualStructuredPruning(prune.BasePruningMethod):
    """
    Context7-enhanced structured channel pruning implementation.
    
    Implements structured pruning that removes entire channels/filters,
    providing actual speedup gains unlike unstructured pruning.
    """
    PRUNING_TYPE = 'structured'

    def __init__(self, amount: float, dim: int = 0):
        """
        Args:
            amount: Pruning ratio (0.0 to 1.0)
            dim: Dimension to prune (0 for output channels, 1 for input channels)
        """
        self.amount = amount
        self.dim = dim

    def compute_mask(self, tensor: torch.Tensor, default_mask: torch.Tensor) -> torch.Tensor:
        """
        Context7-enhanced structured pruning mask computation.
        
        Args:
            tensor: Prune edilecek weight tensor
            default_mask: Varsayılan mask
            
        Returns:
            Computed structured pruning mask
        """
        # Context7: Different strategies for different tensor types
        if tensor.dim() == 4:  # Conv weight: [out_channels, in_channels, H, W]
            return self._compute_conv_mask(tensor, default_mask)
        elif tensor.dim() == 2:  # Linear weight: [out_features, in_features]
            return self._compute_linear_mask(tensor, default_mask)
        else:
            logger.warning(f"Unsupported tensor dimension for structured pruning: {tensor.dim()}")
            return default_mask

    def _compute_conv_mask(self, tensor: torch.Tensor, default_mask: torch.Tensor) -> torch.Tensor:
        """Context7: Convolutional layer structured pruning"""
        if self.dim == 0:  # Prune output channels
            # L2 norm of each output channel
            channel_norms = torch.norm(tensor.view(tensor.size(0), -1), dim=1)
            num_prune = int(self.amount * tensor.size(0))
            
            if num_prune > 0:
                _, indices = torch.topk(channel_norms, num_prune, largest=False)
                mask = default_mask.clone()
                mask[indices] = 0
                return mask
        elif self.dim == 1:  # Prune input channels
            # L2 norm of each input channel
            channel_norms = torch.norm(tensor.view(tensor.size(1), -1), dim=1)
            num_prune = int(self.amount * tensor.size(1))
            
            if num_prune > 0:
                _, indices = torch.topk(channel_norms, num_prune, largest=False)
                mask = default_mask.clone()
                mask[:, indices] = 0
                return mask
                
        return default_mask

    def _compute_linear_mask(self, tensor: torch.Tensor, default_mask: torch.Tensor) -> torch.Tensor:
        """Context7: Linear layer structured pruning"""
        if self.dim == 0:  # Prune output features
            feature_norms = torch.norm(tensor, dim=1)
            num_prune = int(self.amount * tensor.size(0))
            
            if num_prune > 0:
                _, indices = torch.topk(feature_norms, num_prune, largest=False)
                mask = default_mask.clone()
                mask[indices] = 0
                return mask
        elif self.dim == 1:  # Prune input features
            feature_norms = torch.norm(tensor, dim=0)
            num_prune = int(self.amount * tensor.size(1))
            
            if num_prune > 0:
                _, indices = torch.topk(feature_norms, num_prune, largest=False)
                mask = default_mask.clone()
                mask[:, indices] = 0
                return mask
                
        return default_mask


class PruningManager:
    """
    M³TM Model Pruning Manager - Context7 Enhanced
    
    Context7 documentation'dan alınan best practices ile geliştirilmiş
    pruning manager. Hem structured hem unstructured pruning destekler.
    
    Features:
    - Structured pruning for real speedup gains
    - Unstructured pruning for maximum compression
    - Global pruning optimization
    - FX Graph Mode compatibility
    - Progressive pruning with validation
    """

    def __init__(self,
                 benchmarker: Optional[ModelBenchmarker] = None,
                 enable_experimental: bool = True):
        """
        Context7-enhanced PruningManager initialization.
        
        Args:
            benchmarker: Performans ölçüm aracı
            enable_experimental: Experimental structured pruning kullanılsın mı
        """
        self.benchmarker = benchmarker or ModelBenchmarker()
        self.enable_experimental = enable_experimental and EXPERIMENTAL_PRUNING_AVAILABLE
        
        if self.enable_experimental:
            logger.info("Experimental structured pruning enabled")
        else:
            logger.info("Using standard PyTorch pruning APIs")

    def apply_unstructured_pruning(self,
                                   model: Union[nn.Module, BaseModel],
                                   sparsity: float = 0.2,
                                   method: str = "l1",
                                   global_pruning: bool = True,
                                   benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Context7-enhanced unstructured pruning - maximum compression ratio.
        
        Args:
            model: Prune edilecek model
            sparsity: Pruning ratio (0.0 to 1.0)
            method: Pruning method ('l1', 'l2', 'random')
            global_pruning: Global pruning optimization kullanılsın mı
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[pruned_model, metrics]
        """
        logger.info(f"Context7-enhanced unstructured pruning başlatılıyor (method={method}, sparsity={sparsity})")
        
        original_model = model
        pruned_model = model
        
        try:
            # Context7: Global pruning for better optimization
            if global_pruning:
                parameters_to_prune = []
                for name, module in pruned_model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        parameters_to_prune.append((module, 'weight'))
                
                if parameters_to_prune:
                    if method == "l1":
                        prune.global_unstructured(
                            parameters_to_prune,
                            pruning_method=prune.L1Unstructured,
                            amount=sparsity
                        )
                    elif method == "l2":
                        prune.global_unstructured(
                            parameters_to_prune,
                            pruning_method=prune.LnStructured,
                            amount=sparsity,
                            n=2
                        )
                    elif method == "random":
                        prune.global_unstructured(
                            parameters_to_prune,
                            pruning_method=prune.RandomUnstructured,
                            amount=sparsity
                        )
                    else:
                        raise ValueError(f"Unsupported pruning method: {method}")
            else:
                # Context7: Layer-wise pruning
                for name, module in pruned_model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        if method == "l1":
                            prune.l1_unstructured(module, name='weight', amount=sparsity)
                        elif method == "l2":
                            prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=0)
                        elif method == "random":
                            prune.random_unstructured(module, name='weight', amount=sparsity)
                            
            logger.info("Unstructured pruning tamamlandı")
            
            # Benchmark if requested
            metrics = {}
            if benchmark:
                metrics = self._benchmark_models(original_model, pruned_model, "unstructured")
                
            return pruned_model, metrics
            
        except Exception as e:
            logger.error(f"Unstructured pruning failed: {e}")
            return original_model, {}

    def apply_structured_pruning(self,
                                 model: Union[nn.Module, BaseModel],
                                 sparsity: float = 0.3,
                                 structured_config: Optional[Dict] = None,
                                 benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Context7-enhanced structured pruning - real speedup gains.
        
        Args:
            model: Prune edilecek model
            sparsity: Pruning ratio (0.0 to 1.0)
            structured_config: Structured pruning konfigürasyonu
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[pruned_model, metrics]
        """
        logger.info(f"Context7-enhanced structured pruning başlatılıyor (sparsity={sparsity})")
        
        original_model = model
        
        # Context7: Try experimental structured pruning first
        if self.enable_experimental:
            try:
                return self._apply_experimental_structured_pruning(
                    model, sparsity, structured_config, benchmark
                )
            except Exception as e:
                logger.warning(f"Experimental structured pruning failed, falling back to standard: {e}")
        
        # Context7: Fallback to standard structured pruning
        return self._apply_standard_structured_pruning(
            model, sparsity, structured_config, benchmark
        )

    def _apply_experimental_structured_pruning(self,
                                               model: Union[nn.Module, BaseModel],
                                               sparsity: float,
                                               structured_config: Optional[Dict],
                                               benchmark: bool) -> Tuple[nn.Module, Dict]:
        """
        Context7: Experimental structured pruning using latest PyTorch APIs
        """
        logger.info("Applying experimental structured pruning...")
        
        original_model = model
        
        try:
            # Context7: Check if model is FX traceable
            try:
                symbolic_trace(model)
                logger.info("Model is FX symbolically traceable")
            except Exception as e:
                logger.warning(f"Model not FX traceable: {e}")
                
            # Default structured config
            if structured_config is None:
                structured_config = {
                    'sparsity_level': sparsity,
                    'sparse_block_shape': (1, 4),  # Context7: 1x4 structured sparsity
                    'zeros_per_block': 2
                }
                
            # Initialize SaliencyPruner
            pruner = SaliencyPruner()
            
            # Prepare model for structured pruning
            pruner.prepare(model, structured_config)
            
            # Enable mask updates and take pruning step
            pruner.enable_mask_update = True
            pruner.step()
            
            # Apply pruning patterns and get pruned model
            pruned_model = pruner.prune()
            
            logger.info("Experimental structured pruning completed")
            
            # Benchmark if requested
            metrics = {}
            if benchmark:
                metrics = self._benchmark_models(original_model, pruned_model, "structured_experimental")
                
            return pruned_model, metrics
            
        except Exception as e:
            logger.error(f"Experimental structured pruning failed: {e}")
            raise

    def _apply_standard_structured_pruning(self,
                                           model: Union[nn.Module, BaseModel],
                                           sparsity: float,
                                           structured_config: Optional[Dict],
                                           benchmark: bool) -> Tuple[nn.Module, Dict]:
        """
        Context7: Standard structured pruning fallback
        """
        logger.info("Applying standard structured pruning...")
        
        original_model = model
        pruned_model = model
        
        try:
            # Apply custom structured pruning to conv and linear layers
            for name, module in pruned_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Prune output channels
                    ContextualStructuredPruning.apply(
                        module, name='weight', amount=sparsity, dim=0
                    )
                elif isinstance(module, nn.Linear):
                    # Prune output features
                    ContextualStructuredPruning.apply(
                        module, name='weight', amount=sparsity, dim=0
                    )
                    
            logger.info("Standard structured pruning completed")
            
            # Benchmark if requested
            metrics = {}
            if benchmark:
                metrics = self._benchmark_models(original_model, pruned_model, "structured_standard")
                
            return pruned_model, metrics
            
        except Exception as e:
            logger.error(f"Standard structured pruning failed: {e}")
            return original_model, {}

    def apply_progressive_pruning(self,
                                  model: Union[nn.Module, BaseModel],
                                  target_sparsity: float = 0.5,
                                  num_steps: int = 5,
                                  validation_fn: Optional[callable] = None,
                                  accuracy_threshold: float = 0.95,
                                  method: str = "structured",
                                  benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Context7-enhanced progressive pruning with validation checkpoints.
        
        Args:
            model: Prune edilecek model
            target_sparsity: Hedef sparsity ratio
            num_steps: Progressive pruning adım sayısı
            validation_fn: Validasyon fonksiyonu
            accuracy_threshold: Minimum accuracy threshold
            method: Pruning method ('structured', 'unstructured')
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[pruned_model, metrics]
        """
        logger.info(f"Context7-enhanced progressive pruning başlatılıyor (target={target_sparsity}, steps={num_steps})")
        
        current_model = model
        current_sparsity = 0.0
        step_size = target_sparsity / num_steps
        
        metrics_history = []
        
        try:
            for step in range(num_steps):
                current_sparsity += step_size
                logger.info(f"Progressive pruning step {step+1}/{num_steps}, sparsity: {current_sparsity:.3f}")
                
                # Apply pruning for this step
                if method == "structured":
                    current_model, step_metrics = self.apply_structured_pruning(
                        current_model, step_size, benchmark=False
                    )
                else:
                    current_model, step_metrics = self.apply_unstructured_pruning(
                        current_model, step_size, benchmark=False
                    )
                
                # Validate accuracy if validation function provided
                if validation_fn:
                    accuracy = validation_fn(current_model)
                    step_metrics['validation_accuracy'] = accuracy
                    
                    if accuracy < accuracy_threshold:
                        logger.warning(f"Accuracy dropped below threshold ({accuracy:.3f} < {accuracy_threshold:.3f})")
                        logger.warning("Stopping progressive pruning")
                        break
                        
                step_metrics['step'] = step + 1
                step_metrics['current_sparsity'] = current_sparsity
                metrics_history.append(step_metrics)
                
            # Final benchmark
            final_metrics = {
                "progressive_pruning_steps": len(metrics_history),
                "final_sparsity": current_sparsity,
                "steps_history": metrics_history
            }
            
            if benchmark:
                bench_metrics = self._benchmark_models(model, current_model, "progressive")
                final_metrics.update(bench_metrics)
                
            logger.info("Progressive pruning completed")
            return current_model, final_metrics
            
        except Exception as e:
            logger.error(f"Progressive pruning failed: {e}")
            return model, {}

    def remove_pruning_reparameterization(self, model: nn.Module) -> nn.Module:
        """
        Context7-enhanced pruning mask removal for final deployment.
        
        Args:
            model: Pruned model with masks
            
        Returns:
            Model with masks permanently applied and removed
        """
        logger.info("Removing pruning reparameterization...")
        
        try:
            for name, module in model.named_modules():
                if prune.is_pruned(module):
                    prune.remove(module, 'weight')
                    if hasattr(module, 'bias') and prune.is_pruned(module):
                        prune.remove(module, 'bias')
                        
            logger.info("Pruning reparameterization removed")
            return model
            
        except Exception as e:
            logger.error(f"Failed to remove pruning reparameterization: {e}")
            return model

    def get_sparsity_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Context7-enhanced sparsity statistics calculation.
        
        Args:
            model: Model to analyze
            
        Returns:
            Detailed sparsity statistics
        """
        try:
            total_params = 0
            zero_params = 0
            layer_stats = {}
            
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if hasattr(module, 'weight'):
                        weight = module.weight
                        layer_total = weight.numel()
                        layer_zeros = (weight == 0).sum().item()
                        
                        total_params += layer_total
                        zero_params += layer_zeros
                        
                        layer_stats[name] = {
                            'total_params': layer_total,
                            'zero_params': layer_zeros,
                            'sparsity': layer_zeros / layer_total if layer_total > 0 else 0.0
                        }
                        
            global_sparsity = zero_params / total_params if total_params > 0 else 0.0
            
            return {
                'global_sparsity': global_sparsity,
                'total_parameters': total_params,
                'zero_parameters': zero_params,
                'layer_statistics': layer_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate sparsity stats: {e}")
            return {}

    def _benchmark_models(self,
                          original_model: nn.Module,
                          pruned_model: nn.Module,
                          method: str) -> Dict[str, Any]:
        """
        Context7-enhanced model benchmarking for pruning.
        
        Args:
            original_model: Orijinal model
            pruned_model: Pruned model
            method: Pruning yöntemi
            
        Returns:
            Benchmark metrikleri
        """
        try:
            # Calculate sparsity statistics
            sparsity_stats = self.get_sparsity_stats(pruned_model)
            
            # Model size comparison
            def get_model_size(model):
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                return param_size
                
            original_size = get_model_size(original_model)
            pruned_size = get_model_size(pruned_model)
            size_reduction = ((original_size - pruned_size) / original_size) * 100
            
            metrics = {
                "pruning_method": method,
                "original_size_bytes": original_size,
                "pruned_size_bytes": pruned_size,
                "size_reduction_percent": size_reduction,
                "compression_ratio": original_size / pruned_size if pruned_size > 0 else 0,
                **sparsity_stats
            }
            
            # Use benchmarker if available
            if self.benchmarker:
                bench_results = self.benchmarker.compare_models(original_model, pruned_model)
                metrics.update(bench_results)
                
            logger.info(f"Model boyutu {size_reduction:.1f}% azaldı, sparsity: {sparsity_stats.get('global_sparsity', 0):.1%} ({method})")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {"error": str(e)}


def create_pruning_pipeline(enable_experimental: bool = True) -> PruningManager:
    """
    Context7-enhanced pruning pipeline oluşturucu.
    
    Args:
        enable_experimental: Experimental structured pruning kullanılsın mı
        
    Returns:
        PruningManager instance
    """
    return PruningManager(enable_experimental=enable_experimental)


# Context7: Export recommended configurations
MOBILE_PRUNING_CONFIG = {
    "method": "structured",
    "sparsity": 0.3,
    "progressive_steps": 5,
    "accuracy_threshold": 0.95
}

AGGRESSIVE_PRUNING_CONFIG = {
    "method": "unstructured",
    "sparsity": 0.8,
    "global_pruning": True,
    "progressive_steps": 10,
    "accuracy_threshold": 0.90
}
