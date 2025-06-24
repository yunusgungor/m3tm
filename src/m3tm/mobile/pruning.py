"""
M³TM Model Pruning Modülü

Bu modül PyTorch'un pruning API'sini kullanarak model parametrelerini
azaltmayı ve model boyutunu küçültmeyi amaçlar.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker

logger = logging.getLogger(__name__)


class StructuredChannelPruning(prune.BasePruningMethod):
    """
    Structured channel pruning implementation.
    """
    PRUNING_TYPE = 'structured'

    def __init__(self, amount):
        self.amount = amount

    def compute_mask(self, tensor, default_mask):
        """
        Structured pruning mask hesaplar.

        Args:
            tensor: Prune edilecek weight tensor
            default_mask: Varsayılan mask

        Returns:
            Computed pruning mask
        """
        # Channel-wise structured pruning
        if tensor.dim() == 4:  # Conv weight: [out_channels, in_channels, H, W]
            # Prune output channels
            channel_norms = torch.norm(
                tensor.view(tensor.size(0), -1), dim=1)
            num_prune = int(self.amount * tensor.size(0))
            if num_prune > 0:
                _, indices = torch.topk(
                    channel_norms, num_prune, largest=False)
                mask = torch.ones_like(tensor)
                mask[indices] = 0
                return mask
        elif tensor.dim() == 2:  # Linear: [out_features, in_features]
            # Prune output neurons
            neuron_norms = torch.norm(tensor, dim=1)
            num_prune = int(self.amount * tensor.size(0))
            if num_prune > 0:
                _, indices = torch.topk(neuron_norms, num_prune, largest=False)
                mask = torch.ones_like(tensor)
                mask[indices] = 0
                return mask

        return default_mask

    @classmethod
    def apply(cls, module, name, amount):
        """
        Modüle structured pruning uygular.

        Args:
            module: Prune edilecek modül
            name: Parameter adı
            amount: Prune miktarı

        Returns:
            StructuredChannelPruning instance
        """
        return super(StructuredChannelPruning, cls).apply(
            module, name, amount=amount
        )


class PruningManager:
    """
    M³TM Model Pruning Manager

    Bu sınıf PyTorch pruning API'sini kullanarak model boyutunu
    azaltmaya odaklanır. Context7 uyumlu best practices ile
    geliştirilmiştir.
    """

    def __init__(self,
                 pruning_type: str = "unstructured",
                 amount: float = 0.3,
                 benchmarker: Optional[ModelBenchmarker] = None):
        """
        PruningManager initialization.

        Args:
            pruning_type: 'structured' veya 'unstructured'
            amount: Prune edilecek parametre yüzdesi (0.0-1.0)
            benchmarker: Performans ölçüm aracı
        """
        self.pruning_type = pruning_type
        self.amount = amount
        self.benchmarker = benchmarker or ModelBenchmarker()

        logger.info(
            f"PruningManager initialized - Type: {pruning_type}, "
            f"Amount: {amount}"
        )

    def apply_unstructured_pruning(
            self,
            model: Union[nn.Module, BaseModel],
            modules_to_prune: Optional[List] = None,
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

        original_metrics = None
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)

        try:
            # Determine modules to prune
            if modules_to_prune is None:
                modules_to_prune = self._get_default_modules_to_prune(model)

            # Apply L1 unstructured pruning
            for module, parameter_name in modules_to_prune:
                prune.l1_unstructured(
                    module, name=parameter_name, amount=self.amount)

            # Calculate sparsity
            sparsity_info = self._calculate_sparsity(model, modules_to_prune)

            # Benchmark pruned model
            pruned_metrics = None
            compression_metrics = {}

            if benchmark and original_metrics:
                pruned_metrics = self.benchmarker.get_model_metrics(model)
                compression_metrics = self._calculate_pruning_metrics(
                    original_metrics, pruned_metrics, sparsity_info)

                logger.info(
                    f"Unstructured pruning tamamlandı - "
                    f"Sparsity: {sparsity_info.get('global_sparsity', 0):.1f}%"
                )

            return model, {
                'original_metrics': original_metrics,
                'pruned_metrics': pruned_metrics,
                'compression_metrics': compression_metrics,
                'sparsity_info': sparsity_info,
                'pruning_type': 'unstructured'
            }

        except Exception as e:
            logger.error(f"Unstructured pruning hatası: {str(e)}")
            return model, {'error': str(e)}

    def apply_structured_pruning(
            self,
            model: Union[nn.Module, BaseModel],
            modules_to_prune: Optional[List] = None,
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

        original_metrics = None
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)

        try:
            # Determine modules to prune
            if modules_to_prune is None:
                modules_to_prune = self._get_default_modules_to_prune(model)

            # Apply structured pruning
            for module, parameter_name in modules_to_prune:
                StructuredChannelPruning.apply(
                    module,
                    parameter_name,
                    amount=self.amount
                )

            # Calculate sparsity
            sparsity_info = self._calculate_sparsity(model, modules_to_prune)

            # Benchmark pruned model
            pruned_metrics = None
            compression_metrics = {}

            if benchmark and original_metrics:
                pruned_metrics = self.benchmarker.get_model_metrics(model)
                compression_metrics = self._calculate_pruning_metrics(
                    original_metrics, pruned_metrics, sparsity_info)

                logger.info(
                    f"Structured pruning tamamlandı - "
                    f"Sparsity: {sparsity_info.get('global_sparsity', 0):.1f}%"
                )

            return model, {
                'original_metrics': original_metrics,
                'pruned_metrics': pruned_metrics,
                'compression_metrics': compression_metrics,
                'sparsity_info': sparsity_info,
                'pruning_type': 'structured'
            }

        except Exception as e:
            logger.error(f"Structured pruning hatası: {str(e)}")
            return model, {'error': str(e)}

    def apply_progressive_pruning(
            self,
            model: Union[nn.Module, BaseModel],
            target_sparsity: float = 0.8,
            num_steps: int = 5,
            fine_tune_callback=None,
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
        logger.info("Progressive pruning başlatılıyor...")

        original_metrics = None
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)

        step_metrics = []
        modules_to_prune = self._get_default_modules_to_prune(model)

        try:
            for step in range(num_steps):
                current_target = (step + 1) * target_sparsity / num_steps

                logger.info(
                    f"Pruning step {step + 1}/{num_steps} "
                    f"(target sparsity: {current_target:.1f}%)"
                )

                # Apply incremental pruning
                step_amount = target_sparsity / num_steps
                for module, parameter_name in modules_to_prune:
                    if self.pruning_type == "structured":
                        StructuredChannelPruning.apply(
                            module, parameter_name, step_amount)
                    else:
                        prune.l1_unstructured(
                            module, name=parameter_name, amount=step_amount)

                # Fine-tuning if provided
                if fine_tune_callback:
                    fine_tune_callback(model, step)

                # Measure current state
                current_sparsity = self._calculate_sparsity(
                    model, modules_to_prune)
                if benchmark:
                    current_metrics = self.benchmarker.get_model_metrics(model)
                    step_metrics.append({
                        'step': step + 1,
                        'sparsity': current_sparsity,
                        'metrics': current_metrics
                    })

            # Final measurements
            final_sparsity_info = self._calculate_sparsity(
                model, modules_to_prune)
            final_metrics = None
            compression_metrics = {}

            if benchmark and original_metrics:
                final_metrics = self.benchmarker.get_model_metrics(model)
                compression_metrics = self._calculate_pruning_metrics(
                    original_metrics, final_metrics, final_sparsity_info)

                logger.info(
                    f"Progressive pruning tamamlandı - "
                    f"Final sparsity: "
                    f"{final_sparsity_info.get('global_sparsity', 0):.1f}%"
                )

            return model, {
                'original_metrics': original_metrics,
                'final_metrics': final_metrics,
                'compression_metrics': compression_metrics,
                'sparsity_info': final_sparsity_info,
                'step_metrics': step_metrics,
                'pruning_type': f'progressive_{self.pruning_type}'
            }

        except Exception as e:
            logger.error(f"Progressive pruning hatası: {str(e)}")
            return model, {'error': str(e)}

    def remove_pruning_masks(self, model: nn.Module) -> nn.Module:
        """
        Pruning mask'lerini kaldırır ve sparse parametreleri kalıcı hale
        getirir.

        Args:
            model: Mask'leri kaldırılacak model

        Returns:
            Permanent pruned model
        """
        logger.info("Pruning mask'leri kaldırılıyor...")

        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')

        logger.info("Pruning mask'leri kaldırıldı")
        return model

    def _get_default_modules_to_prune(self, model: nn.Module) -> List:
        """
        Varsayılan olarak prune edilecek modülleri belirler.

        Args:
            model: Analiz edilecek model

        Returns:
            (module, parameter_name) tuple'larının listesi
        """
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))

        return modules_to_prune

    def _calculate_sparsity(self, model: nn.Module,
                            modules_to_prune: List) -> Dict:
        """
        Model sparsity'sini hesaplar.

        Args:
            model: Sparsity hesaplanacak model
            modules_to_prune: Prune edilmiş modüller

        Returns:
            Sparsity information dict
        """
        total_params = 0
        zero_params = 0

        for module, parameter_name in modules_to_prune:
            param = getattr(module, parameter_name)
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        global_sparsity = (
            (zero_params / total_params * 100) if total_params > 0 else 0
        )

        return {
            'global_sparsity': global_sparsity,
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'remaining_parameters': total_params - zero_params
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
        metrics = {}

        # Parameter reduction
        if ('param_count' in original_metrics and
                'param_count' in pruned_metrics):
            original_params = original_metrics['param_count']
            pruned_params = pruned_metrics['param_count']
            if original_params > 0:
                param_reduction = (
                    (original_params - pruned_params) / original_params
                ) * 100
                metrics['param_reduction_percent'] = param_reduction

        # Add sparsity info
        metrics.update(sparsity_info)

        return metrics

    def save_pruned_model(self, model: nn.Module, save_path: Union[str, Path],
                          metadata: Optional[Dict] = None,
                          remove_masks: bool = True):
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

        if remove_masks:
            model = self.remove_pruning_masks(model)

        # Save model
        torch.save(model.state_dict(), save_path)

        # Save metadata
        if metadata:
            metadata_path = save_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Pruned model kaydedildi: {save_path}")


def create_pruning_pipeline(pruning_type: str = "unstructured",
                            amount: float = 0.3,
                            pruning_method: str = "magnitude",
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
    return PruningManager(
        pruning_type=pruning_type,
        amount=amount
    )
