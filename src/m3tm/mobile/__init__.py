"""
M³TM v2.3 - PyTorch Mobile entegrasyonu

Mobil platformlarda model çalıştırma, optimizasyon ve platform özgü işlemler için modüller.
Gelişmiş quantization, pruning, knowledge distillation ve TorchScript optimizasyonları içerir.
"""

from .optimization_pipeline import OptimizationPipeline, create_optimization_pipeline
from .quantization import QuantizationManager, create_quantization_pipeline
from .pruning import PruningManager, create_pruning_pipeline
from .knowledge_distillation import DistillationTrainer, create_distillation_pipeline
from .torchscript_converter import TorchScriptConverter, create_torchscript_pipeline
from .benchmark_utils import ModelBenchmarker, create_benchmarker

# Legacy imports
from .optimization import MobileOptimizer, optimize_model_for_mobile
from .model_converter import ModelConverter
from .benchmark import benchmark_model

__all__ = [
    # New optimized modules
    "OptimizationPipeline", "create_optimization_pipeline",
    "QuantizationManager", "create_quantization_pipeline",
    "PruningManager", "create_pruning_pipeline",
    "DistillationTrainer", "create_distillation_pipeline",
    "TorchScriptConverter", "create_torchscript_pipeline",
    "ModelBenchmarker", "create_benchmarker",

    # Legacy modules
    "MobileOptimizer", "optimize_model_for_mobile",
    "ModelConverter", "benchmark_model"
]