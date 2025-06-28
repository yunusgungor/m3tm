#!/usr/bin/env python3
"""
Mobile Optimization Utilities

Bu modül, PyTorch modellerini mobil cihazlar için optimize etmek üzere
gelişmiş araçlar sağlar. Quantization, pruning, model compression ve
performance optimization özelliklerini içerir.

Features:
- Dynamic/Static Quantization
- Structured/Unstructured Pruning  
- TorchScript Optimization
- ONNX Export
- Mobile Benchmarking
- Model Size Analysis
- Performance Profiling

Author: GitHub Copilot
Date: 28 Haziran 2025
"""

import os
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.quantization
from torch.nn.utils import prune
import torchvision

# Additional optimization libraries
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import numpy as np
import psutil


@dataclass
class MobileOptimizationConfig:
    """Mobil optimizasyon konfigürasyonu."""
    
    # Quantization Configuration
    quantization_enabled: bool = True
    quantization_method: str = "dynamic"  # dynamic, static, qat
    quantization_dtype: str = "qint8"  # qint8, qint32, float16
    calibration_samples: int = 100
    
    # Pruning Configuration
    pruning_enabled: bool = True
    pruning_sparsity: float = 0.3  # 30% sparsity
    pruning_structured: bool = False  # Unstructured pruning
    pruning_global: bool = True  # Global vs layer-wise pruning
    
    # Optimization Configuration
    optimize_for_mobile: bool = True
    backend: str = "qnnpack"  # qnnpack, fbgemm
    
    # Export Configuration
    export_formats: List[str] = field(default_factory=lambda: ["torchscript", "onnx"])
    onnx_opset_version: int = 11
    onnx_optimize: bool = True
    
    # Validation Configuration
    validate_model: bool = True
    run_benchmark: bool = True
    target_platforms: List[str] = field(default_factory=lambda: ["android", "ios"])
    
    # Performance Targets
    max_model_size_mb: float = 500.0  # Maximum model size
    max_inference_time_ms: float = 1000.0  # Maximum inference time
    min_accuracy_retention: float = 0.95  # Minimum accuracy retention


class ModelSizeAnalyzer:
    """Model boyut analizi için yardımcı sınıf."""
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """Model boyutunu analiz eder."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "param_size_bytes": param_size,
            "param_size_mb": param_size / (1024 * 1024),
            "buffer_size_bytes": buffer_size,
            "buffer_size_mb": buffer_size / (1024 * 1024),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    
    @staticmethod
    def compare_models(original_model: nn.Module, 
                      optimized_model: nn.Module) -> Dict[str, Any]:
        """İki modeli karşılaştırır."""
        original_stats = ModelSizeAnalyzer.get_model_size(original_model)
        optimized_stats = ModelSizeAnalyzer.get_model_size(optimized_model)
        
        size_reduction = (
            (original_stats["total_size_mb"] - optimized_stats["total_size_mb"]) /
            original_stats["total_size_mb"] * 100
        )
        
        param_reduction = (
            (original_stats["num_parameters"] - optimized_stats["num_parameters"]) /
            original_stats["num_parameters"] * 100
        )
        
        return {
            "original": original_stats,
            "optimized": optimized_stats,
            "size_reduction_percent": size_reduction,
            "param_reduction_percent": param_reduction,
            "compression_ratio": original_stats["total_size_mb"] / optimized_stats["total_size_mb"]
        }


class QuantizationOptimizer:
    """Model quantization için optimizasyon sınıfı."""
    
    def __init__(self, config: MobileOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Dynamic quantization uygular."""
        self.logger.info("Dynamic quantization uygulanıyor...")
        
        # Quantization dtype
        dtype_map = {
            "qint8": torch.qint8,
            "qint32": torch.qint32,
            "float16": torch.float16,
        }
        
        qconfig_spec = {
            nn.Linear: torch.quantization.default_dynamic_qconfig,
            nn.LSTM: torch.quantization.default_dynamic_qconfig,
            nn.GRU: torch.quantization.default_dynamic_qconfig,
        }
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec,
            dtype=dtype_map.get(self.config.quantization_dtype, torch.qint8)
        )
        
        self.logger.info("Dynamic quantization tamamlandı")
        return quantized_model
    
    def apply_static_quantization(self, 
                                 model: nn.Module,
                                 calibration_data: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """Static quantization uygular."""
        self.logger.info("Static quantization uygulanıyor...")
        
        # Model'i evaluation mode'a al
        model.eval()
        
        # Quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig(self.config.backend)
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(model)
        
        # Calibration
        if calibration_data:
            self.logger.info("Calibration yapılıyor...")
            with torch.no_grad():
                for i, (data, _) in enumerate(calibration_data):
                    if i >= self.config.calibration_samples:
                        break
                    prepared_model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        self.logger.info("Static quantization tamamlandı")
        return quantized_model
    
    def apply_qat(self, 
                  model: nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  num_epochs: int = 3) -> nn.Module:
        """Quantization Aware Training uygular."""
        self.logger.info("Quantization Aware Training başlıyor...")
        
        # Prepare model for QAT
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(self.config.backend)
        prepared_model = torch.quantization.prepare_qat(model)
        
        # Training loop (simplified)
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = prepared_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    self.logger.info(f"QAT Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Convert to quantized model
        prepared_model.eval()
        quantized_model = torch.quantization.convert(prepared_model)
        
        self.logger.info("Quantization Aware Training tamamlandı")
        return quantized_model


class PruningOptimizer:
    """Model pruning için optimizasyon sınıfı."""
    
    def __init__(self, config: MobileOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def apply_unstructured_pruning(self, model: nn.Module) -> nn.Module:
        """Unstructured pruning uygular."""
        self.logger.info(f"Unstructured pruning uygulanıyor (sparsity: {self.config.pruning_sparsity})")
        
        # Pruning yapılacak modülleri belirle
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))
        
        # Global pruning uygula
        if self.config.pruning_global:
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config.pruning_sparsity,
            )
        else:
            # Layer-wise pruning
            for module, param_name in modules_to_prune:
                prune.l1_unstructured(module, name=param_name, amount=self.config.pruning_sparsity)
        
        # Pruning'i kalıcı hale getir
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        self.logger.info("Unstructured pruning tamamlandı")
        return model
    
    def apply_structured_pruning(self, model: nn.Module) -> nn.Module:
        """Structured pruning uygular."""
        self.logger.info(f"Structured pruning uygulanıyor (sparsity: {self.config.pruning_sparsity})")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Linear layer için structured pruning
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=self.config.pruning_sparsity, 
                    n=2, 
                    dim=0
                )
            elif isinstance(module, nn.Conv2d):
                # Conv2d layer için structured pruning
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=self.config.pruning_sparsity, 
                    n=2, 
                    dim=0
                )
        
        # Pruning'i kalıcı hale getir
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.remove(module, 'weight')
        
        self.logger.info("Structured pruning tamamlandı")
        return model
    
    def get_sparsity_stats(self, model: nn.Module) -> Dict[str, float]:
        """Model sparsity istatistiklerini hesaplar."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "sparsity": sparsity,
            "sparsity_percent": sparsity * 100
        }


class MobileExporter:
    """Mobile platform için model export işlemleri."""
    
    def __init__(self, config: MobileOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def export_torchscript(self, 
                          model: nn.Module, 
                          example_input: torch.Tensor,
                          output_path: str) -> str:
        """TorchScript formatında export eder."""
        self.logger.info("TorchScript export başlıyor...")
        
        model.eval()
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Mobile için optimize et
        if self.config.optimize_for_mobile:
            optimized_model = optimize_for_mobile(traced_model)
        else:
            optimized_model = traced_model
        
        # Save
        torchscript_path = f"{output_path}/model.ptl"
        optimized_model._save_for_lite_interpreter(torchscript_path)
        
        self.logger.info(f"TorchScript model kaydedildi: {torchscript_path}")
        return torchscript_path
    
    def export_onnx(self, 
                   model: nn.Module, 
                   example_input: torch.Tensor,
                   output_path: str) -> Optional[str]:
        """ONNX formatında export eder."""
        if not ONNX_AVAILABLE:
            self.logger.warning("ONNX kütüphanesi mevcut değil, export atlanıyor")
            return None
        
        self.logger.info("ONNX export başlıyor...")
        
        model.eval()
        onnx_path = f"{output_path}/model.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Optimize ONNX model
        if self.config.onnx_optimize:
            try:
                import onnxoptimizer
                onnx_model = onnx.load(onnx_path)
                optimized_model = onnxoptimizer.optimize(onnx_model)
                onnx.save(optimized_model, onnx_path)
                self.logger.info("ONNX model optimize edildi")
            except ImportError:
                self.logger.warning("onnxoptimizer mevcut değil, optimization atlanıyor")
        
        self.logger.info(f"ONNX model kaydedildi: {onnx_path}")
        return onnx_path


class MobileBenchmark:
    """Mobile platform için performance benchmark."""
    
    def __init__(self, config: MobileOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def benchmark_inference(self, 
                           model: nn.Module, 
                           example_input: torch.Tensor,
                           num_runs: int = 100) -> Dict[str, float]:
        """Model inference performance'ını benchmark eder."""
        self.logger.info(f"Inference benchmark başlıyor ({num_runs} runs)...")
        
        model.eval()
        device = next(model.parameters()).device
        example_input = example_input.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(example_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        
        results = {
            "total_time_seconds": total_time,
            "average_time_seconds": avg_time,
            "average_time_ms": avg_time * 1000,
            "throughput_fps": 1.0 / avg_time,
            "num_runs": num_runs
        }
        
        self.logger.info(f"Benchmark tamamlandı: {avg_time*1000:.2f}ms per inference")
        return results
    
    def benchmark_memory_usage(self, 
                              model: nn.Module, 
                              example_input: torch.Tensor) -> Dict[str, float]:
        """Memory usage benchmark eder."""
        self.logger.info("Memory benchmark başlıyor...")
        
        device = next(model.parameters()).device
        example_input = example_input.to(device)
        
        # GPU memory tracking
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Before inference
            memory_before = torch.cuda.memory_allocated()
            
            # Inference
            with torch.no_grad():
                output = model(example_input)
            
            # After inference
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            results = {
                "memory_before_mb": memory_before / (1024 * 1024),
                "memory_after_mb": memory_after / (1024 * 1024),
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "memory_delta_mb": (memory_after - memory_before) / (1024 * 1024)
            }
        else:
            # CPU memory tracking (approximation)
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            with torch.no_grad():
                output = model(example_input)
            
            memory_after = process.memory_info().rss
            
            results = {
                "memory_before_mb": memory_before / (1024 * 1024),
                "memory_after_mb": memory_after / (1024 * 1024),
                "memory_delta_mb": (memory_after - memory_before) / (1024 * 1024)
            }
        
        self.logger.info(f"Memory benchmark tamamlandı")
        return results


class MobileOptimizer:
    """Ana mobile optimization sınıfı."""
    
    def __init__(self, config: MobileOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sub-optimizers
        self.quantizer = QuantizationOptimizer(config)
        self.pruner = PruningOptimizer(config)
        self.exporter = MobileExporter(config)
        self.benchmark = MobileBenchmark(config)
        
        self.logger.info("MobileOptimizer initialized")
    
    def optimize_model(self, 
                      model: nn.Module,
                      example_input: torch.Tensor,
                      output_dir: str,
                      calibration_data: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, Any]:
        """Modeli mobil platform için kapsamlı optimize eder."""
        self.logger.info("🚀 Mobile optimization başlıyor...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {
            "original_model_stats": ModelSizeAnalyzer.get_model_size(model),
            "optimizations_applied": [],
            "export_paths": {},
            "benchmarks": {}
        }
        
        # Original model benchmark
        self.logger.info("Original model benchmark...")
        results["benchmarks"]["original"] = {
            "inference": self.benchmark.benchmark_inference(model, example_input),
            "memory": self.benchmark.benchmark_memory_usage(model, example_input)
        }
        
        optimized_model = model
        
        # 1. Pruning
        if self.config.pruning_enabled:
            self.logger.info("Pruning uygulanıyor...")
            if self.config.pruning_structured:
                optimized_model = self.pruner.apply_structured_pruning(optimized_model)
            else:
                optimized_model = self.pruner.apply_unstructured_pruning(optimized_model)
            
            results["optimizations_applied"].append("pruning")
            results["pruning_stats"] = self.pruner.get_sparsity_stats(optimized_model)
        
        # 2. Quantization
        if self.config.quantization_enabled:
            self.logger.info("Quantization uygulanıyor...")
            if self.config.quantization_method == "dynamic":
                optimized_model = self.quantizer.apply_dynamic_quantization(optimized_model)
            elif self.config.quantization_method == "static":
                optimized_model = self.quantizer.apply_static_quantization(optimized_model, calibration_data)
            
            results["optimizations_applied"].append("quantization")
        
        # Model comparison
        results["model_comparison"] = ModelSizeAnalyzer.compare_models(model, optimized_model)
        
        # Optimized model benchmark
        self.logger.info("Optimized model benchmark...")
        results["benchmarks"]["optimized"] = {
            "inference": self.benchmark.benchmark_inference(optimized_model, example_input),
            "memory": self.benchmark.benchmark_memory_usage(optimized_model, example_input)
        }
        
        # 3. Export
        self.logger.info("Model export işlemleri...")
        if "torchscript" in self.config.export_formats:
            torchscript_path = self.exporter.export_torchscript(
                optimized_model, example_input, output_dir
            )
            results["export_paths"]["torchscript"] = torchscript_path
        
        if "onnx" in self.config.export_formats:
            onnx_path = self.exporter.export_onnx(
                optimized_model, example_input, output_dir
            )
            if onnx_path:
                results["export_paths"]["onnx"] = onnx_path
        
        # Save optimized PyTorch model
        pytorch_path = f"{output_dir}/optimized_model.pth"
        torch.save(optimized_model.state_dict(), pytorch_path)
        results["export_paths"]["pytorch"] = pytorch_path
        
        # 4. Validation
        if self.config.validate_model:
            validation_results = self._validate_optimization(model, optimized_model, example_input)
            results["validation"] = validation_results
        
        # Save results
        results_path = f"{output_dir}/optimization_results.json"
        with open(results_path, 'w') as f:
            # Convert torch tensors to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info("✅ Mobile optimization tamamlandı!")
        return results
    
    def _validate_optimization(self, 
                              original_model: nn.Module,
                              optimized_model: nn.Module, 
                              example_input: torch.Tensor) -> Dict[str, Any]:
        """Optimization sonuçlarını validate eder."""
        self.logger.info("Optimization validation...")
        
        original_model.eval()
        optimized_model.eval()
        
        validation_results = {
            "passed": True,
            "issues": [],
            "metrics": {}
        }
        
        try:
            # Output comparison
            with torch.no_grad():
                original_output = original_model(example_input)
                optimized_output = optimized_model(example_input)
            
            # Calculate differences
            if isinstance(original_output, torch.Tensor) and isinstance(optimized_output, torch.Tensor):
                mse = F.mse_loss(original_output, optimized_output).item()
                mae = F.l1_loss(original_output, optimized_output).item()
                max_diff = torch.max(torch.abs(original_output - optimized_output)).item()
                
                validation_results["metrics"] = {
                    "mse": mse,
                    "mae": mae,
                    "max_absolute_difference": max_diff
                }
                
                # Check thresholds
                if mse > 0.1:
                    validation_results["issues"].append(f"High MSE: {mse}")
                if max_diff > 1.0:
                    validation_results["issues"].append(f"High max difference: {max_diff}")
            
            # Size validation
            original_size = ModelSizeAnalyzer.get_model_size(original_model)["total_size_mb"]
            optimized_size = ModelSizeAnalyzer.get_model_size(optimized_model)["total_size_mb"]
            
            if optimized_size > self.config.max_model_size_mb:
                validation_results["issues"].append(
                    f"Model size {optimized_size:.1f}MB exceeds limit {self.config.max_model_size_mb}MB"
                )
            
            # Performance validation
            benchmark_results = self.benchmark.benchmark_inference(optimized_model, example_input, num_runs=10)
            if benchmark_results["average_time_ms"] > self.config.max_inference_time_ms:
                validation_results["issues"].append(
                    f"Inference time {benchmark_results['average_time_ms']:.1f}ms exceeds limit {self.config.max_inference_time_ms}ms"
                )
            
            validation_results["passed"] = len(validation_results["issues"]) == 0
            
        except Exception as e:
            validation_results["passed"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def _make_json_serializable(self, obj):
        """Objeyi JSON serializable hale getirir."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.numel() <= 100 else f"Tensor{list(obj.shape)}"
        elif isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size <= 100 else f"Array{list(obj.shape)}"
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


def optimize_model_for_mobile(model: nn.Module,
                             example_input: torch.Tensor,
                             output_dir: str,
                             config: Optional[MobileOptimizationConfig] = None) -> Dict[str, Any]:
    """
    Model'i mobil platform için optimize eden convenience function.
    
    Args:
        model: Optimize edilecek model
        example_input: Örnek input tensor
        output_dir: Çıktı dizini
        config: Optimization konfigürasyonu
        
    Returns:
        Optimization sonuçları
    """
    if config is None:
        config = MobileOptimizationConfig()
    
    optimizer = MobileOptimizer(config)
    return optimizer.optimize_model(model, example_input, output_dir)


def create_example_input(model_type: str = "transformer", 
                        batch_size: int = 1,
                        seq_length: int = 128,
                        vocab_size: int = 50000) -> torch.Tensor:
    """
    Farklı model türleri için örnek input oluşturur.
    
    Args:
        model_type: Model türü (transformer, cnn, etc.)
        batch_size: Batch boyutu
        seq_length: Sequence uzunluğu
        vocab_size: Vocabulary boyutu
        
    Returns:
        Örnek input tensor
    """
    if model_type == "transformer":
        return torch.randint(0, vocab_size, (batch_size, seq_length))
    elif model_type == "cnn":
        return torch.randn(batch_size, 3, 224, 224)
    elif model_type == "audio":
        return torch.randn(batch_size, 1, 16000)  # 1 second at 16kHz
    else:
        return torch.randn(batch_size, 128)  # Generic input


if __name__ == "__main__":
    # Test mobile optimization
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Test optimization
    model = SimpleModel()
    example_input = torch.randn(1, 128)
    output_dir = "./test_mobile_optimization"
    
    config = MobileOptimizationConfig(
        quantization_enabled=True,
        pruning_enabled=True,
        pruning_sparsity=0.2
    )
    
    results = optimize_model_for_mobile(model, example_input, output_dir, config)
    
    print("Mobile Optimization Test Results:")
    print(f"Size reduction: {results['model_comparison']['size_reduction_percent']:.1f}%")
    print(f"Compression ratio: {results['model_comparison']['compression_ratio']:.2f}x")
    print(f"Validation passed: {results['validation']['passed']}")
    print(f"Export paths: {list(results['export_paths'].keys())}")
