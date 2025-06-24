"""
Model Benchmark ve Performans Ölçüm Araçları

Bu modül model boyutu, inference hızı, hafıza kullanımı ve 
doğruluk metriklerini ölçmek için araçlar sağlar.
"""

import time
import logging
import psutil
import os
from typing import Dict, Optional, Union, Any, Tuple, List, Callable
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.utils.benchmark as benchmark


logger = logging.getLogger(__name__)


class ModelBenchmarker:
    """Model performans ölçüm sınıfı."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: Benchmark yapılacak cihaz
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_model_metrics(self, model: nn.Module, 
                          example_inputs: Optional[torch.Tensor] = None) -> Dict:
        """
        Model hakkında temel metrikleri hesaplar.
        
        Args:
            model: Analiz edilecek model
            example_inputs: Örnek girdi (inference metrics için gerekli)
            
        Returns:
            Model metrics dictionary
        """
        metrics = {}
        
        try:
            # Parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            metrics.update({
                'param_count': total_params,
                'trainable_params': trainable_params,
                'non_trainable_params': total_params - trainable_params
            })
            
            # Model size estimation (FP32 bytes)
            model_size_bytes = total_params * 4  # 4 bytes per float32
            model_size_mb = model_size_bytes / (1024 * 1024)
            
            metrics.update({
                'model_size_bytes': model_size_bytes,
                'model_size_mb': model_size_mb
            })
            
            # Model structure info
            layer_count = len(list(model.modules()))
            conv_layers = len([m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))])
            linear_layers = len([m for m in model.modules() if isinstance(m, nn.Linear)])
            
            metrics.update({
                'total_layers': layer_count,
                'conv_layers': conv_layers,
                'linear_layers': linear_layers
            })
            
            # Inference metrics (eğer example_inputs varsa)
            if example_inputs is not None:
                inference_metrics = self.measure_inference_performance(model, example_inputs)
                metrics.update(inference_metrics)
                
        except Exception as e:
            logger.warning(f"Model metrics hesaplanırken hata: {e}")
            
        return metrics
    
    def measure_inference_performance(self, model: nn.Module, 
                                      example_inputs: torch.Tensor,
                                      num_warmup: int = 10,
                                      num_iterations: int = 100) -> Dict:
        """
        Model inference performansını ölçer.
        
        Args:
            model: Test edilecek model
            example_inputs: Örnek girdi tensörü
            num_warmup: Warmup iterasyon sayısı  
            num_iterations: Ölçüm iterasyon sayısı
            
        Returns:
            Inference performance metrics
        """
        model.eval()
        model = model.to(self.device)
        example_inputs = example_inputs.to(self.device)
        
        metrics = {}
        
        try:
            with torch.no_grad():
                # Warmup
                for _ in range(num_warmup):
                    _ = model(example_inputs)
                
                # Synchronize for accurate timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Time measurement
                start_time = time.perf_counter()
                
                for _ in range(num_iterations):
                    _ = model(example_inputs)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                end_time = time.perf_counter()
                
                # Calculate metrics
                total_time = end_time - start_time
                avg_inference_time = total_time / num_iterations
                throughput = 1.0 / avg_inference_time  # inferences per second
                
                metrics.update({
                    'avg_inference_time_ms': avg_inference_time * 1000,
                    'throughput_fps': throughput,
                    'total_benchmark_time_s': total_time,
                    'num_iterations': num_iterations
                })
                
        except Exception as e:
            logger.warning(f"Inference performance ölçümünde hata: {e}")
            
        return metrics
    
    def measure_memory_usage(self, model: nn.Module, 
                             example_inputs: torch.Tensor) -> Dict:
        """
        Model hafıza kullanımını ölçer.
        
        Args:
            model: Test edilecek model
            example_inputs: Örnek girdi tensörü
            
        Returns:
            Memory usage metrics
        """
        model.eval()
        model = model.to(self.device)
        example_inputs = example_inputs.to(self.device)
        
        metrics = {}
        
        try:
            # System memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                gpu_memory_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
            # Model forward pass
            with torch.no_grad():
                output = model(example_inputs)
                
            # System memory after
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_delta = memory_after - memory_before
            
            metrics.update({
                'system_memory_before_mb': memory_before,
                'system_memory_after_mb': memory_after,
                'system_memory_delta_mb': memory_delta
            })
            
            if self.device.type == 'cuda':
                gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                gpu_memory_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                
                metrics.update({
                    'gpu_memory_before_mb': gpu_memory_before,
                    'gpu_memory_after_mb': gpu_memory_after,
                    'gpu_memory_peak_mb': gpu_memory_peak,
                    'gpu_memory_delta_mb': gpu_memory_after - gpu_memory_before
                })
                
        except Exception as e:
            logger.warning(f"Memory usage ölçümünde hata: {e}")
            
        return metrics
    
    def profile_model(self, model: nn.Module, 
                      example_inputs: torch.Tensor,
                      output_path: Optional[str] = None) -> Dict:
        """
        PyTorch Profiler kullanarak detaylı model profiling yapar.
        
        Args:
            model: Profile edilecek model
            example_inputs: Örnek girdi tensörü
            output_path: Profil çıktısının kaydedileceği dosya yolu
            
        Returns:
            Profiling summary metrics
        """
        model.eval()
        model = model.to(self.device)
        example_inputs = example_inputs.to(self.device)
        
        metrics = {}
        
        try:
            activities = [ProfilerActivity.CPU]
            if self.device.type == 'cuda':
                activities.append(ProfilerActivity.CUDA)
            
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    with torch.no_grad():
                        _ = model(example_inputs)
            
            # Profiling summary
            cpu_time_total = prof.key_averages().total_average().cpu_time_total
            cuda_time_total = prof.key_averages().total_average().cuda_time_total if self.device.type == 'cuda' else 0
            
            metrics.update({
                'profile_cpu_time_ms': cpu_time_total / 1000,  # microseconds to milliseconds
                'profile_cuda_time_ms': cuda_time_total / 1000,
                'profile_available': True
            })
            
            # Save profiler output if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(output_path))
                logger.info(f"Profiling çıktısı kaydedildi: {output_path}")
                
        except Exception as e:
            logger.warning(f"Model profiling'de hata: {e}")
            metrics['profile_available'] = False
            
        return metrics
    
    def compare_models(self, original_model: nn.Module, 
                       optimized_model: nn.Module,
                       example_inputs: torch.Tensor) -> Dict:
        """
        İki modeli karşılaştırır (orijinal vs optimize edilmiş).
        
        Args:
            original_model: Orijinal model
            optimized_model: Optimize edilmiş model
            example_inputs: Örnek girdi tensörü
            
        Returns:
            Comparison metrics
        """
        logger.info("Model karşılaştırması başlatılıyor...")
        
        # Her iki model için metrics al
        original_metrics = self.get_model_metrics(original_model, example_inputs)
        optimized_metrics = self.get_model_metrics(optimized_model, example_inputs)
        
        # Comparison metrics hesapla
        comparison = {}
        
        # Size comparison
        if 'model_size_mb' in original_metrics and 'model_size_mb' in optimized_metrics:
            original_size = original_metrics['model_size_mb']
            optimized_size = optimized_metrics['model_size_mb']
            
            if original_size > 0:
                size_reduction = (original_size - optimized_size) / original_size * 100
                compression_ratio = original_size / optimized_size if optimized_size > 0 else float('inf')
                
                comparison.update({
                    'size_reduction_percent': size_reduction,
                    'compression_ratio': compression_ratio
                })
        
        # Speed comparison  
        if 'avg_inference_time_ms' in original_metrics and 'avg_inference_time_ms' in optimized_metrics:
            original_time = original_metrics['avg_inference_time_ms']
            optimized_time = optimized_metrics['avg_inference_time_ms']
            
            if original_time > 0:
                speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
                time_reduction = (original_time - optimized_time) / original_time * 100
                
                comparison.update({
                    'speedup_ratio': speedup,
                    'time_reduction_percent': time_reduction
                })
        
        # Parameter count comparison
        if 'param_count' in original_metrics and 'param_count' in optimized_metrics:
            original_params = original_metrics['param_count']
            optimized_params = optimized_metrics['param_count']
            
            if original_params > 0:
                param_reduction = (original_params - optimized_params) / original_params * 100
                
                comparison.update({
                    'param_reduction_percent': param_reduction
                })
        
        return {
            'original_metrics': original_metrics,
            'optimized_metrics': optimized_metrics,
            'comparison': comparison
        }
    
    def create_benchmark_report(self, results: Dict, 
                                output_path: Optional[str] = None) -> str:
        """
        Benchmark sonuçlarından detaylı rapor oluşturur.
        
        Args:
            results: Benchmark sonuçları dictionary
            output_path: Rapor dosyasının kaydedileceği yol
            
        Returns:
            Rapor string içeriği
        """
        report_lines = [
            "# Model Optimization Benchmark Report",
            "=" * 50,
            ""
        ]
        
        # Original metrics
        if 'original_metrics' in results:
            report_lines.extend([
                "## Original Model Metrics",
                "-" * 30,
                ""
            ])
            
            orig_metrics = results['original_metrics']
            report_lines.append(f"Model Size: {orig_metrics.get('model_size_mb', 'N/A'):.2f} MB")
            report_lines.append(f"Parameter Count: {orig_metrics.get('param_count', 'N/A'):,}")
            report_lines.append(f"Inference Time: {orig_metrics.get('avg_inference_time_ms', 'N/A'):.2f} ms")
            report_lines.append("")
        
        # Optimized metrics
        if 'optimized_metrics' in results:
            report_lines.extend([
                "## Optimized Model Metrics", 
                "-" * 30,
                ""
            ])
            
            opt_metrics = results['optimized_metrics']
            report_lines.append(f"Model Size: {opt_metrics.get('model_size_mb', 'N/A'):.2f} MB")
            report_lines.append(f"Parameter Count: {opt_metrics.get('param_count', 'N/A'):,}")
            report_lines.append(f"Inference Time: {opt_metrics.get('avg_inference_time_ms', 'N/A'):.2f} ms")
            report_lines.append("")
        
        # Comparison
        if 'comparison' in results:
            report_lines.extend([
                "## Optimization Results",
                "-" * 30,
                ""
            ])
            
            comparison = results['comparison']
            report_lines.append(f"Size Reduction: {comparison.get('size_reduction_percent', 'N/A'):.1f}%")
            report_lines.append(f"Compression Ratio: {comparison.get('compression_ratio', 'N/A'):.2f}x")
            report_lines.append(f"Speed Improvement: {comparison.get('speedup_ratio', 'N/A'):.2f}x")
            report_lines.append(f"Parameter Reduction: {comparison.get('param_reduction_percent', 'N/A'):.1f}%")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"Benchmark raporu kaydedildi: {output_path}")
        
        return report_content


class PerformanceTracker:
    """Performans tracking sınıfı."""
    
    def __init__(self):
        self.results_history = []
        
    def add_result(self, model_name: str, metrics: Dict) -> None:
        """
        Yeni benchmark sonucu ekler.
        
        Args:
            model_name: Model adı
            metrics: Metrics dictionary
        """
        result_entry = {
            'timestamp': time.time(),
            'model_name': model_name,
            'metrics': metrics
        }
        
        self.results_history.append(result_entry)
        
    def get_performance_trends(self) -> Dict:
        """
        Performans trendlerini analiz eder.
        
        Returns:
            Trend analysis results
        """
        if len(self.results_history) < 2:
            return {'trend_available': False}
            
        # Simple trend analysis implementation
        size_trend = []
        speed_trend = []
        
        for result in self.results_history:
            metrics = result.get('metrics', {})
            if 'model_size_mb' in metrics:
                size_trend.append(metrics['model_size_mb'])
            if 'avg_inference_time_ms' in metrics:
                speed_trend.append(metrics['avg_inference_time_ms'])
        
        trends = {'trend_available': True}
        
        if len(size_trend) > 1:
            size_improvement = (size_trend[0] - size_trend[-1]) / size_trend[0] * 100
            trends['size_trend_percent'] = size_improvement
            
        if len(speed_trend) > 1:
            speed_improvement = (speed_trend[0] - speed_trend[-1]) / speed_trend[0] * 100
            trends['speed_trend_percent'] = speed_improvement
            
        return trends


def create_benchmarker(device: Optional[str] = None) -> ModelBenchmarker:
    """
    ModelBenchmarker instance oluşturucu fonksiyon.
    
    Args:
        device: Benchmark cihazı ('cpu', 'cuda', vs.)
        
    Returns:
        ModelBenchmarker instance
    """
    if device:
        torch_device = torch.device(device)
    else:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    return ModelBenchmarker(torch_device)
