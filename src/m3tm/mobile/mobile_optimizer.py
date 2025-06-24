"""
M³TM Advanced Mobile Optimization Engine
Enhanced with Context7 PyTorch Mobile Best Practices

Implements QNNPACK quantization, FX graph mode optimization,
hardware-specific acceleration, and intelligent caching.
"""

import torch
import torch.ao.quantization as quantization
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import platform
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardwareDetector:
    """Detect and configure optimal hardware backend"""
    
    @staticmethod
    def detect_platform() -> str:
        """Detect current platform"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'android':
            return 'android'
        elif system == 'darwin':
            if 'arm' in machine:
                return 'ios_arm64'
            else:
                return 'macos_x86'
        elif 'arm' in machine or 'aarch64' in machine:
            return 'linux_arm64'
        else:
            return 'linux_x86'
    
    @staticmethod
    def get_optimal_backend() -> str:
        """Get optimal quantization backend for current platform"""
        platform_type = HardwareDetector.detect_platform()
        
        backend_map = {
            'android': 'qnnpack',
            'ios_arm64': 'qnnpack', 
            'linux_arm64': 'qnnpack',
            'linux_x86': 'x86',
            'macos_x86': 'x86'
        }
        
        return backend_map.get(platform_type, 'qnnpack')
    
    @staticmethod
    def configure_backends():
        """Configure PyTorch backends for optimal performance"""
        backend = HardwareDetector.get_optimal_backend()
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Configure additional optimizations
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Enable MKL optimizations if available
        if hasattr(torch.backends, 'mkl'):
            torch.backends.mkl.enabled = True
        
        logger.info(f"Configured backend: {backend}")
        return backend

class QuantizationOptimizer:
    """Advanced quantization optimizer with Context7 best practices"""
    
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or HardwareDetector.get_optimal_backend()
        HardwareDetector.configure_backends()
        
    def prepare_model_for_quantization(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization with proper fusing"""
        model.eval()
        
        # Add QuantStub and DeQuantStub if not present
        if not hasattr(model, 'quant'):
            model.quant = quantization.QuantStub()
        if not hasattr(model, 'dequant'):
            model.dequant = quantization.DeQuantStub()
        
        return model
    
    def fuse_modules(self, model: nn.Module, fusion_list: List[List[str]]) -> nn.Module:
        """Fuse modules for better quantization performance"""
        try:
            fused_model = quantization.fuse_modules(model, fusion_list)
            logger.info(f"Successfully fused modules: {fusion_list}")
            return fused_model
        except Exception as e:
            logger.warning(f"Module fusion failed: {e}")
            return model
    
    def apply_post_training_static_quantization(
        self, 
        model: nn.Module, 
        calibration_loader: torch.utils.data.DataLoader,
        fusion_list: Optional[List[List[str]]] = None
    ) -> nn.Module:
        """Apply Post-Training Static Quantization with QNNPACK/x86 backend"""
        
        # Prepare model
        model = self.prepare_model_for_quantization(model)
        
        # Apply module fusing if specified
        if fusion_list:
            model = self.fuse_modules(model, fusion_list)
        
        # Set qconfig
        qconfig = quantization.get_default_qconfig(self.backend)
        model.qconfig = qconfig
        
        # Prepare for calibration
        model_prepared = quantization.prepare(model)
        
        # Calibration
        logger.info("Running calibration...")
        model_prepared.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                model_prepared(data)
                if batch_idx >= 100:  # Limit calibration samples
                    break
        
        # Convert to quantized model
        model_quantized = quantization.convert(model_prepared)
        logger.info(f"Successfully applied PTQ with {self.backend} backend")
        
        return model_quantized
    
    def apply_fx_graph_mode_quantization(
        self,
        model: nn.Module,
        example_inputs: Tuple,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        mode: str = 'static'
    ) -> nn.Module:
        """Apply FX Graph Mode Quantization"""
        
        model.eval()
        
        if mode == 'dynamic':
            # Dynamic quantization
            qconfig_mapping = QConfigMapping().set_global(
                quantization.default_dynamic_qconfig
            )
            model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
            model_quantized = quantize_fx.convert_fx(model_prepared)
            
        elif mode == 'static':
            # Static quantization
            qconfig_mapping = get_default_qconfig_mapping(self.backend)
            model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
            
            # Calibration if loader provided
            if calibration_loader:
                logger.info("Running FX mode calibration...")
                model_prepared.eval()
                with torch.no_grad():
                    for batch_idx, (data, _) in enumerate(calibration_loader):
                        model_prepared(data)
                        if batch_idx >= 100:
                            break
            
            model_quantized = quantize_fx.convert_fx(model_prepared)
        
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        logger.info(f"Successfully applied FX Graph Mode {mode} quantization")
        return model_quantized

class IntelligentCache:
    """Multi-level intelligent caching system"""
    
    def __init__(
        self,
        embedding_cache_size: int = 1000,
        feature_cache_size: int = 500,
        result_cache_size: int = 100,
        ttl_seconds: int = 3600
    ):
        self.embedding_cache = {}
        self.feature_cache = {}
        self.result_cache = {}
        
        self.embedding_cache_size = embedding_cache_size
        self.feature_cache_size = feature_cache_size
        self.result_cache_size = result_cache_size
        self.ttl_seconds = ttl_seconds
        
        self.access_times = {}
        
    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, torch.Tensor):
            return str(hash(data.detach().cpu().numpy().tobytes()))
        else:
            return str(hash(str(data)))
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - timestamp > self.ttl_seconds
    
    def _evict_lru(self, cache: Dict, max_size: int):
        """Evict least recently used items"""
        if len(cache) >= max_size:
            # Sort by access time and remove oldest
            sorted_items = sorted(
                cache.items(),
                key=lambda x: self.access_times.get(x[0], 0)
            )
            for key, _ in sorted_items[:len(cache) - max_size + 1]:
                cache.pop(key, None)
                self.access_times.pop(key, None)
    
    def get_embedding(self, key: str) -> Optional[torch.Tensor]:
        """Get cached embedding"""
        if key in self.embedding_cache:
            timestamp, value = self.embedding_cache[key]
            if not self._is_expired(timestamp):
                self.access_times[key] = time.time()
                return value
            else:
                del self.embedding_cache[key]
        return None
    
    def set_embedding(self, key: str, value: torch.Tensor):
        """Cache embedding with LRU eviction"""
        self._evict_lru(self.embedding_cache, self.embedding_cache_size)
        self.embedding_cache[key] = (time.time(), value)
        self.access_times[key] = time.time()
    
    def get_features(self, key: str) -> Optional[torch.Tensor]:
        """Get cached features"""
        if key in self.feature_cache:
            timestamp, value = self.feature_cache[key]
            if not self._is_expired(timestamp):
                self.access_times[key] = time.time()
                return value
            else:
                del self.feature_cache[key]
        return None
    
    def set_features(self, key: str, value: torch.Tensor):
        """Cache features with LRU eviction"""
        self._evict_lru(self.feature_cache, self.feature_cache_size)
        self.feature_cache[key] = (time.time(), value)
        self.access_times[key] = time.time()
    
    def get_result(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if key in self.result_cache:
            timestamp, value = self.result_cache[key]
            if not self._is_expired(timestamp):
                self.access_times[key] = time.time()
                return value
            else:
                del self.result_cache[key]
        return None
    
    def set_result(self, key: str, value: Any):
        """Cache result with LRU eviction"""
        self._evict_lru(self.result_cache, self.result_cache_size)
        self.result_cache[key] = (time.time(), value)
        self.access_times[key] = time.time()
    
    def clear_expired(self):
        """Clear all expired cache entries"""
        current_time = time.time()
        
        for cache in [self.embedding_cache, self.feature_cache, self.result_cache]:
            expired_keys = [
                key for key, (timestamp, _) in cache.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            for key in expired_keys:
                cache.pop(key, None)
                self.access_times.pop(key, None)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'embedding_cache_size': len(self.embedding_cache),
            'feature_cache_size': len(self.feature_cache),
            'result_cache_size': len(self.result_cache),
            'total_access_times': len(self.access_times)
        }

class PerformanceProfiler:
    """Real-time performance metrics collection with mobile benchmarking"""
    
    def __init__(self):
        self.metrics = {
            'inference_times': [],
            'memory_usage': [],
            'cache_hit_rates': [],
            'model_accuracy': [],
            'throughput': []
        }
        self.start_time = None
        
    def start_profiling(self):
        """Start performance profiling"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def end_profiling(self) -> float:
        """End profiling and return inference time"""
        if self.start_time is None:
            return 0.0
        
        inference_time = time.time() - self.start_time
        self.metrics['inference_times'].append(inference_time)
        
        # Record memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory_usage = 0  # CPU memory tracking would require psutil
        
        self.metrics['memory_usage'].append(memory_usage)
        
        self.start_time = None
        return inference_time
    
    def record_cache_hit_rate(self, hit_rate: float):
        """Record cache hit rate"""
        self.metrics['cache_hit_rates'].append(hit_rate)
    
    def record_accuracy(self, accuracy: float):
        """Record model accuracy"""
        self.metrics['model_accuracy'].append(accuracy)
    
    def record_throughput(self, throughput: float):
        """Record throughput (samples/second)"""
        self.metrics['throughput'].append(throughput)
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                summary[metric_name] = {
                    'mean': 0, 'min': 0, 'max': 0, 'count': 0
                }
        
        return summary
    
    def benchmark_model(
        self, 
        model: nn.Module, 
        test_loader: torch.utils.data.DataLoader,
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> Dict:
        """Comprehensive model benchmarking"""
        
        model.eval()
        inference_times = []
        
        # Warmup
        logger.info(f"Warming up for {num_warmup} iterations...")
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= num_warmup:
                    break
                _ = model(data)
        
        # Benchmark
        logger.info(f"Benchmarking for {num_iterations} iterations...")
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= num_iterations:
                    break
                
                self.start_profiling()
                _ = model(data)
                inference_time = self.end_profiling()
                inference_times.append(inference_time)
        
        # Calculate statistics
        return {
            'mean_inference_time': sum(inference_times) / len(inference_times),
            'min_inference_time': min(inference_times),
            'max_inference_time': max(inference_times),
            'throughput_fps': len(inference_times) / sum(inference_times),
            'iterations': len(inference_times)
        }

class CacheManager:
    """Intelligent caching system for mobile optimization"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 100, enable_persistence: bool = True):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.enable_persistence = enable_persistence
        self.cache = {}
        self.access_times = {}
        
        if enable_persistence and self.cache_dir.exists():
            self._load_persistent_cache()
    
    def store(self, key: str, tensor: torch.Tensor):
        """Store tensor in cache with LRU eviction"""
        if self._should_evict():
            self._evict_lru()
        
        self.cache[key] = tensor.clone()
        self.access_times[key] = time.time()
        
        if self.enable_persistence:
            self._save_to_disk(key, tensor)
    
    def retrieve(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve tensor from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key].clone()
        
        if self.enable_persistence:
            return self._load_from_disk(key)
        
        return None
    
    def _should_evict(self) -> bool:
        """Check if cache should evict items"""
        current_size = sum(
            tensor.nelement() * tensor.element_size() 
            for tensor in self.cache.values()
        ) / (1024 * 1024)  # Convert to MB
        
        return current_size > self.max_size_mb
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def _save_to_disk(self, key: str, tensor: torch.Tensor):
        """Save tensor to disk"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, self.cache_dir / f"{key}.pt")
    
    def _load_from_disk(self, key: str) -> Optional[torch.Tensor]:
        """Load tensor from disk"""
        file_path = self.cache_dir / f"{key}.pt"
        if file_path.exists():
            return torch.load(file_path)
        return None
    
    def _load_persistent_cache(self):
        """Load persistent cache from disk"""
        if not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.pt"):
            key = cache_file.stem
            try:
                tensor = torch.load(cache_file)
                self.cache[key] = tensor
                self.access_times[key] = cache_file.stat().st_mtime
            except Exception as e:
                logger.warning(f"Failed to load cached tensor {key}: {e}")

class PerformanceBenchmark:
    """Performance benchmarking for mobile models"""
    
    def __init__(self, warmup_iterations: int = 10, benchmark_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
    
    def benchmark_model(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Benchmark model performance"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_inference_time = (total_time / self.benchmark_iterations) * 1000  # ms
        throughput = self.benchmark_iterations / total_time  # inferences/sec
        
        # Memory usage
        memory_usage = self._estimate_memory_usage(model, input_tensor)
        
        return {
            'inference_time_ms': avg_inference_time,
            'throughput_fps': throughput,
            'memory_usage_mb': memory_usage,
            'total_benchmark_time_s': total_time
        }
    
    def _estimate_memory_usage(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """Estimate memory usage in MB"""
        # Model parameters memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Input memory
        input_memory = input_tensor.nelement() * input_tensor.element_size()
        
        # Rough estimate of activation memory (simplified)
        activation_memory = input_memory * 4  # Rough multiplier for activations
        
        total_memory = param_memory + input_memory + activation_memory
        return total_memory / (1024 * 1024)  # Convert to MB

class MobileOptimizer:
    """Complete mobile optimization pipeline"""
    
    def __init__(
        self, 
        cache_size_mb: int = 100,
        enable_profiling: bool = False,
        optimization_level: int = 2
    ):
        self.cache_manager = CacheManager(
            cache_dir="./cache", 
            max_size_mb=cache_size_mb
        )
        self.benchmark = PerformanceBenchmark()
        self.quantization_optimizer = QuantizationOptimizer()
        self.enable_profiling = enable_profiling
        self.optimization_level = optimization_level
        
        logger.info(f"Initialized MobileOptimizer with optimization level {optimization_level}")
    
    def optimize_model(
        self, 
        model: nn.Module, 
        example_inputs: Tuple[torch.Tensor, ...],
        optimization_config: Optional[Dict] = None
    ) -> nn.Module:
        """Complete model optimization pipeline"""
        if optimization_config is None:
            optimization_config = self.get_optimization_profile("balanced")
        
        optimized_model = model
        
        # Apply quantization if enabled
        if optimization_config.get('quantization', True):
            cache_key = f"quantized_{hash(str(model))}"
            cached_model = self.cache_manager.retrieve(cache_key)
            
            if cached_model is not None:
                logger.info("Using cached quantized model")
                optimized_model = cached_model
            else:
                logger.info("Applying quantization...")
                optimized_model = self.quantization_optimizer.apply_dynamic_quantization(model)
                self.cache_manager.store(cache_key, optimized_model)
        
        # Apply graph optimization if enabled
        if optimization_config.get('graph_optimization', True):
            try:
                logger.info("Applying graph optimization...")
                optimized_model = torch.jit.script(optimized_model)
                optimized_model = torch.jit.optimize_for_inference(optimized_model)
            except Exception as e:
                logger.warning(f"Graph optimization failed: {e}")
        
        # Apply hardware acceleration if enabled
        if optimization_config.get('hardware_acceleration', False):
            logger.info("Configuring hardware acceleration...")
            HardwareDetector.configure_backends()
        
        return optimized_model
    
    def get_optimization_profile(self, profile_name: str) -> Dict:
        """Get predefined optimization profile"""
        profiles = {
            'speed': {
                'quantization': True,
                'graph_optimization': True,
                'aggressive_optimization': True,
                'hardware_acceleration': True
            },
            'balanced': {
                'quantization': True,
                'graph_optimization': True,
                'aggressive_optimization': False,
                'hardware_acceleration': True
            },
            'quality': {
                'quantization': False,
                'graph_optimization': True,
                'aggressive_optimization': False,
                'hardware_acceleration': False
            }
        }
        
        return profiles.get(profile_name, profiles['balanced'])
    
    def get_device_specific_configs(self) -> Dict:
        """Get device-specific optimization configurations"""
        return {
            'android_low_end': {
                'max_memory_mb': 50,
                'optimization_level': 3,
                'quantization': True,
                'graph_optimization': True
            },
            'android_high_end': {
                'max_memory_mb': 200,
                'optimization_level': 2,
                'quantization': False,
                'graph_optimization': True
            },
            'ios_modern': {
                'max_memory_mb': 150,
                'optimization_level': 2,
                'quantization': True,
                'graph_optimization': True
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Test hardware detection
    backend = HardwareDetector.get_optimal_backend()
    print(f"Detected optimal backend: {backend}")
    
    # Test quantization
    optimizer = MobileOptimizer()
    
    # Create dummy model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = quantization.QuantStub()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)
            self.dequant = quantization.DeQuantStub()
        
        def forward(self, x):
            x = self.quant(x)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.dequant(x)
            return x
    
    model = TestModel()
    example_inputs = (torch.randn(1, 3, 224, 224),)
    
    print("Testing quantization...")
    optimized_model = optimizer.optimize_model(
        model, 
        example_inputs,
        quantization_mode='fx_dynamic'
    )
    
    print("Optimization test completed successfully!")
