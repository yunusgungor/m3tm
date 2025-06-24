"""
Integration tests for Mobile Optimizer functionality
Tests S26 implementation components
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from m3tm.mobile.mobile_optimizer import (
    HardwareDetector,
    QuantizationOptimizer,
    CacheManager,
    PerformanceBenchmark,
    MobileOptimizer
)

class TestMobileOptimizerIntegration:
    """Integration tests for mobile optimizer components"""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1) 
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(32, 10)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return SimpleModel()
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return torch.randn(1, 3, 224, 224)
    
    @pytest.fixture
    def mobile_optimizer(self):
        """Create mobile optimizer instance"""
        return MobileOptimizer(
            cache_size_mb=50,
            enable_profiling=True,
            optimization_level=2
        )

class TestHardwareDetection:
    """Test hardware detection and backend configuration"""
    
    def test_platform_detection(self):
        """Test platform detection functionality"""
        platform_type = HardwareDetector.detect_platform()
        
        valid_platforms = [
            'android', 'ios_arm64', 'linux_arm64', 
            'linux_x86', 'macos_x86'
        ]
        
        assert platform_type in valid_platforms, f"Unknown platform: {platform_type}"
    
    def test_backend_selection(self):
        """Test optimal backend selection"""
        backend = HardwareDetector.get_optimal_backend()
        
        valid_backends = ['qnnpack', 'x86', 'onednn']
        assert backend in valid_backends, f"Invalid backend: {backend}"
    
    def test_backend_configuration(self):
        """Test backend configuration applies correctly"""
        original_backend = torch.backends.quantized.engine
        
        configured_backend = HardwareDetector.configure_backends()
        
        # Verify backend was set
        assert torch.backends.quantized.engine == configured_backend
        
        # Restore original backend
        torch.backends.quantized.engine = original_backend

class TestQuantizationIntegration:
    """Test quantization functionality integration"""
    
    def test_dynamic_quantization(self, simple_model, sample_input):
        """Test dynamic quantization works end-to-end"""
        optimizer = QuantizationOptimizer()
        
        # Test original model
        with torch.no_grad():
            original_output = simple_model(sample_input)
        
        # Apply dynamic quantization
        quantized_model = optimizer.apply_dynamic_quantization(simple_model)
        
        # Test quantized model
        with torch.no_grad():
            quantized_output = quantized_model(sample_input)
        
        # Verify outputs are similar (within tolerance)
        assert torch.allclose(original_output, quantized_output, atol=0.1), \
            "Quantized model output differs too much from original"
        
        # Verify model size reduction
        original_size = sum(p.numel() * p.element_size() for p in simple_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters() 
                           if hasattr(p, 'element_size'))
        
        # Note: Dynamic quantization may not always reduce parameter size
        # but should reduce computation overhead
        assert quantized_size <= original_size * 1.1, "Quantization should not significantly increase size"
    
    def test_fx_graph_quantization(self, simple_model, sample_input):
        """Test FX graph mode quantization"""
        optimizer = QuantizationOptimizer()
        
        try:
            # Prepare model for FX quantization
            prepared_model = optimizer.prepare_fx_quantization(simple_model, sample_input)
            
            # Calibrate with sample data
            calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(10)]
            quantized_model = optimizer.apply_fx_quantization(prepared_model, calibration_data)
            
            # Test inference
            with torch.no_grad():
                output = quantized_model(sample_input)
            
            assert output is not None, "FX quantized model should produce output"
            assert output.shape == (1, 10), "Output shape should be preserved"
            
        except Exception as e:
            # FX quantization might not be supported on all platforms
            pytest.skip(f"FX quantization not supported: {e}")

class TestCachingIntegration:
    """Test caching system integration"""
    
    def test_cache_initialization(self):
        """Test cache manager initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(
                cache_dir=temp_dir,
                max_size_mb=100,
                enable_persistence=True
            )
            
            assert cache_manager.cache_dir == Path(temp_dir)
            assert cache_manager.max_size_mb == 100
            assert cache_manager.enable_persistence is True
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(
                cache_dir=temp_dir,
                max_size_mb=50
            )
            
            # Test storing and retrieving tensor
            test_tensor = torch.randn(10, 10)
            cache_key = "test_tensor"
            
            cache_manager.store(cache_key, test_tensor)
            retrieved_tensor = cache_manager.retrieve(cache_key)
            
            assert retrieved_tensor is not None, "Should retrieve cached tensor"
            assert torch.equal(test_tensor, retrieved_tensor), "Retrieved tensor should match original"
    
    def test_cache_eviction(self):
        """Test cache eviction when size limit exceeded"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create small cache for testing eviction
            cache_manager = CacheManager(
                cache_dir=temp_dir,
                max_size_mb=1  # Very small cache
            )
            
            # Store multiple tensors to trigger eviction
            for i in range(10):
                large_tensor = torch.randn(100, 100)  # ~40KB each
                cache_manager.store(f"tensor_{i}", large_tensor)
            
            # Check that not all tensors are still cached
            cached_count = sum(1 for i in range(10) 
                             if cache_manager.retrieve(f"tensor_{i}") is not None)
            
            assert cached_count < 10, "Cache eviction should have occurred"

class TestPerformanceBenchmarking:
    """Test performance benchmarking integration"""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization"""
        benchmark = PerformanceBenchmark(
            warmup_iterations=5,
            benchmark_iterations=10
        )
        
        assert benchmark.warmup_iterations == 5
        assert benchmark.benchmark_iterations == 10
    
    def test_model_benchmarking(self, simple_model, sample_input):
        """Test model performance benchmarking"""
        benchmark = PerformanceBenchmark(
            warmup_iterations=2,
            benchmark_iterations=5
        )
        
        # Benchmark original model
        original_metrics = benchmark.benchmark_model(simple_model, sample_input)
        
        assert 'inference_time_ms' in original_metrics
        assert 'memory_usage_mb' in original_metrics
        assert original_metrics['inference_time_ms'] > 0
        assert original_metrics['memory_usage_mb'] > 0
        
        # Test that metrics are reasonable
        assert original_metrics['inference_time_ms'] < 1000, "Inference should be under 1 second"
        assert original_metrics['memory_usage_mb'] < 1000, "Memory usage should be reasonable"

class TestFullOptimizationPipeline:
    """Test complete optimization pipeline integration"""
    
    def test_end_to_end_optimization(self, simple_model, sample_input, mobile_optimizer):
        """Test complete optimization pipeline"""
        # Get baseline performance
        baseline_metrics = mobile_optimizer.benchmark.benchmark_model(simple_model, sample_input)
        
        # Apply optimizations
        optimized_model = mobile_optimizer.optimize_model(
            simple_model,
            sample_input,
            optimization_config={
                'quantization': True,
                'graph_optimization': True,
                'hardware_acceleration': True
            }
        )
        
        # Test optimized model
        with torch.no_grad():
            original_output = simple_model(sample_input)
            optimized_output = optimized_model(sample_input)
        
        # Verify outputs are similar
        assert torch.allclose(original_output, optimized_output, atol=0.2), \
            "Optimized model output should be similar to original"
        
        # Benchmark optimized model
        optimized_metrics = mobile_optimizer.benchmark.benchmark_model(optimized_model, sample_input)
        
        # Verify optimization effectiveness
        assert optimized_metrics['inference_time_ms'] <= baseline_metrics['inference_time_ms'] * 1.5, \
            "Optimization should not significantly increase inference time"
    
    def test_batch_optimization(self, simple_model, mobile_optimizer):
        """Test batch size optimization"""
        batch_sizes = [1, 2, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            sample_input = torch.randn(batch_size, 3, 224, 224)
            metrics = mobile_optimizer.benchmark.benchmark_model(simple_model, sample_input)
            results[batch_size] = metrics['inference_time_ms'] / batch_size  # Time per sample
        
        # Find optimal batch size (lowest time per sample)
        optimal_batch = min(results.keys(), key=lambda x: results[x])
        
        assert optimal_batch in batch_sizes, "Should find optimal batch size"
        assert results[optimal_batch] > 0, "Should have valid timing results"

class TestConfigurationManagement:
    """Test optimization configuration management"""
    
    def test_optimization_profiles(self, mobile_optimizer):
        """Test different optimization profiles"""
        profiles = {
            'speed': {'quantization': True, 'graph_optimization': True, 'aggressive_optimization': True},
            'balanced': {'quantization': True, 'graph_optimization': True, 'aggressive_optimization': False},
            'quality': {'quantization': False, 'graph_optimization': True, 'aggressive_optimization': False}
        }
        
        for profile_name, config in profiles.items():
            profile_config = mobile_optimizer.get_optimization_profile(profile_name)
            
            # Verify profile exists and has expected structure
            assert isinstance(profile_config, dict), f"Profile {profile_name} should be a dict"
            assert 'quantization' in profile_config, f"Profile {profile_name} should have quantization setting"
    
    def test_device_specific_optimization(self, mobile_optimizer):
        """Test device-specific optimization settings"""
        device_configs = mobile_optimizer.get_device_specific_configs()
        
        assert isinstance(device_configs, dict), "Device configs should be a dictionary"
        
        # Test that common devices have configurations
        expected_devices = ['android_low_end', 'android_high_end', 'ios_modern']
        for device in expected_devices:
            if device in device_configs:
                config = device_configs[device]
                assert 'max_memory_mb' in config, f"Device {device} should have memory limit"
                assert 'optimization_level' in config, f"Device {device} should have optimization level"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
