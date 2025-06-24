"""
M³TM Mobile Optimizasyon Test Modülü

Bu modül mobile optimizasyon pipeline'ının test edilmesi için
kapsamlı test senaryoları sağlar.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil

from m3tm.mobile.optimization_pipeline import OptimizationPipeline, create_optimization_pipeline
from m3tm.mobile.quantization import QuantizationManager, create_quantization_pipeline
from m3tm.mobile.pruning import PruningManager, create_pruning_pipeline
from m3tm.mobile.knowledge_distillation import DistillationTrainer, create_distillation_pipeline
from m3tm.mobile.torchscript_converter import TorchScriptConverter, create_torchscript_pipeline
from m3tm.mobile.benchmark_utils import ModelBenchmarker, create_benchmarker


class SimpleTestModel(nn.Module):
    """Test için basit model."""
    
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestConvModel(nn.Module):
    """Test için basit CNN model."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@pytest.fixture
def simple_model():
    """Simple test model fixture."""
    return SimpleTestModel()


@pytest.fixture
def conv_model():
    """CNN test model fixture."""
    return TestConvModel()


@pytest.fixture
def example_inputs():
    """Example inputs fixture."""
    return torch.randn(1, 784)


@pytest.fixture
def conv_inputs():
    """CNN example inputs fixture.""" 
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def temp_dir():
    """Temporary directory fixture."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestQuantization:
    """Quantization test sınıfı."""
    
    def test_dynamic_quantization(self, simple_model, example_inputs):
        """Dynamic quantization testi."""
        quantizer = create_quantization_pipeline()
        
        # Quantization uygula
        quantized_model, metrics = quantizer.apply_dynamic_quantization(
            simple_model, benchmark=True
        )
        
        # Assertions
        assert quantized_model is not None
        assert 'compression_metrics' in metrics
        assert 'quantized_metrics' in metrics
        
        # Model çalışabilir olmalı
        with torch.no_grad():
            output = quantized_model(example_inputs)
            assert output.shape == (1, 10)
    
    def test_quantization_config(self):
        """Quantization config testi."""
        from m3tm.mobile.quantization import QuantizationConfig
        
        config = QuantizationConfig(
            backend="fbgemm",
            calibration_iterations=50
        )
        
        assert config.backend == "fbgemm"
        assert config.calibration_iterations == 50
        
        quantizer = QuantizationManager(config)
        assert quantizer.config.backend == "fbgemm"


class TestPruning:
    """Pruning test sınıfı."""
    
    def test_unstructured_pruning(self, simple_model, example_inputs):
        """Unstructured pruning testi."""
        pruner = create_pruning_pipeline(
            pruning_type="unstructured",
            amount=0.2
        )
        
        # Pruning uygula
        pruned_model, metrics = pruner.apply_unstructured_pruning(
            simple_model, benchmark=True
        )
        
        # Assertions
        assert pruned_model is not None
        assert 'sparsity_info' in metrics
        assert metrics['sparsity_info']['global_sparsity'] > 0
        
        # Model çalışabilir olmalı
        with torch.no_grad():
            output = pruned_model(example_inputs)
            assert output.shape == (1, 10)
    
    def test_structured_pruning(self, conv_model, conv_inputs):
        """Structured pruning testi."""
        pruner = create_pruning_pipeline(
            pruning_type="structured",
            amount=0.1
        )
        
        # Pruning uygula  
        pruned_model, metrics = pruner.apply_structured_pruning(
            conv_model, benchmark=True
        )
        
        # Assertions
        assert pruned_model is not None
        assert 'sparsity_info' in metrics
        
        # Model çalışabilir olmalı
        with torch.no_grad():
            output = pruned_model(conv_inputs)
            assert output.shape == (1, 10)
    
    def test_progressive_pruning(self, simple_model, example_inputs):
        """Progressive pruning testi."""
        pruner = create_pruning_pipeline()
        
        # Progressive pruning uygula
        pruned_model, metrics = pruner.apply_progressive_pruning(
            simple_model, 
            target_sparsity=0.3,
            num_steps=3,
            benchmark=True
        )
        
        # Assertions
        assert pruned_model is not None
        assert 'final_sparsity_info' in metrics
        assert len(metrics['step_metrics']) == 3


class TestKnowledgeDistillation:
    """Knowledge distillation test sınıfı."""
    
    def test_student_model_creation(self, simple_model):
        """Student model oluşturma testi."""
        distiller = create_distillation_pipeline()
        
        # Student model oluştur
        student_model = distiller.create_student_model(
            simple_model, 
            compression_ratio=0.5
        )
        
        # Assertions
        assert student_model is not None
        
        # Student daha küçük olmalı
        teacher_params = sum(p.numel() for p in simple_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        
        assert student_params < teacher_params
    
    def test_distillation_loss(self):
        """Distillation loss testi."""
        from m3tm.mobile.knowledge_distillation import DistillationLoss
        
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.7, beta=0.3)
        
        # Dummy outputs
        student_logits = torch.randn(2, 10)
        teacher_logits = torch.randn(2, 10)
        labels = torch.randint(0, 10, (2,))
        
        # Loss hesapla
        loss = loss_fn(student_logits, teacher_logits, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestTorchScriptConverter:
    """TorchScript converter test sınıfı."""
    
    def test_trace_conversion(self, simple_model, example_inputs):
        """Trace conversion testi."""
        converter = create_torchscript_pipeline(method="trace")
        
        # TorchScript'e dönüştür
        scripted_model, metrics = converter.convert_to_torchscript(
            simple_model, example_inputs, benchmark=True
        )
        
        # Assertions
        assert scripted_model is not None
        assert isinstance(scripted_model, torch.jit.ScriptModule)
        assert 'conversion_metrics' in metrics
        
        # Model çalışabilir olmalı
        with torch.no_grad():
            output = scripted_model(example_inputs)
            assert output.shape == (1, 10)
    
    def test_mobile_optimization(self, simple_model, example_inputs, temp_dir):
        """Mobile optimization testi."""
        converter = create_torchscript_pipeline(optimize_for_mobile=True)
        
        # Convert ve save
        scripted_model, metrics = converter.convert_to_torchscript(
            simple_model, example_inputs
        )
        
        # Save model
        save_path = temp_dir / "mobile_model.pt"
        converter.save_torchscript_model(scripted_model, save_path)
        
        assert save_path.exists()
        
        # Load model
        loaded_model, metadata = converter.load_torchscript_model(save_path)
        assert loaded_model is not None


class TestBenchmarkUtils:
    """Benchmark utilities test sınıfı."""
    
    def test_model_metrics(self, simple_model, example_inputs):
        """Model metrics testi."""
        benchmarker = create_benchmarker()
        
        metrics = benchmarker.get_model_metrics(simple_model, example_inputs)
        
        # Assertions
        assert 'param_count' in metrics
        assert 'model_size_mb' in metrics
        assert 'avg_inference_time_ms' in metrics
        assert metrics['param_count'] > 0
        assert metrics['model_size_mb'] > 0
    
    def test_model_comparison(self, simple_model, example_inputs):
        """Model comparison testi."""
        benchmarker = create_benchmarker()
        
        # Optimize edilmiş model oluştur (quantization)
        quantizer = create_quantization_pipeline()
        optimized_model, _ = quantizer.apply_dynamic_quantization(simple_model)
        
        # Compare models
        comparison = benchmarker.compare_models(
            simple_model, optimized_model, example_inputs
        )
        
        # Assertions
        assert 'original_metrics' in comparison
        assert 'optimized_metrics' in comparison
        assert 'comparison' in comparison


class TestOptimizationPipeline:
    """Optimization pipeline test sınıfı."""
    
    def test_basic_optimization(self, simple_model, example_inputs, temp_dir):
        """Temel optimizasyon pipeline testi."""
        pipeline = create_optimization_pipeline(
            target_size_reduction=0.3,
            target_speed_improvement=1.5,
            optimization_techniques=["quantization", "pruning"]
        )
        
        # Optimize model
        optimized_model, results = pipeline.optimize_model(
            simple_model, example_inputs, output_dir=temp_dir
        )
        
        # Assertions
        assert optimized_model is not None
        assert 'original_metrics' in results
        assert 'final_metrics' in results
        assert 'final_comparison' in results
        assert 'success_evaluation' in results
        
        # Reports oluşturulmuş olmalı
        reports_dir = temp_dir / "reports"
        assert reports_dir.exists()
        assert (reports_dir / "optimization_report.md").exists()
        assert (reports_dir / "optimization_metrics.json").exists()
    
    def test_progressive_optimization(self, conv_model, conv_inputs, temp_dir):
        """Progressive optimization testi."""
        from m3tm.mobile.optimization_pipeline import OptimizationConfig
        
        config = OptimizationConfig(
            target_size_reduction=0.4,
            progressive_optimization=True,
            optimization_techniques=["pruning", "quantization", "torchscript"]
        )
        
        pipeline = OptimizationPipeline(config)
        
        # Optimize model
        optimized_model, results = pipeline.optimize_model(
            conv_model, conv_inputs, output_dir=temp_dir
        )
        
        # Assertions
        assert optimized_model is not None
        assert len(results['optimization_steps']) >= 2
        
        # Success evaluation
        success = results['success_evaluation']
        assert 'targets_met' in success
        assert 'overall_success' in success


class TestIntegration:
    """Integration test sınıfı."""
    
    def test_full_pipeline_integration(self, conv_model, conv_inputs, temp_dir):
        """Full pipeline integration testi."""
        # Create comprehensive pipeline
        pipeline = create_optimization_pipeline(
            target_size_reduction=0.6,
            target_speed_improvement=2.0,
            optimization_techniques=["quantization", "pruning", "torchscript"]
        )
        
        # Run full optimization
        optimized_model, results = pipeline.optimize_model(
            conv_model, 
            conv_inputs,
            output_dir=temp_dir
        )
        
        # Comprehensive assertions
        assert optimized_model is not None
        
        # Metrics validation
        original_size = results['original_metrics']['model_size_mb']
        final_size = results['final_metrics']['model_size_mb']
        assert final_size < original_size  # Size should be reduced
        
        # Functional validation
        with torch.no_grad():
            output = optimized_model(conv_inputs)
            assert output.shape == (1, 10)
        
        # Platform exports validation
        platform_exports = results.get('platform_exports', {})
        assert len(platform_exports) > 0
        
        # Reports validation
        reports_dir = temp_dir / "reports"
        assert reports_dir.exists()


# Utility functions for testing
def create_test_dataloader(input_shape=(1, 784), batch_size=8, num_batches=10):
    """Test için dummy data loader oluşturur."""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Random data oluştur
    data = torch.randn(batch_size * num_batches, *input_shape[1:])
    labels = torch.randint(0, 10, (batch_size * num_batches,))
    
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # Manual test çalıştırma
    print("M³TM Mobile Optimization Tests")
    print("=" * 40)
    
    # Simple model test
    model = SimpleTestModel()
    inputs = torch.randn(1, 784)
    
    # Quantization test
    print("Testing Quantization...")
    quantizer = create_quantization_pipeline()
    quantized_model, metrics = quantizer.apply_dynamic_quantization(model, benchmark=True)
    print(f"Quantization completed. Size reduction: {metrics['compression_metrics'].get('size_reduction_percent', 0):.1f}%")
    
    # Pruning test
    print("Testing Pruning...")
    pruner = create_pruning_pipeline(amount=0.2)
    pruned_model, metrics = pruner.apply_unstructured_pruning(model, benchmark=True)
    print(f"Pruning completed. Sparsity: {metrics['sparsity_info']['global_sparsity']:.1f}%")
    
    print("All tests completed successfully!")
