"""
M³TM Mobile Optimization Örnek Script

Bu script M³TM modellerinin mobil platformlar için
optimize edilmesi sürecini gösterir.
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
import argparse

# M³TM mobile optimization imports
from m3tm.mobile.optimization_pipeline import create_optimization_pipeline
from m3tm.mobile.quantization import create_quantization_pipeline
from m3tm.mobile.pruning import create_pruning_pipeline
from m3tm.mobile.knowledge_distillation import create_distillation_pipeline
from m3tm.mobile.torchscript_converter import create_torchscript_pipeline
from m3tm.mobile.benchmark_utils import create_benchmarker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleM3TMModel(nn.Module):
    """
    Örnek M³TM benzeri model.
    Gerçek projede m3tm.core.base_model kullanılacak.
    """
    
    def __init__(self, 
                 text_dim=512, 
                 image_dim=2048, 
                 fusion_dim=768,
                 output_dim=256):
        super().__init__()
        
        # Text encoder simulation
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim//2),
            nn.ReLU(),
            nn.Linear(fusion_dim//2, output_dim)
        )
        
        # Image encoder simulation
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim//2),
            nn.ReLU(),
            nn.Linear(fusion_dim//2, output_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, output_dim)
        )
        
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: (batch_size, text_dim)
            image_features: (batch_size, image_dim)
        """
        text_encoded = self.text_encoder(text_features)
        image_encoded = self.image_encoder(image_features)
        
        # Concatenate and fuse
        combined = torch.cat([text_encoded, image_encoded], dim=1)
        fused = self.fusion(combined)
        
        return fused


def create_example_data(batch_size=1, text_dim=512, image_dim=2048):
    """Örnek test verisi oluşturur."""
    text_features = torch.randn(batch_size, text_dim)
    image_features = torch.randn(batch_size, image_dim)
    return text_features, image_features


def demonstrate_quantization():
    """Quantization optimizasyonu gösterimi."""
    logger.info("=" * 60)
    logger.info("Quantization Optimizasyonu Demonstrasyonu")
    logger.info("=" * 60)
    
    # Model ve data oluştur
    model = ExampleM3TMModel()
    text_features, image_features = create_example_data()
    
    # Benchmark original model
    benchmarker = create_benchmarker()
    original_metrics = benchmarker.get_model_metrics(model)
    
    logger.info(f"Orijinal Model:")
    logger.info(f"  Size: {original_metrics['model_size_mb']:.2f} MB")
    logger.info(f"  Parameters: {original_metrics['param_count']:,}")
    
    # Apply quantization
    quantizer = create_quantization_pipeline(backend="fbgemm")
    
    def model_forward(inputs):
        """Model wrapper for quantization."""
        text_feat, image_feat = inputs[0], inputs[1]
        return model(text_feat, image_feat)
    
    # Create a traced version for quantization
    model.eval()
    with torch.no_grad():
        example_output = model(text_features, image_features)
        
    # Dynamic quantization
    quantized_model, quant_metrics = quantizer.apply_dynamic_quantization(
        model, benchmark=True
    )
    
    logger.info(f"Quantized Model:")
    logger.info(f"  Size: {quant_metrics['quantized_metrics']['model_size_mb']:.2f} MB")
    logger.info(f"  Size Reduction: {quant_metrics['compression_metrics']['size_reduction_percent']:.1f}%")
    
    # Test functionality
    with torch.no_grad():
        quantized_output = quantized_model(text_features, image_features)
        output_similarity = torch.allclose(example_output, quantized_output, rtol=1e-2)
        logger.info(f"  Output Similarity: {'✅ Pass' if output_similarity else '❌ Fail'}")
    
    return quantized_model


def demonstrate_pruning():
    """Pruning optimizasyonu gösterimi."""
    logger.info("=" * 60)
    logger.info("Pruning Optimizasyonu Demonstrasyonu")
    logger.info("=" * 60)
    
    # Model oluştur
    model = ExampleM3TMModel()
    text_features, image_features = create_example_data()
    
    # Original metrics
    benchmarker = create_benchmarker()
    original_metrics = benchmarker.get_model_metrics(model)
    
    logger.info(f"Orijinal Model:")
    logger.info(f"  Parameters: {original_metrics['param_count']:,}")
    
    # Apply pruning
    pruner = create_pruning_pipeline(
        pruning_type="unstructured",
        pruning_method="magnitude",
        amount=0.3  # %30 pruning
    )
    
    pruned_model, pruning_metrics = pruner.apply_unstructured_pruning(
        model, benchmark=True
    )
    
    sparsity_info = pruning_metrics['sparsity_info']
    logger.info(f"Pruned Model:")
    logger.info(f"  Global Sparsity: {sparsity_info['global_sparsity']:.1f}%")
    logger.info(f"  Zero Parameters: {sparsity_info['total_zeros']:,}")
    
    # Test functionality
    with torch.no_grad():
        pruned_output = pruned_model(text_features, image_features)
        logger.info(f"  Output Shape: {pruned_output.shape}")
        logger.info(f"  Model Still Functional: ✅")
    
    return pruned_model


def demonstrate_knowledge_distillation():
    """Knowledge distillation gösterimi."""
    logger.info("=" * 60)
    logger.info("Knowledge Distillation Demonstrasyonu")
    logger.info("=" * 60)
    
    # Teacher model oluştur
    teacher_model = ExampleM3TMModel()
    
    # Teacher metrics
    benchmarker = create_benchmarker()
    teacher_metrics = benchmarker.get_model_metrics(teacher_model)
    
    logger.info(f"Teacher Model:")
    logger.info(f"  Parameters: {teacher_metrics['param_count']:,}")
    logger.info(f"  Size: {teacher_metrics['model_size_mb']:.2f} MB")
    
    # Create student model
    distiller = create_distillation_pipeline(temperature=4.0, alpha=0.7)
    
    student_model = distiller.create_student_model(
        teacher_model, 
        compression_ratio=0.4,  # %60 smaller
        architecture_type="proportional"
    )
    
    # Student metrics
    student_metrics = benchmarker.get_model_metrics(student_model)
    
    logger.info(f"Student Model:")
    logger.info(f"  Parameters: {student_metrics['param_count']:,}")
    logger.info(f"  Size: {student_metrics['model_size_mb']:.2f} MB")
    
    actual_compression = student_metrics['param_count'] / teacher_metrics['param_count']
    logger.info(f"  Actual Compression Ratio: {actual_compression:.3f}")
    
    # Test functionality
    text_features, image_features = create_example_data()
    with torch.no_grad():
        teacher_output = teacher_model(text_features, image_features)
        student_output = student_model(text_features, image_features)
        logger.info(f"  Teacher Output Shape: {teacher_output.shape}")
        logger.info(f"  Student Output Shape: {student_output.shape}")
        logger.info(f"  Student Model Functional: ✅")
    
    return student_model


def demonstrate_torchscript_conversion():
    """TorchScript conversion gösterimi."""
    logger.info("=" * 60)
    logger.info("TorchScript Conversion Demonstrasyonu")
    logger.info("=" * 60)
    
    # Model oluştur
    model = ExampleM3TMModel()
    text_features, image_features = create_example_data()
    
    # Original metrics
    benchmarker = create_benchmarker()
    original_metrics = benchmarker.get_model_metrics(model)
    
    logger.info(f"Original PyTorch Model:")
    logger.info(f"  Size: {original_metrics['model_size_mb']:.2f} MB")
    
    # TorchScript converter
    converter = create_torchscript_pipeline(
        method="trace",
        optimize_for_mobile=True
    )
    
    # Model için wrapper oluştur (tracing için)
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            # x'i text ve image feature'lara böl
            text_dim = 512
            text_feat = x[:, :text_dim]
            image_feat = x[:, text_dim:]
            return self.model(text_feat, image_feat)
    
    # Wrapper ve combined input oluştur
    wrapper_model = ModelWrapper(model)
    combined_input = torch.cat([text_features, image_features], dim=1)
    
    # Convert to TorchScript
    scripted_model, conversion_metrics = converter.convert_to_torchscript(
        wrapper_model, combined_input, benchmark=True
    )
    
    logger.info(f"TorchScript Model:")
    torchscript_metrics = conversion_metrics['torchscript_metrics']
    logger.info(f"  Size: {torchscript_metrics['model_size_mb']:.2f} MB")
    
    conversion_comp = conversion_metrics['conversion_metrics']
    if 'size_change_percent' in conversion_comp:
        logger.info(f"  Size Change: {conversion_comp['size_change_percent']:.1f}%")
    
    # Test functionality
    with torch.no_grad():
        scripted_output = scripted_model(combined_input)
        logger.info(f"  TorchScript Output Shape: {scripted_output.shape}")
        logger.info(f"  TorchScript Model Functional: ✅")
    
    return scripted_model


def demonstrate_full_pipeline():
    """Full optimization pipeline gösterimi."""
    logger.info("=" * 60)
    logger.info("Full Optimization Pipeline Demonstrasyonu")
    logger.info("=" * 60)
    
    # Model oluştur
    model = ExampleM3TMModel()
    text_features, image_features = create_example_data()
    
    # Combined input for pipeline
    combined_input = torch.cat([text_features, image_features], dim=1)
    
    # Model wrapper
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            text_dim = 512
            text_feat = x[:, :text_dim]
            image_feat = x[:, text_dim:]
            return self.model(text_feat, image_feat)
    
    wrapper_model = ModelWrapper(model)
    
    # Create comprehensive optimization pipeline
    pipeline = create_optimization_pipeline(
        target_size_reduction=0.6,  # %60 boyut azaltımı
        target_speed_improvement=2.0,  # 2x hız artışı
        optimization_techniques=["quantization", "pruning", "torchscript"]
    )
    
    # Run optimization
    output_dir = Path("optimization_results")
    optimized_model, results = pipeline.optimize_model(
        wrapper_model,
        combined_input,
        output_dir=output_dir
    )
    
    # Results summary
    original_metrics = results['original_metrics']
    final_metrics = results['final_metrics']
    comparison = results['final_comparison']
    success = results['success_evaluation']
    
    logger.info(f"Optimization Results:")
    logger.info(f"  Original Size: {original_metrics['model_size_mb']:.2f} MB")
    logger.info(f"  Final Size: {final_metrics['model_size_mb']:.2f} MB")
    logger.info(f"  Size Reduction: {comparison.get('size_reduction_percent', 0):.1f}%")
    logger.info(f"  Speed Improvement: {comparison.get('speedup_ratio', 1.0):.1f}x")
    logger.info(f"  Overall Success: {'✅' if success['overall_success'] else '❌'}")
    logger.info(f"  Success Score: {success['success_score']:.2f}/1.0")
    
    # Platform exports
    platform_exports = results.get('platform_exports', {})
    if platform_exports:
        logger.info(f"Platform Exports:")
        for platform, path in platform_exports.items():
            logger.info(f"    {platform}: {Path(path).name}")
    
    logger.info(f"Detailed reports saved to: {output_dir}/reports/")
    
    return optimized_model, results


def main():
    """Ana demonstration fonksiyonu."""
    parser = argparse.ArgumentParser(description="M³TM Mobile Optimization Demo")
    parser.add_argument("--demo", 
                        choices=["quantization", "pruning", "distillation", "torchscript", "full", "all"],
                        default="all",
                        help="Hangi demonstrasyonu çalıştırılacağı")
    
    args = parser.parse_args()
    
    logger.info("M³TM Mobile Optimization Demonstrasyonu")
    logger.info("=" * 60)
    
    if args.demo in ["quantization", "all"]:
        demonstrate_quantization()
        
    if args.demo in ["pruning", "all"]:
        demonstrate_pruning()
        
    if args.demo in ["distillation", "all"]:
        demonstrate_knowledge_distillation()
        
    if args.demo in ["torchscript", "all"]:
        demonstrate_torchscript_conversion()
        
    if args.demo in ["full", "all"]:
        demonstrate_full_pipeline()
    
    logger.info("=" * 60)
    logger.info("Demonstrasyon tamamlandı!")


if __name__ == "__main__":
    main()
