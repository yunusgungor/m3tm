"""
M³TM Model Optimization Pipeline

Bu modül tüm optimizasyon tekniklerini birleştiren
ana pipeline'ı sağlar.
"""

import logging
from typing import Dict, Optional, Union, Any, Tuple, List, Callable
from pathlib import Path
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker, create_benchmarker
from m3tm.mobile.quantization import QuantizationManager, create_quantization_pipeline
from m3tm.mobile.pruning import PruningManager, create_pruning_pipeline
from m3tm.mobile.knowledge_distillation import DistillationTrainer, create_distillation_pipeline
from m3tm.mobile.torchscript_converter import TorchScriptConverter, create_torchscript_pipeline


logger = logging.getLogger(__name__)


class OptimizationConfig:
    """Optimizasyon pipeline yapılandırma sınıfı."""
    
    def __init__(self,
                 target_size_reduction: float = 0.6,
                 target_speed_improvement: float = 2.0,
                 target_memory_reduction: float = 0.5,
                 max_accuracy_loss: float = 0.05,
                 optimization_techniques: Optional[List[str]] = None,
                 progressive_optimization: bool = True,
                 platform_targets: Optional[List[str]] = None):
        """
        Args:
            target_size_reduction: Hedef boyut azaltımı (0.0-1.0)
            target_speed_improvement: Hedef hız iyileştirmesi (kat)
            target_memory_reduction: Hedef hafıza azaltımı (0.0-1.0)
            max_accuracy_loss: Maksimum kabul edilebilir doğruluk kaybı
            optimization_techniques: Kullanılacak teknikler listesi
            progressive_optimization: Aşamalı optimizasyon kullan
            platform_targets: Hedef platformlar ['android', 'ios', 'desktop']
        """
        self.target_size_reduction = target_size_reduction
        self.target_speed_improvement = target_speed_improvement
        self.target_memory_reduction = target_memory_reduction
        self.max_accuracy_loss = max_accuracy_loss
        self.optimization_techniques = optimization_techniques or [
            "quantization", "pruning", "knowledge_distillation", "torchscript"
        ]
        self.progressive_optimization = progressive_optimization
        self.platform_targets = platform_targets or ["android", "ios", "desktop"]
        
        # Validation
        if not 0.0 <= target_size_reduction <= 1.0:
            raise ValueError("Target size reduction 0.0-1.0 arasında olmalı")
        if target_speed_improvement < 1.0:
            raise ValueError("Target speed improvement >= 1.0 olmalı")
        if not 0.0 <= target_memory_reduction <= 1.0:
            raise ValueError("Target memory reduction 0.0-1.0 arasında olmalı")
        if not 0.0 <= max_accuracy_loss <= 1.0:
            raise ValueError("Max accuracy loss 0.0-1.0 arasında olmalı")


class OptimizationPipeline:
    """Kapsamlı model optimizasyon pipeline'ı."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Args:
            config: Optimizasyon yapılandırma objesi
        """
        self.config = config or OptimizationConfig()
        self.benchmarker = create_benchmarker()
        
        # Sub-pipeline'ları initialize et
        self.quantization_manager = create_quantization_pipeline()
        self.pruning_manager = create_pruning_pipeline()
        self.distillation_trainer = create_distillation_pipeline()
        self.torchscript_converter = create_torchscript_pipeline()
        
        # Optimization history
        self.optimization_history = []
        self.best_model = None
        self.best_metrics = None
        
    def optimize_model(self,
                       model: Union[nn.Module, BaseModel],
                       example_inputs: torch.Tensor,
                       calibration_data_loader: Optional[DataLoader] = None,
                       validation_data_loader: Optional[DataLoader] = None,
                       output_dir: Union[str, Path] = "optimized_models") -> Tuple[Any, Dict]:
        """
        Ana optimizasyon pipeline'ını çalıştırır.
        
        Args:
            model: Optimize edilecek model
            example_inputs: Örnek girdi tensörü
            calibration_data_loader: Kalibrasyon data loader (quantization için)
            validation_data_loader: Validation data loader
            output_dir: Çıktı dizini
            
        Returns:
            Tuple[optimized_model, comprehensive_metrics]
        """
        logger.info("Ana optimizasyon pipeline'ı başlatılıyor...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Başlangıç metrikleri
        start_time = time.time()
        original_metrics = self.benchmarker.get_model_metrics(model, example_inputs)
        
        logger.info(f"Orijinal model metrics:")
        logger.info(f"  Size: {original_metrics.get('model_size_mb', 'N/A'):.2f} MB")
        logger.info(f"  Parameters: {original_metrics.get('param_count', 'N/A'):,}")
        logger.info(f"  Inference Time: {original_metrics.get('avg_inference_time_ms', 'N/A'):.2f} ms")
        
        comprehensive_results = {
            'original_metrics': original_metrics,
            'optimization_steps': [],
            'final_comparison': {},
            'platform_exports': {},
            'config': {
                'target_size_reduction': self.config.target_size_reduction,
                'target_speed_improvement': self.config.target_speed_improvement,
                'optimization_techniques': self.config.optimization_techniques
            }
        }
        
        current_model = model
        step_number = 0
        
        try:
            # Optimization pipeline steps
            if self.config.progressive_optimization:
                current_model, comprehensive_results = self._run_progressive_optimization(
                    current_model, example_inputs, calibration_data_loader, 
                    validation_data_loader, comprehensive_results
                )
            else:
                current_model, comprehensive_results = self._run_sequential_optimization(
                    current_model, example_inputs, calibration_data_loader,
                    validation_data_loader, comprehensive_results
                )
            
            # Final metrics
            final_metrics = self.benchmarker.get_model_metrics(current_model, example_inputs)
            comprehensive_results['final_metrics'] = final_metrics
            
            # Overall comparison
            comparison_metrics = self._calculate_overall_metrics(original_metrics, final_metrics)
            comprehensive_results['final_comparison'] = comparison_metrics
            
            # Platform exports
            platform_exports = self._export_for_all_platforms(current_model, output_dir)
            comprehensive_results['platform_exports'] = platform_exports
            
            # Success evaluation
            success_metrics = self._evaluate_optimization_success(comparison_metrics)
            comprehensive_results['success_evaluation'] = success_metrics
            
            # Total time
            total_time = time.time() - start_time
            comprehensive_results['total_optimization_time_s'] = total_time
            
            # Generate reports
            self._generate_optimization_report(comprehensive_results, output_dir)
            
            logger.info("Optimizasyon pipeline'ı tamamlandı")
            logger.info(f"Final Results:")
            logger.info(f"  Size Reduction: {comparison_metrics.get('size_reduction_percent', 'N/A'):.1f}%")
            logger.info(f"  Speed Improvement: {comparison_metrics.get('speedup_ratio', 'N/A'):.1f}x")
            logger.info(f"  Total Time: {total_time:.1f}s")
            
            return current_model, comprehensive_results
            
        except Exception as e:
            logger.error(f"Optimizasyon pipeline hatası: {e}")
            raise
    
    def _run_progressive_optimization(self,
                                      model: nn.Module,
                                      example_inputs: torch.Tensor,
                                      calibration_data_loader: Optional[DataLoader],
                                      validation_data_loader: Optional[DataLoader],
                                      results: Dict) -> Tuple[nn.Module, Dict]:
        """
        Aşamalı optimizasyon çalıştırır.
        
        Args:
            model: Optimize edilecek model
            example_inputs: Örnek girdi tensörü
            calibration_data_loader: Kalibrasyon data loader
            validation_data_loader: Validation data loader
            results: Sonuçlar dictionary
            
        Returns:
            Tuple[optimized_model, updated_results]
        """
        logger.info("Aşamalı optimizasyon modu")
        
        current_model = model
        step_metrics = []
        
        # Step 1: Pruning (hafif)
        if "pruning" in self.config.optimization_techniques:
            logger.info("Step 1: Hafif pruning")
            current_model, pruning_results = self.pruning_manager.apply_unstructured_pruning(
                current_model, benchmark=True
            )
            step_metrics.append({"step": "pruning_light", "results": pruning_results})
        
        # Step 2: Dynamic Quantization
        if "quantization" in self.config.optimization_techniques:
            logger.info("Step 2: Dynamic quantization")
            current_model, quant_results = self.quantization_manager.apply_dynamic_quantization(
                current_model, benchmark=True
            )
            step_metrics.append({"step": "quantization_dynamic", "results": quant_results})
        
        # Step 3: Knowledge Distillation (opsiyonel)
        if "knowledge_distillation" in self.config.optimization_techniques:
            logger.info("Step 3: Knowledge distillation")
            try:
                student_model = self.distillation_trainer.create_student_model(
                    current_model, compression_ratio=0.6
                )
                
                # Mini distillation training (gerçek projede daha kapsamlı olmalı)
                if validation_data_loader:
                    # Basit optimizer oluştur
                    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
                    
                    student_model, distill_results = self.distillation_trainer.train_student(
                        current_model, student_model, validation_data_loader, 
                        optimizer, num_epochs=5
                    )
                    
                    current_model = student_model
                    step_metrics.append({"step": "knowledge_distillation", "results": distill_results})
                    
            except Exception as e:
                logger.warning(f"Knowledge distillation atlandı: {e}")
        
        # Step 4: TorchScript conversion
        if "torchscript" in self.config.optimization_techniques:
            logger.info("Step 4: TorchScript conversion")
            current_model, torchscript_results = self.torchscript_converter.convert_to_torchscript(
                current_model, example_inputs, benchmark=True
            )
            step_metrics.append({"step": "torchscript", "results": torchscript_results})
        
        results['optimization_steps'] = step_metrics
        return current_model, results
    
    def _run_sequential_optimization(self,
                                     model: nn.Module,
                                     example_inputs: torch.Tensor,
                                     calibration_data_loader: Optional[DataLoader],
                                     validation_data_loader: Optional[DataLoader],
                                     results: Dict) -> Tuple[nn.Module, Dict]:
        """
        Sıralı optimizasyon çalıştırır.
        
        Args:
            model: Optimize edilecek model
            example_inputs: Örnek girdi tensörü
            calibration_data_loader: Kalibrasyon data loader
            validation_data_loader: Validation data loader
            results: Sonuçlar dictionary
            
        Returns:
            Tuple[optimized_model, updated_results]
        """
        logger.info("Sıralı optimizasyon modu")
        
        current_model = model
        step_metrics = []
        
        # Her tekniği sırayla uygula
        for technique in self.config.optimization_techniques:
            if technique == "quantization":
                logger.info("Quantization uygulanıyor...")
                if calibration_data_loader:
                    current_model, results_step = self.quantization_manager.apply_static_quantization(
                        current_model, calibration_data_loader, example_inputs, benchmark=True
                    )
                else:
                    current_model, results_step = self.quantization_manager.apply_dynamic_quantization(
                        current_model, benchmark=True
                    )
                step_metrics.append({"step": "quantization", "results": results_step})
                
            elif technique == "pruning":
                logger.info("Pruning uygulanıyor...")
                current_model, results_step = self.pruning_manager.apply_unstructured_pruning(
                    current_model, benchmark=True
                )
                step_metrics.append({"step": "pruning", "results": results_step})
                
            elif technique == "knowledge_distillation":
                logger.info("Knowledge distillation uygulanıyor...")
                try:
                    student_model = self.distillation_trainer.create_student_model(current_model)
                    current_model = student_model  # Simplified - gerçek projede training gerekir
                    
                    results_step = {"student_created": True}
                    step_metrics.append({"step": "knowledge_distillation", "results": results_step})
                except Exception as e:
                    logger.warning(f"Knowledge distillation atlandı: {e}")
                    
            elif technique == "torchscript":
                logger.info("TorchScript conversion uygulanıyor...")
                current_model, results_step = self.torchscript_converter.convert_to_torchscript(
                    current_model, example_inputs, benchmark=True
                )
                step_metrics.append({"step": "torchscript", "results": results_step})
        
        results['optimization_steps'] = step_metrics
        return current_model, results
    
    def _calculate_overall_metrics(self, original_metrics: Dict, final_metrics: Dict) -> Dict:
        """
        Genel optimizasyon metriklerini hesaplar.
        
        Args:
            original_metrics: Orijinal model metrikleri
            final_metrics: Final model metrikleri
            
        Returns:
            Overall comparison metrics
        """
        comparison = {}
        
        # Size comparison
        if 'model_size_mb' in original_metrics and 'model_size_mb' in final_metrics:
            original_size = original_metrics['model_size_mb']
            final_size = final_metrics['model_size_mb']
            
            if original_size > 0:
                size_reduction = (original_size - final_size) / original_size
                compression_ratio = original_size / final_size if final_size > 0 else float('inf')
                
                comparison.update({
                    'size_reduction_percent': size_reduction * 100,
                    'compression_ratio': compression_ratio,
                    'original_size_mb': original_size,
                    'final_size_mb': final_size
                })
        
        # Speed comparison
        if 'avg_inference_time_ms' in original_metrics and 'avg_inference_time_ms' in final_metrics:
            original_time = original_metrics['avg_inference_time_ms']
            final_time = final_metrics['avg_inference_time_ms']
            
            if original_time > 0:
                speedup = original_time / final_time if final_time > 0 else float('inf')
                time_reduction = (original_time - final_time) / original_time
                
                comparison.update({
                    'speedup_ratio': speedup,
                    'time_reduction_percent': time_reduction * 100
                })
        
        # Parameter comparison
        if 'param_count' in original_metrics and 'param_count' in final_metrics:
            original_params = original_metrics['param_count']
            final_params = final_metrics['param_count']
            
            if original_params > 0:
                param_reduction = (original_params - final_params) / original_params
                
                comparison.update({
                    'param_reduction_percent': param_reduction * 100,
                    'original_params': original_params,
                    'final_params': final_params
                })
        
        return comparison
    
    def _evaluate_optimization_success(self, comparison_metrics: Dict) -> Dict:
        """
        Optimizasyon başarısını değerlendirir.
        
        Args:
            comparison_metrics: Karşılaştırma metrikleri
            
        Returns:
            Success evaluation metrics
        """
        success_eval = {
            'targets_met': {},
            'overall_success': False,
            'success_score': 0.0
        }
        
        targets_met = 0
        total_targets = 0
        
        # Size reduction target
        size_reduction = comparison_metrics.get('size_reduction_percent', 0) / 100
        target_size_reduction = self.config.target_size_reduction
        
        size_target_met = size_reduction >= target_size_reduction
        success_eval['targets_met']['size_reduction'] = {
            'target': target_size_reduction,
            'achieved': size_reduction,
            'met': size_target_met
        }
        
        if size_target_met:
            targets_met += 1
        total_targets += 1
        
        # Speed improvement target
        speedup = comparison_metrics.get('speedup_ratio', 1.0)
        target_speedup = self.config.target_speed_improvement
        
        speed_target_met = speedup >= target_speedup
        success_eval['targets_met']['speed_improvement'] = {
            'target': target_speedup,
            'achieved': speedup,
            'met': speed_target_met
        }
        
        if speed_target_met:
            targets_met += 1
        total_targets += 1
        
        # Overall success
        success_score = targets_met / total_targets if total_targets > 0 else 0
        success_eval['success_score'] = success_score
        success_eval['overall_success'] = success_score >= 0.5  # 50% threshold
        
        return success_eval
    
    def _export_for_all_platforms(self, model: Any, output_dir: Path) -> Dict[str, Path]:
        """
        Tüm target platformlar için export yapar.
        
        Args:
            model: Export edilecek model
            output_dir: Çıktı dizini
            
        Returns:
            Platform export paths
        """
        exports_dir = output_dir / "platform_exports"
        exports_dir.mkdir(exist_ok=True)
        
        platform_exports = {}
        
        for platform in self.config.platform_targets:
            try:
                if hasattr(self.torchscript_converter, 'export_for_platform'):
                    exports = self.torchscript_converter.export_for_platform(
                        model, platform, exports_dir
                    )
                    platform_exports.update(exports)
                else:
                    # Fallback: generic save
                    platform_path = exports_dir / f"model_{platform}.pt"
                    if hasattr(model, 'save'):
                        torch.jit.save(model, platform_path)
                    else:
                        torch.save(model.state_dict(), platform_path)
                    platform_exports[platform] = platform_path
                    
            except Exception as e:
                logger.warning(f"Platform export hatası ({platform}): {e}")
        
        return platform_exports
    
    def _generate_optimization_report(self, results: Dict, output_dir: Path) -> None:
        """
        Optimizasyon raporu oluşturur.
        
        Args:
            results: Optimizasyon sonuçları
            output_dir: Çıktı dizini
        """
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Markdown report
        report_content = self._create_markdown_report(results)
        with open(reports_dir / "optimization_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # JSON metrics
        import json
        with open(reports_dir / "optimization_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Optimizasyon raporları oluşturuldu: {reports_dir}")
    
    def _create_markdown_report(self, results: Dict) -> str:
        """Markdown optimizasyon raporu oluşturur."""
        
        original = results.get('original_metrics', {})
        final = results.get('final_metrics', {})
        comparison = results.get('final_comparison', {})
        success = results.get('success_evaluation', {})
        
        lines = [
            "# M³TM Model Optimization Report",
            "=" * 50,
            "",
            "## Executive Summary",
            "",
            f"**Optimization Duration:** {results.get('total_optimization_time_s', 'N/A'):.1f} seconds",
            f"**Overall Success:** {'✅ Başarılı' if success.get('overall_success', False) else '❌ Kısmi Başarı'}",
            f"**Success Score:** {success.get('success_score', 0):.1f}/1.0",
            "",
            "## Model Metrics Comparison",
            "",
            "| Metric | Original | Optimized | Improvement |",
            "|--------|----------|-----------|-------------|",
            f"| Model Size (MB) | {original.get('model_size_mb', 'N/A'):.2f} | {final.get('model_size_mb', 'N/A'):.2f} | {comparison.get('size_reduction_percent', 'N/A'):.1f}% |",
            f"| Parameters | {original.get('param_count', 'N/A'):,} | {final.get('param_count', 'N/A'):,} | {comparison.get('param_reduction_percent', 'N/A'):.1f}% |",
            f"| Inference Time (ms) | {original.get('avg_inference_time_ms', 'N/A'):.2f} | {final.get('avg_inference_time_ms', 'N/A'):.2f} | {comparison.get('speedup_ratio', 'N/A'):.1f}x |",
            "",
            "## Target Achievement",
            ""
        ]
        
        # Target achievement details
        targets = success.get('targets_met', {})
        for target_name, target_info in targets.items():
            status = "✅" if target_info.get('met', False) else "❌"
            lines.append(f"- **{target_name.replace('_', ' ').title()}:** {status}")
            lines.append(f"  - Target: {target_info.get('target', 'N/A')}")
            lines.append(f"  - Achieved: {target_info.get('achieved', 'N/A'):.3f}")
            lines.append("")
        
        # Optimization steps
        lines.extend([
            "## Optimization Steps Applied",
            ""
        ])
        
        for i, step in enumerate(results.get('optimization_steps', []), 1):
            step_name = step.get('step', 'Unknown')
            lines.append(f"{i}. **{step_name.replace('_', ' ').title()}**")
            lines.append("")
        
        # Platform exports
        platform_exports = results.get('platform_exports', {})
        if platform_exports:
            lines.extend([
                "## Platform Exports",
                ""
            ])
            
            for platform, path in platform_exports.items():
                lines.append(f"- **{platform.capitalize()}:** `{Path(path).name}`")
            
            lines.append("")
        
        return "\n".join(lines)


def create_optimization_pipeline(target_size_reduction: float = 0.6,
                                 target_speed_improvement: float = 2.0,
                                 optimization_techniques: Optional[List[str]] = None) -> OptimizationPipeline:
    """
    Optimization pipeline oluşturucu fonksiyon.
    
    Args:
        target_size_reduction: Hedef boyut azaltımı
        target_speed_improvement: Hedef hız iyileştirmesi
        optimization_techniques: Kullanılacak teknikler
        
    Returns:
        OptimizationPipeline instance
    """
    config = OptimizationConfig(
        target_size_reduction=target_size_reduction,
        target_speed_improvement=target_speed_improvement,
        optimization_techniques=optimization_techniques
    )
    
    return OptimizationPipeline(config)
