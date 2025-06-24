"""
M³TM Model Optimization Pipeline - Context7 Enhanced

Bu modül tüm optimizasyon tekniklerini birleştiren ana pipeline'ı sağlar.
Context7 documentation'dan alınan current best practices uygulanmıştır.

Pipeline Features (Context7 Based):
- Multi-technique optimization orchestration
- Progressive optimization with validation checkpoints
- Mobile-first optimization strategies
- Comprehensive benchmarking and validation
- Platform-specific optimization paths
- State-of-the-art PyTorch optimization APIs

Supported Optimization Stack:
- Quantization (PTDQ, PTSQ, QAT, FX Graph Mode)
- Pruning (Structured, Unstructured, Progressive)
- Knowledge Distillation (Teacher-Student)
- TorchScript Compilation
- Module Fusion and Graph Optimization
"""

import logging
from typing import Dict, Optional, Union, Any, Tuple, List, Callable
from pathlib import Path
import time
import warnings
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

# Context7: Import latest optimization modules
from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker, create_benchmarker
from m3tm.mobile.quantization import QuantizationManager, create_quantization_pipeline, MOBILE_QUANTIZATION_CONFIG
from m3tm.mobile.pruning import PruningManager, create_pruning_pipeline, MOBILE_PRUNING_CONFIG
from m3tm.mobile.knowledge_distillation import DistillationTrainer, create_distillation_pipeline
from m3tm.mobile.torchscript_converter import TorchScriptConverter, create_torchscript_pipeline

# S26: Advanced mobile optimization modules (Context7 enhanced)
from m3tm.mobile.advanced_cache import AdvancedCacheManager, CacheConfig
from m3tm.mobile.advanced_profiler import AdvancedPerformanceProfiler, ProfilingConfig
from m3tm.mobile.hardware_acceleration import HardwareAccelerationManager, AccelerationConfig

logger = logging.getLogger(__name__)


class Context7OptimizationConfig:
    """
    Context7-enhanced optimization pipeline configuration.
    
    Implements mobile-first optimization strategies based on current
    PyTorch best practices from official documentation.
    S26 Enhanced: Advanced caching, profiling, and hardware acceleration.
    """
    
    def __init__(self,
                 # Story 22 targets from Context7 analysis
                 target_size_reduction: float = 0.6,  # 60%+ reduction
                 target_speed_improvement: float = 2.0,  # 2x+ speedup
                 target_memory_reduction: float = 0.5,  # 50%+ memory reduction
                 max_accuracy_loss: float = 0.05,  # <5% accuracy loss
                 
                 # Context7: Mobile optimization strategy
                 optimization_techniques: Optional[List[str]] = None,
                 progressive_optimization: bool = True,
                 platform_targets: Optional[List[str]] = None,
                 
                 # Context7: Advanced configuration
                 quantization_backend: str = "qnnpack",  # Mobile ARM optimal
                 enable_fx_quantization: bool = True,
                 enable_structured_pruning: bool = True,
                 enable_knowledge_distillation: bool = True,
                 enable_torch_compile: bool = True,
                 
                 # S26: Advanced mobile optimization features
                 enable_advanced_caching: bool = True,
                 enable_advanced_profiling: bool = True,
                 enable_hardware_acceleration: bool = True,
                 cache_strategy: str = "multi_level",  # multi_level, memory_only, persistent
                 profiling_level: str = "comprehensive",  # basic, standard, comprehensive
                 hardware_targets: Optional[List[str]] = None,  # ["nnapi", "coreml", "gpu"]
                 
                 # Context7: Validation and safety
                 validation_frequency: int = 1,  # Validate after each technique
                 early_stopping_threshold: float = 0.02,  # Stop if accuracy drops too much
                 checkpoint_enabled: bool = True):
        """
        Context7-enhanced optimization configuration.
        S26 Enhanced: Advanced caching, profiling, and hardware acceleration.
        
        Args:
            target_size_reduction: Hedef boyut azaltımı (0.0-1.0)
            target_speed_improvement: Hedef hız iyileştirmesi (kat)
            target_memory_reduction: Hedef hafıza azaltımı (0.0-1.0)
            max_accuracy_loss: Maksimum kabul edilebilir doğruluk kaybı
            optimization_techniques: Kullanılacak teknikler ['quantization', 'pruning', 'distillation', 'torchscript']
            progressive_optimization: Aşamalı optimizasyon kullan
            platform_targets: Hedef platformlar ['android', 'ios', 'desktop']
            quantization_backend: Quantization backend ('qnnpack', 'x86')
            enable_fx_quantization: FX Graph Mode quantization kullan
            enable_structured_pruning: Structured pruning kullan (real speedup)
            enable_knowledge_distillation: Knowledge distillation kullan
            enable_torch_compile: torch.compile optimization kullan
            enable_advanced_caching: S26 Advanced caching system
            enable_advanced_profiling: S26 Comprehensive profiling
            enable_hardware_acceleration: S26 Hardware acceleration
            cache_strategy: Cache stratejisi
            profiling_level: Profiling detay seviyesi
            hardware_targets: Donanım hedefleri
            validation_frequency: Validation sıklığı
            early_stopping_threshold: Erken durdurma eşiği
            checkpoint_enabled: Checkpoint kaydetme aktif
        """
        self.target_size_reduction = target_size_reduction
        self.target_speed_improvement = target_speed_improvement
        self.target_memory_reduction = target_memory_reduction
        self.max_accuracy_loss = max_accuracy_loss
        
        # Context7: Default mobile optimization stack
        if optimization_techniques is None:
            optimization_techniques = ["quantization", "pruning", "torchscript"]
            if enable_knowledge_distillation:
                optimization_techniques.insert(-1, "distillation")
                
        self.optimization_techniques = optimization_techniques
        self.progressive_optimization = progressive_optimization
        self.platform_targets = platform_targets or ["android", "ios"]
        
        # Context7: Advanced settings
        self.quantization_backend = quantization_backend
        self.enable_fx_quantization = enable_fx_quantization
        self.enable_structured_pruning = enable_structured_pruning
        self.enable_knowledge_distillation = enable_knowledge_distillation
        self.enable_torch_compile = enable_torch_compile
        
        # S26: Advanced mobile optimization settings
        self.enable_advanced_caching = enable_advanced_caching
        self.enable_advanced_profiling = enable_advanced_profiling
        self.enable_hardware_acceleration = enable_hardware_acceleration
        self.cache_strategy = cache_strategy
        self.profiling_level = profiling_level
        self.hardware_targets = hardware_targets or ["nnapi", "coreml", "gpu"]
        
        # Context7: Validation and safety
        self.validation_frequency = validation_frequency
        self.early_stopping_threshold = early_stopping_threshold
        self.checkpoint_enabled = checkpoint_enabled
        
        logger.info(f"Context7 optimization config: techniques={optimization_techniques}, "
                   f"targets=size:{target_size_reduction:.0%}, speed:{target_speed_improvement}x, "
                   f"memory:{target_memory_reduction:.0%}")
        logger.info(f"S26 advanced features: cache={enable_advanced_caching}, "
                   f"profile={enable_advanced_profiling}, hw_accel={enable_hardware_acceleration}")


class OptimizationPipeline:
    """
    M³TM Model Optimization Pipeline - Context7 Enhanced
    
    Context7 documentation'dan alınan best practices ile geliştirilmiş
    comprehensive optimization pipeline. Mobile deployment için optimize edilmiştir.
    
    S26 Enhanced Features:
    - Advanced multi-level persistent caching system
    - Comprehensive device profiling with battery/thermal monitoring
    - Hardware acceleration layer (NNAPI/CoreML/GPU)
    - Real-time performance analytics and regression detection
    
    Features:
    - Multi-technique orchestration with optimal ordering
    - Progressive optimization with safety checkpoints
    - Mobile-first platform optimization
    - State-of-the-art PyTorch APIs integration
    - Comprehensive benchmarking and validation
    """

    def __init__(self,
                 config: Optional[Context7OptimizationConfig] = None,
                 benchmarker: Optional[ModelBenchmarker] = None):
        """
        Context7-enhanced OptimizationPipeline initialization.
        S26 Enhanced: Advanced caching, profiling, and hardware acceleration.
        
        Args:
            config: Optimization configuration
            benchmarker: Performance measurement tool
        """
        self.config = config or Context7OptimizationConfig()
        self.benchmarker = benchmarker or create_benchmarker()
        
        # Context7: Initialize optimization managers with current best practices
        self.quantization_manager = create_quantization_pipeline(
            backend=self.config.quantization_backend
        )
        self.pruning_manager = create_pruning_pipeline(
            enable_experimental=self.config.enable_structured_pruning
        )
        self.distillation_trainer = create_distillation_pipeline()
        self.torchscript_converter = create_torchscript_pipeline()
        
        # S26: Initialize advanced mobile optimization modules
        self.advanced_cache = None
        self.advanced_profiler = None
        self.hardware_accelerator = None
        
        if self.config.enable_advanced_caching:
            cache_config = CacheConfig(
                strategy=self.config.cache_strategy,
                max_cache_size_mb=512,  # 512MB cache
                enable_persistent=True,
                enable_analytics=True
            )
            self.advanced_cache = AdvancedCacheManager(cache_config)
            logger.info("S26 Advanced caching initialized")
        
        if self.config.enable_advanced_profiling:
            profiling_config = ProfilingConfig(
                level=self.config.profiling_level,
                enable_battery_monitoring=True,
                enable_thermal_monitoring=True,
                enable_regression_detection=True,
                enable_device_analytics=True
            )
            self.advanced_profiler = AdvancedPerformanceProfiler()
            logger.info("S26 Advanced profiling initialized")
        
        if self.config.enable_hardware_acceleration:
            accel_config = AccelerationConfig(
                target_platforms=self.config.hardware_targets,
                enable_fallback=True,
                auto_select_optimal=True
            )
            self.hardware_accelerator = HardwareAccelerationManager(accel_config)
            logger.info("S26 Hardware acceleration initialized")
        
        # Context7: Track optimization state
        self.optimization_history = []
        self.checkpoints = {}
        self.current_metrics = {}
        
        logger.info(f"OptimizationPipeline initialized with Context7 config")

    def optimize_model(self,
                       model: Union[nn.Module, BaseModel],
                       train_dataloader: Optional[DataLoader] = None,
                       val_dataloader: Optional[DataLoader] = None,
                       validation_fn: Optional[Callable] = None,
                       save_path: Optional[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Context7-enhanced comprehensive model optimization.
        
        Implements state-of-the-art optimization pipeline based on
        current PyTorch best practices for mobile deployment.
        
        Args:
            model: Optimize edilecek model
            train_dataloader: Training data (distillation ve QAT için)
            val_dataloader: Validation data
            validation_fn: Custom validation function
            save_path: Model kaydetme yolu
            
        Returns:
            Tuple[optimized_model, comprehensive_metrics]
        """
        logger.info("Starting Context7-enhanced model optimization pipeline")
        
        # S26: Advanced initialization - cache, profiling, hardware acceleration
        if self.advanced_cache:
            # Check cache for previously optimized model
            cache_key = self.advanced_cache.generate_cache_key(model, self.config)
            cached_result = self.advanced_cache.get_cached_result(cache_key)
            if cached_result:
                logger.info("S26: Found cached optimized model, returning cached result")
                return cached_result['model'], cached_result['metrics']
                
        if self.advanced_profiler:
            # Start comprehensive profiling
            self.advanced_profiler.start_profiling_session(model)
            logger.info("S26: Advanced profiling session started")
            
        if self.hardware_accelerator:
            # Initialize hardware acceleration
            self.hardware_accelerator.initialize_for_model(model)
            logger.info("S26: Hardware acceleration initialized")
        
        # Context7: Initial benchmarking
        original_model = copy.deepcopy(model)
        current_model = model
        
        # Baseline metrics
        baseline_metrics = self._get_baseline_metrics(original_model)
        self.current_metrics = baseline_metrics.copy()
        
        # S26: Enhanced baseline metrics with advanced profiling
        if self.advanced_profiler:
            advanced_baseline = self.advanced_profiler.profile_model_inference(original_model)
            baseline_metrics.update(advanced_baseline)
        
        logger.info(f"Baseline metrics: size={baseline_metrics.get('model_size_mb', 0):.1f}MB, "
                   f"params={baseline_metrics.get('total_parameters', 0):,}")
        if self.advanced_profiler:
            logger.info(f"S26 Advanced baseline: battery={baseline_metrics.get('battery_impact', 'N/A')}, "
                       f"thermal={baseline_metrics.get('thermal_impact', 'N/A')}")
        
        try:
            # Context7: Execute optimization techniques in optimal order
            for technique in self.config.optimization_techniques:
                logger.info(f"Applying optimization technique: {technique}")
                
                # Create checkpoint before technique
                if self.config.checkpoint_enabled:
                    self.checkpoints[f"before_{technique}"] = copy.deepcopy(current_model)
                
                # Apply technique
                if technique == "quantization":
                    current_model, technique_metrics = self._apply_quantization(
                        current_model, train_dataloader, val_dataloader
                    )
                elif technique == "pruning":
                    current_model, technique_metrics = self._apply_pruning(
                        current_model, validation_fn
                    )
                elif technique == "distillation":
                    current_model, technique_metrics = self._apply_knowledge_distillation(
                        current_model, train_dataloader, val_dataloader
                    )
                elif technique == "torchscript":
                    current_model, technique_metrics = self._apply_torchscript_optimization(
                        current_model
                    )
                else:
                    logger.warning(f"Unknown optimization technique: {technique}")
                    continue
                
                # S26: Advanced profiling after each technique
                if self.advanced_profiler:
                    step_profiling = self.advanced_profiler.profile_model_inference(current_model)
                    technique_metrics.update({f"{technique}_profiling": step_profiling})
                    
                    # Check for performance regression at each step
                    if len(self.optimization_history) > 0:
                        previous_metrics = self.optimization_history[-1]['metrics']
                        regression_check = self.advanced_profiler.detect_performance_regression(
                            previous_metrics, technique_metrics
                        )
                        technique_metrics[f"{technique}_regression_check"] = regression_check
                
                # Context7: Validate after each technique
                if self.config.validation_frequency > 0:
                    validation_metrics = self._validate_optimization_step(
                        original_model, current_model, technique, validation_fn
                    )
                    technique_metrics.update(validation_metrics)
                    
                    # Early stopping check
                    accuracy_loss = validation_metrics.get('accuracy_loss', 0)
                    if accuracy_loss > self.config.early_stopping_threshold:
                        logger.warning(f"Early stopping triggered: accuracy loss {accuracy_loss:.3f} > {self.config.early_stopping_threshold:.3f}")
                        # Revert to previous checkpoint
                        if f"before_{technique}" in self.checkpoints:
                            current_model = self.checkpoints[f"before_{technique}"]
                        break
                
                # Record technique results
                self.optimization_history.append({
                    "technique": technique,
                    "metrics": technique_metrics,
                    "timestamp": time.time()
                })
                
                logger.info(f"Technique {technique} completed: {technique_metrics}")
                
            # Context7: Final comprehensive evaluation
            final_metrics = self._get_final_metrics(original_model, current_model)
            
            # S26: Enhanced final evaluation with advanced modules
            if self.advanced_profiler:
                # Final profiling and regression detection
                final_profiling = self.advanced_profiler.profile_model_inference(current_model)
                regression_analysis = self.advanced_profiler.detect_performance_regression(
                    baseline_metrics, final_profiling
                )
                final_metrics.update(final_profiling)
                final_metrics['regression_analysis'] = regression_analysis
                
                # Stop profiling session
                profiling_summary = self.advanced_profiler.stop_profiling_session()
                final_metrics['profiling_summary'] = profiling_summary
                
            if self.hardware_accelerator:
                # Apply hardware acceleration to optimized model
                try:
                    accelerated_model = self.hardware_accelerator.accelerate_model(current_model)
                    acceleration_metrics = self.hardware_accelerator.benchmark_acceleration(
                        current_model, accelerated_model
                    )
                    current_model = accelerated_model
                    final_metrics['hardware_acceleration'] = acceleration_metrics
                    logger.info("S26: Hardware acceleration applied successfully")
                except Exception as e:
                    logger.warning(f"S26: Hardware acceleration failed: {e}")
                    final_metrics['hardware_acceleration'] = {"error": str(e)}
                    
            if self.advanced_cache:
                # Cache the optimized model for future use
                cache_key = self.advanced_cache.generate_cache_key(original_model, self.config)
                cache_result = {
                    'model': current_model,
                    'metrics': final_metrics,
                    'optimization_history': self.optimization_history
                }
                self.advanced_cache.cache_result(cache_key, cache_result)
                
                # Update cache analytics
                cache_analytics = self.advanced_cache.get_cache_analytics()
                final_metrics['cache_analytics'] = cache_analytics
                logger.info("S26: Optimized model cached successfully")
            
            # Context7: Apply torch.compile if enabled and supported
            if self.config.enable_torch_compile:
                try:
                    current_model = self._apply_torch_compile(current_model)
                    final_metrics['torch_compile_applied'] = True
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
                    final_metrics['torch_compile_applied'] = False
            
            # Context7: Save optimized model if path provided
            if save_path:
                self._save_optimized_model(current_model, save_path, final_metrics)
            
            # Context7: Validate targets achievement
            target_validation = self._validate_optimization_targets(final_metrics)
            final_metrics['target_validation'] = target_validation
            
            logger.info("Context7 optimization pipeline completed successfully")
            logger.info(f"Final metrics: {final_metrics}")
            
            return current_model, final_metrics
            
        except Exception as e:
            logger.error(f"Optimization pipeline failed: {e}")
            # Return original model on failure
            return original_model, {"error": str(e), "baseline_metrics": baseline_metrics}

    def _apply_quantization(self,
                            model: nn.Module,
                            train_dataloader: Optional[DataLoader],
                            val_dataloader: Optional[DataLoader]) -> Tuple[nn.Module, Dict]:
        """Context7-enhanced quantization application"""
        logger.info("Applying Context7-enhanced quantization...")
        
        try:
            # Context7: Choose optimal quantization method based on model type
            if self.config.enable_fx_quantization and hasattr(model, 'forward'):
                # Try FX Graph Mode first for better optimization
                try:
                    if train_dataloader:
                        # QAT for highest accuracy
                        return self.quantization_manager.apply_qat(
                            model, train_dataloader, val_dataloader,
                            num_epochs=3, benchmark=True
                        )
                    elif val_dataloader:
                        # Static quantization for CNNs
                        example_inputs = next(iter(val_dataloader))[0][:1]
                        return self.quantization_manager.apply_static_quantization(
                            model, val_dataloader, example_inputs, benchmark=True
                        )
                except Exception as e:
                    logger.warning(f"Advanced quantization failed, falling back to dynamic: {e}")
            
            # Context7: Fallback to dynamic quantization (recommended for transformers)
            return self.quantization_manager.apply_dynamic_quantization(
                model, benchmark=True
            )
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model, {"error": str(e)}

    def _apply_pruning(self,
                       model: nn.Module,
                       validation_fn: Optional[Callable]) -> Tuple[nn.Module, Dict]:
        """Context7-enhanced pruning application"""
        logger.info("Applying Context7-enhanced pruning...")
        
        try:
            # Context7: Progressive structured pruning for real speedup
            if self.config.enable_structured_pruning:
                return self.pruning_manager.apply_progressive_pruning(
                    model,
                    target_sparsity=MOBILE_PRUNING_CONFIG["sparsity"],
                    num_steps=MOBILE_PRUNING_CONFIG["progressive_steps"],
                    validation_fn=validation_fn,
                    accuracy_threshold=MOBILE_PRUNING_CONFIG["accuracy_threshold"],
                    method="structured",
                    benchmark=True
                )
            else:
                # Fallback to unstructured pruning
                return self.pruning_manager.apply_unstructured_pruning(
                    model, sparsity=0.3, method="l1", global_pruning=True, benchmark=True
                )
                
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model, {"error": str(e)}

    def _apply_knowledge_distillation(self,
                                      model: nn.Module,
                                      train_dataloader: Optional[DataLoader],
                                      val_dataloader: Optional[DataLoader]) -> Tuple[nn.Module, Dict]:
        """Context7-enhanced knowledge distillation application"""
        logger.info("Applying Context7-enhanced knowledge distillation...")
        
        try:
            if not train_dataloader:
                logger.warning("Knowledge distillation requires training data, skipping...")
                return model, {"skipped": "no_training_data"}
                
            # Context7: Create efficient student model (60-80% parameter reduction)
            student_model = self.distillation_trainer.create_student_model(
                model, compression_ratio=0.3  # 70% reduction
            )
            
            # Train student model with teacher guidance
            trained_student, metrics = self.distillation_trainer.train_student(
                teacher_model=model,
                student_model=student_model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=5,
                benchmark=True
            )
            
            return trained_student, metrics
            
        except Exception as e:
            logger.error(f"Knowledge distillation failed: {e}")
            return model, {"error": str(e)}

    def _apply_torchscript_optimization(self, model: nn.Module) -> Tuple[nn.Module, Dict]:
        """Context7-enhanced TorchScript optimization"""
        logger.info("Applying Context7-enhanced TorchScript optimization...")
        
        try:
            # Context7: Convert to TorchScript for mobile deployment
            scripted_model, metrics = self.torchscript_converter.convert_to_torchscript(
                model, optimization_level="mobile", benchmark=True
            )
            
            return scripted_model, metrics
            
        except Exception as e:
            logger.error(f"TorchScript optimization failed: {e}")
            return model, {"error": str(e)}

    def _apply_torch_compile(self, model: nn.Module) -> nn.Module:
        """Context7: Apply torch.compile optimization if available"""
        try:
            if hasattr(torch, 'compile'):
                # Context7: Use inductor backend for best performance
                compiled_model = torch.compile(model, backend="inductor")
                logger.info("torch.compile optimization applied")
                return compiled_model
            else:
                logger.warning("torch.compile not available in this PyTorch version")
                return model
                
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            return model

    def _get_baseline_metrics(self, model: nn.Module) -> Dict[str, Any]:
        """Context7-enhanced baseline metrics calculation"""
        try:
            return self.benchmarker.get_model_metrics(model, example_inputs=torch.randn(1, 768))
        except Exception as e:
            logger.error(f"Failed to get baseline metrics: {e}")
            return {}

    def _get_final_metrics(self, original_model: nn.Module, optimized_model: nn.Module) -> Dict[str, Any]:
        """Context7-enhanced final metrics calculation"""
        try:
            comparison_metrics = self.benchmarker.compare_models(original_model, optimized_model)
            
            # Add optimization history
            comparison_metrics['optimization_history'] = self.optimization_history
            comparison_metrics['optimization_techniques_applied'] = self.config.optimization_techniques
            
            return comparison_metrics
            
        except Exception as e:
            logger.error(f"Failed to get final metrics: {e}")
            return {}

    def _validate_optimization_step(self,
                                    original_model: nn.Module,
                                    current_model: nn.Module,
                                    technique: str,
                                    validation_fn: Optional[Callable]) -> Dict[str, Any]:
        """Context7-enhanced optimization step validation"""
        try:
            validation_metrics = {}
            
            # Size and parameter comparison
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters())
                
            original_params = count_parameters(original_model)
            current_params = count_parameters(current_model)
            param_reduction = (original_params - current_params) / original_params
            
            validation_metrics['parameter_reduction'] = param_reduction
            validation_metrics['current_parameters'] = current_params
            
            # Custom validation function
            if validation_fn:
                try:
                    original_accuracy = validation_fn(original_model)
                    current_accuracy = validation_fn(current_model)
                    accuracy_loss = original_accuracy - current_accuracy
                    
                    validation_metrics['original_accuracy'] = original_accuracy
                    validation_metrics['current_accuracy'] = current_accuracy
                    validation_metrics['accuracy_loss'] = accuracy_loss
                    
                except Exception as e:
                    logger.warning(f"Custom validation failed: {e}")
                    
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}

    def _validate_optimization_targets(self, final_metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Context7: Validate if optimization targets were achieved"""
        try:
            size_reduction = final_metrics.get('size_reduction_percent', 0) / 100
            speed_improvement = final_metrics.get('speed_improvement', 1.0)
            memory_reduction = final_metrics.get('memory_reduction_percent', 0) / 100
            accuracy_loss = final_metrics.get('accuracy_loss', 0)
            
            validation = {
                'size_target_achieved': size_reduction >= self.config.target_size_reduction,
                'speed_target_achieved': speed_improvement >= self.config.target_speed_improvement,
                'memory_target_achieved': memory_reduction >= self.config.target_memory_reduction,
                'accuracy_maintained': accuracy_loss <= self.config.max_accuracy_loss
            }
            
            validation['all_targets_achieved'] = all(validation.values())
            
            return validation
            
        except Exception as e:
            logger.error(f"Target validation failed: {e}")
            return {}

    def _save_optimized_model(self,
                              model: nn.Module,
                              save_path: str,
                              metrics: Dict[str, Any]) -> bool:
        """Context7-enhanced optimized model saving"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model in multiple formats for different platforms
            base_path = save_path.stem
            save_dir = save_path.parent
            
            # PyTorch format
            torch.save(model.state_dict(), save_dir / f"{base_path}_optimized.pth")
            
            # TorchScript format for mobile
            try:
                scripted = torch.jit.script(model)
                scripted.save(str(save_dir / f"{base_path}_mobile.pt"))
            except Exception as e:
                logger.warning(f"TorchScript save failed: {e}")
                
            # Save metrics
            import json
            with open(save_dir / f"{base_path}_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
                
            logger.info(f"Optimized model saved to {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save optimized model: {e}")
            return False


def create_optimization_pipeline(config: Optional[Context7OptimizationConfig] = None) -> OptimizationPipeline:
    """
    Context7-enhanced optimization pipeline oluşturucu.
    
    Args:
        config: Optimization configuration
        
    Returns:
        OptimizationPipeline instance
    """
    return OptimizationPipeline(config=config)


# Context7: Export Story 22 optimized configuration
STORY_22_OPTIMIZATION_CONFIG = Context7OptimizationConfig(
    target_size_reduction=0.6,  # %60+ model size reduction
    target_speed_improvement=2.0,  # 2x+ inference speedup
    target_memory_reduction=0.5,  # %50+ memory reduction
    max_accuracy_loss=0.05,  # <5% accuracy loss
    optimization_techniques=["quantization", "pruning", "distillation", "torchscript"],
    quantization_backend="qnnpack",  # Mobile ARM optimization
    enable_fx_quantization=True,
    enable_structured_pruning=True,
    enable_knowledge_distillation=True,
    enable_torch_compile=True,
    progressive_optimization=True,
    platform_targets=["android", "ios"],
    validation_frequency=1,
    early_stopping_threshold=0.02,
    checkpoint_enabled=True
)
