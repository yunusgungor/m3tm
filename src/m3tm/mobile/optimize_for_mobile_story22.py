#!/usr/bin/env python3
"""
M³TM Mobile Optimization Script - Context7 Enhanced
Story 22: Model küçültme ve optimizasyon

Bu script Context7 documentation'dan alınan current best practices
kullanarak M³TM modelini mobil deployment için optimize eder.

Optimization Targets (Story 22):
- Model boyutu: %60+ azalma
- Inference hızı: 2x+ hızlanma  
- Memory kullanımı: %50+ azalma
- Accuracy korunması: %95+ korunma

Context7 Techniques Applied:
- QNNPACK quantization backend for mobile ARM
- Structured pruning for real speedup gains
- FX Graph Mode quantization when possible
- Progressive optimization with validation
- torch.compile for additional performance
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# M³TM imports
from m3tm.core.base_model import BaseModel
from m3tm.mobile.optimization_pipeline import (
    OptimizationPipeline, 
    Context7OptimizationConfig,
    STORY_22_OPTIMIZATION_CONFIG
)
from m3tm.mobile.benchmark_utils import ModelBenchmarker

# Try to import example data, fallback to synthetic data
try:
    from m3tm.examples.adapter_example import load_example_model_and_data
    EXAMPLE_DATA_AVAILABLE = True
except ImportError:
    EXAMPLE_DATA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MobileOptimizationRunner:
    """
    Context7-enhanced mobile optimization runner for Story 22.
    
    Implements comprehensive optimization pipeline using current
    PyTorch best practices for mobile deployment.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config: Optional[Context7OptimizationConfig] = None,
                 output_dir: str = "./optimized_models"):
        """
        Initialize mobile optimization runner.
        
        Args:
            model_path: Path to model checkpoint
            config: Optimization configuration
            output_dir: Output directory for optimized models
        """
        self.model_path = model_path
        self.config = config or STORY_22_OPTIMIZATION_CONFIG
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Context7: Initialize optimization pipeline
        self.optimization_pipeline = OptimizationPipeline(config=self.config)
        self.benchmarker = ModelBenchmarker()
        
        logger.info(f"MobileOptimizationRunner initialized")
        logger.info(f"Targets: size reduction {self.config.target_size_reduction:.0%}, "
                   f"speed improvement {self.config.target_speed_improvement}x, "
                   f"memory reduction {self.config.target_memory_reduction:.0%}")

    def load_model_and_data(self) -> tuple:
        """
        Load model and prepare datasets for optimization.
        
        Returns:
            Tuple of (model, train_dataloader, val_dataloader, validation_fn)
        """
        logger.info("Loading model and preparing data...")
        
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load from checkpoint
                logger.info(f"Loading model from {self.model_path}")
                model = torch.load(self.model_path, map_location='cpu')
            else:
                # Load example model or create synthetic model
                logger.info("Creating example M³TM model...")
                if EXAMPLE_DATA_AVAILABLE:
                    model, example_data = load_example_model_and_data()
                else:
                    # Create a simple synthetic model for demonstration
                    model = nn.Sequential(
                        nn.Linear(768, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10)
                    )
                
            # Prepare example datasets for optimization
            # In real usage, use your actual training/validation data
            example_inputs = torch.randn(100, 768)  # Example text embeddings
            example_labels = torch.randint(0, 10, (100,))  # Example labels
            
            train_dataset = TensorDataset(example_inputs[:80], example_labels[:80])
            val_dataset = TensorDataset(example_inputs[80:], example_labels[80:])
            
            train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=16)
            
            # Example validation function
            def validation_fn(model):
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_dataloader:
                        outputs = model(inputs)
                        if outputs.dim() > 1:
                            _, predicted = torch.max(outputs.data, 1)
                        else:
                            predicted = (outputs > 0.5).long()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                return correct / total if total > 0 else 0.0
            
            logger.info(f"Model loaded successfully: {type(model).__name__}")
            logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
            
            return model, train_dataloader, val_dataloader, validation_fn
            
        except Exception as e:
            logger.error(f"Failed to load model and data: {e}")
            raise

    def run_optimization(self) -> Dict[str, Any]:
        """
        Run comprehensive mobile optimization pipeline.
        
        Returns:
            Optimization results and metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Context7-Enhanced Mobile Optimization (Story 22)")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load model and data
            model, train_dataloader, val_dataloader, validation_fn = self.load_model_and_data()
            
            # Context7: Get baseline metrics
            logger.info("Measuring baseline performance...")
            baseline_metrics = self.benchmarker.get_model_metrics(model, example_inputs=torch.randn(1, 768))
            logger.info(f"Baseline - Size: {baseline_metrics.get('model_size_mb', 0):.1f}MB, "
                       f"Parameters: {baseline_metrics.get('total_parameters', 0):,}")
            
            # Context7: Run optimization pipeline
            logger.info("Running optimization pipeline...")
            optimized_model, optimization_metrics = self.optimization_pipeline.optimize_model(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                validation_fn=validation_fn,
                save_path=str(self.output_dir / "optimized_model")
            )
            
            # Context7: Analyze results
            results = self._analyze_optimization_results(
                baseline_metrics, optimization_metrics, time.time() - start_time
            )
            
            # Save detailed results
            self._save_optimization_report(results)
            
            logger.info("=" * 60)
            logger.info("Mobile Optimization Completed Successfully!")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {"error": str(e)}

    def _analyze_optimization_results(self, 
                                      baseline_metrics: Dict[str, Any],
                                      optimization_metrics: Dict[str, Any],
                                      total_time: float) -> Dict[str, Any]:
        """
        Analyze and report optimization results against Story 22 targets.
        
        Args:
            baseline_metrics: Baseline performance metrics
            optimization_metrics: Optimization results
            total_time: Total optimization time
            
        Returns:
            Comprehensive analysis results
        """
        logger.info("Analyzing optimization results...")
        
        try:
            # Extract key metrics
            size_reduction = optimization_metrics.get('size_reduction_percent', 0) / 100
            speed_improvement = optimization_metrics.get('speed_improvement', 1.0)
            memory_reduction = optimization_metrics.get('memory_reduction_percent', 0) / 100
            accuracy_loss = optimization_metrics.get('accuracy_loss', 0)
            
            # Story 22 target validation
            targets_achieved = {
                'size_reduction': {
                    'achieved': size_reduction,
                    'target': self.config.target_size_reduction,
                    'success': size_reduction >= self.config.target_size_reduction
                },
                'speed_improvement': {
                    'achieved': speed_improvement,
                    'target': self.config.target_speed_improvement,
                    'success': speed_improvement >= self.config.target_speed_improvement
                },
                'memory_reduction': {
                    'achieved': memory_reduction,
                    'target': self.config.target_memory_reduction,
                    'success': memory_reduction >= self.config.target_memory_reduction
                },
                'accuracy_preservation': {
                    'achieved': 1.0 - accuracy_loss,
                    'target': 1.0 - self.config.max_accuracy_loss,
                    'success': accuracy_loss <= self.config.max_accuracy_loss
                }
            }
            
            # Overall success rate
            success_count = sum(1 for target in targets_achieved.values() if target['success'])
            overall_success_rate = success_count / len(targets_achieved)
            
            # Detailed analysis
            analysis = {
                'story_22_targets': targets_achieved,
                'overall_success_rate': overall_success_rate,
                'optimization_time_seconds': total_time,
                'techniques_applied': optimization_metrics.get('optimization_techniques_applied', []),
                'baseline_metrics': baseline_metrics,
                'final_metrics': optimization_metrics,
                'context7_enhanced': True
            }
            
            # Log results
            logger.info("=" * 50)
            logger.info("STORY 22 TARGET ANALYSIS")
            logger.info("=" * 50)
            
            for target_name, target_data in targets_achieved.items():
                status = "✅ ACHIEVED" if target_data['success'] else "❌ MISSED"
                logger.info(f"{target_name.upper()}: {status}")
                logger.info(f"  Target: {target_data['target']:.2%}")
                logger.info(f"  Achieved: {target_data['achieved']:.2%}")
                
            logger.info(f"\nOverall Success Rate: {overall_success_rate:.1%}")
            logger.info(f"Optimization Time: {total_time:.1f} seconds")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze results: {e}")
            return {"error": str(e)}

    def _save_optimization_report(self, results: Dict[str, Any]) -> bool:
        """
        Save comprehensive optimization report.
        
        Args:
            results: Optimization analysis results
            
        Returns:
            Success status
        """
        try:
            import json
            from datetime import datetime
            
            # Add timestamp and metadata
            report = {
                'timestamp': datetime.now().isoformat(),
                'story_id': 'story_22',
                'optimization_type': 'mobile_optimization',
                'context7_enhanced': True,
                'pytorch_version': torch.__version__,
                **results
            }
            
            # Save JSON report
            report_path = self.output_dir / "optimization_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            # Save summary text report
            summary_path = self.output_dir / "optimization_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("M³TM MOBILE OPTIMIZATION REPORT - STORY 22\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Optimization completed at: {report['timestamp']}\n")
                f.write(f"Total time: {results.get('optimization_time_seconds', 0):.1f} seconds\n\n")
                
                f.write("TARGET ACHIEVEMENTS:\n")
                f.write("-" * 20 + "\n")
                for target_name, target_data in results.get('story_22_targets', {}).items():
                    status = "ACHIEVED" if target_data['success'] else "MISSED"
                    f.write(f"{target_name}: {status} ({target_data['achieved']:.2%})\n")
                    
                f.write(f"\nOverall Success Rate: {results.get('overall_success_rate', 0):.1%}\n")
                
            logger.info(f"Optimization report saved to {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save optimization report: {e}")
            return False


def main():
    """
    Main execution function for Story 22 mobile optimization.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="M³TM Mobile Optimization - Story 22")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./optimized_models", 
                       help="Output directory for optimized models")
    parser.add_argument("--config", type=str, choices=["story22", "aggressive", "conservative"],
                       default="story22", help="Optimization configuration preset")
    parser.add_argument("--techniques", nargs="+", 
                       choices=["quantization", "pruning", "distillation", "torchscript"],
                       help="Specific optimization techniques to apply")
    
    args = parser.parse_args()
    
    # Configure optimization based on arguments
    if args.config == "story22":
        config = STORY_22_OPTIMIZATION_CONFIG
    elif args.config == "aggressive":
        config = Context7OptimizationConfig(
            target_size_reduction=0.8,
            target_speed_improvement=3.0,
            target_memory_reduction=0.7,
            max_accuracy_loss=0.1
        )
    else:  # conservative
        config = Context7OptimizationConfig(
            target_size_reduction=0.3,
            target_speed_improvement=1.5,
            target_memory_reduction=0.3,
            max_accuracy_loss=0.02
        )
    
    # Override techniques if specified
    if args.techniques:
        config.optimization_techniques = args.techniques
    
    try:
        # Run optimization
        runner = MobileOptimizationRunner(
            model_path=args.model_path,
            config=config,
            output_dir=args.output_dir
        )
        
        results = runner.run_optimization()
        
        # Exit with appropriate code
        if "error" in results:
            sys.exit(1)
        elif results.get('overall_success_rate', 0) >= 0.75:  # 75% target achievement
            logger.info("🎉 Optimization highly successful!")
            sys.exit(0)
        else:
            logger.warning("⚠️ Optimization partially successful")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
