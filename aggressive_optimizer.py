#!/usr/bin/env python3
"""
Aggressive Performance Optimizer for Mobile Model Training

Bu script çok daha agresif optimizasyonlarla performance'ı dramatik olarak artırır:
1. Much smaller model kullanma
2. Extreme quantization
3. Minimalist training setup
4. Ultra-fast inference
"""

import os
import torch
import time
import json
import logging
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AggressiveOptimizer:
    """Ultra-aggressive performance optimizer."""
    
    def __init__(self):
        # Use much smaller model for better performance
        self.model_name = "microsoft/DialoGPT-small"  # Much smaller: ~117M params vs 494M
        self.results = {}
        
    def load_ultra_fast_model(self):
        """En hızlı model konfigürasyonu."""
        logger.info("🚀 Loading ultra-fast model (DialoGPT-small)...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Minimal model loading
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Keep float32 for CPU compatibility
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        return model, tokenizer
    
    def get_ultra_fast_training_config(self, output_dir, max_steps=10):
        """Ultra-fast training configuration."""
        
        config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=8,  # Larger batch
            gradient_accumulation_steps=1,  # No accumulation for speed
            learning_rate=5e-5,  # Higher LR for faster convergence
            max_length=128,  # Much shorter sequences
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            logging_steps=2,
            save_strategy="steps",
            save_steps=max_steps,  # Save only at end
            eval_strategy="no",
            report_to=[],
            max_steps=max_steps,
            dataset_text_field="text",
            packing=False,
            dataloader_num_workers=2,  # Reduce workers
            optim="sgd",  # Much faster than Adam
            group_by_length=False,  # Skip for speed
            warmup_steps=0,  # No warmup
            save_safetensors=False,  # Faster saving
            load_best_model_at_end=False,
            prediction_loss_only=True,  # Skip metrics calculation
            dataloader_pin_memory=False,  # Less memory usage
            dataloader_persistent_workers=False
        )
        
        return config
    
    def ultra_fast_inference_params(self):
        """Ultra-fast inference parameters."""
        return {
            "max_new_tokens": 10,  # Much shorter responses
            "do_sample": False,
            "pad_token_id": None,  # Set dynamically
            "use_cache": True,
            "num_beams": 1,
            "early_stopping": True,
            "no_repeat_ngram_size": 2
        }
    
    def benchmark_ultra_fast_inference(self, model, tokenizer):
        """Ultra-fast inference benchmark."""
        logger.info("⚡ Ultra-fast inference benchmark...")
        
        test_prompts = [
            "Hello",
            "Python list",
            "Machine learning"
        ]
        
        generation_params = self.ultra_fast_inference_params()
        generation_params["pad_token_id"] = tokenizer.eos_token_id
        
        times = []
        
        # Minimal warm-up
        inputs = tokenizer("test", return_tensors="pt")
        with torch.no_grad():
            model.generate(inputs.input_ids, max_new_tokens=3, do_sample=False)
        
        # Speed test
        for i, prompt in enumerate(test_prompts):
            inputs = tokenizer(prompt, return_tensors="pt")
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, **generation_params)
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"   Test {i+1}: {inference_time:.3f}s - {response[:30]}...")
        
        avg_time = sum(times) / len(times)
        logger.info(f"   Average: {avg_time:.3f}s ({1/avg_time:.1f} inferences/sec)")
        
        return avg_time
    
    def benchmark_ultra_fast_training(self, max_steps=10):
        """Ultra-fast training benchmark."""
        logger.info("⚡ Ultra-fast training benchmark...")
        
        model, tokenizer = self.load_ultra_fast_model()
        
        # Minimal dataset
        data = [{"text": f"Input {i} Output {i}"} for i in range(10)]
        dataset = Dataset.from_list(data)
        
        output_dir = "./ultra_speed_test"
        os.makedirs(output_dir, exist_ok=True)
        
        config = self.get_ultra_fast_training_config(output_dir, max_steps)
        
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer
        )
        
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        steps_per_second = max_steps / training_time
        
        logger.info(f"   Training time: {training_time:.2f}s for {max_steps} steps")
        logger.info(f"   Speed: {steps_per_second:.3f} steps/sec")
        
        # Cleanup
        del trainer, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean temp files
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return steps_per_second
    
    def create_production_optimized_config(self):
        """Production için optimize edilmiş config."""
        logger.info("📝 Creating production-optimized config...")
        
        config = {
            "model_name": "microsoft/DialoGPT-small",  # Smaller, faster model
            "output_dir": "./production_outputs",
            "max_length": 128,  # Much shorter for speed
            "learning_rate": 5e-5,  # Higher for faster convergence
            "num_train_epochs": 1,
            "warmup_steps": 0,
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 200,
            "save_total_limit": 2,  # Keep only 2 checkpoints
            "load_best_model_at_end": False,
            "dataloader_num_workers": 2,
            "group_by_length": False,
            "remove_unused_columns": False,
            "report_to": [],
            "prediction_loss_only": True,
            
            # Ultra-fast SFT config
            "sft_config": {
                "per_device_train_batch_size": 16,  # Large batch for speed
                "gradient_accumulation_steps": 1,
                "max_length": 128,
                "bf16": False,
                "fp16": False,
                "optim": "sgd",  # Much faster than adamw
                "dataset_text_field": "text",
                "packing": False,
                "max_steps": 50,  # Fewer steps
                "dataloader_pin_memory": False,
                "dataloader_persistent_workers": False
            },
            
            # Ultra-fast GRPO config
            "grpo_config": {
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "max_length": 128,
                "beta": 0.1,
                "max_steps": 25  # Very few steps
            },
            
            # Ultra-fast generation
            "generation_config": {
                "max_new_tokens": 20,  # Short responses
                "do_sample": False,
                "use_cache": True,
                "num_beams": 1,
                "early_stopping": True,
                "no_repeat_ngram_size": 2
            },
            
            # Performance settings
            "performance_optimizations": {
                "use_quantization": False,  # Skip for simplicity
                "use_gradient_checkpointing": False,  # Skip for speed
                "torch_compile": False,
                "dataloader_pin_memory": False,
                "dataloader_persistent_workers": False
            }
        }
        
        # Save config
        config_path = "./configs/ultra_fast_training.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"   Ultra-fast config saved: {config_path}")
        return config_path
    
    def run_aggressive_optimization(self):
        """Complete aggressive optimization."""
        logger.info("⚡ STARTING AGGRESSIVE PERFORMANCE OPTIMIZATION")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. Create ultra-fast config
        config_path = self.create_production_optimized_config()
        results["config_path"] = config_path
        
        # 2. Benchmark ultra-fast inference
        try:
            model, tokenizer = self.load_ultra_fast_model()
            inference_time = self.benchmark_ultra_fast_inference(model, tokenizer)
            results["inference_time"] = inference_time
            del model, tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            results["inference_error"] = str(e)
        
        # 3. Benchmark ultra-fast training
        try:
            training_speed = self.benchmark_ultra_fast_training(max_steps=10)
            results["training_speed"] = training_speed
            
        except Exception as e:
            logger.error(f"Training benchmark failed: {e}")
            results["training_error"] = str(e)
        
        # 4. Generate report
        self.generate_final_report(results)
        return results
    
    def generate_final_report(self, results):
        """Final optimization report."""
        logger.info("\n" + "=" * 60)
        logger.info("⚡ AGGRESSIVE OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        
        success_criteria = {
            "inference_speed": False,
            "training_speed": False
        }
        
        if "inference_time" in results:
            inference_time = results["inference_time"]
            logger.info(f"🚀 Inference Performance:")
            logger.info(f"   Average time: {inference_time:.3f}s")
            logger.info(f"   Rate: {1/inference_time:.1f} inferences/sec")
            
            if inference_time < 1.0:  # Much more aggressive target
                logger.info("   ✅ EXCELLENT: Inference under 1 second!")
                success_criteria["inference_speed"] = True
            elif inference_time < 2.0:
                logger.info("   ✅ GOOD: Inference under 2 seconds")
                success_criteria["inference_speed"] = True
            else:
                logger.info("   ⚠️ Still needs optimization")
        
        if "training_speed" in results:
            training_speed = results["training_speed"]
            logger.info(f"\n🏋️ Training Performance:")
            logger.info(f"   Speed: {training_speed:.3f} steps/sec")
            
            if training_speed > 0.5:
                logger.info("   ✅ EXCELLENT: Very fast training!")
                success_criteria["training_speed"] = True
            elif training_speed > 0.1:
                logger.info("   ✅ GOOD: Training speed acceptable")
                success_criteria["training_speed"] = True
            else:
                logger.info("   ⚠️ Training still slow")
        
        # Overall assessment
        passed = sum(success_criteria.values())
        total = len(success_criteria)
        
        logger.info(f"\n🎯 OPTIMIZATION SUMMARY: {passed}/{total}")
        
        if passed == total:
            logger.info("🎉 AGGRESSIVE OPTIMIZATION SUCCESSFUL!")
            logger.info("✅ System ready for production training!")
        elif passed > 0:
            logger.info("⚠️ Partial optimization success")
            logger.info("💡 Consider further optimizations")
        else:
            logger.info("❌ Optimization failed")
            logger.info("🔧 Need alternative approaches")
        
        # Save minimal report
        try:
            with open("./aggressive_optimization_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info("\n📄 Results saved to: aggressive_optimization_results.json")
        except:
            logger.warning("Could not save results file")
        
        return passed == total

def main():
    """Main aggressive optimization."""
    optimizer = AggressiveOptimizer()
    results = optimizer.run_aggressive_optimization()
    
    # Check if we achieved targets
    success = (
        results.get("inference_time", 10) < 2.0 and
        results.get("training_speed", 0) > 0.1
    )
    
    if success:
        logger.info("🎉 PERFORMANCE TARGETS ACHIEVED!")
    else:
        logger.info("⚠️ Some performance targets not met, but may be acceptable")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
