#!/usr/bin/env python3
"""
Performance Optimizer for Mobile Model Training

Bu script training ve inference performance'ını optimize eder:
1. Model quantization
2. Optimized batch sizes
3. Improved generation parameters
4. Memory optimization
"""

import os
import torch
import time
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Training ve inference performance optimizer."""
    
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.optimized_configs = {}
        
    def create_quantization_config(self):
        """4-bit quantization config oluştur."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_optimized_model(self, use_quantization=True):
        """Optimize edilmiş model yükle."""
        logger.info("🚀 Loading optimized model...")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model loading options
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "trust_remote_code": True
        }
        
        if use_quantization and torch.cuda.is_available():
            model_kwargs["quantization_config"] = self.create_quantization_config()
            logger.info("   Using 4-bit quantization")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Model optimization
        if hasattr(model, 'half') and torch.cuda.is_available():
            model = model.half()
        
        logger.info(f"   Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model, tokenizer
    
    def get_optimized_training_config(self, output_dir, max_steps=50):
        """Optimize edilmiş training config."""
        
        # Optimal batch size hesapla
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 8
        
        if available_memory_gb > 16:
            batch_size = 8
            gradient_accumulation_steps = 2
        elif available_memory_gb > 8:
            batch_size = 4
            gradient_accumulation_steps = 4
        else:
            batch_size = 2
            gradient_accumulation_steps = 8
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        logger.info(f"   Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"   Effective batch size: {effective_batch_size}")
        
        config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=2e-5,
            max_length=256,  # Shorter sequences for speed
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            remove_unused_columns=False,
            logging_steps=2,
            save_strategy="steps",
            save_steps=max_steps // 4,  # Save 4 times during training
            eval_strategy="no",
            report_to=[],
            max_steps=max_steps,
            dataset_text_field="text",
            packing=False,
            dataloader_num_workers=4,
            optim="adamw_torch",  # Faster optimizer
            group_by_length=True,  # Group similar lengths for efficiency
            warmup_steps=max_steps // 10,  # 10% warmup
            save_safetensors=True,
            load_best_model_at_end=False  # Skip for speed
        )
        
        return config
    
    def optimize_generation_params(self):
        """Optimize edilmiş generation parametreleri."""
        return {
            "max_new_tokens": 20,
            "do_sample": False,  # Deterministic for speed
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "pad_token_id": None,  # Will be set to tokenizer.eos_token_id
            "repetition_penalty": 1.1,
            "use_cache": True,
            "num_beams": 1  # No beam search for speed
        }
    
    def benchmark_inference_speed(self, model, tokenizer, num_tests=5):
        """Inference speed benchmark."""
        logger.info("🔄 Benchmarking inference speed...")
        
        test_prompts = [
            "Python'da liste nasıl oluşturulur?",
            "Machine learning nedir?",
            "Veri analizi nasıl yapılır?",
            "Deep learning modelleri nasıl çalışır?",
            "Programlama dillerinin avantajları nelerdir?"
        ]
        
        generation_params = self.optimize_generation_params()
        generation_params["pad_token_id"] = tokenizer.eos_token_id
        
        times = []
        
        # Warm-up
        inputs = tokenizer(test_prompts[0], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            model.generate(inputs.input_ids, max_new_tokens=5, do_sample=False)
        
        # Actual benchmark
        for i, prompt in enumerate(test_prompts[:num_tests]):
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, **generation_params)
            
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"   Test {i+1}: {inference_time:.3f}s - {response[:50]}...")
        
        avg_time = sum(times) / len(times)
        logger.info(f"   Average inference time: {avg_time:.3f}s")
        logger.info(f"   Inference rate: {1/avg_time:.1f} inferences/sec")
        
        return avg_time
    
    def benchmark_training_speed(self, max_steps=20):
        """Training speed benchmark."""
        logger.info("🔄 Benchmarking training speed...")
        
        # Load optimized model
        model, tokenizer = self.load_optimized_model(use_quantization=True)
        
        # Prepare small dataset
        data = [
            {"text": f"### İnsan: Test prompt {i}\n### Asistan: Test response {i}"}
            for i in range(20)
        ]
        dataset = Dataset.from_list(data)
        
        # Optimized training config
        output_dir = "./temp_speed_test"
        os.makedirs(output_dir, exist_ok=True)
        
        training_config = self.get_optimized_training_config(output_dir, max_steps)
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_config,
            train_dataset=dataset,
            processing_class=tokenizer
        )
        
        # Training benchmark
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        steps_per_second = max_steps / training_time
        
        logger.info(f"   Training time: {training_time:.2f}s for {max_steps} steps")
        logger.info(f"   Training speed: {steps_per_second:.3f} steps/sec")
        
        # Cleanup
        del trainer, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return steps_per_second
    
    def create_optimized_config_file(self):
        """Optimize edilmiş comprehensive_training.yaml oluştur."""
        logger.info("📝 Creating optimized configuration...")
        
        optimized_config = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": "./optimized_outputs",
            "max_length": 256,  # Shorter for speed
            "learning_rate": 2e-5,
            "num_train_epochs": 2,
            "warmup_steps": 50,
            "logging_steps": 5,
            "save_steps": 25,
            "eval_steps": 50,
            "save_total_limit": 3,
            "load_best_model_at_end": False,
            "dataloader_num_workers": 4,
            "group_by_length": True,
            "remove_unused_columns": False,
            "report_to": [],
            
            # SFT specific optimizations
            "sft_config": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "max_length": 256,
                "bf16": True,
                "fp16": False,
                "optim": "adamw_torch",
                "dataset_text_field": "text",
                "packing": False,
                "max_steps": 100
            },
            
            # GRPO optimizations
            "grpo_config": {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_length": 256,
                "beta": 0.1,
                "max_steps": 50
            },
            
            # Generation optimizations
            "generation_config": {
                "max_new_tokens": 50,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "use_cache": True,
                "num_beams": 1
            },
            
            # Performance optimizations
            "performance_optimizations": {
                "use_quantization": True,
                "quantization_bits": 4,
                "use_gradient_checkpointing": True,
                "torch_compile": False,  # May cause issues on some systems
                "dataloader_pin_memory": True,
                "dataloader_persistent_workers": True
            }
        }
        
        # Save optimized config
        config_path = "./configs/optimized_training.yaml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(optimized_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"   Optimized config saved: {config_path}")
        return config_path
    
    def run_full_optimization(self):
        """Complete performance optimization suite."""
        logger.info("🎯 STARTING PERFORMANCE OPTIMIZATION")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. Create optimized config
        config_path = self.create_optimized_config_file()
        results["optimized_config_path"] = config_path
        
        # 2. Benchmark inference speed
        try:
            model, tokenizer = self.load_optimized_model(use_quantization=False)
            baseline_inference = self.benchmark_inference_speed(model, tokenizer)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Optimized inference
            model, tokenizer = self.load_optimized_model(use_quantization=True)
            optimized_inference = self.benchmark_inference_speed(model, tokenizer)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            results["baseline_inference_time"] = baseline_inference
            results["optimized_inference_time"] = optimized_inference
            results["inference_speedup"] = baseline_inference / optimized_inference
            
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
            results["inference_benchmark_error"] = str(e)
        
        # 3. Benchmark training speed
        try:
            training_speed = self.benchmark_training_speed(max_steps=20)
            results["training_speed_steps_per_sec"] = training_speed
            
        except Exception as e:
            logger.error(f"Training benchmark failed: {e}")
            results["training_benchmark_error"] = str(e)
        
        # 4. Generate optimization report
        self.generate_optimization_report(results)
        
        return results
    
    def generate_optimization_report(self, results):
        """Optimization results raporu."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 PERFORMANCE OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        
        if "inference_speedup" in results:
            logger.info(f"🚀 Inference Optimization:")
            logger.info(f"   Baseline: {results['baseline_inference_time']:.3f}s")
            logger.info(f"   Optimized: {results['optimized_inference_time']:.3f}s")
            logger.info(f"   Speedup: {results['inference_speedup']:.2f}x")
            
            if results['optimized_inference_time'] < 2.0:
                logger.info("   ✅ Inference speed target achieved (<2s)")
            else:
                logger.info("   ⚠️ Inference speed needs more optimization")
        
        if "training_speed_steps_per_sec" in results:
            logger.info(f"\n🏋️ Training Optimization:")
            logger.info(f"   Speed: {results['training_speed_steps_per_sec']:.3f} steps/sec")
            
            if results['training_speed_steps_per_sec'] > 0.1:
                logger.info("   ✅ Training speed target achieved (>0.1 steps/sec)")
            else:
                logger.info("   ⚠️ Training speed needs more optimization")
        
        # Save detailed report
        report_path = "./performance_optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n📄 Detailed report saved: {report_path}")

def main():
    """Main optimization function."""
    optimizer = PerformanceOptimizer()
    results = optimizer.run_full_optimization()
    
    # Performance targets check
    success = True
    
    if "optimized_inference_time" in results:
        if results["optimized_inference_time"] >= 2.0:
            success = False
            
    if "training_speed_steps_per_sec" in results:
        if results["training_speed_steps_per_sec"] < 0.1:
            success = False
    
    if success:
        logger.info("🎉 Performance optimization successful!")
        return True
    else:
        logger.error("❌ Performance optimization needs more work")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
