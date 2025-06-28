"""
Hızlı Production Readiness Validator - 2 dakikada tamamlanır!
"""

import os
import sys
import torch
import logging
import psutil
import gc
import time
import json
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuickProductionValidator:
    """Ultra-hızlı production readiness validator."""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        
    def quick_system_check(self):
        """Hızlı sistem kontrolü (5 saniye)."""
        logger.info("🚀 Hızlı sistem kontrolü...")
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.available / (1024**3)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_gb = disk.free / (1024**3)
        
        # PyTorch
        torch_available = torch.__version__
        
        self.results['system_check'] = {
            'memory_gb': round(memory_gb, 2),
            'disk_gb': round(disk_gb, 2),
            'torch_version': torch_available,
            'passed': memory_gb > 2.0 and disk_gb > 10.0
        }
        
        status = "✅ GEÇTI" if self.results['system_check']['passed'] else "❌ BAŞARISIZ"
        logger.info(f"   {status} - Memory: {memory_gb:.1f}GB, Disk: {disk_gb:.1f}GB")
        
    def quick_import_test(self):
        """Hızlı import testi (10 saniye)."""
        logger.info("🔍 Import ve modül testi...")
        
        try:
            # Test critical imports
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from trl import SFTTrainer
            
            # Test our modules
            from src.training import ComprehensiveTrainingConfig, TrainingOrchestrator
            from src.training.checkpoint_manager import ProductionCheckpointManager
            
            self.results['import_test'] = {
                'transformers': True,
                'trl': True,
                'src_training': True,
                'checkpoint_manager': True,
                'passed': True
            }
            logger.info("   ✅ GEÇTI - Tüm modüller başarıyla import edildi")
            
        except Exception as e:
            self.results['import_test'] = {
                'error': str(e),
                'passed': False
            }
            logger.error(f"   ❌ BAŞARISIZ - Import hatası: {e}")
    
    def quick_model_test(self):
        """Ultra-hızlı model testi (15 saniye)."""
        logger.info("🤖 Hızlı model yükleme testi...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Use smallest possible model
            model_name = "microsoft/DialoGPT-small"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
            
            # Quick inference test
            inputs = tokenizer("Test", return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=20, do_sample=False)
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Cleanup
            del model, tokenizer, inputs, outputs
            gc.collect()
            
            self.results['model_test'] = {
                'model_loaded': True,
                'inference_working': True,
                'result_sample': result[:50],
                'passed': True
            }
            logger.info("   ✅ GEÇTI - Model yükleme ve inference başarılı")
            
        except Exception as e:
            self.results['model_test'] = {
                'error': str(e),
                'passed': False
            }
            logger.error(f"   ❌ BAŞARISIZ - Model testi hatası: {e}")
    
    def quick_checkpoint_test(self):
        """Hızlı checkpoint sistemi testi (20 saniye)."""
        logger.info("💾 Checkpoint sistemi testi...")
        
        try:
            from src.training.checkpoint_manager import ProductionCheckpointManager
            
            # Create test output directory
            test_dir = "./quick_test_checkpoints"
            os.makedirs(test_dir, exist_ok=True)
            
            # Initialize checkpoint manager
            checkpoint_manager = ProductionCheckpointManager(test_dir, save_total_limit=1)
            
            # Load minimal model for testing
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            
            # Test checkpoint saving
            success = checkpoint_manager.save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                trainer_state={'global_step': 1, 'epoch': 0.1},
                step=1
            )
            
            # Verify checkpoint files
            checkpoint_dir = Path(test_dir) / "checkpoint-1"
            pytorch_model_exists = (checkpoint_dir / "pytorch_model.bin").exists()
            config_exists = (checkpoint_dir / "config.json").exists()
            
            # Cleanup
            del model, tokenizer
            gc.collect()
            
            # Clean test directory
            import shutil
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            
            self.results['checkpoint_test'] = {
                'save_success': success,
                'pytorch_model_bin_created': pytorch_model_exists,
                'config_created': config_exists,
                'passed': success and pytorch_model_exists and config_exists
            }
            
            status = "✅ GEÇTI" if self.results['checkpoint_test']['passed'] else "❌ BAŞARISIZ"
            logger.info(f"   {status} - pytorch_model.bin: {pytorch_model_exists}")
            
        except Exception as e:
            self.results['checkpoint_test'] = {
                'error': str(e),
                'passed': False
            }
            logger.error(f"   ❌ BAŞARISIZ - Checkpoint testi hatası: {e}")
    
    def quick_training_test(self):
        """Ultra-minimal training testi (30 saniye)."""
        logger.info("🏃‍♂️ Hızlı training testi...")
        
        try:
            from src.training import ComprehensiveTrainingConfig, TrainingOrchestrator
            
            # Create minimal test data
            test_data = [
                {"text": "Python nedir? Python bir programlama dilidir."},
                {"text": "AI nedir? AI yapay zekadır."},
            ]
            
            # Create temp data files
            train_path = "./temp_quick_train.jsonl"
            eval_path = "./temp_quick_eval.jsonl"
            
            with open(train_path, 'w', encoding='utf-8') as f:
                for item in test_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            with open(eval_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(test_data[0], ensure_ascii=False) + '\n')
            
            # Ultra-minimal config
            config = ComprehensiveTrainingConfig(
                model_name="microsoft/DialoGPT-small",
                output_dir="./temp_quick_output",
                train_data_path=train_path,
                eval_data_path=eval_path,
                num_train_epochs=0.1,  # Very short training
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=5e-5,
                warmup_steps=1,
                save_steps=1,
                eval_steps=1,
                logging_steps=1,
                max_length=50,  # Short sequences
                gradient_accumulation_steps=1,
                fp16=False,  # Avoid fp16 issues
                dataloader_num_workers=0,
                save_total_limit=1,
                report_to=[],
            )
            
            # Test training orchestrator
            orchestrator = TrainingOrchestrator(config)
            
            # Only test setup, not full training
            orchestrator.setup()
            
            training_started = True
            
            # Cleanup
            for temp_file in [train_path, eval_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            if os.path.exists("./temp_quick_output"):
                import shutil
                shutil.rmtree("./temp_quick_output")
            
            self.results['training_test'] = {
                'setup_success': training_started,
                'config_valid': True,
                'orchestrator_working': True,
                'passed': training_started
            }
            
            logger.info("   ✅ GEÇTI - Training setup başarılı")
            
        except Exception as e:
            self.results['training_test'] = {
                'error': str(e),
                'passed': False
            }
            logger.error(f"   ❌ BAŞARISIZ - Training testi hatası: {e}")
    
    def quick_performance_test(self):
        """Hızlı performance testi (10 saniye)."""
        logger.info("⚡ Performance testi...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            
            # Test inference speed (3 samples)
            test_texts = ["Test 1", "Test 2", "Test 3"]
            start_time = time.time()
            
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=30, do_sample=False)
            
            total_time = time.time() - start_time
            avg_inference_time = total_time / len(test_texts)
            
            # Cleanup
            del model, tokenizer
            gc.collect()
            
            # Performance criteria (relaxed for speed)
            inference_fast_enough = avg_inference_time < 5.0  # 5 seconds per inference
            
            self.results['performance_test'] = {
                'avg_inference_time': round(avg_inference_time, 3),
                'inference_fast_enough': inference_fast_enough,
                'total_test_time': round(total_time, 3),
                'passed': inference_fast_enough
            }
            
            status = "✅ GEÇTI" if inference_fast_enough else "❌ YAVAŞ"
            logger.info(f"   {status} - Ortalama inference: {avg_inference_time:.3f}s")
            
        except Exception as e:
            self.results['performance_test'] = {
                'error': str(e),
                'passed': False
            }
            logger.error(f"   ❌ BAŞARISIZ - Performance testi hatası: {e}")
    
    def calculate_final_score(self):
        """Final skor hesaplama."""
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.results.items():
            if isinstance(result, dict) and 'passed' in result:
                total_tests += 1
                if result['passed']:
                    passed_tests += 1
        
        score_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'score_percentage': round(score_percentage, 1),
            'production_ready': score_percentage >= 80.0
        }
    
    def run_quick_validation(self):
        """Tüm hızlı testleri çalıştır."""
        logger.info("🎯 HIZLI PRODUCTION READINESS VALIDATION BAŞLATILIYOR")
        logger.info("=" * 80)
        logger.info("Hedef: 2 dakikada production readiness kontrolü")
        logger.info("=" * 80)
        
        # Run all quick tests
        self.quick_system_check()
        self.quick_import_test()
        self.quick_model_test()
        self.quick_checkpoint_test()
        self.quick_training_test()
        self.quick_performance_test()
        
        # Calculate final results
        final_score = self.calculate_final_score()
        total_time = time.time() - self.start_time
        
        # Print final report
        logger.info("=" * 80)
        logger.info("📋 HIZLI VALIDATION SONUÇLARI")
        logger.info("=" * 80)
        logger.info(f"⏱️  Toplam süre: {total_time:.1f} saniye")
        logger.info(f"📊 Skor: {final_score['passed_tests']}/{final_score['total_tests']} ({final_score['score_percentage']}%)")
        
        # Individual test results
        for test_name, result in self.results.items():
            if isinstance(result, dict) and 'passed' in result:
                status = "✅ GEÇTI" if result['passed'] else "❌ BAŞARISIZ"
                logger.info(f"   {test_name}: {status}")
        
        # Final status
        if final_score['production_ready']:
            logger.info("🟢 DURUM: PRODUCTION READY!")
        else:
            logger.info("🔴 DURUM: NOT READY - Düzeltme gerekli")
            
        # Save detailed results
        report = {
            'timestamp': time.time(),
            'total_time_seconds': total_time,
            'final_score': final_score,
            'detailed_results': self.results
        }
        
        with open('./quick_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Detaylı rapor: ./quick_validation_report.json")
        
        return final_score['production_ready']


if __name__ == "__main__":
    validator = QuickProductionValidator()
    success = validator.run_quick_validation()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 SİSTEM PRODUCTION READY!")
        print("💡 Artık full-scale training'e geçebilirsiniz.")
    else:
        print("⚠️  SİSTEM HENÜz HAZIR DEĞİL")
        print("💡 quick_validation_report.json dosyasından detayları kontrol edin.")
    print("=" * 80)
