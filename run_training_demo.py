#!/usr/bin/env python3
"""
Kapsamlı Mobil Model Eğitim Demo Pipeline

Bu script, geliştirdiğimiz mobil model eğitim mimarisinin tam özellikli
bir demo uygulamasını çalıştırır.

Features:
- SFT (Supervised Fine-Tuning) eğitimi
- GRPO (Group Relative Policy Optimization) 
- Gelişmiş reward fonksiyonları
- Mobil optimizasyon (quantization, pruning)
- WandB monitoring
- Kapsamlı raporlama

Author: GitHub Copilot  
Date: 28 Haziran 2025
"""

import os
import sys
import time
import logging
from pathlib import Path

# Ana dizini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training_architecture import TrainingOrchestrator, ComprehensiveTrainingConfig
from src.reward_functions import ComprehensiveRewardFunction, RewardConfig
from src.mobile_optimizer import MobileOptimizer

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_demo_config():
    """Demo eğitimi için optimized konfigürasyon oluşturur."""
    
    # Demo için hızlı eğitim parametreleri
    quick_sft_config = {
        "num_train_epochs": 1,  # Demo için 1 epoch
        "per_device_train_batch_size": 2,  # Küçük batch size
        "per_device_eval_batch_size": 4,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        # "max_seq_length": 256,  # Bu TrainingArguments parametresi değil
        "fp16": True,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "eval_strategy": "steps",
        "eval_steps": 50,  # Daha sık eval
        "save_strategy": "steps",
        "save_steps": 100,
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": ["tensorboard", "wandb"],
    }
    
    quick_grpo_config = {
        "num_train_epochs": 1,  # Demo için 1 epoch
        "per_device_train_batch_size": 8,  # GRPO için 8'e çıkarıldı (generations divisible olması için)
        "learning_rate": 1e-6,  # GRPO için daha düşük
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 5,
        "use_vllm": False,  # Demo için devre dışı
    }
    
    config = ComprehensiveTrainingConfig(
        experiment_name="mobile_model_demo",
        model_name_or_path="microsoft/DialoGPT-small",  # Küçük model demo için
        output_dir="./demo_outputs",
        model_max_length=256,
        
        # Demo için hızlandırılmış stages
        enable_sft=True,
        enable_reward_model=False,  # Demo için skip
        enable_grpo=True,
        enable_dpo=False,  # Demo için skip
        enable_mobile_optimization=True,
        
        sft_config=quick_sft_config,
        grpo_config=quick_grpo_config,
    )
    
    return config

def run_training_demo():
    """Ana demo eğitim pipeline'ını çalıştırır."""
    
    logger.info("🚀 Mobil Model Eğitim Demo Pipeline Başlıyor")
    logger.info("=" * 60)
    
    try:
        # 1. Konfigürasyon oluştur
        logger.info("📋 Demo konfigürasyonu hazırlanıyor...")
        config = create_demo_config()
        logger.info(f"✅ Model: {config.model_name_or_path}")
        logger.info(f"✅ Output: {config.output_dir}")
        
        # 2. Training Orchestrator başlat
        logger.info("\n🔧 Training Orchestrator başlatılıyor...")
        orchestrator = TrainingOrchestrator(config)
        logger.info("✅ Training Orchestrator hazır")
        
        # 3. Reward function hazırla
        logger.info("\n🎯 Reward fonksiyonları hazırlanıyor...")
        reward_config = RewardConfig(
            target_length=50,  # Demo için kısa
            length_tolerance=20
        )
        reward_function = ComprehensiveRewardFunction(reward_config)
        logger.info("✅ Comprehensive reward function hazır")
        
        # 4. SFT eğitimi çalıştır
        logger.info("\n📚 SFT (Supervised Fine-Tuning) başlıyor...")
        start_time = time.time()
        
        sft_model_path = orchestrator.run_sft_training()
        
        sft_time = time.time() - start_time
        logger.info(f"✅ SFT tamamlandı ({sft_time:.2f} saniye)")
        logger.info(f"📊 Model path: {sft_model_path}")
        
        # 5. GRPO eğitimi çalıştır
        logger.info("\n🎮 GRPO (Group Relative Policy Optimization) başlıyor...")
        start_time = time.time()
        
        grpo_model_path = orchestrator.run_grpo_training(sft_model_path)
        
        grpo_time = time.time() - start_time
        logger.info(f"✅ GRPO tamamlandı ({grpo_time:.2f} saniye)")
        logger.info(f"📊 Model path: {grpo_model_path}")
        
        # 6. Mobil optimizasyon
        logger.info("\n📱 Mobil optimizasyon başlıyor...")
        start_time = time.time()
        
        mobile_config = {
            'quantization': {
                'enabled': True,
                'method': 'dynamic'
            },
            'pruning': {
                'enabled': False  # Demo için skip
            },
            'export': {
                'torchscript': True,
                'onnx': False  # Demo için skip
            }
        }
        
        mobile_optimizer = MobileOptimizer(mobile_config)
        optimized_model_path = orchestrator.run_mobile_optimization(grpo_model_path)
        
        mobile_time = time.time() - start_time
        logger.info(f"✅ Mobil optimizasyon tamamlandı ({mobile_time:.2f} saniye)")
        logger.info(f"📊 Optimized model path: {optimized_model_path}")
        
        # 7. Sonuçları özetle
        logger.info("\n" + "=" * 60)
        logger.info("🎯 DEMO EĞİTİM PİPELİNE SONUÇLARI")
        logger.info("=" * 60)
        
        total_time = sft_time + grpo_time + mobile_time
        
        logger.info(f"⏱️  Toplam süre: {total_time:.2f} saniye")
        logger.info(f"📚 SFT süresi: {sft_time:.2f} saniye")
        logger.info(f"🎮 GRPO süresi: {grpo_time:.2f} saniye")
        logger.info(f"📱 Mobil opt. süresi: {mobile_time:.2f} saniye")
        
        logger.info(f"\n📊 Model Performansı:")
        logger.info(f"   • SFT model: {sft_model_path}")
        logger.info(f"   • GRPO model: {grpo_model_path}")
        logger.info(f"   • Optimized model: {optimized_model_path}")
        
        logger.info(f"\n📁 Çıktı dosyaları:")
        logger.info(f"   • Model: {config.output_dir}")
        logger.info(f"   • Loglar: ./training_demo.log")
        logger.info(f"   • TensorBoard: {config.output_dir}/tensorboard")
        
        logger.info("\n🎉 Demo pipeline başarıyla tamamlandı!")
        
        return {
            'success': True,
            'total_time': total_time,
            'sft_model_path': sft_model_path,
            'grpo_model_path': grpo_model_path,
            'optimized_model_path': optimized_model_path
        }
        
    except Exception as e:
        logger.error(f"❌ Demo pipeline hatası: {e}")
        logger.exception("Detaylı hata:")
        return {
            'success': False,
            'error': str(e)
        }

def run_quick_validation():
    """Hızlı model validasyon testi."""
    logger.info("\n🔍 Hızlı model validasyonu başlıyor...")
    
    try:
        # Test prompt'ları
        test_prompts = [
            "Merhaba, nasılsın?",
            "Python programlama hakkında ne düşünüyorsun?",
            "Mobil uygulamalar için hangi teknolojiler önerirsin?"
        ]
        
        # Eğitilmiş model ile generation testi
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_path = "./demo_outputs/sft"
        if Path(model_path).exists():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            logger.info("✅ Eğitilmiş model yüklendi")
            
            for i, prompt in enumerate(test_prompts):
                inputs = tokenizer.encode(prompt, return_tensors='pt')
                with torch.no_grad():
                    outputs = model.generate(
                        inputs, 
                        max_length=inputs.shape[1] + 30,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"📝 Test {i+1}:")
                logger.info(f"   Prompt: {prompt}")
                logger.info(f"   Response: {response}")
        else:
            logger.warning("⚠️ Eğitilmiş model bulunamadı, validation atlandı")
            
    except Exception as e:
        logger.error(f"❌ Validasyon hatası: {e}")

if __name__ == "__main__":
    logger.info("🎬 Mobil Model Eğitim Demo Pipeline")
    logger.info(f"📅 Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"💻 Working directory: {os.getcwd()}")
    
    # Demo pipeline çalıştır
    result = run_training_demo()
    
    if result['success']:
        # Hızlı validasyon
        run_quick_validation()
        
        logger.info("\n🏆 Demo başarıyla tamamlandı!")
        logger.info("🔗 WandB dashboard'u için yukarıdaki linki kontrol edin")
        sys.exit(0)
    else:
        logger.error("💥 Demo başarısız oldu!")
        sys.exit(1)
