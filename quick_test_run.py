#!/usr/bin/env python3
"""
Quick Training Test Run

Bu script, gerçek training'den önce çok kısa bir test run yapar.
Tüm pipeline'ın çalıştığını ve büyük hatalar olmadığını kontrol eder.

Author: GitHub Copilot
Date: 28 Haziran 2025
"""

import os
import sys
import yaml
import json
import logging
import shutil
from pathlib import Path

# Gerekli imports
from src.training_architecture import ComprehensiveTrainingConfig, TrainingOrchestrator


def setup_test_config():
    """Test için çok kısa config oluşturur."""
    
    # Ana config'i yükle
    with open('configs/full_production_training.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Test için modifikasyonlar
    test_config = config.copy()
    
    # Çok kısa eğitim parametreleri
    test_config['sft_config'].update({
        'num_train_epochs': 1,
        'max_steps': 5,  # Sadece 5 step
        'per_device_train_batch_size': 2,
        'eval_steps': 2,
        'save_steps': 10,
        'logging_steps': 1,
    })
    
    test_config['grpo_config'].update({
        'num_train_epochs': 1,
        'max_steps': 3,  # GRPO için 3 step
        'per_device_train_batch_size': 2,
        'eval_steps': 2,
        'save_steps': 10,
        'logging_steps': 1,
    })
    
    # Test output directory
    test_config['output_dir'] = './test_run_output'
    test_config['experiment_name'] = 'quick_test_run'
    
    # WandB'yi devre dışı bırak
    test_config['monitoring']['enable_wandb'] = False
    test_config['sft_config']['report_to'] = []
    
    # Test config'i kaydet
    os.makedirs('configs', exist_ok=True)
    with open('configs/test_run_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)
    
    return test_config


def setup_logging():
    """Test için logging setup."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    """Ana test fonksiyonu."""
    logger = setup_logging()
    
    print("🧪 QUICK TRAINING TEST RUN BAŞLIYOR")
    print("=" * 60)
    
    try:
        # Test output directory'sini temizle
        if os.path.exists('./test_run_output'):
            shutil.rmtree('./test_run_output')
        
        # Test config'i oluştur
        print("📋 Test config'i hazırlanıyor...")
        test_config = setup_test_config()
        
        # Config'i ComprehensiveTrainingConfig'e dönüştür
        print("⚙️ Training config'i yükleniyor...")
        config_obj = ComprehensiveTrainingConfig(
            experiment_name=test_config.get('experiment_name', 'test'),
            output_dir=test_config.get('output_dir', './test_run_output'),
            model_name_or_path=test_config.get('model_name_or_path', 'Qwen/Qwen2.5-0.5B-Instruct'),
            max_length=test_config.get('max_length', 2048),
            sft_config=test_config.get('sft_config', {}),
            grpo_config=test_config.get('grpo_config', {}),
            sft_dataset_name=test_config.get('train_data_path', './data/expanded/sft_train_expanded.jsonl'),
            sft_eval_dataset_name=test_config.get('val_data_path', './data/expanded/sft_val_expanded.jsonl'),
            grpo_dataset_name=test_config.get('grpo_train_data_path', './data/expanded/grpo_train_expanded.jsonl'),
            grpo_eval_dataset_name=test_config.get('grpo_val_data_path', './data/expanded/grpo_val_expanded.jsonl'),
            enable_sft=True,
            enable_grpo=True,
            enable_reward_model=False,  # Test'te devre dışı
            enable_mobile_optimization=False,  # Test'te devre dışı
        )
        
        # TrainingOrchestrator'u başlat
        print("🚀 TrainingOrchestrator'u başlatıyor...")
        orchestrator = TrainingOrchestrator(config_obj)
        
        # Sadece SFT'yi test et
        print("\n📚 SFT Quick Test...")
        print("- Model yüklenecek")
        print("- Tokenizer yüklenecek") 
        print("- Veri yüklenecek")
        print("- 5 step SFT training yapılacak")
        print("- Model kaydedilecek")
        
        # SFT'yi çalıştır
        sft_result = orchestrator.run_sft_training()
        print(f"✅ SFT test başarılı: {sft_result}")
        
        # GRPO'yu test et
        print("\n🎯 GRPO Quick Test...")
        print("- SFT model yüklenecek")
        print("- Reward functions hazırlanacak")
        print("- 3 step GRPO training yapılacak")
        print("- Model kaydedilecek")
        
        # GRPO'yu çalıştır
        grpo_result = orchestrator.run_grpo_training(sft_result)
        print(f"✅ GRPO test başarılı: {grpo_result}")
        
        print("\n" + "=" * 60)
        print("🎉 QUICK TEST RUN BAŞARILI!")
        print("📁 Test sonuçları: ./test_run_output/")
        print("🚀 Full-scale training'e hazırsınız!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST RUN HATASI: {e}")
        print("\n📝 Hata detayları:")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Bu hatayı çözün ve tekrar deneyin.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
