#!/usr/bin/env python3
"""
Training Configuration Validation Script

Bu script, eğitime başlamadan önce tüm config parametrelerini ve 
setup'ı validate eder. Hataları önceden yakalar.

Author: GitHub Copilot
Date: 28 Haziran 2025
"""

import os
import sys
import yaml
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List

# HuggingFace ve TRL imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, GRPOConfig, SFTTrainer, GRPOTrainer
    from datasets import load_dataset
    print("✅ TRL ve Transformers import başarılı")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    sys.exit(1)

# Local imports
try:
    from src.training_architecture import ComprehensiveTrainingConfig, TrainingOrchestrator
    print("✅ Training architecture import başarılı")
except ImportError as e:
    print(f"❌ Training architecture import hatası: {e}")
    sys.exit(1)


def setup_logging():
    """Logging kurulumu."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_config_file(config_path: str) -> Dict[str, Any]:
    """Config dosyasını validate eder."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Config dosyası başarıyla yüklendi: {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Config dosyası okunamadı: {e}")


def validate_sft_config(sft_config: Dict[str, Any]) -> List[str]:
    """SFT config parametrelerini validate eder."""
    errors = []
    
    # TRL SFTConfig'de desteklenen parametreler
    valid_sft_params = {
        'num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size',
        'learning_rate', 'warmup_ratio', 'weight_decay', 'max_length',
        'bf16', 'fp16', 'gradient_checkpointing', 'dataloader_num_workers',
        'remove_unused_columns', 'eval_strategy', 'eval_steps', 'save_strategy',
        'save_steps', 'logging_steps', 'load_best_model_at_end', 'metric_for_best_model',
        'greater_is_better', 'report_to', 'packing', 'dataset_kwargs', 'output_dir',
        'lr_scheduler_type', 'save_total_limit', 'max_grad_norm', 'optim',
        'neftune_noise_alpha', 'use_liger_kernel', 'padding_free'
    }
    
    # Desteklenmeyen parametreleri kontrol et
    for param in sft_config:
        if param not in valid_sft_params:
            errors.append(f"SFT Config'te desteklenmeyen parametre: {param}")
    
    # Kritik parametreleri kontrol et
    if 'max_seq_length' in sft_config:
        errors.append("SFT Config'te 'max_seq_length' yerine 'max_length' kullanın")
    
    if 'cosine_restarts' in str(sft_config.get('lr_scheduler_type', '')):
        errors.append("lr_scheduler_type 'cosine_restarts' desteklenmiyor, 'cosine' kullanın")
    
    # Learning rate formatını kontrol et
    lr = sft_config.get('learning_rate')
    if lr and isinstance(lr, str):
        errors.append(f"learning_rate string formatında: {lr}, float olmalı")
    
    return errors


def validate_grpo_config(grpo_config: Dict[str, Any]) -> List[str]:
    """GRPO config parametrelerini validate eder."""
    errors = []
    
    # TRL GRPOConfig'de desteklenen parametreler
    valid_grpo_params = {
        'num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size',
        'learning_rate', 'warmup_ratio', 'weight_decay', 'bf16', 'fp16',
        'gradient_checkpointing', 'logging_steps', 'use_vllm', 'vllm_server_host',
        'ds3_gather_for_generation', 'max_new_tokens', 'missing_eos_penalty',
        'beta', 'output_dir', 'eval_strategy', 'eval_steps', 'save_strategy',
        'save_steps', 'save_total_limit', 'max_grad_norm'
    }
    
    # Desteklenmeyen parametreleri kontrol et
    for param in grpo_config:
        if param not in valid_grpo_params:
            errors.append(f"GRPO Config'te desteklenmeyen parametre: {param}")
    
    return errors


def validate_data_files(config: Dict[str, Any]) -> List[str]:
    """Veri dosyalarını validate eder."""
    errors = []
    
    # SFT veri dosyaları
    sft_train = config.get('train_data_path')
    sft_val = config.get('val_data_path')
    
    if sft_train and not os.path.exists(sft_train):
        errors.append(f"SFT train dosyası bulunamadı: {sft_train}")
    
    if sft_val and not os.path.exists(sft_val):
        errors.append(f"SFT validation dosyası bulunamadı: {sft_val}")
    
    # GRPO veri dosyaları
    grpo_train = config.get('grpo_train_data_path')
    grpo_val = config.get('grpo_val_data_path')
    
    if grpo_train and not os.path.exists(grpo_train):
        errors.append(f"GRPO train dosyası bulunamadı: {grpo_train}")
    
    if grpo_val and not os.path.exists(grpo_val):
        errors.append(f"GRPO validation dosyası bulunamadı: {grpo_val}")
    
    return errors


def test_model_and_tokenizer_loading(model_name: str) -> List[str]:
    """Model ve tokenizer yükleme testi."""
    errors = []
    
    try:
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer başarıyla yüklendi: {model_name}")
        
        # Test model loading (sadece config)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        print(f"✅ Model config başarıyla yüklendi: {model_name}")
        
    except Exception as e:
        errors.append(f"Model/tokenizer yükleme hatası: {e}")
    
    return errors


def test_sft_config_creation(sft_config: Dict[str, Any]) -> List[str]:
    """SFTConfig oluşturma testi."""
    errors = []
    
    try:
        # Test output_dir ekleme
        test_config = sft_config.copy()
        test_config['output_dir'] = './test_output'
        
        # SFTConfig oluşturmayı dene
        training_args = SFTConfig(**test_config)
        print("✅ SFTConfig başarıyla oluşturuldu")
        
    except Exception as e:
        errors.append(f"SFTConfig oluşturma hatası: {e}")
    
    return errors


def test_grpo_config_creation(grpo_config: Dict[str, Any]) -> List[str]:
    """GRPOConfig oluşturma testi."""
    errors = []
    
    try:
        # Test output_dir ekleme
        test_config = grpo_config.copy()
        test_config['output_dir'] = './test_output'
        
        # GRPOConfig oluşturmayı dene
        training_args = GRPOConfig(**test_config)
        print("✅ GRPOConfig başarıyla oluşturuldu")
        
    except Exception as e:
        errors.append(f"GRPOConfig oluşturma hatası: {e}")
    
    return errors


def test_data_loading(data_path: str) -> List[str]:
    """Veri yükleme testi."""
    errors = []
    
    if not data_path or not os.path.exists(data_path):
        return errors
    
    try:
        # JSONL dosyasını test et
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    errors.append(f"Veri dosyası boş: {data_path}")
                else:
                    # İlk satırı parse etmeyi dene
                    first_line = json.loads(lines[0])
                    print(f"✅ Veri dosyası başarıyla test edildi: {data_path} ({len(lines)} satır)")
                    
    except Exception as e:
        errors.append(f"Veri dosyası okuma hatası ({data_path}): {e}")
    
    return errors


def main():
    """Ana validation fonksiyonu."""
    logger = setup_logging()
    
    print("🔍 Training Configuration Validation Başlıyor...")
    print("=" * 60)
    
    # Config dosyasını yükle
    config_path = "configs/full_production_training.yaml"
    
    try:
        config = validate_config_file(config_path)
    except Exception as e:
        print(f"❌ Config dosyası yükleme hatası: {e}")
        return False
    
    all_errors = []
    
    # 1. SFT config validation
    print("\n📋 SFT Config Validation...")
    sft_config = config.get('sft_config', {})
    sft_errors = validate_sft_config(sft_config)
    all_errors.extend(sft_errors)
    
    if not sft_errors:
        print("✅ SFT config geçerli")
    
    # 2. GRPO config validation  
    print("\n📋 GRPO Config Validation...")
    grpo_config = config.get('grpo_config', {})
    grpo_errors = validate_grpo_config(grpo_config)
    all_errors.extend(grpo_errors)
    
    if not grpo_errors:
        print("✅ GRPO config geçerli")
    
    # 3. Veri dosyaları validation
    print("\n📂 Veri Dosyaları Validation...")
    data_errors = validate_data_files(config)
    all_errors.extend(data_errors)
    
    if not data_errors:
        print("✅ Veri dosyaları mevcut")
    
    # 4. Model ve tokenizer test
    print("\n🤖 Model ve Tokenizer Test...")
    model_name = config.get('model_name_or_path', 'Qwen/Qwen2.5-0.5B-Instruct')
    model_errors = test_model_and_tokenizer_loading(model_name)
    all_errors.extend(model_errors)
    
    # 5. Config creation tests
    print("\n⚙️ Config Creation Tests...")
    sft_creation_errors = test_sft_config_creation(sft_config)
    all_errors.extend(sft_creation_errors)
    
    grpo_creation_errors = test_grpo_config_creation(grpo_config)
    all_errors.extend(grpo_creation_errors)
    
    # 6. Veri yükleme tests
    print("\n📊 Veri Yükleme Tests...")
    for data_path in [config.get('train_data_path'), config.get('val_data_path'), 
                      config.get('grpo_train_data_path'), config.get('grpo_val_data_path')]:
        if data_path:
            data_load_errors = test_data_loading(data_path)
            all_errors.extend(data_load_errors)
    
    # Sonuçları göster
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SONUÇLARI:")
    
    if all_errors:
        print(f"❌ {len(all_errors)} HATA BULUNדו:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        
        print("\n💡 Bu hataları düzelttikten sonra training'i başlatın!")
        return False
    else:
        print("✅ TÜM VALIDATIONLAR BAŞARILI!")
        print("🚀 Training'e başlamaya hazırsınız!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
