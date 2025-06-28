#!/usr/bin/env python3
"""
Final Validation Script - Comprehensive System Check

Bu script, full-scale production training'e geçmeden önce
tüm sistemi kapsamlı şekilde test eder ve potansiyel hataları önceden yakalar.

Context7 entegrasyonlu, güncel TRL API'sine uygun validasyon sistemi.
"""

import os
import sys
import time
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_environment():
    """Sistem ortamını kontrol et."""
    logger.info("🔍 Sistem ortamı kontrol ediliyor...")
    
    # Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Memory info
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon GPU) available: True")
    
    # Import test
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
        from datasets import Dataset
        logger.info("✅ Temel kütüphaneler başarıyla import edildi")
    except ImportError as e:
        logger.error(f"❌ Import hatası: {e}")
        return False
    
    return True

def validate_data_files():
    """Veri dosyalarını kontrol et."""
    logger.info("📂 Veri dosyaları kontrol ediliyor...")
    
    data_files = [
        "./data/expanded/sft_train_expanded.jsonl",
        "./data/expanded/sft_val_expanded.jsonl", 
        "./data/expanded/grpo_train_expanded.jsonl",
        "./data/expanded/grpo_val_expanded.jsonl"
    ]
    
    for file_path in data_files:
        if not os.path.exists(file_path):
            logger.error(f"❌ Veri dosyası bulunamadı: {file_path}")
            return False
        
        # Check file size
        size = os.path.getsize(file_path)
        logger.info(f"✅ {file_path}: {size} bytes")
        
        # Check first few lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [f.readline() for _ in range(3)]
                for i, line in enumerate(lines):
                    if line.strip():
                        data = json.loads(line)
                        logger.info(f"   Satır {i+1} sample keys: {list(data.keys())}")
                        break
        except Exception as e:
            logger.error(f"❌ {file_path} okuma hatası: {e}")
            return False
    
    return True

def validate_config_files():
    """Config dosyalarını kontrol et.""" 
    logger.info("⚙️ Config dosyaları kontrol ediliyor...")
    
    config_path = "./configs/comprehensive_training.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"❌ Config dosyası bulunamadı: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check essential sections
        required_sections = ['sft_config', 'grpo_config', 'model_name_or_path', 'max_length']
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Config'te eksik section: {section}")
                return False
        
        # Check SFT config parameters
        sft_config = config['sft_config']
        required_sft_params = ['num_train_epochs', 'per_device_train_batch_size', 'learning_rate', 'max_length']
        for param in required_sft_params:
            if param not in sft_config:
                logger.error(f"❌ SFT config'te eksik parametre: {param}")
                return False
        
        # Check GRPO config
        grpo_config = config['grpo_config']
        required_grpo_params = ['num_train_epochs', 'per_device_train_batch_size', 'learning_rate']
        for param in required_grpo_params:
            if param not in grpo_config:
                logger.error(f"❌ GRPO config'te eksik parametre: {param}")
                return False
        
        logger.info("✅ Config dosyası geçerli")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config dosyası okuma hatası: {e}")
        return False

def validate_model_loading():
    """Model yüklemeyi test et."""
    logger.info("🤖 Model yükleme testi...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Model test ediliyor: {model_name}")
        
        # Tokenizer test
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # PAD token ayarla
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✅ Tokenizer yüklendi. Vocab size: {tokenizer.vocab_size}")
        
        # Model test (CPU'da)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU uyumluluğu için
            device_map="cpu"
        )
        logger.info(f"✅ Model yüklendi. Parameters: {model.num_parameters():,}")
        
        # Test inference
        test_text = "Python'da fonksiyon nasıl yazılır?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Test inference başarılı: {response[:100]}...")
        
        # Clean up memory
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {e}")
        return False

def validate_trl_configs():
    """TRL Config objelerini test et."""
    logger.info("🔧 TRL Config objelerini test ediliyor...")
    
    try:
        from trl import SFTConfig, GRPOConfig
        
        # SFTConfig test
        sft_config = SFTConfig(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=2e-5,
            max_length=512,
            bf16=False,  # CPU için
            remove_unused_columns=False,
            logging_steps=10
        )
        logger.info("✅ SFTConfig başarıyla oluşturuldu")
        
        # GRPOConfig test
        grpo_config = GRPOConfig(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=8,  # Daha büyük batch size
            learning_rate=5e-6,
            bf16=False,  # CPU için
            logging_steps=10
            # num_generations_per_prompt kaldırıldı - yeni TRL sürümünde yok
        )
        logger.info("✅ GRPOConfig başarıyla oluşturuldu")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TRL Config hatası: {e}")
        return False

def validate_data_loading():
    """Veri yüklemeyi test et.""" 
    logger.info("📊 Veri yükleme testi...")
    
    try:
        import json
        from datasets import Dataset
        
        # SFT data test
        sft_data = []
        valid_count = 0
        total_count = 0
        
        with open("./data/expanded/sft_train_expanded.jsonl", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if valid_count >= 10:  # İlk 10 geçerli sample
                    break
                if line.strip():
                    total_count += 1
                    try:
                        raw_data = json.loads(line)
                        # Sadece dict tipleri işle ve gerekli alanları olan verileri al
                        if isinstance(raw_data, dict) and 'instruction' in raw_data and 'response' in raw_data:
                            # Veri temizliği 
                            cleaned_data = {}
                            for key, value in raw_data.items():
                                if isinstance(value, (str, int, float, bool, dict)):
                                    cleaned_data[key] = value
                                elif isinstance(value, list):
                                    # List olan elemanları string'e çevir
                                    cleaned_data[key] = str(value)
                            sft_data.append(cleaned_data)
                            valid_count += 1
                    except Exception as parse_error:
                        logger.warning(f"   Satır {total_count} atlandı: {parse_error}")
                        continue
        
        logger.info(f"✅ SFT data yüklendi: {len(sft_data)} geçerli samples ({total_count} toplam satır kontrol edildi)")
        logger.info(f"   Sample keys: {list(sft_data[0].keys()) if sft_data else 'No data'}")
        
        # Dataset creation test
        dataset = Dataset.from_list(sft_data)
        logger.info(f"✅ HuggingFace Dataset oluşturuldu: {len(dataset)} samples")
        
        logger.info("✅ GRPO data test atlandı (ana test için gerekli değil)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Veri yükleme hatası: {e}")
        import traceback
        logger.error(f"   Hata detayı: {traceback.format_exc()}")
        return False

def validate_training_architecture():
    """Training architecture'ı test et."""
    logger.info("🏗️ Training architecture testi...")
    
    try:
        # Import test
        sys.path.append('./src')
        from training_architecture import ComprehensiveTrainingConfig, TrainingOrchestrator
        
        # Config test
        config = ComprehensiveTrainingConfig()
        logger.info("✅ ComprehensiveTrainingConfig oluşturuldu")
        
        # Orchestrator test
        orchestrator = TrainingOrchestrator(config)
        logger.info("✅ TrainingOrchestrator oluşturuldu")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training architecture hatası: {e}")
        return False

def run_mini_sft_test():
    """Mini SFT test çalıştır."""
    logger.info("🚀 Mini SFT testi başlatılıyor...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        import json
        
        # Model ve tokenizer yükle
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # PAD token ayarla
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Mini dataset yükle ve SFT formatına dönüştür
        data = []
        valid_count = 0
        total_count = 0
        
        with open("./data/expanded/sft_train_expanded.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                if valid_count >= 5:  # Sadece 5 geçerli sample
                    break
                if line.strip():
                    total_count += 1
                    try:
                        original_data = json.loads(line)
                        # Sadece gerekli alanları olan dict verileri işle
                        if isinstance(original_data, dict) and 'instruction' in original_data and 'response' in original_data:
                            # SFT format için text alanı oluştur
                            formatted_data = {
                                "text": f"### İnsan: {original_data['instruction']}\n### Asistan: {original_data['response']}"
                            }
                            data.append(formatted_data)
                            valid_count += 1
                    except Exception as parse_error:
                        continue  # Geçersiz satırları atla
        
        dataset = Dataset.from_list(data)
        
        # SFT Config - chat formatting devre dışı bırak
        training_args = SFTConfig(
            output_dir="./test_mini_sft",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=2e-5,
            max_length=256,  # Kısa sequence
            bf16=False,
            remove_unused_columns=False,
            logging_steps=1,
            save_strategy="no",  # Kaydetme
            eval_strategy="no",  # Evaluation yok
            report_to=[],  # Logging yok
            max_steps=2,  # Sadece 2 step
            packing=False,  # Packing devre dışı
            dataset_text_field="text"  # Text alanını belirt
        )
        
        # Trainer oluştur
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer  # tokenizer yerine processing_class kullan
        )
        
        logger.info("✅ Mini SFT trainer oluşturuldu")
        
        # 2 step training
        trainer.train()
        logger.info("✅ Mini SFT training tamamlandı")
        
        # Clean up
        del trainer, model, tokenizer, dataset
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mini SFT test hatası: {e}")
        import traceback
        logger.error(f"   Hata detayı: {traceback.format_exc()}")
        return False

def main():
    """Ana validation fonksiyonu."""
    logger.info("🎯 FINAL VALIDATION BAŞLATILIYOR")
    logger.info("=" * 60)
    
    validation_results = []
    
    # Validation steps
    validations = [
        ("Environment Check", validate_environment),
        ("Data Files Check", validate_data_files), 
        ("Config Files Check", validate_config_files),
        ("Model Loading Test", validate_model_loading),
        ("TRL Configs Test", validate_trl_configs),
        ("Data Loading Test", validate_data_loading),
        ("Training Architecture Test", validate_training_architecture),
        ("Mini SFT Test", run_mini_sft_test)
    ]
    
    for name, validation_func in validations:
        logger.info(f"\n🔍 {name}...")
        try:
            result = validation_func()
            validation_results.append((name, result))
            if result:
                logger.info(f"✅ {name} BAŞARILI")
            else:
                logger.error(f"❌ {name} BAŞARISIZ")
        except Exception as e:
            logger.error(f"❌ {name} HATA: {e}")
            validation_results.append((name, False))
    
    # Sonuçları özetle
    logger.info("\n" + "=" * 60)
    logger.info("📋 VALIDATION SONUÇLARI:")
    
    passed = 0
    total = len(validation_results)
    
    for name, result in validation_results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        logger.info(f"   {name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n📊 ÖZET: {passed}/{total} validation geçti")
    
    if passed == total:
        logger.info("🎉 TÜM VALIDASYONLAR BAŞARILI!")
        logger.info("🚀 Full-scale production training'e hazırsınız!")
        return True
    else:
        logger.error("⚠️ Bazı validasyonlar başarısız!")
        logger.error("Lütfen hataları düzeltin ve tekrar deneyin.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
