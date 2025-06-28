#!/usr/bin/env python3
"""
Kapsamlı Eğitim Mimarisi Entegrasyon Testi
Bu script, geliştirdiğimiz mobil model eğitim mimarisinin tüm bileşenlerini test eder.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Ana dizini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Tüm modüllerin import edilebilirliğini test eder."""
    logger.info("🔍 Modül import testleri başlıyor...")
    
    try:
        # Ana eğitim mimarisi
        from src.training_architecture import TrainingOrchestrator, ComprehensiveTrainingConfig
        logger.info("✅ training_architecture.py başarıyla import edildi")
        
        # Reward fonksiyonları
        from src.reward_functions import (
            LengthRewardFunction, QualityRewardFunction, FormatRewardFunction, 
            SemanticRewardFunction, SafetyRewardFunction, ComprehensiveRewardFunction
        )
        logger.info("✅ reward_functions.py başarıyla import edildi")
        
        # Mobil optimizasyon
        from src.mobile_optimizer import MobileOptimizer
        logger.info("✅ mobile_optimizer.py başarıyla import edildi")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import hatası: {e}")
        return False

def test_config_loading():
    """Konfigürasyon dosyasının yüklenebilirliğini test eder."""
    logger.info("🔍 Konfigürasyon yükleme testi başlıyor...")
    
    try:
        config_path = project_root / "configs" / "comprehensive_training.yaml"
        
        if not config_path.exists():
            logger.error(f"❌ Konfigürasyon dosyası bulunamadı: {config_path}")
            return False
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        logger.info("✅ Konfigürasyon dosyası başarıyla yüklendi")
        logger.info(f"📋 Pipeline adımları: {len(config.get('pipeline', {}).get('stages', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Konfigürasyon yükleme hatası: {e}")
        return False

def test_pipeline_initialization():
    """Pipeline'ın başlatılabilirliğini test eder."""
    logger.info("🔍 Pipeline başlatma testi başlıyor...")
    
    try:
        from src.training_architecture import TrainingOrchestrator, ComprehensiveTrainingConfig
        
        # Basit test konfigürasyonu
        test_config = ComprehensiveTrainingConfig(
            experiment_name="test_training",
            model_name_or_path='microsoft/DialoGPT-small',
            output_dir='./test_outputs'
        )
        
        # Pipeline'ı başlat
        pipeline = TrainingOrchestrator(test_config)
        logger.info("✅ TrainingOrchestrator başarıyla başlatıldı")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline başlatma hatası: {e}")
        return False

def test_reward_functions():
    """Reward fonksiyonlarının çalışabilirliğini test eder."""
    logger.info("🔍 Reward fonksiyonları testi başlıyor...")
    
    try:
        from src.reward_functions import (
            LengthRewardFunction, QualityRewardFunction, FormatRewardFunction, 
            ComprehensiveRewardFunction, RewardConfig
        )
        
        # Test metinleri
        test_inputs = ["Bu bir test metnidir."]
        test_outputs = ["Bu iyi bir cevaptır ve yeterli uzunluktadır."]
        
        # Reward config
        config = RewardConfig()
        
        # Reward fonksiyonlarını test et
        length_reward = LengthRewardFunction(config)
        quality_reward = QualityRewardFunction(config)
        format_reward = FormatRewardFunction(config)
        
        # Basit testler
        length_score = length_reward(test_outputs)
        logger.info(f"✅ Length reward: {length_score}")
        
        quality_score = quality_reward(test_outputs)
        logger.info(f"✅ Quality reward: {quality_score}")
        
        format_score = format_reward(test_outputs)
        logger.info(f"✅ Format reward: {format_score}")
        
        # Kapsamlı reward
        comprehensive_reward = ComprehensiveRewardFunction(config)
        comprehensive_score = comprehensive_reward(test_outputs)
        logger.info(f"✅ Comprehensive reward: {comprehensive_score}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Reward fonksiyonları hatası: {e}")
        return False

def test_mobile_optimizer():
    """Mobil optimizasyon modülünün çalışabilirliğini test eder."""
    logger.info("🔍 Mobil optimizasyon testi başlıyor...")
    
    try:
        from src.mobile_optimizer import MobileOptimizer
        
        # Test konfigürasyonu
        config = {
            'quantization': {
                'enabled': True,
                'method': 'dynamic'
            },
            'pruning': {
                'enabled': False  # Test için devre dışı
            },
            'export': {
                'torchscript': True,
                'onnx': False  # Test için devre dışı
            }
        }
        
        optimizer = MobileOptimizer(config)
        logger.info("✅ MobileOptimizer başarıyla başlatıldı")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mobil optimizasyon hatası: {e}")
        return False

def test_data_files():
    """Veri dosyalarının varlığını kontrol eder."""
    logger.info("🔍 Veri dosyaları kontrolü başlıyor...")
    
    data_files = [
        'data/sft_train.jsonl',
        'data/sft_val.jsonl',
        'data/grpo_train.jsonl',
        'data/grpo_val.jsonl'
    ]
    
    all_exist = True
    for file_path in data_files:
        full_path = project_root / file_path
        if full_path.exists():
            file_size = full_path.stat().st_size
            logger.info(f"✅ {file_path} mevcut ({file_size} bytes)")
        else:
            logger.warning(f"⚠️ {file_path} bulunamadı")
            all_exist = False
    
    return all_exist

def run_comprehensive_test():
    """Tüm testleri çalıştırır."""
    logger.info("🚀 Kapsamlı Eğitim Mimarisi Entegrasyon Testi Başlıyor")
    logger.info("=" * 60)
    
    tests = [
        ("Import Testleri", test_imports),
        ("Konfigürasyon Yükleme", test_config_loading),
        ("Pipeline Başlatma", test_pipeline_initialization),
        ("Reward Fonksiyonları", test_reward_functions),
        ("Mobil Optimizasyon", test_mobile_optimizer),
        ("Veri Dosyaları", test_data_files)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 {test_name} çalıştırılıyor...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} beklenmedik hata: {e}")
            results[test_name] = False
    
    # Sonuçları özetle
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SONUÇLARI")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Toplam: {passed}/{total} test başarılı")
    
    if passed == total:
        logger.info("🎉 Tüm testler başarıyla geçti! Sistem entegrasyona hazır.")
    else:
        logger.warning("⚠️ Bazı testler başarısız. Lütfen hataları düzeltin.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
