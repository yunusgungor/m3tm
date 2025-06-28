#!/usr/bin/env python3
"""
Full-Scale Production Model Training Launcher
Bu script, genişletilmiş veri setiyle uzun dönemli, production-grade model eğitimi başlatır.
"""

import os
import sys
import json
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime
import torch
import psutil

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.training_architecture import TrainingOrchestrator

# Logging configuration
def setup_logging():
    """Production-grade logging setup"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"full_production_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def check_system_requirements():
    """Sistem gereksinimlerini kontrol et"""
    logger = logging.getLogger(__name__)
    
    # Memory check
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    logger.info(f"Sistem RAM: {memory_gb:.2f} GB")
    
    if memory_gb < 8:
        logger.warning("Uyarı: 8GB'den az RAM tespit edildi. Eğitim yavaş olabilir.")
    
    # Disk space check
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / (1024**3)
    logger.info(f"Boş disk alanı: {disk_free_gb:.2f} GB")
    
    if disk_free_gb < 5:
        logger.error("Hata: Yetersiz disk alanı! En az 5GB gerekli.")
        return False
    
    # CPU check
    cpu_count = psutil.cpu_count()
    logger.info(f"CPU çekirdek sayısı: {cpu_count}")
    
    # PyTorch check
    logger.info(f"PyTorch versiyonu: {torch.__version__}")
    logger.info(f"CUDA mevcut: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA cihaz sayısı: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return True

def validate_data_files(config):
    """Veri dosyalarının varlığını kontrol et"""
    logger = logging.getLogger(__name__)
    
    required_files = [
        config.get('train_data_path'),
        config.get('val_data_path'),
        config.get('grpo_train_data_path'),
        config.get('grpo_val_data_path')
    ]
    
    for file_path in required_files:
        if file_path:
            if not Path(file_path).exists():
                logger.error(f"Veri dosyası bulunamadı: {file_path}")
                return False
            else:
                # Dosya boyutunu kontrol et
                with open(file_path, 'r') as f:
                    line_count = sum(1 for line in f if line.strip())
                logger.info(f"✓ {file_path}: {line_count} satır")
    
    return True

def backup_existing_models(output_dir):
    """Mevcut modelleri yedekle"""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    if output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = output_path.parent / f"{output_path.name}_backup_{timestamp}"
        
        logger.info(f"Mevcut model yedekleniyor: {backup_dir}")
        output_path.rename(backup_dir)

def estimate_training_time(config):
    """Tahmini eğitim süresini hesapla"""
    logger = logging.getLogger(__name__)
    
    # SFT parameters
    sft_epochs = config.get('sft_config', {}).get('num_train_epochs', 15)
    sft_batch_size = config.get('sft_config', {}).get('per_device_train_batch_size', 16)
    
    # GRPO parameters  
    grpo_epochs = config.get('grpo_config', {}).get('num_train_epochs', 8)
    grpo_batch_size = config.get('grpo_config', {}).get('per_device_train_batch_size', 8)
    
    # Rough estimates (these are very approximate)
    sft_data_size = 1000  # From expanded data
    grpo_data_size = 500
    
    # Steps per epoch
    sft_steps_per_epoch = max(1, sft_data_size // sft_batch_size)
    grpo_steps_per_epoch = max(1, grpo_data_size // grpo_batch_size)
    
    # Total steps
    total_sft_steps = sft_epochs * sft_steps_per_epoch
    total_grpo_steps = grpo_epochs * grpo_steps_per_epoch
    
    # Rough time estimates (seconds per step - very approximate)
    cpu_time_per_step = 5  # CPU training is slower
    
    estimated_sft_time = total_sft_steps * cpu_time_per_step
    estimated_grpo_time = total_grpo_steps * cpu_time_per_step
    estimated_mobile_opt_time = 300  # 5 minutes for mobile optimization
    
    total_estimated_time = estimated_sft_time + estimated_grpo_time + estimated_mobile_opt_time
    
    # Convert to hours
    estimated_hours = total_estimated_time / 3600
    
    logger.info(f"Tahmini eğitim süresi:")
    logger.info(f"  SFT: {estimated_sft_time/3600:.1f} saat ({total_sft_steps} steps)")
    logger.info(f"  GRPO: {estimated_grpo_time/3600:.1f} saat ({total_grpo_steps} steps)")
    logger.info(f"  Mobil Optimizasyon: {estimated_mobile_opt_time/3600:.1f} saat")
    logger.info(f"  TOPLAM: {estimated_hours:.1f} saat")
    
    return estimated_hours

def main():
    """Ana fonksiyon - Full production training"""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("🚀 FULL-SCALE PRODUCTION MODEL TRAINING BAŞLATIYOR")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Sistem gereksinimlerini kontrol et
        logger.info("📋 Sistem gereksinimleri kontrol ediliyor...")
        if not check_system_requirements():
            logger.error("Sistem gereksinimleri karşılanmıyor!")
            return False
        
        # 2. Config dosyasını yükle
        config_path = "configs/full_production_training.yaml"
        logger.info(f"📄 Config dosyası yükleniyor: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Config yüklendi: {config['experiment_name']}")
        
        # 3. Veri dosyalarını kontrol et
        logger.info("📊 Veri dosyaları kontrol ediliyor...")
        if not validate_data_files(config):
            logger.error("Veri dosyaları eksik!")
            return False
        
        # 4. Eğitim süresi tahmini
        logger.info("⏱️  Eğitim süresi tahmini hesaplanıyor...")
        estimated_hours = estimate_training_time(config)
        
        # 5. Kullanıcı onayı
        print(f"\n🤔 Bu eğitim yaklaşık {estimated_hours:.1f} saat sürecek.")
        print("📱 Mobil model optimizasyonu ile birlikte tam pipeline çalışacak.")
        print("💾 Tüm checkpoint'ler ve loglar kaydedilecek.")
        
        user_input = input("\n✅ Devam etmek istiyor musunuz? (y/N): ").strip().lower()
        if user_input not in ['y', 'yes', 'evet']:
            logger.info("❌ Eğitim kullanıcı tarafından iptal edildi.")
            return False
        
        # 6. Mevcut modelleri yedekle
        logger.info("💾 Mevcut modeller yedekleniyor...")
        backup_existing_models(config['output_dir'])
        
        # 7. Training pipeline'ı başlat
        logger.info("🔥 Training Orchestrator başlatılıyor...")
        
        # Config'i ComprehensiveTrainingConfig object'ine dönüştür
        from src.training_architecture import ComprehensiveTrainingConfig
        config_obj = ComprehensiveTrainingConfig()
        
        # Config değerlerini set et
        for key, value in config.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
        
        orchestrator = TrainingOrchestrator(config_obj)
        
        # 8. Full training çalıştır
        logger.info("🎯 FULL-SCALE TRAINING BAŞLIYOR!")
        logger.info("-" * 50)
        
        results = orchestrator.run_comprehensive_training()
        
        # 9. Sonuçları raporla
        end_time = time.time()
        actual_duration = (end_time - start_time) / 3600
        
        logger.info("=" * 80)
        logger.info("🎉 FULL-SCALE TRAINING TAMAMLANDI!")
        logger.info("=" * 80)
        logger.info(f"⏱️  Toplam süre: {actual_duration:.2f} saat")
        logger.info(f"📊 Tahmini süre: {estimated_hours:.1f} saat")
        logger.info(f"📈 Gerçek vs Tahmini: %{(actual_duration/estimated_hours)*100:.1f}")
        
        if results:
            logger.info("✅ Training başarıyla tamamlandı!")
            logger.info(f"📁 Çıktı dizini: {config['output_dir']}")
            
            # Sonuç özetini yazdır
            if 'sft_results' in results:
                logger.info(f"🔧 SFT Loss: {results['sft_results'].get('final_loss', 'N/A')}")
            if 'grpo_results' in results:
                logger.info(f"🎖️  GRPO Reward: {results['grpo_results'].get('final_reward', 'N/A')}")
            if 'mobile_optimization_results' in results:
                mobile_results = results['mobile_optimization_results']
                logger.info(f"📱 Model Boyutu: {mobile_results.get('model_size_mb', 'N/A')} MB")
                logger.info(f"⚡ Inference Hızı: {mobile_results.get('avg_inference_time_ms', 'N/A')} ms")
            
        else:
            logger.error("❌ Training tamamlanamadı!")
            return False
        
        # 10. Next steps
        logger.info("\n🔮 SONRAKI ADIMLAR:")
        logger.info("1. Model performansını değerlendirin")
        logger.info("2. Gerçek cihazda test edin")
        logger.info("3. Production deployment için hazırlayın")
        logger.info("4. A/B testing yapın")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Eğitim kullanıcı tarafından durduruldu (Ctrl+C)")
        return False
    except Exception as e:
        logger.error(f"❌ Eğitim sırasında hata oluştu: {e}")
        logger.exception("Detaylı hata:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
