#!/usr/bin/env python3
"""
Cache Temizleme Script'i
Tüm __pycache__ klasörlerini, .pyc dosyalarını ve diğer cache dosyalarını siler
"""

import os
import shutil
import sys
import logging
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_pycache(root_dir="."):
    """__pycache__ klasörlerini ve .pyc dosyalarını temizle"""
    removed_dirs = 0
    removed_files = 0
    total_size = 0
    
    logger.info(f"🧹 {root_dir} dizininde cache temizliği başlatılıyor...")
    
    for root, dirs, files in os.walk(root_dir):
        # __pycache__ klasörlerini sil
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                # Boyutunu hesapla
                for dirpath, dirnames, filenames in os.walk(pycache_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except:
                            pass
                
                shutil.rmtree(pycache_path)
                removed_dirs += 1
                logger.info(f"   ✅ Silindi: {pycache_path}")
                dirs.remove('__pycache__')  # Alt dizinlerde arama yapmasın
            except Exception as e:
                logger.error(f"   ❌ {pycache_path} silinemedi: {e}")
        
        # .pyc ve .pyo dosyalarını sil
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    removed_files += 1
                    total_size += file_size
                    logger.info(f"   ✅ Silindi: {file_path}")
                except Exception as e:
                    logger.error(f"   ❌ {file_path} silinemedi: {e}")
    
    return removed_dirs, removed_files, total_size

def cleanup_torch_cache():
    """PyTorch cache'lerini temizle"""
    logger.info("🔥 PyTorch cache temizliği...")
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("   ✅ CUDA cache temizlendi")
        else:
            logger.info("   ℹ️ CUDA mevcut değil")
    except ImportError:
        logger.info("   ℹ️ PyTorch yüklü değil")
    except Exception as e:
        logger.error(f"   ❌ PyTorch cache temizleme hatası: {e}")

def cleanup_huggingface_cache():
    """Hugging Face cache'lerini temizle"""
    logger.info("🤗 Hugging Face cache temizliği...")
    
    # Yaygın HF cache konumları
    hf_cache_paths = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
        os.path.expanduser("~/.cache/transformers"),
        "./.cache",
        "./cache",
        "./model_cache"
    ]
    
    removed_size = 0
    removed_items = 0
    
    for cache_path in hf_cache_paths:
        if os.path.exists(cache_path):
            try:
                # Boyutu hesapla
                for root, dirs, files in os.walk(cache_path):
                    for file in files:
                        try:
                            removed_size += os.path.getsize(os.path.join(root, file))
                            removed_items += 1
                        except:
                            pass
                
                shutil.rmtree(cache_path)
                logger.info(f"   ✅ Silindi: {cache_path}")
            except Exception as e:
                logger.error(f"   ❌ {cache_path} silinemedi: {e}")
    
    return removed_items, removed_size

def cleanup_temp_dirs():
    """Geçici dizinleri temizle"""
    logger.info("🗂️ Geçici dizin temizliği...")
    
    temp_patterns = [
        "./temp*",
        "./tmp*",
        "./*_temp",
        "./*_tmp",
        "./temp_validation*",
        "./temp_ultra_fast_test*",
        "./checkpoint*",
        "./.pytest_cache",
        "./lightning_logs",
        "./wandb"
    ]
    
    removed_items = 0
    removed_size = 0
    
    for pattern in temp_patterns:
        import glob
        for path in glob.glob(pattern):
            if os.path.exists(path):
                try:
                    # Boyutu hesapla
                    if os.path.isdir(path):
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                try:
                                    removed_size += os.path.getsize(os.path.join(root, file))
                                    removed_items += 1
                                except:
                                    pass
                        shutil.rmtree(path)
                    else:
                        removed_size += os.path.getsize(path)
                        removed_items += 1
                        os.remove(path)
                    
                    logger.info(f"   ✅ Silindi: {path}")
                except Exception as e:
                    logger.error(f"   ❌ {path} silinemedi: {e}")
    
    return removed_items, removed_size

def cleanup_log_files():
    """Log dosyalarını temizle"""
    logger.info("📋 Log dosyası temizliği...")
    
    log_patterns = [
        "./*.log",
        "./logs/*.log",
        "./**/*.log",
        "./nohup.out",
        "./debug.log",
        "./error.log",
        "./training.log"
    ]
    
    removed_items = 0
    removed_size = 0
    
    for pattern in log_patterns:
        import glob
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                try:
                    removed_size += os.path.getsize(path)
                    os.remove(path)
                    removed_items += 1
                    logger.info(f"   ✅ Silindi: {path}")
                except Exception as e:
                    logger.error(f"   ❌ {path} silinemedi: {e}")
    
    return removed_items, removed_size

def format_size(bytes_size):
    """Boyutu okunabilir formata çevir"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def main():
    """Ana temizlik fonksiyonu"""
    logger.info("🧹 CACHE TEMİZLİK SCRIPT'İ BAŞLATILIYOR")
    logger.info("=" * 60)
    
    total_removed_items = 0
    total_removed_size = 0
    
    # 1. __pycache__ temizliği
    dirs, files, size = cleanup_pycache()
    total_removed_items += dirs + files
    total_removed_size += size
    logger.info(f"   📊 __pycache__: {dirs} klasör, {files} dosya silindi")
    
    # 2. PyTorch cache temizliği
    cleanup_torch_cache()
    
    # 3. Hugging Face cache temizliği
    items, size = cleanup_huggingface_cache()
    total_removed_items += items
    total_removed_size += size
    logger.info(f"   📊 HuggingFace cache: {items} öğe silindi")
    
    # 4. Geçici dizin temizliği
    items, size = cleanup_temp_dirs()
    total_removed_items += items
    total_removed_size += size
    logger.info(f"   📊 Geçici dizinler: {items} öğe silindi")
    
    # 5. Log dosyası temizliği
    items, size = cleanup_log_files()
    total_removed_items += items
    total_removed_size += size
    logger.info(f"   📊 Log dosyaları: {items} dosya silindi")
    
    # Özet
    logger.info("=" * 60)
    logger.info("📈 TEMİZLİK ÖZETİ:")
    logger.info(f"   Toplam silinen öğe: {total_removed_items}")
    logger.info(f"   Toplam temizlenen alan: {format_size(total_removed_size)}")
    
    if total_removed_items > 0:
        logger.info("🎉 Cache temizliği başarıyla tamamlandı!")
    else:
        logger.info("ℹ️ Temizlenecek cache bulunamadı.")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n❌ Kullanıcı tarafından iptal edildi.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Beklenmeyen hata: {e}")
        sys.exit(1)
