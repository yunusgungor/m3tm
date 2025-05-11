#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Android SDK testlerini çalıştırmak için yardımcı script.
Bu script test öncesi gereksinimleri kontrol eder ve gerekli testleri çalıştırır.
"""

import os
import sys
import subprocess
import argparse
import unittest
import tempfile
import json
from pathlib import Path

def check_requirements():
    """Gerekli ortam değişkenlerini ve bağımlılıkları kontrol eder."""
    print("Gereksinimler kontrol ediliyor...")
    req_script = Path(__file__).parent / "android_sdk_test_requirements.sh"
    
    if not req_script.exists():
        print(f"HATA: Gereksinim kontrol scripti bulunamadı: {req_script}")
        return False
        
    try:
        result = subprocess.run(["sh", str(req_script)], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Gereksinim kontrolü başarısız oldu: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def prepare_test_environment():
    """Test ortamını hazırlar."""
    print("Test ortamı hazırlanıyor...")
    
    # Test için gerekli model dosyasını kontrol et
    model_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model_checkpoints')))
    test_model_path = model_dir / "test_model.pt"
    
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
        print(f"Model dizini oluşturuldu: {model_dir}")
    
    # Test modeli yoksa basit bir örnek model oluştur
    if not test_model_path.exists():
        print(f"Test modeli bulunamadı: {test_model_path}")
        print("Basit bir test modeli oluşturuluyor...")
        
        try:
            import torch
            
            # Basit bir model oluştur
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super(SimpleModel, self).__init__()
                    self.fc = torch.nn.Linear(10, 2)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            torch.save(model.state_dict(), test_model_path)
            print(f"Test modeli oluşturuldu: {test_model_path}")
            
        except ImportError:
            print("UYARI: PyTorch yüklü değil, test modeli oluşturulamadı.")
            print("Testler sırasında model ile ilgili hatalar olabilir.")
    else:
        print(f"Test modeli mevcut: {test_model_path}")
    
    # Android SDK'yı derle
    print("Android SDK derleniyor...")
    android_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'android')))
    gradle_wrapper = android_dir / "gradlew"
    
    if not gradle_wrapper.exists():
        print(f"HATA: Gradle Wrapper bulunamadı: {gradle_wrapper}")
        return False
    
    try:
        result = subprocess.run(["sh", str(gradle_wrapper), "assembleDebug"], 
                               cwd=android_dir,
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        print("Android SDK derleme başarılı.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Android SDK derleme başarısız oldu: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def discover_and_run_tests(test_filter=None, verbose=False):
    """Android SDK testlerini keşfeder ve çalıştırır."""
    print("Android SDK testleri başlatılıyor...")
    
    # Test dizinini belirle
    test_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))
    
    # Test runner oluştur
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    
    # Testleri keşfet
    if test_filter:
        print(f"Filtre ile testler aranıyor: {test_filter}")
        pattern = f"*{test_filter}*.py"
        test_suite = loader.discover(test_dir, pattern=pattern)
    else:
        print("Tüm Android SDK testleri aranıyor...")
        test_suite = loader.discover(test_dir, pattern="test_android*.py")
    
    # Test sonuçlarını kaydetmek için bir dizin oluştur
    results_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'test_results')))
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    # Testleri çalıştır
    result = runner.run(test_suite)
    
    # Test sonuçlarını dışa aktar
    with open(results_dir / "android_sdk_test_results.json", "w") as f:
        results = {
            "tests": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
            "success": result.wasSuccessful()
        }
        json.dump(results, f, indent=2)
    
    print(f"\nTest Sonuçları: Toplam {result.testsRun} test, {len(result.failures)} başarısız, {len(result.errors)} hata")
    print(f"Sonuçlar kaydedildi: {results_dir / 'android_sdk_test_results.json'}")
    
    return result.wasSuccessful()

def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(description="Android SDK testlerini çalıştırır.")
    parser.add_argument("--skip-requirements", action="store_true", help="Gereksinim kontrolünü atla")
    parser.add_argument("--skip-build", action="store_true", help="Android SDK derlemeyi atla")
    parser.add_argument("--filter", type=str, help="Test adı filtresi")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ayrıntılı çıktı")
    
    args = parser.parse_args()
    
    # Gereksinimleri kontrol et
    if not args.skip_requirements:
        if not check_requirements():
            print("Gereksinim kontrolü başarısız oldu. Testler çalıştırılamıyor.")
            return 1
    
    # Test ortamını hazırla
    if not args.skip_build:
        if not prepare_test_environment():
            print("Test ortamı hazırlanamadı. Testler çalıştırılamıyor.")
            return 1
    
    # Testleri çalıştır
    success = discover_and_run_tests(args.filter, args.verbose)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 