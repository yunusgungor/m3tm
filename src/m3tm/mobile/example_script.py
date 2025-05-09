"""
PyTorch modelini mobil platformda kullanmak için örnek betik.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np

from m3tm.core.base_model import BaseModel
from m3tm.config.model_config import get_tiny_config, M3TMConfig
from m3tm.mobile.model_converter import ModelConverter
from m3tm.mobile.optimization import MobileOptimizer
from m3tm.mobile.benchmark import MobileBenchmark

# Demo için basit bir model
class SimpleClassifier(nn.Module):
    """Örnek amaçlı basit bir sınıflandırıcı model."""
    
    def __init__(self, input_size=64, hidden_size=32, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Örnek bir inherit edilmiş BaseModel
class SimpleBaseModel(BaseModel):
    """M3TM BaseModel'den türetilmiş basit bir model örneği."""
    
    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.text_config.embed_dim
        
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

def create_example_model(model_type="simple", save_dir="models"):
    """Örnek model oluşturur ve kaydeder."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    if model_type == "simple":
        model = SimpleClassifier()
        example_input = torch.rand(1, 64)
        model_save_path = os.path.join(save_dir, "simple_model.pt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Basit model oluşturuldu ve kaydedildi: {model_save_path}")
        
    elif model_type == "base":
        config = get_tiny_config()
        model = SimpleBaseModel(config)
        example_input = torch.rand(1, config.text_config.embed_dim)
        model_save_path = os.path.join(save_dir, "base_model_dir")
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)
        print(f"BaseModel türünde model oluşturuldu ve kaydedildi: {model_save_path}")
    
    else:
        raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
    
    return model, example_input

def convert_for_mobile_demo(model, example_input, output_dir="mobile_models"):
    """Modeli mobil için dönüştüren demo."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Temel TorchScript dönüşümü
    torchscript_path = os.path.join(output_dir, "model_torchscript.pt")
    traced_model = ModelConverter.to_torchscript(
        model,
        example_input,
        save_path=torchscript_path,
        optimize=True
    )
    
    # 2. Android için optimize edilmiş model
    android_path = os.path.join(output_dir, "model_android.pt")
    ModelConverter.convert_for_android(
        model,
        example_input,
        save_path=android_path,
        optimize=True
    )
    
    # 3. Sıkıştırılmış model (quantize + optimize)
    optimized_path = os.path.join(output_dir, "model_optimized.pt")
    optimized_model = MobileOptimizer.compress_model(
        model,
        example_input,
        methods=["quantize", "optimize"],
        save_path=optimized_path
    )
    
    # 4. Benchmark testleri
    results = MobileBenchmark.compare_models(
        {
            "Original": model,
            "TorchScript": traced_model,
            "Optimized": optimized_model
        },
        example_input,
        num_runs=50,
        warmup_runs=5
    )
    
    # 5. Rapor oluştur
    report_path = os.path.join(output_dir, "benchmark_report.md")
    MobileBenchmark.generate_report(results, report_path)
    
    return results

def load_and_run_mobile_model(model_path, example_input):
    """Kaydedilmiş bir TorchScript modelini yükler ve çalıştırır."""
    
    print(f"TorchScript modeli yükleniyor: {model_path}")
    model = torch.jit.load(model_path)
    
    print("Model yüklendi, çıkarım yapılıyor...")
    
    # Modeli değerlendirme moduna geçir ve çıkarım yap
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(example_input)
        end_time = time.time()
    
    print(f"Çıkarım tamamlandı. Çıkarım süresi: {(end_time - start_time)*1000:.2f} ms")
    print(f"Çıktı şekli: {output.shape}")
    
    return output

def main():
    parser = argparse.ArgumentParser(description="PyTorch Mobile Örnek Betik")
    parser.add_argument('--mode', choices=['create', 'convert', 'load'], default='convert',
                        help='İşlem modu: model oluştur, dönüştür veya yükle')
    parser.add_argument('--model-type', choices=['simple', 'base'], default='simple',
                        help='Oluşturulacak model tipi')
    parser.add_argument('--model-path', default='mobile_models/model_torchscript.pt',
                        help='Yüklenecek model yolu')
    parser.add_argument('--output-dir', default='mobile_models',
                        help='Çıktı dizini')
    
    args = parser.parse_args()
    
    if args.mode == 'create':
        model, example_input = create_example_model(args.model_type)
        print("Model başarıyla oluşturuldu.")
        
    elif args.mode == 'convert':
        # Önce bir model oluştur
        model, example_input = create_example_model(args.model_type)
        
        # Sonra dönüşüm yap
        results = convert_for_mobile_demo(model, example_input, args.output_dir)
        print("Model başarıyla dönüştürüldü ve benchmark sonuçları oluşturuldu.")
        
    elif args.mode == 'load':
        # Örnek girdi oluştur (modele göre değişebilir)
        if args.model_type == 'simple':
            example_input = torch.rand(1, 64)
        else:  # 'base'
            config = get_tiny_config()
            example_input = torch.rand(1, config.text_config.embed_dim)
            
        output = load_and_run_mobile_model(args.model_path, example_input)
        print("Model başarıyla yüklendi ve çalıştırıldı.")
    
if __name__ == "__main__":
    main() 