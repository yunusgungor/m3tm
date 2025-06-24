"""
Mobil platformlarda model performansını ölçmek için benchmark araçları.
"""

import time
from typing import Dict, List, Union, Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from m3tm.core.base_model import BaseModel


class MobileBenchmark:
    """Modellerin mobil platformlardaki performansını ölçen sınıf."""
    
    @staticmethod
    def benchmark_inference_time(
        model: Union[nn.Module, BaseModel, torch.jit.ScriptModule],
        example_inputs: Any,
        num_runs: int = 100,
        warmup_runs: int = 10,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Model çıkarım süresini ölçer.
        
        Args:
            model: Benchmark yapılacak model
            example_inputs: Örnek girdiler
            num_runs: Kaç kez ölçüm yapılacağı
            warmup_runs: Ölçüm öncesi ısınma turu sayısı
            device: Çalıştırılacak cihaz ("cpu" veya "cuda")
            
        Returns:
            Ölçüm sonuçları içeren sözlük
        """
        if isinstance(model, torch.jit.ScriptModule):
            is_torchscript = True
        else:
            is_torchscript = False
            
        # Cihazı ayarla
        if device == "cuda" and torch.cuda.is_available():
            if not is_torchscript:
                model = model.cuda()
            if torch.is_tensor(example_inputs):
                example_inputs = example_inputs.cuda()
            elif isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
                example_inputs = [x.cuda() if torch.is_tensor(x) else x for x in example_inputs]
        
        # Önce modeli ısınma turlarında çalıştır
        with torch.no_grad():
            for _ in range(warmup_runs):
                if is_torchscript:
                    if isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
                        model(*example_inputs)
                    else:
                        model(example_inputs)
                else:
                    model(example_inputs)
        
        # Zamanlamayı ölç
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                
                if is_torchscript:
                    if isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
                        model(*example_inputs)
                    else:
                        model(example_inputs)
                else:
                    model(example_inputs)
                    
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # ms cinsinden
        
        # İstatistikleri hesapla
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        std_latency = np.std(latencies)
        
        return {
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "std_latency_ms": std_latency,
            "num_runs": num_runs
        }
    
    @staticmethod
    def benchmark_memory_usage(
        model: Union[nn.Module, BaseModel, torch.jit.ScriptModule],
        example_inputs: Any,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Model bellek kullanımını ölçer.
        
        Args:
            model: Benchmark yapılacak model
            example_inputs: Örnek girdiler
            device: Çalıştırılacak cihaz ("cpu" veya "cuda")
            
        Returns:
            Bellek kullanım istatistikleri
        """
        # TorchScript model için model boyutu
        if isinstance(model, torch.jit.ScriptModule):
            model_size_mb = 0  # TorchScript model boyutunu ölçme yöntemi platform bağımlıdır
            is_torchscript = True
        else:
            # PyTorch modeli için parametre sayısını ve boyutunu hesapla
            num_params = sum(p.numel() for p in model.parameters())
            model_size_mb = num_params * 4 / (1024 * 1024)  # 4 bytes per float32 parametre
            is_torchscript = False
        
        # Cihazı ayarla
        if device == "cuda" and torch.cuda.is_available():
            if not is_torchscript:
                model = model.cuda()
            if torch.is_tensor(example_inputs):
                example_inputs = example_inputs.cuda()
            elif isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
                example_inputs = [x.cuda() if torch.is_tensor(x) else x for x in example_inputs]
                
            # CUDA bellek kullanımını ölç
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
            # Çıkarım yap
            with torch.no_grad():
                if is_torchscript:
                    if isinstance(example_inputs, tuple) or isinstance(example_inputs, list):
                        model(*example_inputs)
                    else:
                        model(example_inputs)
                else:
                    model(example_inputs)
            
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            memory_stats = {
                "model_size_mb": model_size_mb,
                "peak_memory_usage_mb": peak_memory_mb,
                "device": device
            }
        else:
            # CPU bellek ölçümü - gerçek mobil platformlarda farklı yöntemler kullanılacak
            memory_stats = {
                "model_size_mb": model_size_mb,
                "device": device,
                "note": "CPU bellek kullanımı tam olarak ölçülemedi. Gerçek cihazda test edilmeli."
            }
        
        return memory_stats
    
    @staticmethod
    def compare_models(
        models: Dict[str, Union[nn.Module, BaseModel, torch.jit.ScriptModule]],
        example_inputs: Any,
        num_runs: int = 100,
        warmup_runs: int = 10,
        device: str = "cpu"
    ) -> Dict[str, Dict[str, float]]:
        """
        Birden fazla modeli karşılaştırır.
        
        Args:
            models: İsim/model ikilisi içeren sözlük
            example_inputs: Örnek girdiler
            num_runs: Kaç kez ölçüm yapılacağı
            warmup_runs: Isınma turu sayısı
            device: Çalıştırılacak cihaz
            
        Returns:
            Her model için ölçüm sonuçlarını içeren sözlük
        """
        results = {}
        
        for name, model in models.items():
            print(f"{name} modeli için benchmark başlatılıyor...")
            
            # Çıkarım zamanı ve bellek kullanımı
            latency_results = MobileBenchmark.benchmark_inference_time(
                model, 
                example_inputs, 
                num_runs, 
                warmup_runs, 
                device
            )
            
            memory_results = MobileBenchmark.benchmark_memory_usage(
                model, 
                example_inputs, 
                device
            )
            
            # Sonuçları birleştir
            model_results = {**latency_results, **memory_results}
            results[name] = model_results
            
            print(f"{name} benchmark sonuçları:")
            print(f"  Ortalama çıkarım süresi: {model_results.get('mean_latency_ms', 'N/A'):.2f} ms")
            print(f"  Model boyutu: {model_results.get('model_size_mb', 'N/A'):.2f} MB")
            if 'peak_memory_usage_mb' in model_results:
                print(f"  Tepe bellek kullanımı: {model_results.get('peak_memory_usage_mb'):.2f} MB")
            print("")
            
        return results
    
    @staticmethod
    def generate_report(
        results: Dict[str, Dict[str, float]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Benchmark sonuçlarından rapor oluşturur.
        
        Args:
            results: Benchmark sonuçları
            output_path: Raporun kaydedileceği dosya yolu
            
        Returns:
            Oluşturulan rapor metni
        """
        report = "# M³TM Mobil Benchmark Raporu\n\n"
        report += f"Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Çıkarım Süreleri (ms)\n\n"
        report += "| Model | Ortalama | Ortanca | P95 | Min | Max | Std. Sapma |\n"
        report += "|-------|----------|---------|-----|-----|-----|------------|\n"
        
        for name, result in results.items():
            mean = result.get('mean_latency_ms', 'N/A')
            median = result.get('median_latency_ms', 'N/A')
            p95 = result.get('p95_latency_ms', 'N/A')
            min_lat = result.get('min_latency_ms', 'N/A')
            max_lat = result.get('max_latency_ms', 'N/A')
            std = result.get('std_latency_ms', 'N/A')
            
            report += f"| {name} | {mean:.2f} | {median:.2f} | {p95:.2f} | {min_lat:.2f} | {max_lat:.2f} | {std:.2f} |\n"
        
        report += "\n## Bellek Kullanımı (MB)\n\n"
        report += "| Model | Model Boyutu | Tepe Bellek Kullanımı | Cihaz |\n"
        report += "|-------|-------------|------------------------|-------|\n"
        
        for name, result in results.items():
            model_size = result.get('model_size_mb', 'N/A')
            if isinstance(model_size, float):
                model_size = f"{model_size:.2f}"
                
            peak_memory = result.get('peak_memory_usage_mb', 'N/A')
            if isinstance(peak_memory, float):
                peak_memory = f"{peak_memory:.2f}"
                
            device = result.get('device', 'N/A')
            
            report += f"| {name} | {model_size} | {peak_memory} | {device} |\n"
        
        report += "\n## Notlar\n\n"
        report += "- Ölçümler, her model için aynı girdilerle ve aynı cihazda yapılmıştır.\n"
        report += "- Bellek kullanımı değerleri gerçek mobil cihazlarda farklılık gösterebilir.\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Rapor başarıyla kaydedildi: {output_path}")
            
        return report 


def benchmark_model(
    model: Union[nn.Module, BaseModel, torch.jit.ScriptModule],
    example_inputs: Any,
    model_name: str = "model",
    device: str = "cpu",
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, Any]:
    """
    Legacy function - Model benchmarking için kullanılan ana fonksiyon.
    
    Args:
        model: Benchmark yapılacak model
        example_inputs: Örnek girdiler  
        model_name: Model adı
        device: Çalıştırılacak cihaz
        num_runs: Ölçüm sayısı
        warmup_runs: Isınma turu sayısı
        
    Returns:
        Benchmark sonuçları
    """
    benchmark = MobileBenchmark()
    
    # Çıkarım süresi benchmarkı
    latency_results = benchmark.benchmark_inference_time(
        model=model,
        example_inputs=example_inputs,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        device=device
    )
    
    # Bellek kullanımı benchmarkı
    memory_results = benchmark.benchmark_memory_usage(
        model=model,
        example_inputs=example_inputs,
        device=device
    )
    
    # Model boyutu benchmarkı
    size_results = benchmark.benchmark_model_size(model)
    
    # Sonuçları birleştir
    results = {
        "model_name": model_name,
        "device": device,
        **latency_results,
        **memory_results,
        **size_results
    }
    
    return results