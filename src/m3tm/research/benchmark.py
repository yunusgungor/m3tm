"""
Araştırma Mekanizmaları Karşılaştırma Aracı

Bu modül, farklı dikkat ve konvolüsyon mekanizmalarının 
performans karşılaştırması için kullanılan benchmark 
işlevleri ve yardımcı araçları içerir.

Örüntü: BenchmarkStrategy (PT-004)
"""

import time
import math
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from m3tm.research.attention_mechanisms import get_attention_mechanism_by_name, get_available_mechanisms as get_available_attentions
from m3tm.research.convolution_mechanisms import get_convolution_mechanism_by_name, get_available_mechanisms as get_available_convolutions


@dataclass
class BenchmarkConfig:
    """Benchmark yapılandırması."""
    num_runs: int = 100  # Kaç defa çalıştırılacağı
    warmup_runs: int = 10  # Isınma turları sayısı
    device: str = "cpu"  # Çalıştırılacak cihaz (cpu, cuda, mps)
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 16])  # Test edilecek batch boyutları
    input_dims: List[int] = field(default_factory=lambda: [32, 64, 128])  # Test edilecek girdi boyutları
    seq_lengths: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])  # Test edilecek sekans uzunlukları (dikkat için)
    image_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(32, 32), (64, 64), (128, 128)])  # Test edilecek görüntü boyutları
    measure_memory: bool = True  # Bellek kullanımını ölçme
    save_results: bool = True  # Sonuçları kaydetme
    output_dir: str = "benchmark_results"  # Sonuçların kaydedileceği dizin
    
    def __post_init__(self):
        """Yapılandırma tutarlılığını kontrol eder."""
        # Desteklenen cihaz olduğundan emin ol
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Unsupported device: {self.device}")
        
        # CUDA veya MPS kullanılacaksa, kullanılabilir olduğundan emin ol
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            self.device = "cpu"
        
        if self.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("MPS is not available, falling back to CPU")
            self.device = "cpu"
        
        # Çıktı dizininin var olduğundan emin ol
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)


class BenchmarkResults:
    """Benchmark sonuçlarını saklama ve analiz etme sınıfı."""
    
    def __init__(self, name: str, config: BenchmarkConfig):
        """
        Args:
            name: Benchmark adı (ör. "attention_mechanisms")
            config: Benchmark yapılandırması
        """
        self.name = name
        self.config = config
        self.results = {}
        self.metrics = {}
    
    def add_result(self, mechanism_name: str, params: Dict[str, Any], metrics: Dict[str, Any]):
        """Yeni bir benchmark sonucu ekler.
        
        Args:
            mechanism_name: Mekanizma adı
            params: Benchmark parametreleri (batch_size, dim, seq_len vb.)
            metrics: Ölçülen metrikler (latency, memory, throughput vb.)
        """
        if mechanism_name not in self.results:
            self.results[mechanism_name] = []
        
        self.results[mechanism_name].append({
            "params": params,
            "metrics": metrics
        })
    
    def add_mechanism_metrics(self, mechanism_name: str, metrics: Dict[str, Any]):
        """Mekanizma hakkında genel metrikler ekler (parametre sayısı, FLOP vb.)
        
        Args:
            mechanism_name: Mekanizma adı
            metrics: Mekanizma metrikleri
        """
        self.metrics[mechanism_name] = metrics
    
    def save(self):
        """Benchmark sonuçlarını JSON dosyası olarak kaydeder."""
        if not self.config.save_results:
            return
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.name}_benchmark_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                "name": self.name,
                "config": {k: str(v) if isinstance(v, (list, tuple, dict)) and k != "device" else v 
                           for k, v in self.config.__dict__.items()},
                "results": self.results,
                "metrics": self.metrics,
                "timestamp": timestamp
            }, f, indent=2)
        
        print(f"Benchmark results saved to {filepath}")
        return filepath
    
    def generate_comparison_plots(self):
        """Karşılaştırma grafikleri oluşturur ve kaydeder."""
        if not self.config.save_results or not self.results:
            return
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_dir = os.path.join(self.config.output_dir, f"{self.name}_plots_{timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Farklı boyutlara göre latency karşılaştırma grafikleri
        self._plot_latency_vs_size(plot_dir)
        
        # Performans/parametre karşılaştırması
        self._plot_performance_vs_params(plot_dir)
        
        # Bellek kullanımı karşılaştırması (eğer ölçüldüyse)
        if self.config.measure_memory:
            self._plot_memory_usage(plot_dir)
        
        # Tüm mekanizmalar için genel karşılaştırma grafiği
        self._plot_overall_comparison(plot_dir)
        
        print(f"Comparison plots saved to {plot_dir}")
        return plot_dir
    
    def _plot_latency_vs_size(self, plot_dir: str):
        """Boyutlara göre latency grafiği oluşturur.
        
        Args:
            plot_dir: Grafiklerin kaydedileceği dizin
        """
        # Basit implementasyon (gerçekte daha karmaşık olabilir)
        plt.figure(figsize=(10, 6))
        
        for mechanism_name, results in self.results.items():
            # Farklı boyutlara göre ortalama gecikme sürelerini topla
            sizes = []
            latencies = []
            
            for result in results:
                # Karşılaştırma boyutu (seq_len, dim vb.) - bu basitleştirilmiş
                if "seq_len" in result["params"]:
                    size = result["params"]["seq_len"]
                elif "image_size" in result["params"]:
                    size = result["params"]["image_size"][0]  # Kare görüntü varsayar
                else:
                    size = result["params"].get("dim", 0)
                
                latency = result["metrics"].get("mean_latency_ms", 0)
                
                sizes.append(size)
                latencies.append(latency)
            
            # Boyuta göre sırala
            sorted_data = sorted(zip(sizes, latencies))
            if sorted_data:
                sizes, latencies = zip(*sorted_data)
                plt.plot(sizes, latencies, marker='o', label=mechanism_name)
        
        plt.xlabel('Input Size')
        plt.ylabel('Latency (ms)')
        plt.title(f'{self.name.capitalize()} Mechanisms: Latency vs Size')
        plt.legend()
        plt.grid(True)
        
        # Grafik dosyasını kaydet
        plt.savefig(os.path.join(plot_dir, 'latency_vs_size.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_vs_params(self, plot_dir: str):
        """Parametre sayısına göre performans grafiği.
        
        Args:
            plot_dir: Grafiklerin kaydedileceği dizin
        """
        # Basit implementasyon
        plt.figure(figsize=(10, 6))
        
        labels = []
        param_counts = []
        mean_latencies = []
        
        for mechanism_name, metric in self.metrics.items():
            param_count = metric.get("parameter_count", 0)
            
            # Bu mekanizmanın tüm testlerinden ortalama latency'yi hesapla
            all_latencies = [r["metrics"].get("mean_latency_ms", 0) 
                            for r in self.results.get(mechanism_name, [])]
            if all_latencies:
                mean_latency = sum(all_latencies) / len(all_latencies)
            else:
                mean_latency = 0
            
            labels.append(mechanism_name)
            param_counts.append(param_count)
            mean_latencies.append(mean_latency)
        
        # Parametre sayısına göre scatter plot
        plt.scatter(param_counts, mean_latencies, s=100)
        
        # Noktaları etiketle
        for i, label in enumerate(labels):
            plt.annotate(label, (param_counts[i], mean_latencies[i]),
                        textcoords="offset points", xytext=(0, 10), ha='center')
        
        plt.xlabel('Parameter Count')
        plt.ylabel('Mean Latency (ms)')
        plt.title(f'{self.name.capitalize()} Mechanisms: Performance vs Parameters')
        plt.grid(True)
        
        # Y ekseni, daha iyi görselleştirme için sıfırdan başlasın
        plt.ylim(bottom=0)
        
        # Eğer parametre sayıları arasında büyük fark varsa, logaritmik ölçek kullan
        if max(param_counts) / (min(param_counts) + 1) > 100:
            plt.xscale('log')
        
        # Grafik dosyasını kaydet
        plt.savefig(os.path.join(plot_dir, 'performance_vs_params.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, plot_dir: str):
        """Bellek kullanımı karşılaştırma grafiği.
        
        Args:
            plot_dir: Grafiklerin kaydedileceği dizin
        """
        # Gerçek uygulamada memroy kullanım ölçümü daha karmaşık olacaktır
        # Bu basitleştirilmiş bir örnek
        plt.figure(figsize=(10, 6))
        
        mechanisms = list(self.results.keys())
        memory_usages = []
        
        for mechanism in mechanisms:
            # Her mekanizma için ortalama bellek kullanımını hesapla
            mem_usages = []
            for result in self.results[mechanism]:
                # Teorik bellek kullanımı (gerçek ölçüm farklı olabilir)
                if "memory_tokens_complexity" in result["metrics"]:
                    mem_usages.append(result["metrics"]["memory_tokens_complexity"])
                elif "memory_complexity" in result["metrics"]:
                    mem_usages.append(result["metrics"]["memory_complexity"])
            
            if mem_usages:
                memory_usages.append(sum(mem_usages) / len(mem_usages))
            else:
                memory_usages.append(0)
        
        # Bar plot oluştur
        plt.bar(mechanisms, memory_usages)
        plt.xlabel('Mechanism')
        plt.ylabel('Memory Complexity (theoretical)')
        plt.title(f'{self.name.capitalize()} Mechanisms: Memory Usage Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Grafik dosyasını kaydet
        plt.savefig(os.path.join(plot_dir, 'memory_usage.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_comparison(self, plot_dir: str):
        """Tüm mekanizmalar için genel karşılaştırma grafiği.
        
        Args:
            plot_dir: Grafiklerin kaydedileceği dizin
        """
        # Bu grafik, normalleştirilmiş performans, bellek, ve parametre metriklerini gösterir
        plt.figure(figsize=(12, 8))
        
        mechanisms = list(self.results.keys())
        num_mechanisms = len(mechanisms)
        
        # Metrikler: Latency, Bellek Kullanımı, Parametre Sayısı, ve İşlem Karmaşıklığı
        metrics = ['Latency', 'Memory', 'Parameters', 'Compute']
        num_metrics = len(metrics)
        
        # Radar grafiği için açılar
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Grafiği kapatmak için ilk açıyı ekle
        
        # Çokgen için eksenler oluştur
        ax = plt.subplot(111, polar=True)
        
        # Eksen etiketleri
        plt.xticks(angles[:-1], metrics)
        
        # Y eksenini gizle
        ax.set_yticklabels([])
        
        # Her mekanizma için değerleri hesapla ve çiz
        for i, mechanism in enumerate(mechanisms):
            # Metrik değerlerini topla
            latencies = [r["metrics"].get("mean_latency_ms", 0) for r in self.results[mechanism]]
            memories = []
            computes = []
            
            for result in self.results[mechanism]:
                if "memory_tokens_complexity" in result["metrics"]:
                    memories.append(result["metrics"]["memory_tokens_complexity"])
                elif "memory_complexity" in result["metrics"]:
                    memories.append(result["metrics"]["memory_complexity"])
                
                if "compute_complexity" in result["metrics"]:
                    computes.append(result["metrics"]["compute_complexity"])
            
            param_count = self.metrics.get(mechanism, {}).get("parameter_count", 0)
            
            # Ortalama değerleri hesapla (veya varsayılan 0)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            avg_memory = sum(memories) / len(memories) if memories else 0
            avg_compute = sum(computes) / len(computes) if computes else 0
            
            # Değerleri normalleştirmek için, tüm mekanizmaların değerlerini topla
            all_latencies = [sum([r["metrics"].get("mean_latency_ms", 0) for r in self.results[m]]) / 
                            len(self.results[m]) for m in mechanisms]
            all_memories = []
            all_computes = []
            all_params = []
            
            for m in mechanisms:
                m_memories = []
                m_computes = []
                
                for result in self.results[m]:
                    if "memory_tokens_complexity" in result["metrics"]:
                        m_memories.append(result["metrics"]["memory_tokens_complexity"])
                    elif "memory_complexity" in result["metrics"]:
                        m_memories.append(result["metrics"]["memory_complexity"])
                    
                    if "compute_complexity" in result["metrics"]:
                        m_computes.append(result["metrics"]["compute_complexity"])
                
                if m_memories:
                    all_memories.append(sum(m_memories) / len(m_memories))
                else:
                    all_memories.append(0)
                
                if m_computes:
                    all_computes.append(sum(m_computes) / len(m_computes))
                else:
                    all_computes.append(0)
                
                all_params.append(self.metrics.get(m, {}).get("parameter_count", 0))
            
            # Maksimum değerleri bul
            max_latency = max(all_latencies) if all_latencies else 1
            max_memory = max(all_memories) if all_memories else 1
            max_param = max(all_params) if all_params else 1
            max_compute = max(all_computes) if all_computes else 1
            
            # Normalleştirilmiş metrik değerleri (değer düşükse, performans iyidir)
            # Ters çevir: 1 - (değer / maksimum)
            norm_latency = 1 - (avg_latency / max_latency if max_latency > 0 else 0)
            norm_memory = 1 - (avg_memory / max_memory if max_memory > 0 else 0)
            norm_param = 1 - (param_count / max_param if max_param > 0 else 0)
            norm_compute = 1 - (avg_compute / max_compute if max_compute > 0 else 0)
            
            # Çokgen için değerler
            values = [norm_latency, norm_memory, norm_param, norm_compute]
            values += values[:1]  # Grafiği kapatmak için ilk değeri ekle
            
            # Çokgen çiz
            ax.plot(angles, values, linewidth=2, label=mechanism)
            ax.fill(angles, values, alpha=0.1)
        
        # Lejant ekle
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title(f'{self.name.capitalize()} Mechanisms: Overall Comparison\n(Higher is better)')
        
        # Grafik dosyasını kaydet
        plt.savefig(os.path.join(plot_dir, 'overall_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


class MechanismBenchmarker:
    """Mekanizmaların benchmark işlemlerini yürüten sınıf."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Args:
            config: Benchmark yapılandırması
        """
        self.config = config
        self.device = torch.device(config.device)
    
    def benchmark_attention_mechanisms(self, mechanisms: List[str] = None) -> BenchmarkResults:
        """Dikkat mekanizmalarını benchmark yapar.
        
        Args:
            mechanisms: Test edilecek dikkat mekanizmaları listesi
                        None ise, tüm kayıtlı mekanizmalar test edilir
        
        Returns:
            Benchmark sonuçları
        """
        if mechanisms is None:
            mechanisms = get_available_attentions()
        
        results = BenchmarkResults("attention_mechanisms", self.config)
        
        for mechanism_name in mechanisms:
            print(f"Benchmarking attention mechanism: {mechanism_name}")
            
            for batch_size in self.config.batch_sizes:
                for input_dim in self.config.input_dims:
                    for seq_len in self.config.seq_lengths:
                        # Mekanizma ve yapılandırma oluştur
                        mech_class, config = get_attention_mechanism_by_name(
                            mechanism_name, 
                            input_dim=input_dim,
                            num_heads=min(8, input_dim // 8)
                        )
                        model = mech_class(config).to(self.device)
                        
                        # Örnek girdi oluştur
                        x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
                        mask = None  # İhtiyaç halinde maske eklenebilir
                        
                        # Isınma turları
                        with torch.no_grad():
                            for _ in range(self.config.warmup_runs):
                                _ = model(x, mask)
                        
                        # Benchmark ölçümleri
                        latencies = []
                        with torch.no_grad():
                            for _ in range(self.config.num_runs):
                                start_time = time.time()
                                output, model_metrics = model(x, mask)
                                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                                latencies.append((time.time() - start_time) * 1000)  # ms
                        
                        # İstatistikler
                        metrics = {
                            "mean_latency_ms": np.mean(latencies),
                            "median_latency_ms": np.median(latencies),
                            "min_latency_ms": np.min(latencies),
                            "max_latency_ms": np.max(latencies),
                            "std_latency_ms": np.std(latencies),
                            "p95_latency_ms": np.percentile(latencies, 95),
                            "throughput_seqs_per_sec": 1000 * batch_size / np.mean(latencies),
                        }
                        
                        # Model metriklerini ekle
                        metrics.update({k: v for k, v in model_metrics.items()})
                        
                        # Bellek kullanımı ölçümü
                        if self.config.measure_memory and self.device.type == 'cuda':
                            torch.cuda.reset_peak_memory_stats()
                            with torch.no_grad():
                                _ = model(x, mask)
                            cuda_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                            metrics["peak_cuda_memory_mb"] = cuda_memory
                        
                        # Sonucu ekle
                        results.add_result(
                            mechanism_name,
                            {
                                "batch_size": batch_size,
                                "dim": input_dim,
                                "seq_len": seq_len
                            },
                            metrics
                        )
            
            # Mekanizma genel metriklerini ekle
            # Bu örnekte basit olarak parametre sayısını ekliyoruz
            mech_class, config = get_attention_mechanism_by_name(
                mechanism_name, 
                input_dim=64,
                num_heads=8
            )
            model = mech_class(config)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results.add_mechanism_metrics(mechanism_name, {
                "parameter_count": param_count,
                "type": "attention"
            })
        
        # Sonuçları kaydet ve grafikleri oluştur
        if self.config.save_results:
            results.save()
            results.generate_comparison_plots()
        
        return results
    
    def benchmark_convolution_mechanisms(self, mechanisms: List[str] = None) -> BenchmarkResults:
        """Konvolüsyon mekanizmalarını benchmark yapar.
        
        Args:
            mechanisms: Test edilecek konvolüsyon mekanizmaları listesi
                        None ise, tüm kayıtlı mekanizmalar test edilir
        
        Returns:
            Benchmark sonuçları
        """
        if mechanisms is None:
            mechanisms = get_available_convolutions()
        
        results = BenchmarkResults("convolution_mechanisms", self.config)
        
        for mechanism_name in mechanisms:
            print(f"Benchmarking convolution mechanism: {mechanism_name}")
            
            for batch_size in self.config.batch_sizes:
                for in_channels in self.config.input_dims:
                    out_channels = in_channels  # Çoğu testler için aynı kanal sayısı
                    
                    for img_size in self.config.image_sizes:
                        height, width = img_size
                        
                        # Mekanizma ve yapılandırma oluştur
                        mech_class, config = get_convolution_mechanism_by_name(
                            mechanism_name, 
                            in_channels=in_channels,
                            out_channels=out_channels
                        )
                        model = mech_class(config).to(self.device)
                        
                        # Örnek girdi oluştur
                        x = torch.randn(batch_size, in_channels, height, width).to(self.device)
                        
                        # Isınma turları
                        with torch.no_grad():
                            for _ in range(self.config.warmup_runs):
                                _ = model(x)
                        
                        # Benchmark ölçümleri
                        latencies = []
                        with torch.no_grad():
                            for _ in range(self.config.num_runs):
                                start_time = time.time()
                                output, model_metrics = model(x)
                                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                                latencies.append((time.time() - start_time) * 1000)  # ms
                        
                        # İstatistikler
                        metrics = {
                            "mean_latency_ms": np.mean(latencies),
                            "median_latency_ms": np.median(latencies),
                            "min_latency_ms": np.min(latencies),
                            "max_latency_ms": np.max(latencies),
                            "std_latency_ms": np.std(latencies),
                            "p95_latency_ms": np.percentile(latencies, 95),
                            "throughput_imgs_per_sec": 1000 * batch_size / np.mean(latencies),
                        }
                        
                        # Model metriklerini ekle
                        metrics.update({k: v for k, v in model_metrics.items()})
                        
                        # Bellek kullanımı ölçümü
                        if self.config.measure_memory and self.device.type == 'cuda':
                            torch.cuda.reset_peak_memory_stats()
                            with torch.no_grad():
                                _ = model(x)
                            cuda_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                            metrics["peak_cuda_memory_mb"] = cuda_memory
                        
                        # Sonucu ekle
                        results.add_result(
                            mechanism_name,
                            {
                                "batch_size": batch_size,
                                "in_channels": in_channels,
                                "out_channels": out_channels,
                                "image_size": img_size
                            },
                            metrics
                        )
            
            # Mekanizma genel metriklerini ekle
            mech_class, config = get_convolution_mechanism_by_name(
                mechanism_name, 
                in_channels=64,
                out_channels=64
            )
            model = mech_class(config)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results.add_mechanism_metrics(mechanism_name, {
                "parameter_count": param_count,
                "type": "convolution"
            })
        
        # Sonuçları kaydet ve grafikleri oluştur
        if self.config.save_results:
            results.save()
            results.generate_comparison_plots()
        
        return results


def run_attention_benchmark(config: BenchmarkConfig = None, mechanisms: List[str] = None) -> BenchmarkResults:
    """Dikkat mekanizmaları benchmark'ını çalıştırır.
    
    Args:
        config: Benchmark yapılandırması, None ise varsayılan kullanılır
        mechanisms: Test edilecek mekanizmalar listesi, None ise tümü kullanılır
    
    Returns:
        Benchmark sonuçları
    """
    if config is None:
        config = BenchmarkConfig()
    
    benchmarker = MechanismBenchmarker(config)
    return benchmarker.benchmark_attention_mechanisms(mechanisms)


def run_convolution_benchmark(config: BenchmarkConfig = None, mechanisms: List[str] = None) -> BenchmarkResults:
    """Konvolüsyon mekanizmaları benchmark'ını çalıştırır.
    
    Args:
        config: Benchmark yapılandırması, None ise varsayılan kullanılır
        mechanisms: Test edilecek mekanizmalar listesi, None ise tümü kullanılır
    
    Returns:
        Benchmark sonuçları
    """
    if config is None:
        config = BenchmarkConfig()
    
    benchmarker = MechanismBenchmarker(config)
    return benchmarker.benchmark_convolution_mechanisms(mechanisms) 