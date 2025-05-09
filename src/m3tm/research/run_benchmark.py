"""
Benchmark Çalıştırma Aracı

Bu script, dikkat ve konvolüsyon mekanizmalarının benchmark karşılaştırmasını çalıştırır.
Komut satırından çalıştırılabilir ve çeşitli parametreleri kabul eder.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from m3tm.research.benchmark import (
    BenchmarkConfig, 
    run_attention_benchmark,
    run_convolution_benchmark
)
from m3tm.research.attention_mechanisms import get_available_mechanisms as get_available_attentions
from m3tm.research.convolution_mechanisms import get_available_mechanisms as get_available_convolutions


def main():
    """Ana çalıştırma fonksiyonu."""
    
    parser = argparse.ArgumentParser(description='Run benchmarks for attention and convolution mechanisms')
    
    # Benchmark tipi
    parser.add_argument('--type', choices=['attention', 'convolution', 'both'], default='both',
                        help='Type of mechanisms to benchmark (default: both)')
    
    # Cihaz seçimi
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                        help='Device to run benchmarks on (default: cpu)')
    
    # Kaydetme ve çıktı seçenekleri
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to save benchmark results (default: benchmark_results)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save benchmark results')
    
    # Mini benchmark modu (daha az boyut/iterasyon, daha hızlı çalışma)
    parser.add_argument('--mini', action='store_true',
                        help='Run a smaller set of benchmarks (faster)')
    
    # Benchmark parametreleri
    parser.add_argument('--num-runs', type=int, default=100,
                        help='Number of runs for timing measurement (default: 100)')
    parser.add_argument('--warmup-runs', type=int, default=10,
                        help='Number of warmup runs (default: 10)')
    
    # Boyut parametreleri
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 16],
                        help='Batch sizes to test (default: 1 4 16)')
    parser.add_argument('--input-dims', type=int, nargs='+', default=[32, 64, 128],
                        help='Input dimensions to test (default: 32 64 128)')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[32, 64, 128, 256, 512],
                        help='Sequence lengths for attention (default: 32 64 128 256 512)')
    
    # Bellek ölçüm seçenekleri
    parser.add_argument('--no-memory', action='store_true',
                        help='Do not measure memory usage')
    
    # Mekanizma filtreleme seçenekleri
    parser.add_argument('--attention-mechanisms', type=str, nargs='+', default=None,
                        help='Specific attention mechanisms to benchmark (default: all)')
    parser.add_argument('--convolution-mechanisms', type=str, nargs='+', default=None,
                        help='Specific convolution mechanisms to benchmark (default: all)')
    
    args = parser.parse_args()
    
    # CUDA kontrolü
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is not available, falling back to CPU")
        args.device = 'cpu'
    
    # MPS kontrolü (Apple Silicon)
    if args.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("Warning: MPS is not available, falling back to CPU")
        args.device = 'cpu'
    
    # Mini benchmark modu ayarları
    if args.mini:
        args.num_runs = min(args.num_runs, 20)
        args.warmup_runs = min(args.warmup_runs, 5)
        args.batch_sizes = [1]
        args.input_dims = [32]
        args.seq_lengths = [32, 128]
        print("Running in mini benchmark mode with reduced parameters")
    
    # Çıktı dizini oluştur
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Benchmark yapılandırması
    config = BenchmarkConfig(
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        device=args.device,
        batch_sizes=args.batch_sizes,
        input_dims=args.input_dims,
        seq_lengths=args.seq_lengths,
        image_sizes=[(d, d) for d in args.input_dims],  # Kare görüntüler
        measure_memory=not args.no_memory,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    # Mekanizma listelerini oluştur
    available_attentions = get_available_attentions()
    available_convolutions = get_available_convolutions()
    
    if args.attention_mechanisms:
        for mech in args.attention_mechanisms:
            if mech not in available_attentions:
                print(f"Warning: Attention mechanism '{mech}' is not available")
        attention_mechanisms = [m for m in args.attention_mechanisms if m in available_attentions]
    else:
        attention_mechanisms = available_attentions
    
    if args.convolution_mechanisms:
        for mech in args.convolution_mechanisms:
            if mech not in available_convolutions:
                print(f"Warning: Convolution mechanism '{mech}' is not available")
        convolution_mechanisms = [m for m in args.convolution_mechanisms if m in available_convolutions]
    else:
        convolution_mechanisms = available_convolutions
    
    print(f"Available attention mechanisms: {available_attentions}")
    print(f"Available convolution mechanisms: {available_convolutions}")
    print(f"Selected attention mechanisms: {attention_mechanisms}")
    print(f"Selected convolution mechanisms: {convolution_mechanisms}")
    
    # Benchmark'ları çalıştır
    start_time = time.time()
    
    if args.type in ['attention', 'both'] and attention_mechanisms:
        print("\n=== Running Attention Mechanism Benchmarks ===")
        attention_results = run_attention_benchmark(
            config=config,
            mechanisms=attention_mechanisms
        )
        print(f"Attention benchmarks completed")
    
    if args.type in ['convolution', 'both'] and convolution_mechanisms:
        print("\n=== Running Convolution Mechanism Benchmarks ===")
        convolution_results = run_convolution_benchmark(
            config=config,
            mechanisms=convolution_mechanisms
        )
        print(f"Convolution benchmarks completed")
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main() 