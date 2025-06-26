#!/usr/bin/env python3
"""
Distributed Training Launcher

Bu script, M³TM modelini distributed training ile çalıştırmak için kullanılır.
"""

import os
import sys
import argparse
import subprocess
import socket
from pathlib import Path

# M3TM root dizinini sys.path'e ekle
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def find_free_port():
    """Boş port bulur."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def run_single_node_multi_gpu(
    config_path: str,
    output_dir: str,
    num_gpus: int,
    mode: str = "full",
    master_port: int = None
):
    """
    Tek node üzerinde multi-GPU eğitimi çalıştırır.
    
    Args:
        config_path: Konfigürasyon dosyası yolu
        output_dir: Çıktı dizini
        num_gpus: GPU sayısı
        mode: Eğitim modu
        master_port: Master port
    """
    if master_port is None:
        master_port = find_free_port()
    
    print(f"🚀 Tek node multi-GPU eğitimi başlatılıyor...")
    print(f"   GPU sayısı: {num_gpus}")
    print(f"   Master port: {master_port}")
    print(f"   Konfigürasyon: {config_path}")
    print(f"   Çıktı dizini: {output_dir}")
    
    # Environment variables
    env = os.environ.copy()
    env.update({
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': str(master_port),
        'WORLD_SIZE': str(num_gpus),
    })
    
    processes = []
    
    try:
        for rank in range(num_gpus):
            # Her GPU için ayrı process
            env_rank = env.copy()
            env_rank.update({
                'RANK': str(rank),
                'LOCAL_RANK': str(rank),
                'CUDA_VISIBLE_DEVICES': str(rank)
            })
            
            cmd = [
                sys.executable, 'train.py',
                '--config', config_path,
                '--mode', mode,
                '--output-dir', output_dir,
                '--distributed',
                '--world-size', str(num_gpus),
                '--rank', str(rank),
                '--local-rank', str(rank),
                '--master-addr', 'localhost',
                '--master-port', str(master_port)
            ]
            
            print(f"   GPU {rank} başlatılıyor: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                env=env_rank,
                cwd=project_root
            )
            processes.append(process)
        
        # Tüm process'lerin bitmesini bekle
        for i, process in enumerate(processes):
            return_code = process.wait()
            if return_code != 0:
                print(f"❌ GPU {i} process'i hata ile sonlandı: {return_code}")
            else:
                print(f"✅ GPU {i} process'i başarıyla tamamlandı")
        
        print("🎉 Distributed training tamamlandı!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Kullanıcı tarafından durduruldu, process'ler sonlandırılıyor...")
        for process in processes:
            process.terminate()
        
        for process in processes:
            process.wait()
        
        print("🛑 Tüm process'ler sonlandırıldı")


def run_multi_node(
    config_path: str,
    output_dir: str,
    num_nodes: int,
    gpus_per_node: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    mode: str = "full"
):
    """
    Multi-node eğitimi çalıştırır.
    
    Args:
        config_path: Konfigürasyon dosyası yolu
        output_dir: Çıktı dizini
        num_nodes: Node sayısı
        gpus_per_node: Node başına GPU sayısı
        node_rank: Bu node'un rank'i
        master_addr: Master node adresi
        master_port: Master port
        mode: Eğitim modu
    """
    world_size = num_nodes * gpus_per_node
    
    print(f"🌐 Multi-node eğitimi başlatılıyor...")
    print(f"   Node sayısı: {num_nodes}")
    print(f"   Node başına GPU: {gpus_per_node}")
    print(f"   Bu node rank: {node_rank}")
    print(f"   Toplam world size: {world_size}")
    print(f"   Master: {master_addr}:{master_port}")
    
    # Environment variables
    env = os.environ.copy()
    env.update({
        'MASTER_ADDR': master_addr,
        'MASTER_PORT': str(master_port),
        'WORLD_SIZE': str(world_size),
    })
    
    processes = []
    
    try:
        for local_rank in range(gpus_per_node):
            global_rank = node_rank * gpus_per_node + local_rank
            
            env_rank = env.copy()
            env_rank.update({
                'RANK': str(global_rank),
                'LOCAL_RANK': str(local_rank),
                'CUDA_VISIBLE_DEVICES': str(local_rank)
            })
            
            cmd = [
                sys.executable, 'train.py',
                '--config', config_path,
                '--mode', mode,
                '--output-dir', output_dir,
                '--distributed',
                '--world-size', str(world_size),
                '--rank', str(global_rank),
                '--local-rank', str(local_rank),
                '--master-addr', master_addr,
                '--master-port', str(master_port)
            ]
            
            print(f"   Node {node_rank}, GPU {local_rank} (Global rank {global_rank}) başlatılıyor")
            
            process = subprocess.Popen(
                cmd,
                env=env_rank,
                cwd=project_root
            )
            processes.append(process)
        
        # Tüm process'lerin bitmesini bekle
        for i, process in enumerate(processes):
            return_code = process.wait()
            if return_code != 0:
                print(f"❌ Local GPU {i} process'i hata ile sonlandı: {return_code}")
            else:
                print(f"✅ Local GPU {i} process'i başarıyla tamamlandı")
        
        print("🎉 Multi-node training tamamlandı!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Kullanıcı tarafından durduruldu, process'ler sonlandırılıyor...")
        for process in processes:
            process.terminate()
        
        for process in processes:
            process.wait()
        
        print("🛑 Tüm process'ler sonlandırıldı")


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(description="M³TM Distributed Training Launcher")
    
    # Temel argümanlar
    parser.add_argument("--config", type=str, required=True, help="Konfigürasyon dosyası")
    parser.add_argument("--output-dir", type=str, default="./outputs/distributed", help="Çıktı dizini")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "adapter", "task_head"], help="Eğitim modu")
    
    # Single-node multi-GPU
    parser.add_argument("--gpus", type=int, help="GPU sayısı (single-node için)")
    
    # Multi-node
    parser.add_argument("--num-nodes", type=int, default=1, help="Node sayısı")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Node başına GPU sayısı")
    parser.add_argument("--node-rank", type=int, default=0, help="Bu node'un rank'i")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master node adresi")
    parser.add_argument("--master-port", type=int, help="Master port")
    
    args = parser.parse_args()
    
    # Port belirle
    if args.master_port is None:
        args.master_port = find_free_port()
    
    # Çıktı dizinini oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.gpus:
        # Single-node multi-GPU
        run_single_node_multi_gpu(
            config_path=args.config,
            output_dir=args.output_dir,
            num_gpus=args.gpus,
            mode=args.mode,
            master_port=args.master_port
        )
    else:
        # Multi-node
        run_multi_node(
            config_path=args.config,
            output_dir=args.output_dir,
            num_nodes=args.num_nodes,
            gpus_per_node=args.gpus_per_node,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
            mode=args.mode
        )


if __name__ == "__main__":
    main()
