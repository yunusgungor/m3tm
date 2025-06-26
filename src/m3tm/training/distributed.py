"""
Distributed Training Desteği

Bu modül, M³TM modeli için multi-GPU ve multi-node distributed training
desteği sağlar.
"""

import os
import logging
import socket
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DistributedConfig:
    """Distributed training konfigürasyonu."""
    enabled: bool = False
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # Multi-node ayarları
    node_rank: int = 0
    num_nodes: int = 1
    gpus_per_node: int = 1
    
    # Performans ayarları
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False


class DistributedManager:
    """Distributed training yöneticisi."""
    
    def __init__(self, config: DistributedConfig):
        """
        DistributedManager'ı başlatır.
        
        Args:
            config: Distributed konfigürasyonu
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
        self.local_rank = config.local_rank
        self.world_size = config.world_size
        self.rank = config.rank
        
    def setup(self) -> None:
        """Distributed training'i kurar."""
        if not self.config.enabled:
            self.logger.info("Distributed training devre dışı")
            return
        
        # Environment variables ayarla
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.rank)
        os.environ['LOCAL_RANK'] = str(self.config.local_rank)
        
        # CUDA device ayarla
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")
            self.logger.warning("CUDA mevcut değil, CPU kullanılıyor")
        
        # Process group başlat
        try:
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            self.is_initialized = True
            
            self.logger.info(
                f"Distributed training başlatıldı - "
                f"Rank: {self.rank}/{self.world_size}, "
                f"Local Rank: {self.local_rank}, "
                f"Backend: {self.config.backend}, "
                f"Device: {device}"
            )
            
        except Exception as e:
            self.logger.error(f"Distributed training başlatılamadı: {e}")
            raise
    
    def cleanup(self) -> None:
        """Distributed training'i temizler."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            self.logger.info("Distributed training temizlendi")
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Modeli DDP ile sarar.
        
        Args:
            model: Sarılacak model
            
        Returns:
            DDP ile sarılmış model
        """
        if not self.config.enabled or not self.is_initialized:
            return model
        
        # Model'i doğru device'a taşı
        if torch.cuda.is_available():
            model = model.to(f"cuda:{self.local_rank}")
        
        # DDP ile sar
        ddp_model = DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph
        )
        
        self.logger.info(f"Model DDP ile sarıldı - Device: cuda:{self.local_rank}")
        return ddp_model
    
    def create_distributed_sampler(
        self, 
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
        drop_last: bool = False
    ) -> Optional[DistributedSampler]:
        """
        Distributed sampler oluşturur.
        
        Args:
            dataset: Veri seti
            shuffle: Karıştırma
            drop_last: Son batch'i düşür
            
        Returns:
            DistributedSampler veya None
        """
        if not self.config.enabled or not self.is_initialized:
            return None
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last
        )
        
        return sampler
    
    def create_distributed_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
    ) -> DataLoader:
        """
        Distributed DataLoader oluşturur.
        
        Args:
            dataset: Veri seti
            batch_size: Batch boyutu
            shuffle: Karıştırma
            num_workers: Worker sayısı
            pin_memory: Pin memory
            drop_last: Son batch'i düşür
            **kwargs: Ek DataLoader parametreleri
            
        Returns:
            DataLoader
        """
        # Distributed sampler oluştur
        sampler = self.create_distributed_sampler(dataset, shuffle, drop_last)
        
        # Eğer distributed sampler varsa shuffle=False olmalı
        if sampler is not None:
            shuffle = False
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
        
        return dataloader
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """
        All-reduce operasyonu.
        
        Args:
            tensor: Tensor
            op: Reduce operasyonu
            
        Returns:
            Reduced tensor
        """
        if not self.config.enabled or not self.is_initialized:
            return tensor
        
        dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        All-gather operasyonu.
        
        Args:
            tensor: Tensor
            
        Returns:
            Gathered tensor
        """
        if not self.config.enabled or not self.is_initialized:
            return tensor
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        
        return torch.cat(tensor_list, dim=0)
    
    def barrier(self) -> None:
        """Process barrier."""
        if self.config.enabled and self.is_initialized:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """Ana process mi kontrol eder."""
        return self.rank == 0
    
    def get_device(self) -> torch.device:
        """Mevcut device'ı döndürür."""
        if torch.cuda.is_available() and self.config.enabled:
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")
    
    def reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Metrikleri tüm process'ler arasında ortalar.
        
        Args:
            metrics: Metrik sözlüğü
            
        Returns:
            Ortalanmış metrikler
        """
        if not self.config.enabled or not self.is_initialized:
            return metrics
        
        reduced_metrics = {}
        for name, value in metrics.items():
            tensor = torch.tensor(value, device=self.get_device())
            self.all_reduce(tensor)
            reduced_metrics[name] = (tensor / self.world_size).item()
        
        return reduced_metrics


def auto_detect_distributed_config() -> DistributedConfig:
    """
    Otomatik olarak distributed konfigürasyonu algılar.
    
    Returns:
        DistributedConfig
    """
    config = DistributedConfig()
    
    # Environment variables'dan oku
    if 'WORLD_SIZE' in os.environ:
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.enabled = config.world_size > 1
    
    if 'RANK' in os.environ:
        config.rank = int(os.environ['RANK'])
    
    if 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])
    
    if 'MASTER_ADDR' in os.environ:
        config.master_addr = os.environ['MASTER_ADDR']
    
    if 'MASTER_PORT' in os.environ:
        config.master_port = os.environ['MASTER_PORT']
    
    # CUDA device sayısını kontrol et
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and not config.enabled:
            config.enabled = True
            config.world_size = gpu_count
            config.gpus_per_node = gpu_count
    
    # Backend seçimi
    if torch.cuda.is_available():
        config.backend = "nccl"
    else:
        config.backend = "gloo"
    
    return config


def find_free_port() -> int:
    """Boş port bulur."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed_environment(
    world_size: int,
    rank: int,
    master_addr: str = "localhost",
    master_port: Optional[int] = None
) -> None:
    """
    Distributed environment'ı kurar.
    
    Args:
        world_size: Toplam process sayısı
        rank: Process rank'i
        master_addr: Master adres
        master_port: Master port
    """
    if master_port is None:
        master_port = find_free_port()
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    if torch.cuda.is_available():
        os.environ['LOCAL_RANK'] = str(rank % torch.cuda.device_count())


def launch_distributed_training(
    training_function,
    world_size: int,
    *args,
    **kwargs
) -> None:
    """
    Distributed training'i başlatır.
    
    Args:
        training_function: Eğitim fonksiyonu
        world_size: Process sayısı
        *args: Fonksiyon argümanları
        **kwargs: Fonksiyon keyword argümanları
    """
    mp.spawn(
        training_function,
        args=(world_size, *args),
        nprocs=world_size,
        join=True,
        **kwargs
    )
