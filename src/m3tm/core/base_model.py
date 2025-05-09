"""
M³TM modeli için temel sınıf.
"""

import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from m3tm.config.model_config import M3TMConfig


class BaseModel(nn.Module):
    """Tüm M³TM modellerinin temel sınıfı."""
    
    def __init__(self, config: M3TMConfig):
        """
        BaseModel sınıfını başlatır.
        
        Args:
            config: Model yapılandırması
        """
        super().__init__()
        self.config = config
    
    def save_pretrained(self, save_dir: str) -> None:
        """
        Modeli ve yapılandırmasını verilen dizine kaydeder.
        
        Args:
            save_dir: Kaydetme dizini
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Model durumunu kaydet
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.state_dict(), model_path)
        
        # Yapılandırmayı kaydet
        config_path = os.path.join(save_dir, "config.pt")
        torch.save(self.config, config_path)
    
    @classmethod
    def from_pretrained(cls, load_dir: str) -> "BaseModel":
        """
        Önceden eğitilmiş modeli ve yapılandırmasını verilen dizinden yükler.
        
        Args:
            load_dir: Yükleme dizini
            
        Returns:
            Yüklenmiş BaseModel örneği
        """
        # Yapılandırmayı yükle
        config_path = os.path.join(load_dir, "config.pt")
        config = torch.load(config_path)
        
        # Modeli oluştur
        model = cls(config)
        
        # Model durumunu yükle
        model_path = os.path.join(load_dir, "model.pt")
        model.load_state_dict(torch.load(model_path))
        
        return model
    
    def get_param_count(self) -> Dict[str, int]:
        """
        Modeldeki parametrelerin sayısını hesaplar.
        
        Returns:
            Parametre istatistikleri içeren sözlük:
            - total: Toplam parametre sayısı
            - trainable: Eğitilebilir parametre sayısı
            - frozen: Donmuş parametre sayısı
        """
        total_params = 0
        trainable_params = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params
        }
    
    def forward(self, *args, **kwargs):
        """
        İleri beslemeli geçiş.
        Alt sınıflarda uygulanmalıdır.
        """
        raise NotImplementedError("BaseModel.forward alt sınıflarda uygulanmalıdır.") 