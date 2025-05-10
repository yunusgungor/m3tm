"""
Görev Başlıkları için Temel Sınıf

Bu modül, tüm görev başlıkları için temel sınıfı içerir.
Tüm görev başlıkları bu temel sınıftan türetilmelidir.

Örüntüler:
- PluggableComponentStrategy (PT-015): Değiştirilebilir görev başlıkları
- ModelComposite (PT-003): Alt modülleri birleştiren kompozit model yapısı
"""

from typing import Dict, Any, Optional, Tuple, Union, List

import os
import json
import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """
    Görev başlıkları için soyut temel sınıf.
    
    Bu temel sınıf, çeşitli görev başlıklarının uygulanması için gerekli
    ortak arayüzü ve işlevselliği tanımlar. Tüm görev başlıkları bu sınıftan türetilmelidir.
    
    Örüntüler:
    - PluggableComponentStrategy (PT-015): Değiştirilebilir görev başlıkları
    - ModelComposite (PT-003): Alt modülleri birleştiren kompozit model yapısı
    """
    
    def __init__(self, input_dim: int):
        """
        TaskHead temel sınıfını başlatır.
        
        Args:
            input_dim: Girdi boyutu
        """
        super().__init__()
        self.input_dim = input_dim
    
    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        İleri geçiş (forward pass) işlemi. Bu metod alt sınıflar tarafından uygulanmalıdır.
        
        Args:
            inputs: Girdi tensörü (model çıktısı).
                2D [batch_size, hidden_size] ya da 3D [batch_size, seq_len, hidden_size]
            attention_mask: Dikkat maskesi [batch_size, seq_len]
            return_dict: Çıktı sözlük formatında döndürülsün mü?
            
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: Görev başlığı çıktısı
        """
        raise NotImplementedError("Alt sınıflar bu metodu uygulamalıdır.")
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Kayıp fonksiyonunu hesaplar. Bu metod alt sınıflar tarafından uygulanmalıdır.
        
        Args:
            logits: Model çıktı lojistikleri
            labels: Gerçek etiketler
            **kwargs: Ek parametreler
            
        Returns:
            torch.Tensor: Hesaplanan kayıp değeri
        """
        raise NotImplementedError("Alt sınıflar bu metodu uygulamalıdır.")
    
    def predict(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Tahmin işlemi için kolaylık metodu.
        
        Args:
            inputs: Girdi tensörü (model çıktısı)
            attention_mask: Dikkat maskesi
            
        Returns:
            torch.Tensor: Tahmin çıktısı
        """
        outputs = self.forward(inputs, attention_mask, return_dict=True)
        if isinstance(outputs, dict):
            return outputs.get("predictions", outputs.get("logits", None))
        return outputs
    
    def count_parameters(self) -> int:
        """
        Toplam eğitilebilir parametre sayısını hesaplar.
        
        Returns:
            int: Toplam parametre sayısı
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, save_directory: str, save_config: bool = True) -> None:
        """
        Görev başlığını ve yapılandırmasını diske kaydeder.
        
        Args:
            save_directory: Kayıt dizini
            save_config: Yapılandırma dosyasını da kaydet
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Model durumunu kaydet
        model_path = os.path.join(save_directory, "task_head.pt")
        torch.save(self.state_dict(), model_path)
        
        # Yapılandırmayı kaydet (alt sınıflar tarafından uygulanabilir)
        if save_config and hasattr(self, 'config'):
            config_path = os.path.join(save_directory, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """
        Görev başlığını diskten yükler.
        
        Args:
            load_directory: Yükleme dizini
            **kwargs: Ek parametreler
            
        Returns:
            TaskHead: Yüklenen görev başlığı
        """
        # Bu metod alt sınıflar tarafından özelleştirilmelidir
        raise NotImplementedError("Alt sınıflar bu metodu uygulamalıdır.") 