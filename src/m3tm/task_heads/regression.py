"""
Regresyon Başlıkları Modülü

Bu modül, M³TM modeli için regresyon görevlerine özel başlıkları içerir.
Metin regresyon, görüntü regresyon ve çoklu-modalite regresyon
görevleri için kullanılabilir.

Örüntüler:
- ModelComposite (PT-003): Alt modülleri birleştiren kompozit model yapısı
- MetricsCollector (PT-008): Performans metriklerini toplama ve raporlama
"""

from typing import Dict, Any, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskHead
from .config import RegressionHeadConfig


class RegressionHead(TaskHead):
    """
    Regresyon görevi için görev başlığı.
    
    Bu modül, metin veya görüntü özniteliklerini alıp regresyon tahmini yapar.
    Girdi olarak son katman özniteliklerini (embeddings) alır ve sürekli değerler döndürür.
    
    Örüntü: ModelComposite (PT-003)
    """
    
    def __init__(self, config: RegressionHeadConfig):
        """
        RegressionHead modülünü başlatır.
        
        Args:
            config: Regresyon başlığı yapılandırması
        """
        super().__init__(config.input_dim)
        self.config = config
        
        # Havuzlama metodunu belirle
        self.pooling = config.pooling
        
        # Katmanları tanımla
        self.layers = nn.ModuleList()
        
        # Ara katman (hidden layer) varsa ekle
        if config.hidden_dim > 0:
            self.layers.append(nn.Linear(config.input_dim, config.hidden_dim, bias=config.use_bias))
            
            # Layer normalization
            if config.use_layer_norm:
                self.layers.append(nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps))
            
            # Aktivasyon
            self.layers.append(self._get_activation(config.activation))
            
            # Dropout
            if config.dropout_rate > 0:
                self.layers.append(nn.Dropout(config.dropout_rate))
            
            # Çıktı katmanı
            self.layers.append(nn.Linear(config.hidden_dim, config.output_dim, bias=config.use_bias))
        else:
            # Doğrudan çıktı katmanı
            self.layers.append(nn.Linear(config.input_dim, config.output_dim, bias=config.use_bias))
        
        # Son katman aktivasyonu
        self.final_activation = None
        if config.final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif config.final_activation == "tanh":
            self.final_activation = nn.Tanh()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """
        Aktivasyon fonksiyonu modülünü döndürür.
        
        Args:
            activation_name: Aktivasyon fonksiyonu adı
            
        Returns:
            nn.Module: Aktivasyon modülü
            
        Raises:
            ValueError: Bilinmeyen aktivasyon fonksiyonu
        """
        if activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "swish":
            return nn.SiLU()  # SiLU = Swish
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")
    
    def _apply_pooling(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Havuzlama (pooling) işlemi uygular.
        
        Args:
            x: Girdi tensörü [batch_size, seq_len, hidden_size]
            attention_mask: Dikkat maskesi [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Havuzlanmış tensör [batch_size, hidden_size]
        """
        if self.pooling == "cls" and x.size(1) > 0:
            # İlk token'ı al (genellikle CLS token)
            return x[:, 0]
        elif self.pooling == "first" and x.size(1) > 0:
            # İlk token'ı al (CLS ile aynı, ancak terminoloji farkı)
            return x[:, 0]
        elif self.pooling == "mean":
            # Ortalama havuzlama
            if attention_mask is not None:
                # Maske uygula, böylece padding tokenları ortalamanın dışında kalır
                expanded_mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
                sum_embeddings = torch.sum(x * expanded_mask, dim=1)  # [batch_size, hidden_size]
                sum_mask = torch.sum(expanded_mask, dim=1)  # [batch_size, 1]
                # Sıfıra bölmeyi önle
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                return sum_embeddings / sum_mask
            else:
                # Maske yoksa, tüm tokenların ortalamasını al
                return torch.mean(x, dim=1)
        elif self.pooling == "max":
            # Maksimum havuzlama
            if attention_mask is not None:
                # Maske uygula, böylece padding tokenları maksimumun dışında kalır
                # Padding tokenlarını çok küçük bir değer ile değiştir
                expanded_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
                x_masked = x.clone()
                x_masked[expanded_mask == 0] = -1e9  # Padding tokenlarını maskele
                return torch.max(x_masked, dim=1)[0]  # [batch_size, hidden_size]
            else:
                # Maske yoksa, tüm tokenların maksimumunu al
                return torch.max(x, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        RegressionHead modülünün ileri geçişi.
        
        Args:
            inputs: Girdi tensörü.
                2D [batch_size, hidden_size] ya da 3D [batch_size, seq_len, hidden_size]
            attention_mask: Dikkat maskesi [batch_size, seq_len]
            return_dict: Çıktı sözlük formatında döndürülsün mü?
            
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: Regresyon değerleri [batch_size, output_dim]
            veya çıktı sözlüğü
        """
        # 3 boyutlu girdiyse havuzlama uygula (dizi girildi)
        if len(inputs.shape) == 3:
            x = self._apply_pooling(inputs, attention_mask)
        else:
            # Zaten havuzlanmış girdi
            x = inputs
        
        # Katmanları uygula
        for layer in self.layers:
            x = layer(x)
        
        # Son katman aktivasyonu (opsiyonel)
        if self.final_activation is not None:
            x = self.final_activation(x)
        
        # Regresyon değerleri
        values = x
        
        # Metrikleri hesapla
        metrics = {
            "params": self.count_parameters()
        }
        
        if return_dict:
            return {
                "values": values,
                "metrics": metrics
            }
        else:
            return values
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Kayıp fonksiyonunu hesaplar.
        
        Args:
            logits: Model çıktı değerleri [batch_size, output_dim]
            labels: Gerçek değerler [batch_size, output_dim]
            **kwargs: Ek parametreler
            
        Returns:
            torch.Tensor: Hesaplanan kayıp değeri
        """
        if isinstance(logits, dict):
            logits = logits["values"]
        
        # Boyut kontrolü
        if logits.shape != labels.shape:
            # Eğer çıktı tek boyutlu ama etiketler tek boyutlu değilse, squeeze et
            if logits.shape[1] == 1 and len(labels.shape) == 1:
                logits = logits.squeeze(-1)
            # Eğer etiketler tek boyutlu ama çıktı tek boyutlu değilse, unsqueeze et
            elif len(logits.shape) == 2 and len(labels.shape) == 1:
                labels = labels.unsqueeze(-1)
        
        # Kayıp fonksiyonu tipine göre hesapla
        if self.config.loss_type == "mse":
            return F.mse_loss(logits, labels)
        elif self.config.loss_type == "mae":
            return F.l1_loss(logits, labels)
        elif self.config.loss_type == "smooth_l1":
            return F.smooth_l1_loss(logits, labels)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
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
            torch.Tensor: Regresyon tahmin değerleri
        """
        outputs = self.forward(inputs, attention_mask, return_dict=True)
        if isinstance(outputs, dict):
            return outputs["values"]
        return outputs
    
    @classmethod
    def from_config(cls, config: RegressionHeadConfig) -> 'RegressionHead':
        """
        Yapılandırmadan RegressionHead oluşturur.
        
        Args:
            config: Regresyon başlığı yapılandırması
            
        Returns:
            RegressionHead: Oluşturulan regresyon başlığı
        """
        return cls(config)
    
    @classmethod
    def create_single_regressor(cls, input_dim: int = 64, hidden_dim: int = 32) -> 'RegressionHead':
        """
        Tek çıktılı bir regresör oluşturur.
        
        Args:
            input_dim: Girdi boyutu
            hidden_dim: Gizli katman boyutu
            
        Returns:
            RegressionHead: Tek çıktılı regresör
        """
        config = RegressionHeadConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        )
        return cls(config)
    
    @classmethod
    def create_multi_regressor(
        cls,
        input_dim: int = 64,
        hidden_dim: int = 32,
        output_dim: int = 2
    ) -> 'RegressionHead':
        """
        Çoklu çıktılı bir regresör oluşturur.
        
        Args:
            input_dim: Girdi boyutu
            hidden_dim: Gizli katman boyutu
            output_dim: Çıktı boyutu
            
        Returns:
            RegressionHead: Çoklu çıktılı regresör
        """
        config = RegressionHeadConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        return cls(config)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs) -> 'RegressionHead':
        """
        Kaydedilmiş bir regresyon başlığını diskten yükler.
        
        Args:
            load_directory: Yükleme dizini
            **kwargs: Ek parametreler
            
        Returns:
            RegressionHead: Yüklenen regresyon başlığı
        """
        import os
        import json
        
        # Yapılandırmayı yükle
        config_path = os.path.join(load_directory, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Yapılandırma sınıfını oluştur
        config = RegressionHeadConfig(**config_dict)
        
        # Modeli oluştur
        model = cls(config)
        
        # Model durumunu yükle
        model_path = os.path.join(load_directory, "task_head.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
        
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model 