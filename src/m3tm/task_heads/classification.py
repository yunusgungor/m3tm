"""
Sınıflandırma Başlıkları Modülü

Bu modül, M³TM modeli için sınıflandırma görevlerine özel başlıkları içerir.
Metin sınıflandırma, görüntü sınıflandırma ve çoklu-modalite sınıflandırma
görevleri için kullanılabilir.

Örüntüler:
- ModelComposite (PT-003): Alt modülleri birleştiren kompozit model yapısı
- MetricsCollector (PT-008): Performans metriklerini toplama ve raporlama
"""

from typing import Dict, Any, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ClassificationHeadConfig


class ClassificationHead(nn.Module):
    """
    Sınıflandırma görevi için görev başlığı.
    
    Bu modül, metin veya görüntü özniteliklerini alıp sınıflandırma yapar.
    Girdi olarak son katman özniteliklerini (embeddings) alır ve sınıf 
    olasılıklarını veya lojistikleri döndürür.
    
    Örüntü: ModelComposite (PT-003)
    """
    
    def __init__(self, config: ClassificationHeadConfig):
        """
        ClassificationHead modülünü başlatır.
        
        Args:
            config: Sınıflandırma başlığı yapılandırması
        """
        super().__init__()
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
            self.layers.append(nn.Linear(config.hidden_dim, config.num_classes, bias=config.use_bias))
        else:
            # Doğrudan çıktı katmanı
            self.layers.append(nn.Linear(config.input_dim, config.num_classes, bias=config.use_bias))
    
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
        ClassificationHead modülünün ileri geçişi.
        
        Args:
            inputs: Girdi tensörü.
                2D [batch_size, hidden_size] ya da 3D [batch_size, seq_len, hidden_size]
            attention_mask: Dikkat maskesi [batch_size, seq_len]
            return_dict: Çıktı sözlük formatında döndürülsün mü?
            
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: Sınıflandırma lojistikleri [batch_size, num_classes]
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
        
        # Sınıflandırma lojistikleri
        logits = x
        
        # İkili sınıflandırma için sigmoid olasılığı hesapla
        if self.config.num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            # Çok sınıflı sınıflandırma için softmax olasılığı hesapla
            probs = F.softmax(logits, dim=-1)
        
        # Metrikleri hesapla
        metrics = {
            "params": self.count_parameters()
        }
        
        if return_dict:
            return {
                "logits": logits,
                "probs": probs,
                "metrics": metrics
            }
        else:
            return logits
    
    def predict(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sınıflandırma tahmini yapar.
        
        Args:
            inputs: Girdi tensörü
            attention_mask: Dikkat maskesi
            
        Returns:
            torch.Tensor: Tahmin edilen sınıf indeksleri [batch_size]
        """
        outputs = self.forward(inputs, attention_mask, return_dict=True)
        logits = outputs["logits"]
        
        # İkili sınıflandırma
        if self.config.num_classes == 1:
            predictions = (logits > 0).long()
        else:
            # Çok sınıflı sınıflandırma
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions
    
    def count_parameters(self) -> int:
        """
        Eğitilebilir parametre sayısını hesaplar.
        
        Returns:
            int: Eğitilebilir parametre sayısı
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_config(cls, config: ClassificationHeadConfig) -> 'ClassificationHead':
        """
        Yapılandırma nesnesinden ClassificationHead oluşturur.
        
        Args:
            config: Sınıflandırma başlığı yapılandırması
            
        Returns:
            ClassificationHead: Oluşturulan sınıflandırma başlığı
        """
        return cls(config)
    
    @classmethod
    def create_binary_classifier(cls, input_dim: int = 64, hidden_dim: int = 32) -> 'ClassificationHead':
        """
        İkili sınıflandırıcı oluşturur.
        
        Args:
            input_dim: Girdi boyutu
            hidden_dim: Gizli katman boyutu
            
        Returns:
            ClassificationHead: İkili sınıflandırıcı
        """
        config = ClassificationHeadConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=2
        )
        return cls(config)
    
    @classmethod
    def create_multiclass_classifier(
        cls,
        input_dim: int = 64,
        hidden_dim: int = 32,
        num_classes: int = 5
    ) -> 'ClassificationHead':
        """
        Çok sınıflı sınıflandırıcı oluşturur.
        
        Args:
            input_dim: Girdi boyutu
            hidden_dim: Gizli katman boyutu
            num_classes: Sınıf sayısı
            
        Returns:
            ClassificationHead: Çok sınıflı sınıflandırıcı
        """
        config = ClassificationHeadConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        return cls(config) 