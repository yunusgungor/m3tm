"""
Çoklu Etiket Başlıkları Modülü

Bu modül, M³TM modeli için çoklu etiket görevlerine özel başlıkları içerir.
Metin çoklu etiketleme, görüntü çoklu etiketleme ve çoklu-modalite 
çoklu etiketleme görevleri için kullanılabilir.

Örüntüler:
- ModelComposite (PT-003): Alt modülleri birleştiren kompozit model yapısı
- MetricsCollector (PT-008): Performans metriklerini toplama ve raporlama
"""

from typing import Dict, Any, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskHead
from .config import MultiLabelHeadConfig


class MultiLabelHead(TaskHead):
    """
    Çoklu etiket görevi için görev başlığı.
    
    Bu modül, metin veya görüntü özniteliklerini alıp çoklu etiket tahmini yapar.
    Girdi olarak son katman özniteliklerini (embeddings) alır ve her bir
    etiket için olasılık değerleri döndürür.
    
    Örüntü: ModelComposite (PT-003)
    """
    
    def __init__(self, config: MultiLabelHeadConfig):
        """
        MultiLabelHead modülünü başlatır.
        
        Args:
            config: Çoklu etiket başlığı yapılandırması
        """
        super().__init__(config.input_dim)
        self.config = config
        
        # Havuzlama metodunu belirle
        self.pooling = config.pooling
        
        # Eşik değeri
        self.threshold = config.threshold
        
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
            self.layers.append(nn.Linear(config.hidden_dim, config.num_labels, bias=config.use_bias))
        else:
            # Doğrudan çıktı katmanı
            self.layers.append(nn.Linear(config.input_dim, config.num_labels, bias=config.use_bias))
        
        # Kayıp fonksiyonu için pozitif ağırlık
        self.pos_weight = None
        if config.pos_weight is not None:
            self.pos_weight = torch.tensor(config.pos_weight)
    
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
        MultiLabelHead modülünün ileri geçişi.
        
        Args:
            inputs: Girdi tensörü.
                2D [batch_size, hidden_size] ya da 3D [batch_size, seq_len, hidden_size]
            attention_mask: Dikkat maskesi [batch_size, seq_len]
            return_dict: Çıktı sözlük formatında döndürülsün mü?
            
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: Çoklu etiket lojistikleri [batch_size, num_labels]
            ve olasılıkları veya çıktı sözlüğü
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
        
        # Çoklu etiket lojistikleri
        logits = x
        
        # Olasılıklar (sigmoid)
        probs = torch.sigmoid(logits)
        
        # Tahminler (eşik değerine göre)
        predictions = (probs >= self.threshold).float()
        
        # Metrikleri hesapla
        metrics = {
            "params": self.count_parameters()
        }
        
        if return_dict:
            return {
                "logits": logits,
                "probs": probs,
                "predictions": predictions,
                "metrics": metrics
            }
        else:
            return logits
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Kayıp fonksiyonunu hesaplar.
        
        Args:
            logits: Model çıktı lojistikleri [batch_size, num_labels]
            labels: Gerçek etiketler [batch_size, num_labels]
            **kwargs: Ek parametreler
            
        Returns:
            torch.Tensor: Hesaplanan kayıp değeri
        """
        if isinstance(logits, dict):
            logits = logits["logits"]
        
        # Pozitif ağırlık, cihazı uyumlu hale getir
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
        
        # Kayıp fonksiyonu: Binary Cross Entropy with Logits
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="mean"
        )
        
        return loss
    
    def predict(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Tahmin işlemi için kolaylık metodu.
        
        Args:
            inputs: Girdi tensörü (model çıktısı)
            attention_mask: Dikkat maskesi
            threshold: Tahmin eşiği (None ise config'deki değer kullanılır)
            
        Returns:
            torch.Tensor: İkili etiket tahminleri
        """
        outputs = self.forward(inputs, attention_mask, return_dict=True)
        if threshold is None:
            return outputs["predictions"]
        else:
            return (outputs["probs"] >= threshold).float()
    
    def get_top_k_predictions(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        En yüksek olasılıklı k etiketi döndürür.
        
        Args:
            inputs: Girdi tensörü (model çıktısı)
            attention_mask: Dikkat maskesi
            k: Döndürülecek en yüksek olasılıklı etiket sayısı
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: En yüksek k etiket indeksi ve olasılığı
        """
        outputs = self.forward(inputs, attention_mask, return_dict=True)
        probs = outputs["probs"]
        
        # k değeri etiket sayısından büyükse, etiket sayısına eşitle
        k = min(k, probs.shape[1])
        
        # En yüksek k olasılığı bul
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=1)
        
        return top_k_indices, top_k_probs
    
    @classmethod
    def from_config(cls, config: MultiLabelHeadConfig) -> 'MultiLabelHead':
        """
        Yapılandırmadan MultiLabelHead oluşturur.
        
        Args:
            config: Çoklu etiket başlığı yapılandırması
            
        Returns:
            MultiLabelHead: Oluşturulan çoklu etiket başlığı
        """
        return cls(config)
    
    @classmethod
    def create_multi_label_classifier(
        cls,
        input_dim: int = 64,
        hidden_dim: int = 32,
        num_labels: int = 5
    ) -> 'MultiLabelHead':
        """
        Çoklu etiket sınıflandırıcısı oluşturur.
        
        Args:
            input_dim: Girdi boyutu
            hidden_dim: Gizli katman boyutu
            num_labels: Etiket sayısı
            
        Returns:
            MultiLabelHead: Çoklu etiket sınıflandırıcısı
        """
        config = MultiLabelHeadConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels
        )
        return cls(config)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs) -> 'MultiLabelHead':
        """
        Kaydedilmiş bir çoklu etiket başlığını diskten yükler.
        
        Args:
            load_directory: Yükleme dizini
            **kwargs: Ek parametreler
            
        Returns:
            MultiLabelHead: Yüklenen çoklu etiket başlığı
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
        config = MultiLabelHeadConfig(**config_dict)
        
        # Modeli oluştur
        model = cls(config)
        
        # Model durumunu yükle
        model_path = os.path.join(load_directory, "task_head.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
        
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model 