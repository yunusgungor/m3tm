"""
M³TM için ProtoTransformerBlock.

Bu modül, M³TM modelinin temel yapı taşı olan ProtoTransformerBlock'u içerir.
Bu blok, mobil cihazlar için optimize edilmiş bir Transformer bloğudur ve
alternatif dikkat mekanizmaları, mobil-dostu feed-forward ağlar 
ve adapter yuvaları içerir.

Örüntüler:
- PluggableComponentStrategy: Değiştirilebilir dikkat ve FFN mekanizmaları
- ConfigurationComposite (PT-012): Hiyerarşik yapılandırma organizasyonu
"""

from typing import Dict, Tuple, Optional, List, Union, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m3tm.research.attention_mechanisms import get_attention_mechanism_by_name, StandardSelfAttentionConfig
from m3tm.transformer.config import ProtoTransformerConfig, AttentionConfig, FeedForwardConfig, AdapterConfig
from m3tm.transformer.ffn import get_ffn_mechanism, list_available_ffn_mechanisms
from m3tm.transformer.adapter import create_adapter_slots, AdapterSlot


class ProtoTransformerBlock(nn.Module):
    """
    M³TM modeli için optimize edilmiş Transformer bloğu.
    
    Bu sınıf, çeşitli dikkat ve feed-forward ağ mekanizmalarını destekler,
    mobil cihazlarda verimli çalışacak şekilde tasarlanmıştır, ve
    adapter yuvaları ile kişiselleştirilebilir.
    """
    
    def __init__(self, config: ProtoTransformerConfig):
        """
        Args:
            config: Transformer bloğu yapılandırması
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.pre_ln = config.pre_layer_norm
        
        # Dikkat mekanizmasını al (PluggableComponentStrategy)
        attention_params = {
            "input_dim": config.hidden_size,
            **config.attention_config.mechanism_params
        }
        attention_class, attention_config = get_attention_mechanism_by_name(
            config.attention_config.mechanism_name,
            **attention_params
        )
        self.attention = attention_class(attention_config)
        
        # Feed-forward mekanizmasını al (PluggableComponentStrategy)
        ffn_params = config.ffn_config.mechanism_params.copy()
        if "hidden_size" not in ffn_params:
            ffn_params["hidden_size"] = config.hidden_size
        config.ffn_config.mechanism_params = ffn_params
        self.ffn = get_ffn_mechanism(config.ffn_config)
        
        # Normalizasyon katmanları
        self.attn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Adapter yuvaları
        if config.adapter_config.enabled:
            self.adapter_slots = create_adapter_slots(
                config.adapter_config,
                config.hidden_size,
                ["pre_attention", "post_attention", "pre_ffn", "post_ffn"]
            )
        else:
            self.adapter_slots = {}
        
        # Optimize edilmiş operasyonlar
        self.use_flash_attention = config.use_flash_attention
        self.fuse_operations = config.fuse_operations
    
    def _apply_adapters(self, x: torch.Tensor, position: str) -> torch.Tensor:
        """Belirli bir pozisyondaki adaptörleri uygular.
        
        Args:
            x: Girdi tensörü
            position: Adaptör pozisyonu
            
        Returns:
            Adaptör uygulanmış tensör
        """
        if position in self.adapter_slots:
            return self.adapter_slots[position](x)
        return x
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            attention_mask: Dikkat maskesi, şekil (batch_size, seq_len) veya (batch_size, 1, seq_len, seq_len)
            
        Returns:
            Transformer bloğu çıktısı, şekil (batch_size, seq_len, hidden_size)
            ve performans metrikleri
        """
        residual = x
        metrics = {}
        
        # Pre-attention adapter (varsa)
        x = self._apply_adapters(x, "pre_attention")
        
        # Self-attention (Pre-LN veya Post-LN)
        if self.pre_ln:
            attn_input = self.attn_ln(x)
        else:
            attn_input = x
            
        attn_output, attn_metrics = self.attention(attn_input, attention_mask)
        metrics.update({f"attention_{k}": v for k, v in attn_metrics.items()})
        
        # Attention dropout ve artık bağlantı
        attn_output = self.dropout(attn_output)
        x = residual + attn_output
        
        # Post-attention adapter (varsa)
        x = self._apply_adapters(x, "post_attention")
        
        # Feed-forward network için hazırlık
        residual = x
        
        # Pre-ffn adapter (varsa)
        x = self._apply_adapters(x, "pre_ffn")
        
        # Feed-forward network (Pre-LN veya Post-LN)
        if self.pre_ln:
            ffn_input = self.ffn_ln(x)
        else:
            ffn_input = x
            
        ffn_output, ffn_metrics = self.ffn(ffn_input)
        metrics.update({f"ffn_{k}": v for k, v in ffn_metrics.items()})
        
        # FFN dropout ve artık bağlantı
        ffn_output = self.dropout(ffn_output)
        x = residual + ffn_output
        
        # Post-ffn adapter (varsa)
        x = self._apply_adapters(x, "post_ffn")
        
        # Son normalizasyon (Post-LN durumunda)
        if not self.pre_ln:
            x = self.attn_ln(x)
            x = self.ffn_ln(x)
            
        # Toplam parametre sayısı
        metrics["total_parameters"] = self.count_parameters()
        
        return x, metrics
    
    def register_adapter(self, adapter: nn.Module, position: str) -> bool:
        """Belirli bir pozisyona adaptör kaydeder.
        
        Args:
            adapter: Kaydedilecek adaptör
            position: Adaptör pozisyonu
            
        Returns:
            Kayıt başarılı ise True, değilse False
        """
        if position in self.adapter_slots:
            self.adapter_slots[position].register_adapter(adapter)
            return True
        return False
    
    def remove_adapter(self, position: str) -> bool:
        """Belirli bir pozisyondaki adaptörü kaldırır.
        
        Args:
            position: Adaptör pozisyonu
            
        Returns:
            Kaldırma başarılı ise True, değilse False
        """
        if position in self.adapter_slots:
            self.adapter_slots[position].remove_adapter()
            return True
        return False
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_flops(self, seq_len: int) -> Dict[str, int]:
        """Yaklaşık FLOP sayısını hesaplar.
        
        Args:
            seq_len: Dizi uzunluğu
            
        Returns:
            Modül başına FLOP sayısını içeren sözlük
        """
        batch_size = 1
        hidden_size = self.hidden_size
        
        # Örnek girdi boyutları
        x_shape = (batch_size, seq_len, hidden_size)
        
        # Dikkat FLOPs (kaba tahmin - gerçek mekanizmaya göre değişebilir)
        # Tipik Q, K, V projeksiyonları + skor + ağırlıklı toplam
        attention_flops = batch_size * seq_len * (
            3 * 2 * hidden_size * hidden_size +  # QKV projeksiyonları
            2 * seq_len * hidden_size +  # Skor hesaplama
            2 * seq_len * hidden_size  # Ağırlıklı toplam
        )
        
        # FFN FLOPs özel mekanizma tarafından zaten hesaplanıyor
        # Yaklaşık bir değer atayalım
        ffn_flops = batch_size * seq_len * (
            2 * hidden_size * (hidden_size * 4) +  # İlk doğrusal
            hidden_size * 4 +  # Aktivasyon
            2 * (hidden_size * 4) * hidden_size  # İkinci doğrusal
        )
        
        # LayerNorm FLOPs
        ln_flops = batch_size * seq_len * hidden_size * 5  # Mean, var, norm ops
        
        # Adapter FLOPs (etkinse)
        adapter_flops = 0
        if self.config.adapter_config.enabled:
            bottleneck = self.config.adapter_config.bottleneck_dim
            active_slots = len(self.adapter_slots)
            adapter_flops = active_slots * batch_size * seq_len * (
                2 * hidden_size * bottleneck +  # Down-project
                bottleneck * 5 +  # LayerNorm
                bottleneck +  # Activation
                2 * bottleneck * hidden_size  # Up-project
            )
        
        return {
            "attention": attention_flops,
            "ffn": ffn_flops,
            "layer_norm": ln_flops,
            "adapters": adapter_flops,
            "total": attention_flops + ffn_flops + ln_flops + adapter_flops
        }


class ProtoTransformer(nn.Module):
    """
    Birden fazla ProtoTransformerBlock içeren tam bir Transformer modeli.
    
    Bu sınıf, M³TM'nin çekirdek Transformer modelini oluşturur.
    """
    
    def __init__(self, config: ProtoTransformerConfig, num_layers: int = 1):
        """
        Args:
            config: Transformer yapılandırması
            num_layers: Katman sayısı
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            ProtoTransformerBlock(config) for _ in range(num_layers)
        ])
        
        # Son çıktı katmanı normalizasyonu
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            attention_mask: Dikkat maskesi, şekil (batch_size, seq_len) veya (batch_size, 1, seq_len, seq_len)
            
        Returns:
            Transformer çıktısı, şekil (batch_size, seq_len, hidden_size)
            ve performans metrikleri
        """
        all_metrics = {}
        
        for i, layer in enumerate(self.layers):
            x, metrics = layer(x, attention_mask)
            all_metrics[f"layer_{i}"] = metrics
        
        # Son normalizasyon
        x = self.final_layer_norm(x)
        
        # Toplam parametre sayısı
        all_metrics["total_parameters"] = self.count_parameters()
        
        return x, all_metrics
    
    def count_parameters(self) -> int:
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 