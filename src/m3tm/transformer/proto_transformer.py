"""
M³TM için ProtoTransformerBlock.

Bu modül, M³TM modelinin temel yapı taşı olan ProtoTransformerBlock'u içerir.
Bu blok, mobil cihazlar için optimize edilmiş bir Transformer bloğudur ve
alternatif dikkat mekanizmaları, mobil-dostu feed-forward ağlar 
ve adapter yuvaları içerir.

Örüntüler:
- PluggableComponentStrategy: Değiştirilebilir dikkat ve FFN mekanizmaları
- ConfigurationComposite (PT-012): Hiyerarşik yapılandırma organizasyonu
- ModelComposite (PT-003): Ana modelin içine küçük ve özelleştirilmiş modül ekleme
- CompositeAdapter (PT-014): Birden fazla adaptörü sıralı olarak uygulama
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
    
    Örüntüler:
    - ModelComposite (PT-003): Ana modelin içine küçük ve özelleştirilmiş modül ekleme
    - CompositeAdapter (PT-014): Birden fazla adaptörü sıralı olarak uygulama
    """
    
    def __init__(self, config):
        """
        Args:
            config: Transformer bloğu yapılandırması (ProtoTransformerConfig veya TransformerConfig)
        """
        super().__init__()
        self.config = config
        
        # TransformerConfig veya ProtoTransformerConfig arasında uyumluluk
        self.hidden_size = getattr(config, 'hidden_size', getattr(config, 'embed_dim', 768))
        self.pre_ln = getattr(config, 'pre_layer_norm', True)
        
        # Farklı yapılandırma sınıfları için default değerler
        attention_config = getattr(config, 'attention_config', {})
        attention_mechanism_name = getattr(attention_config, 'mechanism_name', 
                                         getattr(config, 'attention_type', 'StandardSelfAttention'))
        
        # Dikkat mekanizmasını al (PluggableComponentStrategy)
        attention_params = {
            "input_dim": self.hidden_size,
            "num_heads": getattr(config, 'num_heads', 8),
            "dropout": getattr(config, 'attention_dropout', getattr(config, 'dropout', 0.1))
        }
        
        if hasattr(attention_config, 'mechanism_params'):
            attention_params.update(attention_config.mechanism_params)
            
        attention_class, attention_config = get_attention_mechanism_by_name(
            attention_mechanism_name,
            **attention_params
        )
        self.attention = attention_class(attention_config)
        
        # Feed-forward mekanizmasını al (PluggableComponentStrategy)
        ffn_config = getattr(config, 'ffn_config', {})
        ffn_mechanism_name = getattr(ffn_config, 'mechanism_name', 
                                    getattr(config, 'ffn_type', 'StandardFFN'))
        
        ffn_params = {
            "hidden_size": self.hidden_size,
            "intermediate_size": getattr(config, 'ffn_hidden_dim', self.hidden_size * 4),
            "dropout": getattr(config, 'dropout', 0.1),
            "activation": getattr(config, 'activation', 'gelu')
        }
        
        if hasattr(ffn_config, 'mechanism_params'):
            ffn_params.update(ffn_config.mechanism_params)
            
        # FFN yapılandırmasını oluştur ve FFN mekanizmasını al
        from m3tm.transformer.config import FeedForwardConfig
        temp_ffn_config = FeedForwardConfig(
            mechanism_name=ffn_mechanism_name,
            mechanism_params=ffn_params
        )
        self.ffn = get_ffn_mechanism(temp_ffn_config)
        
        # Normalizasyon katmanları
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-12)
        self.attn_ln = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.ffn_ln = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.1))
        
        # Adapter yuvaları
        adapter_config = getattr(config, 'adapter_config', None)
        use_adapter_slots = getattr(config, 'use_adapter_slots', False)
        
        if adapter_config and getattr(adapter_config, 'enabled', False):
            # Tüm pozisyonlar için adapter yuvaları oluştur
            self.adapter_slots = create_adapter_slots(
                adapter_config,
                self.hidden_size,
                getattr(adapter_config, 'adapter_positions', 
                       ["pre_attention", "post_attention", "pre_ffn", "post_ffn"])
            )
        elif use_adapter_slots:
            # TransformerConfig için basit adapter konfigürasyonu oluştur
            from m3tm.transformer.config import AdapterConfig
            simple_adapter_config = AdapterConfig(
                enabled=True,
                bottleneck_dim=self.hidden_size // 8,
                adapter_positions=["post_attention", "post_ffn"]
            )
            self.adapter_slots = create_adapter_slots(
                simple_adapter_config,
                self.hidden_size,
                ["post_attention", "post_ffn"]
            )
        else:
            self.adapter_slots = {}
        
        # Optimize edilmiş operasyonlar
        self.use_flash_attention = getattr(config, 'use_flash_attention', False)
        self.fuse_operations = getattr(config, 'fuse_operations', False)
        
        # Eğitim ve çıkarım modları için kontroller
        self.training_adapters = True
    
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
    
    def register_adapter(self, adapter: nn.Module, position: str, name: str = None) -> bool:
        """Belirli bir pozisyona adaptör kaydeder.
        
        Args:
            adapter: Kaydedilecek adaptör
            position: Adaptör pozisyonu
            name: Adaptör adı (None ise otomatik oluşturulur)
            
        Returns:
            Kayıt başarılı ise True, değilse False
        """
        if position in self.adapter_slots:
            return self.adapter_slots[position].register_adapter(adapter, name)
        return False
    
    def remove_adapter(self, position: str, name: str = None) -> bool:
        """Belirli bir pozisyondaki adaptörü kaldırır.
        
        Args:
            position: Adaptör pozisyonu
            name: Kaldırılacak adaptör adı (None ise son eklenen adaptör kaldırılır)
            
        Returns:
            Kaldırma başarılı ise True, değilse False
        """
        if position in self.adapter_slots:
            return self.adapter_slots[position].remove_adapter(name)
        return False
    
    def get_adapter(self, position: str, name: str = None) -> Optional[nn.Module]:
        """Belirli bir pozisyondaki adaptörü alır.
        
        Args:
            position: Adaptör pozisyonu
            name: Adaptör adı (None ise son eklenen adaptör döndürülür)
            
        Returns:
            Adaptör modülü veya None
        """
        if position in self.adapter_slots:
            return self.adapter_slots[position].get_adapter(name)
        return None
    
    def get_adapter_positions(self) -> List[str]:
        """Desteklenen adapter pozisyonlarını döndürür."""
        return list(self.adapter_slots.keys())
    
    def get_adapter_names(self, position: str) -> List[str]:
        """Belirli bir pozisyondaki adaptör isimlerini döndürür.
        
        Args:
            position: Adaptör pozisyonu
            
        Returns:
            Adaptör isimleri listesi
        """
        if position in self.adapter_slots:
            return self.adapter_slots[position].get_adapter_names()
        return []
    
    def set_training_adapters(self, training: bool) -> None:
        """Adaptörlerin eğitim modunu ayarlar.
        
        Args:
            training: True ise adaptörler etkinleştirilir, False ise devre dışı bırakılır
        """
        self.training_adapters = training
        for pos, slot in self.adapter_slots.items():
            slot.set_training_mode(training)
    
    def count_parameters(self) -> int:
        """Modülün toplam eğitilebilir parametre sayısını hesaplar."""
        main_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Adaptor parametrelerini hesapla
        adapter_params = 0
        for pos, slot in self.adapter_slots.items():
            adapter_params += slot.count_parameters()
            
        return main_params
    
    def estimate_flops(self, seq_len: int) -> Dict[str, int]:
        """Bir ileri yayılım için tahmini FLOP sayısını hesaplar.
        
        Args:
            seq_len: Giriş dizisi uzunluğu
            
        Returns:
            FLOP tahminlerini içeren sözlük
        """
        batch_size = 1  # Batch başına tek örnek için hesapla
        
        flops = {}
        
        # Layer Norm FLOP'ları (ortalama, varyans, normalize)
        ln_flops = 2 * batch_size * seq_len * self.hidden_size
        
        # Attention FLOP'ları
        # - Q, K, V projeksiyonları: 3 * batch_size * seq_len * hidden_size * hidden_size
        # - QK matris çarpımı: batch_size * num_heads * seq_len * seq_len * head_dim
        # - Softmax: batch_size * num_heads * seq_len * seq_len
        # - Attention * V: batch_size * num_heads * seq_len * seq_len * head_dim
        # - Final projeksiyon: batch_size * seq_len * hidden_size * hidden_size
        qkv_proj_flops = 3 * batch_size * seq_len * self.hidden_size * self.hidden_size
        num_heads = self.config.attention_config.num_heads
        head_dim = self.hidden_size // num_heads
        qk_matmul_flops = batch_size * num_heads * seq_len * seq_len * head_dim
        softmax_flops = batch_size * num_heads * seq_len * seq_len
        attn_v_flops = batch_size * num_heads * seq_len * seq_len * head_dim
        final_proj_flops = batch_size * seq_len * self.hidden_size * self.hidden_size
        
        attention_flops = qkv_proj_flops + qk_matmul_flops + softmax_flops + attn_v_flops + final_proj_flops
        flops["attention"] = attention_flops
        
        # FFN FLOP'ları
        # - Down projeksiyon: batch_size * seq_len * hidden_size * intermediate_size
        # - Aktivasyon: batch_size * seq_len * intermediate_size
        # - Up projeksiyon: batch_size * seq_len * intermediate_size * hidden_size
        intermediate_size = self.config.ffn_config.intermediate_dim
        down_proj_flops = batch_size * seq_len * self.hidden_size * intermediate_size
        activation_flops = batch_size * seq_len * intermediate_size
        up_proj_flops = batch_size * seq_len * intermediate_size * self.hidden_size
        
        ffn_flops = down_proj_flops + activation_flops + up_proj_flops
        flops["ffn"] = ffn_flops
        
        # Adapter FLOP'ları (tüm pozisyonlar)
        adapter_flops = 0
        for pos, slot in self.adapter_slots.items():
            # Her adaptör için: down_proj + activation + up_proj
            for adapter_name in slot.get_adapter_names():
                adapter = slot.get_adapter(adapter_name)
                if hasattr(adapter, "down_project") and hasattr(adapter, "up_project"):
                    bottleneck_dim = adapter.down_project.out_features
                    down_flops = batch_size * seq_len * self.hidden_size * bottleneck_dim
                    act_flops = batch_size * seq_len * bottleneck_dim
                    up_flops = batch_size * seq_len * bottleneck_dim * self.hidden_size
                    adapter_flops += down_flops + act_flops + up_flops
        
        flops["adapters"] = adapter_flops
        
        # Toplam FLOP'lar
        flops["total"] = 2 * ln_flops + attention_flops + ffn_flops + adapter_flops
        
        return flops
    
    def add_adapter(self, adapter_name: str, adapter_size: int = None) -> None:
        """Test için isim ve boyutla adapter eklemek için kullanılır.
        
        Args:
            adapter_name: Eklenecek adaptörün adı
            adapter_size: Adaptör boyutu (bottleneck boyutu)
        """
        # Standart pozisyonlar için adaptör oluştur
        from m3tm.adapters.adapter import create_adapter, AdapterConfig, AdapterType
        
        # Eğer adapter_size belirtilmemişse, varsayılan bir değer kullan
        if adapter_size is None:
            adapter_size = self.hidden_size // 8
            
        positions = self.get_adapter_positions()
        if not positions:
            positions = ["post_attention", "post_ffn"]
            
        # Her pozisyon için adaptör oluştur ve kaydet
        for position in positions:
            # AdapterConfig oluştur
            adapter_config = AdapterConfig(
                adapter_type=AdapterType.BOTTLENECK,
                bottleneck_dim=adapter_size,
                use_layer_norm=True,
                activation="gelu",
                dropout=0.0
            )
            
            # Adapter'ı oluştur
            adapter = create_adapter(adapter_config, self.hidden_size)
                
            # Eğer pozisyon yoksa, slot oluştur
            if position not in self.adapter_slots:
                from m3tm.transformer.adapter import AdapterSlot
                self.adapter_slots[position] = AdapterSlot(self.hidden_size)
                
            # Adaptörü kaydet
            self.register_adapter(adapter, position, adapter_name)
        
        # Adaptör modunda çalıştığından emin ol
        self.set_training_adapters(True)


class ProtoTransformer(nn.Module):
    """
    Çok katmanlı ProtoTransformerBlock modeli.
    """
    
    def __init__(self, config: ProtoTransformerConfig, num_layers: int = 1):
        """
        Args:
            config: Transformer yapılandırması
            num_layers: Katman sayısı
        """
        super().__init__()
        self.config = config
        self.num_layers = num_layers if num_layers > 0 else config.num_layers
        
        # Transformer blokları
        self.layers = nn.ModuleList([
            ProtoTransformerBlock(config) for _ in range(self.num_layers)
        ])
        
        # Son layer norm
        self.final_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, hidden_size)
            attention_mask: Dikkat maskesi
            
        Returns:
            Transformer çıktısı ve performans metrikleri
        """
        metrics = {"layer_metrics": []}
        
        for i, layer in enumerate(self.layers):
            x, layer_metrics = layer(x, attention_mask)
            metrics["layer_metrics"].append({f"layer_{i}_{k}": v for k, v in layer_metrics.items()})
        
        # Son layer norm
        x = self.final_ln(x)
        
        # Toplam parametre sayısı
        metrics["total_parameters"] = self.count_parameters()
        
        return x, metrics
    
    def register_adapter(self, adapter: nn.Module, layer_idx: int, position: str, name: str = None) -> bool:
        """Belirli bir katmana ve pozisyona adaptör kaydeder.
        
        Args:
            adapter: Kaydedilecek adaptör
            layer_idx: Katman indeksi
            position: Adaptör pozisyonu
            name: Adaptör adı
            
        Returns:
            Kayıt başarılı ise True, değilse False
        """
        if 0 <= layer_idx < len(self.layers):
            return self.layers[layer_idx].register_adapter(adapter, position, name)
        return False
    
    def remove_adapter(self, layer_idx: int, position: str, name: str = None) -> bool:
        """Belirli bir katmandan ve pozisyondan adaptörü kaldırır.
        
        Args:
            layer_idx: Katman indeksi
            position: Adaptör pozisyonu
            name: Adaptör adı
            
        Returns:
            Kaldırma başarılı ise True, değilse False
        """
        if 0 <= layer_idx < len(self.layers):
            return self.layers[layer_idx].remove_adapter(position, name)
        return False
    
    def set_training_adapters(self, training: bool) -> None:
        """Tüm katmanlardaki adaptörlerin eğitim modunu ayarlar.
        
        Args:
            training: True ise adaptörler etkinleştirilir, False ise devre dışı bırakılır
        """
        for layer in self.layers:
            layer.set_training_adapters(training)
    
    def count_parameters(self) -> int:
        """Toplam eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Tüm katmanlar ve pozisyonlar için adaptör bilgilerini döndürür."""
        info = {}
        for i, layer in enumerate(self.layers):
            layer_info = {}
            positions = layer.get_adapter_positions()
            for pos in positions:
                adapter_names = layer.get_adapter_names(pos)
                if adapter_names:
                    layer_info[pos] = adapter_names
            if layer_info:
                info[f"layer_{i}"] = layer_info 