"""
Mobil Dikkat Mekanizmaları Araştırması

Bu modül, mobil cihazlarda verimli çalışabilecek alternatif 
dikkat mekanizmalarının araştırılması ve prototiplenmesi için kullanılır.

Örüntü: ConfigurationDataclass (PT-001)
"""

import math
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionMechanismConfig:
    """Dikkat mekanizması yapılandırması için temel sınıf.
    
    Tüm dikkat mekanizmaları için ortak parametreleri içerir.
    Özel dikkat mekanizmaları bu sınıftan türetilmelidir.
    """
    name: str = ""  # Mekanizma adı
    input_dim: int = 32  # Girdi boyutu
    head_dim: int = 32  # Dikkat başı boyutu
    num_heads: int = 1  # Dikkat başı sayısı
    dropout: float = 0.1  # Dropout oranı
    use_bias: bool = True  # Bias kullanılıp kullanılmayacağı
    
    def __post_init__(self):
        """Yapılandırma parametrelerinin tutarlılığını kontrol eder."""
        if self.input_dim % self.num_heads != 0 and self.head_dim * self.num_heads != self.input_dim:
            raise ValueError(
                f"input_dim {self.input_dim} must be divisible by num_heads {self.num_heads}"
                f" or head_dim {self.head_dim} * num_heads {self.num_heads} must equal input_dim"
            )
            
        if not self.name:
            raise ValueError("Attention mechanism name must be specified")


@dataclass
class StandardSelfAttentionConfig(AttentionMechanismConfig):
    """Standart öz-dikkat mekanizması yapılandırması."""
    name: str = "StandardSelfAttention"
    causal: bool = False  # Nedensellik maskelemesi
    
    
@dataclass
class LinformerAttentionConfig(AttentionMechanismConfig):
    """Linformer dikkat mekanizması yapılandırması."""
    name: str = "LinformerAttention"
    seq_len: int = 512  # Maksimum dizi uzunluğu 
    k: int = 128  # Projeksiyon boyutu (k << seq_len)
    share_kv_projection: bool = False  # K ve V projeksiyonlarını paylaşma


@dataclass
class PerformerAttentionConfig(AttentionMechanismConfig):
    """Performer dikkat mekanizması yapılandırması."""
    name: str = "PerformerAttention"
    feature_dim: int = 32  # Kernel özellik boyutu
    ortho_scaling: float = 0.0  # Ortogonal rastgele özellikler için ölçekleme
    use_relu_kernel: bool = True  # ReLU FAVOR+ kernel kullanımı
    

@dataclass
class LocalAttentionConfig(AttentionMechanismConfig):
    """Yerel dikkat mekanizması yapılandırması."""
    name: str = "LocalAttention"
    window_size: int = 32  # Yerel dikkat pencere boyutu
    strided: bool = False  # Adımlı pencereler kullanımı
    

@dataclass
class MobileAttentionConfig(AttentionMechanismConfig):
    """Özel mobil-dostu dikkat mekanizması yapılandırması."""
    name: str = "MobileAttention"
    depth_wise: bool = True  # Derinlik yönlü konvolüsyon kullanımı
    squeeze_factor: int = 4  # Sıkıştırma faktörü
    use_gating: bool = True  # Geçit mekanizması kullanımı


class AttentionMechanismRegistry:
    """Dikkat mekanizmalarını kaydeden ve erişim sağlayan kayıt sınıfı."""
    
    _registry: Dict[str, Dict] = {
        "models": {},
        "configs": {}
    }
    
    @classmethod
    def register(cls, attention_class=None, config_class=None):
        """Yeni bir dikkat mekanizmasını kaydeder."""
        # İç içe fonksiyon kullanarak hem dekoratör hem de doğrudan çağrı olarak kullanım sağlar
        def decorator(attention_cls):
            # config_class argümanı iç veya dış fonksiyonda verilebilir
            nonlocal config_class
            if config_class is None:
                # Sınıf adından config sınıfını tahmin etmeye çalış
                config_name = f"{attention_cls.__name__}Config"
                # Mevcut modülde bu isimde bir sınıf ara
                import sys
                current_module = sys.modules[attention_cls.__module__]
                if hasattr(current_module, config_name):
                    config_class = getattr(current_module, config_name)
                else:
                    raise ValueError(f"config_class not provided and couldn't find {config_name} in module")
            
            name = config_class().name
            cls._registry["models"][name] = attention_cls
            cls._registry["configs"][name] = config_class
            return attention_cls
        
        # Doğrudan sınıf geçilmişse dekoratörü hemen uygula
        if attention_class is not None:
            return decorator(attention_class)
        
        # Aksi takdirde dekoratörü döndür
        return decorator
    
    @classmethod
    def get_attention(cls, name: str, **kwargs):
        """İsme göre dikkat mekanizması sınıfını ve yapılandırmasını döndürür."""
        if name not in cls._registry["models"]:
            raise ValueError(f"Attention mechanism '{name}' not registered")
        
        config_class = cls._registry["configs"][name]
        config = config_class(**kwargs)
        attention_class = cls._registry["models"][name]
        
        return attention_class, config
    
    @classmethod
    def list_mechanisms(cls):
        """Kayıtlı tüm dikkat mekanizmalarının listesini döndürür."""
        return list(cls._registry["models"].keys())


@AttentionMechanismRegistry.register
class StandardSelfAttention(nn.Module):
    """Standart öz-dikkat mekanizması implementasyonu.
    
    Bu temel implementasyon, karşılaştırma için referans olarak kullanılacaktır.
    Standart çok başlı öz-dikkat mekanizmasını uygular.
    """
    
    def __init__(self, config: StandardSelfAttentionConfig):
        """
        Args:
            config: Dikkat mekanizması yapılandırması
        """
        super().__init__()
        self.config = config
        
        if config.head_dim * config.num_heads != config.input_dim:
            self.head_dim = config.input_dim // config.num_heads
        else:
            self.head_dim = config.head_dim
            
        self.num_heads = config.num_heads
        self.input_dim = config.input_dim
        self.causal = config.causal
        
        # Çok başlı dikkat için projeksiyonlar
        self.q_proj = nn.Linear(config.input_dim, config.input_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.input_dim, config.input_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.input_dim, config.input_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(config.input_dim, config.input_dim, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Modül parametrelerini sıfırlar."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.config.use_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, input_dim)
            mask: Dikkat maskesi, çeşitli şekillerde olabilir:
                  - (batch_size, seq_len): Token maskesi
                  - (batch_size, seq_len, seq_len): Tam dikkat maskesi
                  - (batch_size, 1, seq_len, seq_len): Çok başlı dikkat maskesi
            
        Returns:
            Dikkat çıktısı, şekil (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Projeksiyonları uygula ve başlara ayır
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Boyutları yeniden düzenle: (batch_size, num_heads, seq_len, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Dikkat skorlarını hesapla
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Maskeleme uygula - farklı maske formatlarını destekle
        if mask is not None:
            # Maske boyutunu kontrol et ve gerektiğinde dönüştür
            if mask.dim() == 2:  # (batch_size, seq_len) şeklinde token maskesi
                # Token maskesini attention maskesine dönüştür
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
                # (batch_size, 1, 1, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
                # Her bir sorgu pozisyonu için anahtar pozisyonlarını maskeler 
                mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            elif mask.dim() == 3:  # (batch_size, seq_len, seq_len)
                # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
                # Tüm başlara genişlet
                mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
                
            # Maskeyi uygula (1 -> görünür, 0 -> maskelenmiş)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        elif self.causal:
            # Nedensel maskeleme oluştur ve uygula
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), -1e9
            )
        
        # Softmax ve dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Değerlerle çarp
        attn_output = torch.matmul(attn_weights, v)
        
        # Başları birleştir ve çıktı projeksiyonu uygula
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.input_dim)
        attn_output = self.out_proj(attn_output)
        
        # Mekanizma kompleksliği için ekstra metrikleri döndürür
        # Bu değerler benchmark sırasında kullanılabilir
        metrics = {
            "memory_tokens_complexity": seq_len ** 2,  # O(N²) bellek karmaşıklığı
            "compute_complexity": 2 * seq_len ** 2 * self.input_dim,  # İşlem karmaşıklığı
            "parameter_count": self.count_parameters()
        }
        
        return attn_output, metrics
    
    def count_parameters(self):
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@AttentionMechanismRegistry.register
class LinformerAttention(nn.Module):
    """Linformer dikkat mekanizması implementasyonu.
    
    Referans: "Linformer: Self-Attention with Linear Complexity"
    https://arxiv.org/abs/2006.04768
    
    Bu mekanizma, sekans uzunluğuna göre doğrusal ölçeklenen bir dikkat yapısı sağlar.
    Dikkat matrisinin düşük dereceli bir yaklaşımını kullanır.
    """
    
    def __init__(self, config: LinformerAttentionConfig):
        """
        Args:
            config: Linformer dikkat mekanizması yapılandırması
        """
        super().__init__()
        self.config = config
        
        if config.head_dim * config.num_heads != config.input_dim:
            self.head_dim = config.input_dim // config.num_heads
        else:
            self.head_dim = config.head_dim
            
        self.num_heads = config.num_heads
        self.input_dim = config.input_dim
        self.k_dim = config.k
        
        # Çok başlı dikkat için projeksiyonlar
        self.q_proj = nn.Linear(config.input_dim, config.input_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.input_dim, config.input_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.input_dim, config.input_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(config.input_dim, config.input_dim, bias=config.use_bias)
        
        # Linformer projeksiyon matrisleri
        self.e_proj = nn.Parameter(torch.Tensor(self.num_heads, config.seq_len, self.k_dim))
        if config.share_kv_projection:
            self.f_proj = self.e_proj
        else:
            self.f_proj = nn.Parameter(torch.Tensor(self.num_heads, config.seq_len, self.k_dim))
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Modül parametrelerini sıfırlar."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.xavier_uniform_(self.e_proj)
        if not self.config.share_kv_projection:
            nn.init.xavier_uniform_(self.f_proj)
        
        if self.config.use_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Girdi tensörü, şekil (batch_size, seq_len, input_dim)
            mask: Dikkat maskesi, şekil (batch_size, seq_len, seq_len) veya (batch_size, 1, seq_len, seq_len)
            
        Returns:
            Dikkat çıktısı, şekil (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Girdi dizisi uzunluğunu maksimum değerle sınırla
        seq_len = min(seq_len, self.config.seq_len)
        
        # Projeksiyonları uygula ve başlara ayır
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Boyutları yeniden düzenle: (batch_size, num_heads, seq_len, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Linformer düşük dereceli projeksiyonlarını uygula
        k_proj = torch.matmul(k, self.e_proj[:, :seq_len, :].unsqueeze(0))  # (batch, heads, k_dim, head_dim)
        v_proj = torch.matmul(v, self.f_proj[:, :seq_len, :].unsqueeze(0))  # (batch, heads, k_dim, head_dim)
        
        # Dikkat skorlarını hesapla (artık k_dim boyutunda)
        attn_weights = torch.matmul(q, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Maskeleme - Linformer için kompleks, bu versiyonda uygulanmadı
        # Gerçek uygulamada maske, projeksiyon öncesi uygulanmalıdır
        
        # Softmax ve dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Değerlerle çarp
        attn_output = torch.matmul(attn_weights, v_proj)
        
        # Başları birleştir ve çıktı projeksiyonu uygula
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.input_dim)
        attn_output = self.out_proj(attn_output)
        
        # Performans metrikleri
        metrics = {
            "memory_tokens_complexity": seq_len * self.k_dim,  # O(N*k) bellek karmaşıklığı
            "compute_complexity": 2 * seq_len * self.k_dim * self.input_dim,  # İşlem karmaşıklığı
            "parameter_count": self.count_parameters()
        }
        
        return attn_output, metrics
    
    def count_parameters(self):
        """Modülün eğitilebilir parametre sayısını hesaplar."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_attention_mechanism_by_name(name: str, **kwargs):
    """İsme göre dikkat mekanizması sınıfını ve uygun yapılandırmasını döndürür.
    
    Args:
        name: Dikkat mekanizması adı
        **kwargs: Dikkat mekanizması yapılandırması için ek parametreler
        
    Returns:
        Dikkat mekanizması sınıfı ve yapılandırması
    """
    return AttentionMechanismRegistry.get_attention(name, **kwargs)


def get_available_mechanisms():
    """Kullanılabilir tüm dikkat mekanizmalarının listesini döndürür."""
    return AttentionMechanismRegistry.list_mechanisms() 