"""
Görüntü Embedding Modülü

Bu modül, görüntüleri yamalara bölerek vektör gösterimlerine dönüştüren 
PyTorch modüllerini içerir. ImagePatchEmbeddingConfig yapılandırmasını
kullanarak görüntü yamalarını vektör gösterimlerine dönüştürür.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..image.patch import patchify_image, get_2d_sincos_pos_embed

@dataclass
class ImagePatchEmbeddingConfig:
    """
    Görüntü yama gömme işlemi için yapılandırma sınıfı.
    
    Örüntü: ConfigurationDataclass (PT-001)
    """
    # Görüntü temel özellikleri
    image_size: int = 224  # Görüntü boyutu (kare görüntü varsayılır)
    in_channels: int = 3    # Giriş kanalları (RGB=3, Gri=1)
    
    # Yama özellikleri
    patch_size: int = 16    # Yama boyutu (kare yama varsayılır)
    
    # Gömme boyutları
    embed_dim: int = 192    # Yama gömmelerinin boyutu
    use_embedding_projection: bool = False  # Gömme boyutu projeksiyonu kullanılsın mı?
    projection_dim: Optional[int] = None    # Projeksiyon boyutu (kullanılacaksa)
    
    # Konum gömmesi
    use_position_embedding: bool = True     # Konum gömmesi kullanılsın mı?
    position_embedding_type: str = "sincos"  # Konum gömmesi türü: "learned" veya "sincos"
    
    # Normalizasyon ve düzenlileştirme
    layer_norm_eps: float = 1e-12           # LayerNorm epsilon değeri
    dropout_rate: float = 0.1               # Dropout oranı
    
    # Interpolasyon ayarları (farklı boyutlu görüntüler için)
    interpolate_mode: str = "bicubic"       # Interpolasyon modu
    interpolate_pos_encoding: bool = False  # Pozisyon kodlaması için interpolasyon
    
    # İlklendirme için ek ayarlar
    init_std: float = 0.02                  # İlklendirme standart sapması
    
    def __post_init__(self):
        """Yapılandırma doğrulamasını gerçekleştirir"""
        # Örüntü: ConfigValidationPipeline (PT-009)
        self._validate_image_size()
        self._validate_patch_size()
        self._validate_embed_dims()
        self._validate_position_embedding()
        
    def _validate_image_size(self):
        """Görüntü boyutu doğrulaması"""
        if self.image_size <= 0:
            raise ValueError(f"image_size pozitif olmalıdır, alınan: {self.image_size}")
    
    def _validate_patch_size(self):
        """Yama boyutu doğrulaması"""
        if self.patch_size <= 0:
            raise ValueError(f"patch_size pozitif olmalıdır, alınan: {self.patch_size}")
        
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) patch_size ({self.patch_size})'in "
                f"tam katı olmalıdır."
            )
    
    def _validate_embed_dims(self):
        """Gömme boyutları doğrulaması"""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim pozitif olmalıdır, alınan: {self.embed_dim}")
        
        if self.use_embedding_projection and self.projection_dim is None:
            raise ValueError(
                "use_embedding_projection=True iken projection_dim belirtilmelidir."
            )
            
        if self.use_embedding_projection and self.projection_dim <= 0:
            raise ValueError(f"projection_dim pozitif olmalıdır, alınan: {self.projection_dim}")
    
    def _validate_position_embedding(self):
        """Konum gömmesi doğrulaması"""
        valid_pos_embedding_types = ["learned", "sincos"]
        if (
            self.use_position_embedding and 
            self.position_embedding_type not in valid_pos_embedding_types
        ):
            raise ValueError(
                f"position_embedding_type şunlardan biri olmalıdır: {valid_pos_embedding_types}, "
                f"alınan: {self.position_embedding_type}"
            )
    
    @property
    def num_patches(self) -> int:
        """Görüntüdeki toplam yama sayısını hesaplar."""
        return (self.image_size // self.patch_size) ** 2


class ImagePatchEmbedding(nn.Module):
    """
    Görüntüleri yamalara bölerek embedding vektörleri oluşturan PyTorch modülü.
    
    Bu modül, görüntüleri eşit boyutlu yamalara böler, doğrusal projeksiyon uygular
    ve isteğe bağlı olarak konum gömmesi ekler. ViT (Vision Transformer) mimarisinden
    esinlenmiştir ancak mobil cihazlarda verimli çalışacak şekilde optimize edilmiştir.
    
    Örüntü: ModelComposite (PT-003), VisionTransformerPatching (PT-014)
    """
    
    def __init__(self, config: ImagePatchEmbeddingConfig):
        """
        ImagePatchEmbedding modülünü başlatır.
        
        Args:
            config: Görüntü yama gömme yapılandırması
        """
        super().__init__()
        
        self.config = config
        
        # Yamadan gömmeye projeksiyon
        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Konum gömmesi
        self.position_embedding = None
        if config.use_position_embedding:
            if config.position_embedding_type == "learned":
                self.position_embedding = nn.Parameter(
                    torch.zeros(1, config.num_patches, config.embed_dim)
                )
                nn.init.normal_(self.position_embedding, std=config.init_std)
            elif config.position_embedding_type == "sincos":
                # Sinüzoidal konum kodlaması, ileri geçişte dinamik olarak oluşturulacak
                pass
        
        # Embedding boyut düşürme projeksiyonu (opsiyonel)
        self.projection_layer = None
        if config.use_embedding_projection and config.projection_dim is not None:
            self.projection_layer = nn.Linear(config.embed_dim, config.projection_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(
            config.projection_dim if config.use_embedding_projection and config.projection_dim is not None
            else config.embed_dim,
            eps=config.layer_norm_eps
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Model parametrelerini başlat
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Model ağırlıklarını başlatır."""
        # Projeksiyon katmanını başlat
        nn.init.normal_(self.projection.weight, std=self.config.init_std)
        nn.init.zeros_(self.projection.bias)
        
        # Projeksiyon katmanını başlat (varsa)
        if self.projection_layer is not None:
            nn.init.normal_(self.projection_layer.weight, std=self.config.init_std)
            nn.init.zeros_(self.projection_layer.bias)
        
        # LayerNorm'u başlat
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
    
    def _resize_pos_embed(self, pos_embed: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Konum gömme matrisini görüntü boyutuna göre yeniden boyutlandırır.
        
        Args:
            pos_embed: Konum gömme matrisi [1, orig_h*orig_w, embed_dim]
            height: Hedef yama yüksekliği
            width: Hedef yama genişliği
            
        Returns:
            torch.Tensor: Yeniden boyutlandırılmış konum gömme matrisi [1, height*width, embed_dim]
        """
        if not self.config.interpolate_pos_encoding:
            return pos_embed
            
        # Orijinal boyutları hesapla
        embed_dim = pos_embed.shape[-1]
        orig_size = int(math.sqrt(pos_embed.shape[1]))
        
        # Konum gömmeyi 2D formata yeniden şekillendir [1, orig_size, orig_size, embed_dim]
        pos_embed_2d = pos_embed.reshape(1, orig_size, orig_size, embed_dim)
        
        # 2D gömmeyi interpolasyonla yeniden boyutlandır
        pos_embed_2d = F.interpolate(
            pos_embed_2d.permute(0, 3, 1, 2),  # [1, embed_dim, orig_size, orig_size]
            size=(height, width),
            mode=self.config.interpolate_mode,
            align_corners=False
        )
        
        # 1D formata geri dönüştür [1, height*width, embed_dim]
        pos_embed_2d = pos_embed_2d.permute(0, 2, 3, 1)  # [1, height, width, embed_dim]
        return pos_embed_2d.reshape(1, height * width, embed_dim)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        ImagePatchEmbedding modülünün ileri geçişi.
        
        Args:
            pixel_values: Görüntü pikselleri [batch_size, channels, height, width]
            attention_mask: Dikkat maskesi, opsiyonel [batch_size, height, width]
            return_dict: Çıktı sözlük formatında döndürülsün mü?
            
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: Yama gömme vektörleri
                [batch_size, num_patches, embed_dim] veya çıktı sözlüğü
        """
        batch_size = pixel_values.shape[0]
        
        # Pikselleri nn.Conv2d ile yamalara dönüştür
        x = self.projection(pixel_values)  # [batch_size, embed_dim, grid_h, grid_w]
        
        # Boyutları al
        _, _, height, width = x.shape
        
        # [batch_size, embed_dim, grid_h, grid_w] -> [batch_size, embed_dim, grid_h*grid_w]
        x = x.flatten(2)
        
        # [batch_size, embed_dim, grid_h*grid_w] -> [batch_size, grid_h*grid_w, embed_dim]
        x = x.transpose(1, 2)
        
        # Konum gömmeleri ekle (eğer kullanılıyorsa)
        if self.config.use_position_embedding:
            if self.config.position_embedding_type == "learned":
                # Öğrenilmiş konum gömmelerini kullan
                pos_embed = self.position_embedding
                
                # Gerekirse gömme matrisini yeniden boyutlandır
                if pos_embed.shape[1] != height * width:
                    pos_embed = self._resize_pos_embed(pos_embed, height, width)
                
                x = x + pos_embed
            elif self.config.position_embedding_type == "sincos":
                # Sinüzoidal konum kodlaması oluştur ve ekle
                pos_embed = get_2d_sincos_pos_embed(
                    self.config.embed_dim, height, width, dtype=x.dtype, device=x.device
                )
                x = x + pos_embed.unsqueeze(0)  # [1, grid_h*grid_w, embed_dim]
        
        # Projeksiyon katmanını uygula (kullanılıyorsa)
        if self.projection_layer is not None:
            x = self.projection_layer(x)
        
        # Layer normalization ve dropout uygula
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Attention mask'ı embeddings ile çarp (opsiyonel)
        if attention_mask is not None:
            # Mask'ı uygun boyuta getir
            # Giriş: [batch_size, height, width]
            # Dönüşüm: [batch_size, grid_h, grid_w] -> [batch_size, grid_h*grid_w, 1]
            mask = F.interpolate(
                attention_mask.float().unsqueeze(1),
                size=(height, width),
                mode="nearest"
            ).squeeze(1).flatten(1).unsqueeze(-1)
            
            # Mask'ı embeddings'e uygula
            x = x * mask
        
        if return_dict:
            return {
                "embeddings": x,
                "attention_mask": attention_mask
            }
        else:
            return x


class ImagePatchEmbeddingFactory:
    """
    Farklı görüntü yama embedding türleri oluşturmak için fabrika sınıfı.
    
    Örüntü: FactoryMethod (PT-002)
    """
    
    @staticmethod
    def create_image_patch_embedding(config: ImagePatchEmbeddingConfig) -> ImagePatchEmbedding:
        """
        Yapılandırmaya göre görüntü yama embedding modülü oluşturur.
        
        Args:
            config: Görüntü yama embedding yapılandırması
            
        Returns:
            ImagePatchEmbedding: Oluşturulan görüntü yama embedding modülü
        """
        return ImagePatchEmbedding(config)
    
    @staticmethod
    def create_efficient_embedding(
        config: ImagePatchEmbeddingConfig,
        efficiency_factor: float = 0.5
    ) -> ImagePatchEmbedding:
        """
        Verimli görüntü yama embedding modülü oluşturur.
        
        Daha az kaynak kullanan ve mobil cihazlara optimizasyonu artıran bir embedding oluşturur.
        
        Args:
            config: Görüntü yama embedding yapılandırması
            efficiency_factor: Verimlilik faktörü (0-1 arası)
            
        Returns:
            ImagePatchEmbedding: Verimli görüntü yama embedding modülü
        """
        # Yapılandırmayı kopyala ve değiştir
        efficient_config = ImagePatchEmbeddingConfig(
            image_size=config.image_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            embed_dim=max(32, int(config.embed_dim * efficiency_factor)),  # Minimum 32
            use_embedding_projection=True,
            projection_dim=max(16, int((config.projection_dim or config.embed_dim) * efficiency_factor)),
            use_position_embedding=config.use_position_embedding,
            position_embedding_type="sincos",  # Sincos daha verimli
            layer_norm_eps=config.layer_norm_eps,
            dropout_rate=min(0.2, config.dropout_rate + 0.1),  # Daha fazla dropout
            interpolate_mode="bilinear",  # Daha hızlı interpolasyon
        )
        
        return ImagePatchEmbedding(efficient_config) 