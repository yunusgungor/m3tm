"""
Görüntü Yama İşlemleri

Bu modül, görüntüleri eşit boyutlu yamalara bölmek ve işlemek için
yardımcı fonksiyonlar ve sınıflar içerir.
"""

from typing import Tuple, List, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


def patchify_image(
    images: torch.Tensor,
    patch_size: int,
    flatten: bool = True
) -> torch.Tensor:
    """
    Görüntüleri eşit boyutlu yamalara böler.
    
    Args:
        images: [batch_size, channels, height, width] şeklinde görüntü tensörü
        patch_size: Kare yama boyutu (pixeller)
        flatten: Yamaları düzleştirme bayrağı
        
    Returns:
        Yama tensörü:
            flatten=True ise: [batch_size, num_patches, patch_size*patch_size*channels]
            flatten=False ise: [batch_size, num_patches, channels, patch_size, patch_size]
            
    Raises:
        ValueError: Görüntü boyutları yama boyutunun tam katı değilse
    """
    batch_size, channels, height, width = images.shape
    
    # Görüntü boyutlarının yama boyutuna bölünebilirliğini kontrol et
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Görüntü boyutları ({height}, {width}) yama boyutunun ({patch_size}) "
            f"tam katı olmalıdır."
        )
    
    # Yama sayılarını hesapla
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Yamalar oluştur - İki yaklaşım:
    
    # 1. Yaklaşım: unfold kullanarak (daha bellek verimli)
    # [B, C, H, W] -> [B, C, num_patches_h, patch_size, num_patches_w, patch_size]
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # Boyutları yeniden düzenle
    # [B, C, num_patches_h, patch_size, num_patches_w, patch_size] -> 
    # [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
    patches = patches.contiguous()
    
    if flatten:
        # [B, C, num_patches_h, num_patches_w, patch_size, patch_size] -> 
        # [B, num_patches_h*num_patches_w, C*patch_size*patch_size]
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(batch_size, num_patches, -1)
    else:
        # [B, C, num_patches_h, num_patches_w, patch_size, patch_size] -> 
        # [B, num_patches_h*num_patches_w, C, patch_size, patch_size]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, num_patches, channels, patch_size, patch_size)
    
    return patches


def unpatchify_image(
    patches: torch.Tensor,
    patch_size: int,
    image_size: Optional[Union[Tuple[int, int], int]] = None
) -> torch.Tensor:
    """
    Yamalardan görüntüyü yeniden oluşturur.
    
    Args:
        patches: Görüntü yamaları. Şekil:
            - flatten=True ise: [B, num_patches, channels*patch_size*patch_size]
            - flatten=False ise: [B, num_patches, channels, patch_size, patch_size]
        patch_size: Yama boyutu
        image_size: Orijinal görüntü boyutu (yükseklik, genişlik) veya
                   kare görüntü için tek bir değer olarak boyut.
                   None ise, kare görüntü varsayılır.
    
    Returns:
        torch.Tensor: Yeniden oluşturulmuş görüntü [B, channels, H, W]
    """
    if patches.dim() == 3:
        # [B, num_patches, channels*patch_size*patch_size] -> [B, num_patches, channels, patch_size, patch_size]
        batch_size, num_patches, flattened_dim = patches.shape
        channels = flattened_dim // (patch_size ** 2)
        patches = patches.view(batch_size, num_patches, channels, patch_size, patch_size)
    
    batch_size, num_patches, channels, patch_size, patch_size = patches.shape
    
    # Görüntü boyutunu belirle
    if image_size is None:
        # Kare görüntü varsayılır
        num_patches_per_side = int(math.sqrt(num_patches))
        if num_patches_per_side ** 2 != num_patches:
            raise ValueError(
                f"Kare olmayan görüntüler için image_size belirtilmelidir. "
                f"num_patches={num_patches}"
            )
        image_h = image_w = num_patches_per_side * patch_size
    else:
        # Eğer image_size bir int ise, kare görüntü olarak düşün
        if isinstance(image_size, int):
            image_h = image_w = image_size
        else:
            # Tuple olarak görüntü boyutu
            image_h, image_w = image_size
            
        # Tutarlılık kontrolü
        if (image_h // patch_size) * (image_w // patch_size) != num_patches:
            raise ValueError(
                f"image_size {image_size} ve patch_size {patch_size} değerleri, "
                f"num_patches {num_patches} ile tutarsız."
            )
    
    # Yama sayılarını hesapla
    num_patches_h = image_h // patch_size
    num_patches_w = image_w // patch_size
    
    # [B, N, C, P, P] -> [B, num_patches_h, num_patches_w, C, P, P]
    patches = patches.view(batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
    
    # [B, num_patches_h, num_patches_w, C, P, P] -> [B, C, image_h, image_w]
    # Önce [B, C, num_patches_h, P, num_patches_w, P] haline getir
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    # Sonra [B, C, image_h, image_w] için birleştir
    images = patches.view(batch_size, channels, image_h, image_w)
    
    return images


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    cls_token: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    2B sinüzoidal konum kodlaması oluşturur.
    
    ViT (Vision Transformer) tarzı 2D konum kodlaması, her konum için
    sinüs ve kosinüs değerlerinin bir karışımını kullanır. Bu kodlama,
    her uzamsal konumu benzersiz bir şekilde temsil eder.
    
    Args:
        embed_dim: Gömme boyutu (çift sayı olmalıdır)
        grid_h: Izgara yüksekliği (yama sayısı)
        grid_w: Izgara genişliği (yama sayısı)
        cls_token: [CLS] token pozisyonu dahil edilsin mi?
        dtype: Dönüş tensörünün veri tipi
        device: Dönüş tensörünün cihazı
    
    Returns:
        torch.Tensor: 2D konum kodlaması
            cls_token=False ise: [grid_h*grid_w, embed_dim]
            cls_token=True ise: [1+grid_h*grid_w, embed_dim], sıfır ile başlayan [CLS] kodlaması
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim {embed_dim} çift sayı olmalıdır")
    
    # Konum için kullanılacak boyut sayısı
    omega = torch.arange(embed_dim // 4, dtype=dtype, device=device) / (embed_dim // 4)
    omega = 1. / (10000 ** omega)
    
    # Her iki eksen için pozisyonlar oluştur
    y_pos = torch.arange(grid_h, dtype=dtype, device=device).reshape(-1, 1)  # [grid_h, 1]
    x_pos = torch.arange(grid_w, dtype=dtype, device=device).reshape(-1, 1)  # [grid_w, 1]
    
    # Açıları hesapla: y için sin/cos, x için sin/cos
    y_angles = y_pos * omega.reshape(1, -1)  # [grid_h, embed_dim//4]
    x_angles = x_pos * omega.reshape(1, -1)  # [grid_w, embed_dim//4]
    
    # Sin ve cos değerlerini hesapla
    y_sin = torch.sin(y_angles)  # [grid_h, embed_dim//4]
    y_cos = torch.cos(y_angles)  # [grid_h, embed_dim//4]
    x_sin = torch.sin(x_angles)  # [grid_w, embed_dim//4]
    x_cos = torch.cos(x_angles)  # [grid_w, embed_dim//4]
    
    # Grid pozisyonları için konum kodlamasını oluştur
    pos_embed = torch.zeros(grid_h, grid_w, embed_dim, dtype=dtype, device=device)
    
    # Y ekseni için sin/cos değerlerini yerleştir (embed_dim'in 1/4'ü)
    dim_y_sin = embed_dim // 4
    pos_embed[:, :, :dim_y_sin] = y_sin.unsqueeze(1).repeat(1, grid_w, 1)
    pos_embed[:, :, dim_y_sin:2*dim_y_sin] = y_cos.unsqueeze(1).repeat(1, grid_w, 1)
    
    # X ekseni için sin/cos değerlerini yerleştir (embed_dim'in 1/4'ü)
    dim_x_sin = 2 * dim_y_sin
    dim_x_cos = 3 * dim_y_sin
    pos_embed[:, :, dim_x_sin:dim_x_cos] = x_sin.unsqueeze(0).repeat(grid_h, 1, 1)
    pos_embed[:, :, dim_x_cos:] = x_cos.unsqueeze(0).repeat(grid_h, 1, 1)
    
    # [grid_h, grid_w, embed_dim] -> [grid_h*grid_w, embed_dim]
    pos_embed = pos_embed.view(grid_h * grid_w, embed_dim)
    
    # [CLS] token ekle (opsiyonel)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim, dtype=dtype, device=device), pos_embed], dim=0)
    
    return pos_embed


def get_1d_sincos_pos_embed(
    embed_dim: int,
    length: int,
    cls_token: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    1B sinüzoidal konum kodlaması oluşturur.
    
    Transformer modellerinde yaygın olarak kullanılan 1D konum kodlaması,
    dizideki her konum için sinüs ve kosinüs değerlerinin karışımını kullanarak
    benzersiz bir temsil oluşturur.
    
    Args:
        embed_dim: Gömme boyutu (çift sayı olmalıdır)
        length: Dizi uzunluğu (konum sayısı)
        cls_token: [CLS] token pozisyonu dahil edilsin mi?
        dtype: Dönüş tensörünün veri tipi
        device: Dönüş tensörünün cihazı
    
    Returns:
        torch.Tensor: 1D konum kodlaması
            cls_token=False ise: [length, embed_dim]
            cls_token=True ise: [1+length, embed_dim], sıfır ile başlayan [CLS] kodlaması
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim {embed_dim} çift sayı olmalıdır")
    
    # Konum için kullanılacak boyut sayısı
    omega = torch.arange(embed_dim // 2, dtype=dtype, device=device) / (embed_dim // 2)
    omega = 1. / (10000 ** omega)  # [embed_dim//2]
    
    # Pozisyonlar oluştur
    pos = torch.arange(length, dtype=dtype, device=device).reshape(-1, 1)  # [length, 1]
    
    # Açıları hesapla
    pos_angles = pos * omega.reshape(1, -1)  # [length, embed_dim//2]
    
    # Sin ve cos değerlerini birbiri ardına yerleştir
    pos_embed = torch.zeros(length, embed_dim, dtype=dtype, device=device)
    pos_embed[:, 0::2] = torch.sin(pos_angles)  # Çift indeksler için sin
    pos_embed[:, 1::2] = torch.cos(pos_angles)  # Tek indeksler için cos
    
    # [CLS] token ekle (opsiyonel)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim, dtype=dtype, device=device), pos_embed], dim=0)
    
    return pos_embed


def resize_pos_embed(
    pos_embed: torch.Tensor,
    new_grid_h: int,
    new_grid_w: int,
    has_cls_token: bool = True,
    interpolation_mode: str = "bicubic"
) -> torch.Tensor:
    """
    Önceden oluşturulmuş konum kodlaması matrisini yeni ızgara boyutuna yeniden boyutlandırır.
    
    Bu fonksiyon, farklı boyutlara sahip görüntüler için konum kodlamasını uyarlamak amacıyla
    interpolasyon kullanır.
    
    Args:
        pos_embed: Konum kodlaması matrisi, şekli:
            has_cls_token=True ise: [1, 1+grid_h*grid_w, embed_dim]
            has_cls_token=False ise: [1, grid_h*grid_w, embed_dim]
        new_grid_h: Yeni ızgara yüksekliği (yama sayısı)
        new_grid_w: Yeni ızgara genişliği (yama sayısı)
        has_cls_token: Giriş konum kodlaması [CLS] token içeriyor mu?
        interpolation_mode: Interpolasyon modu ("bicubic", "bilinear", vb.)
        
    Returns:
        torch.Tensor: Yeniden boyutlandırılmış konum kodlaması matrisi
    """
    # Giriş kontrolü
    assert pos_embed.dim() == 3, f"pos_embed boyutu 3 olmalıdır, alınan: {pos_embed.dim()}"
    embed_dim = pos_embed.shape[2]
    
    # [CLS] token'ı varsa ayır
    cls_token_embed = None
    if has_cls_token:
        cls_token_embed = pos_embed[:, 0:1, :]
        pos_embed = pos_embed[:, 1:, :]
    
    # Mevcut ızgara boyutunu hesapla
    num_positions = pos_embed.shape[1]
    grid_side = int(math.sqrt(num_positions))
    
    # Konum kodlamasını ızgara şeklinde yeniden düzenle
    pos_embed = pos_embed.reshape(1, grid_side, grid_side, embed_dim)
    
    # Interpolasyon için kanalları öne al [1, grid_side, grid_side, embed_dim] -> [1, embed_dim, grid_side, grid_side]
    pos_embed = pos_embed.permute(0, 3, 1, 2)
    
    # Yeni ızgara boyutuna göre yeniden boyutlandır
    pos_embed = F.interpolate(
        pos_embed, 
        size=(new_grid_h, new_grid_w), 
        mode=interpolation_mode,
        align_corners=False
    )
    
    # Kanalları geri al [1, embed_dim, new_grid_h, new_grid_w] -> [1, new_grid_h, new_grid_w, embed_dim]
    pos_embed = pos_embed.permute(0, 2, 3, 1)
    
    # Düzleştir [1, new_grid_h, new_grid_w, embed_dim] -> [1, new_grid_h*new_grid_w, embed_dim]
    pos_embed = pos_embed.reshape(1, new_grid_h * new_grid_w, embed_dim)
    
    # [CLS] token'ı geri ekle
    if has_cls_token and cls_token_embed is not None:
        pos_embed = torch.cat([cls_token_embed, pos_embed], dim=1)
    
    return pos_embed 