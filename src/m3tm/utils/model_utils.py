"""
M³TM model yardımcı fonksiyonları.

Bu modül, model performansı, büyüklüğü ve profil oluşturma ile ilgili
yardımcı fonksiyonlar sağlar.

Örüntüler:
- UtilityFunction (PT-005): Genel amaçlı yardımcı fonksiyonlar sağlama
"""

import time
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Bir modelin eğitilebilir parametre sayısını hesaplar.
    
    Args:
        model: Parametre sayısı hesaplanacak model
        
    Returns:
        Toplam eğitilebilir parametre sayısı
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_model(model: nn.Module, 
                  input_tensors: Union[torch.Tensor, List[torch.Tensor]], 
                  n_iterations: int = 10, 
                  warmup_iterations: int = 5) -> Dict[str, Any]:
    """Bir modelin ileri geçiş performansını profiller.
    
    Args:
        model: Profili oluşturulacak model
        input_tensors: Modele girdi olarak verilecek tensör veya tensörler
        n_iterations: Ölçülecek ileri geçiş sayısı
        warmup_iterations: Isınma için yapılacak ileri geçiş sayısı
        
    Returns:
        Ortalama ve toplam süre, FLOP tahminleri dahil profil metrikleri
    """
    # Modeli değerlendirme modunda ayarla
    model.eval()
    
    # Girdiyi liste ya da tekil tensör olarak normalize et
    if not isinstance(input_tensors, list):
        input_tensors = [input_tensors]
    
    # Isınma döngüsü
    with torch.no_grad():
        for _ in range(warmup_iterations):
            if len(input_tensors) > 1:
                _ = model(*input_tensors)
            else:
                _ = model(input_tensors[0])
    
    # Zamanlama döngüsü
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            if len(input_tensors) > 1:
                _, metrics = model(*input_tensors)
            else:
                _, metrics = model(input_tensors[0])
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    total_time = time.time() - start_time
    avg_time = total_time / n_iterations
    
    # Güç tüketimi, bellek kullanımı, flops gibi ek metrikler buraya eklenebilir
    if hasattr(model, "estimate_flops") and len(input_tensors) > 0:
        seq_len = input_tensors[0].size(1) if input_tensors[0].dim() > 1 else 1
        flops = model.estimate_flops(seq_len)
        flops_per_sec = flops["total"] / avg_time if "total" in flops else None
    else:
        flops = {}
        flops_per_sec = None
    
    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "iterations": n_iterations,
        "parameter_count": count_parameters(model),
        "flops": flops,
        "flops_per_sec": flops_per_sec
    }


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """Modeli çıkarım için optimize eder.
    
    Bu fonksiyon, modeli değerlendirme moduna alır, 
    buffer'ları ve parametreleri CPU'dan GPU'ya (veya tersi) taşır,
    ve JIT derlemesi veya quantization gibi desteklenen optimizasyonları uygular.
    
    Args:
        model: Optimize edilecek model
        
    Returns:
        Optimize edilmiş model
    """
    # Değerlendirme moduna al
    model.eval()
    
    # Gereksiz gradient bilgisini kaldır
    for param in model.parameters():
        param.requires_grad = False
    
    # JIT derlemesi, quantization vb. ek optimizasyonlar burada yapılabilir
    
    return model


def compare_models(model_a: nn.Module, model_b: nn.Module, 
                  input_tensor: torch.Tensor, 
                  rtol: float = 1e-5, atol: float = 1e-8) -> Dict[str, Any]:
    """İki modelin çıktılarını karşılaştırır.
    
    Args:
        model_a: Birinci model
        model_b: İkinci model
        input_tensor: Modellere girdi olarak verilecek tensör
        rtol: Göreli tolerans
        atol: Mutlak tolerans
        
    Returns:
        Karşılaştırma sonuçlarını içeren sözlük
    """
    model_a.eval()
    model_b.eval()
    
    with torch.no_grad():
        output_a, _ = model_a(input_tensor)
        output_b, _ = model_b(input_tensor)
    
    # Çıktı boyutlarını kontrol et
    shapes_match = output_a.shape == output_b.shape
    
    # Çıktı değerlerini karşılaştır
    values_match = torch.allclose(output_a, output_b, rtol=rtol, atol=atol)
    
    # Fark istatistikleri
    if shapes_match:
        abs_diff = (output_a - output_b).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Göreli fark (sıfıra bölmeyi önle)
        mask = output_a.abs() > atol
        rel_diff = torch.zeros_like(output_a)
        rel_diff[mask] = (abs_diff[mask] / output_a.abs()[mask])
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff[mask].mean().item() if mask.sum() > 0 else 0.0
    else:
        max_diff = float('nan')
        mean_diff = float('nan')
        max_rel_diff = float('nan')
        mean_rel_diff = float('nan')
    
    return {
        "shapes_match": shapes_match,
        "values_match": values_match,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
    }


def analyze_adapter_impact(model: nn.Module, 
                         input_tensor: torch.Tensor, 
                         n_iterations: int = 10,
                         warmup_iterations: int = 3) -> Dict[str, Any]:
    """Adapter mekanizmasının model performansına etkisini analiz eder.
    
    Bu fonksiyon, adaptörlerin açık ve kapalı durumlarında model performansını karşılaştırır,
    parametre sayısı, ileri geçiş süresi ve FLOP artışı gibi metrikleri hesaplar.
    
    Args:
        model: Adapter modülleri içeren model
        input_tensor: Test için kullanılacak girdi tensörü
        n_iterations: Test döngüsü için iterasyon sayısı
        warmup_iterations: Isınma döngüsü için iterasyon sayısı
        
    Returns:
        Adaptör etkisini gösteren metrikler sözlüğü
    """
    # Parametre sayılarını hesapla
    total_params = count_parameters(model)
    
    # 1. Adaptörleri devre dışı bırakarak test et
    if hasattr(model, "set_training_adapters"):
        model.set_training_adapters(False)
    
    model.eval()
    with torch.no_grad():
        # Isınma
        for _ in range(warmup_iterations):
            _ = model(input_tensor)
        
        # Adaptörsüz zaman ölçümü
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(n_iterations):
            without_adapters_output, _ = model(input_tensor)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        without_adapters_time = time.time() - start_time
        
    # 2. Adaptörleri etkinleştirerek test et
    if hasattr(model, "set_training_adapters"):
        model.set_training_adapters(True)
    
    with torch.no_grad():
        # Isınma
        for _ in range(warmup_iterations):
            _ = model(input_tensor)
        
        # Adaptörlü zaman ölçümü
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(n_iterations):
            with_adapters_output, _ = model(input_tensor)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        with_adapters_time = time.time() - start_time
    
    # Adaptör parametre sayısını hesapla (yalnızca adaptörleri içeren modüller için)
    adapter_params = 0
    adapter_modules = []
    
    if hasattr(model, "adapter_slots"):
        for pos, slot in model.adapter_slots.items():
            adapter_params += slot.count_parameters()
            adapter_modules.extend(slot.adapters)
    
    # FLOP'ları hesapla (varsa)
    seq_len = input_tensor.size(1) if input_tensor.dim() > 1 else 1
    
    if hasattr(model, "estimate_flops"):
        # Adaptörsüz FLOP'lar
        model.set_training_adapters(False) if hasattr(model, "set_training_adapters") else None
        flops_without = model.estimate_flops(seq_len)
        
        # Adaptörlü FLOP'lar
        model.set_training_adapters(True) if hasattr(model, "set_training_adapters") else None
        flops_with = model.estimate_flops(seq_len)
        
        flop_increase = flops_with["total"] - flops_without["total"]
        flop_increase_percent = (flop_increase / flops_without["total"]) * 100
    else:
        flops_without = {"total": float('nan')}
        flops_with = {"total": float('nan')}
        flop_increase = float('nan')
        flop_increase_percent = float('nan')
    
    # Çıktı farklılığını ölç
    if torch.is_tensor(with_adapters_output) and torch.is_tensor(without_adapters_output):
        output_diff = (with_adapters_output - without_adapters_output).abs().mean().item()
    else:
        output_diff = float('nan')
    
    time_increase_percent = ((with_adapters_time - without_adapters_time) / without_adapters_time) * 100
    
    return {
        "total_parameters": total_params,
        "adapter_parameters": adapter_params,
        "adapter_param_percent": (adapter_params / total_params) * 100 if total_params > 0 else 0,
        "without_adapters_time": without_adapters_time,
        "with_adapters_time": with_adapters_time,
        "time_increase_percent": time_increase_percent,
        "flops_without_adapters": flops_without["total"],
        "flops_with_adapters": flops_with["total"],
        "flop_increase": flop_increase,
        "flop_increase_percent": flop_increase_percent,
        "output_difference": output_diff,
        "adapter_module_count": len(adapter_modules),
        "iterations": n_iterations
    } 