"""
AdapterSlot mekanizmasının ProtoTransformerBlock entegrasyonu test örneği.

Bu örnek, geliştirilmiş AdapterSlot mekanizmasıyla ProtoTransformerBlock'un
entegrasyonunu test eder ve çeşitli kullanım senaryolarını gösterir.

Örüntüler:
- CompositeAdapter (PT-014): Birden fazla adaptörü sıralı olarak uygulama
- PluggableComponentStrategy: Değiştirilebilir dikkat ve FFN mekanizmaları
"""

import torch
import torch.nn as nn
import time
import sys
import os

# src klasörünü Python path'ine ekle (doğrudan çalıştırıldığında gerekli)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from m3tm.transformer.config import ProtoTransformerConfig, AdapterConfig
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.transformer.adapter import Adapter, ParallelAdapter
from m3tm.utils.model_utils import count_parameters, profile_model, analyze_adapter_impact


def configure_model():
    """Test için ProtoTransformerBlock konfigürasyonu oluşturur."""
    # Adapter'ları etkinleştiren bir konfigürasyon oluştur
    adapter_config = AdapterConfig(
        enabled=True,
        bottleneck_dim=32,
        use_layer_norm=True,
        adapter_type="bottleneck",
        activation="hardswish",
        dropout=0.1,
        residual_connection=True,
        alpha=1.0,
        adapter_positions=["pre_attention", "post_attention", "pre_ffn", "post_ffn"]
    )
    
    # ProtoTransformerBlock konfigürasyonu
    config = ProtoTransformerConfig(
        hidden_size=256,
        num_layers=1,
        dropout=0.1,
        adapter_config=adapter_config
    )
    
    return config


def test_adapter_slot_integration():
    """AdapterSlot entegrasyonunu test eder."""
    print("AdapterSlot ProtoTransformerBlock Entegrasyon Testi")
    print("-" * 60)
    
    # Model konfigürasyonu
    config = configure_model()
    
    # Model oluştur
    model = ProtoTransformerBlock(config)
    
    # Örnek girdi
    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # İlk çıktıyı al (adapter'sız)
    with torch.no_grad():
        y1, metrics = model(x)
    
    # Parametre sayısını kontrol et
    print(f"Temel model parametre sayısı: {count_parameters(model):,}")
    
    # Adapter pozisyonlarını kontrol et
    print(f"Adapter pozisyonları: {model.get_adapter_positions()}")
    
    # Test 1: Bir pozisyona adapter ekle
    print("\nTest 1: Bir pozisyona adapter ekle")
    adapter1 = Adapter(
        config=AdapterConfig(
            bottleneck_dim=32,
            use_layer_norm=True,
            activation="hardswish",
        ),
        input_dim=hidden_size
    )
    
    success = model.register_adapter(adapter1, "post_attention", "test_adapter_1")
    print(f"Adapter ekleme başarılı: {success}")
    print(f"post_attention pozisyonundaki adapter'lar: {model.get_adapter_names('post_attention')}")
    
    # Adapter'lı çıktıyı al
    with torch.no_grad():
        y2, _ = model(x)
    
    # Çıktı farkını kontrol et
    diff = (y1 - y2).abs().mean().item()
    print(f"Çıktı farkı (adapter ekledikten sonra): {diff:.6f}")
    
    # Test 2: Birden fazla adapter ekle
    print("\nTest 2: Birden fazla adapter ekle")
    adapter2 = ParallelAdapter(
        config=AdapterConfig(
            bottleneck_dim=16,
            use_layer_norm=True,
            activation="gelu",
            alpha=0.5,
        ),
        input_dim=hidden_size
    )
    
    success = model.register_adapter(adapter2, "post_attention", "test_adapter_2")
    print(f"İkinci adapter ekleme başarılı: {success}")
    print(f"post_attention pozisyonundaki adapter'lar: {model.get_adapter_names('post_attention')}")
    
    # İki adapter'lı çıktıyı al
    with torch.no_grad():
        y3, _ = model(x)
    
    # Çıktı farkını kontrol et
    diff = (y2 - y3).abs().mean().item()
    print(f"Çıktı farkı (ikinci adapter ekledikten sonra): {diff:.6f}")
    
    # Test 3: Adapter eğitim modu testi
    print("\nTest 3: Adapter eğitim modu testi")
    
    # Eğitim modunu kapat
    model.set_training_adapters(False)
    
    # Çıkarım modunda çıktıyı al
    model.eval()
    with torch.no_grad():
        y4, _ = model(x)
    
    # Eğitim modunu geri aç
    model.set_training_adapters(True)
    model.train()
    with torch.no_grad():
        y5, _ = model(x)
    
    # Çıktı farklarını kontrol et
    diff_inference = (y1 - y4).abs().mean().item()
    diff_training = (y3 - y5).abs().mean().item()
    print(f"Çıktı farkı (çıkarım modunda vs. başlangıç): {diff_inference:.6f}")
    print(f"Çıktı farkı (eğitim modu geri açıldığında vs. iki adapter'lı): {diff_training:.6f}")
    
    # Test 4: Adapter kaldırma
    print("\nTest 4: Adapter kaldırma")
    
    success = model.remove_adapter("post_attention", "test_adapter_1")
    print(f"Birinci adapter'ı kaldırma başarılı: {success}")
    print(f"post_attention pozisyonundaki adapter'lar: {model.get_adapter_names('post_attention')}")
    
    with torch.no_grad():
        y6, _ = model(x)
    
    diff = (y3 - y6).abs().mean().item()
    print(f"Çıktı farkı (bir adapter kaldırıldıktan sonra): {diff:.6f}")
    
    # Tüm adapter'ları kaldır
    for pos in model.get_adapter_positions():
        adapter_names = model.get_adapter_names(pos).copy()
        for name in adapter_names:
            model.remove_adapter(pos, name)
    
    print(f"Tüm adapter'lar kaldırıldı.")
    for pos in model.get_adapter_positions():
        print(f"{pos} pozisyonundaki adapter'lar: {model.get_adapter_names(pos)}")
    
    # Tüm adapter'lar kaldırıldıktan sonra çıktıyı kontrol et
    with torch.no_grad():
        y7, _ = model(x)
    
    diff = (y1 - y7).abs().mean().item()
    print(f"Çıktı farkı (tüm adapter'lar kaldırıldıktan sonra vs başlangıç): {diff:.6f}")
    
    # Test 5: Performans testi
    print("\nTest 5: Performans testi")
    
    # Adapter'sız performans
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(x)
    adapter_free_time = time.time() - start_time
    print(f"Adapter'sız 100 ileri geçiş süresi: {adapter_free_time:.4f} saniye")
    
    # Tek adapter ekle
    model.register_adapter(adapter1, "post_attention", "perf_test_adapter")
    
    # Adapter'lı performans
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(x)
    adapter_time = time.time() - start_time
    print(f"Adapter'lı 100 ileri geçiş süresi: {adapter_time:.4f} saniye")
    print(f"Performans farkı: {(adapter_time - adapter_free_time) / adapter_free_time * 100:.2f}%")
    
    # Test 6: FLOP ve parametre sayısı analizi
    print("\nTest 6: FLOP ve parametre analizi")
    
    # Adapter'sız FLOP
    for pos in model.get_adapter_positions():
        adapter_names = model.get_adapter_names(pos).copy()
        for name in adapter_names:
            model.remove_adapter(pos, name)
    
    flops_base = model.estimate_flops(seq_len)
    print(f"Adapter'sız FLOP'lar: {flops_base['total']:,}")
    
    # İki adapter ekle (farklı pozisyonlara)
    model.register_adapter(adapter1, "post_attention", "flop_test_adapter1")
    model.register_adapter(adapter2, "post_ffn", "flop_test_adapter2")
    
    flops_with_adapters = model.estimate_flops(seq_len)
    print(f"Adapter'lı FLOP'lar: {flops_with_adapters['total']:,}")
    print(f"Adapter FLOP'ları: {flops_with_adapters['adapters']:,}")
    print(f"FLOP artışı: {(flops_with_adapters['total'] - flops_base['total']) / flops_base['total'] * 100:.2f}%")
    
    # Parametre sayısı analizi
    params_base = count_parameters(model)
    adapter_params = sum(adapter1.count_parameters() + adapter2.count_parameters())
    print(f"Adapter parametre sayısı: {adapter_params:,}")
    print(f"Temel model parametre sayısı: {params_base:,}")
    print(f"Parametre artışı: {adapter_params / params_base * 100:.2f}%")
    
    # Test 7: Detaylı performans analizi
    print("\nTest 7: Detaylı performans analizi")
    
    # Tüm adapter'ları temizle
    for pos in model.get_adapter_positions():
        adapter_names = model.get_adapter_names(pos).copy()
        for name in adapter_names:
            model.remove_adapter(pos, name)
    
    # Farklı sayıda adapter ekleyerek performans analizleri yap
    results = []
    
    # 1. Adapter yok
    impact = analyze_adapter_impact(model, x)
    print("\nAdapter Yok - Performans Analizi:")
    print(f"Adaptör parametre oranı: {impact['adapter_param_percent']:.2f}%")
    print(f"İleri geçiş süre farkı: {impact['time_increase_percent']:.2f}%")
    print(f"FLOP artışı: {impact['flop_increase_percent']:.2f}%")
    results.append(impact)
    
    # 2. Tek adapter ekle
    model.register_adapter(adapter1, "post_attention", "test_adapter")
    impact = analyze_adapter_impact(model, x)
    print("\nTek Adapter - Performans Analizi:")
    print(f"Adaptör parametre oranı: {impact['adapter_param_percent']:.2f}%")
    print(f"İleri geçiş süre farkı: {impact['time_increase_percent']:.2f}%")
    print(f"FLOP artışı: {impact['flop_increase_percent']:.2f}%")
    results.append(impact)
    
    # 3. Çoklu adapter ekle
    model.register_adapter(adapter2, "post_ffn", "test_adapter2")
    model.register_adapter(Adapter(hidden_size, 8), "pre_attention", "test_adapter3")
    impact = analyze_adapter_impact(model, x)
    print("\nÇoklu Adapter (3) - Performans Analizi:")
    print(f"Adaptör parametre oranı: {impact['adapter_param_percent']:.2f}%")
    print(f"İleri geçiş süre farkı: {impact['time_increase_percent']:.2f}%")
    print(f"FLOP artışı: {impact['flop_increase_percent']:.2f}%")
    results.append(impact)
    
    print("\nTest başarıyla tamamlandı!")


if __name__ == "__main__":
    test_adapter_slot_integration() 