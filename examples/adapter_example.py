#!/usr/bin/env python3
"""
M³TM adapter kullanım örneği.

Bu örnek, M³TM modelinde adapter'ların nasıl kullanılacağını gösterir.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from m3tm.transformer.config import ProtoTransformerConfig
from m3tm.transformer.proto_transformer import ProtoTransformerBlock
from m3tm.adapters.adapter import AdapterConfig, AdapterType
from m3tm.adapters.adapter_manager import AdapterManager
from m3tm.adapters.adapter_utils import (
    get_transformer_compatible_adapters,
    register_adapters_to_transformers,
    create_default_adapter_for_model,
    freeze_model_except_adapters,
    get_adapter_parameter_count
)


def create_simple_model():
    """Örnek bir transformer modeli oluşturur."""
    config = ProtoTransformerConfig(
        hidden_size=64,
        num_layers=2,
        max_sequence_length=128,
        vocab_size=1000,
        dropout=0.1
    )
    
    class SimpleTransformerModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.blocks = nn.ModuleList([
                ProtoTransformerBlock(config) for _ in range(config.num_layers)
            ])
            self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            
            for block in self.blocks:
                x, _ = block(x)
                
            x = self.output_layer(x)
            return x
    
    return SimpleTransformerModel(config)


def main():
    parser = argparse.ArgumentParser(description="M³TM adapter örneği")
    parser.add_argument("--save_dir", type=str, default="adapter_checkpoint", help="Adapter'ların kaydedileceği dizin")
    parser.add_argument("--bottleneck_dim", type=int, default=8, help="Adapter darboğaz boyutu")
    parser.add_argument("--adapter_type", type=str, default="bottleneck", choices=["bottleneck", "parallel"], help="Adapter türü")
    parser.add_argument("--positions", type=str, default="post_attention,post_ffn", help="Adapter pozisyonları (virgülle ayrılmış)")
    parser.add_argument("--alpha", type=float, default=0.8, help="Parallel adapter için alfa değeri")
    args = parser.parse_args()
    
    # Pozisyonları parse et
    positions = args.positions.split(",") if args.positions else None
    
    # Modeli oluştur
    print("Model oluşturuluyor...")
    model = create_simple_model()
    
    # Model hakkında bilgi
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam model parametreleri: {total_params:,}")
    
    # Transformer bloklarını bul
    transformer_blocks = get_transformer_compatible_adapters(model)
    print(f"Bulunan transformer blokları: {len(transformer_blocks)}")
    for block_name, block_positions in transformer_blocks.items():
        print(f"  {block_name}: {', '.join(block_positions)}")
    
    # AdapterManager oluştur
    manager = AdapterManager(model)
    
    # Adapter yapılandırmasını oluştur
    adapter_config = AdapterConfig(
        adapter_type=args.adapter_type,
        bottleneck_dim=args.bottleneck_dim,
        alpha=args.alpha if args.adapter_type == "parallel" else 1.0
    )
    
    print("\nAdapter'lar ekleniyor...")
    adapter_ids = {}
    
    # Her blok için adapter ekle
    for i, (block_name, block_positions) in enumerate(transformer_blocks.items()):
        # Modülü bul
        module = None
        for name, m in model.named_modules():
            if name == block_name:
                module = m
                break
        
        if module is None:
            continue
        
        # Belirtilen pozisyonlara adapter ekle
        for position in positions:
            if position in block_positions:
                adapter_name = f"adapter_{i}_{position}"
                adapter_id = manager.register_adapter(
                    module,
                    adapter_name,
                    position,
                    adapter_config
                )
                
                if adapter_id:
                    adapter_ids[f"{block_name}.{position}"] = adapter_id
                    print(f"  {adapter_name} eklendi (ID: {adapter_id})")
    
    # Adapter parametreleri hakkında bilgi
    adapter_params = get_adapter_parameter_count(model)
    print(f"\nAdapter parametre sayısı: {adapter_params['total']:,}")
    print(f"Model parametre yüzdesi: {adapter_params['percentage']:.2f}%")
    
    print("\nAdapter pozisyonlarına göre parametre dağılımı:")
    for pos, count in adapter_params.items():
        if pos not in ["total", "percentage"]:
            print(f"  {pos}: {count:,} parametre")
    
    # Adapter'lar dışındaki tüm model parametrelerini dondur
    print("\nAdapter'lar dışındaki parametreler donduruluyor...")
    freeze_model_except_adapters(model)
    
    # Eğitilebilir parametre sayısını kontrol et
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")
    
    # Küçük bir örnek veri oluştur
    print("\nÖrnek ileri geçiş yapılıyor...")
    batch_size = 4
    seq_len = 16
    x = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Modeli değerlendirme moduna al ve ileri geçiş yap
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Çıktı şekli: {output.shape}")
    
    # Adapter'ları kaydet
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nAdapter'lar {save_dir} dizinine kaydediliyor...")
    manager.save_adapters(save_dir)
    print(f"Kaydedilen adapter sayısı: {len(adapter_ids)}")
    
    # Adapter'ları kaldır
    print("\nAdapter'lar kaldırılıyor...")
    for adapter_id in adapter_ids.values():
        manager.remove_adapter(adapter_id)
    
    # Adapter'ları yükle
    print("\nAdapter'lar geri yükleniyor...")
    loaded_adapters = manager.load_adapters(save_dir)
    print(f"Yüklenen adapter sayısı: {len(loaded_adapters)}")
    
    # Yüklenen adapter'lar ile ileri geçiş yap
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Adapter'lar ile çıktı şekli: {output.shape}")
    
    # Adapter'ları etkinleştir/devre dışı bırak
    print("\nAdapter etkinleştirme/devre dışı bırakma testi:")
    for adapter_id in loaded_adapters:
        # Önce devre dışı bırak
        manager.deactivate_adapter(adapter_id)
        print(f"  {adapter_id} devre dışı bırakıldı")
        
        # Sonra etkinleştir
        manager.activate_adapter(adapter_id)
        print(f"  {adapter_id} etkinleştirildi")
    
    # Tüm aktif adapter'ları listele
    active_adapters = manager.list_adapters(active_only=True)
    print(f"\nAktif adapter sayısı: {len(active_adapters)}")
    
    # Adapter özeti
    summary = manager.summary()
    print("\nAdapter yöneticisi özeti:")
    print(f"  Toplam adapter'lar: {summary['total_adapters']}")
    print(f"  Aktif adapter'lar: {summary['active_adapters']}")
    print(f"  Toplam adapter parametreleri: {summary['total_adapter_parameters']:,}")
    print(f"  Model parametreleri: {summary['model_parameters']:,}")
    print(f"  Adapter yüzdesi: {summary['adapter_percentage']:.2f}%")
    
    print("\nÖrnek tamamlandı!")


if __name__ == "__main__":
    main() 