"""
M³TM Yapılandırma Komut Satırı Arayüzü

Bu modül, yapılandırma dosyalarını yönetmek için bir komut satırı arayüzü sağlar.
Yapılandırma dosyalarını doğrulama, özetleme, dönüştürme ve karşılaştırma gibi işlemleri destekler.

Kullanım:
    python -m m3tm.config validate <file_path>
    python -m m3tm.config summary <file_path>
    python -m m3tm.config convert <input_file> <output_file>
    python -m m3tm.config compare <file1> <file2>
    python -m m3tm.config create <config_type> <output_file>
"""

import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from .model_config import M3TMModelConfig
from .default_configs import get_predefined_config, PREDEFINED_CONFIGS
from .utils import ConfigValidator, print_config_summary, compare_configs
from .config_base import ConfigValidationError


def create_parser() -> argparse.ArgumentParser:
    """Komut satırı argüman ayrıştırıcısını oluşturur."""
    parser = argparse.ArgumentParser(
        description="M³TM Yapılandırma Yönetim Aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Komut")
    
    # Validate komutu
    validate_parser = subparsers.add_parser("validate", help="Yapılandırma dosyasını doğrula")
    validate_parser.add_argument("file_path", help="Doğrulanacak yapılandırma dosyası yolu")
    
    # Summary komutu
    summary_parser = subparsers.add_parser("summary", help="Yapılandırma özeti göster")
    summary_parser.add_argument("file_path", help="Özetlenecek yapılandırma dosyası yolu")
    
    # Convert komutu
    convert_parser = subparsers.add_parser("convert", help="Yapılandırma dosyasını başka bir formata dönüştür")
    convert_parser.add_argument("input_file", help="Dönüştürülecek yapılandırma dosyası yolu")
    convert_parser.add_argument("output_file", help="Çıktı dosyası yolu")
    
    # Compare komutu
    compare_parser = subparsers.add_parser("compare", help="İki yapılandırma dosyasını karşılaştır")
    compare_parser.add_argument("file1", help="İlk yapılandırma dosyası yolu")
    compare_parser.add_argument("file2", help="İkinci yapılandırma dosyası yolu")
    
    # Create komutu
    create_parser = subparsers.add_parser("create", help="Yeni bir yapılandırma dosyası oluştur")
    create_parser.add_argument(
        "config_type", 
        choices=list(PREDEFINED_CONFIGS.keys()),
        help="Yapılandırma tipi"
    )
    create_parser.add_argument("output_file", help="Çıktı dosyası yolu")
    
    # List komutu
    list_parser = subparsers.add_parser("list", help="Kullanılabilir yapılandırma tiplerini listele")
    
    return parser


def validate_config(file_path: str) -> bool:
    """
    Yapılandırma dosyasını doğrular.
    
    Args:
        file_path: Doğrulanacak dosya yolu
        
    Returns:
        bool: Doğrulama başarılıysa True, değilse False
    """
    path = Path(file_path)
    
    # Dosya varlığını kontrol et
    if not ConfigValidator.validate_file_exists(path):
        print(f"Hata: Dosya bulunamadı: {file_path}")
        return False
    
    try:
        # Dosya içeriğini oku
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Format doğrulaması
        if path.suffix.lower() == ".json":
            if not ConfigValidator.validate_json_format(content):
                print(f"Hata: Geçersiz JSON formatı: {file_path}")
                return False
                
            # JSON içeriğini yükle
            config_data = json.loads(content)
            
        elif path.suffix.lower() in [".yaml", ".yml"]:
            if not ConfigValidator.validate_yaml_format(content):
                print(f"Hata: Geçersiz YAML formatı: {file_path}")
                return False
                
            # YAML içeriğini yükle
            config_data = yaml.safe_load(content)
        else:
            print(f"Hata: Desteklenmeyen dosya uzantısı: {path.suffix}")
            return False
        
        # __config_type__ alanını kontrol et
        if "__config_type__" not in config_data:
            print(f"Hata: Yapılandırma türü belirtilmemiş (__config_type__ alanı eksik)")
            return False
        
        # M3TMModelConfig'den bir örnek oluşturmayı dene
        try:
            config = M3TMModelConfig.from_dict(config_data)
            # Config'in kendi doğrulama yöntemini çağır
            config.validate()
            
            print(f"Başarılı: Yapılandırma doğrulaması başarılı. Tür: {config_data['__config_type__']}")
            return True
            
        except ConfigValidationError as e:
            print(f"Doğrulama Hatası: {str(e)}")
            return False
        except Exception as e:
            print(f"Hata: Yapılandırma oluşturulurken hata: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Hata: Dosya işlenirken hata: {str(e)}")
        return False


def show_summary(file_path: str) -> None:
    """
    Yapılandırma dosyasının özetini gösterir.
    
    Args:
        file_path: Özetlenecek dosya yolu
    """
    path = Path(file_path)
    
    # Dosya varlığını kontrol et
    if not ConfigValidator.validate_file_exists(path):
        print(f"Hata: Dosya bulunamadı: {file_path}")
        return
    
    try:
        # Yapılandırmayı yükle
        config = M3TMModelConfig.load(path)
        
        # Özeti yazdır
        summary = print_config_summary(config)
        print("\n=== Yapılandırma Özeti ===")
        print(summary)
        print("\n=== Detaylı Bilgiler ===")
        
        # Bazı önemli parametreleri göster
        if hasattr(config, 'text_config'):
            print(f"Metin Gömme Boyutu: {config.text_config.embed_dim}")
            print(f"Sözlük Boyutu: {config.text_config.vocab_size}")
        
        if hasattr(config, 'image_config'):
            print(f"Görüntü Gömme Boyutu: {config.image_config.embed_dim}")
            print(f"Yama Boyutu: {config.image_config.patch_size}")
        
        if hasattr(config, 'transformer_config'):
            print(f"Transformer Blok Sayısı: {config.num_transformer_blocks}")
            print(f"Dikkat Başı Sayısı: {config.transformer_config.num_heads}")
        
        if hasattr(config, 'fusion_config'):
            print(f"Füzyon Tipi: {config.fusion_config.fusion_type}")
        
        print(f"Aktif Modaliteler: " + 
              (f"Metin {'✓' if config.use_text_modality else '✗'}, " +
               f"Görüntü {'✓' if config.use_image_modality else '✗'}"))
        
    except Exception as e:
        print(f"Hata: Yapılandırma özetlenirken hata: {str(e)}")


def convert_config(input_file: str, output_file: str) -> None:
    """
    Yapılandırma dosyasını başka bir formata dönüştürür.
    
    Args:
        input_file: Dönüştürülecek dosya yolu
        output_file: Çıktı dosyası yolu
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Dosya varlığını kontrol et
    if not ConfigValidator.validate_file_exists(input_path):
        print(f"Hata: Dosya bulunamadı: {input_file}")
        return
    
    # Çıktı formatını belirle
    if output_path.suffix.lower() not in ['.json', '.yaml', '.yml']:
        print(f"Hata: Desteklenmeyen çıktı formatı: {output_path.suffix}")
        print("Desteklenen formatlar: .json, .yaml, .yml")
        return
    
    try:
        # Yapılandırmayı yükle
        config = M3TMModelConfig.load(input_path)
        
        # Belirtilen formatta kaydet
        if output_path.suffix.lower() == '.json':
            config.save(output_path, format="json")
        else:
            config.save(output_path, format="yaml")
        
        print(f"Başarılı: Yapılandırma dönüştürüldü ve kaydedildi: {output_file}")
        
    except Exception as e:
        print(f"Hata: Yapılandırma dönüştürülürken hata: {str(e)}")


def compare_configuration_files(file1: str, file2: str) -> None:
    """
    İki yapılandırma dosyasını karşılaştırır ve farklılıkları gösterir.
    
    Args:
        file1: İlk yapılandırma dosyası yolu
        file2: İkinci yapılandırma dosyası yolu
    """
    path1 = Path(file1)
    path2 = Path(file2)
    
    # Dosya varlığını kontrol et
    if not ConfigValidator.validate_file_exists(path1):
        print(f"Hata: Dosya bulunamadı: {file1}")
        return
    
    if not ConfigValidator.validate_file_exists(path2):
        print(f"Hata: Dosya bulunamadı: {file2}")
        return
    
    try:
        # Yapılandırmaları yükle
        config1 = M3TMModelConfig.load(path1)
        config2 = M3TMModelConfig.load(path2)
        
        # Farklılıkları bul
        differences = compare_configs(config1, config2)
        
        if not differences:
            print("Yapılandırmalar arasında fark bulunmadı.")
            return
        
        # Farklılıkları göster
        print(f"\n=== Yapılandırma Farklılıkları ===")
        print(f"Dosya 1: {file1}")
        print(f"Dosya 2: {file2}")
        print(f"\nFarklı Parametreler:")
        
        for param, values in differences.items():
            # Alt yapılandırmalar için özel durum
            if values['old'] == '...' and values['new'] == '...':
                print(f"  {param}: [Yapılandırma nesnesi farklı]")
            else:
                print(f"  {param}:")
                print(f"    Eski: {values['old']}")
                print(f"    Yeni: {values['new']}")
        
    except Exception as e:
        print(f"Hata: Yapılandırmalar karşılaştırılırken hata: {str(e)}")


def create_config(config_type: str, output_file: str) -> None:
    """
    Belirtilen tipte bir yapılandırma dosyası oluşturur.
    
    Args:
        config_type: Yapılandırma tipi ('tiny', 'small', 'base', vb.)
        output_file: Çıktı dosyası yolu
    """
    output_path = Path(output_file)
    
    # Çıktı formatını belirle
    if output_path.suffix.lower() not in ['.json', '.yaml', '.yml']:
        print(f"Hata: Desteklenmeyen çıktı formatı: {output_path.suffix}")
        print("Desteklenen formatlar: .json, .yaml, .yml")
        return
    
    try:
        # Yapılandırma oluştur
        config = get_predefined_config(config_type)
        
        # Belirtilen formatta kaydet
        if output_path.suffix.lower() == '.json':
            config.save(output_path, format="json")
        else:
            config.save(output_path, format="yaml")
        
        print(f"Başarılı: '{config_type}' tipinde yapılandırma oluşturuldu ve kaydedildi: {output_file}")
        
    except Exception as e:
        print(f"Hata: Yapılandırma oluşturulurken hata: {str(e)}")


def list_config_types() -> None:
    """Kullanılabilir yapılandırma tiplerini listeler."""
    print("\n=== Kullanılabilir Yapılandırma Tipleri ===")
    
    for config_name, config_func in PREDEFINED_CONFIGS.items():
        # Fonksiyonun docstring'inden açıklama çıkar
        doc = config_func.__doc__
        description = ""
        
        if doc:
            # İlk paragrafı al
            first_paragraph = doc.strip().split('\n\n')[0]
            # Docstring formatını temizle
            description = ' '.join(line.strip() for line in first_paragraph.splitlines())
        
        print(f"- {config_name}: {description}")


def main() -> None:
    """Ana program fonksiyonu."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "validate":
        success = validate_config(args.file_path)
        if not success:
            sys.exit(1)
            
    elif args.command == "summary":
        show_summary(args.file_path)
        
    elif args.command == "convert":
        convert_config(args.input_file, args.output_file)
        
    elif args.command == "compare":
        compare_configuration_files(args.file1, args.file2)
        
    elif args.command == "create":
        create_config(args.config_type, args.output_file)
        
    elif args.command == "list":
        list_config_types()
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 