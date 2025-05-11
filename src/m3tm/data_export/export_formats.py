"""
Dışa Aktarma Formatları

Bu modül, farklı dışa aktarma formatları için sınıflar içerir.
"""

import os
import json
import csv
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseExporter(ABC):
    """
    Temel dışa aktarma sınıfı.
    
    Tüm format dışa aktarıcıları için temel sınıf.
    """
    
    @abstractmethod
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi belirtilen dosyaya aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        pass


class JSONExporter(BaseExporter):
    """
    JSON formatında dışa aktarıcı.
    """
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi JSON formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        # Dizini oluştur (varsa atla)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # JSON olarak yaz
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class CSVExporter(BaseExporter):
    """
    CSV formatında dışa aktarıcı.
    """
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi CSV formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        if not data:
            # Boş veri, boş bir CSV dosyası oluştur
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                pass
            return
        
        # Dizini oluştur (varsa atla)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Tüm anahtarları topla - basit veri yapısı için
        # İç içe sözlükler için daha karmaşık bir işleme gerekebilir
        headers = set()
        for item in data:
            headers.update(item.keys())
        
        headers = sorted(list(headers))
        
        # CSV olarak yaz
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)


class TextExporter(BaseExporter):
    """
    Düz metin formatında dışa aktarıcı.
    """
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi düz metin formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        # Dizini oluştur (varsa atla)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Metin içeriği veya tanımlanmış bir gösterim alanı
                content = item.get('content', '')
                if not content and 'text' in item:
                    content = item['text']
                
                # Metaveri
                metadata = item.get('metadata', {})
                id_str = metadata.get('id', item.get('id', ''))
                
                # Gösterilebilir bir çıktı oluştur
                if id_str:
                    f.write(f"[{id_str}]\n")
                
                f.write(f"{content}\n")
                
                # Öğeler arasında boşluk bırak
                f.write("\n---\n\n") 