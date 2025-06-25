"""
Dışa Aktarma Formatları

Bu modül, farklı dışa aktarma formatları için sınıflar içerir.
Enterprise seviye formatlar: Protocol Buffers, XML, Avro desteği dahil.
"""

import os
import json
import csv
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging

# Enterprise integration imports
try:
    import lxml.etree as etree
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

try:
    import google.protobuf.message as pb_message
    from google.protobuf.json_format import MessageToJson, MessageToDict
    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False

try:
    import avro.schema
    import avro.io
    import avro.datafile
    HAS_AVRO = True
except ImportError:
    HAS_AVRO = False

logger = logging.getLogger(__name__)

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


class XMLExporter(BaseExporter):
    """
    XML formatında dışa aktarıcı.
    Enterprise sistemler için XML export desteği.
    """
    
    def __init__(self, root_name: str = "data", item_name: str = "item"):
        """
        XMLExporter sınıfını başlatır.
        
        Args:
            root_name: Kök XML elementi adı
            item_name: Her veri öğesi için XML elementi adı
        """
        self.root_name = root_name
        self.item_name = item_name
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi XML formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        # Dizini oluştur
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # XML root element oluştur
        root = ET.Element(self.root_name)
        root.set("exported_at", datetime.now().isoformat())
        root.set("count", str(len(data)))
        
        for item in data:
            item_element = ET.SubElement(root, self.item_name)
            self._dict_to_xml(item, item_element)
        
        # XML dosyasını yaz
        tree = ET.ElementTree(root)
        if HAS_LXML:
            # lxml varsa pretty print kullan
            ET.indent(tree, space="  ", level=0)
        
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
    
    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element) -> None:
        """
        Dictionary'yi XML elementine dönüştürür.
        
        Args:
            data: Dönüştürülecek dictionary
            parent: Parent XML element
        """
        for key, value in data.items():
            # XML element adını temizle
            clean_key = self._clean_xml_name(key)
            
            if isinstance(value, dict):
                element = ET.SubElement(parent, clean_key)
                self._dict_to_xml(value, element)
            elif isinstance(value, list):
                for item in value:
                    element = ET.SubElement(parent, clean_key)
                    if isinstance(item, dict):
                        self._dict_to_xml(item, element)
                    else:
                        element.text = str(item)
            else:
                element = ET.SubElement(parent, clean_key)
                element.text = str(value) if value is not None else ""
    
    def _clean_xml_name(self, name: str) -> str:
        """
        XML element adını temizler.
        
        Args:
            name: Temizlenecek isim
            
        Returns:
            Temizlenmiş isim
        """
        # XML element adı kurallarına uygun hale getir
        clean_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
        if clean_name and clean_name[0].isdigit():
            clean_name = 'item_' + clean_name
        return clean_name or 'item'


class ProtocolBuffersExporter(BaseExporter):
    """
    Protocol Buffers formatında dışa aktarıcı.
    Enterprise sistemler için yüksek performanslı binary serialization.
    """
    
    def __init__(self, message_class=None):
        """
        ProtocolBuffersExporter sınıfını başlatır.
        
        Args:
            message_class: Protocol Buffers message sınıfı
        """
        if not HAS_PROTOBUF:
            raise ImportError("Protocol Buffers support requires 'protobuf' package")
        
        self.message_class = message_class
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi Protocol Buffers formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        # Dizini oluştur
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Eğer message class belirtilmemişse generic format kullan
        if self.message_class:
            self._export_with_schema(data, file_path)
        else:
            self._export_generic(data, file_path)
    
    def _export_with_schema(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Belirli bir Protocol Buffers schema kullanarak export eder.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        # Bu metodun implementasyonu specific schema'ya bağlı
        # M3TM proto dosyaları compile edildikten sonra implement edilecek
        raise NotImplementedError("Schema-based export will be implemented after proto compilation")
    
    def _export_generic(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Generic format kullanarak Protocol Buffers export eder.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        # Generic olarak JSON benzeri yapıyı binary formatta sakla
        # Bu geçici bir çözüm, gerçek proto schema kullanılacak
        import pickle
        
        with open(file_path, 'wb') as f:
            # Metadata ekle
            export_data = {
                'metadata': {
                    'format': 'protobuf_generic',
                    'exported_at': datetime.now().isoformat(),
                    'count': len(data)
                },
                'data': data
            }
            pickle.dump(export_data, f)
        
        logger.warning("Using generic binary format. Implement schema-based export for production.")


class AvroExporter(BaseExporter):
    """
    Apache Avro formatında dışa aktarıcı.
    Schema-based big data serialization için kullanılır.
    """
    
    def __init__(self, schema_file: Optional[str] = None):
        """
        AvroExporter sınıfını başlatır.
        
        Args:
            schema_file: Avro schema dosyası yolu
        """
        if not HAS_AVRO:
            raise ImportError("Avro support requires 'avro-python3' package")
        
        self.schema_file = schema_file
        self.schema = None
        
        if schema_file and os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                self.schema = avro.schema.parse(f.read())
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi Avro formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        if not HAS_AVRO:
            raise RuntimeError("Avro not available")
        
        # Dizini oluştur
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        if self.schema:
            self._export_with_schema(data, file_path)
        else:
            self._export_schemaless(data, file_path)
    
    def _export_with_schema(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Belirli bir Avro schema kullanarak export eder.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        with open(file_path, 'wb') as out_file:
            writer = avro.datafile.DataFileWriter(
                out_file, 
                avro.io.DatumWriter(), 
                self.schema
            )
            
            for item in data:
                writer.append(item)
            
            writer.close()
    
    def _export_schemaless(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Schema olmadan generic Avro export eder.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        # Generic schema oluştur
        if not data:
            return
        
        # İlk öğeden otomatik schema çıkar (basit yaklaşım)
        sample_item = data[0]
        schema_dict = {
            "type": "record",
            "name": "GenericRecord",
            "fields": []
        }
        
        for key, value in sample_item.items():
            field_type = self._infer_avro_type(value)
            schema_dict["fields"].append({
                "name": key,
                "type": ["null", field_type],
                "default": None
            })
        
        schema = avro.schema.parse(json.dumps(schema_dict))
        
        with open(file_path, 'wb') as out_file:
            writer = avro.datafile.DataFileWriter(
                out_file,
                avro.io.DatumWriter(),
                schema
            )
            
            for item in data:
                writer.append(item)
            
            writer.close()
    
    def _infer_avro_type(self, value: Any) -> str:
        """
        Python değerinden Avro tipini çıkarır.
        
        Args:
            value: Python değeri
            
        Returns:
            Avro tipi
        """
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "long"
        elif isinstance(value, float):
            return "double"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return {"type": "array", "items": "string"}
        elif isinstance(value, dict):
            return {"type": "map", "values": "string"}
        else:
            return "string"


class ParquetExporter(BaseExporter):
    """
    Apache Parquet formatında dışa aktarıcı.
    Columnar storage format for analytics.
    """
    
    def __init__(self, compression: str = "snappy"):
        """
        ParquetExporter sınıfını başlatır.
        
        Args:
            compression: Sıkıştırma algoritması (snappy, gzip, lz4)
        """
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
            self.pd = pd
            self.pa = pa
            self.pq = pq
            self.has_parquet = True
        except ImportError:
            self.has_parquet = False
        
        self.compression = compression
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi Parquet formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        if not self.has_parquet:
            raise ImportError("Parquet support requires 'pandas' and 'pyarrow' packages")
        
        if not data:
            return
        
        # Dizini oluştur
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # DataFrame'e dönüştür
        df = self.pd.DataFrame(data)
        
        # Nested yapıları flatten et
        df = self._flatten_dataframe(df)
        
        # Parquet olarak kaydet
        df.to_parquet(
            file_path,
            compression=self.compression,
            index=False,
            engine='pyarrow'
        )
    
    def _flatten_dataframe(self, df) -> 'pd.DataFrame':
        """
        Nested dictionary'leri flatten eder.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Flattened DataFrame
        """
        # Basit flatten işlemi
        # Production'da daha sofistike bir yaklaşım gerekebilir
        result_data = []
        
        for _, row in df.iterrows():
            flattened_row = {}
            for col, value in row.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flattened_row[f"{col}_{sub_key}"] = sub_value
                else:
                    flattened_row[col] = value
            result_data.append(flattened_row)
        
        return self.pd.DataFrame(result_data)


class StreamingExporter(BaseExporter):
    """
    Streaming export için base sınıf.
    Büyük veri setleri için memory-efficient export.
    """
    
    def __init__(self, batch_size: int = 1000):
        """
        StreamingExporter sınıfını başlatır.
        
        Args:
            batch_size: Batch boyutu
        """
        self.batch_size = batch_size
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Base implementation - alt sınıflar override etmeli.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        # Dizini oluştur
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Batch'ler halinde işle
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        with open(file_path, 'w', encoding='utf-8') as f:
            self._write_header(f)
            
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                self._write_batch(f, batch, batch_num, total_batches)
            
            self._write_footer(f)
    
    def _write_header(self, file_handle) -> None:
        """
        Dosya başlığını yazar.
        
        Args:
            file_handle: Dosya handle
        """
        pass
    
    def _write_batch(self, file_handle, batch: List[Dict[str, Any]], 
                    batch_num: int, total_batches: int) -> None:
        """
        Bir batch'i yazar.
        
        Args:
            file_handle: Dosya handle
            batch: Veri batch'i
            batch_num: Batch numarası
            total_batches: Toplam batch sayısı
        """
        raise NotImplementedError("Subclasses must implement _write_batch")
    
    def _write_footer(self, file_handle) -> None:
        """
        Dosya sonunu yazar.
        
        Args:
            file_handle: Dosya handle
        """
        pass


class StreamingJSONExporter(StreamingExporter):
    """
    Streaming JSON exporter.
    Büyük JSON dosyaları için memory-efficient export.
    """
    
    def _write_header(self, file_handle) -> None:
        """JSON array başlangıcını yazar."""
        file_handle.write('[\n')
    
    def _write_batch(self, file_handle, batch: List[Dict[str, Any]], 
                    batch_num: int, total_batches: int) -> None:
        """JSON batch yazar."""
        for i, item in enumerate(batch):
            json.dump(item, file_handle, ensure_ascii=False, indent=2)
            
            # Son öğe değilse virgül ekle
            if not (batch_num == total_batches and i == len(batch) - 1):
                file_handle.write(',')
            file_handle.write('\n')
    
    def _write_footer(self, file_handle) -> None:
        """JSON array sonunu yazar."""
        file_handle.write(']\n')