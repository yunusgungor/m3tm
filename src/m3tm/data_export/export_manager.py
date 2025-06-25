"""
Dışa Aktarma Yöneticisi

Bu modül, veri dışa aktarma işlemlerini yöneten merkezi bir sınıf sağlar.
"""

import os
import datetime
from typing import List, Dict, Any, Optional, Union

from .export_formats import (
    BaseExporter,
    JSONExporter,
    CSVExporter,
    TextExporter,
    XMLExporter,
    ProtocolBuffersExporter,
    AvroExporter,
    ParquetExporter,
    StreamingJSONExporter
)

# Enterprise exports (with graceful fallback)
try:
    from .enterprise_exporters import (
        ProtobufExporter,
        XMLExporter,
        StreamingExporter,
        EnterpriseIntegrationManager,
        ExportConfiguration
    )
    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False
    ExportConfiguration = None


class ExportManager:
    """
    Dışa aktarma yöneticisi sınıfı.
    
    Bu sınıf, farklı formatlarda veri dışa aktarma işlemlerini yönetir
    ve uygun dışa aktarıcıları seçer. Enterprise formatları destekler.
    
    Context7 Protocol Buffers ve enterprise integration best practices uygulanmıştır.
    """
    
    def __init__(self, enterprise_config: Optional[Any] = None):
        """
        ExportManager sınıfını başlatır.
        
        Args:
            enterprise_config: Enterprise export konfigürasyonu
        """
        # Basic format -> Exporter eşleştirmeleri
        self._exporters = {
            'json': JSONExporter(),
            'csv': CSVExporter(),
            'txt': TextExporter()
        }
        
        # Enterprise formatları ekle (eğer mevcut ise)
        if ENTERPRISE_AVAILABLE:
            self._exporters.update({
                'pb': ProtobufExporter(enterprise_config),
                'protobuf': ProtobufExporter(enterprise_config),
                'pbtxt': ProtobufExporter(enterprise_config),
                'xml': XMLExporter(enterprise_config)
            })
            
            # Enterprise integration manager
            self.enterprise_manager = EnterpriseIntegrationManager(enterprise_config)
            self.streaming_exporter = StreamingExporter(enterprise_config)
        else:
            self.enterprise_manager = None
            self.streaming_exporter = None
        
        # Desteklenen formatlar
        self.supported_formats = list(self._exporters.keys())
        self.enterprise_config = enterprise_config
    
    def export(self, data: List[Dict[str, Any]], file_path: str, 
              format: Optional[str] = None,
              start_date: Optional[datetime.datetime] = None,
              end_date: Optional[datetime.datetime] = None,
              content_type: Optional[str] = None,
              limit: Optional[int] = None) -> None:
        """
        Veriyi belirtilen dosyaya dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
            format: Dışa aktarma formatı (belirtilmezse dosya uzantısından algılanır)
            start_date: Başlangıç tarihi (filtreleme için)
            end_date: Bitiş tarihi (filtreleme için)
            content_type: İçerik türü filtresi (örn. 'text', 'image')
            limit: Maksimum öğe sayısı
        
        Raises:
            ValueError: Geçersiz format veya dosya uzantısı
        """
        # Format algılama
        if format is None:
            format = self._detect_format(file_path)
        
        # Veriyi filtrele
        filtered_data = self._filter_data(data, start_date, end_date, content_type, limit)
        
        # Uygun dışa aktarıcıyı seç ve kullan
        exporter = self._get_exporter(format)
        exporter.export(filtered_data, file_path)
    
    def _detect_format(self, file_path: str) -> str:
        """
        Dosya yolundaki uzantıya göre formatı algılar.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            str: Algılanan format
            
        Raises:
            ValueError: Bilinmeyen format
        """
        extension = os.path.splitext(file_path)[1].lower().lstrip('.')
        
        if extension in self.supported_formats:
            return extension
        
        # Bilinmeyen format
        raise ValueError(f"Unknown export format: {extension}. Supported formats: {', '.join(self.supported_formats)}")
    
    def _get_exporter(self, format: str) -> BaseExporter:
        """
        Belirtilen format için dışa aktarıcı döndürür.
        
        Args:
            format: Dışa aktarma formatı
            
        Returns:
            BaseExporter: Dışa aktarıcı
        
        Raises:
            ValueError: Desteklenmeyen format
        """
        format = format.lower()
        
        if format not in self._exporters:
            raise ValueError(f"Unsupported export format: {format}. Supported formats: {', '.join(self.supported_formats)}")
        
        return self._exporters[format]
    
    def _filter_data(self, data: List[Dict[str, Any]],
                    start_date: Optional[datetime.datetime] = None,
                    end_date: Optional[datetime.datetime] = None,
                    content_type: Optional[str] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Veriyi belirtilen kriterlere göre filtreler.
        
        Args:
            data: Filtrelenecek veri
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            content_type: İçerik türü filtresi (örn. 'text', 'image')
            limit: Maksimum öğe sayısı
            
        Returns:
            List[Dict[str, Any]]: Filtrelenmiş veri
        """
        filtered_data = data
        
        # Tarih filtresi
        if start_date or end_date:
            filtered_data = self._filter_by_date(filtered_data, start_date, end_date)
        
        # İçerik türü filtresi
        if content_type:
            filtered_data = self._filter_by_type(filtered_data, content_type)
        
        # Limit uygula
        if limit is not None and limit > 0:
            filtered_data = filtered_data[:limit]
        
        return filtered_data
    
    def _filter_by_date(self, data: List[Dict[str, Any]],
                       start_date: Optional[datetime.datetime] = None,
                       end_date: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """
        Veriyi tarih aralığına göre filtreler.
        
        Args:
            data: Filtrelenecek veri
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            
        Returns:
            List[Dict[str, Any]]: Filtrelenmiş veri
        """
        result = []
        
        # Dönüştür: naive datetimes (timezone bilgisi olmayan)
        naive_start_date = start_date.replace(tzinfo=None) if start_date else None
        naive_end_date = end_date.replace(tzinfo=None) if end_date else None
        
        for item in data:
            # Tarih değeri bulma yöntemi veri yapısına bağlı olarak değişebilir
            date_str = None
            
            # 'metadata' içinde 'created' alanını kontrol et
            if 'metadata' in item and isinstance(item['metadata'], dict):
                date_str = item['metadata'].get('created')
            
            # Doğrudan 'created' alanını kontrol et
            if not date_str and 'created' in item:
                date_str = item['created']
            
            # Tarih yoksa veya tarih dönüştürülemiyorsa atla
            if not date_str:
                continue
            
            try:
                # ISO format tarih parsing (alternatif formatlar için daha esnek bir yaklaşım gerekebilir)
                # Remove Z and replace with +00:00 for proper parsing
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                
                # Farklı ISO formatları için
                if 'T' in date_str:
                    item_date = datetime.datetime.fromisoformat(date_str)
                else:
                    # Sadece tarih varsa (YYYY-MM-DD)
                    item_date = datetime.datetime.fromisoformat(date_str + 'T00:00:00')
                
                # Timezone bilgisini kaldır - naive datetime haline getir
                naive_item_date = item_date.replace(tzinfo=None)
                
                # Tarih aralığı kontrolü
                if naive_start_date and naive_item_date < naive_start_date:
                    continue
                if naive_end_date and naive_item_date > naive_end_date:
                    continue
                
                # Tarih aralığına uygunsa sonuca ekle
                result.append(item)
            except (ValueError, TypeError) as e:
                # Tarih dönüştürülemiyorsa logla ve atla
                print(f"Warning: Could not parse date '{date_str}': {e}")
                continue
        
        return result
    
    def _filter_by_type(self, data: List[Dict[str, Any]], content_type: str) -> List[Dict[str, Any]]:
        """
        Veriyi içerik türüne göre filtreler.
        
        Args:
            data: Filtrelenecek veri
            content_type: İçerik türü
            
        Returns:
            List[Dict[str, Any]]: Filtrelenmiş veri
        """
        result = []
        
        for item in data:
            # İçerik türü kontrolü
            item_type = None
            
            # 'metadata' içinde 'type' alanını kontrol et
            if 'metadata' in item and isinstance(item['metadata'], dict):
                item_type = item['metadata'].get('type')
            
            # Doğrudan 'type' alanını kontrol et
            if not item_type and 'type' in item:
                item_type = item['type']
            
            # Türü belirtilen türle eşleşiyorsa sonuca ekle
            if item_type and item_type.lower() == content_type.lower():
                result.append(item)
        
        return result
    
    # Enterprise Integration Methods
    
    def register_enterprise_integration(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Enterprise integration kaydeder.
        
        Args:
            name: Integration adı
            config: Integration konfigürasyonu
            
        Returns:
            bool: Başarı durumu
        """
        if not ENTERPRISE_AVAILABLE or not self.enterprise_manager:
            return False
        
        self.enterprise_manager.register_integration(name, config)
        return True
    
    async def export_to_enterprise_system(self, data: List[Dict[str, Any]], 
                                        integration_name: str) -> bool:
        """
        Enterprise sisteme veri export eder.
        
        Args:
            data: Export edilecek veri
            integration_name: Hedef integration adı
            
        Returns:
            bool: Başarı durumu
        """
        if not ENTERPRISE_AVAILABLE or not self.enterprise_manager:
            return False
        
        return await self.enterprise_manager.export_to_integration(integration_name, data)
    
    async def start_real_time_streaming(self, host: str = "localhost", port: int = 8765) -> bool:
        """
        Real-time data streaming başlatır.
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
            
        Returns:
            bool: Başarı durumu
        """
        if not ENTERPRISE_AVAILABLE or not self.streaming_exporter:
            return False
        
        try:
            await self.streaming_exporter.start_websocket_stream(host, port)
            return True
        except Exception:
            return False
    
    async def stream_data_real_time(self, data: Dict[str, Any]) -> bool:
        """
        Veriyi real-time olarak stream eder.
        
        Args:
            data: Stream edilecek veri
            
        Returns:
            bool: Başarı durumu
        """
        if not ENTERPRISE_AVAILABLE or not self.streaming_exporter:
            return False
        
        try:
            await self.streaming_exporter.stream_data(data)
            return True
        except Exception:
            return False
    
    def export_protobuf(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """
        Protocol Buffers formatında export eder.
        
        Args:
            data: Export edilecek veri
            file_path: Hedef dosya yolu
            
        Returns:
            bool: Başarı durumu
        """
        if not ENTERPRISE_AVAILABLE:
            return False
        
        try:
            exporter = ProtobufExporter(self.enterprise_config)
            exporter.export(data, file_path)
            return True
        except Exception:
            return False
    
    def export_xml(self, data: List[Dict[str, Any]], file_path: str) -> bool:
        """
        XML formatında export eder.
        
        Args:
            data: Export edilecek veri
            file_path: Hedef dosya yolu
            
        Returns:
            bool: Başarı durumu
        """
        if not ENTERPRISE_AVAILABLE:
            return False
        
        try:
            exporter = XMLExporter(self.enterprise_config)
            exporter.export(data, file_path)
            return True
        except Exception:
            return False
    
    def get_enterprise_capabilities(self) -> Dict[str, bool]:
        """
        Mevcut enterprise yeteneklerini döndürür.
        
        Returns:
            Dict[str, bool]: Yetenek durumları
        """
        return {
            "enterprise_available": ENTERPRISE_AVAILABLE,
            "protobuf_export": ENTERPRISE_AVAILABLE,
            "xml_export": ENTERPRISE_AVAILABLE,
            "real_time_streaming": ENTERPRISE_AVAILABLE,
            "enterprise_integration": ENTERPRISE_AVAILABLE,
            "supported_formats": self.supported_formats
        }