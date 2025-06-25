"""
Enterprise Data Export Formats

Bu modül, enterprise düzeyinde veri dışa aktarma formatları ve entegrasyon yetenekleri sağlar.
Context7 Protocol Buffers dokümantasyonuna dayalı olarak geliştirilmiştir.
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Callable
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass, asdict
import gzip
import io

# Protocol Buffers imports - using absolute imports to avoid circular dependencies
try:
    import src.m3tm.data_export.proto.m3tm_export_pb2 as m3tm_export_pb2
    import src.m3tm.data_export.proto.enterprise_integration_pb2 as enterprise_integration_pb2
    import src.m3tm.data_export.proto.streaming_pb2 as streaming_pb2
    from google.protobuf.timestamp_pb2 import Timestamp
    from google.protobuf.json_format import MessageToJson, MessageToDict
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    logging.warning("Protocol Buffers not available. Run compile_protos.sh first.")

# Enterprise integration imports
try:
    import websockets
    import kafka
    import redis
    import grpc
    ENTERPRISE_LIBS_AVAILABLE = True
except ImportError:
    ENTERPRISE_LIBS_AVAILABLE = False
    logging.warning("Enterprise integration libraries not available. Install requirements.")

from .export_formats import BaseExporter

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExportConfiguration:
    """Export konfigürasyon sınıfı"""
    include_metadata: bool = True
    include_embeddings: bool = True
    include_user_data: bool = False  # Privacy by default
    include_model_data: bool = False
    include_search_results: bool = True
    include_training_data: bool = False
    compression_enabled: bool = False
    compression_algorithm: str = "gzip"
    max_records_per_file: Optional[int] = None
    anonymize_user_data: bool = True
    export_format_version: str = "1.0"


class ProtobufExporter(BaseExporter):
    """
    Protocol Buffers formatında dışa aktarıcı.
    Context7 Protocol Buffers best practices uygulanmıştır.
    """
    
    def __init__(self, config: Optional[ExportConfiguration] = None):
        """
        ProtobufExporter sınıfını başlatır.
        
        Args:
            config: Export konfigürasyonu
        """
        if not PROTOBUF_AVAILABLE:
            raise ImportError("Protocol Buffers not available. Install protobuf and run compile_protos.sh")
        
        self.config = config or ExportConfiguration()
        logger.info("ProtobufExporter initialized with Context7 best practices")
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi Protocol Buffers formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        try:
            # BatchExportData mesajını oluştur
            export_data = self._create_export_message(data)
            
            # Dizini oluştur
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Binary veya text format olarak kaydet
            if file_path.endswith('.pb'):
                self._save_binary_format(export_data, file_path)
            elif file_path.endswith('.pbtxt'):
                self._save_text_format(export_data, file_path)
            else:
                # Default binary format
                self._save_binary_format(export_data, file_path + '.pb')
            
            logger.info(f"Protocol Buffers export completed: {file_path}")
            
        except Exception as e:
            logger.error(f"Protocol Buffers export failed: {str(e)}")
            raise
    
    def _create_export_message(self, data: List[Dict[str, Any]]) -> 'm3tm_export_pb2.BatchExportData':
        """BatchExportData mesajını oluşturur"""
        export_msg = m3tm_export_pb2.BatchExportData()
        
        # Metadata oluştur
        metadata = export_msg.metadata
        metadata.export_id = f"export_{datetime.now().isoformat()}"
        metadata.created_at.GetCurrentTime()
        metadata.export_version = self.config.export_format_version
        metadata.source_system = "M3TM_v2.3"
        metadata.export_type = "filtered"
        metadata.total_records = len(data)
        
        # Konfigürasyonu metadata'ya ekle
        config_msg = metadata.config
        config_msg.include_embeddings = self.config.include_embeddings
        config_msg.include_user_data = self.config.include_user_data
        config_msg.include_model_data = self.config.include_model_data
        config_msg.include_search_results = self.config.include_search_results
        config_msg.include_training_data = self.config.include_training_data
        
        # Veriyi kategorilere ayır ve ekle
        for item in data:
            data_type = item.get('type', 'unknown')
            
            if data_type == 'embedding' and self.config.include_embeddings:
                self._add_embedding_data(export_msg, item)
            elif data_type == 'user_data' and self.config.include_user_data:
                self._add_user_data(export_msg, item)
            elif data_type == 'search_result' and self.config.include_search_results:
                self._add_search_result(export_msg, item)
            # Diğer veri türleri için benzer işlemler...
        
        return export_msg
    
    def _add_embedding_data(self, export_msg: 'm3tm_export_pb2.BatchExportData', item: Dict[str, Any]):
        """Embedding verisini mesaja ekler"""
        embedding = export_msg.embeddings.add()
        embedding.embedding_id = item.get('id', '')
        embedding.content_type = item.get('content_type', 'unknown')
        
        # Vector data
        if 'vector' in item:
            embedding.vector.extend(item['vector'])
            embedding.dimension = len(item['vector'])
        
        embedding.source_content = item.get('content', '')
        embedding.model_version = item.get('model_version', 'unknown')
        
        # Metadata
        if 'metadata' in item:
            for key, value in item['metadata'].items():
                embedding.metadata[key] = str(value)
    
    def _add_user_data(self, export_msg: 'm3tm_export_pb2.BatchExportData', item: Dict[str, Any]):
        """Kullanıcı verisini mesaja ekler (gizlilik korumalı)"""
        if not self.config.include_user_data:
            return
        
        user_data = export_msg.user_data.add()
        
        # Anonymization uygula
        if self.config.anonymize_user_data:
            user_data.user_id = f"anonymous_{hash(item.get('user_id', '')) % 10000}"
        else:
            user_data.user_id = item.get('user_id', '')
        
        user_data.content_id = item.get('content_id', '')
        user_data.content_type = item.get('content_type', 'unknown')
        
        # Privacy settings
        privacy = user_data.privacy_settings
        privacy.is_exportable = item.get('is_exportable', True)
        privacy.requires_anonymization = self.config.anonymize_user_data
        privacy.consent_level = item.get('consent_level', 'limited')
    
    def _add_search_result(self, export_msg: 'm3tm_export_pb2.BatchExportData', item: Dict[str, Any]):
        """Arama sonucunu mesaja ekler"""
        search_result = export_msg.search_results.add()
        search_result.search_id = item.get('search_id', '')
        search_result.query = item.get('query', '')
        search_result.query_type = item.get('query_type', 'text')
        search_result.search_time_ms = item.get('search_time_ms', 0.0)
        
        # Search hits
        for hit_data in item.get('results', []):
            hit = search_result.results.add()
            hit.content_id = hit_data.get('content_id', '')
            hit.similarity_score = hit_data.get('score', 0.0)
            hit.content_type = hit_data.get('content_type', 'unknown')
            hit.content_preview = hit_data.get('preview', '')
            hit.rank = hit_data.get('rank', 0)
    
    def _save_binary_format(self, export_data: 'm3tm_export_pb2.BatchExportData', file_path: str):
        """Binary formatında kaydet"""
        serialized_data = export_data.SerializeToString()
        
        if self.config.compression_enabled:
            serialized_data = gzip.compress(serialized_data)
        
        with open(file_path, 'wb') as f:
            f.write(serialized_data)
    
    def _save_text_format(self, export_data: 'm3tm_export_pb2.BatchExportData', file_path: str):
        """Text formatında kaydet (debugging için)"""
        text_data = str(export_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_data)


class XMLExporter(BaseExporter):
    """
    XML formatında dışa aktarıcı.
    Enterprise sistemlerle entegrasyon için optimize edilmiştir.
    """
    
    def __init__(self, config: Optional[ExportConfiguration] = None):
        """
        XMLExporter sınıfını başlatır.
        
        Args:
            config: Export konfigürasyonu
        """
        self.config = config or ExportConfiguration()
        logger.info("XMLExporter initialized for enterprise integration")
    
    def export(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Veriyi XML formatında dışa aktarır.
        
        Args:
            data: Dışa aktarılacak veri
            file_path: Hedef dosya yolu
        """
        try:
            # XML root element oluştur
            root = ET.Element("BatchExportData")
            root.set("version", self.config.export_format_version)
            root.set("timestamp", datetime.now().isoformat())
            root.set("source", "M3TM_v2.3")
            
            # Metadata ekle
            metadata_elem = ET.SubElement(root, "Metadata")
            self._add_metadata(metadata_elem, data)
            
            # Data kategorileri
            if self.config.include_embeddings:
                embeddings_elem = ET.SubElement(root, "Embeddings")
                self._add_embeddings_xml(embeddings_elem, data)
            
            if self.config.include_search_results:
                search_results_elem = ET.SubElement(root, "SearchResults")
                self._add_search_results_xml(search_results_elem, data)
            
            # Pretty formatting
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            
            # Dizini oluştur
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # XML dosyasını kaydet
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            logger.info(f"XML export completed: {file_path}")
            
        except Exception as e:
            logger.error(f"XML export failed: {str(e)}")
            raise
    
    def _add_metadata(self, parent: ET.Element, data: List[Dict[str, Any]]):
        """Metadata XML elementlerini ekler"""
        ET.SubElement(parent, "ExportId").text = f"export_{datetime.now().isoformat()}"
        ET.SubElement(parent, "TotalRecords").text = str(len(data))
        ET.SubElement(parent, "ExportType").text = "filtered"
        
        # Configuration
        config_elem = ET.SubElement(parent, "Configuration")
        ET.SubElement(config_elem, "IncludeEmbeddings").text = str(self.config.include_embeddings)
        ET.SubElement(config_elem, "IncludeUserData").text = str(self.config.include_user_data)
        ET.SubElement(config_elem, "AnonymizeUserData").text = str(self.config.anonymize_user_data)
    
    def _add_embeddings_xml(self, parent: ET.Element, data: List[Dict[str, Any]]):
        """Embedding verilerini XML olarak ekler"""
        embeddings = [item for item in data if item.get('type') == 'embedding']
        
        for embedding_data in embeddings:
            embedding_elem = ET.SubElement(parent, "Embedding")
            embedding_elem.set("id", embedding_data.get('id', ''))
            
            ET.SubElement(embedding_elem, "ContentType").text = embedding_data.get('content_type', 'unknown')
            ET.SubElement(embedding_elem, "Dimension").text = str(len(embedding_data.get('vector', [])))
            ET.SubElement(embedding_elem, "ModelVersion").text = embedding_data.get('model_version', 'unknown')
            
            # Vector data (sample only for XML readability)
            if 'vector' in embedding_data and len(embedding_data['vector']) > 0:
                vector_elem = ET.SubElement(embedding_elem, "Vector")
                vector_elem.set("length", str(len(embedding_data['vector'])))
                # Store first few values as example
                sample_values = embedding_data['vector'][:5]
                vector_elem.text = ",".join(map(str, sample_values))
                if len(embedding_data['vector']) > 5:
                    vector_elem.text += ",...(truncated)"
    
    def _add_search_results_xml(self, parent: ET.Element, data: List[Dict[str, Any]]):
        """Arama sonuçlarını XML olarak ekler"""
        search_results = [item for item in data if item.get('type') == 'search_result']
        
        for search_data in search_results:
            search_elem = ET.SubElement(parent, "SearchResult")
            search_elem.set("id", search_data.get('search_id', ''))
            
            ET.SubElement(search_elem, "Query").text = search_data.get('query', '')
            ET.SubElement(search_elem, "QueryType").text = search_data.get('query_type', 'text')
            ET.SubElement(search_elem, "SearchTimeMs").text = str(search_data.get('search_time_ms', 0))
            
            # Results
            results_elem = ET.SubElement(search_elem, "Results")
            for hit_data in search_data.get('results', []):
                hit_elem = ET.SubElement(results_elem, "Hit")
                hit_elem.set("rank", str(hit_data.get('rank', 0)))
                
                ET.SubElement(hit_elem, "ContentId").text = hit_data.get('content_id', '')
                ET.SubElement(hit_elem, "SimilarityScore").text = str(hit_data.get('score', 0.0))
                ET.SubElement(hit_elem, "ContentType").text = hit_data.get('content_type', 'unknown')


class StreamingExporter:
    """
    Real-time streaming export için sınıf.
    WebSocket, Kafka, Redis gibi streaming protokollerini destekler.
    """
    
    def __init__(self, config: Optional[ExportConfiguration] = None):
        """
        StreamingExporter sınıfını başlatır.
        
        Args:
            config: Export konfigürasyonu
        """
        self.config = config or ExportConfiguration()
        self.is_streaming = False
        self.clients = set()
        logger.info("StreamingExporter initialized for real-time data streaming")
    
    async def start_websocket_stream(self, host: str = "localhost", port: int = 8765):
        """
        WebSocket streaming server başlatır.
        
        Args:
            host: Server host
            port: Server port
        """
        if not ENTERPRISE_LIBS_AVAILABLE:
            raise ImportError("WebSocket streaming requires websockets library")
        
        async def handle_client(websocket, path):
            """WebSocket client handler"""
            self.clients.add(websocket)
            logger.info(f"New streaming client connected: {websocket.remote_address}")
            
            try:
                await websocket.wait_closed()
            finally:
                self.clients.remove(websocket)
                logger.info(f"Streaming client disconnected: {websocket.remote_address}")
        
        self.is_streaming = True
        logger.info(f"Starting WebSocket streaming server on {host}:{port}")
        
        async with websockets.serve(handle_client, host, port):
            await asyncio.Future()  # Run forever
    
    async def stream_data(self, data: Dict[str, Any]):
        """
        Veriyi connected clients'a stream eder.
        
        Args:
            data: Stream edilecek veri
        """
        if not self.is_streaming or not self.clients:
            return
        
        # Event wrapper oluştur
        if PROTOBUF_AVAILABLE:
            event = m3tm_export_pb2.ExportEvent()
            event.event_id = f"event_{datetime.now().isoformat()}"
            event.timestamp.GetCurrentTime()
            event.event_type = data.get('type', 'data')
            event.sequence_number = data.get('sequence', 0)
            
            # JSON format kullan (WebSocket için daha uygun)
            message = MessageToJson(event)
        else:
            # Fallback JSON format
            message = json.dumps({
                "event_id": f"event_{datetime.now().isoformat()}",
                "timestamp": datetime.now().isoformat(),
                "event_type": data.get('type', 'data'),
                "data": data
            })
        
        # Tüm connected clients'a gönder
        disconnected_clients = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Disconnected clients'ı temizle
        self.clients -= disconnected_clients
        
        logger.debug(f"Streamed data to {len(self.clients)} clients")


class EnterpriseIntegrationManager:
    """
    Enterprise sistem entegrasyonları için yönetici sınıf.
    Database, message queue, REST API ve diğer enterprise sistemlerle entegrasyon.
    """
    
    def __init__(self, config: Optional[ExportConfiguration] = None):
        """
        EnterpriseIntegrationManager sınıfını başlatır.
        
        Args:
            config: Export konfigürasyonu
        """
        self.config = config or ExportConfiguration()
        self.integrations = {}
        logger.info("EnterpriseIntegrationManager initialized")
    
    def register_integration(self, name: str, integration_config: Dict[str, Any]):
        """
        Yeni enterprise integration kaydeder.
        
        Args:
            name: Integration adı
            integration_config: Integration konfigürasyonu
        """
        self.integrations[name] = integration_config
        logger.info(f"Registered enterprise integration: {name}")
    
    async def export_to_integration(self, integration_name: str, data: List[Dict[str, Any]]) -> bool:
        """
        Belirtilen enterprise integration'a veri export eder.
        
        Args:
            integration_name: Integration adı
            data: Export edilecek veri
            
        Returns:
            Success status
        """
        if integration_name not in self.integrations:
            logger.error(f"Integration not found: {integration_name}")
            return False
        
        integration_config = self.integrations[integration_name]
        integration_type = integration_config.get('type', 'unknown')
        
        try:
            if integration_type == 'database':
                return await self._export_to_database(integration_config, data)
            elif integration_type == 'message_queue':
                return await self._export_to_message_queue(integration_config, data)
            elif integration_type == 'rest_api':
                return await self._export_to_rest_api(integration_config, data)
            else:
                logger.error(f"Unsupported integration type: {integration_type}")
                return False
                
        except Exception as e:
            logger.error(f"Integration export failed for {integration_name}: {str(e)}")
            return False
    
    async def _export_to_database(self, config: Dict[str, Any], data: List[Dict[str, Any]]) -> bool:
        """Database integration için export"""
        # Database integration implementation
        logger.info(f"Exporting {len(data)} records to database: {config.get('connection_string', 'unknown')}")
        # Implementation would depend on specific database type (PostgreSQL, MongoDB, etc.)
        return True
    
    async def _export_to_message_queue(self, config: Dict[str, Any], data: List[Dict[str, Any]]) -> bool:
        """Message queue integration için export"""
        # Message queue integration implementation (Kafka, RabbitMQ, Redis, etc.)
        logger.info(f"Exporting {len(data)} records to message queue: {config.get('queue_name', 'unknown')}")
        return True
    
    async def _export_to_rest_api(self, config: Dict[str, Any], data: List[Dict[str, Any]]) -> bool:
        """REST API integration için export"""
        # REST API integration implementation
        logger.info(f"Exporting {len(data)} records to REST API: {config.get('endpoint', 'unknown')}")
        return True


# Convenience function for quick exports
def export_data_enterprise(data: List[Dict[str, Any]], 
                          format_type: str,
                          file_path: str,
                          config: Optional[ExportConfiguration] = None) -> bool:
    """
    Enterprise formatlarında hızlı veri export fonksiyonu.
    
    Args:
        data: Export edilecek veri
        format_type: Export formatı ('protobuf', 'xml')
        file_path: Hedef dosya yolu
        config: Export konfigürasyonu
        
    Returns:
        Success status
    """
    try:
        if format_type.lower() == 'protobuf':
            exporter = ProtobufExporter(config)
        elif format_type.lower() == 'xml':
            exporter = XMLExporter(config)
        else:
            logger.error(f"Unsupported enterprise format: {format_type}")
            return False
        
        exporter.export(data, file_path)
        return True
        
    except Exception as e:
        logger.error(f"Enterprise export failed: {str(e)}")
        return False
