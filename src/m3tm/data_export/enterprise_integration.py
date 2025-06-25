"""
Enterprise Integration Manager

Bu modül enterprise sistemlerle entegrasyon için gerekli sınıfları içerir.
Real-time streaming, message queues, cloud storage, database integration.
Context7 best practices uygulanmıştır.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import queue
import time
from abc import ABC, abstractmethod

# Enterprise integration imports (optional)
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import grpc
    from grpc import aio as grpc_aio
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

logger = logging.getLogger(__name__)


@dataclass
class ExportConfiguration:
    """
    Export konfigürasyon sınıfı.
    Enterprise sistemler için detaylı konfigürasyon seçenekleri.
    """
    # Basic configuration
    format: str = "json"
    compression: bool = False
    compression_type: str = "gzip"  # gzip, lz4, snappy
    
    # Streaming configuration
    batch_size: int = 1000
    max_retry_count: int = 3
    timeout_seconds: int = 30
    
    # Enterprise integration
    enable_real_time: bool = False
    enable_webhooks: bool = False
    enable_encryption: bool = False
    
    # Cloud storage
    cloud_storage_config: Optional[Dict[str, Any]] = None
    
    # Database integration
    database_config: Optional[Dict[str, Any]] = None
    
    # Message queue integration
    message_queue_config: Optional[Dict[str, Any]] = None
    
    # Authentication
    auth_config: Optional[Dict[str, Any]] = None
    
    # Rate limiting
    rate_limit_requests_per_second: int = 100
    rate_limit_burst_size: int = 200
    
    # Monitoring
    enable_metrics: bool = True
    metrics_collection_interval: int = 60  # seconds
    
    # Custom fields
    custom_config: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """
    Token bucket rate limiter implementation.
    Enterprise API'ler için rate limiting.
    """
    
    def __init__(self, requests_per_second: int, burst_size: int):
        """
        RateLimiter sınıfını başlatır.
        
        Args:
            requests_per_second: Saniye başına istek sayısı
            burst_size: Burst boyutu
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Token alımı yapar.
        
        Args:
            tokens: İstenilen token sayısı
            
        Returns:
            Token alınabilirse True
        """
        with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            self.last_update = now
            
            # Token ekle
            self.tokens = min(
                self.burst_size,
                self.tokens + time_passed * self.requests_per_second
            )
            
            # Yeterli token var mı?
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def acquire_async(self, tokens: int = 1) -> None:
        """
        Async token alımı yapar.
        
        Args:
            tokens: İstenilen token sayısı
        """
        while not self.acquire(tokens):
            await asyncio.sleep(0.01)  # 10ms bekle


class StreamingExportManager:
    """
    Streaming export yöneticisi.
    Büyük veri setleri için memory-efficient streaming export.
    """
    
    def __init__(self, config: ExportConfiguration):
        """
        StreamingExportManager sınıfını başlatır.
        
        Args:
            config: Export konfigürasyonu
        """
        self.config = config
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests_per_second,
            config.rate_limit_burst_size
        )
        self._metrics = {
            'total_exported': 0,
            'bytes_exported': 0,
            'errors': 0,
            'start_time': None
        }
    
    async def stream_export(
        self,
        data_generator: AsyncGenerator[Dict[str, Any], None],
        export_callback: Callable[[List[Dict[str, Any]]], None]
    ) -> Dict[str, Any]:
        """
        Async streaming export yapar.
        
        Args:
            data_generator: Veri generator'ı
            export_callback: Export callback fonksiyonu
            
        Returns:
            Export metrikleri
        """
        self._metrics['start_time'] = datetime.now()
        batch = []
        
        try:
            async for item in data_generator:
                # Rate limiting kontrol et
                await self.rate_limiter.acquire_async()
                
                batch.append(item)
                
                # Batch boyutu doldu mu?
                if len(batch) >= self.config.batch_size:
                    await self._process_batch(batch, export_callback)
                    batch = []
            
            # Kalan batch'i işle
            if batch:
                await self._process_batch(batch, export_callback)
        
        except Exception as e:
            logger.error(f"Streaming export error: {e}")
            self._metrics['errors'] += 1
            raise
        
        return self._get_metrics()
    
    async def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        export_callback: Callable[[List[Dict[str, Any]]], None]
    ) -> None:
        """
        Bir batch'i işler.
        
        Args:
            batch: Veri batch'i
            export_callback: Export callback fonksiyonu
        """
        try:
            # Export callback'ini çağır
            export_callback(batch)
            
            # Metrikleri güncelle
            self._metrics['total_exported'] += len(batch)
            self._metrics['bytes_exported'] += len(json.dumps(batch).encode('utf-8'))
            
            logger.debug(f"Processed batch of {len(batch)} items")
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self._metrics['errors'] += 1
            
            # Retry logic
            for retry in range(self.config.max_retry_count):
                try:
                    await asyncio.sleep(2 ** retry)  # Exponential backoff
                    export_callback(batch)
                    break
                except Exception as retry_error:
                    logger.warning(f"Retry {retry + 1} failed: {retry_error}")
            else:
                raise e
    
    def _get_metrics(self) -> Dict[str, Any]:
        """
        Export metriklerini döndürür.
        
        Returns:
            Export metrikleri
        """
        elapsed = datetime.now() - self._metrics['start_time']
        
        return {
            'total_exported': self._metrics['total_exported'],
            'bytes_exported': self._metrics['bytes_exported'],
            'errors': self._metrics['errors'],
            'duration_seconds': elapsed.total_seconds(),
            'throughput_items_per_second': self._metrics['total_exported'] / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0,
            'throughput_mbps': (self._metrics['bytes_exported'] / 1024 / 1024) / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        }


class WebSocketExporter:
    """
    WebSocket üzerinden real-time export.
    Enterprise real-time integration için kullanılır.
    """
    
    def __init__(self, config: ExportConfiguration):
        """
        WebSocketExporter sınıfını başlatır.
        
        Args:
            config: Export konfigürasyonu
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("WebSocket support requires 'websockets' package")
        
        self.config = config
        self.connections = set()
    
    async def start_server(self, host: str = "localhost", port: int = 8765) -> None:
        """
        WebSocket sunucusunu başlatır.
        
        Args:
            host: Host adresi
            port: Port numarası
        """
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        async def handler(websocket, path):
            self.connections.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.connections.remove(websocket)
        
        await websockets.serve(handler, host, port)
    
    async def broadcast_data(self, data: Dict[str, Any]) -> None:
        """
        Tüm bağlı istemcilere veri gönderir.
        
        Args:
            data: Gönderilecek veri
        """
        if not self.connections:
            return
        
        message = json.dumps(data, ensure_ascii=False)
        
        # Tüm bağlantılara gönder
        disconnected = set()
        for websocket in self.connections:
            try:
                await websocket.send(message)
            except Exception:
                disconnected.add(websocket)
        
        # Kopuk bağlantıları temizle
        self.connections -= disconnected


class KafkaExporter:
    """
    Apache Kafka üzerinden stream export.
    Enterprise message queue integration için kullanılır.
    """
    
    def __init__(self, config: ExportConfiguration):
        """
        KafkaExporter sınıfını başlatır.
        
        Args:
            config: Export konfigürasyonu
        """
        if not HAS_KAFKA:
            raise ImportError("Kafka support requires 'kafka-python' package")
        
        self.config = config
        self.producer = None
        
        # Kafka konfigürasyonu
        kafka_config = config.message_queue_config or {}
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', ['localhost:9092'])
        self.topic = kafka_config.get('topic', 'm3tm_exports')
        
        # Producer oluştur
        self._create_producer()
    
    def _create_producer(self) -> None:
        """Kafka producer oluşturur."""
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8'),
            retries=self.config.max_retry_count,
            request_timeout_ms=self.config.timeout_seconds * 1000
        )
    
    def send_data(self, data: Dict[str, Any], key: Optional[str] = None) -> None:
        """
        Kafka'ya veri gönderir.
        
        Args:
            data: Gönderilecek veri
            key: Kafka message key
        """
        try:
            future = self.producer.send(
                self.topic,
                value=data,
                key=key.encode('utf-8') if key else None
            )
            
            # Async gönderim, hata kontrolü için callback ekle
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
        
        except Exception as e:
            logger.error(f"Kafka send error: {e}")
            raise
    
    def _on_send_success(self, record_metadata) -> None:
        """Başarılı gönderim callback."""
        logger.debug(f"Message sent to topic {record_metadata.topic} partition {record_metadata.partition}")
    
    def _on_send_error(self, exception) -> None:
        """Hatalı gönderim callback."""
        logger.error(f"Kafka send error: {exception}")
    
    def close(self) -> None:
        """Producer'ı kapatır."""
        if self.producer:
            self.producer.flush()
            self.producer.close()


class RedisExporter:
    """
    Redis üzerinden cache ve stream export.
    Enterprise caching ve real-time data için kullanılır.
    """
    
    def __init__(self, config: ExportConfiguration):
        """
        RedisExporter sınıfını başlatır.
        
        Args:
            config: Export konfigürasyonu
        """
        if not HAS_REDIS:
            raise ImportError("Redis support requires 'redis' package")
        
        self.config = config
        
        # Redis konfigürasyonu
        redis_config = config.message_queue_config or {}
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password'),
            socket_timeout=config.timeout_seconds
        )
    
    def cache_data(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """
        Redis'e veri cache'ler.
        
        Args:
            key: Cache key
            data: Cache'lenecek veri
            ttl: Time to live (seconds)
        """
        try:
            serialized_data = json.dumps(data, ensure_ascii=False)
            
            if ttl:
                self.redis_client.setex(key, ttl, serialized_data)
            else:
                self.redis_client.set(key, serialized_data)
        
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            raise
    
    def stream_data(self, stream_name: str, data: Dict[str, Any]) -> None:
        """
        Redis stream'e veri ekler.
        
        Args:
            stream_name: Stream adı
            data: Eklenecek veri
        """
        try:
            # Redis Streams kullan
            self.redis_client.xadd(stream_name, data)
        
        except Exception as e:
            logger.error(f"Redis stream error: {e}")
            raise
    
    def close(self) -> None:
        """Redis bağlantısını kapatır."""
        if self.redis_client:
            self.redis_client.close()


class EnterpriseIntegrationManager:
    """
    Enterprise sistemlerle entegrasyon yöneticisi.
    Tüm enterprise export seçeneklerini koordine eder.
    """
    
    def __init__(self, config: ExportConfiguration):
        """
        EnterpriseIntegrationManager sınıfını başlatır.
        
        Args:
            config: Enterprise konfigürasyonu
        """
        self.config = config
        self.exporters = {}
        
        # Export seçeneklerini başlat
        self._initialize_exporters()
    
    def _initialize_exporters(self) -> None:
        """Export seçeneklerini başlatır."""
        # Streaming export manager
        self.streaming_manager = StreamingExportManager(self.config)
        
        # Real-time exporters (eğer etkin ise)
        if self.config.enable_real_time:
            try:
                self.websocket_exporter = WebSocketExporter(self.config)
                self.exporters['websocket'] = self.websocket_exporter
            except ImportError:
                logger.warning("WebSocket exporter not available")
        
        # Message queue exporters
        if self.config.message_queue_config:
            queue_type = self.config.message_queue_config.get('type', 'kafka')
            
            if queue_type == 'kafka':
                try:
                    self.kafka_exporter = KafkaExporter(self.config)
                    self.exporters['kafka'] = self.kafka_exporter
                except ImportError:
                    logger.warning("Kafka exporter not available")
            
            elif queue_type == 'redis':
                try:
                    self.redis_exporter = RedisExporter(self.config)
                    self.exporters['redis'] = self.redis_exporter
                except ImportError:
                    logger.warning("Redis exporter not available")
    
    async def export_data(
        self,
        data: Union[List[Dict[str, Any]], AsyncGenerator[Dict[str, Any], None]],
        export_targets: List[str] = None
    ) -> Dict[str, Any]:
        """
        Veriyi belirlenen hedeflere export eder.
        
        Args:
            data: Export edilecek veri
            export_targets: Export hedefleri listesi
            
        Returns:
            Export sonuç metrikleri
        """
        if export_targets is None:
            export_targets = list(self.exporters.keys())
        
        results = {}
        
        # Her export hedefi için işlem yap
        for target in export_targets:
            if target not in self.exporters:
                logger.warning(f"Export target '{target}' not available")
                continue
            
            try:
                if target == 'websocket':
                    # WebSocket için tüm verileri broadcast et
                    if isinstance(data, list):
                        for item in data:
                            await self.websocket_exporter.broadcast_data(item)
                        results[target] = {'status': 'success', 'count': len(data)}
                    else:
                        count = 0
                        async for item in data:
                            await self.websocket_exporter.broadcast_data(item)
                            count += 1
                        results[target] = {'status': 'success', 'count': count}
                
                elif target == 'kafka':
                    # Kafka için tüm verileri gönder
                    if isinstance(data, list):
                        for item in data:
                            self.kafka_exporter.send_data(item)
                        results[target] = {'status': 'success', 'count': len(data)}
                    else:
                        count = 0
                        async for item in data:
                            self.kafka_exporter.send_data(item)
                            count += 1
                        results[target] = {'status': 'success', 'count': count}
                
                elif target == 'redis':
                    # Redis için cache ve stream
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            self.redis_exporter.cache_data(f"export_{i}", item)
                            self.redis_exporter.stream_data("m3tm_exports", item)
                        results[target] = {'status': 'success', 'count': len(data)}
                    else:
                        count = 0
                        async for item in data:
                            self.redis_exporter.cache_data(f"export_{count}", item)
                            self.redis_exporter.stream_data("m3tm_exports", item)
                            count += 1
                        results[target] = {'status': 'success', 'count': count}
            
            except Exception as e:
                logger.error(f"Export to {target} failed: {e}")
                results[target] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def close(self) -> None:
        """Tüm bağlantıları kapatır."""
        for exporter in self.exporters.values():
            if hasattr(exporter, 'close'):
                exporter.close()


# Export pattern factory
class ExportPatternFactory:
    """
    Export pattern factory.
    Context7 best practices'e göre factory pattern implementasyonu.
    """
    
    @staticmethod
    def create_enterprise_exporter(
        export_type: str,
        config: ExportConfiguration
    ) -> Union[WebSocketExporter, KafkaExporter, RedisExporter]:
        """
        Enterprise exporter oluşturur.
        
        Args:
            export_type: Export tipi
            config: Konfigürasyon
            
        Returns:
            Enterprise exporter instance
        """
        if export_type == 'websocket':
            return WebSocketExporter(config)
        elif export_type == 'kafka':
            return KafkaExporter(config)
        elif export_type == 'redis':
            return RedisExporter(config)
        else:
            raise ValueError(f"Unsupported export type: {export_type}")
    
    @staticmethod
    def create_streaming_manager(config: ExportConfiguration) -> StreamingExportManager:
        """
        Streaming manager oluşturur.
        
        Args:
            config: Konfigürasyon
            
        Returns:
            StreamingExportManager instance
        """
        return StreamingExportManager(config)
