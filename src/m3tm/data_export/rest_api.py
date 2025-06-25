"""
REST API ve Webhook Support

Bu modül M³TM data export için REST API endpoints ve webhook 
notification sistemini sağlar. Enterprise integration için gerekli.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import threading
import queue
from dataclasses import dataclass
from urllib.parse import urlparse
import hashlib
import hmac

# REST API framework imports (optional)
try:
    from aiohttp import web, ClientSession, ClientTimeout
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, validator
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Webhook konfigürasyon sınıfı."""
    url: str
    secret_token: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 5
    verify_ssl: bool = True
    custom_headers: Optional[Dict[str, str]] = None


class ExportRequest(BaseModel):
    """Export request modeli."""
    export_id: str
    format: str = "json"
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    compression: bool = False
    webhook_url: Optional[str] = None
    
    @validator('format')
    def validate_format(cls, v):
        allowed_formats = ['json', 'csv', 'xml', 'protobuf', 'avro', 'parquet']
        if v not in allowed_formats:
            raise ValueError(f'Format must be one of {allowed_formats}')
        return v


@dataclass
class ExportStatus(BaseModel):
    """Export status modeli."""
    export_id: str
    status: str  # pending, running, completed, failed
    progress_percentage: float = 0.0
    total_records: int = 0
    processed_records: int = 0
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_files: List[str] = []


class WebhookManager:
    """
    Webhook notification yöneticisi.
    Export olayları için webhook notifications gönderir.
    """
    
    def __init__(self):
        """WebhookManager sınıfını başlatır."""
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.notification_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Worker thread'i başlat
        self._start_worker()
    
    def register_webhook(self, webhook_id: str, config: WebhookConfig) -> None:
        """
        Webhook kaydeder.
        
        Args:
            webhook_id: Webhook ID
            config: Webhook konfigürasyonu
        """
        self.webhooks[webhook_id] = config
        logger.info(f"Webhook registered: {webhook_id} -> {config.url}")
    
    def unregister_webhook(self, webhook_id: str) -> None:
        """
        Webhook kaydını siler.
        
        Args:
            webhook_id: Webhook ID
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Webhook unregistered: {webhook_id}")
    
    def notify(
        self,
        event_type: str,
        data: Dict[str, Any],
        webhook_ids: Optional[List[str]] = None
    ) -> None:
        """
        Webhook notification gönderir.
        
        Args:
            event_type: Event tipi
            data: Notification verisi
            webhook_ids: Bildirilecek webhook ID'leri (None ise tümü)
        """
        target_webhooks = webhook_ids or list(self.webhooks.keys())
        
        for webhook_id in target_webhooks:
            if webhook_id in self.webhooks:
                notification = {
                    'webhook_id': webhook_id,
                    'event_type': event_type,
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }
                self.notification_queue.put(notification)
    
    def _start_worker(self) -> None:
        """Webhook worker thread'ini başlatır."""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def _worker_loop(self) -> None:
        """Webhook worker ana döngüsü."""
        while not self.stop_event.is_set():
            try:
                # Queue'dan notification al (timeout ile)
                notification = self.notification_queue.get(timeout=1.0)
                self._send_webhook(notification)
                self.notification_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Webhook worker error: {e}")
    
    def _send_webhook(self, notification: Dict[str, Any]) -> None:
        """
        Webhook gönderir.
        
        Args:
            notification: Notification verisi
        """
        webhook_id = notification['webhook_id']
        config = self.webhooks.get(webhook_id)
        
        if not config:
            logger.warning(f"Webhook config not found: {webhook_id}")
            return
        
        # Async webhook gönderimi için event loop kullan
        try:
            # Yeni event loop oluştur (thread-safe)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            loop.run_until_complete(self._send_webhook_async(config, notification))
            loop.close()
        
        except Exception as e:
            logger.error(f"Webhook send error: {e}")
    
    async def _send_webhook_async(
        self,
        config: WebhookConfig,
        notification: Dict[str, Any]
    ) -> None:
        """
        Async webhook gönderir.
        
        Args:
            config: Webhook konfigürasyonu
            notification: Notification verisi
        """
        if not HAS_AIOHTTP:
            logger.error("aiohttp not available for webhook sending")
            return
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'M3TM-Webhook/1.0'
        }
        
        # Custom headers ekle
        if config.custom_headers:
            headers.update(config.custom_headers)
        
        # HMAC signature ekle (eğer secret token varsa)
        payload = json.dumps(notification, ensure_ascii=False).encode('utf-8')
        if config.secret_token:
            signature = hmac.new(
                config.secret_token.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            headers['X-Hub-Signature-256'] = f'sha256={signature}'
        
        timeout = ClientTimeout(total=config.timeout_seconds)
        
        for attempt in range(config.max_retries + 1):
            try:
                async with ClientSession(timeout=timeout) as session:
                    async with session.post(
                        config.url,
                        data=payload,
                        headers=headers,
                        ssl=config.verify_ssl
                    ) as response:
                        if response.status == 200:
                            logger.debug(f"Webhook sent successfully: {config.url}")
                            return
                        else:
                            logger.warning(f"Webhook response status: {response.status}")
            
            except Exception as e:
                logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                
                if attempt < config.max_retries:
                    await asyncio.sleep(config.retry_delay_seconds * (2 ** attempt))  # Exponential backoff
        
        logger.error(f"Webhook failed after {config.max_retries + 1} attempts: {config.url}")
    
    def close(self) -> None:
        """WebhookManager'ı kapatır."""
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)


class ExportAPI:
    """
    Export REST API sınıfı.
    FastAPI kullanarak export operasyonları için REST endpoints sağlar.
    """
    
    def __init__(self, export_manager, webhook_manager: WebhookManager):
        """
        ExportAPI sınıfını başlatır.
        
        Args:
            export_manager: Export manager instance
            webhook_manager: Webhook manager instance
        """
        if not HAS_FASTAPI:
            raise ImportError("FastAPI support requires 'fastapi' and 'uvicorn' packages")
        
        self.export_manager = export_manager
        self.webhook_manager = webhook_manager
        self.app = FastAPI(
            title="M³TM Data Export API",
            description="Enterprise data export API for M³TM",
            version="1.0.0"
        )
        
        # Export status tracking
        self.export_statuses: Dict[str, ExportStatus] = {}
        
        # Security
        self.security = HTTPBearer()
        
        # Routes'ları tanımla
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """API routes'larını kurar."""
        
        @self.app.post("/api/v1/exports", response_model=Dict[str, str])
        async def create_export(
            request: ExportRequest,
            background_tasks: BackgroundTasks,
            token: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """
            Yeni export işlemi başlatır.
            
            Args:
                request: Export request
                background_tasks: Background tasks
                token: Auth token
                
            Returns:
                Export ID
            """
            # Token validation (basit örnek)
            if not self._validate_token(token.credentials):
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Export status oluştur
            export_status = ExportStatus(
                export_id=request.export_id,
                status="pending",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.export_statuses[request.export_id] = export_status
            
            # Background task olarak export başlat
            background_tasks.add_task(
                self._process_export,
                request.export_id,
                request
            )
            
            # Webhook notification gönder
            self.webhook_manager.notify(
                "export_started",
                {"export_id": request.export_id, "format": request.format}
            )
            
            return {"export_id": request.export_id, "status": "started"}
        
        @self.app.get("/api/v1/exports/{export_id}", response_model=ExportStatus)
        async def get_export_status(
            export_id: str,
            token: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """
            Export durumunu sorgular.
            
            Args:
                export_id: Export ID
                token: Auth token
                
            Returns:
                Export status
            """
            if not self._validate_token(token.credentials):
                raise HTTPException(status_code=401, detail="Invalid token")
            
            if export_id not in self.export_statuses:
                raise HTTPException(status_code=404, detail="Export not found")
            
            return self.export_statuses[export_id]
        
        @self.app.delete("/api/v1/exports/{export_id}")
        async def cancel_export(
            export_id: str,
            token: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """
            Export işlemini iptal eder.
            
            Args:
                export_id: Export ID
                token: Auth token
                
            Returns:
                Cancellation status
            """
            if not self._validate_token(token.credentials):
                raise HTTPException(status_code=401, detail="Invalid token")
            
            if export_id not in self.export_statuses:
                raise HTTPException(status_code=404, detail="Export not found")
            
            # Export'u iptal et
            export_status = self.export_statuses[export_id]
            if export_status.status in ["pending", "running"]:
                export_status.status = "cancelled"
                export_status.updated_at = datetime.now()
                
                # Webhook notification gönder
                self.webhook_manager.notify(
                    "export_cancelled",
                    {"export_id": export_id}
                )
                
                return {"status": "cancelled"}
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel export in status: {export_status.status}"
                )
        
        @self.app.get("/api/v1/exports")
        async def list_exports(
            token: HTTPAuthorizationCredentials = Depends(self.security),
            status: Optional[str] = None,
            limit: int = 100
        ):
            """
            Export listesini döndürür.
            
            Args:
                token: Auth token
                status: Status filter
                limit: Result limit
                
            Returns:
                Export list
            """
            if not self._validate_token(token.credentials):
                raise HTTPException(status_code=401, detail="Invalid token")
            
            exports = list(self.export_statuses.values())
            
            # Status filter uygula
            if status:
                exports = [e for e in exports if e.status == status]
            
            # Limit uygula
            exports = exports[:limit]
            
            return {"exports": exports, "total": len(exports)}
        
        @self.app.post("/api/v1/webhooks")
        async def register_webhook(
            webhook_config: Dict[str, Any],
            token: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """
            Webhook kaydeder.
            
            Args:
                webhook_config: Webhook konfigürasyonu
                token: Auth token
                
            Returns:
                Registration status
            """
            if not self._validate_token(token.credentials):
                raise HTTPException(status_code=401, detail="Invalid token")
            
            config = WebhookConfig(
                url=webhook_config["url"],
                secret_token=webhook_config.get("secret_token"),
                timeout_seconds=webhook_config.get("timeout_seconds", 30),
                max_retries=webhook_config.get("max_retries", 3)
            )
            
            webhook_id = webhook_config.get("id", f"webhook_{len(self.webhook_manager.webhooks)}")
            self.webhook_manager.register_webhook(webhook_id, config)
            
            return {"webhook_id": webhook_id, "status": "registered"}
        
        @self.app.get("/api/v1/health")
        async def health_check():
            """
            Health check endpoint.
            
            Returns:
                Health status
            """
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
    
    def _validate_token(self, token: str) -> bool:
        """
        Token validation (basit örnek).
        
        Args:
            token: Auth token
            
        Returns:
            Geçerli ise True
        """
        # Gerçek uygulamada JWT validation vs. yapılmalı
        return token == "valid_api_key"  # Basit örnek
    
    async def _process_export(self, export_id: str, request: ExportRequest) -> None:
        """
        Export işlemini gerçekleştirir.
        
        Args:
            export_id: Export ID
            request: Export request
        """
        export_status = self.export_statuses[export_id]
        
        try:
            # Status'u running yap
            export_status.status = "running"
            export_status.updated_at = datetime.now()
            
            # Export manager kullanarak export yap
            # Bu basit bir örnek, gerçek implementasyon daha karmaşık olacak
            data = await self._get_export_data(request.filters)
            
            # Progress tracking
            export_status.total_records = len(data)
            
            # Export işlemini simüle et
            for i, item in enumerate(data):
                # İptal kontrolü
                if export_status.status == "cancelled":
                    return
                
                # Progress güncelle
                export_status.processed_records = i + 1
                export_status.progress_percentage = (i + 1) / len(data) * 100
                export_status.updated_at = datetime.now()
                
                # Kısa bekleme (gerçek işlemi simüle et)
                await asyncio.sleep(0.01)
            
            # Export başarılı
            export_status.status = "completed"
            export_status.completed_at = datetime.now()
            export_status.updated_at = datetime.now()
            export_status.output_files = [f"/exports/{export_id}.{request.format}"]
            
            # Webhook notification gönder
            self.webhook_manager.notify(
                "export_completed",
                {
                    "export_id": export_id,
                    "status": "completed",
                    "total_records": export_status.total_records,
                    "output_files": export_status.output_files
                }
            )
        
        except Exception as e:
            logger.error(f"Export {export_id} failed: {e}")
            
            export_status.status = "failed"
            export_status.error_message = str(e)
            export_status.updated_at = datetime.now()
            
            # Webhook notification gönder
            self.webhook_manager.notify(
                "export_failed",
                {
                    "export_id": export_id,
                    "error": str(e)
                }
            )
    
    async def _get_export_data(self, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Export edilecek veriyi alır.
        
        Args:
            filters: Veri filtreleri
            
        Returns:
            Export verisi
        """
        # Bu basit bir örnek, gerçek implementasyon database'den veri alacak
        sample_data = [
            {"id": f"item_{i}", "content": f"Sample content {i}", "timestamp": datetime.now().isoformat()}
            for i in range(1000)  # 1000 örnek veri
        ]
        
        return sample_data
    
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        API sunucusunu başlatır.
        
        Args:
            host: Host adresi
            port: Port numarası
        """
        uvicorn.run(self.app, host=host, port=port)


# Export orchestrator
class ExportOrchestrator:
    """
    Export işlemlerini orkestre eden ana sınıf.
    Tüm export seçeneklerini koordine eder.
    """
    
    def __init__(self, export_manager):
        """
        ExportOrchestrator sınıfını başlatır.
        
        Args:
            export_manager: Export manager instance
        """
        self.export_manager = export_manager
        self.webhook_manager = WebhookManager()
        
        # API server (isteğe bağlı)
        self.api_server = None
        if HAS_FASTAPI:
            self.api_server = ExportAPI(export_manager, self.webhook_manager)
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        REST API sunucusunu başlatır.
        
        Args:
            host: Host adresi
            port: Port numarası
        """
        if not self.api_server:
            raise RuntimeError("API server not available (FastAPI not installed)")
        
        logger.info(f"Starting export API server on {host}:{port}")
        self.api_server.run(host, port)
    
    def close(self) -> None:
        """Orchestrator'ı kapatır."""
        if self.webhook_manager:
            self.webhook_manager.close()
