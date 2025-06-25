#!/usr/bin/env python3
"""
M³TM Enterprise Data Export Demo

Bu demo S28 story için geliştirilen enterprise data export özelliklerini gösterir.
Context7 Protocol Buffers best practices uygulanmıştır.

Kullanım:
    python enterprise_export_demo.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from m3tm.data_export import (
    ExportManager,
    ExportConfiguration,
    ENTERPRISE_AVAILABLE,
    REST_API_AVAILABLE
)

# Optional enterprise imports
if ENTERPRISE_AVAILABLE:
    from m3tm.data_export import (
        StreamingExportManager,
        EnterpriseIntegrationManager,
        ExportPatternFactory
    )

if REST_API_AVAILABLE:
    from m3tm.data_export import (
        WebhookManager,
        ExportOrchestrator
    )

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data() -> List[Dict[str, Any]]:
    """
    Demo için örnek veri üretir.
    
    Returns:
        Örnek veri listesi
    """
    sample_data = []
    
    for i in range(100):
        item = {
            'id': f'item_{i:03d}',
            'content': f'Bu {i+1}. örnek veridir. M³TM enterprise export demo.',
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'type': 'text',
                'user_id': f'user_{i % 10}',
                'tags': ['demo', 'export', 'm3tm'],
                'embedding_vector': [0.1 * j for j in range(10)],  # Fake embedding
                'model_version': '2.3.0'
            },
            'stats': {
                'view_count': i * 2,
                'like_count': i,
                'share_count': i // 2
            }
        }
        sample_data.append(item)
    
    return sample_data


def demo_basic_exports():
    """
    Temel export formatlarını demo eder.
    """
    logger.info("=== Basic Export Formats Demo ===")
    
    # Sample data oluştur
    data = generate_sample_data()
    
    # Export manager oluştur
    export_manager = ExportManager()
    
    # Desteklenen formatları göster
    logger.info(f"Supported formats: {export_manager.supported_formats}")
    
    # Output directory oluştur
    output_dir = os.path.join(os.path.dirname(__file__), 'demo_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON Export
    json_file = os.path.join(output_dir, 'demo_export.json')
    export_manager.export(data, json_file, format='json')
    logger.info(f"JSON export completed: {json_file}")
    
    # CSV Export
    csv_file = os.path.join(output_dir, 'demo_export.csv')
    export_manager.export(data, csv_file, format='csv')
    logger.info(f"CSV export completed: {csv_file}")
    
    # XML Export
    xml_file = os.path.join(output_dir, 'demo_export.xml')
    export_manager.export(data, xml_file, format='xml')
    logger.info(f"XML export completed: {xml_file}")
    
    # Streaming JSON Export (büyük dosyalar için)
    streaming_json_file = os.path.join(output_dir, 'demo_streaming.json')
    export_manager.export(data, streaming_json_file, format='streaming_json')
    logger.info(f"Streaming JSON export completed: {streaming_json_file}")
    
    # Protocol Buffers Export (generic)
    protobuf_file = os.path.join(output_dir, 'demo_export.pb')
    try:
        export_manager.export(data, protobuf_file, format='protobuf')
        logger.info(f"Protocol Buffers export completed: {protobuf_file}")
    except Exception as e:
        logger.warning(f"Protocol Buffers export failed (expected): {e}")


async def demo_enterprise_exports():
    """
    Enterprise export özelliklerini demo eder.
    """
    if not ENTERPRISE_AVAILABLE:
        logger.warning("Enterprise features not available (missing dependencies)")
        return
    
    logger.info("=== Enterprise Export Features Demo ===")
    
    # Enterprise configuration
    config = ExportConfiguration(
        format="json",
        batch_size=50,
        enable_real_time=True,
        enable_webhooks=True,
        rate_limit_requests_per_second=10,
        enable_metrics=True
    )
    
    # Sample data oluştur
    data = generate_sample_data()
    
    # Streaming export manager
    streaming_manager = StreamingExportManager(config)
    
    async def data_generator():
        """Async data generator."""
        for item in data:
            yield item
            await asyncio.sleep(0.01)  # Gerçek stream'i simüle et
    
    def export_callback(batch: List[Dict[str, Any]]):
        """Export callback function."""
        logger.info(f"Processing batch of {len(batch)} items")
        # Gerçek export işlemi burada yapılır
    
    # Streaming export demo
    try:
        metrics = await streaming_manager.stream_export(
            data_generator(),
            export_callback
        )
        logger.info(f"Streaming export metrics: {json.dumps(metrics, indent=2)}")
    except Exception as e:
        logger.error(f"Streaming export error: {e}")
    
    # Enterprise Integration Manager demo
    enterprise_manager = EnterpriseIntegrationManager(config)
    
    try:
        # Mock export to available targets
        results = await enterprise_manager.export_data(
            data[:10],  # İlk 10 öğe
            export_targets=[]  # Boş liste - mevcut target'lar otomatik detect edilir
        )
        logger.info(f"Enterprise export results: {json.dumps(results, indent=2)}")
    except Exception as e:
        logger.error(f"Enterprise export error: {e}")
    
    # Cleanup
    enterprise_manager.close()


def demo_webhook_notifications():
    """
    Webhook notification sistemini demo eder.
    """
    if not REST_API_AVAILABLE:
        logger.warning("REST API features not available (missing dependencies)")
        return
    
    logger.info("=== Webhook Notifications Demo ===")
    
    # Webhook manager oluştur
    webhook_manager = WebhookManager()
    
    # Test webhook register et (gerçek URL yerine placeholder)
    from m3tm.data_export.rest_api import WebhookConfig
    
    webhook_config = WebhookConfig(
        url="https://httpbin.org/post",  # Test endpoint
        secret_token="demo_secret_token",
        timeout_seconds=10,
        max_retries=2
    )
    
    webhook_manager.register_webhook("demo_webhook", webhook_config)
    
    # Test notification gönder
    webhook_manager.notify(
        "export_completed",
        {
            "export_id": "demo_export_001",
            "status": "completed",
            "total_records": 100,
            "format": "json",
            "completed_at": datetime.now().isoformat()
        }
    )
    
    logger.info("Webhook notification sent (check httpbin.org for request)")
    
    # Cleanup
    webhook_manager.close()


def demo_export_patterns():
    """
    Export pattern factory'yi demo eder.
    """
    if not ENTERPRISE_AVAILABLE:
        logger.warning("Export patterns not available (missing dependencies)")
        return
    
    logger.info("=== Export Patterns Demo ===")
    
    # Configuration
    config = ExportConfiguration(
        format="json",
        batch_size=25,
        enable_real_time=True
    )
    
    # Factory pattern kullanarak exporters oluştur
    try:
        # Streaming manager
        streaming_manager = ExportPatternFactory.create_streaming_manager(config)
        logger.info("Streaming manager created successfully")
        
        # WebSocket exporter (eğer websockets mevcut ise)
        try:
            websocket_exporter = ExportPatternFactory.create_enterprise_exporter('websocket', config)
            logger.info("WebSocket exporter created successfully")
        except ImportError:
            logger.warning("WebSocket exporter not available")
        
    except Exception as e:
        logger.error(f"Export pattern creation error: {e}")


def display_feature_matrix():
    """
    Mevcut özelliklerin matrix'ini gösterir.
    """
    logger.info("=== M³TM Enterprise Export Feature Matrix ===")
    
    features = {
        "Basic Formats": {
            "JSON": "✓ Available",
            "CSV": "✓ Available", 
            "XML": "✓ Available",
            "Text": "✓ Available"
        },
        "Enterprise Formats": {
            "Protocol Buffers": "✓ Available (generic)",
            "Apache Avro": f"{'✓ Available' if ENTERPRISE_AVAILABLE else '✗ Missing dependencies'}",
            "Apache Parquet": f"{'✓ Available' if ENTERPRISE_AVAILABLE else '✗ Missing dependencies'}"
        },
        "Streaming & Real-time": {
            "Streaming JSON": "✓ Available",
            "WebSocket": f"{'✓ Available' if ENTERPRISE_AVAILABLE else '✗ Missing dependencies'}",
            "Apache Kafka": f"{'✓ Available' if ENTERPRISE_AVAILABLE else '✗ Missing dependencies'}",
            "Redis Streams": f"{'✓ Available' if ENTERPRISE_AVAILABLE else '✗ Missing dependencies'}"
        },
        "Enterprise Integration": {
            "REST API": f"{'✓ Available' if REST_API_AVAILABLE else '✗ Missing dependencies'}",
            "Webhooks": f"{'✓ Available' if REST_API_AVAILABLE else '✗ Missing dependencies'}",
            "Rate Limiting": f"{'✓ Available' if ENTERPRISE_AVAILABLE else '✗ Missing dependencies'}",
            "Authentication": f"{'✓ Available' if REST_API_AVAILABLE else '✗ Missing dependencies'}"
        }
    }
    
    for category, items in features.items():
        logger.info(f"\n{category}:")
        for feature, status in items.items():
            logger.info(f"  {feature}: {status}")


async def main():
    """
    Ana demo fonksiyonu.
    """
    logger.info("M³TM Enterprise Data Export Demo başlatılıyor...")
    logger.info(f"Enterprise features available: {ENTERPRISE_AVAILABLE}")
    logger.info(f"REST API features available: {REST_API_AVAILABLE}")
    
    # Feature matrix göster
    display_feature_matrix()
    
    # Basic exports demo
    demo_basic_exports()
    
    # Enterprise exports demo
    await demo_enterprise_exports()
    
    # Webhook notifications demo
    demo_webhook_notifications()
    
    # Export patterns demo
    demo_export_patterns()
    
    logger.info("Demo tamamlandı! Çıktı dosyaları 'demo_outputs' klasöründe.")


if __name__ == "__main__":
    # Demo'yu çalıştır
    asyncio.run(main())
