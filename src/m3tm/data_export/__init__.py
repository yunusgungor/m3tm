"""
M³TM Data Export Module

Bu modül M³TM için enterprise-level veri dışa aktarma işlevlerini sağlar:
- Multi-format export (JSON, CSV, XML, Protocol Buffers, Avro, Parquet)
- Real-time streaming integration (WebSocket, Kafka, Redis)
- Enterprise system connectivity
- REST API endpoints with webhook notifications
- Data transformation and validation

Context7 best practices uygulanmıştır.
Generated Protocol Buffer classes are available in the 'proto' subpackage.
"""

from .export_manager import ExportManager
from .export_formats import (
    BaseExporter,
    JSONExporter,
    CSVExporter,
    TextExporter,
    XMLExporter,
    ProtocolBuffersExporter,
    AvroExporter,
    ParquetExporter,
    StreamingExporter,
    StreamingJSONExporter
)

# Enterprise integration (optional imports)
try:
    from .enterprise_integration import (
        ExportConfiguration,
        StreamingExportManager,
        WebSocketExporter,
        KafkaExporter,
        RedisExporter,
        EnterpriseIntegrationManager,
        ExportPatternFactory,
        RateLimiter
    )
    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False

# REST API and webhooks (optional imports)
try:
    from .rest_api import (
        WebhookManager,
        ExportAPI,
        ExportOrchestrator,
        WebhookConfig,
        ExportRequest,
        ExportStatus
    )
    REST_API_AVAILABLE = True
except ImportError:
    REST_API_AVAILABLE = False

__all__ = [
    # Core exports
    'ExportManager',
    'BaseExporter',
    'JSONExporter',
    'CSVExporter', 
    'TextExporter',
    'XMLExporter',
    'ProtocolBuffersExporter',
    'AvroExporter',
    'ParquetExporter',
    'StreamingExporter',
    'StreamingJSONExporter',
]

# Enterprise exports (if available)
if ENTERPRISE_AVAILABLE:
    __all__.extend([
        'ExportConfiguration',
        'StreamingExportManager',
        'WebSocketExporter',
        'KafkaExporter',
        'RedisExporter',
        'EnterpriseIntegrationManager',
        'ExportPatternFactory',
        'RateLimiter'
    ])

# REST API exports (if available)
if REST_API_AVAILABLE:
    __all__.extend([
        'WebhookManager',
        'ExportAPI',
        'ExportOrchestrator',
        'WebhookConfig',
        'ExportRequest',
        'ExportStatus'
    ])

# Version info
__version__ = "2.3.0"
__author__ = "M³TM Development Team"
__description__ = "Enterprise data export system for M³TM with Context7 best practices"
