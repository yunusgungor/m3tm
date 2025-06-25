"""
Generated Protocol Buffers for M3TM Enterprise Integration

This package contains auto-generated Python classes from Protocol Buffer
schema definitions for M3TM's enterprise data export and integration features.

Generated files:
- m3tm_export_pb2.py: Core M3TM data export schemas
- enterprise_integration_pb2.py: Enterprise integration patterns and configurations

Usage:
    from m3tm.data_export.generated import m3tm_export_pb2
    from m3tm.data_export.generated import enterprise_integration_pb2
    
    # Create export data
    export_data = m3tm_export_pb2.M3TMExportData()
    
    # Create integration request
    integration_request = enterprise_integration_pb2.EnterpriseIntegrationRequest()
"""

# Import generated modules for easier access
try:
    from . import m3tm_export_pb2
    from . import enterprise_integration_pb2
    
    __all__ = [
        'm3tm_export_pb2',
        'enterprise_integration_pb2'
    ]
except ImportError:
    # Handle case when modules are not yet generated
    __all__ = []
