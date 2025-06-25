#!/bin/bash

# Protocol Buffers compilation script for M3TM Enterprise Integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔨 Compiling Protocol Buffers schemas for M3TM Enterprise Integration...${NC}"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
PROTO_DIR="$SCRIPT_DIR/proto"
OUTPUT_DIR="$SCRIPT_DIR/generated"

echo -e "${YELLOW}📁 Project root: $PROJECT_ROOT${NC}"
echo -e "${YELLOW}📁 Proto directory: $PROTO_DIR${NC}"
echo -e "${YELLOW}📁 Output directory: $OUTPUT_DIR${NC}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo -e "${RED}❌ Error: protoc (Protocol Buffers compiler) is not installed${NC}"
    echo -e "${YELLOW}💡 Install with: apt-get install protobuf-compiler (Ubuntu/Debian)${NC}"
    echo -e "${YELLOW}💡 Or with: brew install protobuf (macOS)${NC}"
    exit 1
fi

# Check protoc version
PROTOC_VERSION=$(protoc --version)
echo -e "${GREEN}✅ Found protoc: $PROTOC_VERSION${NC}"

# Compile proto files
echo -e "${GREEN}🔄 Compiling proto files...${NC}"

# Compile m3tm_export.proto
echo -e "${YELLOW}  📄 Compiling m3tm_export.proto...${NC}"
protoc --python_out="$OUTPUT_DIR" \
       --proto_path="$PROTO_DIR" \
       "$PROTO_DIR/m3tm_export.proto"

# Compile enterprise_integration.proto
echo -e "${YELLOW}  📄 Compiling enterprise_integration.proto...${NC}"
protoc --python_out="$OUTPUT_DIR" \
       --proto_path="$PROTO_DIR" \
       "$PROTO_DIR/enterprise_integration.proto"

# Create __init__.py files
echo -e "${GREEN}📝 Creating __init__.py files...${NC}"
touch "$OUTPUT_DIR/__init__.py"

# Create Python package imports
cat > "$OUTPUT_DIR/__init__.py" << 'EOF'
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
EOF

# Verify generated files
echo -e "${GREEN}✅ Verifying generated files...${NC}"

EXPECTED_FILES=(
    "m3tm_export_pb2.py"
    "enterprise_integration_pb2.py"
)

for file in "${EXPECTED_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        echo -e "${GREEN}  ✅ $file${NC}"
        # Show file size
        SIZE=$(ls -lh "$OUTPUT_DIR/$file" | awk '{print $5}')
        echo -e "${YELLOW}    📏 Size: $SIZE${NC}"
    else
        echo -e "${RED}  ❌ $file (missing)${NC}"
        exit 1
    fi
done

# Create convenience imports in parent directory
echo -e "${GREEN}📦 Creating convenience imports...${NC}"
cat > "$SCRIPT_DIR/__init__.py" << 'EOF'
"""
M3TM Data Export Module

This module provides enterprise-grade data export capabilities including:
- Multi-format export (JSON, CSV, XML, Protocol Buffers)
- Real-time streaming integration
- Enterprise system connectivity
- Data transformation and validation

Generated Protocol Buffer classes are available in the 'generated' subpackage.
"""

from .export_manager import ExportManager
from .export_formats import (
    BaseExporter,
    JSONExporter,
    CSVExporter,
    TextExporter
)

# Import new enterprise exporters when they become available
try:
    from .enterprise_exporters import (
        ProtobufExporter,
        XMLExporter,
        StreamingExporter,
        EnterpriseIntegrationManager
    )
    __all__ = [
        'ExportManager',
        'BaseExporter', 'JSONExporter', 'CSVExporter', 'TextExporter',
        'ProtobufExporter', 'XMLExporter', 'StreamingExporter',
        'EnterpriseIntegrationManager'
    ]
except ImportError:
    # Fallback when enterprise features are not yet implemented
    __all__ = [
        'ExportManager',
        'BaseExporter', 'JSONExporter', 'CSVExporter', 'TextExporter'
    ]

# Version info
__version__ = "2.3.0"
__author__ = "M3TM Development Team"
__enterprise_features__ = True
EOF

echo -e "${GREEN}🎉 Protocol Buffers compilation completed successfully!${NC}"
echo -e "${YELLOW}💡 Generated files are available in: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}💡 To use in Python:${NC}"
echo -e "${YELLOW}    from m3tm.data_export.generated import m3tm_export_pb2${NC}"
echo -e "${YELLOW}    from m3tm.data_export.generated import enterprise_integration_pb2${NC}"

# Optional: Display file tree
if command -v tree &> /dev/null; then
    echo -e "${GREEN}📁 Generated file structure:${NC}"
    tree "$OUTPUT_DIR" || ls -la "$OUTPUT_DIR"
fi
