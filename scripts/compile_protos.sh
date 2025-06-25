#!/bin/bash

# Protocol Buffers compilation script for M³TM v2.3 S28 Data Export
# This script compiles all .proto files to Python code

set -e

PROTO_DIR="src/m3tm/data_export/proto"
PROTO_OUT_DIR="src/m3tm/data_export/proto"

echo "Compiling Protocol Buffers schemas..."

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "Error: protoc (Protocol Buffers compiler) is not installed."
    echo "Please install it using: brew install protobuf"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$PROTO_OUT_DIR"

# Compile all .proto files
for proto_file in "$PROTO_DIR"/*.proto; do
    if [[ -f "$proto_file" ]]; then
        echo "Compiling $(basename "$proto_file")..."
        protoc --python_out="$PROTO_OUT_DIR" --proto_path="$PROTO_DIR" "$proto_file"
    fi
done

# Create __init__.py for the proto package
cat > "$PROTO_OUT_DIR/__init__.py" << 'EOF'
"""
Protocol Buffers generated modules for M³TM v2.3 Data Export.
Generated from .proto schema files.
"""

# Import generated protobuf modules
try:
    from . import streaming_pb2
    from . import m3tm_export_pb2
    from . import enterprise_integration_pb2
    
    __all__ = [
        'streaming_pb2', 
        'm3tm_export_pb2',
        'enterprise_integration_pb2'
    ]
except ImportError as e:
    # Silently fail - errors will be handled by importing modules
    __all__ = []
EOF

echo "Protocol Buffers compilation completed successfully!"
echo "Generated Python modules are available in $PROTO_OUT_DIR"
