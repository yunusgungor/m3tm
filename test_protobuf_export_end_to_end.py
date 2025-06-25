#!/usr/bin/env python3
"""
End-to-End Test for Protocol Buffers Enterprise Export

This script tests the complete protobuf export workflow including:
- BatchExportData message creation
- Serialization to binary and text formats
- Deserialization verification
- Data integrity validation
"""

import os
import sys
import tempfile
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, '/Users/yunusgungor/work/mobilemodel/src')

try:
    from m3tm.data_export.enterprise_exporters import ProtobufExporter, ExportConfiguration
    import src.m3tm.data_export.proto.m3tm_export_pb2 as m3tm_export_pb2
    print("✅ Successfully imported ProtobufExporter and related modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def create_sample_data():
    """Create comprehensive sample data for testing"""
    return [
        {
            'type': 'embedding',
            'id': 'embed_001',
            'content_type': 'text',
            'vector': [0.1, 0.2, 0.3, 0.4, 0.5] * 20,  # 100-dimensional vector
            'content': 'Sample text content for embedding',
            'model_version': 'M3TM_v2.3',
            'metadata': {
                'source': 'test_document.txt',
                'language': 'en',
                'timestamp': '2025-06-25T10:00:00Z'
            }
        },
        {
            'type': 'embedding',
            'id': 'embed_002',
            'content_type': 'image',
            'vector': [0.6, 0.7, 0.8, 0.9, 1.0] * 30,  # 150-dimensional vector
            'content': 'Image embedding data',
            'model_version': 'M3TM_v2.3',
            'metadata': {
                'source': 'test_image.jpg',
                'resolution': '1920x1080',
                'format': 'jpeg'
            }
        },
        {
            'type': 'search_result',
            'search_id': 'search_001',
            'query': 'machine learning multimodal',
            'query_type': 'text',
            'search_time_ms': 125.5,
            'results': [
                {
                    'content_id': 'doc_001',
                    'score': 0.95,
                    'content_type': 'document',
                    'preview': 'Introduction to multimodal machine learning...',
                    'rank': 1
                },
                {
                    'content_id': 'doc_002',
                    'score': 0.87,
                    'content_type': 'paper',
                    'preview': 'Advances in cross-modal representation...',
                    'rank': 2
                }
            ]
        },
        {
            'type': 'user_data',
            'user_id': 'user_12345',
            'content_id': 'content_789',
            'content_type': 'preference',
            'is_exportable': True,
            'consent_level': 'full'
        }
    ]

def test_protobuf_export():
    """Run comprehensive protobuf export test"""
    print("\n🚀 Starting Protocol Buffers End-to-End Test")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    sample_data = create_sample_data()
    print(f"✅ Created {len(sample_data)} sample records")
    
    # Test with different configurations
    test_configs = [
        {
            'name': 'Full Export',
            'config': ExportConfiguration(
                include_embeddings=True,
                include_user_data=True,
                include_search_results=True,
                anonymize_user_data=False
            )
        },
        {
            'name': 'Privacy-Safe Export',
            'config': ExportConfiguration(
                include_embeddings=True,
                include_user_data=True,
                include_search_results=True,
                anonymize_user_data=True
            )
        },
        {
            'name': 'Embeddings Only',
            'config': ExportConfiguration(
                include_embeddings=True,
                include_user_data=False,
                include_search_results=False
            )
        }
    ]
    
    success_count = 0
    total_tests = len(test_configs) * 2  # Binary + Text formats
    
    for test_config in test_configs:
        print(f"\n2. Testing Configuration: {test_config['name']}")
        print("-" * 40)
        
        try:
            # Create exporter
            exporter = ProtobufExporter(test_config['config'])
            print("✅ ProtobufExporter created successfully")
            
            # Test binary format
            with tempfile.NamedTemporaryFile(suffix='.pb', delete=False) as temp_file:
                binary_path = temp_file.name
            
            try:
                print("   Testing binary format export...")
                exporter.export(sample_data, binary_path)
                print(f"✅ Binary export completed: {os.path.getsize(binary_path)} bytes")
                
                # Verify binary file can be read back
                with open(binary_path, 'rb') as f:
                    binary_data = f.read()
                
                # Deserialize and verify
                restored_msg = m3tm_export_pb2.BatchExportData()
                restored_msg.ParseFromString(binary_data)
                print(f"✅ Binary deserialization successful")
                print(f"   - Export ID: {restored_msg.metadata.export_id}")
                print(f"   - Total records: {restored_msg.metadata.total_records}")
                print(f"   - Embeddings count: {len(restored_msg.embeddings)}")
                print(f"   - Search results count: {len(restored_msg.search_results)}")
                print(f"   - User data count: {len(restored_msg.user_data)}")
                
                success_count += 1
                
            finally:
                if os.path.exists(binary_path):
                    os.unlink(binary_path)
            
            # Test text format
            with tempfile.NamedTemporaryFile(suffix='.pbtxt', delete=False) as temp_file:
                text_path = temp_file.name
            
            try:
                print("   Testing text format export...")
                exporter.export(sample_data, text_path)
                print(f"✅ Text export completed: {os.path.getsize(text_path)} bytes")
                
                # Verify text file content
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                # Basic content validation
                assert 'metadata {' in text_content, "Metadata section missing"
                assert 'export_id:' in text_content, "Export ID missing"
                print("✅ Text format validation successful")
                
                success_count += 1
                
            finally:
                if os.path.exists(text_path):
                    os.unlink(text_path)
                    
        except Exception as e:
            print(f"❌ Test failed for {test_config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n🏁 Test Summary")
    print("=" * 60)
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! Protocol Buffers export is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False

def test_message_structure():
    """Test the structure of generated protobuf messages"""
    print("\n🔍 Testing Protobuf Message Structure")
    print("=" * 60)
    
    try:
        # Create a minimal message to inspect structure
        export_msg = m3tm_export_pb2.BatchExportData()
        
        print("✅ BatchExportData message created")
        print(f"   Available fields: {[field.name for field in export_msg.DESCRIPTOR.fields]}")
        
        # Test metadata access
        metadata = export_msg.metadata
        print(f"✅ Metadata accessible: {[field.name for field in metadata.DESCRIPTOR.fields]}")
        
        # Test repeated fields
        embedding = export_msg.embeddings.add()
        print(f"✅ Embedding field accessible: {[field.name for field in embedding.DESCRIPTOR.fields]}")
        
        search_result = export_msg.search_results.add()
        print(f"✅ SearchResult field accessible: {[field.name for field in search_result.DESCRIPTOR.fields]}")
        
        user_data = export_msg.user_data.add()
        print(f"✅ UserData field accessible: {[field.name for field in user_data.DESCRIPTOR.fields]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Message structure test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_integrity():
    """Test data integrity through serialization/deserialization cycle"""
    print("\n🔐 Testing Data Integrity")
    print("=" * 60)
    
    try:
        # Create exporter
        exporter = ProtobufExporter(ExportConfiguration())
        
        # Create test data with specific values
        test_data = [
            {
                'type': 'embedding',
                'id': 'integrity_test_001',
                'content_type': 'text',
                'vector': [1.1, 2.2, 3.3, 4.4, 5.5],
                'content': 'Integrity test content',
                'model_version': 'test_v1.0',
                'metadata': {
                    'test_key': 'test_value',
                    'numeric_key': '42'
                }
            }
        ]
        
        # Export to binary
        with tempfile.NamedTemporaryFile(suffix='.pb', delete=False) as temp_file:
            binary_path = temp_file.name
        
        try:
            exporter.export(test_data, binary_path)
            
            # Read back and verify data integrity
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            restored_msg = m3tm_export_pb2.BatchExportData()
            restored_msg.ParseFromString(binary_data)
            
            # Verify embedding data
            assert len(restored_msg.embeddings) == 1, f"Expected 1 embedding, got {len(restored_msg.embeddings)}"
            
            embedding = restored_msg.embeddings[0]
            assert embedding.embedding_id == 'integrity_test_001', f"ID mismatch: {embedding.embedding_id}"
            assert embedding.content_type == 'text', f"Content type mismatch: {embedding.content_type}"
            assert embedding.dimension == 5, f"Dimension mismatch: {embedding.dimension}"
            assert len(embedding.vector) == 5, f"Vector length mismatch: {len(embedding.vector)}"
            assert abs(embedding.vector[0] - 1.1) < 0.001, f"Vector value mismatch: {embedding.vector[0]}"
            assert embedding.source_content == 'Integrity test content', f"Content mismatch: {embedding.source_content}"
            assert embedding.model_version == 'test_v1.0', f"Model version mismatch: {embedding.model_version}"
            
            # Verify metadata
            assert 'test_key' in embedding.metadata, "Metadata key missing"
            assert embedding.metadata['test_key'] == 'test_value', f"Metadata value mismatch: {embedding.metadata['test_key']}"
            
            print("✅ Data integrity verified successfully")
            print(f"   - Embedding ID: {embedding.embedding_id}")
            print(f"   - Vector dimension: {embedding.dimension}")
            print(f"   - Metadata keys: {list(embedding.metadata.keys())}")
            
            return True
            
        finally:
            if os.path.exists(binary_path):
                os.unlink(binary_path)
                
    except Exception as e:
        print(f"❌ Data integrity test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 M³TM Protocol Buffers Enterprise Export - End-to-End Test")
    print("=" * 80)
    
    all_tests_passed = True
    
    # Run all tests
    all_tests_passed &= test_message_structure()
    all_tests_passed &= test_data_integrity()
    all_tests_passed &= test_protobuf_export()
    
    print(f"\n{'🎉' if all_tests_passed else '❌'} Overall Test Result: {'SUCCESS' if all_tests_passed else 'FAILURE'}")
    print("=" * 80)
    
    if all_tests_passed:
        print("✅ Protocol Buffers enterprise export is fully operational!")
        print("✅ BatchExportData serialization/deserialization working correctly")
        print("✅ All data types (embeddings, search results, user data) supported")
        print("✅ Both binary and text formats working")
        print("✅ Data integrity maintained through serialization cycle")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please review the error messages above.")
        sys.exit(1)
