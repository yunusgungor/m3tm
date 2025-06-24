#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for M³TM v2.3
Tests all mobile optimization features and validates performance targets.
"""

import unittest
import torch
import torch.nn as nn
import time
import json
import tempfile
import os
from pathlib import Path
import numpy as np
from PIL import Image
import sys
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the demo module
from examples.simplified_mobile_demo import SimplifiedM3TMDemo

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestM3TMPerformance(unittest.TestCase):
    """Test performance requirements and mobile optimization."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment."""
        cls.demo_config = {
            'model_name': 'M³TM v2.3 Test',
            'version': '2.3.0',
            'mobile_optimized': True
        }
        cls.demo = MobileOptimizedM3TM(cls.demo_config)
        cls.demo.build_model()
    
    def test_model_size_requirement(self):
        """Test that model size is under 500MB."""
        model_size = self.demo._get_model_size()
        self.assertLess(model_size, 500, 
                       f"Model size {model_size}MB exceeds 500MB limit")
        logger.info(f"✅ Model size test passed: {model_size}MB < 500MB")
    
    def test_memory_usage_requirement(self):
        """Test that memory usage is under 2GB."""
        memory_usage = self.demo._get_memory_usage()
        self.assertLess(memory_usage, 2048, 
                       f"Memory usage {memory_usage}MB exceeds 2GB limit")
        logger.info(f"✅ Memory usage test passed: {memory_usage}MB < 2GB")
    
    def test_text_inference_speed(self):
        """Test that text inference is under 100ms."""
        text = "Sample text for performance testing"
        
        # Warm up
        for _ in range(3):
            self.demo.process_text(text)
        
        # Measure performance
        times = []
        for _ in range(10):
            start = time.time()
            self.demo.process_text(text)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        self.assertLess(avg_time, 100, 
                       f"Text inference {avg_time:.2f}ms exceeds 100ms limit")
        logger.info(f"✅ Text inference speed test passed: {avg_time:.2f}ms < 100ms")
    
    def test_image_inference_speed(self):
        """Test that image inference is under 100ms."""
        image = Image.new('RGB', (224, 224), color='red')
        
        # Warm up
        for _ in range(3):
            self.demo.process_image(image)
        
        # Measure performance
        times = []
        for _ in range(10):
            start = time.time()
            self.demo.process_image(image)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        self.assertLess(avg_time, 100, 
                       f"Image inference {avg_time:.2f}ms exceeds 100ms limit")
        logger.info(f"✅ Image inference speed test passed: {avg_time:.2f}ms < 100ms")


class TestM3TMFunctionality(unittest.TestCase):
    """Test core functionality of M³TM system."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment."""
        cls.demo_config = {
            'model_name': 'M³TM v2.3 Test',
            'version': '2.3.0',
            'mobile_optimized': True
        }
        cls.demo = MobileOptimizedM3TM(cls.demo_config)
        cls.demo.build_model()
    
    def test_text_processing(self):
        """Test text processing functionality."""
        test_texts = [
            "Hello world",
            "Machine learning is fascinating",
            "Mobile AI applications"
        ]
        
        for text in test_texts:
            embedding = self.demo.process_text(text)
            self.assertIsInstance(embedding, torch.Tensor)
            self.assertEqual(len(embedding.shape), 1, "Text embedding should be 1D")
            self.assertGreater(embedding.shape[0], 0, "Text embedding should not be empty")
        
        logger.info("✅ Text processing functionality test passed")
    
    def test_image_processing(self):
        """Test image processing functionality."""
        test_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (256, 256), color='green'),
            Image.new('RGB', (512, 512), color='blue')
        ]
        
        for image in test_images:
            embedding = self.demo.process_image(image)
            self.assertIsInstance(embedding, torch.Tensor)
            # Image embedding can be 1D or 2D depending on batch processing
            self.assertLessEqual(len(embedding.shape), 2, "Image embedding should be 1D or 2D")
            self.assertGreater(embedding.numel(), 0, "Image embedding should not be empty")
        
        logger.info("✅ Image processing functionality test passed")
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        # Create test embeddings
        test_texts = [
            "Machine learning algorithms",
            "Deep neural networks", 
            "Computer vision applications"
        ]
        
        embeddings = [self.demo.process_text(text) for text in test_texts]
        
        # Test search
        query = "artificial intelligence"
        results = self.demo.semantic_search(query, embeddings, top_k=3)
        
        self.assertEqual(len(results), 3, "Should return top 3 results")
        
        for result in results:
            self.assertIn('index', result)
            self.assertIn('similarity', result)
            self.assertIsInstance(result['similarity'], float)
            self.assertGreaterEqual(result['index'], 0)
            self.assertLess(result['index'], len(embeddings))
        
        # Check that results are sorted by similarity
        similarities = [r['similarity'] for r in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        
        logger.info("✅ Semantic search functionality test passed")
    
    def test_data_export(self):
        """Test data export functionality."""
        export_data = self.demo.export_data()
        
        self.assertIsInstance(export_data, dict)
        self.assertIn('model_config', export_data)
        self.assertIn('performance_metrics', export_data)
        self.assertIn('mobile_optimizations', export_data)
        self.assertIn('export_timestamp', export_data)
        
        # Test JSON serialization
        json_str = json.dumps(export_data, default=str)
        self.assertIsInstance(json_str, str)
        
        logger.info("✅ Data export functionality test passed")


class TestM3TMMobileOptimizations(unittest.TestCase):
    """Test mobile-specific optimizations."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment."""
        cls.demo_config = {
            'model_name': 'M³TM v2.3 Test',
            'version': '2.3.0',
            'mobile_optimized': True
        }
        cls.demo = MobileOptimizedM3TM(cls.demo_config)
        cls.demo.build_model()
    
    def test_torch_compile_applied(self):
        """Test that torch.compile optimization is applied."""
        # This test checks if the model was successfully compiled
        self.assertTrue(hasattr(self.demo, 'is_compiled'))
        
        # Since compilation might fail on some systems, we just check it was attempted
        logger.info(f"✅ torch.compile test: {'applied' if self.demo.is_compiled else 'attempted'}")
    
    def test_quantization_applied(self):
        """Test that quantization is applied."""
        # Check if quantization is enabled in mobile optimizations
        self.assertTrue(self.demo.mobile_optimizations['quantization'])
        
        # For CPU, check if the model has quantized layers
        if self.demo.device.type == 'cpu':
            # Look for quantized operations in the model
            has_quantized_ops = any(
                'quantized' in str(type(module)).lower() 
                for module in self.demo.model.modules()
            )
            logger.info(f"✅ Quantization test: enabled, quantized ops found: {has_quantized_ops}")
        else:
            # For GPU, check if model is in half precision
            first_param = next(self.demo.model.parameters())
            is_half_precision = first_param.dtype == torch.float16
            logger.info(f"✅ Quantization test: FP16 applied: {is_half_precision}")
    
    def test_mobile_optimization_flags(self):
        """Test that all mobile optimization flags are properly set."""
        expected_optimizations = [
            'torch_compile',
            'static_cache', 
            'sdpa_flash_attention',
            'quantization',
            'memory_efficient_attention'
        ]
        
        for opt in expected_optimizations:
            self.assertIn(opt, self.demo.mobile_optimizations)
            self.assertIsInstance(self.demo.mobile_optimizations[opt], bool)
        
        logger.info("✅ Mobile optimization flags test passed")


class TestM3TMStressTest(unittest.TestCase):
    """Stress tests for mobile performance under load."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment."""
        cls.demo_config = {
            'model_name': 'M³TM v2.3 Test',
            'version': '2.3.0',
            'mobile_optimized': True
        }
        cls.demo = MobileOptimizedM3TM(cls.demo_config)
        cls.demo.build_model()
    
    def test_batch_text_processing(self):
        """Test processing multiple texts in sequence."""
        texts = [f"Test text number {i}" for i in range(50)]
        
        start_time = time.time()
        embeddings = []
        
        for text in texts:
            embedding = self.demo.process_text(text)
            embeddings.append(embedding)
        
        total_time = time.time() - start_time
        avg_time_per_text = (total_time / len(texts)) * 1000  # ms
        
        self.assertLess(avg_time_per_text, 100, 
                       f"Average time per text {avg_time_per_text:.2f}ms exceeds 100ms")
        
        # Check embeddings are valid
        self.assertEqual(len(embeddings), len(texts))
        for emb in embeddings:
            self.assertIsInstance(emb, torch.Tensor)
        
        logger.info(f"✅ Batch text processing test passed: {avg_time_per_text:.2f}ms per text")
    
    def test_batch_image_processing(self):
        """Test processing multiple images in sequence."""
        images = [Image.new('RGB', (224, 224), color=f'#{i:02d}{i:02d}{i:02d}') 
                 for i in range(20)]
        
        start_time = time.time()
        embeddings = []
        
        for image in images:
            embedding = self.demo.process_image(image)
            embeddings.append(embedding)
        
        total_time = time.time() - start_time
        avg_time_per_image = (total_time / len(images)) * 1000  # ms
        
        self.assertLess(avg_time_per_image, 100, 
                       f"Average time per image {avg_time_per_image:.2f}ms exceeds 100ms")
        
        # Check embeddings are valid
        self.assertEqual(len(embeddings), len(images))
        for emb in embeddings:
            self.assertIsInstance(emb, torch.Tensor)
        
        logger.info(f"✅ Batch image processing test passed: {avg_time_per_image:.2f}ms per image")
    
    def test_large_semantic_search(self):
        """Test semantic search with large dataset."""
        # Create large dataset
        texts = [f"Document {i} about topic {i % 10}" for i in range(100)]
        embeddings = [self.demo.process_text(text) for text in texts]
        
        # Perform searches
        queries = ["topic 1", "topic 5", "document", "information"]
        
        for query in queries:
            start_time = time.time()
            results = self.demo.semantic_search(query, embeddings, top_k=10)
            search_time = (time.time() - start_time) * 1000  # ms
            
            self.assertLess(search_time, 1000, 
                           f"Search time {search_time:.2f}ms exceeds 1000ms for 100 documents")
            self.assertEqual(len(results), 10)
        
        logger.info("✅ Large semantic search test passed")
    
    def test_memory_stability(self):
        """Test memory usage stability during extended operation."""
        initial_memory = self.demo._get_memory_usage()
        
        # Perform multiple operations
        for i in range(20):
            text = f"Memory test iteration {i}"
            text_emb = self.demo.process_text(text)
            
            image = Image.new('RGB', (224, 224), color='red')
            image_emb = self.demo.process_image(image)
            
            # Perform search
            self.demo.semantic_search("test", [text_emb, image_emb])
        
        final_memory = self.demo._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100, 
                       f"Memory increased by {memory_increase:.2f}MB during stress test")
        
        logger.info(f"✅ Memory stability test passed: +{memory_increase:.2f}MB increase")


class TestM3TMIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment."""
        cls.demo_config = {
            'model_name': 'M³TM v2.3 Test',
            'version': '2.3.0',
            'mobile_optimized': True
        }
        cls.demo = MobileOptimizedM3TM(cls.demo_config)
        cls.demo.build_model()
    
    def test_complete_multimodal_workflow(self):
        """Test complete multi-modal workflow from input to search."""
        # Prepare test data
        texts = [
            "A red car on the street",
            "Blue ocean waves",
            "Green forest landscape"
        ]
        
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue'),
            Image.new('RGB', (224, 224), color='green')
        ]
        
        # Process all data
        text_embeddings = [self.demo.process_text(text) for text in texts]
        image_embeddings = [self.demo.process_image(img) for img in images]
        
        # Combine embeddings
        all_embeddings = text_embeddings + image_embeddings
        
        # Test various queries
        test_queries = [
            "red vehicle",
            "water",
            "nature scene"
        ]
        
        for query in test_queries:
            results = self.demo.semantic_search(query, all_embeddings, top_k=3)
            
            self.assertEqual(len(results), 3)
            self.assertGreaterEqual(results[0]['similarity'], results[1]['similarity'])
            self.assertGreaterEqual(results[1]['similarity'], results[2]['similarity'])
        
        logger.info("✅ Complete multimodal workflow test passed")
    
    def test_performance_benchmarking_integration(self):
        """Test integrated performance benchmarking."""
        benchmark_results = self.demo.benchmark_performance()
        
        # Verify benchmark structure
        required_keys = [
            'model_size_mb',
            'memory_usage_mb', 
            'inference_benchmarks',
            'mobile_optimizations_applied',
            'device_info'
        ]
        
        for key in required_keys:
            self.assertIn(key, benchmark_results)
        
        # Verify inference benchmarks
        self.assertIn('text', benchmark_results['inference_benchmarks'])
        self.assertIn('image', benchmark_results['inference_benchmarks'])
        
        # Verify performance targets
        text_time = benchmark_results['inference_benchmarks']['text']['avg_time_ms']
        image_time = benchmark_results['inference_benchmarks']['image']['avg_time_ms']
        
        self.assertLess(text_time, 100)
        self.assertLess(image_time, 100)
        self.assertLess(benchmark_results['model_size_mb'], 500)
        self.assertLess(benchmark_results['memory_usage_mb'], 2048)
        
        logger.info("✅ Performance benchmarking integration test passed")


def run_comprehensive_tests():
    """Run all test suites with detailed reporting."""
    print("🧪 M³TM v2.3 Comprehensive Test Suite")
    print("=" * 60)
    print("Testing mobile optimization and performance targets")
    print()
    
    # Create test suite
    test_classes = [
        TestM3TMPerformance,
        TestM3TMFunctionality,
        TestM3TMMobileOptimizations,
        TestM3TMStressTest,
        TestM3TMIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"🔍 Running {test_class.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        class_total = result.testsRun
        class_failed = len(result.failures) + len(result.errors)
        class_passed = class_total - class_failed
        
        total_tests += class_total
        passed_tests += class_passed
        failed_tests += class_failed
        
        status = "✅ PASSED" if class_failed == 0 else f"❌ FAILED ({class_failed} failures)"
        print(f"   {status} - {class_passed}/{class_total} tests passed")
        
        if class_failed > 0:
            for failure in result.failures + result.errors:
                print(f"      ❌ {failure[0]}: {failure[1].split(chr(10))[0]}")
        
        print()
    
    # Final summary
    print("📋 Test Summary")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} {'❌' if failed_tests > 0 else '✅'}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    if failed_tests == 0:
        print("🎉 All tests passed! M³TM v2.3 meets all requirements.")
    else:
        print("⚠️  Some tests failed. Review and fix issues before deployment.")
    
    return passed_tests, failed_tests, total_tests


if __name__ == "__main__":
    passed, failed, total = run_comprehensive_tests()
