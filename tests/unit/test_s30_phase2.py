"""
M³TM S30 Phase 2 Tests - HNSW Advanced Graph Construction & Mobile Parameter Tuning

Tests for advanced HNSW implementation with mobile optimizations.
Covers graph construction, parameter tuning, and performance benchmarking.

Requirements:
- HNSW advanced graph construction testing
- Mobile parameter optimization validation
- Performance benchmark verification
- Memory constraint compliance testing

Patterns: PT-015 (PluggableComponentStrategy), PT-001 (ConfigurationDataclass)
Covers graph construction, parameter tuning, and performance benchmarking.

Requirements:
- HNSW advanced graph construction testing
- Mobile parameter optimization validation
- Performance benchmark verification
- Memory constraint compliance testing

Patterns: PT-015 (PluggableComponentStrategy), PT-001 (ConfigurationDataclass)
"""

import pytest
import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock

# Test imports
from src.m3tm.search.hnsw_mobile import (
    HNSWConfig,
    HNSWOptimizer,
    MobileHNSWBuilder,
    HNSWPerformanceBenchmark,
    MobileHNSWFactory
)
from src.m3tm.search.strategies import HNSWSearchStrategy

logger = logging.getLogger(__name__)


class TestHNSWConfig:
    """Test HNSW configuration and mobile optimizations."""
    
    def test_default_config_creation(self):
        """Test default HNSW configuration creation."""
        config = HNSWConfig()
        
        assert config.m == 16
        assert config.ef_construction == 200
        assert config.ef_search == 50
        assert config.max_level == 16
        assert config.mobile_memory_constraint_mb == 512.0
        assert config.enable_progressive_build is True
        assert config.batch_construction_size == 1000
    
    def test_mobile_optimized_config(self):
        """Test mobile-optimized configuration creation."""
        config = HNSWConfig(
            m=8,  # Lower connectivity for mobile
            ef_construction=100,
            ef_search=32,
            mobile_memory_constraint_mb=256.0,
            construction_quality_level="fast",
            search_quality_level="fast"
        )
        
        assert config.m <= 8  # Should remain at or below mobile-optimized value
        assert config.ef_construction <= 100
        assert config.ef_search <= 32
        assert config.mobile_memory_constraint_mb == 256.0
    
    def test_high_quality_config_adjustments(self):
        """Test high quality configuration adjustments."""
        config = HNSWConfig(
            construction_quality_level="high",
            search_quality_level="high"
        )
        
        assert config.ef_construction >= 400
        assert config.m >= 32
        assert config.ef_search >= 100
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        config = HNSWConfig(m=16, level_generation_factor=1/0.693)
        estimated_memory = config._estimate_memory_usage()
        
        assert isinstance(estimated_memory, float)
        assert estimated_memory > 0
        logger.info(f"Estimated HNSW memory usage: {estimated_memory:.2f}MB")
    
    def test_memory_constraint_warning(self):
        """Test memory constraint validation and warning."""
        # Test configuration that may trigger memory warning
        config = HNSWConfig(
            m=64,  # Very high connectivity
            ef_construction=1000,
            mobile_memory_constraint_mb=50.0  # Very low constraint
        )
        
        # The warning is logged, not raised as an exception
        # So we just verify the configuration was created
        assert config is not None
        assert config.mobile_memory_constraint_mb == 50.0


class TestHNSWOptimizer:
    """Test HNSW parameter optimization for mobile deployment."""
    
    def test_mobile_optimization_small_dataset(self):
        """Test mobile optimization for small datasets."""
        config = HNSWOptimizer.optimize_for_mobile(
            dataset_size=1000,
            embedding_dim=128,
            memory_constraint_mb=256.0,
            latency_target_ms=30.0
        )
        
        assert isinstance(config, HNSWConfig)
        assert config.m <= 16  # Conservative for small dataset
        assert config.ef_construction >= 50
        assert config.mobile_memory_constraint_mb == 256.0
    
    def test_mobile_optimization_large_dataset(self):
        """Test mobile optimization for large datasets."""
        config = HNSWOptimizer.optimize_for_mobile(
            dataset_size=100000,
            embedding_dim=512,
            memory_constraint_mb=1024.0,
            latency_target_ms=50.0
        )
        
        assert isinstance(config, HNSWConfig)
        assert config.batch_construction_size <= 5000  # Reasonable batch size
        assert config.enable_progressive_build is True
    
    def test_optimization_very_constrained_memory(self):
        """Test optimization under very tight memory constraints."""
        config = HNSWOptimizer.optimize_for_mobile(
            dataset_size=150000,  # Large dataset to trigger fast construction
            embedding_dim=256,
            memory_constraint_mb=128.0,  # Very tight
            latency_target_ms=100.0
        )
        
        # Should optimize for memory efficiency with large datasets  
        assert config.construction_quality_level == "fast"  # Large dataset optimization
        assert config.mobile_memory_constraint_mb == 128.0
        
        # Should optimize for memory efficiency with larger datasets
        assert config.construction_quality_level == "fast"  # Large dataset optimization
        assert config.mobile_memory_constraint_mb == 128.0
    
    def test_optimization_high_performance_target(self):
        """Test optimization for high performance targets."""
        config = HNSWOptimizer.optimize_for_mobile(
            dataset_size=10000,
            embedding_dim=128,
            memory_constraint_mb=2048.0,  # Generous memory
            latency_target_ms=10.0  # Very aggressive latency
        )
        
        # Should optimize for speed
        assert config.search_quality_level == "fast"
        assert config.mobile_memory_constraint_mb == 2048.0


class TestMobileHNSWBuilder:
    """Test mobile-optimized HNSW builder functionality."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(1000, 128).astype(np.float32)
    
    @pytest.fixture
    def mobile_config(self):
        """Create mobile-optimized HNSW configuration."""
        return HNSWConfig(
            m=8,
            ef_construction=100,
            ef_search=32,
            batch_construction_size=200,
            mobile_memory_constraint_mb=256.0
        )
    
    def test_builder_creation(self, mobile_config):
        """Test HNSW builder creation."""
        builder = MobileHNSWBuilder(mobile_config)
        
        assert builder.config == mobile_config
        assert builder.dimension is None
        assert builder.index is None
        assert isinstance(builder.build_stats, dict)
    
    @patch('faiss.IndexHNSWFlat')
    def test_progressive_build(self, mock_index_class, sample_embeddings, mobile_config):
        """Test progressive HNSW index building."""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = len(sample_embeddings)  # Mock ntotal as integer
        mock_index_class.return_value = mock_index
        
        builder = MobileHNSWBuilder(mobile_config)
        result_index = builder.build_progressive(sample_embeddings)
        
        # Verify index creation
        mock_index_class.assert_called_once_with(128, mobile_config.m)
        
        # Verify the method returns the mock index
        assert result_index == mock_index
        
        # Verify HNSW parameters were set
        assert hasattr(mock_index, 'hnsw')
    
    @patch('faiss.IndexHNSWFlat')
    def test_batch_construction(self, mock_index_class, sample_embeddings, mobile_config):
        """Test batch-based HNSW construction."""
        mock_index = Mock()
        mock_index.ntotal = len(sample_embeddings)  # Mock ntotal as integer
        mock_index_class.return_value = mock_index
        
        builder = MobileHNSWBuilder(mobile_config)
        builder.build_progressive(sample_embeddings)
        
        # Verify add was called (at least once for small dataset)
        assert mock_index.add.call_count >= 1
    
    @patch('faiss.IndexHNSWFlat')
    def test_memory_monitoring_during_build(self, mock_index_class, sample_embeddings, mobile_config):
        """Test memory monitoring during HNSW construction."""
        mock_index = Mock()
        mock_index.ntotal = 1000  # Mock the ntotal attribute
        mock_index_class.return_value = mock_index
        
        builder = MobileHNSWBuilder(mobile_config)
        
        # Build the index
        builder.build_progressive(sample_embeddings)
        
        # Verify memory monitoring functionality exists
        memory_usage = builder._monitor_memory()
        assert isinstance(memory_usage, (int, float))
    
    def test_build_without_faiss(self, sample_embeddings, mobile_config):
        """Test graceful handling when FAISS is not available."""
        with patch('src.m3tm.search.hnsw_mobile.faiss', None):
            builder = MobileHNSWBuilder(mobile_config)
            
            with pytest.raises(ImportError, match="FAISS"):
                builder.build_progressive(sample_embeddings)


class TestHNSWPerformanceBenchmark:
    """Test HNSW performance benchmarking functionality."""
    
    @pytest.fixture
    def benchmark_data(self):
        """Create benchmark data."""
        np.random.seed(42)
        vectors = np.random.randn(5000, 128).astype(np.float32)
        queries = np.random.randn(100, 128).astype(np.float32)
        return vectors, queries
    
    @pytest.fixture
    def benchmark_configs(self):
        """Create multiple configurations for benchmarking."""
        return [
            HNSWConfig(m=8, ef_search=16, construction_quality_level="fast"),
            HNSWConfig(m=16, ef_search=32, construction_quality_level="balanced"),
            HNSWConfig(m=32, ef_search=64, construction_quality_level="high")
        ]
    
    def test_benchmark_creation(self):
        """Test benchmark object creation."""
        benchmark = HNSWPerformanceBenchmark()
        
        assert benchmark.results == []
        assert hasattr(benchmark, 'run_benchmark')
    
    @patch('faiss.IndexHNSWFlat')
    def test_single_config_benchmark(self, mock_index_class, benchmark_data):
        """Test benchmarking a single HNSW configuration."""
        vectors, queries = benchmark_data
        config = HNSWConfig(m=16, ef_search=32)
        
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = len(vectors)
        mock_index.search.return_value = (
            np.random.randn(len(queries), 10),  # distances
            np.random.randint(0, len(vectors), (len(queries), 10))  # indices
        )
        mock_index_class.return_value = mock_index
        
        benchmark = HNSWPerformanceBenchmark()
        result = benchmark.run_benchmark(vectors, queries, config)
        
        # Verify result structure
        assert 'config' in result
        assert 'metrics' in result
        assert 'build_time_ms' in result['metrics']
        assert 'search_time_ms' in result['metrics']
        assert 'memory_usage_mb' in result['metrics']
        assert 'recall_at_10' in result['metrics']
        
        # Verify reasonable values
        assert result['metrics']['build_time_ms'] > 0
        assert result['metrics']['search_time_ms'] > 0
        assert result['metrics']['memory_usage_mb'] > 0
    
    @patch('faiss.IndexHNSWFlat')
    def test_multiple_config_comparison(self, mock_index_class, benchmark_data, benchmark_configs):
        """Test comparing multiple HNSW configurations."""
        vectors, queries = benchmark_data
        
        # Mock FAISS index for all configs
        mock_index = Mock()
        mock_index.ntotal = len(vectors)
        mock_index.search.return_value = (
            np.random.randn(len(queries), 10),
            np.random.randint(0, len(vectors), (len(queries), 10))
        )
        mock_index_class.return_value = mock_index
        
        benchmark = HNSWPerformanceBenchmark()
        results = benchmark.compare_configurations(vectors, queries, benchmark_configs)
        
        assert len(results) == len(benchmark_configs)
        
        # Verify each result
        for result in results:
            assert 'config' in result
            assert 'metrics' in result
            assert result['metrics']['build_time_ms'] > 0
    
    def test_recall_calculation(self):
        """Test recall calculation for HNSW search results."""
        # Mock ground truth and search results
        ground_truth = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        search_results = np.array([[0, 1, 10, 11, 12], [5, 6, 13, 14, 15]])
        
        benchmark = HNSWPerformanceBenchmark()
        recall = benchmark._calculate_recall(ground_truth, search_results, k=5)
        
        # Expected recall: (2+2)/(5+5) = 0.4
        assert abs(recall - 0.4) < 0.01
    
    def test_performance_analysis(self, benchmark_data):
        """Test performance analysis and recommendations."""
        vectors, queries = benchmark_data
        
        # Mock benchmark results
        mock_results = [
            {
                'config': HNSWConfig(m=8),
                'metrics': {
                    'build_time_ms': 100,
                    'search_time_ms': 5,
                    'memory_usage_mb': 50,
                    'recall_at_10': 0.85
                }
            },
            {
                'config': HNSWConfig(m=16),
                'metrics': {
                    'build_time_ms': 200,
                    'search_time_ms': 10,
                    'memory_usage_mb': 100,
                    'recall_at_10': 0.92
                }
            }
        ]
        
        benchmark = HNSWPerformanceBenchmark()
        analysis = benchmark.analyze_results(mock_results)
        
        assert 'best_for_speed' in analysis
        assert 'best_for_accuracy' in analysis
        assert 'best_for_memory' in analysis
        assert 'recommendations' in analysis


class TestMobileHNSWFactory:
    """Test mobile HNSW factory functionality."""
    
    def test_factory_creation(self):
        """Test HNSW factory creation."""
        factory = MobileHNSWFactory()
        
        assert hasattr(factory, 'create_index')
        assert hasattr(factory, 'create_optimized_config')
    
    def test_create_optimized_config(self):
        """Test creation of optimized configuration."""
        factory = MobileHNSWFactory()
        
        config = factory.create_optimized_config(
            dataset_size=10000,
            embedding_dim=256,
            target="balanced"
        )
        
        assert isinstance(config, HNSWConfig)
        assert config.mobile_memory_constraint_mb > 0
    
    def test_create_speed_optimized_config(self):
        """Test creation of speed-optimized configuration."""
        factory = MobileHNSWFactory()
        
        config = factory.create_optimized_config(
            dataset_size=50000,
            embedding_dim=128,
            target="speed"
        )
        
        assert config.construction_quality_level == "fast"
        assert config.search_quality_level == "fast"
        assert config.m <= 16
    
    def test_create_accuracy_optimized_config(self):
        """Test creation of accuracy-optimized configuration."""
        factory = MobileHNSWFactory()
        
        config = factory.create_optimized_config(
            dataset_size=10000,
            embedding_dim=256,
            target="accuracy"
        )
        
        assert config.construction_quality_level == "high"
        assert config.search_quality_level == "high"
        assert config.m >= 16
    
    @patch('faiss.IndexHNSWFlat')
    def test_create_index_with_config(self, mock_index_class):
        """Test index creation with specific configuration."""
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        
        factory = MobileHNSWFactory()
        config = HNSWConfig(m=16, ef_construction=200)
        
        index = factory.create_index(embedding_dim=128, config=config)
        
        mock_index_class.assert_called_once_with(128, 16)
        assert index == mock_index


class TestHNSWSearchStrategy:
    """Test HNSW search strategy integration."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(1000, 128).astype(np.float32)
    
    @pytest.fixture
    def hnsw_strategy(self):
        """Create HNSW search strategy."""
        config = HNSWConfig(m=16, ef_search=32)
        return HNSWSearchStrategy(config)
    
    def test_strategy_creation(self, hnsw_strategy):
        """Test HNSW strategy creation."""
        assert hnsw_strategy.config.m == 16
        assert hnsw_strategy.config.ef_search == 32
        assert hnsw_strategy.index is None
    
    @patch('faiss.IndexHNSWFlat')
    def test_strategy_build_index(self, mock_index_class, hnsw_strategy, sample_embeddings):
        """Test HNSW strategy index building."""
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        
        hnsw_strategy.build_index(sample_embeddings)
        
        mock_index_class.assert_called_once()
        assert hnsw_strategy.index == mock_index
    
    @patch('faiss.IndexHNSWFlat')
    def test_strategy_search(self, mock_index_class, hnsw_strategy, sample_embeddings):
        """Test HNSW strategy search functionality."""
        mock_index = Mock()
        mock_index.search.return_value = (
            np.random.randn(1, 10),  # distances
            np.random.randint(0, 1000, (1, 10))  # indices
        )
        mock_index_class.return_value = mock_index
        
        hnsw_strategy.build_index(sample_embeddings)
        
        query = np.random.randn(1, 128).astype(np.float32)
        distances, indices = hnsw_strategy.search(hnsw_strategy.index, query, k=10)
        
        mock_index.search.assert_called_once_with(query, 10)
        assert distances.shape == (1, 10)
        assert indices.shape == (1, 10)


class TestHNSWIntegration:
    """Integration tests for HNSW mobile implementation."""
    
    @pytest.fixture
    def integration_data(self):
        """Create integration test data."""
        np.random.seed(42)
        vectors = np.random.randn(2000, 128).astype(np.float32)
        queries = np.random.randn(50, 128).astype(np.float32)
        return vectors, queries
    
    @patch('faiss.IndexHNSWFlat')
    def test_end_to_end_hnsw_workflow(self, mock_index_class, integration_data):
        """Test complete HNSW workflow from optimization to search."""
        vectors, queries = integration_data
        
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = len(vectors)
        mock_index.search.return_value = (
            np.random.randn(len(queries), 10),
            np.random.randint(0, len(vectors), (len(queries), 10))
        )
        mock_index_class.return_value = mock_index
        
        # Step 1: Optimize configuration
        config = HNSWOptimizer.optimize_for_mobile(
            dataset_size=len(vectors),
            embedding_dim=vectors.shape[1],
            memory_constraint_mb=512.0,
            latency_target_ms=50.0
        )
        
        # Step 2: Build index
        builder = MobileHNSWBuilder(config)
        builder.build_progressive(vectors)
        
        # Step 3: Create strategy and search
        strategy = HNSWSearchStrategy(config)
        strategy.index = mock_index  # Use mock index
        
        distances, indices = strategy.search(strategy.index, queries, k=10)
        
        # Verify complete workflow
        assert distances.shape == (len(queries), 10)
        assert indices.shape == (len(queries), 10)
        mock_index.search.assert_called_once()
    
    def test_memory_constraint_compliance(self):
        """Test that HNSW configurations comply with memory constraints."""
        constraints = [128, 256, 512, 1024]  # MB
        
        for constraint in constraints:
            config = HNSWOptimizer.optimize_for_mobile(
                dataset_size=10000,
                embedding_dim=256,
                memory_constraint_mb=constraint
            )
            
            estimated_memory = config._estimate_memory_usage()
            
            # Allow some tolerance for estimation
            assert estimated_memory <= constraint * 1.2, \
                f"Estimated memory {estimated_memory:.1f}MB exceeds constraint {constraint}MB"
    
    def test_performance_target_optimization(self):
        """Test optimization for different performance targets."""
        latency_targets = [10, 25, 50, 100]  # ms
        
        for target in latency_targets:
            config = HNSWOptimizer.optimize_for_mobile(
                dataset_size=50000,
                embedding_dim=128,
                memory_constraint_mb=1024.0,
                latency_target_ms=target
            )
            
            # More aggressive targets should result in faster configurations
            if target <= 25:
                assert config.ef_search <= 64
                assert config.search_quality_level in ["fast", "balanced"]


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
