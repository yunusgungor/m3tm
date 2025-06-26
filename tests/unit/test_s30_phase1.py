"""
Tests for S30 Advanced Semantic Search & Large-Scale Indexing

Phase 1: FAISS Integration & Basic ANN Search Tests
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

# Import modules under test
from m3tm.search.index import (
    AdvancedSearchIndexConfig, 
    AdvancedSearchIndexFactory,
    AdvancedSearchIndex,
    SearchAnalytics,
    MemoryMonitor
)
from m3tm.search.strategies import (
    HNSWSearchStrategy,
    IVFPQSearchStrategy,
    AdaptiveSearchStrategy,
    SearchStrategyFactory
)


class TestAdvancedSearchIndexConfig:
    """Test enhanced configuration for advanced search."""
    
    def test_config_initialization_defaults(self):
        """Test default configuration values."""
        config = AdvancedSearchIndexConfig(embedding_dim=128)
        
        assert config.embedding_dim == 128
        assert config.ann_algorithm == "hnsw"
        assert config.hnsw_m == 16
        assert config.max_memory_usage_gb == 2.0
        assert config.enable_arm_optimization == True
        assert config.target_search_latency_ms == 50
    
    def test_config_validation_invalid_algorithm(self):
        """Test validation with invalid ANN algorithm."""
        with pytest.raises(ValueError, match="ann_algorithm 'invalid' invalid"):
            AdvancedSearchIndexConfig(
                embedding_dim=128,
                ann_algorithm="invalid"
            )
    
    def test_factory_string_generation_hnsw(self):
        """Test FAISS factory string generation for HNSW."""
        config = AdvancedSearchIndexConfig(
            embedding_dim=128,
            ann_algorithm="hnsw",
            hnsw_m=16
        )
        
        assert config.index_factory_string == "HNSW16"
    
    def test_factory_string_generation_ivf_pq(self):
        """Test FAISS factory string generation for IVF+PQ."""
        config = AdvancedSearchIndexConfig(
            embedding_dim=128,
            ann_algorithm="ivf_pq",
            ivf_nlist=1024,
            pq_m=8,
            enable_pq_compression=True
        )
        
        assert config.index_factory_string == "IVF1024,PQ8"
    
    def test_mobile_memory_warning(self, caplog):
        """Test warning for excessive memory usage."""
        config = AdvancedSearchIndexConfig(
            embedding_dim=128,
            max_memory_usage_gb=5.0  # Exceeds mobile recommendation
        )
        
        assert "exceeds mobile recommendations" in caplog.text


class TestAdvancedSearchIndexFactory:
    """Test factory for creating mobile-optimized indexes."""
    
    @patch('m3tm.search.index.faiss')
    def test_create_hnsw_index(self, mock_faiss):
        """Test HNSW index creation."""
        mock_faiss.IndexHNSWFlat.return_value = Mock()
        
        index = AdvancedSearchIndexFactory.create_hnsw_index(
            embedding_dim=128,
            m=16,
            ef_construction=200
        )
        
        assert isinstance(index, AdvancedSearchIndex)
        assert index.advanced_config.ann_algorithm == "hnsw"
        assert index.advanced_config.hnsw_m == 16
    
    @patch('m3tm.search.index.faiss')
    def test_create_ivf_pq_index(self, mock_faiss):
        """Test IVF+PQ index creation."""
        mock_faiss.index_factory.return_value = Mock()
        
        index = AdvancedSearchIndexFactory.create_ivf_pq_index(
            embedding_dim=128,
            nlist=1024,
            pq_m=8
        )
        
        assert isinstance(index, AdvancedSearchIndex)
        assert index.advanced_config.ann_algorithm == "ivf_pq"
        assert index.advanced_config.enable_pq_compression == True
    
    def test_get_recommended_config_small_dataset(self):
        """Test recommended configuration for small dataset."""
        config = AdvancedSearchIndexFactory.get_recommended_config_for_dataset_size(
            embedding_dim=128,
            dataset_size=5000
        )
        
        assert config.ann_algorithm == "flat"
    
    def test_get_recommended_config_medium_dataset(self):
        """Test recommended configuration for medium dataset."""
        config = AdvancedSearchIndexFactory.get_recommended_config_for_dataset_size(
            embedding_dim=128,
            dataset_size=50000
        )
        
        assert config.ann_algorithm == "hnsw"
        assert config.hnsw_m == 16
    
    def test_get_recommended_config_large_dataset(self):
        """Test recommended configuration for large dataset."""
        config = AdvancedSearchIndexFactory.get_recommended_config_for_dataset_size(
            embedding_dim=128,
            dataset_size=500000
        )
        
        assert config.ann_algorithm == "ivf_pq"
        assert config.enable_pq_compression == True
        assert config.enable_incremental_updates == True


class TestHNSWSearchStrategy:
    """Test HNSW search strategy implementation."""
    
    def test_strategy_initialization(self):
        """Test HNSW strategy initialization."""
        strategy = HNSWSearchStrategy(m=16, ef_construction=200, ef_search=50)
        
        assert strategy.m == 16
        assert strategy.ef_construction == 200
        assert strategy.ef_search == 50
        assert strategy.get_strategy_name() == "HNSW_M16_ef50"
    
    @patch('m3tm.search.strategies.faiss')
    def test_create_index(self, mock_faiss):
        """Test HNSW index creation."""
        mock_index = Mock()
        mock_index.hnsw = Mock()
        mock_faiss.IndexHNSWFlat.return_value = mock_index
        mock_faiss.METRIC_INNER_PRODUCT = 0
        mock_faiss.METRIC_L2 = 1
        
        strategy = HNSWSearchStrategy(m=16, ef_construction=200)
        config = {"metric_type": "ip"}
        
        index = strategy.create_index(embedding_dim=128, config=config)
        
        mock_faiss.IndexHNSWFlat.assert_called_once_with(128, 16, 0)
        assert index.hnsw.efConstruction == 200
    
    def test_search_query_reshaping(self):
        """Test search with 1D query reshaping."""
        strategy = HNSWSearchStrategy()
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
        
        query = np.array([0.1, 0.2, 0.3])  # 1D query
        distances, indices = strategy.search(mock_index, query, k=2)
        
        # Check that query was reshaped to 2D
        called_query = mock_index.search.call_args[0][0]
        assert called_query.shape == (1, 3)
    
    def test_update_search_parameters(self):
        """Test updating search-time parameters."""
        strategy = HNSWSearchStrategy(ef_search=50)
        strategy.update_search_parameters(ef_search=100)
        
        assert strategy.ef_search == 100
        assert "100" in strategy.get_strategy_name()


class TestIVFPQSearchStrategy:
    """Test IVF+PQ search strategy implementation."""
    
    def test_strategy_initialization(self):
        """Test IVF+PQ strategy initialization."""
        strategy = IVFPQSearchStrategy(nlist=1024, pq_m=8, pq_nbits=8, nprobe=16)
        
        assert strategy.nlist == 1024
        assert strategy.pq_m == 8
        assert strategy.pq_nbits == 8
        assert strategy.nprobe == 16
        assert strategy.get_strategy_name() == "IVF1024_PQ8x8_probe16"
    
    @patch('m3tm.search.strategies.faiss')
    def test_create_index(self, mock_faiss):
        """Test IVF+PQ index creation."""
        mock_quantizer = Mock()
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_quantizer
        mock_faiss.IndexIVFPQ.return_value = mock_index
        mock_faiss.METRIC_L2 = 1
        
        strategy = IVFPQSearchStrategy(nlist=1024, pq_m=8, pq_nbits=8, nprobe=16)
        config = {"metric_type": "l2"}
        
        index = strategy.create_index(embedding_dim=128, config=config)
        
        mock_faiss.IndexIVFPQ.assert_called_once_with(mock_quantizer, 128, 1024, 8, 8, 1)
        assert index.nprobe == 16


class TestAdaptiveSearchStrategy:
    """Test adaptive search strategy implementation."""
    
    def test_strategy_selection_small_dataset(self):
        """Test strategy selection for small dataset."""
        strategy = AdaptiveSearchStrategy()
        selected = strategy.select_strategy(
            dataset_size=5000,
            embedding_dim=128,
            memory_limit_gb=2.0
        )
        
        assert selected == "flat"
        assert strategy.selected_strategy == "flat"
    
    def test_strategy_selection_memory_constrained(self):
        """Test strategy selection for memory-constrained scenario."""
        strategy = AdaptiveSearchStrategy()
        selected = strategy.select_strategy(
            dataset_size=1000000,
            embedding_dim=128,
            memory_limit_gb=0.5  # Very limited memory
        )
        
        assert selected == "ivf_pq"
        assert strategy.selected_strategy == "ivf_pq"
    
    def test_strategy_selection_balanced(self):
        """Test strategy selection for balanced scenario."""
        strategy = AdaptiveSearchStrategy()
        selected = strategy.select_strategy(
            dataset_size=100000,
            embedding_dim=128,
            memory_limit_gb=2.0
        )
        
        assert selected == "hnsw"
        assert strategy.selected_strategy == "hnsw"


class TestSearchStrategyFactory:
    """Test search strategy factory."""
    
    def test_create_hnsw_strategy(self):
        """Test creating HNSW strategy."""
        strategy = SearchStrategyFactory.create_strategy("hnsw", m=16, ef_construction=200)
        
        assert isinstance(strategy, HNSWSearchStrategy)
        assert strategy.m == 16
        assert strategy.ef_construction == 200
    
    def test_create_ivf_pq_strategy(self):
        """Test creating IVF+PQ strategy."""
        strategy = SearchStrategyFactory.create_strategy("ivf_pq", nlist=1024, pq_m=8)
        
        assert isinstance(strategy, IVFPQSearchStrategy)
        assert strategy.nlist == 1024
        assert strategy.pq_m == 8
    
    def test_create_adaptive_strategy(self):
        """Test creating adaptive strategy."""
        strategy = SearchStrategyFactory.create_strategy("adaptive")
        
        assert isinstance(strategy, AdaptiveSearchStrategy)
    
    def test_create_unknown_strategy(self):
        """Test creating unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy type: unknown"):
            SearchStrategyFactory.create_strategy("unknown")
    
    def test_get_mobile_optimized_strategy(self):
        """Test getting mobile-optimized strategy."""
        strategy = SearchStrategyFactory.get_mobile_optimized_strategy(
            dataset_size=50000,
            memory_limit_gb=2.0
        )
        
        assert isinstance(strategy, AdaptiveSearchStrategy)
        assert strategy.selected_strategy is not None


class TestSearchAnalytics:
    """Test search analytics and monitoring."""
    
    def test_analytics_initialization(self):
        """Test analytics initialization."""
        analytics = SearchAnalytics()
        
        assert analytics.total_searches == 0
        assert len(analytics.search_history) == 0
    
    def test_record_search(self):
        """Test recording search analytics."""
        analytics = SearchAnalytics()
        search_data = {
            "search_time_ms": 25.5,
            "results_returned": 10,
            "meets_latency_target": True
        }
        
        analytics.record_search(search_data)
        
        assert analytics.total_searches == 1
        assert len(analytics.search_history) == 1
        assert analytics.search_history[0]["search_time_ms"] == 25.5
    
    def test_summary_stats(self):
        """Test summary statistics calculation."""
        analytics = SearchAnalytics()
        
        # Record multiple searches
        for i in range(5):
            analytics.record_search({
                "search_time_ms": 20.0 + i * 5,
                "meets_latency_target": i < 3  # First 3 meet target
            })
        
        stats = analytics.get_summary_stats()
        
        assert stats["total_searches"] == 5
        assert stats["avg_search_time_ms"] == 30.0  # (20+25+30+35+40)/5
        assert stats["searches_meeting_target"] == 3
        assert stats["target_meeting_rate"] == 0.6
    
    def test_history_pruning(self):
        """Test search history pruning for memory management."""
        analytics = SearchAnalytics()
        
        # Record more than 1000 searches to trigger pruning
        for i in range(1100):
            analytics.record_search({"search_time_ms": float(i)})
        
        assert analytics.total_searches == 1100
        assert len(analytics.search_history) == 500  # Pruned to 500


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(max_memory_gb=2.0)
        
        assert monitor.max_memory_gb == 2.0
        assert monitor.max_memory_bytes == 2.0 * 1024 * 1024 * 1024
    
    @patch('m3tm.search.index.psutil')
    def test_get_current_usage_with_psutil(self, mock_psutil):
        """Test memory usage calculation with psutil."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 500 * 1024 * 1024  # 500 MB
        mock_psutil.Process.return_value = mock_process
        
        monitor = MemoryMonitor(max_memory_gb=2.0)
        usage_mb = monitor.get_current_usage_mb()
        
        assert usage_mb == 500.0
    
    @patch('m3tm.search.index.psutil', side_effect=ImportError)
    def test_get_current_usage_without_psutil(self, mock_psutil):
        """Test memory usage calculation without psutil."""
        monitor = MemoryMonitor(max_memory_gb=2.0)
        usage_mb = monitor.get_current_usage_mb()
        
        assert usage_mb == 0.0
    
    def test_can_add_vectors_within_limit(self):
        """Test vector addition check within memory limit."""
        monitor = MemoryMonitor(max_memory_gb=2.0)
        
        # Mock current usage to 1GB
        with patch.object(monitor, 'get_current_usage_mb', return_value=1024.0):
            # Adding 1000 vectors of 128 dimensions (float32) = ~0.5MB
            can_add = monitor.can_add_vectors(num_vectors=1000, embedding_dim=128)
            assert can_add == True
    
    def test_can_add_vectors_exceeds_limit(self):
        """Test vector addition check exceeding memory limit."""
        monitor = MemoryMonitor(max_memory_gb=1.0)  # 1GB limit
        
        # Mock current usage to 900MB
        with patch.object(monitor, 'get_current_usage_mb', return_value=900.0):
            # Adding 100K vectors of 128 dimensions (float32) = ~50MB, total would exceed 1GB
            can_add = monitor.can_add_vectors(num_vectors=100000, embedding_dim=128)
            assert can_add == False


# Integration tests
class TestS30Phase1Integration:
    """Integration tests for S30 Phase 1 implementation."""
    
    @patch('m3tm.search.index.faiss')
    def test_end_to_end_advanced_search(self, mock_faiss):
        """Test end-to-end advanced search functionality."""
        # Mock FAISS components
        mock_index = Mock()
        mock_index.ntotal = 1000
        mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]), 
            np.array([[0, 1, 2]])
        )
        mock_faiss.index_factory.return_value = mock_index
        mock_faiss.METRIC_INNER_PRODUCT = 0
        
        # Create advanced index with HNSW
        config = AdvancedSearchIndexConfig(
            embedding_dim=128,
            ann_algorithm="hnsw",
            enable_search_analytics=True
        )
        
        index = AdvancedSearchIndex(config)
        
        # Perform search with analytics
        query = np.random.rand(128).astype(np.float32)
        distances, indices, metadata, analytics = index.search_with_analytics(query, k=3)
        
        # Verify results
        assert len(distances[0]) == 3
        assert len(indices[0]) == 3
        assert "search_time_ms" in analytics
        assert "meets_latency_target" in analytics
        assert analytics["algorithm"] == "hnsw"
    
    def test_mobile_optimization_recommendations(self):
        """Test that mobile optimization recommendations are properly applied."""
        # Test small dataset recommendation
        config_small = AdvancedSearchIndexFactory.get_recommended_config_for_dataset_size(
            embedding_dim=128,
            dataset_size=5000,
            memory_constraint_gb=1.0
        )
        assert config_small.ann_algorithm == "flat"
        
        # Test large dataset recommendation  
        config_large = AdvancedSearchIndexFactory.get_recommended_config_for_dataset_size(
            embedding_dim=128,
            dataset_size=1000000,
            memory_constraint_gb=2.0
        )
        assert config_large.ann_algorithm == "ivf_pq"
        assert config_large.enable_pq_compression == True
        assert config_large.enable_incremental_updates == True
