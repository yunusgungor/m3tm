"""
S30 Phase 4 Tests: Large-scale Optimization & Analytics
Tests for incremental updates, scalability optimization, and search analytics.
"""

import pytest
import torch
import numpy as np
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.m3tm.search.multimodal_fusion import (
    ScalabilityOptimizer,
    IncrementalUpdateManager, 
    SearchAnalytics,
    LargeScaleMultiModalSearch,
    MultiModalConfig,
    FusionStrategy
)


class TestScalabilityOptimizer:
    """Test large-scale dataset optimization"""
    
    def test_optimization_for_small_dataset(self):
        """Test optimization configuration for small datasets"""
        optimizer = ScalabilityOptimizer(max_memory_gb=2.0, chunk_size=10000)
        
        config = optimizer.optimize_for_dataset_size(1000)
        
        assert config["chunk_size"] == 100  # dataset_size // 10
        assert config["batch_size"] == 64
        assert config["index_type"] == "IVF"
        assert not config["memory_mapping"]
        assert config["parallel_workers"] == 1
        
    def test_optimization_for_large_dataset(self):
        """Test optimization configuration for large datasets"""
        optimizer = ScalabilityOptimizer(max_memory_gb=2.0, chunk_size=10000)
        
        config = optimizer.optimize_for_dataset_size(1000000)
        
        assert config["chunk_size"] == 10000
        assert config["batch_size"] == 32
        assert config["index_type"] == "HNSW" 
        assert config["memory_mapping"]
        assert config["parallel_workers"] == 4
        
    def test_memory_estimation(self):
        """Test memory usage estimation"""
        optimizer = ScalabilityOptimizer()
        
        memory_1k = optimizer._estimate_memory_usage(1000)
        memory_1m = optimizer._estimate_memory_usage(1000000)
        
        assert memory_1k < memory_1m
        assert memory_1k > 0
        assert memory_1m > 1.0  # Should be over 1GB for 1M items
        
    def test_memory_constraint_optimization(self):
        """Test optimization when memory constraints are exceeded"""
        optimizer = ScalabilityOptimizer(max_memory_gb=0.5)  # Very low limit
        
        config = optimizer.optimize_for_dataset_size(1000000)
        
        assert config.get("use_quantization", False)
        assert config.get("precision") == "fp16"
        assert config["chunk_size"] < 10000  # Reduced chunk size
        
    def test_optimization_stats_tracking(self):
        """Test that optimization statistics are tracked"""
        optimizer = ScalabilityOptimizer()
        
        config1 = optimizer.optimize_for_dataset_size(1000)
        config2 = optimizer.optimize_for_dataset_size(100000)
        
        stats = optimizer.optimization_stats["dataset_optimizations"]
        assert len(stats) == 2
        assert stats[0]["dataset_size"] == 1000
        assert stats[1]["dataset_size"] == 100000


class TestIncrementalUpdateManager:
    """Test incremental update system"""
    
    def test_add_items_below_threshold(self):
        """Test adding items below batch threshold"""
        manager = IncrementalUpdateManager(update_threshold=100)
        
        items = [{"id": f"item_{i}"} for i in range(10)]
        embeddings = torch.randn(10, 512)
        
        result = manager.add_items(items, embeddings)
        
        assert not result  # No batch update triggered
        assert len(manager.pending_updates) == 10
        
    def test_add_items_trigger_batch_update(self):
        """Test that batch update is triggered when threshold is reached"""
        manager = IncrementalUpdateManager(update_threshold=5)
        
        items = [{"id": f"item_{i}"} for i in range(10)]
        embeddings = torch.randn(10, 512)
        
        result = manager.add_items(items, embeddings)
        
        assert result  # Batch update should be triggered
        assert len(manager.pending_updates) == 0  # Should be cleared after update
        assert len(manager.update_history) == 1
        
    def test_remove_items(self):
        """Test item removal functionality"""
        manager = IncrementalUpdateManager(update_threshold=5)
        
        item_ids = [f"item_{i}" for i in range(3)]
        
        result = manager.remove_items(item_ids)
        
        assert not result  # Below threshold
        assert len(manager.pending_updates) == 3
        assert all(u["operation"] == "remove" for u in manager.pending_updates)
        
    def test_force_update(self):
        """Test forced update of pending changes"""
        manager = IncrementalUpdateManager(update_threshold=100)
        
        # Add some pending updates
        items = [{"id": f"item_{i}"} for i in range(5)]
        embeddings = torch.randn(5, 512)
        manager.add_items(items, embeddings)
        
        result = manager.force_update()
        
        assert result
        assert len(manager.pending_updates) == 0
        assert len(manager.update_history) == 1
        
    def test_update_history_tracking(self):
        """Test that update history is properly tracked"""
        manager = IncrementalUpdateManager(update_threshold=2)
        
        # Trigger multiple updates
        for i in range(3):
            items = [{"id": f"batch_{i}_item_{j}"} for j in range(2)]
            embeddings = torch.randn(2, 512)
            manager.add_items(items, embeddings)
            
        assert len(manager.update_history) == 3
        for record in manager.update_history:
            assert "timestamp" in record
            assert "adds_count" in record
            assert "processing_time" in record


class TestSearchAnalytics:
    """Test search analytics and monitoring"""
    
    def test_log_search_basic(self):
        """Test basic search logging"""
        analytics = SearchAnalytics(max_history=1000)
        
        analytics.log_search(
            query="test query",
            results_count=10,
            latency_ms=25.5,
            modalities=["text"],
            strategy="attention_fusion"
        )
        
        assert len(analytics.search_history) == 1
        search_record = analytics.search_history[0]
        assert search_record["query"] == "test query"
        assert search_record["latency_ms"] == 25.5
        assert not search_record["multimodal"]
        
    def test_multimodal_search_logging(self):
        """Test logging of multimodal searches"""
        analytics = SearchAnalytics()
        
        analytics.log_search(
            query="multimodal test",
            results_count=15,
            latency_ms=45.2,
            modalities=["text", "image"],
            strategy="late_fusion"
        )
        
        search_record = analytics.search_history[0]
        assert search_record["multimodal"]
        assert search_record["modalities"] == ["text", "image"]
        
    def test_performance_stats_calculation(self):
        """Test performance statistics calculation"""
        analytics = SearchAnalytics()
        
        # Log multiple searches with different latencies
        latencies = [10, 20, 30, 40, 50]
        for i, latency in enumerate(latencies):
            analytics.log_search(
                query=f"query_{i}",
                results_count=10,
                latency_ms=latency,
                modalities=["text"],
                strategy="attention_fusion"
            )
            
        stats = analytics.get_performance_stats()
        
        assert stats["total_searches"] == 5
        assert stats["average_latency_ms"] == 30.0
        assert stats["p95_latency_ms"] >= 45.0  # Allow for percentile calculation variance
        assert stats["multimodal_search_ratio"] == 0.0
        
    def test_query_pattern_tracking(self):
        """Test that query patterns are tracked correctly"""
        analytics = SearchAnalytics()
        
        # Log searches with different patterns
        analytics.log_search("q1", 10, 25, ["text"], "attention_fusion")
        analytics.log_search("q2", 10, 30, ["text"], "attention_fusion") 
        analytics.log_search("q3", 10, 35, ["text", "image"], "late_fusion")
        
        stats = analytics.get_performance_stats()
        patterns = stats["query_patterns"]
        
        assert patterns["text_attention_fusion"] == 2
        assert patterns["image+text_late_fusion"] == 1
        
    def test_performance_trend_calculation(self):
        """Test performance trend calculation"""
        analytics = SearchAnalytics()
        
        # Add enough data for trend calculation
        # First 50 searches with higher latency (older)
        for i in range(50):
            analytics.log_search(f"old_query_{i}", 10, 50.0, ["text"], "attention_fusion")
            
        # Next 50 searches with lower latency (recent)  
        for i in range(50):
            analytics.log_search(f"new_query_{i}", 10, 25.0, ["text"], "attention_fusion")
            
        stats = analytics.get_performance_stats()
        assert stats["performance_trend"] == "improving"
        
    def test_analytics_export(self):
        """Test analytics data export"""
        analytics = SearchAnalytics()
        
        # Add some search data
        analytics.log_search("test", 10, 25, ["text"], "attention_fusion")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
            
        try:
            result = analytics.export_analytics(temp_path)
            assert result
            assert temp_path.exists()
            
            # Verify exported data
            import json
            with open(temp_path) as f:
                data = json.load(f)
                
            assert "performance_stats" in data
            assert "recent_searches" in data
            assert "export_timestamp" in data
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestLargeScaleMultiModalSearch:
    """Test large-scale multi-modal search system"""
    
    def test_initialization_for_dataset(self):
        """Test system initialization optimized for dataset size"""
        config = MultiModalConfig()
        search_system = LargeScaleMultiModalSearch(config)
        
        result = search_system.initialize_for_dataset(100000)
        
        assert result["initialization_status"] == "success"
        assert "optimization_applied" in result
        assert "estimated_memory_gb" in result
        
    def test_multimodal_search_with_analytics(self):
        """Test multimodal search with analytics logging"""
        config = MultiModalConfig()
        search_system = LargeScaleMultiModalSearch(config)
        
        result = search_system.search_multimodal(
            text_query="test query",
            image_query=torch.randn(1, 3, 224, 224),
            top_k=5,
            strategy="attention_fusion"
        )
        
        assert "results" in result
        assert "latency_ms" in result
        assert result["modalities"] == ["text", "image"]
        assert result["strategy_used"] == "attention_fusion"
        
        # Check analytics were logged
        assert len(search_system.analytics.search_history) == 1
        
    def test_incremental_item_addition(self):
        """Test incremental item addition"""
        config = MultiModalConfig(incremental_update_threshold=2)
        search_system = LargeScaleMultiModalSearch(config)
        
        items = [{"id": "item1"}, {"id": "item2"}, {"id": "item3"}]
        embeddings = torch.randn(3, 512)
        
        result = search_system.add_items_incremental(items, embeddings)
        
        assert result["items_added"] == 3
        assert result["batch_update_triggered"]  # Should trigger due to low threshold
        
    def test_system_analytics_comprehensive(self):
        """Test comprehensive system analytics"""
        config = MultiModalConfig()
        search_system = LargeScaleMultiModalSearch(config)
        
        # Perform some operations to generate analytics data
        search_system.search_multimodal(text_query="test", top_k=5)
        search_system.add_items_incremental([{"id": "test"}], torch.randn(1, 512))
        
        analytics = search_system.get_system_analytics()
        
        assert "search_analytics" in analytics
        assert "optimization_stats" in analytics  
        assert "update_history" in analytics
        assert "system_status" in analytics
        
        system_status = analytics["system_status"]
        assert "pending_updates" in system_status
        assert "total_searches" in system_status
        
    def test_text_only_search(self):
        """Test text-only search functionality"""
        config = MultiModalConfig()
        search_system = LargeScaleMultiModalSearch(config)
        
        result = search_system.search_multimodal(
            text_query="text only query",
            top_k=10
        )
        
        assert result["modalities"] == ["text"]
        assert len(result["results"]) <= 10
        
    def test_image_only_search(self):
        """Test image-only search functionality"""
        config = MultiModalConfig()
        search_system = LargeScaleMultiModalSearch(config)
        
        result = search_system.search_multimodal(
            image_query=torch.randn(1, 3, 224, 224),
            top_k=5
        )
        
        assert result["modalities"] == ["image"]
        assert len(result["results"]) <= 5


@pytest.mark.performance
class TestPhase4Performance:
    """Performance tests for Phase 4 components"""
    
    def test_large_dataset_optimization_performance(self):
        """Test optimization performance for large datasets"""
        optimizer = ScalabilityOptimizer()
        
        start_time = time.time()
        config = optimizer.optimize_for_dataset_size(1000000)
        optimization_time = time.time() - start_time
        
        # Optimization should be fast
        assert optimization_time < 0.1  # Less than 100ms
        assert config["index_type"] == "HNSW"
        
    def test_incremental_update_performance(self):
        """Test incremental update performance"""
        manager = IncrementalUpdateManager(update_threshold=1000)
        
        # Add many items
        start_time = time.time()
        for batch in range(10):
            items = [{"id": f"batch_{batch}_item_{i}"} for i in range(100)]
            embeddings = torch.randn(100, 512)
            manager.add_items(items, embeddings)
            
        total_time = time.time() - start_time
        
        # Should complete quickly
        assert total_time < 1.0  # Less than 1 second for 1000 items
        
    def test_analytics_logging_performance(self):
        """Test analytics logging performance under load"""
        analytics = SearchAnalytics(max_history=10000)
        
        start_time = time.time()
        for i in range(1000):
            analytics.log_search(
                query=f"query_{i}",
                results_count=10,
                latency_ms=25.0,
                modalities=["text"],
                strategy="attention_fusion"
            )
            
        logging_time = time.time() - start_time
        
        # Should handle 1000 logs quickly
        assert logging_time < 0.5  # Less than 500ms
        assert len(analytics.search_history) == 1000


@pytest.mark.integration  
class TestPhase4Integration:
    """Integration tests for Phase 4 with previous phases"""
    
    def test_phase4_with_fusion_strategies(self):
        """Test Phase 4 components work with all fusion strategies"""
        strategies = [
            "early_fusion",
            "late_fusion", 
            "attention_fusion",
            "contrastive_fusion"
        ]
        
        config = MultiModalConfig()
        search_system = LargeScaleMultiModalSearch(config)
        
        for strategy in strategies:
            result = search_system.search_multimodal(
                text_query="test",
                image_query=torch.randn(1, 3, 224, 224),
                strategy=strategy
            )
            
            assert result["strategy_used"] == strategy
            assert "latency_ms" in result
            
        # Check all strategies were logged
        patterns = search_system.analytics.get_performance_stats()["query_patterns"]
        assert len(patterns) == len(strategies)
        
    def test_phase4_scalability_with_multimodal_config(self):
        """Test Phase 4 scalability optimization with multimodal config"""
        config = MultiModalConfig(
            max_memory_gb=1.0,
            chunk_size=5000,
            incremental_update_threshold=500
        )
        
        search_system = LargeScaleMultiModalSearch(config)
        
        # Initialize for large dataset
        result = search_system.initialize_for_dataset(500000)
        
        assert result["initialization_status"] == "success"
        opt_config = result["optimization_applied"]
        assert opt_config["chunk_size"] <= config.chunk_size
        assert opt_config["index_type"] == "HNSW"  # Large dataset should use HNSW
        
    def test_end_to_end_phase4_workflow(self):
        """Test complete Phase 4 workflow: optimization -> search -> analytics"""
        config = MultiModalConfig(incremental_update_threshold=2)
        search_system = LargeScaleMultiModalSearch(config)
        
        # 1. Initialize for dataset
        init_result = search_system.initialize_for_dataset(10000)
        assert init_result["initialization_status"] == "success"
        
        # 2. Add items incrementally  
        items = [{"id": f"item_{i}"} for i in range(5)]
        embeddings = torch.randn(5, 512)
        add_result = search_system.add_items_incremental(items, embeddings)
        assert add_result["batch_update_triggered"]
        
        # 3. Perform searches
        for i in range(3):
            search_result = search_system.search_multimodal(
                text_query=f"query_{i}",
                top_k=5
            )
            assert len(search_result["results"]) > 0
            
        # 4. Get comprehensive analytics
        analytics = search_system.get_system_analytics()
        assert analytics["system_status"]["total_searches"] == 3
        assert len(analytics["update_history"]) > 0
        
        # 5. Verify performance
        search_stats = analytics["search_analytics"]
        assert search_stats["total_searches"] == 3
        assert search_stats["average_latency_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
