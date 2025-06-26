"""
M³TM Advanced Indexing System

Advanced indexing capabilities including multi-index management,
index optimization, and mobile-specific indexing strategies.

Patterns: PT-003 (ModelComposite), PT-015 (PluggableComponentStrategy), PT-002 (FactoryMethod)
"""

from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
import json
from pathlib import Path
import numpy as np
import torch

try:
    import faiss
except ImportError:
    faiss = None

from .hnsw_mobile import HNSWConfig, MobileHNSWBuilder, HNSWOptimizer
from .strategies import SearchStrategy, SearchStrategyFactory

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Index types supported by the advanced indexing system."""
    FLAT = "flat"
    HNSW = "hnsw"
    IVF_FLAT = "ivf_flat"
    IVF_PQ = "ivf_pq"
    MULTI_INDEX = "multi_index"
    ADAPTIVE = "adaptive"


@dataclass
class IndexPerformanceMetrics:
    """Performance metrics for index evaluation."""
    build_time_ms: float = 0.0
    search_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    accuracy_recall_at_10: float = 0.0
    throughput_qps: float = 0.0
    index_size_mb: float = 0.0
    last_updated: str = ""


@dataclass
class AdvancedIndexConfig:
    """Configuration for advanced indexing system."""
    # Core configuration
    embedding_dim: int = 128
    expected_dataset_size: int = 100000
    memory_budget_mb: float = 512.0
    
    # Performance targets
    target_latency_ms: float = 50.0
    target_recall: float = 0.95
    target_throughput_qps: float = 100.0
    
    # Index selection strategy
    auto_index_selection: bool = True
    fallback_strategy: str = "hnsw"
    enable_multi_index: bool = False
    
    # Mobile optimizations
    mobile_tier: str = "mid"  # low, mid, high
    enable_compression: bool = True
    enable_quantization: bool = True
    
    # Monitoring and adaptation
    enable_performance_monitoring: bool = True
    adaptation_threshold_degradation: float = 0.1  # 10% performance degradation
    reindex_frequency_hours: int = 24
    
    # Quality vs speed tradeoffs
    optimization_objective: str = "balanced"  # speed, accuracy, memory, balanced


class IndexSelector:
    """
    Intelligent index selector for mobile deployment.
    
    Selects optimal index type and configuration based on dataset
    characteristics, performance requirements, and mobile constraints.
    """
    
    def __init__(self, config: AdvancedIndexConfig):
        """Initialize index selector."""
        self.config = config
        self.performance_history = {}
        
    def select_optimal_index(self, 
                           dataset_characteristics: Dict[str, Any]) -> Tuple[IndexType, Dict[str, Any]]:
        """
        Select optimal index type and configuration.
        
        Args:
            dataset_characteristics: Information about the dataset
            
        Returns:
            Tuple of (IndexType, configuration dict)
        """
        dataset_size = dataset_characteristics.get("size", self.config.expected_dataset_size)
        dimensionality = dataset_characteristics.get("dim", self.config.embedding_dim)
        data_distribution = dataset_characteristics.get("distribution", "uniform")
        
        # Decision matrix based on dataset characteristics
        if dataset_size < 1000:
            # Very small dataset - use flat for accuracy
            return IndexType.FLAT, self._get_flat_config()
            
        elif dataset_size < 10000:
            # Small dataset - HNSW with high accuracy settings
            return IndexType.HNSW, self._get_hnsw_config("high_accuracy")
            
        elif dataset_size < 100000:
            # Medium dataset - balanced HNSW
            return IndexType.HNSW, self._get_hnsw_config("balanced")
            
        elif self.config.memory_budget_mb < 256:
            # Memory constrained - use IVF+PQ
            return IndexType.IVF_PQ, self._get_ivf_pq_config("memory_optimized")
            
        elif self.config.target_latency_ms < 25:
            # Latency critical - fast HNSW
            return IndexType.HNSW, self._get_hnsw_config("speed_optimized")
            
        else:
            # Large dataset - adaptive strategy
            return IndexType.ADAPTIVE, self._get_adaptive_config()
    
    def _get_flat_config(self) -> Dict[str, Any]:
        """Get configuration for flat index."""
        return {
            "metric_type": "ip",
            "enable_gpu": False
        }
    
    def _get_hnsw_config(self, optimization_mode: str) -> Dict[str, Any]:
        """Get HNSW configuration based on optimization mode."""
        base_config = {
            "metric_type": "ip",
            "enable_gpu": False,
            "mobile_tier": self.config.mobile_tier
        }
        
        if optimization_mode == "speed_optimized":
            base_config.update({
                "m": 8,
                "ef_construction": 100,
                "ef_search": 16,
                "construction_quality_level": "fast"
            })
        elif optimization_mode == "high_accuracy":
            base_config.update({
                "m": 32,
                "ef_construction": 400,
                "ef_search": 100,
                "construction_quality_level": "high"
            })
        else:  # balanced
            base_config.update({
                "m": 16,
                "ef_construction": 200,
                "ef_search": 50,
                "construction_quality_level": "balanced"
            })
        
        return base_config
    
    def _get_ivf_pq_config(self, optimization_mode: str) -> Dict[str, Any]:
        """Get IVF+PQ configuration based on optimization mode."""
        dataset_size = self.config.expected_dataset_size
        
        # Calculate appropriate nlist based on dataset size
        nlist = min(int(np.sqrt(dataset_size)), 4096)
        
        base_config = {
            "metric_type": "ip",
            "enable_gpu": False,
            "nlist": nlist,
            "nprobe": 16
        }
        
        if optimization_mode == "memory_optimized":
            base_config.update({
                "pq_m": 8,
                "pq_nbits": 8,
                "enable_compression": True
            })
        else:
            base_config.update({
                "pq_m": 16,
                "pq_nbits": 8,
                "enable_compression": False
            })
        
        return base_config
    
    def _get_adaptive_config(self) -> Dict[str, Any]:
        """Get adaptive configuration."""
        return {
            "primary_strategy": "hnsw",
            "fallback_strategy": "ivf_pq",
            "adaptation_enabled": True,
            "performance_monitoring": True
        }


class MultiIndexManager:
    """
    Manager for multiple index instances with load balancing and failover.
    
    Supports hybrid indexing strategies and automatic index selection
    based on query characteristics.
    """
    
    def __init__(self, config: AdvancedIndexConfig):
        """Initialize multi-index manager."""
        self.config = config
        self.indexes = {}
        self.index_metadata = {}
        self.query_router = QueryRouter()
        self.performance_monitor = IndexPerformanceMonitor()
        self._lock = threading.RLock()
        
    def add_index(self, 
                  index_id: str, 
                  index: Any, 
                  index_type: IndexType,
                  specialization: Optional[str] = None):
        """
        Add an index to the multi-index system.
        
        Args:
            index_id: Unique identifier for the index
            index: The index instance
            index_type: Type of the index
            specialization: Optional specialization (e.g., "text", "image")
        """
        with self._lock:
            self.indexes[index_id] = index
            self.index_metadata[index_id] = {
                "type": index_type,
                "specialization": specialization,
                "created_at": time.time(),
                "last_used": time.time(),
                "query_count": 0,
                "total_latency_ms": 0.0
            }
            
        logger.info(f"Added index {index_id} (type: {index_type.value})")
    
    def search_multi_index(self, 
                          query_embedding: np.ndarray,
                          k: int = 10,
                          query_context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Search across multiple indexes with intelligent routing.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            query_context: Additional context for routing decisions
            
        Returns:
            Tuple of (distances, indices, search_metadata)
        """
        if not self.indexes:
            raise ValueError("No indexes available for search")
        
        # Route query to appropriate index
        selected_index_id = self.query_router.route_query(
            query_embedding, query_context, self.index_metadata
        )
        
        # Perform search
        start_time = time.time()
        index = self.indexes[selected_index_id]
        
        if hasattr(index, 'search'):
            distances, indices = index.search(
                query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding, 
                k
            )
        else:
            raise ValueError(f"Index {selected_index_id} does not support search")
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Update metadata
        with self._lock:
            metadata = self.index_metadata[selected_index_id]
            metadata["last_used"] = time.time()
            metadata["query_count"] += 1
            metadata["total_latency_ms"] += search_time_ms
        
        # Record performance
        self.performance_monitor.record_search(selected_index_id, search_time_ms, len(indices[0]))
        
        search_metadata = {
            "selected_index": selected_index_id,
            "search_time_ms": search_time_ms,
            "index_type": self.index_metadata[selected_index_id]["type"].value
        }
        
        return distances, indices, search_metadata
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics for all managed indexes."""
        stats = {}
        
        with self._lock:
            for index_id, metadata in self.index_metadata.items():
                index = self.indexes[index_id]
                
                avg_latency = (
                    metadata["total_latency_ms"] / metadata["query_count"]
                    if metadata["query_count"] > 0 else 0.0
                )
                
                stats[index_id] = {
                    "type": metadata["type"].value,
                    "specialization": metadata["specialization"],
                    "query_count": metadata["query_count"],
                    "avg_latency_ms": avg_latency,
                    "size": getattr(index, 'ntotal', 0),
                    "last_used": metadata["last_used"]
                }
        
        return stats
    
    def optimize_indexes(self):
        """Optimize all indexes based on usage patterns."""
        stats = self.get_index_statistics()
        
        for index_id, stat in stats.items():
            if stat["avg_latency_ms"] > self.config.target_latency_ms * 1.2:
                logger.warning(f"Index {index_id} performance degraded: "
                             f"{stat['avg_latency_ms']:.1f}ms avg latency")
                # Could trigger reindexing or parameter tuning
        
        logger.info("Index optimization completed")


class QueryRouter:
    """
    Intelligent query router for multi-index systems.
    
    Routes queries to the most appropriate index based on query
    characteristics and index specializations.
    """
    
    def route_query(self, 
                   query_embedding: np.ndarray,
                   query_context: Optional[Dict[str, Any]],
                   index_metadata: Dict[str, Any]) -> str:
        """
        Route query to the most appropriate index.
        
        Args:
            query_embedding: Query vector
            query_context: Additional context (modality, domain, etc.)
            index_metadata: Metadata about available indexes
            
        Returns:
            Selected index ID
        """
        if not index_metadata:
            raise ValueError("No indexes available")
        
        # Simple routing strategy: prefer specialized indexes
        if query_context and "modality" in query_context:
            modality = query_context["modality"]
            
            # Look for specialized index
            for index_id, metadata in index_metadata.items():
                if metadata.get("specialization") == modality:
                    logger.debug(f"Routed to specialized index {index_id} for {modality}")
                    return index_id
        
        # Fallback to index with best recent performance
        best_index_id = min(
            index_metadata.keys(),
            key=lambda idx: (
                index_metadata[idx]["total_latency_ms"] / 
                max(index_metadata[idx]["query_count"], 1)
            )
        )
        
        logger.debug(f"Routed to best performing index {best_index_id}")
        return best_index_id


class IndexPerformanceMonitor:
    """
    Performance monitoring system for indexes.
    
    Tracks search latency, throughput, accuracy, and memory usage
    across different index configurations.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.history = {}
        
    def record_search(self, index_id: str, latency_ms: float, results_count: int):
        """Record search performance metrics."""
        if index_id not in self.metrics:
            self.metrics[index_id] = IndexPerformanceMetrics()
            self.history[index_id] = []
        
        # Update metrics
        metrics = self.metrics[index_id]
        metrics.search_time_ms = latency_ms
        metrics.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Update history for trend analysis
        self.history[index_id].append({
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "results_count": results_count
        })
        
        # Keep only recent history (last 1000 searches)
        if len(self.history[index_id]) > 1000:
            self.history[index_id] = self.history[index_id][-500:]
    
    def get_performance_summary(self, index_id: str) -> Dict[str, Any]:
        """Get performance summary for an index."""
        if index_id not in self.metrics:
            return {}
        
        metrics = self.metrics[index_id]
        history = self.history.get(index_id, [])
        
        if not history:
            return {"index_id": index_id, "status": "no_data"}
        
        recent_latencies = [h["latency_ms"] for h in history[-100:]]  # Last 100 searches
        
        return {
            "index_id": index_id,
            "current_latency_ms": metrics.search_time_ms,
            "avg_latency_ms": np.mean(recent_latencies),
            "p95_latency_ms": np.percentile(recent_latencies, 95),
            "min_latency_ms": min(recent_latencies),
            "max_latency_ms": max(recent_latencies),
            "total_searches": len(history),
            "last_updated": metrics.last_updated
        }


class AdvancedIndexingSystem:
    """
    Advanced indexing system with mobile optimization and intelligent management.
    
    Main interface for the S30 advanced indexing functionality.
    Combines multiple indexing strategies with performance monitoring
    and automatic optimization.
    """
    
    def __init__(self, config: AdvancedIndexConfig):
        """Initialize advanced indexing system."""
        self.config = config
        self.index_selector = IndexSelector(config)
        self.multi_index_manager = MultiIndexManager(config)
        self.performance_monitor = IndexPerformanceMonitor()
        
        # System state
        self.is_initialized = False
        self.primary_index = None
        self.fallback_index = None
        
        logger.info("Advanced indexing system initialized")
    
    def initialize_system(self, 
                         dataset_characteristics: Dict[str, Any],
                         initial_vectors: Optional[np.ndarray] = None):
        """
        Initialize the indexing system with optimal configuration.
        
        Args:
            dataset_characteristics: Information about the dataset
            initial_vectors: Optional initial vectors to index
        """
        # Select optimal index configuration
        index_type, index_config = self.index_selector.select_optimal_index(dataset_characteristics)
        
        logger.info(f"Selected index type: {index_type.value}")
        
        # Create primary index
        if index_type == IndexType.HNSW:
            self.primary_index = self._create_hnsw_index(index_config, initial_vectors)
        elif index_type == IndexType.IVF_PQ:
            self.primary_index = self._create_ivf_pq_index(index_config, initial_vectors)
        elif index_type == IndexType.FLAT:
            self.primary_index = self._create_flat_index(index_config, initial_vectors)
        elif index_type == IndexType.ADAPTIVE:
            self.primary_index = self._create_adaptive_index(index_config, initial_vectors)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add to multi-index manager
        self.multi_index_manager.add_index("primary", self.primary_index, index_type)
        
        # Create fallback index if needed
        if self.config.fallback_strategy and self.config.fallback_strategy != index_type.value:
            fallback_type = IndexType(self.config.fallback_strategy)
            fallback_config = self._get_fallback_config(fallback_type)
            
            if fallback_type == IndexType.HNSW:
                self.fallback_index = self._create_hnsw_index(fallback_config, initial_vectors)
            else:
                self.fallback_index = self._create_flat_index(fallback_config, initial_vectors)
            
            self.multi_index_manager.add_index("fallback", self.fallback_index, fallback_type)
        
        self.is_initialized = True
        logger.info("Advanced indexing system initialization completed")
    
    def search(self, 
              query_embedding: Union[torch.Tensor, np.ndarray],
              k: int = 10,
              query_context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Perform intelligent search across the indexing system.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            query_context: Additional search context
            
        Returns:
            Tuple of (distances, indices, search_metadata)
        """
        if not self.is_initialized:
            raise RuntimeError("Indexing system not initialized")
        
        # Convert to numpy if needed
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Perform multi-index search
        try:
            distances, indices, metadata = self.multi_index_manager.search_multi_index(
                query_embedding, k, query_context
            )
            
            # Add system-level metadata
            metadata.update({
                "system_latency_ms": metadata["search_time_ms"],
                "meets_target_latency": metadata["search_time_ms"] <= self.config.target_latency_ms,
                "system_status": "healthy"
            })
            
            return distances, indices, metadata
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            
            # Fallback to primary index if multi-index fails
            if self.primary_index:
                start_time = time.time()
                distances, indices = self.primary_index.search(
                    query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding,
                    k
                )
                search_time_ms = (time.time() - start_time) * 1000
                
                metadata = {
                    "selected_index": "primary_fallback",
                    "search_time_ms": search_time_ms,
                    "system_status": "degraded",
                    "error": str(e)
                }
                
                return distances, indices, metadata
            
            raise
    
    def add_vectors(self, 
                   vectors: Union[torch.Tensor, np.ndarray],
                   metadata: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add vectors to the indexing system.
        
        Args:
            vectors: Vectors to add
            metadata: Optional metadata for each vector
            
        Returns:
            List of assigned IDs
        """
        if not self.is_initialized:
            raise RuntimeError("Indexing system not initialized")
        
        # Convert to numpy if needed
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.cpu().numpy()
        
        # Add to primary index
        if hasattr(self.primary_index, 'add'):
            start_size = getattr(self.primary_index, 'ntotal', 0)
            self.primary_index.add(vectors)
            end_size = getattr(self.primary_index, 'ntotal', 0)
            
            # Generate IDs
            new_ids = list(range(start_size, end_size))
            
            logger.info(f"Added {len(vectors)} vectors to primary index")
            return new_ids
        
        raise RuntimeError("Primary index does not support adding vectors")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "initialized": self.is_initialized,
            "config": {
                "target_latency_ms": self.config.target_latency_ms,
                "target_recall": self.config.target_recall,
                "memory_budget_mb": self.config.memory_budget_mb
            }
        }
        
        if self.is_initialized:
            status.update({
                "index_statistics": self.multi_index_manager.get_index_statistics(),
                "performance_summary": {
                    idx: self.performance_monitor.get_performance_summary(idx)
                    for idx in ["primary", "fallback"] 
                    if idx in self.multi_index_manager.indexes
                }
            })
        
        return status
    
    # Helper methods for creating different index types
    def _create_hnsw_index(self, config: Dict[str, Any], vectors: Optional[np.ndarray]) -> Any:
        """Create HNSW index with mobile optimization."""
        if faiss is None:
            raise ImportError("FAISS not available")
        
        hnsw_config = HNSWConfig(
            m=config.get("m", 16),
            ef_construction=config.get("ef_construction", 200),
            ef_search=config.get("ef_search", 50),
            mobile_memory_constraint_mb=self.config.memory_budget_mb
        )
        
        builder = MobileHNSWBuilder(hnsw_config)
        
        metric_type = config.get("metric_type", "ip")
        index = builder.build_index(self.config.embedding_dim, vectors, metric_type)
        
        return index
    
    def _create_ivf_pq_index(self, config: Dict[str, Any], vectors: Optional[np.ndarray]) -> Any:
        """Create IVF+PQ index."""
        if faiss is None:
            raise ImportError("FAISS not available")
        
        metric = faiss.METRIC_INNER_PRODUCT if config.get("metric_type") == "ip" else faiss.METRIC_L2
        
        # Create quantizer
        quantizer = faiss.IndexFlatIP(self.config.embedding_dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.config.embedding_dim)
        
        # Create IVF+PQ index
        nlist = config.get("nlist", 1024)
        pq_m = config.get("pq_m", 8)
        pq_nbits = config.get("pq_nbits", 8)
        
        index = faiss.IndexIVFPQ(quantizer, self.config.embedding_dim, nlist, pq_m, pq_nbits, metric)
        index.nprobe = config.get("nprobe", 16)
        
        # Train and add vectors if provided
        if vectors is not None and len(vectors) > 0:
            index.train(vectors)
            index.add(vectors)
        
        return index
    
    def _create_flat_index(self, config: Dict[str, Any], vectors: Optional[np.ndarray]) -> Any:
        """Create flat index."""
        if faiss is None:
            raise ImportError("FAISS not available")
        
        metric = faiss.METRIC_INNER_PRODUCT if config.get("metric_type") == "ip" else faiss.METRIC_L2
        
        if metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(self.config.embedding_dim)
        else:
            index = faiss.IndexFlatL2(self.config.embedding_dim)
        
        # Add vectors if provided
        if vectors is not None and len(vectors) > 0:
            index.add(vectors)
        
        return index
    
    def _create_adaptive_index(self, config: Dict[str, Any], vectors: Optional[np.ndarray]) -> Any:
        """Create adaptive index (currently delegates to HNSW)."""
        # For now, adaptive index uses HNSW with balanced settings
        hnsw_config = {
            "m": 16,
            "ef_construction": 200,
            "ef_search": 50,
            "metric_type": "ip"
        }
        return self._create_hnsw_index(hnsw_config, vectors)
    
    def _get_fallback_config(self, fallback_type: IndexType) -> Dict[str, Any]:
        """Get configuration for fallback index."""
        if fallback_type == IndexType.HNSW:
            return {
                "m": 8,
                "ef_construction": 100,
                "ef_search": 32,
                "metric_type": "ip"
            }
        else:  # FLAT
            return {
                "metric_type": "ip"
            }


# Factory for creating advanced indexing components
class AdvancedIndexingFactory:
    """
    Factory for creating advanced indexing system components.
    
    Pattern: PT-002 (FactoryMethod)
    """
    
    @staticmethod
    def create_system(config: AdvancedIndexConfig) -> AdvancedIndexingSystem:
        """Create advanced indexing system."""
        return AdvancedIndexingSystem(config)
    
    @staticmethod
    def create_mobile_optimized_system(dataset_size: int,
                                     embedding_dim: int = 128,
                                     mobile_tier: str = "mid") -> AdvancedIndexingSystem:
        """
        Create mobile-optimized indexing system.
        
        Args:
            dataset_size: Expected number of vectors
            embedding_dim: Dimension of embedding vectors
            mobile_tier: Mobile device tier ("low", "mid", "high")
            
        Returns:
            Configured AdvancedIndexingSystem
        """
        # Memory budgets based on mobile tier
        memory_budgets = {
            "low": 256.0,
            "mid": 512.0,
            "high": 1024.0
        }
        
        config = AdvancedIndexConfig(
            embedding_dim=embedding_dim,
            expected_dataset_size=dataset_size,
            memory_budget_mb=memory_budgets.get(mobile_tier, 512.0),
            mobile_tier=mobile_tier,
            target_latency_ms=50.0,
            target_recall=0.95,
            auto_index_selection=True,
            enable_performance_monitoring=True
        )
        
        return AdvancedIndexingSystem(config)
