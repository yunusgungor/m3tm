"""
M³TM Advanced Search Strategies Module

Advanced search algorithm implementations using Strategy pattern.
Optimized for mobile deployment with Context7 FAISS best practices.

Patterns: PT-015 (PluggableComponentStrategy), PT-002 (FactoryMethod)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, List, Optional, TYPE_CHECKING
import logging
import time
import numpy as np
import torch

if TYPE_CHECKING:
    from .hnsw_mobile import HNSWConfig

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""
    
    @abstractmethod
    def create_index(self, embedding_dim: int, config: Dict[str, Any]) -> Any:
        """Create index for this strategy."""
        pass
    
    @abstractmethod
    def search(self, index: Any, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform search using this strategy."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get name of this strategy."""
        pass


class HNSWSearchStrategy(SearchStrategy):
    """
    HNSW (Hierarchical Navigable Small World) search strategy.
    
    Optimized for mobile deployment with Context7 recommendations:
    - Lower M parameter for memory efficiency
    - Balanced efConstruction for build time
    - Adaptive efSearch for runtime performance
    """
    
    def __init__(self, config_or_m: Union[int, 'HNSWConfig'] = 16, ef_construction: int = 200, ef_search: int = 50):
        """
        Initialize HNSW strategy with mobile-optimized parameters.
        
        Args:
            config_or_m: Either HNSWConfig object or M parameter (for backward compatibility)
            ef_construction: Size of candidate list during construction
            ef_search: Size of candidate list during search
        """
        # Handle both HNSWConfig object and integer M parameter
        if hasattr(config_or_m, 'm'):  # It's an HNSWConfig object
            self.config = config_or_m
            self.m = config_or_m.m
            self.ef_construction = config_or_m.ef_construction
            self.ef_search = config_or_m.ef_search
        else:  # It's an integer M parameter
            self.config = None
            self.m = config_or_m
            self.ef_construction = ef_construction
            self.ef_search = ef_search
        
        self.index = None
        
        logger.info(f"HNSW strategy initialized: M={self.m}, efConstruction={self.ef_construction}, efSearch={self.ef_search}")
    
    def create_index(self, embedding_dim: int, config: Dict[str, Any]) -> Any:
        """
        Create HNSW index optimized for mobile deployment.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            config: Additional configuration parameters
            
        Returns:
            Configured HNSW index
        """
        if faiss is None:
            raise ImportError("FAISS not available")
        
        # Determine metric type
        metric = faiss.METRIC_INNER_PRODUCT if config.get("metric_type") == "ip" else faiss.METRIC_L2
        
        # Create HNSW index
        index = faiss.IndexHNSWFlat(embedding_dim, self.m, metric)
        
        # Configure parameters
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        
        return index
    
    def search(self, index: Any, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform HNSW search with mobile optimization.
        
        Args:
            index: HNSW index
            query: Query vector(s)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        # Perform search
        distances, indices = index.search(query, k)
        
        return distances, indices
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return f"HNSW_M{self.m}_ef{self.ef_search}"
    
    def update_search_parameters(self, ef_search: int):
        """Update search-time parameters."""
        self.ef_search = ef_search
        logger.info(f"HNSW efSearch updated to {ef_search}")
    
    def build_index(self, vectors: np.ndarray) -> Any:
        """
        Build HNSW index with the given vectors.
        
        Args:
            vectors: Embedding vectors to index
            
        Returns:
            Built HNSW index
        """
        embedding_dim = vectors.shape[1]
        config = {"metric_type": "ip"}
        
        self.index = self.create_index(embedding_dim, config)
        self.index.add(vectors)
        
        return self.index


class IVFPQSearchStrategy(SearchStrategy):
    """
    IVF+PQ (Inverted File + Product Quantization) search strategy.
    
    Optimized for memory-efficient large-scale search on mobile devices.
    """
    
    def __init__(self, nlist: int = 1024, pq_m: int = 8, pq_nbits: int = 8, nprobe: int = 16):
        """
        Initialize IVF+PQ strategy with mobile-optimized parameters.
        
        Args:
            nlist: Number of inverted lists
            pq_m: Number of sub-quantizers
            pq_nbits: Bits per sub-quantizer
            nprobe: Number of lists to probe during search
        """
        self.nlist = nlist
        self.pq_m = pq_m
        self.pq_nbits = pq_nbits
        self.nprobe = nprobe
        
        logger.info(f"IVF+PQ strategy initialized: nlist={nlist}, PQ={pq_m}x{pq_nbits}, nprobe={nprobe}")
    
    def create_index(self, embedding_dim: int, config: Dict[str, Any]) -> Any:
        """
        Create IVF+PQ index for memory-efficient search.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            config: Additional configuration parameters
            
        Returns:
            Configured IVF+PQ index
        """
        if faiss is None:
            raise ImportError("FAISS not available")
        
        # Determine metric type
        metric = faiss.METRIC_INNER_PRODUCT if config.get("metric_type") == "ip" else faiss.METRIC_L2
        
        # Create quantizer (flat index for centroids)
        quantizer = faiss.IndexFlatL2(embedding_dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(embedding_dim)
        
        # Create IVF+PQ index
        index = faiss.IndexIVFPQ(quantizer, embedding_dim, self.nlist, self.pq_m, self.pq_nbits, metric)
        
        # Set search parameters
        index.nprobe = self.nprobe
        
        return index
    
    def search(self, index: Any, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform IVF+PQ search with mobile optimization.
        
        Args:
            index: IVF+PQ index
            query: Query vector(s)
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        # Perform search
        distances, indices = index.search(query, k)
        
        return distances, indices
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return f"IVF{self.nlist}_PQ{self.pq_m}x{self.pq_nbits}_probe{self.nprobe}"
    
    def update_search_parameters(self, nprobe: int):
        """Update search-time parameters."""
        self.nprobe = nprobe
        logger.info(f"IVF+PQ nprobe updated to {nprobe}")


class AdaptiveSearchStrategy(SearchStrategy):
    """
    Adaptive search strategy that selects optimal algorithm based on dataset characteristics.
    
    Implements mobile-first approach with automatic algorithm selection.
    """
    
    def __init__(self):
        """Initialize adaptive strategy."""
        self.strategies = {
            "hnsw": HNSWSearchStrategy(),
            "ivf_pq": IVFPQSearchStrategy(),
            "flat": FlatSearchStrategy()
        }
        self.selected_strategy = None
        
        logger.info("Adaptive search strategy initialized")
    
    def select_strategy(self, dataset_size: int, embedding_dim: int, memory_limit_gb: float) -> str:
        """
        Select optimal strategy based on dataset characteristics.
        
        Args:
            dataset_size: Number of vectors in dataset
            embedding_dim: Dimension of embedding vectors
            memory_limit_gb: Memory constraint
            
        Returns:
            Selected strategy name
        """
        # Estimate memory requirements
        vector_memory_gb = (dataset_size * embedding_dim * 4) / (1024**3)  # float32
        
        if dataset_size < 10000:
            # Small dataset - use flat for accuracy
            strategy_name = "flat"
        elif vector_memory_gb > memory_limit_gb * 0.7:
            # Memory constrained - use IVF+PQ for compression
            strategy_name = "ivf_pq"
        else:
            # Balanced case - use HNSW for performance
            strategy_name = "hnsw"
        
        self.selected_strategy = strategy_name
        logger.info(f"Selected strategy: {strategy_name} for dataset_size={dataset_size}, memory_limit={memory_limit_gb}GB")
        
        return strategy_name
    
    def create_index(self, embedding_dim: int, config: Dict[str, Any]) -> Any:
        """Create index using selected strategy."""
        if self.selected_strategy is None:
            # Auto-select based on config
            dataset_size = config.get("expected_dataset_size", 100000)
            memory_limit = config.get("memory_limit_gb", 2.0)
            self.select_strategy(dataset_size, embedding_dim, memory_limit)
        
        return self.strategies[self.selected_strategy].create_index(embedding_dim, config)
    
    def search(self, index: Any, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform search using selected strategy."""
        return self.strategies[self.selected_strategy].search(index, query, k)
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return f"Adaptive_{self.selected_strategy}" if self.selected_strategy else "Adaptive_unselected"


class FlatSearchStrategy(SearchStrategy):
    """Flat (exhaustive) search strategy for small datasets or high accuracy requirements."""
    
    def create_index(self, embedding_dim: int, config: Dict[str, Any]) -> Any:
        """Create flat index."""
        if faiss is None:
            raise ImportError("FAISS not available")
        
        metric = faiss.METRIC_INNER_PRODUCT if config.get("metric_type") == "ip" else faiss.METRIC_L2
        
        if metric == faiss.METRIC_INNER_PRODUCT:
            return faiss.IndexFlatIP(embedding_dim)
        else:
            return faiss.IndexFlatL2(embedding_dim)
    
    def search(self, index: Any, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform flat search."""
        if query.ndim == 1:
            query = query.reshape(1, -1)
        return index.search(query, k)
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "Flat"


class SearchStrategyFactory:
    """
    Factory for creating search strategies.
    
    Pattern: PT-002 (FactoryMethod)
    """
    
    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> SearchStrategy:
        """
        Create search strategy by type.
        
        Args:
            strategy_type: Type of strategy ("hnsw", "ivf_pq", "adaptive", "flat")
            **kwargs: Strategy-specific parameters
            
        Returns:
            SearchStrategy instance
        """
        if strategy_type == "hnsw":
            return HNSWSearchStrategy(**kwargs)
        elif strategy_type == "ivf_pq":
            return IVFPQSearchStrategy(**kwargs)
        elif strategy_type == "adaptive":
            return AdaptiveSearchStrategy()
        elif strategy_type == "flat":
            return FlatSearchStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    @staticmethod
    def get_mobile_optimized_strategy(dataset_size: int, memory_limit_gb: float = 2.0) -> SearchStrategy:
        """
        Get mobile-optimized strategy based on dataset characteristics.
        
        Args:
            dataset_size: Expected number of vectors
            memory_limit_gb: Mobile memory constraint
            
        Returns:
            Optimized SearchStrategy
        """
        adaptive_strategy = AdaptiveSearchStrategy()
        adaptive_strategy.select_strategy(dataset_size, 128, memory_limit_gb)  # Assume 128D embeddings
        return adaptive_strategy
