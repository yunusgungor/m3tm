"""
Advanced Intelligent Caching System for M³TM Mobile
Context7 Enhanced Implementation with Persistent Storage, Cache Warming, and Analytics

Features:
- Multi-level caching (embedding, features, results)
- Persistent storage with SQLite backend
- LRU eviction with TTL and size limits
- Cache warming strategies
- Cache analytics and optimization
- Thread-safe concurrent access
- Compression and deduplication
"""

import sqlite3
import pickle
import hashlib
import threading
import time
import json
import gzip
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import OrderedDict
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value_hash: str
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: float
    compression_ratio: float = 1.0

@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    cache_hit_rate: float
    average_access_time: float
    storage_efficiency: float

@dataclass
class CacheConfig:
    """Cache configuration for advanced caching system"""
    strategy: str = "multi_level"  # multi_level, memory_only, persistent
    max_cache_size_mb: int = 512
    enable_persistent: bool = True
    enable_analytics: bool = True
    cache_dir: str = "./cache"
    embedding_cache_size: int = 10000
    feature_cache_size: int = 5000
    result_cache_size: int = 1000
    default_ttl: int = 3600
    enable_compression: bool = True

class PersistentCache:
    """Persistent cache with SQLite backend"""
    
    def __init__(self, cache_dir: str, cache_name: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / f"{cache_name}.db"
        self.data_dir = self.cache_dir / f"{cache_name}_data"
        self.data_dir.mkdir(exist_ok=True)
        
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value_hash TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER NOT NULL,
                    ttl REAL NOT NULL,
                    compression_ratio REAL DEFAULT 1.0,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count)
            """)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value_hash, timestamp, ttl FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                value_hash, timestamp, ttl = row
                
                # Check TTL
                if time.time() - timestamp > ttl:
                    self.delete(key)
                    return None
                
                # Update access stats
                conn.execute(
                    "UPDATE cache_entries SET access_count = access_count + 1, timestamp = ? WHERE key = ?",
                    (time.time(), key)
                )
                
                # Load value from file
                value_path = self.data_dir / f"{value_hash}.pkl.gz"
                if value_path.exists():
                    with gzip.open(value_path, 'rb') as f:
                        return pickle.load(f)
                
        except Exception as e:
            logger.error(f"Error getting cache entry {key}: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: float = 3600):
        """Set value in persistent cache"""
        try:
            # Serialize and compress value
            serialized = pickle.dumps(value)
            compressed = gzip.compress(serialized)
            
            value_hash = hashlib.sha256(compressed).hexdigest()
            size_bytes = len(compressed)
            compression_ratio = len(serialized) / len(compressed) if compressed else 1.0
            
            # Save compressed data
            value_path = self.data_dir / f"{value_hash}.pkl.gz"
            with open(value_path, 'wb') as f:
                f.write(compressed)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value_hash, timestamp, size_bytes, ttl, compression_ratio, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (key, value_hash, time.time(), size_bytes, ttl, compression_ratio, time.time()))
                
        except Exception as e:
            logger.error(f"Error setting cache entry {key}: {e}")
    
    def delete(self, key: str):
        """Delete entry from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value_hash FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    value_hash = row[0]
                    
                    # Delete file
                    value_path = self.data_dir / f"{value_hash}.pkl.gz"
                    if value_path.exists():
                        os.remove(value_path)
                    
                    # Delete from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    
        except Exception as e:
            logger.error(f"Error deleting cache entry {key}: {e}")
    
    def cleanup_expired(self):
        """Clean up expired entries"""
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get expired entries
            cursor = conn.execute(
                "SELECT key, value_hash FROM cache_entries WHERE ? - timestamp > ttl",
                (current_time,)
            )
            expired_entries = cursor.fetchall()
            
            for key, value_hash in expired_entries:
                # Delete file
                value_path = self.data_dir / f"{value_hash}.pkl.gz"
                if value_path.exists():
                    os.remove(value_path)
            
            # Delete from database
            conn.execute("DELETE FROM cache_entries WHERE ? - timestamp > ttl", (current_time,))
            
            logger.info(f"Cleaned up {len(expired_entries)} expired cache entries")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(size_bytes) as total_size,
                    SUM(access_count) as total_accesses,
                    AVG(compression_ratio) as avg_compression
                FROM cache_entries
            """)
            row = cursor.fetchone()
            
            if row:
                return {
                    'total_entries': row[0] or 0,
                    'total_size_bytes': row[1] or 0,
                    'total_accesses': row[2] or 0,
                    'average_compression_ratio': row[3] or 1.0
                }
        
        return {'total_entries': 0, 'total_size_bytes': 0, 'total_accesses': 0, 'average_compression_ratio': 1.0}

class AdvancedCacheManager:
    """Advanced multi-level cache manager with intelligence"""
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None
    ):
        self.config = config or CacheConfig()
        
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent caches
        self.embedding_cache = PersistentCache(self.config.cache_dir, "embeddings")
        self.feature_cache = PersistentCache(self.config.cache_dir, "features") 
        self.result_cache = PersistentCache(self.config.cache_dir, "results")
        
        # Memory caches for hot data
        self.memory_caches = {
            'embeddings': OrderedDict(),
            'features': OrderedDict(),
            'results': OrderedDict()
        }
        
        self.cache_sizes = {
            'embeddings': self.config.embedding_cache_size,
            'features': self.config.feature_cache_size,
            'results': self.config.result_cache_size
        }
        
        self.default_ttl = self.config.default_ttl
        self.enable_analytics = self.config.enable_analytics
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Analytics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'access_times': []
        }
        
        # Cache warming strategies
        self.warming_strategies = []
        
    def _generate_key(self, data: Any, prefix: str = "") -> str:
        """Generate consistent cache key"""
        if isinstance(data, torch.Tensor):
            data_bytes = data.detach().cpu().numpy().tobytes()
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = str(data).encode()
        
        key_hash = hashlib.sha256(data_bytes).hexdigest()
        return f"{prefix}:{key_hash}" if prefix else key_hash
    
    def _evict_lru_memory(self, cache_type: str):
        """Evict LRU items from memory cache"""
        memory_cache = self.memory_caches[cache_type]
        max_size = self.cache_sizes[cache_type] // 10  # Keep 10% in memory
        
        while len(memory_cache) >= max_size:
            memory_cache.popitem(last=False)
            self.stats['evictions'] += 1
    
    def get_embedding(self, input_data: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached embedding"""
        key = self._generate_key(input_data, "emb")
        
        with self.lock:
            start_time = time.time()
            
            # Check memory cache first
            if key in self.memory_caches['embeddings']:
                self.memory_caches['embeddings'].move_to_end(key)
                result = self.memory_caches['embeddings'][key]
                self.stats['hits'] += 1
            else:
                # Check persistent cache
                result = self.embedding_cache.get(key)
                if result is not None:
                    # Add to memory cache
                    self._evict_lru_memory('embeddings')
                    self.memory_caches['embeddings'][key] = result
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
            
            if self.enable_analytics:
                self.stats['access_times'].append(time.time() - start_time)
            
            return result
    
    def set_embedding(self, input_data: torch.Tensor, embedding: torch.Tensor, ttl: Optional[int] = None):
        """Cache embedding"""
        key = self._generate_key(input_data, "emb")
        ttl = ttl or self.default_ttl
        
        with self.lock:
            # Set in persistent cache
            self.embedding_cache.set(key, embedding, ttl)
            
            # Set in memory cache
            self._evict_lru_memory('embeddings')
            self.memory_caches['embeddings'][key] = embedding
    
    def get_features(self, input_data: Any) -> Optional[torch.Tensor]:
        """Get cached features"""
        key = self._generate_key(input_data, "feat")
        
        with self.lock:
            start_time = time.time()
            
            if key in self.memory_caches['features']:
                self.memory_caches['features'].move_to_end(key)
                result = self.memory_caches['features'][key]
                self.stats['hits'] += 1
            else:
                result = self.feature_cache.get(key)
                if result is not None:
                    self._evict_lru_memory('features')
                    self.memory_caches['features'][key] = result
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
            
            if self.enable_analytics:
                self.stats['access_times'].append(time.time() - start_time)
            
            return result
    
    def set_features(self, input_data: Any, features: torch.Tensor, ttl: Optional[int] = None):
        """Cache features"""
        key = self._generate_key(input_data, "feat")
        ttl = ttl or self.default_ttl
        
        with self.lock:
            self.feature_cache.set(key, features, ttl)
            self._evict_lru_memory('features')
            self.memory_caches['features'][key] = features
    
    def get_result(self, input_data: Any) -> Optional[Any]:
        """Get cached result"""
        key = self._generate_key(input_data, "res")
        
        with self.lock:
            start_time = time.time()
            
            if key in self.memory_caches['results']:
                self.memory_caches['results'].move_to_end(key)
                result = self.memory_caches['results'][key]
                self.stats['hits'] += 1
            else:
                result = self.result_cache.get(key)
                if result is not None:
                    self._evict_lru_memory('results')
                    self.memory_caches['results'][key] = result
                    self.stats['hits'] += 1
                else:
                    self.stats['misses'] += 1
            
            if self.enable_analytics:
                self.stats['access_times'].append(time.time() - start_time)
            
            return result
    
    def set_result(self, input_data: Any, result: Any, ttl: Optional[int] = None):
        """Cache result"""
        key = self._generate_key(input_data, "res")
        ttl = ttl or self.default_ttl
        
        with self.lock:
            self.result_cache.set(key, result, ttl)
            self._evict_lru_memory('results')
            self.memory_caches['results'][key] = result
    
    def add_warming_strategy(self, strategy_func, priority: int = 1):
        """Add cache warming strategy"""
        self.warming_strategies.append((priority, strategy_func))
        self.warming_strategies.sort(key=lambda x: x[0])
    
    def warm_cache(self, data_loader: Optional[Any] = None):
        """Execute cache warming strategies"""
        logger.info("Starting cache warming...")
        
        for priority, strategy_func in self.warming_strategies:
            try:
                strategy_func(self, data_loader)
                logger.info(f"Cache warming strategy (priority {priority}) completed")
            except Exception as e:
                logger.error(f"Cache warming strategy failed: {e}")
    
    def cleanup(self):
        """Clean up expired entries"""
        self.embedding_cache.cleanup_expired()
        self.feature_cache.cleanup_expired()
        self.result_cache.cleanup_expired()
    
    def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        avg_access_time = sum(self.stats['access_times']) / len(self.stats['access_times']) if self.stats['access_times'] else 0
        
        # Get persistent cache stats
        emb_stats = self.embedding_cache.get_stats()
        feat_stats = self.feature_cache.get_stats()
        res_stats = self.result_cache.get_stats()
        
        total_size = emb_stats['total_size_bytes'] + feat_stats['total_size_bytes'] + res_stats['total_size_bytes']
        total_entries = emb_stats['total_entries'] + feat_stats['total_entries'] + res_stats['total_entries']
        
        storage_efficiency = (
            emb_stats['average_compression_ratio'] + 
            feat_stats['average_compression_ratio'] + 
            res_stats['average_compression_ratio']
        ) / 3 if total_entries > 0 else 1.0
        
        return CacheStats(
            total_entries=total_entries,
            total_size_bytes=total_size,
            hit_count=self.stats['hits'],
            miss_count=self.stats['misses'],
            eviction_count=self.stats['evictions'],
            cache_hit_rate=hit_rate,
            average_access_time=avg_access_time,
            storage_efficiency=storage_efficiency
        )
    
    def export_analytics(self, output_path: str):
        """Export cache analytics to JSON"""
        stats = self.get_cache_stats()
        analytics_data = {
            'cache_stats': asdict(stats),
            'detailed_stats': {
                'embeddings': self.embedding_cache.get_stats(),
                'features': self.feature_cache.get_stats(),
                'results': self.result_cache.get_stats()
            },
            'memory_cache_sizes': {
                cache_type: len(cache) 
                for cache_type, cache in self.memory_caches.items()
            },
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(analytics_data, f, indent=2)
        
        logger.info(f"Cache analytics exported to {output_path}")

# Cache warming strategies
def warm_embeddings_strategy(cache_manager: AdvancedCacheManager, data_loader):
    """Strategy to warm embedding cache with common inputs"""
    if data_loader is None:
        return
    
    logger.info("Warming embedding cache...")
    for i, (data, _) in enumerate(data_loader):
        if i >= 100:  # Limit warming samples
            break
        
        # This would be called by the actual model during warming
        # cache_manager.set_embedding(data, computed_embedding)
        
def warm_frequent_patterns_strategy(cache_manager: AdvancedCacheManager, data_loader):
    """Strategy to warm cache with frequently accessed patterns"""
    logger.info("Warming cache with frequent patterns...")
    # Implementation would analyze access patterns and pre-compute common combinations
