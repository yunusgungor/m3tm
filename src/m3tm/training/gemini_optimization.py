"""
Gemini API Optimization Module

Bu modül, Gemini API kullanımını optimize etmek için gelişmiş
caching, batch processing ve rate limiting özellikleri sağlar.
"""

import os
import time
import json
import hashlib
import pickle
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3


@dataclass
class CacheConfig:
    """Gelişmiş cache konfigürasyonu"""
    cache_dir: str = "cache/gemini_advanced"
    max_cache_size_mb: int = 1000  # 1GB
    default_ttl_hours: int = 24  # 24 saat
    compression_enabled: bool = True
    sqlite_enabled: bool = True
    memory_cache_size: int = 100  # Memory'de tutulacak item sayısı


@dataclass
class BatchConfig:
    """Intelligent batch processing konfigürasyonu"""
    max_batch_size: int = 10
    min_batch_size: int = 2
    batch_timeout_seconds: float = 5.0
    parallel_batches: int = 3
    adaptive_sizing: bool = True
    priority_queue: bool = True


@dataclass
class RateLimitConfig:
    """Gelişmiş rate limiting konfigürasyonu"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_capacity: int = 10  # Token bucket burst capacity
    adaptive_limiting: bool = True
    backoff_multiplier: float = 1.5
    max_backoff_seconds: float = 300.0


class AdvancedCache:
    """Gelişmiş caching sistemi"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache (LRU)
        self.memory_cache = {}
        self.memory_access_order = deque()
        
        # SQLite cache
        if config.sqlite_enabled:
            self.db_path = self.cache_dir / "cache.db"
            self._init_sqlite()
        
        self.logger = logging.getLogger(__name__)
        
    def _init_sqlite(self):
        """SQLite cache database'ini başlat"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    ttl_hours REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
            """)
    
    def _generate_key(self, prompt: str, config: Dict[str, Any]) -> str:
        """Content-based cache key oluştur"""
        # Prompt ve config'i normalize et
        normalized_config = {
            k: v for k, v in sorted(config.items()) 
            if k in ['temperature', 'max_output_tokens', 'top_p', 'top_k', 'model_name']
        }
        
        content = f"{prompt}|{json.dumps(normalized_config, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, prompt: str, config: Dict[str, Any]) -> Optional[str]:
        """Cache'den değer al"""
        key = self._generate_key(prompt, config)
        
        # Memory cache'den kontrol et
        if key in self.memory_cache:
            self._update_memory_access(key)
            return self.memory_cache[key]
        
        # SQLite cache'den kontrol et
        if self.config.sqlite_enabled:
            result = self._get_from_sqlite(key)
            if result:
                # Memory cache'e ekle
                self._add_to_memory(key, result)
                return result
        
        return None
    
    def set(self, prompt: str, config: Dict[str, Any], value: str, ttl_hours: Optional[float] = None):
        """Cache'e değer ekle"""
        key = self._generate_key(prompt, config)
        ttl = ttl_hours or self.config.default_ttl_hours
        
        # Memory cache'e ekle
        self._add_to_memory(key, value)
        
        # SQLite cache'e ekle
        if self.config.sqlite_enabled:
            self._add_to_sqlite(key, value, ttl)
    
    def _get_from_sqlite(self, key: str) -> Optional[str]:
        """SQLite'dan değer al"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT value, created_at, ttl_hours FROM cache_entries 
                    WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                value_blob, created_at, ttl_hours = row
                
                # TTL kontrolü
                if time.time() - created_at > ttl_hours * 3600:
                    # Expired, sil
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    return None
                
                # Access count'u güncelle
                conn.execute("""
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE key = ?
                """, (time.time(), key))
                
                # Deserialize
                if self.config.compression_enabled:
                    return pickle.loads(value_blob)
                else:
                    return value_blob.decode('utf-8')
                    
        except Exception as e:
            self.logger.error(f"SQLite cache read error: {e}")
            return None
    
    def _add_to_sqlite(self, key: str, value: str, ttl_hours: float):
        """SQLite'a değer ekle"""
        try:
            # Serialize
            if self.config.compression_enabled:
                value_blob = pickle.dumps(value)
            else:
                value_blob = value.encode('utf-8')
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, ttl_hours, last_accessed)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, value_blob, time.time(), ttl_hours, time.time()))
                
        except Exception as e:
            self.logger.error(f"SQLite cache write error: {e}")
    
    def _add_to_memory(self, key: str, value: str):
        """Memory cache'e ekle (LRU)"""
        if key in self.memory_cache:
            # Mevcut key'i güncelle
            self.memory_access_order.remove(key)
        elif len(self.memory_cache) >= self.config.memory_cache_size:
            # LRU eviction
            oldest_key = self.memory_access_order.popleft()
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
        self.memory_access_order.append(key)
    
    def _update_memory_access(self, key: str):
        """Memory cache access order'ı güncelle"""
        self.memory_access_order.remove(key)
        self.memory_access_order.append(key)
    
    def cleanup_expired(self):
        """Expired cache entries'leri temizle"""
        if not self.config.sqlite_enabled:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Expired entries'leri sil
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE (? - created_at) > (ttl_hours * 3600)
                """, (time.time(),))
                
                # Cache size kontrolü
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                count = cursor.fetchone()[0]
                
                if count > 10000:  # Çok fazla entry varsa, eski olanları sil
                    conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE key IN (
                            SELECT key FROM cache_entries 
                            ORDER BY last_accessed ASC 
                            LIMIT ?
                        )
                    """, (count - 8000,))
                    
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Cache istatistikleri"""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_limit": self.config.memory_cache_size
        }
        
        if self.config.sqlite_enabled:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                    stats["sqlite_cache_size"] = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT SUM(access_count) FROM cache_entries")
                    stats["total_cache_hits"] = cursor.fetchone()[0] or 0
                    
            except Exception as e:
                self.logger.error(f"Cache stats error: {e}")
        
        return stats


class TokenBucket:
    """Token bucket rate limiting algoritması"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Token tüket, başarılıysa True döndür"""
        with self.lock:
            now = time.time()
            
            # Token'ları yenile
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # Token tüketimi
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Gerekli token'lar için bekleme süresi"""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.refill_rate


class AdvancedRateLimiter:
    """Gelişmiş rate limiting sistemi"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        
        # Token buckets for different time windows
        self.minute_bucket = TokenBucket(config.requests_per_minute, config.requests_per_minute / 60.0)
        self.hour_bucket = TokenBucket(config.requests_per_hour, config.requests_per_hour / 3600.0)
        self.day_bucket = TokenBucket(config.requests_per_day, config.requests_per_day / 86400.0)
        
        # Burst capacity
        self.burst_bucket = TokenBucket(config.burst_capacity, config.requests_per_minute / 60.0)
        
        # Adaptive limiting
        self.recent_errors = deque(maxlen=100)
        self.success_rate = 1.0
        
        self.logger = logging.getLogger(__name__)
    
    def acquire(self, priority: int = 1) -> float:
        """Rate limit token'ı al, bekleme süresi döndür"""
        # Adaptive rate limiting
        if self.config.adaptive_limiting:
            self._update_success_rate()
            if self.success_rate < 0.8:  # %80'den düşükse rate'i azalt
                effective_rate = 0.5
            else:
                effective_rate = 1.0
        else:
            effective_rate = 1.0
        
        # Token bucket'lardan kontrol et
        buckets = [self.minute_bucket, self.hour_bucket, self.day_bucket]
        
        # Burst capacity kontrolü
        if priority > 1 and self.burst_bucket.consume():
            return 0.0
        
        # Normal rate limiting
        max_wait = 0.0
        for bucket in buckets:
            if not bucket.consume():
                wait_time = bucket.wait_time() / effective_rate
                max_wait = max(max_wait, wait_time)
        
        return max_wait
    
    def record_success(self):
        """Başarılı request kaydı"""
        self.recent_errors.append(False)
    
    def record_error(self, error_type: str = "unknown"):
        """Hata kaydı"""
        self.recent_errors.append(True)
        self.logger.warning(f"Rate limiter recorded error: {error_type}")
    
    def _update_success_rate(self):
        """Success rate'i güncelle"""
        if len(self.recent_errors) < 10:
            return
        
        errors = sum(self.recent_errors)
        self.success_rate = 1.0 - (errors / len(self.recent_errors))
    
    def get_stats(self) -> Dict[str, Any]:
        """Rate limiter istatistikleri"""
        return {
            "minute_tokens": self.minute_bucket.tokens,
            "hour_tokens": self.hour_bucket.tokens,
            "day_tokens": self.day_bucket.tokens,
            "burst_tokens": self.burst_bucket.tokens,
            "success_rate": self.success_rate,
            "recent_errors": len([e for e in self.recent_errors if e])
        }


@dataclass
class BatchRequest:
    """Batch request item"""
    prompt: str
    config: Dict[str, Any]
    priority: int = 1
    callback: Optional[callable] = None
    future: Optional[asyncio.Future] = None
    created_at: float = field(default_factory=time.time)


class IntelligentBatchProcessor:
    """Intelligent batch processing sistemi"""

    def __init__(self, config: BatchConfig, gemini_client):
        self.config = config
        self.gemini_client = gemini_client

        # Request queues (priority-based)
        self.high_priority_queue = deque()
        self.normal_priority_queue = deque()
        self.low_priority_queue = deque()

        # Processing state
        self.processing = False
        self.active_batches = 0
        self.batch_stats = defaultdict(int)

        # Adaptive sizing
        self.recent_batch_times = deque(maxlen=50)
        self.optimal_batch_size = config.max_batch_size // 2

        self.logger = logging.getLogger(__name__)

        # Start background processor
        self._start_processor()

    def _start_processor(self):
        """Background batch processor'ı başlat"""
        def processor():
            while True:
                try:
                    self._process_batches()
                    time.sleep(0.1)  # 100ms check interval
                except Exception as e:
                    self.logger.error(f"Batch processor error: {e}")
                    time.sleep(1.0)

        thread = threading.Thread(target=processor, daemon=True)
        thread.start()

    async def submit_request(
        self,
        prompt: str,
        config: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """Async request submission"""
        future = asyncio.Future()
        request = BatchRequest(
            prompt=prompt,
            config=config,
            priority=priority,
            future=future
        )

        # Priority queue'ya ekle
        if priority >= 3:
            self.high_priority_queue.append(request)
        elif priority >= 2:
            self.normal_priority_queue.append(request)
        else:
            self.low_priority_queue.append(request)

        return await future

    def submit_request_sync(
        self,
        prompt: str,
        config: Dict[str, Any],
        priority: int = 1,
        timeout: float = 30.0
    ) -> str:
        """Sync request submission"""
        import concurrent.futures

        future = concurrent.futures.Future()
        request = BatchRequest(
            prompt=prompt,
            config=config,
            priority=priority,
            callback=lambda result: future.set_result(result)
        )

        # Priority queue'ya ekle
        if priority >= 3:
            self.high_priority_queue.append(request)
        elif priority >= 2:
            self.normal_priority_queue.append(request)
        else:
            self.low_priority_queue.append(request)

        return future.result(timeout=timeout)

    def _process_batches(self):
        """Batch'leri işle"""
        if self.active_batches >= self.config.parallel_batches:
            return

        # Request'leri topla
        batch = self._collect_batch()
        if not batch:
            return

        # Batch'i işle
        self.active_batches += 1
        thread = threading.Thread(
            target=self._execute_batch,
            args=(batch,),
            daemon=True
        )
        thread.start()

    def _collect_batch(self) -> List[BatchRequest]:
        """Optimal batch oluştur"""
        batch = []

        # Priority order: high -> normal -> low
        queues = [
            self.high_priority_queue,
            self.normal_priority_queue,
            self.low_priority_queue
        ]

        target_size = self._get_optimal_batch_size()

        for queue in queues:
            while queue and len(batch) < target_size:
                batch.append(queue.popleft())

        # Minimum batch size kontrolü
        if len(batch) < self.config.min_batch_size:
            # Timeout kontrolü
            if batch and time.time() - batch[0].created_at > self.config.batch_timeout_seconds:
                return batch  # Timeout olmuş, küçük batch'i işle
            else:
                # Batch'i geri koy
                for req in reversed(batch):
                    if req.priority >= 3:
                        self.high_priority_queue.appendleft(req)
                    elif req.priority >= 2:
                        self.normal_priority_queue.appendleft(req)
                    else:
                        self.low_priority_queue.appendleft(req)
                return []

        return batch

    def _get_optimal_batch_size(self) -> int:
        """Adaptive batch size hesapla"""
        if not self.config.adaptive_sizing:
            return self.config.max_batch_size

        if len(self.recent_batch_times) < 5:
            return self.optimal_batch_size

        # Son batch'lerin ortalama süresine göre optimize et
        avg_time = sum(self.recent_batch_times) / len(self.recent_batch_times)

        if avg_time > 10.0:  # Çok yavaş
            self.optimal_batch_size = max(
                self.config.min_batch_size,
                self.optimal_batch_size - 1
            )
        elif avg_time < 3.0:  # Çok hızlı
            self.optimal_batch_size = min(
                self.config.max_batch_size,
                self.optimal_batch_size + 1
            )

        return self.optimal_batch_size

    def _execute_batch(self, batch: List[BatchRequest]):
        """Batch'i execute et"""
        start_time = time.time()

        try:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {}

                for request in batch:
                    future = executor.submit(
                        self.gemini_client.generate_text,
                        request.prompt,
                        **request.config
                    )
                    futures[future] = request

                # Results'ları topla
                for future in as_completed(futures):
                    request = futures[future]
                    try:
                        result = future.result()
                        self._complete_request(request, result)
                        self.batch_stats['success'] += 1
                    except Exception as e:
                        self._complete_request(request, None, str(e))
                        self.batch_stats['error'] += 1

        except Exception as e:
            self.logger.error(f"Batch execution error: {e}")
            # Tüm request'leri error ile complete et
            for request in batch:
                self._complete_request(request, None, str(e))

        finally:
            # Stats güncelle
            batch_time = time.time() - start_time
            self.recent_batch_times.append(batch_time)
            self.batch_stats['total_batches'] += 1
            self.batch_stats['total_requests'] += len(batch)
            self.active_batches -= 1

            self.logger.debug(f"Batch completed: {len(batch)} requests in {batch_time:.2f}s")

    def _complete_request(self, request: BatchRequest, result: Optional[str], error: Optional[str] = None):
        """Request'i complete et"""
        if request.future:
            # Async completion
            if error:
                request.future.set_exception(Exception(error))
            else:
                request.future.set_result(result)
        elif request.callback:
            # Sync completion
            if error:
                request.callback(Exception(error))
            else:
                request.callback(result)

    def get_stats(self) -> Dict[str, Any]:
        """Batch processor istatistikleri"""
        total_queued = (
            len(self.high_priority_queue) +
            len(self.normal_priority_queue) +
            len(self.low_priority_queue)
        )

        return {
            "queued_requests": total_queued,
            "high_priority_queue": len(self.high_priority_queue),
            "normal_priority_queue": len(self.normal_priority_queue),
            "low_priority_queue": len(self.low_priority_queue),
            "active_batches": self.active_batches,
            "optimal_batch_size": self.optimal_batch_size,
            "recent_avg_batch_time": (
                sum(self.recent_batch_times) / len(self.recent_batch_times)
                if self.recent_batch_times else 0.0
            ),
            **dict(self.batch_stats)
        }
