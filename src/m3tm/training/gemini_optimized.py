"""
Optimized Gemini Client

Bu modül, gelişmiş caching, batch processing ve rate limiting
özellikleri ile optimize edilmiş Gemini client sağlar.
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .gemini_integration import GeminiClient, GeminiConfig
from .gemini_optimization import (
    AdvancedCache, CacheConfig,
    AdvancedRateLimiter, RateLimitConfig,
    IntelligentBatchProcessor, BatchConfig
)


@dataclass
class OptimizedGeminiConfig:
    """Optimized Gemini client konfigürasyonu"""
    # Base Gemini config
    gemini_config: GeminiConfig = None
    
    # Optimization configs
    cache_config: CacheConfig = None
    rate_limit_config: RateLimitConfig = None
    batch_config: BatchConfig = None
    
    # Model selection
    model_selection_enabled: bool = True
    task_specific_models: Dict[str, str] = None
    
    # Performance monitoring
    performance_monitoring: bool = True
    stats_logging_interval: int = 300  # 5 minutes
    
    def __post_init__(self):
        if self.gemini_config is None:
            self.gemini_config = GeminiConfig()
        
        if self.cache_config is None:
            self.cache_config = CacheConfig()
        
        if self.rate_limit_config is None:
            self.rate_limit_config = RateLimitConfig()
        
        if self.batch_config is None:
            self.batch_config = BatchConfig()
        
        if self.task_specific_models is None:
            self.task_specific_models = {
                "code_generation": "gemini-2.0-flash",
                "text_generation": "gemini-2.0-flash",
                "reasoning": "gemini-2.0-flash",
                "creative_writing": "gemini-2.0-flash",
                "data_analysis": "gemini-2.0-flash"
            }


class OptimizedGeminiClient:
    """Production-ready optimized Gemini client"""
    
    def __init__(self, config: OptimizedGeminiConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Base Gemini client
        self.base_client = GeminiClient(config.gemini_config)
        
        # Optimization components
        self.cache = AdvancedCache(config.cache_config)
        self.rate_limiter = AdvancedRateLimiter(config.rate_limit_config)
        self.batch_processor = IntelligentBatchProcessor(config.batch_config, self.base_client)
        
        # Performance monitoring
        self.request_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limited": 0,
            "errors": 0,
            "total_time": 0.0
        }
        
        # Start monitoring
        if config.performance_monitoring:
            self._start_monitoring()
        
        self.logger.info("OptimizedGeminiClient initialized")
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: str = "text_generation",
        priority: int = 1,
        use_cache: bool = True,
        use_batch: bool = False,
        max_retries: int = 3
    ) -> str:
        """Optimized text generation"""
        start_time = time.time()
        self.request_stats["total_requests"] += 1
        
        try:
            # Model selection
            model_name = self._select_model(task_type)
            
            # Config oluştur
            config = {
                "system_prompt": system_prompt,
                "model_name": model_name,
                "temperature": self.config.gemini_config.temperature,
                "max_output_tokens": self.config.gemini_config.max_output_tokens,
                "top_p": self.config.gemini_config.top_p,
                "top_k": self.config.gemini_config.top_k
            }
            
            # Cache kontrolü
            if use_cache:
                cached_result = self.cache.get(prompt, config)
                if cached_result:
                    self.request_stats["cache_hits"] += 1
                    self.logger.debug("Cache hit")
                    return cached_result
                else:
                    self.request_stats["cache_misses"] += 1
            
            # Rate limiting
            wait_time = self.rate_limiter.acquire(priority)
            if wait_time > 0:
                self.request_stats["rate_limited"] += 1
                self.logger.info(f"Rate limited, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            
            # Request execution
            if use_batch and priority < 3:
                # Batch processing (non-urgent requests)
                result = self.batch_processor.submit_request_sync(
                    prompt, config, priority, timeout=60.0
                )
            else:
                # Direct processing (urgent requests)
                result = self._direct_request(prompt, config, max_retries)
            
            # Cache result
            if use_cache and result:
                self.cache.set(prompt, config, result)
            
            # Success stats
            self.rate_limiter.record_success()
            
            return result
            
        except Exception as e:
            self.request_stats["errors"] += 1
            self.rate_limiter.record_error(str(type(e).__name__))
            self.logger.error(f"Generate text error: {e}")
            raise
        
        finally:
            self.request_stats["total_time"] += time.time() - start_time
    
    async def generate_text_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: str = "text_generation",
        priority: int = 1,
        use_cache: bool = True
    ) -> str:
        """Async optimized text generation"""
        # Model selection
        model_name = self._select_model(task_type)
        
        # Config oluştur
        config = {
            "system_prompt": system_prompt,
            "model_name": model_name,
            "temperature": self.config.gemini_config.temperature,
            "max_output_tokens": self.config.gemini_config.max_output_tokens,
            "top_p": self.config.gemini_config.top_p,
            "top_k": self.config.gemini_config.top_k
        }
        
        # Cache kontrolü
        if use_cache:
            cached_result = self.cache.get(prompt, config)
            if cached_result:
                return cached_result
        
        # Batch processing
        result = await self.batch_processor.submit_request(prompt, config, priority)
        
        # Cache result
        if use_cache and result:
            self.cache.set(prompt, config, result)
        
        return result
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        task_type: str = "text_generation",
        priority: int = 1,
        use_cache: bool = True
    ) -> List[str]:
        """Batch text generation with optimization"""
        results = []
        
        for prompt in prompts:
            try:
                result = self.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    task_type=task_type,
                    priority=priority,
                    use_cache=use_cache,
                    use_batch=True  # Force batch processing
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch item failed: {e}")
                results.append("")  # Empty result for failed items
        
        return results
    
    def _select_model(self, task_type: str) -> str:
        """Task-specific model selection"""
        if not self.config.model_selection_enabled:
            return self.config.gemini_config.model_name
        
        return self.config.task_specific_models.get(
            task_type, 
            self.config.gemini_config.model_name
        )
    
    def _direct_request(self, prompt: str, config: Dict[str, Any], max_retries: int) -> str:
        """Direct API request with retries"""
        for attempt in range(max_retries):
            try:
                return self.base_client.generate_text(
                    prompt=prompt,
                    system_prompt=config.get("system_prompt"),
                    max_retries=1  # Single attempt per retry
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * self.config.rate_limit_config.backoff_multiplier
                    wait_time = min(wait_time, self.config.rate_limit_config.max_backoff_seconds)
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise
    
    def _start_monitoring(self):
        """Performance monitoring başlat"""
        def monitor():
            while True:
                try:
                    time.sleep(self.config.stats_logging_interval)
                    self._log_stats()
                    self.cache.cleanup_expired()
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        import threading
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _log_stats(self):
        """İstatistikleri logla"""
        stats = self.get_comprehensive_stats()
        self.logger.info(f"Gemini Client Stats: {json.dumps(stats, indent=2)}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Comprehensive statistics"""
        cache_stats = self.cache.get_stats()
        rate_limiter_stats = self.rate_limiter.get_stats()
        batch_stats = self.batch_processor.get_stats()
        
        # Calculate derived metrics
        total_requests = self.request_stats["total_requests"]
        if total_requests > 0:
            cache_hit_rate = self.request_stats["cache_hits"] / total_requests
            error_rate = self.request_stats["errors"] / total_requests
            avg_response_time = self.request_stats["total_time"] / total_requests
        else:
            cache_hit_rate = 0.0
            error_rate = 0.0
            avg_response_time = 0.0
        
        return {
            "request_stats": {
                **self.request_stats,
                "cache_hit_rate": cache_hit_rate,
                "error_rate": error_rate,
                "avg_response_time": avg_response_time
            },
            "cache_stats": cache_stats,
            "rate_limiter_stats": rate_limiter_stats,
            "batch_stats": batch_stats
        }
    
    def optimize_performance(self):
        """Performance optimization önerileri"""
        stats = self.get_comprehensive_stats()
        
        recommendations = []
        
        # Cache optimization
        if stats["request_stats"]["cache_hit_rate"] < 0.3:
            recommendations.append("Cache hit rate düşük, TTL'yi artırın veya cache size'ı büyütün")
        
        # Rate limiting optimization
        if stats["request_stats"]["rate_limited"] > stats["request_stats"]["total_requests"] * 0.1:
            recommendations.append("Çok fazla rate limiting, batch processing'i artırın")
        
        # Error rate optimization
        if stats["request_stats"]["error_rate"] > 0.05:
            recommendations.append("Error rate yüksek, retry stratejisini gözden geçirin")
        
        # Batch optimization
        if stats["batch_stats"]["queued_requests"] > 50:
            recommendations.append("Batch queue çok dolu, parallel batch sayısını artırın")
        
        return recommendations
