#!/usr/bin/env python3
"""
M³TM v2.3 Simplified Demo - Mobile Optimization Showcase
Context7 Enhanced with Current Best Practices
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
from PIL import Image
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedM3TMDemo:
    """Simplified M³TM v2.3 demo with Context7 mobile optimizations."""
    
    def __init__(self):
        self.performance_metrics = {}
        self.mobile_optimizations_applied = []
        self.device = self._setup_device()
        self.model = None
        
        logger.info(f"Initialized M³TM Demo on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup device with mobile optimization considerations."""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            # Apply Context7 mobile GPU optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.mobile_optimizations_applied.append("CUDA TF32 optimization")
            logger.info("CUDA available, using GPU with mobile optimizations")
        else:
            device = torch.device('cpu')
            # Apply Context7 CPU optimizations
            torch.set_num_threads(4)  # Mobile-optimized thread count
            self.mobile_optimizations_applied.append("CPU thread optimization")
            logger.info("Using CPU with mobile optimizations")
        
        return device
    
    def build_model(self):
        """Build simplified M³TM model with Context7 mobile best practices."""
        logger.info("Building simplified M³TM model with Context7 optimizations...")
        
        # Simple unified embedding model
        self.model = nn.Sequential(
            nn.Linear(512, 256),  # Common projection layer
            nn.ReLU(),
            nn.Linear(256, 128),  # Final embedding size
            nn.LayerNorm(128)
        ).to(self.device)
        
        # Apply Context7 quantization best practices
        self._apply_mobile_quantization()
        
        # Apply Context7 torch.compile optimization
        self._apply_torch_compile()
        
        logger.info("Simplified M³TM model built successfully")
    
    def _apply_mobile_quantization(self):
        """Apply quantization using Context7 best practices."""
        logger.info("Applying Context7 mobile quantization...")
        
        if self.device.type == 'cuda':
            # FP16 for CUDA as recommended in Context7
            self.model = self.model.half()
            self.mobile_optimizations_applied.append("FP16 quantization")
            logger.info("Applied FP16 quantization for CUDA")
        else:
            # Dynamic quantization for CPU as per Context7 best practices
            self.model = torch.quantization.quantize_dynamic(
                self.model, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
            self.mobile_optimizations_applied.append("INT8 dynamic quantization")
            logger.info("Applied dynamic INT8 quantization for CPU")
    
    def _apply_torch_compile(self):
        """Apply torch.compile with Context7 mobile settings."""
        logger.info("Applying torch.compile with Context7 mobile optimizations...")
        
        try:
            # Set fallback for quantized models as per Context7 best practices
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            # Use Context7 recommended settings for mobile
            self.model = torch.compile(
                self.model, 
                mode="reduce-overhead",
                fullgraph=False  # Allow fallback for quantized ops
            )
            self.mobile_optimizations_applied.append("torch.compile optimization")
            logger.info("torch.compile applied successfully with fallback support")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}, continuing without compilation")
            self.mobile_optimizations_applied.append("torch.compile attempted (fallback used)")
    
    def process_text(self, text: str) -> torch.Tensor:
        """Process text with mobile-optimized inference."""
        start_time = time.time()
        
        # Simple text encoding (hash-based for demo)
        words = text.lower().split()
        # Create simple feature vector
        features = torch.zeros(512, device=self.device)
        for i, word in enumerate(words[:50]):  # Limit to 50 words
            hash_val = hash(word) % 512
            features[hash_val] += 1.0
        
        # Normalize
        features = features / (features.norm() + 1e-8)
        
        # Apply model with Context7 inference mode optimization
        with torch.inference_mode():
            embedding = self.model(features.unsqueeze(0)).squeeze(0)
        
        inference_time = time.time() - start_time
        self._update_performance_metrics('text_inference', inference_time)
        
        return embedding
    
    def process_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Process image with mobile-optimized inference."""
        start_time = time.time()
        
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image)
        
        # Simple image encoding (color histogram for demo)
        image_rgb = image.convert('RGB').resize((32, 32))  # Mobile-friendly size
        pixels = np.array(image_rgb)
        
        # Create simple feature vector from color channels
        features = torch.zeros(512, device=self.device)
        for c in range(3):  # RGB channels
            channel = pixels[:, :, c].flatten()
            hist, _ = np.histogram(channel, bins=170, range=(0, 256))
            start_idx = c * 170
            features[start_idx:start_idx+170] = torch.tensor(hist, dtype=torch.float32)
        
        # Normalize
        features = features / (features.norm() + 1e-8)
        
        # Apply model with Context7 inference mode optimization
        with torch.inference_mode():
            embedding = self.model(features.unsqueeze(0)).squeeze(0)
        
        inference_time = time.time() - start_time
        self._update_performance_metrics('image_inference', inference_time)
        
        return embedding
    
    def semantic_search(self, query: str, data_embeddings: List[torch.Tensor], 
                       top_k: int = 5) -> List[Dict]:
        """Perform semantic search with mobile optimization."""
        start_time = time.time()
        
        # Process query
        query_embedding = self.process_text(query)
        
        # Compute similarities (optimized for mobile)
        similarities = []
        with torch.inference_mode():
            for i, data_emb in enumerate(data_embeddings):
                similarity = torch.cosine_similarity(
                    query_embedding.unsqueeze(0), 
                    data_emb.unsqueeze(0)
                ).item()
                similarities.append((i, similarity))
        
        # Sort and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]
        
        search_time = time.time() - start_time
        self._update_performance_metrics('search_time', search_time)
        
        return [{'index': i, 'similarity': sim} for i, sim in results]
    
    def on_device_training_demo(self):
        """Demonstrate on-device training concepts."""
        logger.info("Demonstrating on-device training concepts...")
        
        # Create simple training scenario
        sample_data = [
            torch.randn(512, device=self.device) for _ in range(10)
        ]
        
        # Add a simple adapter layer for training
        adapter = nn.Linear(128, 128).to(self.device)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(3):
            epoch_loss = 0.0
            for data in sample_data:
                optimizer.zero_grad()
                
                # Forward pass - remove inference_mode to allow gradients
                # Only compute features needed for adapter training
                with torch.no_grad():
                    base_features = self.model(data.unsqueeze(0)).squeeze(0)
                
                # Detach and clone to break the computation graph but allow new gradients
                base_features = base_features.detach().clone().requires_grad_(False)
                adapted_features = adapter(base_features)
                
                # Simple loss (distance from target)
                target = torch.randn_like(adapted_features, device=self.device)
                loss = nn.functional.mse_loss(adapted_features, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            logger.info(f"Training epoch {epoch+1}: loss = {epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        self._update_performance_metrics('training_time', training_time)
        
        logger.info(f"On-device training demo completed in {training_time:.2f} seconds")
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Comprehensive mobile performance benchmarking."""
        logger.info("Running mobile performance benchmarks...")
        
        # Model size
        model_size_mb = self._get_model_size()
        
        # Memory usage
        memory_usage_mb = self._get_memory_usage()
        
        # Text inference benchmark
        sample_text = "This is a mobile optimization test for text processing."
        text_times = []
        for _ in range(10):
            start = time.time()
            _ = self.process_text(sample_text)
            text_times.append(time.time() - start)
        
        # Image inference benchmark
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        image_times = []
        for _ in range(10):
            start = time.time()
            _ = self.process_image(dummy_image)
            image_times.append(time.time() - start)
        
        # Compile results
        benchmark_results = {
            'model_size_mb': model_size_mb,
            'memory_usage_mb': memory_usage_mb,
            'text_inference': {
                'avg_time_ms': np.mean(text_times) * 1000,
                'std_time_ms': np.std(text_times) * 1000,
                'min_time_ms': np.min(text_times) * 1000,
                'max_time_ms': np.max(text_times) * 1000
            },
            'image_inference': {
                'avg_time_ms': np.mean(image_times) * 1000,
                'std_time_ms': np.std(image_times) * 1000,
                'min_time_ms': np.min(image_times) * 1000,
                'max_time_ms': np.max(image_times) * 1000
            },
            'mobile_optimizations_applied': self.mobile_optimizations_applied,
            'performance_summary': self.performance_metrics,
            'device_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available()
            }
        }
        
        logger.info("Performance benchmark completed")
        return benchmark_results
    
    def export_data(self) -> Dict[str, Any]:
        """Demonstrate data export functionality."""
        logger.info("Exporting demo data...")
        
        export_data = {
            'model_info': {
                'type': 'Simplified M³TM v2.3',
                'device': str(self.device),
                'optimizations': self.mobile_optimizations_applied
            },
            'performance_metrics': self.performance_metrics,
            'export_timestamp': time.time(),
            'context7_optimizations': [
                'torch.compile with reduce-overhead mode',
                'Dynamic quantization for CPU',
                'FP16 quantization for CUDA',
                'Inference mode optimization',
                'Mobile-optimized thread count'
            ]
        }
        
        return export_data
    
    def _get_model_size(self) -> float:
        """Get model size in MB."""
        if self.model is None:
            return 0.0
        
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        size_mb = param_size / 1024 / 1024
        return round(size_mb, 2)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def _update_performance_metrics(self, metric_name: str, value: float):
        """Update performance metrics tracking."""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        self.performance_metrics[metric_name].append(value)


def run_simplified_demo():
    """Run the simplified M³TM v2.3 mobile demo."""
    print("🚀 M³TM v2.3 Simplified Mobile Demo")
    print("=" * 50)
    print("Context7 Enhanced Mobile AI Demonstration")
    print()
    
    # Create demo instance
    demo = SimplifiedM3TMDemo()
    
    try:
        # Build model
        print("📱 Building mobile-optimized model...")
        demo.build_model()
        print("✅ Model built with Context7 optimizations")
        print()
        
        # Test text processing
        print("📝 Testing text processing...")
        sample_texts = [
            "Mobile AI is the future",
            "Privacy-preserving machine learning", 
            "On-device neural networks"
        ]
        
        text_embeddings = []
        for text in sample_texts:
            embedding = demo.process_text(text)
            text_embeddings.append(embedding)
            print(f"   ✓ Processed: '{text[:30]}...'")
        print()
        
        # Test image processing
        print("🖼️  Testing image processing...")
        sample_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        
        image_embeddings = []
        for i, image in enumerate(sample_images):
            embedding = demo.process_image(image)
            image_embeddings.append(embedding)
            print(f"   ✓ Processed image {i+1}")
        print()
        
        # Test semantic search
        print("🔍 Testing semantic search...")
        search_query = "artificial intelligence"
        all_embeddings = text_embeddings + image_embeddings
        results = demo.semantic_search(search_query, all_embeddings, top_k=3)
        
        print(f"   Query: '{search_query}'")
        for result in results:
            print(f"   Result {result['index']}: similarity = {result['similarity']:.4f}")
        print()
        
        # Test on-device training
        print("🎯 Testing on-device training demo...")
        demo.on_device_training_demo()
        print("✅ On-device training demonstration completed")
        print()
        
        # Performance benchmarking
        print("⚡ Running performance benchmarks...")
        benchmark_results = demo.benchmark_performance()
        
        print("📊 Performance Results:")
        print(f"   Model Size: {benchmark_results['model_size_mb']:.2f} MB")
        print(f"   Memory Usage: {benchmark_results['memory_usage_mb']:.2f} MB")
        print(f"   Text Inference: {benchmark_results['text_inference']['avg_time_ms']:.2f} ms")
        print(f"   Image Inference: {benchmark_results['image_inference']['avg_time_ms']:.2f} ms")
        print()
        
        # Data export
        print("💾 Testing data export...")
        export_data = demo.export_data()
        print(f"   ✓ Exported {len(export_data)} data sections")
        print()
        
        # Final summary
        print("🎉 M³TM v2.3 Simplified Demo Completed!")
        print("=" * 50)
        
        # Check performance targets
        text_time = benchmark_results['text_inference']['avg_time_ms']
        image_time = benchmark_results['image_inference']['avg_time_ms']
        model_size = benchmark_results['model_size_mb']
        memory_usage = benchmark_results['memory_usage_mb']
        
        print("📋 Performance Target Results:")
        print(f"   Text inference: {text_time:.1f}ms (target: <100ms) {'✅' if text_time < 100 else '❌'}")
        print(f"   Image inference: {image_time:.1f}ms (target: <100ms) {'✅' if image_time < 100 else '❌'}")
        print(f"   Model size: {model_size:.1f}MB (target: <500MB) {'✅' if model_size < 500 else '❌'}")
        print(f"   Memory usage: {memory_usage:.1f}MB (target: <2GB) {'✅' if memory_usage < 2048 else '❌'}")
        print()
        
        print("Context7 Mobile Optimizations Applied:")
        for opt in benchmark_results['mobile_optimizations_applied']:
            print(f"   ✓ {opt}")
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    results = run_simplified_demo()
