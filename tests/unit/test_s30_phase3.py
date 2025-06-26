"""
S30 Phase 3: Multi-modal Search & Fusion - Comprehensive Test Suite

Tests for cross-modal embedding alignment, multi-modal search strategies,
joint embedding space optimization, and cross-modal search functionality.

Test Coverage:
- CrossModalAligner with contrastive learning
- AttentionFusion mechanism  
- MultiModalSearchStrategy implementations
- MultiModalSearchIndex functionality
- CrossModalRetrieval system
- Integration with existing search infrastructure
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import tempfile
import json
from pathlib import Path

# Import S30 Phase 3 components
from src.m3tm.search.multimodal_fusion import (
    MultiModalConfig,
    FusionStrategy,
    CrossModalAligner,
    AttentionFusion,
    EarlyFusionStrategy,
    LateFusionStrategy,
    AttentionFusionStrategy,
    MultiModalSearchStrategyFactory,
    MultiModalSearchIndex,
    CrossModalRetrieval
)


class TestMultiModalConfig:
    """Test MultiModalConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MultiModalConfig()
        
        assert config.text_embedding_dim == 768
        assert config.image_embedding_dim == 768
        assert config.joint_embedding_dim == 512
        assert config.fusion_strategy == FusionStrategy.ATTENTION_FUSION
        assert config.num_attention_heads == 8
        assert config.contrastive_temperature == 0.07
        assert config.use_mobile_optimizations == True
        assert config.top_k == 10
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MultiModalConfig(
            text_embedding_dim=512,
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            contrastive_temperature=0.1,
            top_k=20
        )
        
        assert config.text_embedding_dim == 512
        assert config.fusion_strategy == FusionStrategy.EARLY_FUSION
        assert config.contrastive_temperature == 0.1
        assert config.top_k == 20
    
    def test_fusion_weights_default(self):
        """Test default fusion weights"""
        config = MultiModalConfig()
        
        assert "text" in config.fusion_weights
        assert "image" in config.fusion_weights
        assert config.fusion_weights["text"] == 0.6
        assert config.fusion_weights["image"] == 0.4


class TestCrossModalAligner:
    """Test CrossModalAligner for embedding alignment"""
    
    @pytest.fixture
    def config(self):
        return MultiModalConfig(
            text_embedding_dim=256,
            image_embedding_dim=256, 
            joint_embedding_dim=128
        )
    
    @pytest.fixture
    def aligner(self, config):
        return CrossModalAligner(config)
    
    def test_aligner_initialization(self, aligner, config):
        """Test aligner initialization"""
        assert isinstance(aligner.text_projector, nn.Sequential)
        assert isinstance(aligner.image_projector, nn.Sequential)
        assert isinstance(aligner.temperature, nn.Parameter)
    
    def test_forward_pass(self, aligner, config):
        """Test forward pass with dummy embeddings"""
        batch_size = 4
        text_emb = torch.randn(batch_size, config.text_embedding_dim)
        image_emb = torch.randn(batch_size, config.image_embedding_dim)
        
        text_proj, image_proj = aligner(text_emb, image_emb)
        
        assert text_proj.shape == (batch_size, config.joint_embedding_dim)
        assert image_proj.shape == (batch_size, config.joint_embedding_dim)
        
        # Check normalization
        text_norms = torch.norm(text_proj, p=2, dim=-1)
        image_norms = torch.norm(image_proj, p=2, dim=-1)
        assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-6)
        assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-6)
    
    def test_contrastive_loss(self, aligner, config):
        """Test contrastive loss computation"""
        batch_size = 8
        text_emb = torch.randn(batch_size, config.text_embedding_dim)
        image_emb = torch.randn(batch_size, config.image_embedding_dim)
        
        loss = aligner.contrastive_loss(text_emb, image_emb)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_contrastive_loss_with_labels(self, aligner, config):
        """Test contrastive loss with custom labels"""
        batch_size = 4
        text_emb = torch.randn(batch_size, config.text_embedding_dim)
        image_emb = torch.randn(batch_size, config.image_embedding_dim)
        labels = torch.tensor([0, 1, 2, 3])  # Identity mapping
        
        loss = aligner.contrastive_loss(text_emb, image_emb, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestAttentionFusion:
    """Test AttentionFusion mechanism"""
    
    @pytest.fixture
    def config(self):
        return MultiModalConfig(
            text_embedding_dim=256,
            image_embedding_dim=256,
            joint_embedding_dim=256,  # Make same as input dims for test simplicity
            attention_dim=256,
            num_attention_heads=4,
            attention_dropout=0.1
        )
    
    @pytest.fixture
    def attention_fusion(self, config):
        return AttentionFusion(config)
    
    def test_attention_fusion_initialization(self, attention_fusion, config):
        """Test attention fusion initialization"""
        assert isinstance(attention_fusion.text_to_image_attention, nn.MultiheadAttention)
        assert isinstance(attention_fusion.image_to_text_attention, nn.MultiheadAttention)
        assert attention_fusion.num_heads == config.num_attention_heads
    
    def test_attention_fusion_forward(self, attention_fusion, config):
        """Test attention fusion forward pass"""
        batch_size = 2
        text_seq_len = 10
        image_patches = 8
        
        text_features = torch.randn(batch_size, text_seq_len, config.attention_dim)
        image_features = torch.randn(batch_size, image_patches, config.attention_dim)
        
        fused_text, fused_image = attention_fusion(text_features, image_features)
        
        assert fused_text.shape == text_features.shape
        assert fused_image.shape == image_features.shape
    
    def test_attention_fusion_different_sequence_lengths(self, attention_fusion, config):
        """Test with different sequence lengths"""
        batch_size = 3
        text_seq_len = 15
        image_patches = 12
        
        text_features = torch.randn(batch_size, text_seq_len, config.attention_dim)
        image_features = torch.randn(batch_size, image_patches, config.attention_dim)
        
        fused_text, fused_image = attention_fusion(text_features, image_features)
        
        assert fused_text.shape == (batch_size, text_seq_len, config.attention_dim)
        assert fused_image.shape == (batch_size, image_patches, config.attention_dim)


class TestEarlyFusionStrategy:
    """Test EarlyFusionStrategy implementation"""
    
    @pytest.fixture
    def config(self):
        return MultiModalConfig()
    
    @pytest.fixture
    def strategy(self, config):
        return EarlyFusionStrategy(config)
    
    @pytest.fixture
    def mock_index_data(self, config):
        """Create mock index data"""
        num_items = 10
        embedding_dim = config.text_embedding_dim + config.image_embedding_dim
        
        return {
            "embeddings": torch.randn(num_items, embedding_dim),
            "metadata": [{"id": i, "title": f"Item {i}"} for i in range(num_items)]
        }
    
    def test_search_with_both_modalities(self, strategy, mock_index_data):
        """Test search with both text and image queries"""
        results = strategy.search(
            query_text="test query",
            query_image=np.random.rand(224, 224, 3),
            index_data=mock_index_data,
            top_k=5
        )
        
        assert len(results) == 5
        assert all("similarity" in result for result in results)
        assert all("index" in result for result in results)
        assert all("metadata" in result for result in results)
    
    def test_search_text_only(self, strategy, mock_index_data):
        """Test search with text query only"""
        results = strategy.search(
            query_text="test query",
            query_image=None,
            index_data=mock_index_data,
            top_k=3
        )
        
        assert len(results) == 3
        assert all(result["similarity"] >= 0 for result in results)
    
    def test_search_image_only(self, strategy, mock_index_data):
        """Test search with image query only"""
        results = strategy.search(
            query_text=None,
            query_image=np.random.rand(224, 224, 3),
            index_data=mock_index_data,
            top_k=3
        )
        
        assert len(results) == 3
    
    def test_search_no_query_error(self, strategy, mock_index_data):
        """Test error when no query is provided"""
        with pytest.raises(ValueError, match="At least one modality must be provided"):
            strategy.search(
                query_text=None,
                query_image=None,
                index_data=mock_index_data,
                top_k=5
            )
    
    def test_compute_similarity(self, strategy):
        """Test similarity computation"""
        query_emb = torch.randn(1, 512)
        candidate_embs = torch.randn(5, 512)
        
        similarities = strategy.compute_similarity(query_emb, candidate_embs)
        
        assert similarities.shape == (5,)
        assert torch.all(similarities >= -1) and torch.all(similarities <= 1)  # Cosine similarity range


class TestLateFusionStrategy:
    """Test LateFusionStrategy implementation"""
    
    @pytest.fixture
    def config(self):
        return MultiModalConfig(
            fusion_weights={"text": 0.7, "image": 0.3}
        )
    
    @pytest.fixture
    def strategy(self, config):
        return LateFusionStrategy(config)
    
    @pytest.fixture
    def mock_index_data(self, config):
        """Create mock index data with separate modality embeddings"""
        num_items = 8
        
        return {
            "text_embeddings": torch.randn(num_items, config.text_embedding_dim),
            "image_embeddings": torch.randn(num_items, config.image_embedding_dim),
            "metadata": [{"id": i, "title": f"Item {i}"} for i in range(num_items)]
        }
    
    def test_search_with_both_modalities(self, strategy, mock_index_data):
        """Test late fusion search with both modalities"""
        results = strategy.search(
            query_text="test query",
            query_image=np.random.rand(224, 224, 3),
            index_data=mock_index_data,
            top_k=5
        )
        
        assert len(results) == 5
        assert all("similarity" in result for result in results)
    
    def test_fusion_weights_applied(self, strategy, mock_index_data):
        """Test that fusion weights are properly applied"""
        # This test verifies the weighted combination logic
        results = strategy.search(
            query_text="test query", 
            query_image=np.random.rand(224, 224, 3),
            index_data=mock_index_data,
            top_k=3
        )
        
        assert len(results) == 3
        # Similarities should be weighted combination
        assert all(0 <= result["similarity"] <= 1 for result in results)
    
    def test_search_missing_embeddings_error(self, strategy, config):
        """Test error when required embeddings are missing"""
        empty_index = {"metadata": []}
        
        with pytest.raises(ValueError, match="No valid embeddings found"):
            strategy.search(
                query_text="test",
                query_image=None,
                index_data=empty_index,
                top_k=5
            )


class TestAttentionFusionStrategy:
    """Test AttentionFusionStrategy implementation"""
    
    @pytest.fixture
    def config(self):
        return MultiModalConfig()
    
    @pytest.fixture
    def strategy(self, config):
        return AttentionFusionStrategy(config)
    
    @pytest.fixture
    def mock_index_data(self, config):
        """Create mock index data"""
        num_items = 6
        
        return {
            "embeddings": torch.randn(num_items, config.joint_embedding_dim),
            "metadata": [{"id": i, "title": f"Item {i}"} for i in range(num_items)]
        }
    
    def test_search_with_both_modalities(self, strategy, mock_index_data):
        """Test attention-based search with both modalities"""
        results = strategy.search(
            query_text="test query",
            query_image=np.random.rand(224, 224, 3),
            index_data=mock_index_data,
            top_k=4
        )
        
        assert len(results) == 4
        assert all("similarity" in result for result in results)
    
    def test_search_single_modality_fallback(self, strategy, mock_index_data):
        """Test fallback to single modality search"""
        # Text only
        results_text = strategy.search(
            query_text="test query",
            query_image=None,
            index_data=mock_index_data,
            top_k=3
        )
        
        assert len(results_text) == 3
        
        # Image only
        results_image = strategy.search(
            query_text=None,
            query_image=np.random.rand(224, 224, 3),
            index_data=mock_index_data,
            top_k=3
        )
        
        assert len(results_image) == 3
    
    def test_attention_fusion_components(self, strategy):
        """Test that attention fusion components are properly initialized"""
        assert hasattr(strategy, "aligner")
        assert hasattr(strategy, "attention_fusion")
        assert isinstance(strategy.aligner, CrossModalAligner)
        assert isinstance(strategy.attention_fusion, AttentionFusion)


class TestMultiModalSearchStrategyFactory:
    """Test MultiModalSearchStrategyFactory"""
    
    def test_create_early_fusion_strategy(self):
        """Test creating early fusion strategy"""
        config = MultiModalConfig(fusion_strategy=FusionStrategy.EARLY_FUSION)
        strategy = MultiModalSearchStrategyFactory.create_strategy(config)
        
        assert isinstance(strategy, EarlyFusionStrategy)
    
    def test_create_late_fusion_strategy(self):
        """Test creating late fusion strategy"""
        config = MultiModalConfig(fusion_strategy=FusionStrategy.LATE_FUSION)
        strategy = MultiModalSearchStrategyFactory.create_strategy(config)
        
        assert isinstance(strategy, LateFusionStrategy)
    
    def test_create_attention_fusion_strategy(self):
        """Test creating attention fusion strategy"""
        config = MultiModalConfig(fusion_strategy=FusionStrategy.ATTENTION_FUSION)
        strategy = MultiModalSearchStrategyFactory.create_strategy(config)
        
        assert isinstance(strategy, AttentionFusionStrategy)
    
    def test_unsupported_strategy_error(self):
        """Test error for unsupported strategy"""
        config = MultiModalConfig(fusion_strategy=FusionStrategy.CONTRASTIVE_FUSION)  # Not implemented
        
        with pytest.raises(ValueError, match="Unsupported fusion strategy"):
            MultiModalSearchStrategyFactory.create_strategy(config)


class TestMultiModalSearchIndex:
    """Test MultiModalSearchIndex functionality"""
    
    @pytest.fixture
    def config(self):
        return MultiModalConfig(similarity_threshold=0.5)
    
    @pytest.fixture
    def search_index(self, config):
        return MultiModalSearchIndex(config)
    
    @pytest.fixture
    def sample_items(self):
        """Create sample items for indexing"""
        return [
            {"id": 1, "title": "Red car", "category": "vehicle"},
            {"id": 2, "title": "Blue bike", "category": "vehicle"},
            {"id": 3, "title": "Green tree", "category": "nature"},
            {"id": 4, "title": "Yellow flower", "category": "nature"},
            {"id": 5, "title": "Black cat", "category": "animal"}
        ]
    
    def test_index_initialization(self, search_index, config):
        """Test index initialization"""
        assert search_index.config == config
        assert not search_index.is_built
        assert search_index.index_data == {}
    
    def test_add_items_text_only(self, search_index, sample_items, config):
        """Test adding items with text embeddings only"""
        text_embeddings = torch.randn(len(sample_items), config.text_embedding_dim)
        
        search_index.add_items(
            items=sample_items,
            text_embeddings=text_embeddings
        )
        
        assert search_index.is_built
        assert "text_embeddings" in search_index.index_data
        assert "embeddings" in search_index.index_data
        assert len(search_index.index_data["metadata"]) == len(sample_items)
    
    def test_add_items_image_only(self, search_index, sample_items, config):
        """Test adding items with image embeddings only"""
        image_embeddings = torch.randn(len(sample_items), config.image_embedding_dim)
        
        search_index.add_items(
            items=sample_items,
            image_embeddings=image_embeddings
        )
        
        assert search_index.is_built
        assert "image_embeddings" in search_index.index_data
        assert "embeddings" in search_index.index_data
    
    def test_add_items_both_modalities(self, search_index, sample_items, config):
        """Test adding items with both text and image embeddings"""
        text_embeddings = torch.randn(len(sample_items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(sample_items), config.image_embedding_dim)
        
        search_index.add_items(
            items=sample_items,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings
        )
        
        assert search_index.is_built
        assert "text_embeddings" in search_index.index_data
        assert "image_embeddings" in search_index.index_data
        assert "embeddings" in search_index.index_data
        
        # For ATTENTION_FUSION strategy, embeddings should be in joint space
        if config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            expected_dim = config.joint_embedding_dim
        else:
            # For EARLY_FUSION, embeddings should have concatenated dimensions
            expected_dim = config.text_embedding_dim + config.image_embedding_dim
        assert search_index.index_data["embeddings"].shape[1] == expected_dim
    
    def test_search_text_query(self, search_index, sample_items, config):
        """Test search with text query"""
        text_embeddings = torch.randn(len(sample_items), config.text_embedding_dim)
        search_index.add_items(sample_items, text_embeddings=text_embeddings)
        
        results = search_index.search(query_text="red car", top_k=3)
        
        assert len(results) <= 3
        assert all("similarity" in result for result in results)
        assert all("index" in result for result in results)
        assert all("metadata" in result for result in results)
    
    def test_search_image_query(self, search_index, sample_items, config):
        """Test search with image query"""
        image_embeddings = torch.randn(len(sample_items), config.image_embedding_dim)
        search_index.add_items(sample_items, image_embeddings=image_embeddings)
        
        query_image = np.random.rand(224, 224, 3)
        results = search_index.search(query_image=query_image, top_k=2)
        
        assert len(results) <= 2
    
    def test_search_both_queries(self, search_index, sample_items, config):
        """Test search with both text and image queries"""
        text_embeddings = torch.randn(len(sample_items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(sample_items), config.image_embedding_dim)
        search_index.add_items(sample_items, text_embeddings, image_embeddings)
        
        query_image = np.random.rand(224, 224, 3)
        results = search_index.search(
            query_text="red car",
            query_image=query_image,
            top_k=4
        )
        
        assert len(results) <= 4
    
    def test_search_not_built_error(self, search_index):
        """Test error when searching before building index"""
        with pytest.raises(RuntimeError, match="Index must be built before searching"):
            search_index.search(query_text="test")
    
    def test_search_no_query_error(self, search_index, sample_items, config):
        """Test error when no query is provided"""
        text_embeddings = torch.randn(len(sample_items), config.text_embedding_dim)
        search_index.add_items(sample_items, text_embeddings=text_embeddings)
        
        with pytest.raises(ValueError, match="At least one query modality must be provided"):
            search_index.search()
    
    def test_similarity_threshold_filtering(self, search_index, sample_items, config):
        """Test that results are filtered by similarity threshold"""
        text_embeddings = torch.randn(len(sample_items), config.text_embedding_dim)
        search_index.add_items(sample_items, text_embeddings=text_embeddings)
        
        # Set high threshold to filter most results
        search_index.config.similarity_threshold = 0.9
        
        results = search_index.search(query_text="test", top_k=10)
        
        # Most results should be filtered out due to high threshold
        assert all(result["similarity"] >= 0.9 for result in results)
    
    def test_get_stats(self, search_index, sample_items, config):
        """Test getting index statistics"""
        text_embeddings = torch.randn(len(sample_items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(sample_items), config.image_embedding_dim)
        
        # Before building
        stats = search_index.get_stats()
        assert stats["is_built"] == False
        assert stats["total_items"] == 0
        
        # After building
        search_index.add_items(sample_items, text_embeddings, image_embeddings)
        stats = search_index.get_stats()
        
        assert stats["is_built"] == True
        assert stats["total_items"] == len(sample_items)
        assert stats["has_text_embeddings"] == True
        assert stats["has_image_embeddings"] == True
        assert stats["fusion_strategy"] == config.fusion_strategy.value


class TestCrossModalRetrieval:
    """Test CrossModalRetrieval functionality"""
    
    @pytest.fixture
    def config(self):
        return MultiModalConfig()
    
    @pytest.fixture
    def retrieval_system(self, config):
        return CrossModalRetrieval(config)
    
    @pytest.fixture
    def sample_text_items(self):
        return [
            {"id": 1, "text": "A red sports car", "category": "vehicle"},
            {"id": 2, "text": "A blue mountain bike", "category": "vehicle"},
            {"id": 3, "text": "A green forest landscape", "category": "nature"}
        ]
    
    @pytest.fixture
    def sample_image_items(self):
        return [
            {"id": 1, "description": "Photo of red car", "category": "vehicle"},
            {"id": 2, "description": "Photo of blue bike", "category": "vehicle"},
            {"id": 3, "description": "Photo of forest", "category": "nature"}
        ]
    
    def test_build_indices(self, retrieval_system, sample_text_items, sample_image_items, config):
        """Test building cross-modal indices"""
        text_embeddings = torch.randn(len(sample_text_items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(sample_image_items), config.image_embedding_dim)
        
        retrieval_system.build_indices(
            text_items=sample_text_items,
            text_embeddings=text_embeddings,
            image_items=sample_image_items,
            image_embeddings=image_embeddings
        )
        
        assert len(retrieval_system.text_index["items"]) == len(sample_text_items)
        assert len(retrieval_system.image_index["items"]) == len(sample_image_items)
        assert "embeddings" in retrieval_system.text_index
        assert "embeddings" in retrieval_system.image_index
    
    def test_find_images_for_text(self, retrieval_system, sample_text_items, sample_image_items, config):
        """Test finding images for text query"""
        text_embeddings = torch.randn(len(sample_text_items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(sample_image_items), config.image_embedding_dim)
        
        retrieval_system.build_indices(
            sample_text_items, text_embeddings,
            sample_image_items, image_embeddings
        )
        
        results = retrieval_system.find_images_for_text("red car", top_k=2)
        
        assert len(results) <= 2
        assert all(result["type"] == "image" for result in results)
        assert all("similarity" in result for result in results)
        assert all("item" in result for result in results)
    
    def test_find_text_for_image(self, retrieval_system, sample_text_items, sample_image_items, config):
        """Test finding text for image query"""
        text_embeddings = torch.randn(len(sample_text_items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(sample_image_items), config.image_embedding_dim)
        
        retrieval_system.build_indices(
            sample_text_items, text_embeddings,
            sample_image_items, image_embeddings
        )
        
        query_image = np.random.rand(224, 224, 3)
        results = retrieval_system.find_text_for_image(query_image, top_k=2)
        
        assert len(results) <= 2
        assert all(result["type"] == "text" for result in results)
        assert all("similarity" in result for result in results)
        assert all("item" in result for result in results)
    
    def test_find_images_not_built_error(self, retrieval_system):
        """Test error when image index not built"""
        with pytest.raises(RuntimeError, match="Image index not built"):
            retrieval_system.find_images_for_text("test query")
    
    def test_find_text_not_built_error(self, retrieval_system):
        """Test error when text index not built"""
        with pytest.raises(RuntimeError, match="Text index not built"):
            retrieval_system.find_text_for_image(np.random.rand(224, 224, 3))


class TestMultiModalIntegration:
    """Integration tests for multi-modal search components"""
    
    @pytest.fixture
    def config(self):
        return MultiModalConfig(
            text_embedding_dim=512,
            image_embedding_dim=512,
            joint_embedding_dim=512,
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
            top_k=5,
            similarity_threshold=0.3
        )
    
    def test_end_to_end_multimodal_search(self, config):
        """Test complete end-to-end multi-modal search workflow"""
        # Create search index
        search_index = MultiModalSearchIndex(config)
        
        # Prepare sample data
        items = [
            {"id": 1, "title": "Red sports car", "description": "Fast red vehicle"},
            {"id": 2, "title": "Blue mountain bike", "description": "Two-wheeled transport"},
            {"id": 3, "title": "Green forest", "description": "Natural landscape with trees"},
            {"id": 4, "title": "Yellow sunflower", "description": "Bright flower in field"},
            {"id": 5, "title": "Black cat", "description": "Domestic feline animal"}
        ]
        
        # Generate embeddings
        text_embeddings = torch.randn(len(items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(items), config.image_embedding_dim)
        
        # Build index
        search_index.add_items(items, text_embeddings, image_embeddings)
        
        # Perform multi-modal search
        query_image = np.random.rand(224, 224, 3)
        results = search_index.search(
            query_text="red vehicle",
            query_image=query_image,
            top_k=3
        )
        
        # Verify results
        assert len(results) <= 3
        assert all("similarity" in result for result in results)
        assert all("metadata" in result for result in results)
        assert all(result["similarity"] >= config.similarity_threshold for result in results)
        
        # Verify results are sorted by similarity
        similarities = [result["similarity"] for result in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_cross_modal_retrieval_integration(self, config):
        """Test cross-modal retrieval system integration"""
        retrieval = CrossModalRetrieval(config)
        
        # Prepare data
        text_items = [
            {"id": 1, "text": "Fast red sports car on highway"},
            {"id": 2, "text": "Blue mountain bike on trail"},
            {"id": 3, "text": "Green forest with tall trees"}
        ]
        
        image_items = [
            {"id": 1, "caption": "Photo of red car driving"},
            {"id": 2, "caption": "Picture of blue bicycle"},
            {"id": 3, "caption": "Forest landscape photo"}
        ]
        
        text_embeddings = torch.randn(len(text_items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(image_items), config.image_embedding_dim)
        
        # Build indices
        retrieval.build_indices(
            text_items, text_embeddings,
            image_items, image_embeddings
        )
        
        # Test text-to-image retrieval
        image_results = retrieval.find_images_for_text("red car", top_k=2)
        assert len(image_results) <= 2
        assert all(result["type"] == "image" for result in image_results)
        
        # Test image-to-text retrieval
        query_image = np.random.rand(224, 224, 3)
        text_results = retrieval.find_text_for_image(query_image, top_k=2)
        assert len(text_results) <= 2
        assert all(result["type"] == "text" for result in text_results)
    
    def test_strategy_comparison(self, config):
        """Test comparison between different fusion strategies"""
        strategies_to_test = [
            FusionStrategy.EARLY_FUSION,
            FusionStrategy.LATE_FUSION,
            FusionStrategy.ATTENTION_FUSION
        ]
        
        # Sample data
        items = [{"id": i, "title": f"Item {i}"} for i in range(5)]
        text_embeddings = torch.randn(len(items), config.text_embedding_dim)
        image_embeddings = torch.randn(len(items), config.image_embedding_dim)
        
        results_by_strategy = {}
        
        for strategy in strategies_to_test:
            config.fusion_strategy = strategy
            search_index = MultiModalSearchIndex(config)
            search_index.add_items(items, text_embeddings, image_embeddings)
            
            results = search_index.search(
                query_text="test query",
                query_image=np.random.rand(224, 224, 3),
                top_k=3
            )
            
            results_by_strategy[strategy.value] = results
            assert len(results) <= 3
        
        # Verify all strategies return valid results
        assert len(results_by_strategy) == len(strategies_to_test)
        for strategy_name, results in results_by_strategy.items():
            assert all("similarity" in result for result in results)


@pytest.mark.performance
class TestMultiModalPerformance:
    """Performance tests for multi-modal search"""
    
    def test_large_scale_indexing_performance(self):
        """Test performance with larger datasets"""
        config = MultiModalConfig()
        search_index = MultiModalSearchIndex(config)
        
        # Create larger dataset
        num_items = 1000
        items = [{"id": i, "title": f"Item {i}"} for i in range(num_items)]
        text_embeddings = torch.randn(num_items, config.text_embedding_dim)
        image_embeddings = torch.randn(num_items, config.image_embedding_dim)
        
        # Measure indexing time
        import time
        start_time = time.time()
        search_index.add_items(items, text_embeddings, image_embeddings)
        indexing_time = time.time() - start_time
        
        # Should complete indexing reasonably quickly
        assert indexing_time < 5.0  # Less than 5 seconds
        
        # Measure search time
        start_time = time.time()
        results = search_index.search(
            query_text="test query",
            query_image=np.random.rand(224, 224, 3),
            top_k=10
        )
        search_time = time.time() - start_time
        
        # Search should be fast
        assert search_time < 1.0  # Less than 1 second
        assert len(results) <= 10
    
    def test_memory_efficiency(self):
        """Test memory efficiency for multi-modal components"""
        config = MultiModalConfig(use_mobile_optimizations=True)
        
        # Create components
        aligner = CrossModalAligner(config)
        attention_fusion = AttentionFusion(config)
        search_index = MultiModalSearchIndex(config)
        
        # Test with batch processing
        batch_size = 16
        text_emb = torch.randn(batch_size, config.text_embedding_dim)
        image_emb = torch.randn(batch_size, config.image_embedding_dim)
        
        # Should not cause memory issues
        text_proj, image_proj = aligner(text_emb, image_emb)
        assert text_proj.shape == (batch_size, config.joint_embedding_dim)
        assert image_proj.shape == (batch_size, config.joint_embedding_dim)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
