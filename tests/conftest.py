"""
Pytest yapılandırması ve test ortamı kurulumu
"""
import os
import sys
import pytest
import torch
import numpy as np
import random

# Çalışma dizinine src eklemek için
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture(scope="session")
def seed_everything():
    """Test tekrarlanabilirliği için seed ayarı"""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
@pytest.fixture(scope="session")
def device():
    """Test için cihaz ayarı"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def test_data_dir():
    """Test verileri dizini"""
    return os.path.join(os.path.dirname(__file__), 'test_data')

@pytest.fixture(scope="function")
def mock_image_tensor():
    """Test için sahte görüntü verisi oluşturur"""
    return torch.rand(2, 3, 224, 224)  # batch_size, channels, height, width

@pytest.fixture(scope="function")
def mock_text_tensor():
    """Test için sahte metin verisi oluşturur"""
    # token_id'leri içeren sahte batch
    return torch.randint(0, 1000, (2, 32))  # batch_size, sequence_length

@pytest.fixture(scope="function")
def mock_mixed_batch():
    """Test için metin ve görüntü içeren karma batch"""
    return {
        'text': torch.randint(0, 1000, (2, 32)),
        'image': torch.rand(2, 3, 224, 224)
    }

@pytest.fixture(scope="function")
def create_temp_directory(tmpdir_factory):
    """Test için geçici dizin oluşturur"""
    temp_dir = tmpdir_factory.mktemp("temp_test_dir")
    return temp_dir

@pytest.fixture(scope="function")
def create_model_configs():
    """Test için model konfigürasyonları oluşturur"""
    from m3tm.config.config_base import ConfigBase
    from dataclasses import dataclass
    
    @dataclass
    class MockTextConfig(ConfigBase):
        vocab_size: int = 1000
        embed_dim: int = 32
        max_seq_len: int = 128
        pad_token_id: int = 0
    
    @dataclass
    class MockImageConfig(ConfigBase):
        patch_size: int = 16
        in_channels: int = 3
        embed_dim: int = 32
        image_size: int = 224
    
    @dataclass
    class MockTransformerConfig(ConfigBase):
        embed_dim: int = 32
        num_heads: int = 2
        mlp_dim: int = 64
        dropout: float = 0.1
        
    @dataclass
    class MockModelConfig(ConfigBase):
        text_config: MockTextConfig = MockTextConfig()
        image_config: MockImageConfig = MockImageConfig()
        transformer_config: MockTransformerConfig = MockTransformerConfig()
        num_core_blocks: int = 2
        fused_embed_dim: int = 64
        search_embed_dim: int = 128
    
    return MockModelConfig() 