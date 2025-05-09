"""
Temel model sınıfı için birim testler.
"""

import os
import tempfile

import pytest
import torch

from m3tm.config.model_config import M3TMConfig, get_tiny_config
from m3tm.core.base_model import BaseModel


class TestModelImpl(BaseModel):
    """Test için BaseModel uygulaması."""
    
    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


def test_base_model_init():
    """BaseModel'in doğru şekilde başlatılıp başlatılmadığını kontrol eder."""
    config = get_tiny_config()
    model = TestModelImpl(config)
    
    assert model.config == config
    assert isinstance(model.linear, torch.nn.Linear)


def test_get_param_count():
    """get_param_count metodunun doğru parametre sayılarını döndürdüğünü kontrol eder."""
    config = get_tiny_config()
    model = TestModelImpl(config)
    
    param_count = model.get_param_count()
    
    # Linear(10, 5) = (10 * 5) + 5 (ağırlıklar + bias) = 55 parametre
    assert param_count["total"] == 55
    assert param_count["trainable"] == 55
    assert param_count["frozen"] == 0
    
    # Parametrelerin bir kısmını dondur
    model.linear.weight.requires_grad = False
    
    param_count = model.get_param_count()
    
    # Linear(10, 5) = (10 * 5) + 5 (ağırlıklar + bias) = 55 parametre
    # Dondurulan: (10 * 5) = 50 parametre
    # Eğitilebilir: 5 (bias)
    assert param_count["total"] == 55
    assert param_count["trainable"] == 5
    assert param_count["frozen"] == 50


def test_save_and_load_pretrained():
    """save_pretrained ve from_pretrained metodlarının doğru çalıştığını kontrol eder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Model oluştur ve kaydet
        config = get_tiny_config()
        model = TestModelImpl(config)
        
        # Modele bazı ağırlıklar ver
        torch.nn.init.constant_(model.linear.weight, 0.5)
        torch.nn.init.constant_(model.linear.bias, 0.1)
        
        model.save_pretrained(tmpdir)
        
        # Kaydedilen dosyaların var olduğunu kontrol et
        assert os.path.exists(os.path.join(tmpdir, "model.pt"))
        assert os.path.exists(os.path.join(tmpdir, "config.pt"))
        
        # Modeli yükle
        loaded_model = TestModelImpl.from_pretrained(tmpdir)
        
        # Yüklenen modelin yapılandırmasını kontrol et
        assert loaded_model.config.text_config.vocab_size == config.text_config.vocab_size
        assert loaded_model.config.image_config.embed_dim == config.image_config.embed_dim
        
        # Yüklenen modelin ağırlıklarını kontrol et
        assert torch.allclose(loaded_model.linear.weight, torch.full_like(loaded_model.linear.weight, 0.5))
        assert torch.allclose(loaded_model.linear.bias, torch.full_like(loaded_model.linear.bias, 0.1))


def test_forward_not_implemented():
    """BaseModel.forward metodunun NotImplementedError yükselttiğini kontrol eder."""
    config = get_tiny_config()
    model = BaseModel(config)
    
    with pytest.raises(NotImplementedError):
        model.forward(torch.randn(1, 10)) 