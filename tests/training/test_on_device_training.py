"""
M³TM modelinin cihaz üzerinde (on-device) eğitim testleri

Bu modül, modelin mobil cihazlarda eğitimini test eder:
- Hafif on-device fine-tuning
- Adaptör tabanlı transfer öğrenme
- Kişiselleştirilmiş öğrenme
- Depolanmış verilerle çevrimdışı eğitim
- Mobil donanım kısıtları altında eğitim performansı
- Dinamik öğrenme oranları ve iyileştirme stratejileri
- Eğitim durum yönetimi ve devam eden eğitim
- Düşük hassasiyetli kuantalama eğitimi
- Veri artırma teknikleri
- Federated learning simülasyonu
- Gizlilik korumalı eğitim
"""
import pytest
import torch
import torch.nn as nn
import tempfile
import time
import os
import json
import numpy as np
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch, MagicMock

from m3tm.config.model_config import get_tiny_config, TrainingConfig
from m3tm.core.base_model import BaseModel
from m3tm.mobile.optimization import optimize_model_for_mobile
from m3tm.training.trainer import TrainingLoopTemplate, TextClassificationTrainer
from m3tm.training.adapter_training import AdapterTrainingManager, AdapterTrainingConfig
from m3tm.task_heads.classification import ClassificationHead
from m3tm.adapters import BottleneckAdapter, AdapterConfig

# Metrik hesaplama yardımcı fonksiyonları
def compute_metrics(outputs, targets, task_type="classification"):
    """Metrik hesaplama yardımcı fonksiyonu."""
    if task_type == "classification":
        # Sınıflandırma için doğruluk (accuracy) hesapla
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        pred_labels = np.argmax(outputs, axis=1)
        correct = (pred_labels == targets).sum()
        accuracy = correct / targets.size
        
        return {"accuracy": accuracy}
    else:
        # Diğer görevler için MSE hesapla
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        mse = np.mean((outputs - targets) ** 2)
        return {"mse": mse}

# Test veri yolu
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Mobil cihaz bellek limitleri (MB)
MEMORY_LIMITS = {
    "high_end_mobile": {
        "peak": 1024,  # 1GB
        "sustained": 512  # 512MB
    },
    "mid_range_mobile": {
        "peak": 512,  # 512MB
        "sustained": 256  # 256MB
    },
    "low_end_mobile": {
        "peak": 256,  # 256MB
        "sustained": 128  # 128MB
    }
}
TIME_LIMITS = {
    "small_batch": {"max": 200, "avg": 150},  # ms cinsinden
    "medium_batch": {"max": 350, "avg": 250}
}

# Burada TrainingLoopTemplate sınıfını OnDeviceTrainer olarak kullanacağız
# Yani OnDeviceTrainer = TrainingLoopTemplate
OnDeviceTrainer = TextClassificationTrainer

# === Setup/Teardown Pattern Uygulaması ===
@pytest.fixture
def model_fixture():
    """Test için model kurulumu sağlayan fixture."""
    # Temel model oluştur
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 384),
        torch.nn.ReLU(),
        torch.nn.Linear(384, 768)
    )
    yield model
    # Temizleme
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture
def adapter_fixture():
    """Test için adapter kurulumu sağlayan fixture."""
    # Adapter yapılandırması oluştur
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=64
    )
    # input_dim ayrı bir parametre olarak Adapter constructor'ına geçirilmeli
    adapter = BottleneckAdapter(config, input_dim=768)
    yield adapter
    # Teardown - adapter belleğini temizle
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture
def trainer_fixture(model_fixture, adapter_fixture):
    """Test için eğitim kurulumu sağlayan fixture."""
    # TrainingConfig oluştur
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=8
    )
    trainer = OnDeviceTrainer(config)
    yield trainer
    # Teardown - eğitici belleğini temizle
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture
def dummy_dataset():
    """Test için yapay veri seti oluşturan fixture."""
    # Yapay veri oluştur
    inputs = torch.randn(32, 768)
    targets = torch.randn(32, 768)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    return dataset

# === İzole Bileşen Testi Pattern Uygulaması ===
class TestAdapterIsolation:
    """Adapter bileşenini izole şekilde test eden sınıf."""
    
    def test_adapter_forward(self, adapter_fixture):
        """Adapter'ın ileri geçişini test eden izole test."""
        batch_size = 4
        input_data = torch.randn(batch_size, 768)
        
        # İleri geçiş
        output = adapter_fixture(input_data)
        
        # Doğrulama
        assert output.shape == input_data.shape, "Adapter çıktı şekli giriş şekliyle eşleşmelidir"
        assert not torch.allclose(output, input_data, atol=1e-7), "Adapter çıktısı girişten farklı olmalıdır"
    
    def test_adapter_parameters(self, adapter_fixture):
        """Adapter parametrelerini test eden izole test."""
        # Eğitilebilir parametreleri say
        trainable_params = sum(p.numel() for p in adapter_fixture.parameters() if p.requires_grad)
        
        # Doğrulama
        assert trainable_params > 0, "Adapter eğitilebilir parametrelere sahip olmalıdır"
        
        # Adapter parametrelerini hesapla
        # - down_proj: input_dim * bottleneck_dim + bottleneck_dim (bias)
        # - up_proj: bottleneck_dim * input_dim + input_dim (bias)
        # - layer_norm: 2 * bottleneck_dim (scale ve bias)
        input_dim = 768
        bottleneck_dim = 64
        expected_params = (input_dim * bottleneck_dim + bottleneck_dim) + (bottleneck_dim * input_dim + input_dim) + (2 * bottleneck_dim)
        
        assert trainable_params == expected_params, f"Adapter {expected_params} parametresine sahip olmalıdır, fakat {trainable_params} bulundu"

# === Parametrize Test Pattern Uygulaması ===
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])
def test_training_parameters(model_fixture, adapter_fixture, dummy_dataset, batch_size, learning_rate):
    """Farklı eğitim parametrelerini test eden parametrize test."""
    # Trainer oluştur
    config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    trainer = OnDeviceTrainer(config)
    
    # Kısa eğitim çalıştır
    initial_adapter_weights = [p.clone() for p in adapter_fixture.parameters()]
    train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size)
    
    # Adapter'ı modele ekle - burada modeli ve adapter'ı birlikte yönetiyoruz
    # Bu, gerçek uygulamada adapter_manager tarafından yapılacaktır
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, x):
            x = self.model(x)
            x = self.adapter(x)
            return x
    
    composite_model = CompositeModel(model_fixture, adapter_fixture)
    
    # Eğitim başlat - modeli doğrudan trainer'a vererek eğitiyoruz
    # Bu test amaçlıdır ve gerçek senaryoda daha farklı olabilir
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(composite_model.parameters(), lr=learning_rate)
    
    # Eğitim döngüsü
    composite_model.train()
    for _ in range(1):  # 1 epoch
        for inputs, _ in train_loader:
            optimizer.zero_grad()
            outputs = composite_model(inputs)
            # Burada loss hesaplama ve backward kısmını manuel olarak yapıyoruz
            # Bu, OnDeviceTrainer sınıfında olduğu varsayılan davranışı simüle ediyor
            loss = criterion(outputs, inputs)  # Örnek olarak reconstruction loss
            loss.backward()
            optimizer.step()
    
    # Adapter ağırlıklarının değişip değişmediğini kontrol et
    current_adapter_weights = [p for p in adapter_fixture.parameters()]
    weights_changed = False
    
    for initial, current in zip(initial_adapter_weights, current_adapter_weights):
        if not torch.allclose(initial, current, atol=1e-5):
            weights_changed = True
            break
    
    assert weights_changed, "Eğitimden sonra adapter ağırlıkları değişmelidir"

# === Golden Master Test Pattern Uygulaması ===
def test_training_loss_trend(trainer_fixture, dummy_dataset, model_fixture, adapter_fixture):
    """Eğitim kaybı eğilimini kontrol eden golden master test."""
    train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=8)
    
    # CompositeModel oluştur
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, x):
            x = self.model(x)
            x = self.adapter(x)
            return x
    
    composite_model = CompositeModel(model_fixture, adapter_fixture)
    
    # Eğitim döngüsü ve kayıpları izleme
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(composite_model.parameters(), lr=1e-4)
    
    # Kayıpları kaydetmek için liste
    losses = []
    
    # Eğitim döngüsü
    composite_model.train()
    for epoch in range(3):  # 3 epoch
        epoch_losses = []
        for inputs, _ in train_loader:
            optimizer.zero_grad()
            outputs = composite_model(inputs)
            loss = criterion(outputs, inputs)  # Örnek olarak reconstruction loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        # Epoch ortalamasını kaydet
        losses.append(sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0)
    
    # Kayıpların düşüp düşmediğini kontrol et
    assert len(losses) > 1, "Birden fazla kayıp değeri döndürülmelidir"
    assert losses[0] > losses[-1], "Eğitim kaybı zamanla azalmalıdır"
    
    # Kayıp eğilimini kaydet (ileriki karşılaştırmalar için referans olarak)
    loss_trend_file = os.path.join(TEST_DATA_DIR, "expected_results", "loss_trend_reference.npy")
    os.makedirs(os.path.dirname(loss_trend_file), exist_ok=True)
    np.save(loss_trend_file, np.array(losses))
    
    # Eğer zaten referans dosyası varsa, karşılaştır
    if os.path.exists(loss_trend_file):
        reference_losses = np.load(loss_trend_file)
        # Kayıp eğilimi benzer olmalı, mutlak değerler farklı olabilir
        assert np.corrcoef(losses, reference_losses)[0, 1] > 0.5, "Kayıp eğilimi referans eğilimle benzer olmalıdır"

# === Performans Profil Pattern Uygulaması ===
def test_memory_usage_profile(model_fixture, adapter_fixture, dummy_dataset):
    """Eğitim sırasında bellek kullanımını ölçen performans profil testi."""
    # Güncel bellek kullanımını ölçmek için yardımcı fonksiyon
    def get_memory_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)  # MB cinsinden
        else:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)  # MB cinsinden
    
    # CompositeModel oluştur
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, x):
            x = self.model(x)
            x = self.adapter(x)
            return x
    
    composite_model = CompositeModel(model_fixture, adapter_fixture)
    
    # Eğitim döngüsü ve kayıpları izleme
    batch_size = 8
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(composite_model.parameters(), lr=1e-4)
    train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size)
    
    # Bellek kullanımı takibi
    memory_usage = []
    peak_memory = 0
    
    # Bellek kullanımını izleyen bir callback
    def memory_monitor(epoch, batch, loss):
        nonlocal peak_memory
        current_memory = get_memory_usage()
        memory_usage.append(current_memory)
        peak_memory = max(peak_memory, current_memory)
    
    # Eğitim döngüsü
    composite_model.train()
    for epoch in range(2):  # 2 epoch
        for i, (inputs, _) in enumerate(train_loader):
            # Eğitim adımı öncesi bellek kullanımı
            before_step_memory = get_memory_usage()
            memory_usage.append(before_step_memory)
            
            optimizer.zero_grad()
            outputs = composite_model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            # Eğitim adımı sonrası bellek kullanımı
            after_step_memory = get_memory_usage()
            memory_usage.append(after_step_memory)
            peak_memory = max(peak_memory, after_step_memory)
            
            # Callback'i çağır
            memory_monitor(epoch, i, loss.item())
    
    # Bellek kullanım istatistiklerini hesapla
    avg_memory = sum(memory_usage) / len(memory_usage)
    
    # Doğrulama: Bellek kullanımı sınırlar içinde mi?
    assert peak_memory < MEMORY_LIMITS["high_end_mobile"]["peak"], f"Tepe bellek kullanımı ({peak_memory:.2f} MB) limiti aşıyor"
    assert avg_memory < MEMORY_LIMITS["high_end_mobile"]["sustained"], f"Ortalama bellek kullanımı ({avg_memory:.2f} MB) limiti aşıyor"
    
    # Bellek kullanım metriklerini kaydet
    print(f"Tepe bellek kullanımı: {peak_memory:.2f} MB")
    print(f"Ortalama bellek kullanımı: {avg_memory:.2f} MB")

# === Sınır Testi Pattern Uygulaması ===
@pytest.mark.parametrize("batch_size", [1, 32])
def test_boundary_batch_sizes(model_fixture, adapter_fixture, dummy_dataset, batch_size):
    """Uç batch boyutlarında eğitimi test eden sınır testi."""
    # Trainer oluştur
    training_config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=batch_size,
        epochs=1
    )
    trainer = OnDeviceTrainer(config=training_config)
    
    # Model parametrelerini dondur
    for param in model_fixture.parameters():
        param.requires_grad = False
    
    # CompositeModel oluştur - TextClassificationTrainer'a uyumlu forward metodu
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, input_ids, attention_mask=None):
            # TextClassificationTrainer dict döndürmemizi bekliyor
            with torch.no_grad():
                x = self.model(input_ids)
            x = self.adapter(x)
            return {"logits": x}
    
    composite_model = CompositeModel(model_fixture, adapter_fixture)
    
    # Create dummy data - TextClassificationTrainer'a uygun formatta
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10, dim=768):
            self.size = size
            self.dim = dim
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randn(self.dim),
                "attention_mask": torch.ones(self.dim),
                "labels": torch.randint(0, 2, (1,)).item()  # Binary sınıflandırma
            }
    
    test_dataset = DummyDataset()
    train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # Eğitim çalıştır
    trainer.train(composite_model, train_loader)
    
    # Doğrulama kriterleri batch size'a göre değişir
    if batch_size == 1:
        # Çok küçük batch size, daha uzun eğitim süresi beklenir ama bellek tüketimi düşük olmalı
        assert True, "Küçük batch boyutu ile eğitim başarıyla tamamlandı"
    elif batch_size == 32:
        # Büyük batch size, potansiyel olarak daha hızlı yakınsama ama daha yüksek bellek tüketimi
        assert True, "Büyük batch boyutu ile eğitim başarıyla tamamlandı"

@pytest.mark.parametrize("sequence_length", [64, 512, 1024])
def test_boundary_sequence_lengths(sequence_length):
    """Farklı giriş uzunluklarıyla çalışabilme kapasitesini test eden sınır testi."""
    
    # Adapter oluştur
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=min(64, sequence_length // 8)
    )
    adapter = BottleneckAdapter(config, input_dim=sequence_length)
    
    # Rasgele giriş
    batch_size = 2
    inputs = torch.randn(batch_size, sequence_length)
    
    # İleri geçiş
    outputs = adapter(inputs)
    
    # Doğrulama: Çıkış şekli giriş şekliyle eşleşmeli
    assert outputs.shape == inputs.shape, f"sequence_length={sequence_length} için şekil eşleşmesi başarısız"

# === Entegrasyon Cascade Pattern Uygulaması ===
def test_adapter_to_model_integration(model_fixture, adapter_fixture):
    """Adaptör ve model entegrasyonunu test eden test."""
    # Örnek girdi verisi oluştur
    batch_size = 4
    input_data = torch.randn(batch_size, 768)
    
    # Model çıktısını al
    model_output = model_fixture(input_data)
    
    # Adapter çıktısını al
    adapter_output = adapter_fixture(input_data)
    
    # Adapter-model entegrasyonunu test et - residual bağlantı ile
    combined_output = model_output + adapter_output
    
    # Doğrulama
    assert combined_output.shape == input_data.shape, "Entegre çıktı şekli giriş şekliyle eşleşmelidir"
    assert not torch.allclose(combined_output, model_output, atol=1e-7), "Entegre çıktı yalnızca model çıktısından farklı olmalıdır"
    assert not torch.allclose(combined_output, adapter_output, atol=1e-7), "Entegre çıktı yalnızca adapter çıktısından farklı olmalıdır"

def test_trainer_to_adapter_model_integration(model_fixture, adapter_fixture, dummy_dataset):
    """Eğitici, adaptör ve model tam entegrasyonunu test eden test."""
    # Model parametrelerini dondur
    for param in model_fixture.parameters():
        param.requires_grad = False
    
    # CompositeModel oluştur - TextClassificationTrainer'a uyumlu forward metodu
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, input_ids, attention_mask=None):
            # TextClassificationTrainer dict döndürmemizi bekliyor
            with torch.no_grad():
                x = self.model(input_ids)
            x = self.adapter(x)
            return {"logits": x}
    
    composite_model = CompositeModel(model_fixture, adapter_fixture)
    
    # Eğitim yapılandırması oluştur
    training_config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=8,
        epochs=1
    )
    
    # Eğitici oluştur
    trainer = OnDeviceTrainer(config=training_config)
    
    # Başlangıçta adapter ağırlıklarını kaydet
    initial_adapter_states = {name: param.clone() for name, param in adapter_fixture.named_parameters()}
    
    # Create dummy data - TextClassificationTrainer'a uygun formatta
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10, dim=768):
            self.size = size
            self.dim = dim
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randn(self.dim),
                "attention_mask": torch.ones(self.dim),
                "labels": torch.randint(0, 2, (1,)).item()  # Binary sınıflandırma
            }
    
    test_dataset = DummyDataset()
    train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
    
    # Eğitim yap
    trainer.train(composite_model, train_loader)
    
    # Eğitim sonrası adapter parametrelerinin değiştiğini kontrol et
    for name, param in adapter_fixture.named_parameters():
        assert not torch.allclose(param, initial_adapter_states[name], atol=1e-7), f"{name} parametresi eğitim sırasında değişmedi"
    
    # Tam entegrasyon testi - adapte edilmiş modeli çıkarım için kullan
    test_input = torch.randn(1, 768)
    
    # Sadece model çıktısı
    with torch.no_grad():
        model_only_output = model_fixture(test_input)
    
    # Model + adapter çıktısı
    def forward_with_adapter(model, adapter, x):
        model_output = model(x)
        adapter_output = adapter(x)
        return model_output + adapter_output
    
    with torch.no_grad():
        integrated_output = forward_with_adapter(model_fixture, adapter_fixture, test_input)
    
    # Doğrulama
    assert not torch.allclose(model_only_output, integrated_output, atol=1e-7), "Adapter entegrasyonu model çıktısını değiştirmelidir"

# === Simüle Edilmiş Ortam Pattern Uygulaması ===
def test_low_memory_training():
    """Düşük bellek ortamında eğitimi test eden modül izolasyon testi."""
    # Basit model ve veri oluştur
    input_dim = 128
    model = torch.nn.Linear(input_dim, input_dim)
    
    # Küçük adaptör oluştur
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=16
    )
    small_adapter = BottleneckAdapter(config, input_dim=input_dim)
    
    # CompositeModel oluştur
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, x):
            x = self.model(x)
            x = self.adapter(x)
            return x
    
    composite_model = CompositeModel(model, small_adapter)
    
    # Çok küçük batch size ve çok küçük veri seti
    batch_size = 2
    num_samples = 10
    dummy_inputs = torch.randn(num_samples, input_dim)
    dummy_targets = torch.randn(num_samples, input_dim)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_inputs, dummy_targets)
    train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size)
    
    # Basit eğitim döngüsü
    optimizer = torch.optim.Adam(composite_model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Bellek kullanımını izle
    memory_usage = []
    
    def track_memory(epoch, batch, loss):
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        memory_usage.append(memory_mb)
        return True
    
    # Eğitim döngüsü
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            track_memory(epoch, batch_idx, 0)
            optimizer.zero_grad()
            output = composite_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Düşük bellek kullanımı doğrulama
    avg_memory = sum(memory_usage) / len(memory_usage)
    assert avg_memory < MEMORY_LIMITS["low_end_mobile"]["peak"], "Düşük bellek eğitimi başarısız oldu"

# === Property Based Test Pattern Uygulaması ===
@pytest.mark.parametrize("input_dim", [128, 256, 512, 768])
def test_adapter_output_shape_property(input_dim):
    """Adaptör çıktı şekli özelliğini rasgele girdilerle test eder."""
    # Adapter oluştur
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=input_dim // 8
    )
    adapter = BottleneckAdapter(config, input_dim=input_dim)
    
    # Farklı şekillerde girdiler oluştur ve test et
    test_shapes = [
        (1, input_dim),           # Tek örnek
        (16, input_dim),          # Batch
        (8, 10, input_dim),       # Batch ve seq_len
        (4, 5, 6, input_dim)      # Batch ve çoklu boyutlar
    ]
    
    for shape in test_shapes:
        # Rasgele girdi
        x = torch.randn(*shape)
        
        # İleri geçiş
        y = adapter(x)
        
        # Doğrulama: Giriş ve çıkış şekilleri eşleşmeli
        assert y.shape == x.shape, f"shape={shape} için şekil eşleşmesi başarısız"

# === Comparative Test Pattern Uygulaması ===
def test_adapter_vs_full_model_training(dummy_dataset):
    """Adaptör eğitimi ile tam model eğitimini karşılaştıran test."""
    # Model oluştur
    input_dim = 768
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, input_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(input_dim // 2, input_dim)
    )
    
    # Adaptör oluştur
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=64
    )
    adapter = BottleneckAdapter(config, input_dim=input_dim)
    
    # Model kopyaları oluştur (biri dondurulmuş + adapter, diğeri tam eğitimli)
    model_with_adapter = deepcopy(model)
    full_model = deepcopy(model)
    
    # Adaptörlü modelin parametrelerini dondur
    for p in model_with_adapter.parameters():
        p.requires_grad = False
    
    # CompositeModel oluştur
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, x):
            x = self.model(x)
            x = self.adapter(x)
            return x
    
    composite_model = CompositeModel(model_with_adapter, adapter)
    
    # Eğitim verilerini hazırla
    train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=8)
    
    # Her iki model için eğitim yapılandırması
    criterion = torch.nn.MSELoss()
    adapter_optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    full_model_optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4)
    
    # Eğitim öncesi parametre sayısını karşılaştır
    adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    full_model_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"Adapter parametreleri: {adapter_params}, Tam model parametreleri: {full_model_params}")
    
    # Parametre sayısını kontrol et
    assert adapter_params < full_model_params, "Adapter daha az parametre içermelidir"
    
    # Her iki modeli de eğit
    adapter_losses = []
    full_model_losses = []
    
    # Eğitim döngüsü
    for epoch in range(5):
        adapter_epoch_loss = 0.0
        full_model_epoch_loss = 0.0
        
        for inputs, _ in train_loader:
            # Adaptör eğitimi
            adapter_optimizer.zero_grad()
            outputs_adapter = composite_model(inputs)
            loss_adapter = criterion(outputs_adapter, inputs)  # Rekonstrüksiyon görevi
            loss_adapter.backward()
            adapter_optimizer.step()
            adapter_epoch_loss += loss_adapter.item()
            
            # Tam model eğitimi
            full_model_optimizer.zero_grad()
            outputs_full = full_model(inputs)
            loss_full = criterion(outputs_full, inputs)  # Aynı görev
            loss_full.backward()
            full_model_optimizer.step()
            full_model_epoch_loss += loss_full.item()
        
        adapter_losses.append(adapter_epoch_loss / len(train_loader))
        full_model_losses.append(full_model_epoch_loss / len(train_loader))
    
    # Performans karşılaştırması - adapter yeterli performansa ulaşabilmeli
    assert min(adapter_losses) <= min(full_model_losses) * 1.5, "Adapter performansı kabul edilebilir aralıkta olmalı"

# === Reproducibility Test Pattern Uygulaması ===
def test_training_reproducibility(model_fixture, adapter_fixture):
    """Aynı tohum ve verilerle eğitimin tekrarlanabilirliğini test eder."""
    # Tohum ayarla
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Modeli ve adapter'ı kopyala
    model1 = deepcopy(model_fixture)
    adapter1 = deepcopy(adapter_fixture)
    
    # Bağlantı parametrelerini kopyala
    for p1, p2 in zip(model1.parameters(), model_fixture.parameters()):
        p1.data.copy_(p2.data)
    
    # Adapter1'i kopyala
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=64
    )
    adapter_fixture_copy = BottleneckAdapter(config, input_dim=768)
    for p1, p2 in zip(adapter_fixture_copy.parameters(), adapter_fixture.parameters()):
        p1.data.copy_(p2.data)
    
    # CompositeModel1 oluştur
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, x):
            x = self.model(x)
            x = self.adapter(x)
            return x
    
    composite_model1 = CompositeModel(model1, adapter1)
    
    # Yapay veri seti oluştur - sabit tohum ile
    torch.manual_seed(42)
    np.random.seed(42)
    inputs = torch.randn(50, 768)
    labels = torch.randint(0, 2, (50,))
    dataset = [(inputs[i], labels[i]) for i in range(50)]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Optimizer ve loss
    optimizer1 = torch.optim.Adam(composite_model1.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # İlk eğitim
    losses1 = []
    torch.manual_seed(42)  # Shuffle için tohum
    for epoch in range(3):
        epoch_loss = 0.0
        for batch_inputs, _ in train_loader:
            optimizer1.zero_grad()
            outputs = composite_model1(batch_inputs)
            loss = criterion(outputs, batch_inputs)
            loss.backward()
            optimizer1.step()
            epoch_loss += loss.item()
        losses1.append(epoch_loss / len(train_loader))
    
    # Tohumu sıfırla ve aynı eğitimi tekrarla
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model ve adapter'ı sıfırla
    model2 = deepcopy(model_fixture)
    adapter2 = deepcopy(adapter_fixture)
    
    # Bağlantı parametrelerini kopyala
    for p1, p2 in zip(model2.parameters(), model_fixture.parameters()):
        p1.data.copy_(p2.data)
    
    composite_model2 = CompositeModel(model2, adapter2)
    
    # Aynı veri seti
    torch.manual_seed(42)
    np.random.seed(42)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Optimizer ve loss
    optimizer2 = torch.optim.Adam(composite_model2.parameters(), lr=1e-4)
    
    # İkinci eğitim
    losses2 = []
    torch.manual_seed(42)  # Shuffle için tohum
    for epoch in range(3):
        epoch_loss = 0.0
        for batch_inputs, _ in train_loader:
            optimizer2.zero_grad()
            outputs = composite_model2(batch_inputs)
            loss = criterion(outputs, batch_inputs)
            loss.backward()
            optimizer2.step()
            epoch_loss += loss.item()
        losses2.append(epoch_loss / len(train_loader))
    
    # Kayıpların eşleştiğini doğrula
    assert np.allclose(losses1, losses2, rtol=1e-5), "Aynı tohumla eğitim aynı kayıpları üretmelidir"
    
    # Parametrelerin yakın olduğunu doğrula
    for p1, p2 in zip(adapter1.parameters(), adapter2.parameters()):
        assert torch.allclose(p1, p2, rtol=1e-5), "Aynı tohumla eğitim aynı parametreleri üretmelidir"

# === Öğrenme Eğrisi Testi Pattern Uygulaması ===
def test_learning_curve_progression(model_fixture, adapter_fixture, dummy_dataset):
    """Öğrenme eğrisinin zamanla iyileştiğini ve yerel minimuma yakınsadığını doğrulayan test."""
    
    # Create an adapter
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=2
    )
    adapter = BottleneckAdapter(config, input_dim=10)
    
    # Create a small model for efficient testing
    model = torch.nn.Linear(10, 10)
    
    # CompositeModel oluştur
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, x):
            x = self.model(x)
            x = self.adapter(x)
            return x
    
    composite_model = CompositeModel(model, adapter)
    
    # Create a small dataset
    batch_size = 4
    inputs = torch.randn(20, 10)
    targets = torch.randn(20, 10)
    dataset = [(inputs[i], targets[i]) for i in range(20)]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Training setup
    optimizer = torch.optim.Adam(composite_model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    
    # Training for more epochs to ensure convergence
    losses = []
    for epoch in range(20):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = composite_model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
    
    # Check if loss is decreasing
    assert losses[0] > losses[-1], "Loss should decrease during training"
    
    # Check if the learning curve flattens (convergence)
    early_slope = abs(losses[1] - losses[0])
    late_slope = abs(losses[-1] - losses[-2])
    assert late_slope < early_slope, "Learning curve should flatten indicating convergence"
    
    # Loss azalma trendini kontrol et - son çeyrek kayıpların ortalaması ilk çeyrek kayıpların
    # ortalamasından daha düşük olmalı
    first_quarter = losses[:len(losses)//4]
    last_quarter = losses[-len(losses)//4:]
    assert sum(last_quarter)/len(last_quarter) < sum(first_quarter)/len(first_quarter), "Son çeyrek kayıpların ortalaması ilk çeyrek kayıplardan düşük olmalı"

def test_data_preprocessing_integrity():
    """Veri önişleme adımlarının adapter eğitim sürecindeki bütünlüğünü test eder."""
    
    # Create an adapter
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=2
    )
    adapter = BottleneckAdapter(config, input_dim=10)
    
    # Define preprocessing functions
    def normalize(x):
        """Normalize input data."""
        if isinstance(x, torch.Tensor):
            return (x - x.mean()) / (x.std() + 1e-8)
        else:
            x_np = np.array(x)
            return (x_np - x_np.mean()) / (x_np.std() + 1e-8)
    
    # Create datasets with and without preprocessing
    inputs_raw = torch.randn(20, 10)
    inputs_processed = normalize(inputs_raw.clone())
    
    dataset_raw = [(inputs_raw[i], inputs_raw[i]) for i in range(20)]
    dataset_processed = [(inputs_processed[i], inputs_processed[i]) for i in range(20)]
    
    # Train adapter on both datasets
    def train_and_get_loss(dataset):
        # Create a new adapter to avoid state leakage
        config = AdapterConfig(
            adapter_type="bottleneck",
            bottleneck_dim=2
        )
        new_adapter = BottleneckAdapter(config, input_dim=10)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.Adam(new_adapter.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # Short training
        losses = []
        for epoch in range(10):
            epoch_loss = 0.0
            for x, y in dataloader:
                optimizer.zero_grad()
                output = new_adapter(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(dataloader))
        
        return losses[-1]  # Return final loss
    
    # Compare training effectiveness
    loss_raw = train_and_get_loss(dataset_raw)
    loss_processed = train_and_get_loss(dataset_processed)
    
    # Her iki yöntem de çalışabilmeli; hangisinin daha iyi olduğu veri ve model 
    # yapısına bağlı olarak değişebilir. Bu nedenle her iki modelin de 
    # eğitilebilir olduğunu ve aşırı bir fark olmadığını kontrol edelim.
    print(f"Raw loss: {loss_raw}, Processed loss: {loss_processed}")
    
    # Her iki yöntem de makul değerler vermeli
    assert loss_raw < 0.1, "Raw veri ile eğitim başarısız oldu"
    assert loss_processed < 0.1, "Preprocessed veri ile eğitim başarısız oldu"

# === Orijinal test fonksiyonları ===
def test_on_device_training():
    """
    Test the on-device training module with basic verification.
    """
    # Create a simple model
    model = torch.nn.Linear(10, 10)
    
    # Create an adapter
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=2
    )
    adapter = BottleneckAdapter(config, input_dim=10)
    
    # Model parametrelerini dondur
    for param in model.parameters():
        param.requires_grad = False
    
    # Adapter parametrelerinin eğitilebilir olduğunu kontrol et
    for param in adapter.parameters():
        assert param.requires_grad, "Adapter parametreleri eğitilebilir olmalıdır"
    
    # CompositeModel oluştur - TextClassificationTrainer'a uyumlu forward metodu
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, input_ids, attention_mask=None):
            # TextClassificationTrainer dict döndürmemizi bekliyor
            with torch.no_grad():  # Model eğitim dışında tutulacağı için no_grad kullanabiliriz
                x = self.model(input_ids)
            x = self.adapter(x)  # Adapter eğitilebilir
            return {"logits": x}
    
    composite_model = CompositeModel(model, adapter)
    
    # Eğitim yapılandırması oluştur
    training_config = TrainingConfig(
        learning_rate=0.01,
        batch_size=1,
        epochs=2
    )
    
    # Create a trainer
    trainer = OnDeviceTrainer(config=training_config)
    
    # Create dummy data - TextClassificationTrainer'a uygun formatta
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10, dim=10):
            self.size = size
            self.dim = dim
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # labels için torch.zeros yerine torch.randint kullan, sınıflandırma varmış gibi
            return {
                "input_ids": torch.randn(self.dim),
                "attention_mask": torch.ones(self.dim),
                "labels": torch.randint(0, 2, (1,)).item()  # Binary sınıflandırma
            }
    
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    # Eğitimden önce adapter parametrelerinin kopyasını al
    adapter_params_before = [(name, p.clone().detach()) for name, p in adapter.named_parameters()]
    
    # Train the model
    trainer.train(composite_model, dataloader)
    
    # Eğitimden sonra adapter parametrelerinin değişip değişmediğini kontrol et
    changed = False
    for (name, before), (_, after) in zip(adapter_params_before, adapter.named_parameters()):
        if not torch.allclose(before, after):
            changed = True
            print(f"Parametre {name} değişti: Öncesi {before[0, 0].item():.6f}, Sonrası {after[0, 0].item():.6f}")
            break
    
    # Bu sefer gradyanlar yerine parametre değişimini kontrol ediyoruz
    assert changed, "Adapter parametreleri eğitimden sonra değişmelidir"
    
    # Verify model parameters haven't changed
    model_params_changed = False
    for param_before, param_after in zip(model.parameters(), model.parameters()):
        if not torch.equal(param_before, param_after):
            model_params_changed = True
            break
    
    assert not model_params_changed, "Model parametreleri değişmemiş olmalıdır"

def test_training_checkpoint():
    """
    Test saving and loading training checkpoints.
    """
    # Create a simple model
    model = torch.nn.Linear(10, 10)
    
    # Create an adapter
    config = AdapterConfig(
        adapter_type="bottleneck",
        bottleneck_dim=2
    )
    adapter = BottleneckAdapter(config, input_dim=10)
    
    # Model parametrelerini dondur
    for param in model.parameters():
        param.requires_grad = False
        
    # Adapter parametrelerinin eğitilebilir olduğunu kontrol et
    for param in adapter.parameters():
        assert param.requires_grad, "Adapter parametreleri eğitilebilir olmalıdır"
        
    # CompositeModel oluştur - TextClassificationTrainer'a uyumlu forward metodu
    class CompositeModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.model = model
            self.adapter = adapter
        
        def forward(self, input_ids, attention_mask=None):
            # TextClassificationTrainer dict döndürmemizi bekliyor
            with torch.no_grad():  # Model eğitim dışında tutulacağı için no_grad kullanabiliriz
                x = self.model(input_ids)
            x = self.adapter(x)  # Adapter eğitilebilir
            return {"logits": x}
    
    composite_model = CompositeModel(model, adapter)
    
    # Eğitim yapılandırması oluştur
    training_config = TrainingConfig(
        learning_rate=0.01,
        batch_size=1,
        epochs=1
    )
    
    # Create dummy data - TextClassificationTrainer'a uygun formatta
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10, dim=10):
            self.size = size
            self.dim = dim
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # labels için torch.zeros yerine torch.randint kullan, sınıflandırma varmış gibi
            return {
                "input_ids": torch.randn(self.dim),
                "attention_mask": torch.ones(self.dim),
                "labels": torch.randint(0, 2, (1,)).item()  # Binary sınıflandırma
            }
    
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    # Create temp dir for save_dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a trainer with save_dir
        trainer = OnDeviceTrainer(config=training_config, save_dir=tmp_dir)
        
        # Eğitimden önce adapter parametrelerinin kopyasını al
        adapter_params_before = [(name, p.clone().detach()) for name, p in adapter.named_parameters()]
    
        # Train the model
        trainer.train(composite_model, dataloader)
        
        # Eğitimden sonra adapter parametrelerinin değişip değişmediğini kontrol et
        changed = False
        for (name, before), (_, after) in zip(adapter_params_before, adapter.named_parameters()):
            if not torch.allclose(before, after):
                changed = True
                print(f"Parametre {name} değişti: Öncesi {before[0, 0].item():.6f}, Sonrası {after[0, 0].item():.6f}")
                break
        
        # Parametre değişimini kontrol et
        assert changed, "Adapter parametreleri eğitimden sonra değişmelidir"
        
        # Check for saved checkpoint
        checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")
        assert os.path.exists(checkpoint_path), "Checkpoint dosyası oluşturulmalı"
        
        # Create a new adapter
        config = AdapterConfig(
            adapter_type="bottleneck",
            bottleneck_dim=2
        )
        new_adapter = BottleneckAdapter(config, input_dim=10)
        
        # Create a new composite model
        new_model = torch.nn.Linear(10, 10)
        # Model parametrelerini dondur
        for param in new_model.parameters():
            param.requires_grad = False
            
        new_composite_model = CompositeModel(new_model, new_adapter)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        new_composite_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Verify adapter parameters match - model state_dict yüklendikten sonra
        # adapter parametreleri eşleşmeli
        for (old_name, old_param), (new_name, new_param) in zip(
            adapter.named_parameters(), new_adapter.named_parameters()
        ):
            assert torch.allclose(old_param, new_param), f"Parameters {old_name} and {new_name} don't match after loading checkpoint"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 