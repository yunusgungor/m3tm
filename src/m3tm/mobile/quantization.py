"""
M³TM Model Quantization Modülü - Context7 Enhanced

Bu modül PyTorch'un en güncel quantization teknikleri kullanarak
model boyutunu minimize etmeyi amaçlar. Context7 documentation'dan
alınan current best practices uygulanmıştır.

Desteklenen Teknikler (Context7 Based):
- Post Training Dynamic Quantization (PTDQ) - LSTM/Transformer için optimal
- Post Training Static Quantization (PTSQ) - CNN'ler için optimal
- Quantization Aware Training (QAT) - En yüksek accuracy için
- FX Graph Mode Quantization - Gelişmiş optimizasyon için

Mobile Backend Optimizations:
- QNNPACK backend for ARM mobile CPUs
- Automatic module fusion (conv + relu, conv + bn + relu)
- QuantStub/DeQuantStub integration for custom models
"""

import copy
import io
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple, List, Callable

import torch
import torch.nn as nn

# Context7 recommended imports - latest PyTorch quantization APIs
try:
    import torch.ao.quantization as quantization
    from torch.ao.quantization import (
        quantize_fx, prepare_fx, convert_fx, fuse_fx,
        prepare_qat_fx, convert_fx as convert_qat_fx,
        get_default_qconfig, get_default_qat_qconfig,
        get_default_qconfig_mapping, get_default_qat_qconfig_mapping,
        quantize_dynamic, QuantStub, DeQuantStub
    )
    # Backend configuration for mobile
    torch.backends.quantized.engine = 'qnnpack'  # Context7: Mobile ARM CPUs için optimal
    QUANTIZATION_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Latest quantization APIs not available, falling back to legacy")
    try:
        from torch.quantization import (
            quantize_fx, prepare_fx, convert_fx,
            get_default_qconfig, get_default_qat_qconfig,
            quantize_dynamic
        )
        QUANTIZATION_AVAILABLE = True
    except ImportError:
        QUANTIZATION_AVAILABLE = False

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker

logger = logging.getLogger(__name__)


class QuantizationManager:
    """
    M³TM Model Quantization Manager - Context7 Enhanced
    
    Context7 documentation'dan alınan best practices ile geliştirilmiş
    quantization manager. Mobile deployment için optimize edilmiştir.
    
    Features:
    - Multi-method quantization support (PTDQ, PTSQ, QAT)
    - Mobile-optimized backend configuration (QNNPACK)
    - Automatic module fusion for performance
    - FX Graph Mode support for advanced optimization
    - Comprehensive validation and benchmarking
    """

    def __init__(self,
                 backend: str = "qnnpack",  # Context7: Mobile için optimal
                 calibration_iterations: int = 100,
                 benchmarker: Optional[ModelBenchmarker] = None):
        """
        Context7-enhanced QuantizationManager initialization.
        
        Args:
            backend: Quantization backend ('qnnpack' for mobile, 'x86' for server)
            calibration_iterations: Kalibrasyon iterasyon sayısı
            benchmarker: Performans ölçüm aracı
        """
        self.backend = backend
        self.calibration_iterations = calibration_iterations
        self.benchmarker = benchmarker or ModelBenchmarker()
        
        # Context7: Set optimal backend for mobile deployment
        if backend == "qnnpack":
            torch.backends.quantized.engine = 'qnnpack'  # ARM mobile CPUs
        elif backend == "x86":
            torch.backends.quantized.engine = 'x86'  # Server inference
        elif backend == "fbgemm":  # Legacy support
            torch.backends.quantized.engine = 'fbgemm'
            
        # Context7: Verify quantization availability
        if not QUANTIZATION_AVAILABLE:
            logger.warning("Advanced quantization APIs not available, falling back to basic dynamic quantization")
            
        logger.info(f"QuantizationManager initialized with backend: {backend}")

    def apply_dynamic_quantization(self,
                                   model: Union[nn.Module, BaseModel],
                                   qconfig_spec: Optional[Dict] = None,
                                   dtype: torch.dtype = torch.qint8,
                                   benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Context7-enhanced dynamic quantization - optimal for LSTM/Transformer models.
        
        Best for models where execution time is dominated by weight loading,
        such as LSTM and Transformer models with small batch sizes.
        
        Args:
            model: Quantize edilecek model
            qconfig_spec: Özel quantization config spesifikasyonu  
            dtype: Hedef veri tipi (torch.qint8 recommended)
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[quantized_model, metrics]
        """
        logger.info("Context7-enhanced dynamic quantization başlatılıyor...")
        
        if not QUANTIZATION_AVAILABLE:
            logger.error("Quantization APIs not available")
            return model, {}
            
        original_model = model
        
        # Context7 pattern: Default layer spec for dynamic quantization
        if qconfig_spec is None:
            qconfig_spec = {torch.nn.Linear}  # Most common for transformers
            
        try:
            # Context7: Apply dynamic quantization using latest API
            quantized_model = quantize_dynamic(
                original_model,
                qconfig_spec,
                dtype=dtype
            )
            
            logger.info("Dynamic quantization tamamlandı")
            
            # Benchmark if requested
            metrics = {}
            if benchmark:
                metrics = self._benchmark_models(original_model, quantized_model, "dynamic")
                
            return quantized_model, metrics
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            return original_model, {}

    def apply_static_quantization(self,
                                  model: Union[nn.Module, BaseModel],
                                  calibration_dataloader,
                                  example_inputs: torch.Tensor,
                                  fuse_modules: Optional[List[List[str]]] = None,
                                  benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Context7-enhanced static quantization - optimal for CNN models.
        
        Post Training Static Quantization quantizes both weights and activations,
        often fusing activations into preceding layers. Requires calibration with
        representative dataset.
        
        Args:
            model: Quantize edilecek model
            calibration_dataloader: Kalibrasyon data loader'ı
            example_inputs: Model için örnek input tensoru
            fuse_modules: Füzyon için modül listesi (örn: [['conv', 'relu']])
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[quantized_model, metrics]
        """
        logger.info("Context7-enhanced static quantization başlatılıyor...")
        
        if not QUANTIZATION_AVAILABLE:
            logger.error("Static quantization APIs not available")
            return model, {}
            
        original_model = copy.deepcopy(model)
        
        try:
            # Context7: Set model to eval mode (required for static quantization)
            original_model.eval()
            
            # Context7: Set quantization config for mobile backend
            original_model.qconfig = get_default_qconfig(self.backend)
            
            # Context7: Module fusion for performance
            if fuse_modules:
                logger.info(f"Applying module fusion: {fuse_modules}")
                original_model = quantization.fuse_modules(original_model, fuse_modules)
            
            # Context7: Prepare model for static quantization
            prepared_model = quantization.prepare(original_model)
            
            # Context7: Calibration phase with representative data
            logger.info("Calibrating model with representative data...")
            prepared_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(calibration_dataloader):
                    if i >= self.calibration_iterations:
                        break
                    if isinstance(batch, (list, tuple)):
                        prepared_model(batch[0])
                    else:
                        prepared_model(batch)
                        
            # Context7: Convert to quantized model
            quantized_model = quantization.convert(prepared_model)
            
            logger.info("Static quantization tamamlandı")
            
            # Benchmark if requested
            metrics = {}
            if benchmark:
                metrics = self._benchmark_models(model, quantized_model, "static")
                
            return quantized_model, metrics
            
        except Exception as e:
            logger.error(f"Static quantization failed: {e}")
            return model, {}

    def apply_qat(self,
                  model: Union[nn.Module, BaseModel],
                  train_dataloader,
                  val_dataloader,
                  num_epochs: int = 5,
                  learning_rate: float = 1e-4,
                  fuse_modules: Optional[List[List[str]]] = None,
                  benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Context7-enhanced Quantization Aware Training - highest accuracy method.
        
        QAT simulates quantization effects during training using fake quantization,
        allowing the model to adapt to quantization noise.
        
        Args:
            model: Eğitilecek model
            train_dataloader: Eğitim data loader'ı
            val_dataloader: Validasyon data loader'ı
            num_epochs: Eğitim epoch sayısı
            learning_rate: Öğrenme oranı
            fuse_modules: Füzyon için modül listesi
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[quantized_model, metrics]
        """
        logger.info("Context7-enhanced QAT başlatılıyor...")
        
        if not QUANTIZATION_AVAILABLE:
            logger.error("QAT APIs not available")
            return model, {}
            
        original_model = copy.deepcopy(model)
        
        try:
            # Context7: Set model to eval mode for fusion
            original_model.eval()
            
            # Context7: Set QAT quantization config
            original_model.qconfig = get_default_qat_qconfig(self.backend)
            
            # Context7: Module fusion
            if fuse_modules:
                logger.info(f"Applying module fusion for QAT: {fuse_modules}")
                original_model = quantization.fuse_modules(original_model, fuse_modules)
                
            # Context7: Prepare for QAT (model must be in train mode)
            prepared_model = quantization.prepare_qat(original_model.train())
            
            # Context7: QAT training loop
            optimizer = torch.optim.Adam(prepared_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(num_epochs):
                prepared_model.train()
                total_loss = 0
                
                for batch_idx, batch in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch
                    else:
                        inputs, targets = batch, None
                        
                    outputs = prepared_model(inputs)
                    
                    if targets is not None:
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        
                logger.info(f"QAT Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}")
                
            # Context7: Convert to quantized model
            prepared_model.eval()
            quantized_model = quantization.convert(prepared_model)
            
            logger.info("QAT tamamlandı")
            
            # Benchmark if requested
            metrics = {}
            if benchmark:
                metrics = self._benchmark_models(model, quantized_model, "qat")
                
            return quantized_model, metrics
            
        except Exception as e:
            logger.error(f"QAT failed: {e}")
            return model, {}

    def apply_fx_quantization(self,
                              model: Union[nn.Module, BaseModel],
                              example_inputs: torch.Tensor,
                              calibration_dataloader = None,
                              qat_mode: bool = False,
                              benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Context7-enhanced FX Graph Mode quantization - advanced optimization.
        
        FX Graph Mode provides more advanced quantization capabilities and better
        performance than eager mode quantization.
        
        Args:
            model: Quantize edilecek model
            example_inputs: Model için örnek input
            calibration_dataloader: Kalibrasyon dataloader'ı (static için gerekli)
            qat_mode: QAT modu kullanılsın mı
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[quantized_model, metrics]
        """
        logger.info("Context7-enhanced FX quantization başlatılıyor...")
        
        if not QUANTIZATION_AVAILABLE or not hasattr(quantization, 'quantize_fx'):
            logger.error("FX quantization APIs not available")
            return model, {}
            
        original_model = copy.deepcopy(model)
        
        try:
            # Context7: Get appropriate qconfig mapping
            if qat_mode:
                qconfig_mapping = get_default_qat_qconfig_mapping(self.backend)
                original_model.train()
            else:
                qconfig_mapping = get_default_qconfig_mapping(self.backend)
                original_model.eval()
                
            # Context7: Module fusion using FX
            fused_model = fuse_fx(original_model)
            
            # Context7: Prepare model using FX
            if qat_mode:
                prepared_model = prepare_qat_fx(fused_model, qconfig_mapping, example_inputs)
                # QAT training would go here...
                prepared_model.eval()
            else:
                prepared_model = prepare_fx(fused_model, qconfig_mapping, example_inputs)
                
                # Context7: Calibration for static quantization
                if calibration_dataloader:
                    logger.info("FX calibration başlatılıyor...")
                    with torch.no_grad():
                        for i, batch in enumerate(calibration_dataloader):
                            if i >= self.calibration_iterations:
                                break
                            if isinstance(batch, (list, tuple)):
                                prepared_model(batch[0])
                            else:
                                prepared_model(batch)
                                
            # Context7: Convert to quantized model
            quantized_model = convert_fx(prepared_model)
            
            logger.info("FX quantization tamamlandı")
            
            # Benchmark if requested
            metrics = {}
            if benchmark:
                metrics = self._benchmark_models(model, quantized_model, "fx")
                
            return quantized_model, metrics
            
        except Exception as e:
            logger.error(f"FX quantization failed: {e}")
            return model, {}

    def save_quantized_model(self,
                             model: nn.Module,
                             save_path: Union[str, Path],
                             save_format: str = "state_dict") -> bool:
        """
        Context7-enhanced model saving with recommended best practices.
        
        Args:
            model: Kaydedilecek quantized model
            save_path: Dosya yolu
            save_format: Kayıt formatı ('state_dict', 'full_model', 'torchscript')
            
        Returns:
            Başarı durumu
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_format == "state_dict":
                # Context7: Recommended method for quantized models
                torch.save(model.state_dict(), save_path)
            elif save_format == "full_model":
                torch.save(model, save_path)
            elif save_format == "torchscript":
                # Convert to TorchScript for mobile deployment
                scripted_model = torch.jit.script(model)
                scripted_model.save(str(save_path))
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
                
            logger.info(f"Quantized model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save quantized model: {e}")
            return False

    def load_quantized_model(self,
                             model_class: nn.Module,
                             load_path: Union[str, Path],
                             load_format: str = "state_dict") -> Optional[nn.Module]:
        """
        Context7-enhanced quantized model loading.
        
        Args:
            model_class: Model sınıfı (state_dict için gerekli)
            load_path: Dosya yolu
            load_format: Yükleme formatı ('state_dict', 'full_model', 'torchscript')
            
        Returns:
            Yüklenen model veya None
        """
        try:
            load_path = Path(load_path)
            
            if load_format == "state_dict":
                # Context7: Recommended loading method
                model = model_class
                model.load_state_dict(torch.load(load_path))
                return model
            elif load_format == "full_model":
                return torch.load(load_path)
            elif load_format == "torchscript":
                return torch.jit.load(str(load_path))
            else:
                raise ValueError(f"Unsupported load format: {load_format}")
                
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            return None

    def _benchmark_models(self,
                          original_model: nn.Module,
                          quantized_model: nn.Module,
                          method: str) -> Dict[str, Any]:
        """
        Context7-enhanced model benchmarking.
        
        Args:
            original_model: Orijinal model
            quantized_model: Quantized model
            method: Quantization yöntemi
            
        Returns:
            Benchmark metrikleri
        """
        try:
            # Model size comparison
            def get_model_size(model):
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                return param_size
                
            original_size = get_model_size(original_model)
            quantized_size = get_model_size(quantized_model)
            size_reduction = ((original_size - quantized_size) / original_size) * 100
            
            metrics = {
                "quantization_method": method,
                "original_size_bytes": original_size,
                "quantized_size_bytes": quantized_size,
                "size_reduction_percent": size_reduction,
                "compression_ratio": original_size / quantized_size if quantized_size > 0 else 0
            }
            
            # Use benchmarker if available
            if self.benchmarker:
                bench_results = self.benchmarker.compare_models(original_model, quantized_model)
                metrics.update(bench_results)
                
            logger.info(f"Model boyutu {size_reduction:.1f}% azaldı ({method})")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {"error": str(e)}


def create_quantization_pipeline(backend: str = "qnnpack",
                                 calibration_iterations: int = 100) -> QuantizationManager:
    """
    Context7-enhanced quantization pipeline oluşturucu.
    
    Args:
        backend: Quantization backend ('qnnpack' for mobile recommended)
        calibration_iterations: Kalibrasyon iterasyon sayısı
        
    Returns:
        QuantizationManager instance
    """
    return QuantizationManager(
        backend=backend,
        calibration_iterations=calibration_iterations
    )


# Context7: Export recommended configurations
MOBILE_QUANTIZATION_CONFIG = {
    "backend": "qnnpack",
    "dtype": torch.qint8,
    "calibration_iterations": 100
}

SERVER_QUANTIZATION_CONFIG = {
    "backend": "x86",
    "dtype": torch.qint8,
    "calibration_iterations": 200
}
