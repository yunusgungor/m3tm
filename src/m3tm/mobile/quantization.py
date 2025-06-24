"""
M³TM Model Quantization Modülü

Bu modül PyTorch'un en güncel quantization teknikleri kullanarak 
model boyutunu minimize etmeyi amaçlar.
"""

import logging
from typing import Dict, Optional, Union, Any, Tuple, List, Callable
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_fx, prepare_fx, convert_fx
from torch.quantization.qconfig import get_default_qconfig, get_default_qat_qconfig
from torch.quantization.quantize_fx import prepare_qat_fx
import torch.quantization.quantize_fx as quantize_fx_module
from torch.ao.quantization import quantize, quantize_qat, QuantStub, DeQuantStub

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker


logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Quantization yapılandırma sınıfı."""
    
    def __init__(self, 
                 backend: str = "fbgemm",
                 qconfig_spec: Optional[Dict] = None,
                 calibration_iterations: int = 100,
                 preserve_sparsity: bool = False):
        """
        Args:
            backend: Quantization backend ('fbgemm', 'qnnpack', 'x86', 'onednn')
            qconfig_spec: Özel quantization config spesifikasyonu
            calibration_iterations: Kalibrasyon için iterasyon sayısı
            preserve_sparsity: Pruning sparsity'sini koruyup korumaması
        """
        self.backend = backend
        self.qconfig_spec = qconfig_spec or {}
        self.calibration_iterations = calibration_iterations
        self.preserve_sparsity = preserve_sparsity
        
        # Backend'e göre varsayılan qconfig
        if backend == "fbgemm":
            self.default_qconfig = get_default_qconfig('fbgemm')
        elif backend == "qnnpack":
            self.default_qconfig = get_default_qconfig('qnnpack')
        else:
            self.default_qconfig = get_default_qconfig('fbgemm')


class QuantizationManager:
    """Gelişmiş model quantization yöneticisi."""
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Args:
            config: Quantization yapılandırma objesi
        """
        self.config = config or QuantizationConfig()
        self.benchmarker = ModelBenchmarker()
        
    def apply_dynamic_quantization(self,
                                   model: Union[nn.Module, BaseModel],
                                   qconfig_spec: Optional[Dict] = None,
                                   dtype: torch.dtype = torch.qint8,
                                   benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Gelişmiş dinamik quantization uygular.
        
        Args:
            model: Quantize edilecek model
            qconfig_spec: Özel quantization config spesifikasyonu
            dtype: Hedef veri tipi
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[quantized_model, metrics]
        """
        logger.info("Dinamik quantization başlatılıyor...")
        
        # Orijinal model metrics
        original_metrics = {}
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)
        
        try:
            # Quantize edilecek layer türlerini belirle
            modules_to_quantize = {nn.Linear, nn.Conv2d, nn.Conv1d, nn.LSTM, nn.LSTMCell}
            
            # Özel qconfig varsa kullan
            if qconfig_spec:
                quantized_model = quantize_dynamic(
                    model, 
                    qconfig_spec=qconfig_spec,
                    dtype=dtype
                )
            else:
                quantized_model = quantize_dynamic(
                    model, 
                    modules_to_quantize, 
                    dtype=dtype
                )
            
            # Quantized model metrics
            quantized_metrics = {}
            if benchmark:
                quantized_metrics = self.benchmarker.get_model_metrics(quantized_model)
                
            # Compression metrics hesapla
            compression_metrics = self._calculate_compression_metrics(
                original_metrics, quantized_metrics
            )
            
            logger.info(f"Dinamik quantization tamamlandı. "
                       f"Boyut azaltımı: {compression_metrics.get('size_reduction_percent', 0):.1f}%")
            
            return quantized_model, {
                'original_metrics': original_metrics,
                'quantized_metrics': quantized_metrics,
                'compression_metrics': compression_metrics,
                'quantization_type': 'dynamic',
                'dtype': str(dtype)
            }
            
        except Exception as e:
            logger.error(f"Dinamik quantization hatası: {e}")
            raise
    
    def apply_static_quantization(self,
                                  model: Union[nn.Module, BaseModel], 
                                  calibration_data_loader: Any,
                                  example_inputs: torch.Tensor,
                                  qconfig_spec: Optional[Dict] = None,
                                  benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Statik (post-training) quantization uygular.
        
        Args:
            model: Quantize edilecek model
            calibration_data_loader: Kalibrasyon veri yükleyici
            example_inputs: Örnek girdi tensörü
            qconfig_spec: Özel quantization config
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[quantized_model, metrics]
        """
        logger.info("Statik quantization başlatılıyor...")
        
        # Orijinal model metrics
        original_metrics = {}
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)
        
        try:
            # Model evaluation moda al
            model.eval()
            
            # FX Graph Mode Quantization kullan (PyTorch 1.8+)
            qconfig = qconfig_spec or self.config.default_qconfig
            qconfig_dict = {"": qconfig}
            
            # Model prepare et
            model_prepared = prepare_fx(model, qconfig_dict, example_inputs)
            
            # Kalibrasyon yap
            logger.info("Kalibrasyon işlemi başlatılıyor...")
            self._calibrate_model(model_prepared, calibration_data_loader)
            
            # Quantized model'e dönüştür
            quantized_model = convert_fx(model_prepared)
            
            # Quantized model metrics
            quantized_metrics = {}
            if benchmark:
                quantized_metrics = self.benchmarker.get_model_metrics(quantized_model)
                
            # Compression metrics hesapla
            compression_metrics = self._calculate_compression_metrics(
                original_metrics, quantized_metrics
            )
            
            logger.info(f"Statik quantization tamamlandı. "
                       f"Boyut azaltımı: {compression_metrics.get('size_reduction_percent', 0):.1f}%")
            
            return quantized_model, {
                'original_metrics': original_metrics,
                'quantized_metrics': quantized_metrics,
                'compression_metrics': compression_metrics,
                'quantization_type': 'static',
                'calibration_iterations': self.config.calibration_iterations
            }
            
        except Exception as e:
            logger.error(f"Statik quantization hatası: {e}")
            raise
    
    def apply_quantization_aware_training(self,
                                          model: Union[nn.Module, BaseModel],
                                          example_inputs: torch.Tensor,
                                          qconfig_spec: Optional[Dict] = None,
                                          benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        Quantization Aware Training (QAT) için model hazırlar.
        
        Args:
            model: QAT için hazırlanacak model
            example_inputs: Örnek girdi tensörü  
            qconfig_spec: Özel quantization config
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[qat_prepared_model, metrics]
        """
        logger.info("Quantization Aware Training hazırlığı başlatılıyor...")
        
        # Orijinal model metrics
        original_metrics = {}
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)
        
        try:
            # QAT config
            qat_qconfig = qconfig_spec or get_default_qat_qconfig(self.config.backend)
            qconfig_dict = {"": qat_qconfig}
            
            # QAT için model prepare et
            model.train()  # Training modda olmalı
            qat_prepared_model = prepare_qat_fx(model, qconfig_dict, example_inputs)
            
            logger.info("Model QAT için hazırlandı. Training loop ile eğitim yapılması gerekiyor.")
            
            # QAT prepared model metrics
            qat_metrics = {}
            if benchmark:
                qat_metrics = self.benchmarker.get_model_metrics(qat_prepared_model)
            
            return qat_prepared_model, {
                'original_metrics': original_metrics,
                'qat_prepared_metrics': qat_metrics,
                'quantization_type': 'qat_prepared',
                'qconfig': str(qat_qconfig)
            }
            
        except Exception as e:
            logger.error(f"QAT hazırlık hatası: {e}")
            raise
    
    def finalize_qat_model(self, qat_trained_model: nn.Module, 
                           benchmark: bool = True) -> Tuple[nn.Module, Dict]:
        """
        QAT ile eğitilmiş modeli nihai quantized model'e dönüştürür.
        
        Args:
            qat_trained_model: QAT ile eğitilmiş model
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[final_quantized_model, metrics]
        """
        logger.info("QAT model finalize ediliyor...")
        
        try:
            # Evaluation moda al
            qat_trained_model.eval()
            
            # QAT modelini nihai quantized model'e dönüştür
            quantized_model = convert_fx(qat_trained_model)
            
            # Metrics hesapla
            quantized_metrics = {}
            if benchmark:
                quantized_metrics = self.benchmarker.get_model_metrics(quantized_model)
            
            logger.info("QAT model başarıyla finalize edildi.")
            
            return quantized_model, {
                'quantized_metrics': quantized_metrics,
                'quantization_type': 'qat_final'
            }
            
        except Exception as e:
            logger.error(f"QAT finalize hatası: {e}")
            raise
    
    def _calibrate_model(self, model: nn.Module, calibration_data_loader: Any) -> None:
        """
        Model kalibrasyonu yapar.
        
        Args:
            model: Kalibrasyon yapılacak prepared model
            calibration_data_loader: Kalibrasyon veri yükleyici
        """
        model.eval()
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data_loader):
                if i >= self.config.calibration_iterations:
                    break
                    
                # Model forward pass
                if isinstance(inputs, (list, tuple)):
                    model(*inputs)
                else:
                    model(inputs)
        
        logger.info(f"Kalibrasyon tamamlandı ({self.config.calibration_iterations} iterasyon)")
    
    def _calculate_compression_metrics(self, original_metrics: Dict, 
                                       quantized_metrics: Dict) -> Dict:
        """
        Compression metrics hesaplar.
        
        Args:
            original_metrics: Orijinal model metrikleri
            quantized_metrics: Quantized model metrikleri
            
        Returns:
            Compression metrics dict
        """
        if not original_metrics or not quantized_metrics:
            return {}
        
        compression_metrics = {}
        
        # Size reduction
        if 'model_size_mb' in original_metrics and 'model_size_mb' in quantized_metrics:
            original_size = original_metrics['model_size_mb']
            quantized_size = quantized_metrics['model_size_mb']
            
            if original_size > 0:
                size_reduction = (original_size - quantized_size) / original_size * 100
                compression_ratio = original_size / quantized_size if quantized_size > 0 else float('inf')
                
                compression_metrics.update({
                    'size_reduction_percent': size_reduction,
                    'compression_ratio': compression_ratio,
                    'original_size_mb': original_size,
                    'quantized_size_mb': quantized_size
                })
        
        # Parameter count reduction
        if 'param_count' in original_metrics and 'param_count' in quantized_metrics:
            original_params = original_metrics['param_count']
            quantized_params = quantized_metrics['param_count']
            
            if original_params > 0:
                param_reduction = (original_params - quantized_params) / original_params * 100
                compression_metrics.update({
                    'param_reduction_percent': param_reduction,
                    'original_params': original_params,
                    'quantized_params': quantized_params
                })
        
        return compression_metrics
    
    def save_quantized_model(self, model: nn.Module, save_path: Union[str, Path],
                             metadata: Optional[Dict] = None) -> None:
        """
        Quantized modeli kaydeder.
        
        Args:
            model: Kaydedilecek quantized model
            save_path: Kaydetme yolu
            metadata: Ek metadata bilgileri
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Model state dict kaydet
        model_state = {
            'model_state_dict': model.state_dict(),
            'quantization_metadata': metadata or {},
            'model_type': 'quantized',
            'backend': self.config.backend
        }
        
        torch.save(model_state, save_path)
        logger.info(f"Quantized model kaydedildi: {save_path}")


def create_quantization_pipeline(backend: str = "fbgemm",
                                 calibration_iterations: int = 100) -> QuantizationManager:
    """
    Quantization pipeline oluşturucu fonksiyon.
    
    Args:
        backend: Quantization backend
        calibration_iterations: Kalibrasyon iterasyon sayısı
        
    Returns:
        QuantizationManager instance
    """
    config = QuantizationConfig(
        backend=backend,
        calibration_iterations=calibration_iterations
    )
    
    return QuantizationManager(config)
