"""
M³TM Model Quantization Modülü

Bu modül PyTorch'un en güncel quantization teknikleri kullanarak
model boyutunu minimize etmeyi amaçlar.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

# Import compatibility handling
try:
    from torch.ao.quantization import (
        quantize_fx, prepare_fx, convert_fx,
        get_default_qconfig, get_default_qat_qconfig
    )
except ImportError:
    try:
        from torch.quantization import (
            quantize_fx, prepare_fx, convert_fx,
            get_default_qconfig, get_default_qat_qconfig
        )
    except ImportError:
        # Fallback for older versions
        quantize_fx = None
        prepare_fx = None
        convert_fx = None
        get_default_qconfig = None
        get_default_qat_qconfig = None

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker

logger = logging.getLogger(__name__)


class QuantizationManager:
    """
    M³TM Model Quantization Manager

    Bu sınıf PyTorch quantization API'sini kullanarak model boyutunu
    minimize etmeye odaklanır. Context7 uyumlu best practices ile
    geliştirilmiştir.
    """

    def __init__(self,
                 backend: str = "fbgemm",
                 calibration_iterations: int = 100,
                 benchmarker: Optional[ModelBenchmarker] = None):
        """
        QuantizationManager initialization.

        Args:
            backend: Quantization backend ('fbgemm', 'qnnpack')
            calibration_iterations: Kalibrasyon iterasyon sayısı
            benchmarker: Performans ölçüm aracı
        """
        self.backend = backend
        self.calibration_iterations = calibration_iterations
        self.benchmarker = benchmarker or ModelBenchmarker()

        # Set torch backend
        if backend == "fbgemm":
            torch.backends.quantized.engine = 'fbgemm'
        elif backend == "qnnpack":
            torch.backends.quantized.engine = 'qnnpack'

        logger.info(f"QuantizationManager initialized with backend: {backend}")

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
        logger.info("Dynamic quantization başlatılıyor...")

        original_model = model
        original_metrics = None

        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model)

        try:
            # Create a clean copy of the model
            device = next(model.parameters()).device

            # Create a clean state dict copy
            state_dict = copy.deepcopy(model.state_dict())
            clean_model = type(model)()

            # Handle different model types
            if hasattr(model, 'load_state_dict'):
                clean_model.load_state_dict(state_dict)
            else:
                # Fallback: try to recreate model structure
                try:
                    clean_model = copy.deepcopy(model)
                except:
                    logger.warning("Could not create clean copy, using original model")
                    clean_model = model

            clean_model.eval()

            # Apply quantization
            if qconfig_spec:
                quantized_model = quantize_dynamic(
                    clean_model,
                    qconfig_spec,
                    dtype=dtype
                )
            else:
                # Default modules to quantize
                modules_to_quantize = {nn.Linear, nn.Conv2d}
                quantized_model = quantize_dynamic(
                    clean_model,
                    modules_to_quantize,
                    dtype=dtype
                )

            quantized_metrics = None
            compression_metrics = {}

            if benchmark and original_metrics:
                quantized_metrics = self.benchmarker.get_model_metrics(quantized_model)
                compression_metrics = self._calculate_compression_metrics(
                    original_metrics, quantized_metrics)

                logger.info(f"Dynamic quantization tamamlandı - "
                           f"Boyut azaltımı: {compression_metrics.get('size_reduction_percent', 0):.1f}%")

            return quantized_model, {
                'original_metrics': original_metrics,
                'quantized_metrics': quantized_metrics,
                'compression_metrics': compression_metrics,
                'quantization_type': 'dynamic'
            }

        except Exception as e:
            logger.error(f"Dynamic quantization hatası: {str(e)}")
            return original_model, {'error': str(e)}

    def apply_static_quantization(self,
                                  model: Union[nn.Module, BaseModel],
                                  calibration_data_loader,
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
        logger.info("Static quantization başlatılıyor...")

        original_metrics = None
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model, example_inputs)

        try:
            model.eval()

            # FX-based quantization if available
            if prepare_fx and convert_fx:
                prepared_model = prepare_fx(model, qconfig_spec or {"": get_default_qconfig(self.backend)})
                self._calibrate_model(prepared_model, calibration_data_loader)
                quantized_model = convert_fx(prepared_model)
            else:
                logger.warning("FX quantization not available, using legacy method")
                return self.apply_dynamic_quantization(model, qconfig_spec, benchmark=benchmark)

            quantized_metrics = None
            compression_metrics = {}

            if benchmark and original_metrics:
                quantized_metrics = self.benchmarker.get_model_metrics(quantized_model, example_inputs)
                compression_metrics = self._calculate_compression_metrics(
                    original_metrics, quantized_metrics)

                logger.info(f"Static quantization tamamlandı - "
                           f"Boyut azaltımı: {compression_metrics.get('size_reduction_percent', 0):.1f}%")

            return quantized_model, {
                'original_metrics': original_metrics,
                'quantized_metrics': quantized_metrics,
                'compression_metrics': compression_metrics,
                'quantization_type': 'static'
            }

        except Exception as e:
            logger.error(f"Static quantization hatası: {str(e)}")
            return model, {'error': str(e)}

    def prepare_qat_model(self,
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
        logger.info("QAT model preparation başlatılıyor...")

        original_metrics = None
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model, example_inputs)

        try:
            model.train()

            # Prepare for QAT
            if prepare_fx and get_default_qat_qconfig:
                qat_model = prepare_fx(
                    model,
                    qconfig_spec or {"": get_default_qat_qconfig(self.backend)},
                    example_inputs
                )
            else:
                logger.warning("QAT preparation not available")
                return model, {'error': 'QAT not supported in this PyTorch version'}

            preparation_metrics = None
            if benchmark:
                preparation_metrics = self.benchmarker.get_model_metrics(qat_model, example_inputs)

            logger.info("QAT model preparation tamamlandı")

            return qat_model, {
                'original_metrics': original_metrics,
                'prepared_metrics': preparation_metrics,
                'preparation_type': 'qat'
            }

        except Exception as e:
            logger.error(f"QAT preparation hatası: {str(e)}")
            return model, {'error': str(e)}

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
        logger.info("QAT model finalization başlatılıyor...")

        try:
            qat_trained_model.eval()

            if convert_fx:
                final_model = convert_fx(qat_trained_model)
            else:
                logger.warning("QAT finalization not available")
                return qat_trained_model, {'error': 'QAT finalization not supported'}

            finalization_metrics = None
            if benchmark:
                finalization_metrics = self.benchmarker.get_model_metrics(final_model)

            logger.info("QAT model finalization tamamlandı")

            return final_model, {
                'finalized_metrics': finalization_metrics,
                'finalization_type': 'qat'
            }

        except Exception as e:
            logger.error(f"QAT finalization hatası: {str(e)}")
            return qat_trained_model, {'error': str(e)}

    def _calibrate_model(self, model: nn.Module, calibration_data_loader):
        """
        Model kalibrasyonu yapar.

        Args:
            model: Kalibrasyon yapılacak prepared model
            calibration_data_loader: Kalibrasyon veri yükleyici
        """
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data_loader):
                if i >= self.calibration_iterations:
                    break
                model(inputs)

        logger.info(f"Model kalibrasyonu tamamlandı - {i+1} iterasyon")

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
        metrics = {}

        # Size reduction
        if 'model_size_mb' in original_metrics and 'model_size_mb' in quantized_metrics:
            original_size = original_metrics['model_size_mb']
            quantized_size = quantized_metrics['model_size_mb']
            if original_size > 0:
                size_reduction = ((original_size - quantized_size) / original_size) * 100
                metrics['size_reduction_percent'] = size_reduction
                metrics['compression_ratio'] = original_size / quantized_size if quantized_size > 0 else 0

        # Speed improvement
        if ('avg_inference_time_ms' in original_metrics and
                'avg_inference_time_ms' in quantized_metrics):
            original_time = original_metrics['avg_inference_time_ms']
            quantized_time = quantized_metrics['avg_inference_time_ms']
            if quantized_time > 0:
                speedup = original_time / quantized_time
                metrics['speedup_ratio'] = speedup

        return metrics

    def save_quantized_model(self, model: nn.Module, save_path: Union[str, Path],
                             metadata: Optional[Dict] = None):
        """
        Quantized modeli kaydeder.

        Args:
            model: Kaydedilecek quantized model
            save_path: Kaydetme yolu
            metadata: Ek metadata bilgileri
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(model.state_dict(), save_path)

        # Save metadata
        if metadata:
            metadata_path = save_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

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
    return QuantizationManager(
        backend=backend,
        calibration_iterations=calibration_iterations
    )
