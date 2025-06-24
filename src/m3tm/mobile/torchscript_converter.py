"""
M³TM TorchScript Converter Modülü

Bu modül PyTorch modellerini TorchScript'e dönüştürmeyi ve 
mobil platformlar için optimize etmeyi amaçlar.
"""

import logging
from typing import Dict, Optional, Union, Any, Tuple, List, Callable
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.jit import ScriptModule, trace, script
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.backends.quantized

from m3tm.core.base_model import BaseModel
from m3tm.mobile.benchmark_utils import ModelBenchmarker


logger = logging.getLogger(__name__)


class TorchScriptConfig:
    """TorchScript dönüştürme yapılandırma sınıfı."""
    
    def __init__(self,
                 method: str = "trace",
                 optimize_for_mobile: bool = True,
                 preserved_methods: Optional[List[str]] = None,
                 optimization_blocklist: Optional[Set[str]] = None,
                 backend: str = "CPU"):
        """
        Args:
            method: 'trace' veya 'script' dönüştürme methodu
            optimize_for_mobile: Mobil optimizasyonları uygula
            preserved_methods: Korunacak method'lar
            optimization_blocklist: Optimize edilmeyecek operasyonlar
            backend: Target backend ('CPU', 'Vulkan', 'Metal')
        """
        self.method = method
        self.optimize_for_mobile = optimize_for_mobile
        self.preserved_methods = preserved_methods or []
        self.optimization_blocklist = optimization_blocklist or set()
        self.backend = backend
        
        if method not in ["trace", "script"]:
            raise ValueError("Method 'trace' veya 'script' olmalı")


class TorchScriptConverter:
    """TorchScript dönüştürme yöneticisi."""
    
    def __init__(self, config: Optional[TorchScriptConfig] = None):
        """
        Args:
            config: TorchScript yapılandırma objesi
        """
        self.config = config or TorchScriptConfig()
        self.benchmarker = ModelBenchmarker()
    
    def convert_to_torchscript(self,
                               model: Union[nn.Module, BaseModel],
                               example_inputs: torch.Tensor,
                               benchmark: bool = True) -> Tuple[ScriptModule, Dict]:
        """
        PyTorch modelini TorchScript'e dönüştürür.
        
        Args:
            model: Dönüştürülecek model
            example_inputs: Örnek girdi tensörü
            benchmark: Performans ölçümü yapılsın mı
            
        Returns:
            Tuple[torchscript_model, conversion_metrics]
        """
        logger.info(f"TorchScript dönüştürme başlatılıyor ({self.config.method} method)")
        
        # Orijinal model metrics
        original_metrics = {}
        if benchmark:
            original_metrics = self.benchmarker.get_model_metrics(model, example_inputs)
        
        try:
            # Model evaluation moduna al
            model.eval()
            
            # TorchScript dönüştürme
            if self.config.method == "trace":
                scripted_model = self._trace_model(model, example_inputs)
            elif self.config.method == "script":
                scripted_model = self._script_model(model)
            else:
                raise ValueError(f"Desteklenmeyen method: {self.config.method}")
            
            # Mobil optimizasyonları uygula
            if self.config.optimize_for_mobile:
                scripted_model = self._apply_mobile_optimizations(scripted_model)
            
            # TorchScript model metrics
            torchscript_metrics = {}
            if benchmark:
                torchscript_metrics = self._get_torchscript_metrics(scripted_model, example_inputs)
            
            # Conversion metrics hesapla
            conversion_metrics = self._calculate_conversion_metrics(
                original_metrics, torchscript_metrics
            )
            
            logger.info("TorchScript dönüştürme tamamlandı")
            
            return scripted_model, {
                'original_metrics': original_metrics,
                'torchscript_metrics': torchscript_metrics,
                'conversion_metrics': conversion_metrics,
                'conversion_method': self.config.method,
                'mobile_optimized': self.config.optimize_for_mobile
            }
            
        except Exception as e:
            logger.error(f"TorchScript dönüştürme hatası: {e}")
            raise
    
    def _trace_model(self, model: nn.Module, example_inputs: torch.Tensor) -> ScriptModule:
        """
        Model'i trace method ile dönüştürür.
        
        Args:
            model: Trace edilecek model
            example_inputs: Örnek girdi tensörü
            
        Returns:
            Traced TorchScript model
        """
        logger.info("Model trace ediliyor...")
        
        try:
            # Model'i eval modunda trace et
            with torch.no_grad():
                traced_model = trace(model, example_inputs)
            
            # Trace verification
            self._verify_trace(model, traced_model, example_inputs)
            
            return traced_model
            
        except Exception as e:
            logger.error(f"Model trace hatası: {e}")
            raise
    
    def _script_model(self, model: nn.Module) -> ScriptModule:
        """
        Model'i script method ile dönüştürür.
        
        Args:
            model: Script edilecek model
            
        Returns:
            Scripted TorchScript model
        """
        logger.info("Model script ediliyor...")
        
        try:
            # Modeli script et
            scripted_model = script(model)
            
            return scripted_model
            
        except Exception as e:
            logger.error(f"Model script hatası: {e}")
            logger.info("Script hatası durumunda trace method denenebilir")
            raise
    
    def _verify_trace(self, original_model: nn.Module, 
                      traced_model: ScriptModule, 
                      example_inputs: torch.Tensor) -> bool:
        """
        Trace edilen model'in doğruluğunu kontrol eder.
        
        Args:
            original_model: Orijinal PyTorch model
            traced_model: Trace edilmiş TorchScript model
            example_inputs: Test girdileri
            
        Returns:
            Verification success flag
        """
        try:
            with torch.no_grad():
                original_output = original_model(example_inputs)
                traced_output = traced_model(example_inputs)
                
                # Output similarity check
                if isinstance(original_output, torch.Tensor):
                    similarity = torch.allclose(original_output, traced_output, rtol=1e-5, atol=1e-6)
                    
                    if similarity:
                        logger.info("Trace verification başarılı")
                        return True
                    else:
                        max_diff = (original_output - traced_output).abs().max().item()
                        logger.warning(f"Trace verification uyarısı - Max difference: {max_diff}")
                        return False
                else:
                    logger.info("Complex output structure - manual verification required")
                    return True
                    
        except Exception as e:
            logger.warning(f"Trace verification hatası: {e}")
            return False
    
    def _apply_mobile_optimizations(self, scripted_model: ScriptModule) -> ScriptModule:
        """
        Mobil optimizasyonları uygular.
        
        Args:
            scripted_model: Optimize edilecek TorchScript model
            
        Returns:
            Mobile optimized TorchScript model
        """
        logger.info("Mobil optimizasyonları uygulanıyor...")
        
        try:
            # PyTorch Mobile optimizer kullan
            optimization_blocklist = self.config.optimization_blocklist
            preserved_methods = self.config.preserved_methods
            
            optimized_model = optimize_for_mobile(
                scripted_model,
                optimization_blocklist=optimization_blocklist,
                preserved_methods=preserved_methods,
                backend=self.config.backend
            )
            
            logger.info("Mobil optimizasyonları tamamlandı")
            return optimized_model
            
        except Exception as e:
            logger.warning(f"Mobil optimizasyon hatası: {e}")
            logger.info("Optimizasyon olmadan devam ediliyor")
            return scripted_model
    
    def _get_torchscript_metrics(self, scripted_model: ScriptModule, 
                                 example_inputs: torch.Tensor) -> Dict:
        """
        TorchScript model metrics hesaplar.
        
        Args:
            scripted_model: TorchScript model
            example_inputs: Örnek girdi tensörü
            
        Returns:
            TorchScript model metrics
        """
        metrics = {}
        
        try:
            # Model size estimation
            temp_path = Path("temp_torchscript_model.pt")
            torch.jit.save(scripted_model, temp_path)
            
            if temp_path.exists():
                model_size_bytes = temp_path.stat().st_size
                model_size_mb = model_size_bytes / (1024 * 1024)
                
                metrics.update({
                    'model_size_bytes': model_size_bytes,
                    'model_size_mb': model_size_mb
                })
                
                # Cleanup
                temp_path.unlink()
            
            # Performance metrics
            inference_metrics = self.benchmarker.measure_inference_performance(
                scripted_model, example_inputs
            )
            metrics.update(inference_metrics)
            
        except Exception as e:
            logger.warning(f"TorchScript metrics hesaplama hatası: {e}")
            
        return metrics
    
    def _calculate_conversion_metrics(self, original_metrics: Dict, 
                                      torchscript_metrics: Dict) -> Dict:
        """
        Conversion metrics hesaplar.
        
        Args:
            original_metrics: Orijinal model metrikleri
            torchscript_metrics: TorchScript model metrikleri
            
        Returns:
            Conversion metrics dict
        """
        if not original_metrics or not torchscript_metrics:
            return {}
        
        conversion_metrics = {}
        
        # Size comparison
        if 'model_size_mb' in original_metrics and 'model_size_mb' in torchscript_metrics:
            original_size = original_metrics['model_size_mb']
            torchscript_size = torchscript_metrics['model_size_mb']
            
            if original_size > 0:
                size_change = (torchscript_size - original_size) / original_size * 100
                conversion_metrics.update({
                    'size_change_percent': size_change,
                    'original_size_mb': original_size,
                    'torchscript_size_mb': torchscript_size
                })
        
        # Speed comparison
        if 'avg_inference_time_ms' in original_metrics and 'avg_inference_time_ms' in torchscript_metrics:
            original_time = original_metrics['avg_inference_time_ms']
            torchscript_time = torchscript_metrics['avg_inference_time_ms']
            
            if original_time > 0:
                speedup = original_time / torchscript_time if torchscript_time > 0 else float('inf')
                time_change = (torchscript_time - original_time) / original_time * 100
                
                conversion_metrics.update({
                    'speedup_ratio': speedup,
                    'time_change_percent': time_change
                })
        
        return conversion_metrics
    
    def save_torchscript_model(self, scripted_model: ScriptModule, 
                               save_path: Union[str, Path],
                               metadata: Optional[Dict] = None,
                               include_extra_files: bool = True) -> None:
        """
        TorchScript modeli kaydeder.
        
        Args:
            scripted_model: Kaydedilecek TorchScript model
            save_path: Kaydetme yolu
            metadata: Ek metadata bilgileri
            include_extra_files: Extra files dahil et
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extra files hazırla
            extra_files = {}
            if include_extra_files and metadata:
                import json
                extra_files['metadata.json'] = json.dumps(metadata, indent=2)
                extra_files['config.json'] = json.dumps({
                    'conversion_method': self.config.method,
                    'mobile_optimized': self.config.optimize_for_mobile,
                    'backend': self.config.backend
                }, indent=2)
            
            # Model kaydet
            if extra_files:
                torch.jit.save(scripted_model, save_path, _extra_files=extra_files)
            else:
                torch.jit.save(scripted_model, save_path)
            
            logger.info(f"TorchScript model kaydedildi: {save_path}")
            
        except Exception as e:
            logger.error(f"TorchScript model kaydetme hatası: {e}")
            raise
    
    def load_torchscript_model(self, model_path: Union[str, Path],
                               device: Optional[torch.device] = None) -> Tuple[ScriptModule, Dict]:
        """
        TorchScript modeli yükler.
        
        Args:
            model_path: Model dosya yolu
            device: Target device
            
        Returns:
            Tuple[loaded_model, metadata]
        """
        model_path = Path(model_path)
        device = device or torch.device('cpu')
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        try:
            # Model yükle
            scripted_model = torch.jit.load(model_path, map_location=device)
            
            # Extra files yükle (varsa)
            metadata = {}
            try:
                extra_files = {'metadata.json': '', 'config.json': ''}
                torch.jit.load(model_path, map_location=device, _extra_files=extra_files)
                
                import json
                if extra_files.get('metadata.json'):
                    metadata.update(json.loads(extra_files['metadata.json']))
                if extra_files.get('config.json'):
                    metadata['config'] = json.loads(extra_files['config.json'])
                    
            except Exception:
                logger.info("Extra files yüklenemedi, devam ediliyor")
            
            logger.info(f"TorchScript model yüklendi: {model_path}")
            return scripted_model, metadata
            
        except Exception as e:
            logger.error(f"TorchScript model yükleme hatası: {e}")
            raise
    
    def export_for_platform(self, scripted_model: ScriptModule,
                             platform: str,
                             output_dir: Union[str, Path],
                             model_name: str = "optimized_model") -> Dict[str, Path]:
        """
        Platform-specific export yapar.
        
        Args:
            scripted_model: Export edilecek TorchScript model
            platform: 'android', 'ios', 'desktop'
            output_dir: Çıktı dizini
            model_name: Model dosya adı
            
        Returns:
            Export edilen dosya yolları
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        try:
            if platform.lower() == "android":
                # Android için .ptl format
                android_path = output_dir / f"{model_name}_android.ptl"
                torch.jit.save(scripted_model, android_path)
                exported_files['android'] = android_path
                
            elif platform.lower() == "ios":
                # iOS için .ptl format
                ios_path = output_dir / f"{model_name}_ios.ptl"
                torch.jit.save(scripted_model, ios_path)
                exported_files['ios'] = ios_path
                
            elif platform.lower() == "desktop":
                # Desktop için .pt format
                desktop_path = output_dir / f"{model_name}_desktop.pt"
                torch.jit.save(scripted_model, desktop_path)
                exported_files['desktop'] = desktop_path
                
            else:
                # Generic export
                generic_path = output_dir / f"{model_name}.pt"
                torch.jit.save(scripted_model, generic_path)
                exported_files['generic'] = generic_path
            
            logger.info(f"Platform export tamamlandı: {platform}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Platform export hatası: {e}")
            raise
    
    def create_deployment_package(self, scripted_model: ScriptModule,
                                  output_dir: Union[str, Path],
                                  include_examples: bool = True,
                                  include_benchmarks: bool = True) -> Path:
        """
        Deployment package oluşturur.
        
        Args:
            scripted_model: Package edilecek model
            output_dir: Çıktı dizini
            include_examples: Örnek kodları dahil et
            include_benchmarks: Benchmark sonuçlarını dahil et
            
        Returns:
            Package dizin yolu
        """
        output_dir = Path(output_dir)
        package_dir = output_dir / "deployment_package"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Model files
            models_dir = package_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Platform exports
            platform_exports = {}
            for platform in ["android", "ios", "desktop"]:
                try:
                    exports = self.export_for_platform(scripted_model, platform, models_dir)
                    platform_exports.update(exports)
                except Exception as e:
                    logger.warning(f"Platform export hatası ({platform}): {e}")
            
            # Documentation
            docs_dir = package_dir / "docs"
            docs_dir.mkdir(exist_ok=True)
            
            # README oluştur
            readme_content = self._generate_deployment_readme(platform_exports)
            with open(docs_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            # Example codes
            if include_examples:
                examples_dir = package_dir / "examples"
                examples_dir.mkdir(exist_ok=True)
                self._create_example_codes(examples_dir, platform_exports)
            
            # Benchmarks
            if include_benchmarks:
                benchmarks_dir = package_dir / "benchmarks"
                benchmarks_dir.mkdir(exist_ok=True)
                # Benchmark results burada eklenebilir
            
            logger.info(f"Deployment package oluşturuldu: {package_dir}")
            return package_dir
            
        except Exception as e:
            logger.error(f"Deployment package oluşturma hatası: {e}")
            raise
    
    def _generate_deployment_readme(self, platform_exports: Dict[str, Path]) -> str:
        """Deployment README içeriği oluşturur."""
        content = [
            "# M³TM Mobile Model Deployment Package",
            "",
            "Bu package M³TM modelinin mobil platformlarda kullanımı için optimize edilmiş dosyaları içerir.",
            "",
            "## Model Dosyaları",
            ""
        ]
        
        for platform, path in platform_exports.items():
            content.append(f"- **{platform.capitalize()}**: `{path.name}`")
        
        content.extend([
            "",
            "## Kullanım",
            "",
            "### Android",
            "```java",
            "// Android örnek kodu examples/android/ klasöründe",
            "```",
            "",
            "### iOS",
            "```swift",
            "// iOS örnek kodu examples/ios/ klasöründe",
            "```",
            "",
            "### Desktop",
            "```python",
            "import torch",
            "model = torch.jit.load('models/optimized_model_desktop.pt')",
            "```"
        ])
        
        return "\n".join(content)
    
    def _create_example_codes(self, examples_dir: Path, platform_exports: Dict[str, Path]) -> None:
        """Platform-specific örnek kodları oluşturur."""
        
        # Python example
        python_example = '''
import torch
import numpy as np

def load_and_run_model(model_path, input_data):
    """
    TorchScript modelini yükler ve çalıştırır.
    
    Args:
        model_path: Model dosya yolu
        input_data: Girdi verisi (numpy array)
    
    Returns:
        Model çıktısı
    """
    # Model yükle
    model = torch.jit.load(model_path)
    model.eval()
    
    # Input'u tensor'e dönüştür
    input_tensor = torch.from_numpy(input_data).float()
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    return output.numpy()

# Örnek kullanım
if __name__ == "__main__":
    model_path = "models/optimized_model_desktop.pt"
    # Örnek girdi (model'e göre ayarlanmalı)
    example_input = np.random.randn(1, 3, 224, 224)  # Batch, Channel, Height, Width
    
    result = load_and_run_model(model_path, example_input)
    print(f"Model output shape: {result.shape}")
'''
        
        with open(examples_dir / "python_example.py", 'w', encoding='utf-8') as f:
            f.write(python_example)
        
        # Android example (Java)
        android_example = '''
// Android TorchScript model kullanımı örneği
// build.gradle'a dependency ekleyin:
// implementation 'org.pytorch:pytorch_android:1.13.1'

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class ModelInference {
    private Module model;
    
    public void loadModel(String modelPath) {
        model = Module.load(modelPath);
    }
    
    public float[] runInference(float[] inputData, int[] shape) {
        Tensor inputTensor = Tensor.fromBlob(inputData, shape);
        IValue output = model.forward(IValue.from(inputTensor));
        return output.toTensor().getDataAsFloatArray();
    }
}
'''
        
        android_dir = examples_dir / "android"
        android_dir.mkdir(exist_ok=True)
        with open(android_dir / "ModelInference.java", 'w', encoding='utf-8') as f:
            f.write(android_example)


def create_torchscript_pipeline(method: str = "trace",
                                optimize_for_mobile: bool = True,
                                backend: str = "CPU") -> TorchScriptConverter:
    """
    TorchScript conversion pipeline oluşturucu fonksiyon.
    
    Args:
        method: Conversion method ('trace' veya 'script')
        optimize_for_mobile: Mobil optimizasyonları uygula
        backend: Target backend
        
    Returns:
        TorchScriptConverter instance
    """
    config = TorchScriptConfig(
        method=method,
        optimize_for_mobile=optimize_for_mobile,
        backend=backend
    )
    
    return TorchScriptConverter(config)
