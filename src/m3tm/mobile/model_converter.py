"""
PyTorch modellerini mobil platformlarda kullanılabilir formatlara çeviren modül.
"""

import os
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn

from m3tm.core.base_model import BaseModel


class ModelConverter:
    """PyTorch modellerini mobil platformlar için uygun formatlara dönüştürür."""
    
    @staticmethod
    def to_torchscript(
        model: Union[nn.Module, BaseModel],
        example_inputs: Any,
        save_path: Optional[str] = None,
        method: str = "trace",
        optimize: bool = True
    ) -> torch.jit.ScriptModule:
        """
        PyTorch modelini TorchScript formatına dönüştürür.
        
        Args:
            model: Dönüştürülecek PyTorch modeli
            example_inputs: Modele giriş için örnek veri
            save_path: TorchScript modelinin kaydedileceği dosya yolu (opsiyonel)
            method: Dönüştürme yöntemi, "trace" veya "script"
            optimize: Modeli optimize etmek için torch.jit.optimize_for_mobile kullan
            
        Returns:
            TorchScript modeli
        """
        if method == "trace":
            traced_model = torch.jit.trace(model, example_inputs)
            script_model = torch.jit.script(traced_model)
        elif method == "script":
            script_model = torch.jit.script(model)
        else:
            raise ValueError(f"Geçersiz method: {method}. 'trace' veya 'script' kullanın.")
        
        if optimize:
            script_model = torch.jit.optimize_for_mobile(script_model)
            
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.jit.save(script_model, save_path)
            print(f"Model başarıyla kaydedildi: {save_path}")
            
        return script_model
    
    @staticmethod
    def optimize_for_mobile(
        model: Union[nn.Module, BaseModel],
        quantize: bool = False,
        quantization_config: Optional[Dict] = None,
        example_inputs: Optional[Any] = None,
        save_path: Optional[str] = None
    ) -> Union[torch.jit.ScriptModule, nn.Module]:
        """
        PyTorch modelini mobil cihazlar için optimize eder.
        
        Args:
            model: Optimize edilecek model
            quantize: Modeli nicelendirmek için
            quantization_config: Nicelendirme parametreleri
            example_inputs: Dinamik nicelendirme için örnek girdiler
            save_path: Optimize edilmiş modelin kaydedileceği dosya yolu
            
        Returns:
            Optimize edilmiş model
        """
        if not quantize:
            # Sadece TorchScript'e dönüştür ve mobil için optimize et
            return ModelConverter.to_torchscript(
                model, 
                example_inputs, 
                save_path=save_path, 
                optimize=True
            )
        
        # Burada daha gelişmiş quantization işlemleri eklenecek
        # PyTorch'un post-training quantization veya QAT (Quantization Aware Training) ile
        # Bu kısım henüz prototip aşamasında
        
        # Basit bir dynamic quantization örneği:
        if example_inputs is not None:
            # İlk olarak TorchScript'e dönüştür
            scripted_model = ModelConverter.to_torchscript(
                model, 
                example_inputs, 
                optimize=False
            )
            
            # TODO: Mobil için uygun quantization stratejileri eklenecek
            # Şu an için basit bir placeholder
            print("Quantization henüz tam olarak uygulanmadı.")
            
            # Sonucu kaydet
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.jit.save(scripted_model, save_path)
                print(f"Optimize edilmiş model başarıyla kaydedildi: {save_path}")
                
            return scripted_model
        else:
            raise ValueError("Quantization için example_inputs gereklidir.")
            
    @staticmethod
    def convert_for_android(
        model: Union[nn.Module, BaseModel],
        example_inputs: Any,
        save_path: str,
        optimize: bool = True,
        quantize: bool = False
    ) -> str:
        """
        PyTorch modelini Android platformu için uygun formata dönüştürür.
        
        Args:
            model: Dönüştürülecek model
            example_inputs: Örnek girdiler
            save_path: Kaydedilecek dosya yolu (.pt uzantılı)
            optimize: Mobil için optimize edilsin mi
            quantize: Nicelendirme yapılsın mı
            
        Returns:
            Kaydedilen dosyanın yolu
        """
        if quantize:
            ModelConverter.optimize_for_mobile(
                model, 
                quantize=True, 
                example_inputs=example_inputs,
                save_path=save_path
            )
        else:
            ModelConverter.to_torchscript(
                model, 
                example_inputs,
                save_path=save_path,
                optimize=optimize
            )
        
        return save_path
    
    @staticmethod
    def convert_for_ios(
        model: Union[nn.Module, BaseModel],
        example_inputs: Any,
        save_path: str,
        optimize: bool = True,
        quantize: bool = False
    ) -> str:
        """
        PyTorch modelini iOS platformu için uygun formata dönüştürür.
        
        Args:
            model: Dönüştürülecek model
            example_inputs: Örnek girdiler
            save_path: Kaydedilecek dosya yolu (.pt uzantılı)
            optimize: Mobil için optimize edilsin mi
            quantize: Nicelendirme yapılsın mı
            
        Returns:
            Kaydedilen dosyanın yolu
        """
        # iOS için dönüşüm genellikle Android ile aynıdır, ancak 
        # ileride platform-spesifik optimizasyonlar eklenebilir
        return ModelConverter.convert_for_android(
            model,
            example_inputs,
            save_path,
            optimize,
            quantize
        ) 