"""
M³TM v2.3 Çıkarım Köprüsü modülü.

Bu modül, model çıkarım işlemlerini Java/Kotlin tarafından 
çağrılabilir JNI işlevleri olarak sunar.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import time

import torch
import torch.nn.functional as F
import numpy as np

from .jni_helpers import (
    convert_tensor_to_java,
    convert_java_to_tensor,
    convert_dict_to_java_map,
    convert_list_to_java_list,
    JNIError,
    JNIRuntimeError,
    safe_jni_call
)
from .model_loader import ModelLoader, ModelNotFoundError

class InferenceError(Exception):
    """Çıkarım işlemi sırasında oluşan hatalar için istisna sınıfı."""
    pass

class InferenceBridge:
    """
    Model çıkarım işlemlerini JNI üzerinden Java/Kotlin'e sunan köprü sınıfı.
    
    Bu sınıf, modelleri kullanarak metin ve görüntüler üzerinde çıkarım yapmak,
    çoklu-modalite füzyonu gerçekleştirmek ve sonuçları Java formatında döndürmek
    için işlevler sağlar.
    """
    
    def __init__(self, model_loader: ModelLoader):
        """
        InferenceBridge sınıfını başlatır.
        
        Args:
            model_loader: Model yükleme ve yönetimi için ModelLoader örneği.
        """
        self.model_loader = model_loader
        logging.info("InferenceBridge başlatıldı")
    
    def run_text_inference(
        self,
        model_id: str,
        text: str,
        task_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Metin verisi üzerinde model çıkarımı yapar.
        
        Args:
            model_id: Kullanılacak modelin benzersiz tanımlayıcısı.
            text: İşlenecek metin.
            task_name: İsteğe bağlı görev adı (birden fazla görev başlığı destekleniyorsa).
            options: Çıkarım seçenekleri sözlüğü.
            
        Returns:
            result: Çıkarım sonuçları.
            
        Raises:
            ModelNotFoundError: Model bulunamadığında.
            InferenceError: Çıkarım işlemi sırasında hata oluştuğunda.
        """
        # Varsayılan seçenekleri ayarla
        options = options or {}
        
        # Modeli al
        try:
            model = self.model_loader.get_model(model_id)
        except ModelNotFoundError as e:
            raise e
        
        # Giriş doğrulama
        if not text or not isinstance(text, str):
            raise InferenceError("Geçerli bir metin girdisi gerekiyor")
        
        try:
            # İşlemi zamanla
            start_time = time.time()
            
            # Çıkarım işlemi
            with torch.no_grad():
                # Modelin doğrudan çıkarım işlevini kullan
                if hasattr(model, "run_text_inference"):
                    outputs = model.run_text_inference(text, task_name, options)
                else:
                    # Varsayılan çıkarım işlevi
                    inputs = {"text": text, "task": task_name, "options": options}
                    outputs = model.forward(inputs)
            
            # İşlem süresini hesapla
            inference_time = time.time() - start_time
            
            # Çıktıları hazırla
            result = {
                "outputs": outputs,
                "metadata": {
                    "model_id": model_id,
                    "task": task_name or "default",
                    "inference_time": inference_time,
                    "text_length": len(text)
                }
            }
            
            return result
            
        except Exception as e:
            raise InferenceError(f"Metin çıkarımı sırasında hata: {str(e)}")
    
    def run_image_inference(
        self,
        model_id: str,
        image_tensor: torch.Tensor,
        task_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Görüntü verisi üzerinde model çıkarımı yapar.
        
        Args:
            model_id: Kullanılacak modelin benzersiz tanımlayıcısı.
            image_tensor: İşlenecek görüntü tensörü.
            task_name: İsteğe bağlı görev adı (birden fazla görev başlığı destekleniyorsa).
            options: Çıkarım seçenekleri sözlüğü.
            
        Returns:
            result: Çıkarım sonuçları.
            
        Raises:
            ModelNotFoundError: Model bulunamadığında.
            InferenceError: Çıkarım işlemi sırasında hata oluştuğunda.
        """
        # Varsayılan seçenekleri ayarla
        options = options or {}
        
        # Modeli al
        try:
            model = self.model_loader.get_model(model_id)
        except ModelNotFoundError as e:
            raise e
        
        # Giriş doğrulama
        if not isinstance(image_tensor, torch.Tensor):
            raise InferenceError("Geçerli bir görüntü tensörü gerekiyor")
        
        try:
            # İşlemi zamanla
            start_time = time.time()
            
            # Çıkarım işlemi
            with torch.no_grad():
                # Modelin doğrudan çıkarım işlevini kullan
                if hasattr(model, "run_image_inference"):
                    outputs = model.run_image_inference(image_tensor, task_name, options)
                else:
                    # Varsayılan çıkarım işlevi
                    inputs = {"image": image_tensor, "task": task_name, "options": options}
                    outputs = model.forward(inputs)
            
            # İşlem süresini hesapla
            inference_time = time.time() - start_time
            
            # Çıktıları hazırla
            result = {
                "outputs": outputs,
                "metadata": {
                    "model_id": model_id,
                    "task": task_name or "default",
                    "inference_time": inference_time,
                    "image_shape": list(image_tensor.shape)
                }
            }
            
            return result
            
        except Exception as e:
            raise InferenceError(f"Görüntü çıkarımı sırasında hata: {str(e)}")
    
    def run_multimodal_inference(
        self,
        model_id: str,
        text: Optional[str] = None,
        image_tensor: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Metin ve görüntü verilerini birleştirerek çoklu-modalite çıkarımı yapar.
        
        Args:
            model_id: Kullanılacak modelin benzersiz tanımlayıcısı.
            text: İşlenecek metin (opsiyonel).
            image_tensor: İşlenecek görüntü tensörü (opsiyonel).
            task_name: İsteğe bağlı görev adı (birden fazla görev başlığı destekleniyorsa).
            options: Çıkarım seçenekleri sözlüğü.
            
        Returns:
            result: Çıkarım sonuçları.
            
        Raises:
            ModelNotFoundError: Model bulunamadığında.
            InferenceError: Çıkarım işlemi sırasında hata oluştuğunda.
        """
        # Varsayılan seçenekleri ayarla
        options = options or {}
        
        # Modeli al
        try:
            model = self.model_loader.get_model(model_id)
        except ModelNotFoundError as e:
            raise e
        
        # Giriş doğrulama
        if text is None and image_tensor is None:
            raise InferenceError("En az bir metin veya görüntü girdisi gerekiyor")
        
        if image_tensor is not None and not isinstance(image_tensor, torch.Tensor):
            raise InferenceError("Geçerli bir görüntü tensörü gerekiyor")
        
        try:
            # İşlemi zamanla
            start_time = time.time()
            
            # Çıkarım işlemi
            with torch.no_grad():
                # Modelin doğrudan çıkarım işlevini kullan
                if hasattr(model, "run_multimodal_inference"):
                    outputs = model.run_multimodal_inference(text, image_tensor, task_name, options)
                else:
                    # Varsayılan çıkarım işlevi
                    inputs = {
                        "text": text,
                        "image": image_tensor,
                        "task": task_name,
                        "options": options
                    }
                    outputs = model.forward(inputs)
            
            # İşlem süresini hesapla
            inference_time = time.time() - start_time
            
            # Çıktıları hazırla
            result = {
                "outputs": outputs,
                "metadata": {
                    "model_id": model_id,
                    "task": task_name or "default",
                    "inference_time": inference_time,
                    "has_text": text is not None,
                    "has_image": image_tensor is not None,
                    "text_length": len(text) if text else 0,
                    "image_shape": list(image_tensor.shape) if image_tensor is not None else None
                }
            }
            
            return result
            
        except Exception as e:
            raise InferenceError(f"Çoklu-modalite çıkarımı sırasında hata: {str(e)}")
    
    # JNI köprü işlevleri
    
    def jni_run_text_inference(
        self,
        model_id: str,
        text: str,
        task_name: Optional[str] = None,
        options_java: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek metin çıkarım işlevi.
        
        Args:
            model_id: Kullanılacak modelin benzersiz tanımlayıcısı.
            text: İşlenecek metin.
            task_name: İsteğe bağlı görev adı.
            options_java: Java formatındaki çıkarım seçenekleri.
            
        Returns:
            result: İşlem sonucu ve çıkarım sonuçları.
        """
        # Java seçeneklerini Python sözlüğüne dönüştür
        options = None
        if options_java:
            try:
                # Burada Java Map'ten Python dict'e dönüşüm yapılacak
                # (JNI implementasyonunda bu adım uygulanacak)
                options = options_java
            except Exception as e:
                logging.warning(f"Seçenekler dönüştürülemedi: {e}")
        
        # Güvenli çağrı ile çıkarım yap
        result = safe_jni_call(self.run_text_inference, model_id, text, task_name, options)
        
        # Başarılı durumda, çıktıları Java formatına dönüştür
        if result["status"] == "success":
            outputs_dict = result["data"]["outputs"]
            metadata_dict = result["data"]["metadata"]
            
            # Çıktıları Java formatına dönüştür
            java_outputs = {}
            for key, value in outputs_dict.items():
                if isinstance(value, torch.Tensor):
                    java_outputs[key] = convert_tensor_to_java(value)
                elif isinstance(value, np.ndarray):
                    java_outputs[key] = convert_tensor_to_java(torch.from_numpy(value))
                elif isinstance(value, (dict, list)):
                    if isinstance(value, dict):
                        java_outputs[key] = convert_dict_to_java_map(value)
                    else:
                        java_outputs[key] = convert_list_to_java_list(value)
                else:
                    # Primitive türler doğrudan kalsın
                    java_outputs[key] = value
            
            # Metadata'yı Java formatına dönüştür
            java_metadata = convert_dict_to_java_map(metadata_dict)
            
            # Sonuç paketini güncelle
            result["data"] = {
                "outputs": convert_dict_to_java_map(java_outputs),
                "metadata": java_metadata
            }
            
            # Tüm veri paketini Java Map'e dönüştür
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result
    
    def jni_run_image_inference(
        self,
        model_id: str,
        image_data_java: Dict[str, Any],
        task_name: Optional[str] = None,
        options_java: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek görüntü çıkarım işlevi.
        
        Args:
            model_id: Kullanılacak modelin benzersiz tanımlayıcısı.
            image_data_java: Java formatında görüntü verisi.
            task_name: İsteğe bağlı görev adı.
            options_java: Java formatındaki çıkarım seçenekleri.
            
        Returns:
            result: İşlem sonucu ve çıkarım sonuçları.
        """
        # Java görüntü verisini PyTorch tensörüne dönüştür
        image_tensor = None
        try:
            image_tensor = convert_java_to_tensor(image_data_java)
        except Exception as e:
            return {
                "status": "error",
                "error": {
                    "exception_type": "ImageConversionError",
                    "message": f"Görüntü verisi tensöre dönüştürülemedi: {str(e)}",
                    "stack_trace": ""
                }
            }
        
        # Java seçeneklerini Python sözlüğüne dönüştür
        options = None
        if options_java:
            try:
                # Burada Java Map'ten Python dict'e dönüşüm yapılacak
                options = options_java
            except Exception as e:
                logging.warning(f"Seçenekler dönüştürülemedi: {e}")
        
        # Güvenli çağrı ile çıkarım yap
        result = safe_jni_call(self.run_image_inference, model_id, image_tensor, task_name, options)
        
        # Başarılı durumda, çıktıları Java formatına dönüştür
        if result["status"] == "success":
            outputs_dict = result["data"]["outputs"]
            metadata_dict = result["data"]["metadata"]
            
            # Çıktıları Java formatına dönüştür
            java_outputs = {}
            for key, value in outputs_dict.items():
                if isinstance(value, torch.Tensor):
                    java_outputs[key] = convert_tensor_to_java(value)
                elif isinstance(value, np.ndarray):
                    java_outputs[key] = convert_tensor_to_java(torch.from_numpy(value))
                elif isinstance(value, (dict, list)):
                    if isinstance(value, dict):
                        java_outputs[key] = convert_dict_to_java_map(value)
                    else:
                        java_outputs[key] = convert_list_to_java_list(value)
                else:
                    # Primitive türler doğrudan kalsın
                    java_outputs[key] = value
            
            # Metadata'yı Java formatına dönüştür
            java_metadata = convert_dict_to_java_map(metadata_dict)
            
            # Sonuç paketini güncelle
            result["data"] = {
                "outputs": convert_dict_to_java_map(java_outputs),
                "metadata": java_metadata
            }
            
            # Tüm veri paketini Java Map'e dönüştür
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result
    
    def jni_run_multimodal_inference(
        self,
        model_id: str,
        text: Optional[str] = None,
        image_data_java: Optional[Dict[str, Any]] = None,
        task_name: Optional[str] = None,
        options_java: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek çoklu-modalite çıkarım işlevi.
        
        Args:
            model_id: Kullanılacak modelin benzersiz tanımlayıcısı.
            text: İşlenecek metin (opsiyonel).
            image_data_java: Java formatında görüntü verisi (opsiyonel).
            task_name: İsteğe bağlı görev adı.
            options_java: Java formatındaki çıkarım seçenekleri.
            
        Returns:
            result: İşlem sonucu ve çıkarım sonuçları.
        """
        # Java görüntü verisini PyTorch tensörüne dönüştür
        image_tensor = None
        if image_data_java:
            try:
                image_tensor = convert_java_to_tensor(image_data_java)
            except Exception as e:
                return {
                    "status": "error",
                    "error": {
                        "exception_type": "ImageConversionError",
                        "message": f"Görüntü verisi tensöre dönüştürülemedi: {str(e)}",
                        "stack_trace": ""
                    }
                }
        
        # Java seçeneklerini Python sözlüğüne dönüştür
        options = None
        if options_java:
            try:
                # Burada Java Map'ten Python dict'e dönüşüm yapılacak
                options = options_java
            except Exception as e:
                logging.warning(f"Seçenekler dönüştürülemedi: {e}")
        
        # Güvenli çağrı ile çıkarım yap
        result = safe_jni_call(
            self.run_multimodal_inference,
            model_id,
            text,
            image_tensor,
            task_name,
            options
        )
        
        # Başarılı durumda, çıktıları Java formatına dönüştür
        if result["status"] == "success":
            outputs_dict = result["data"]["outputs"]
            metadata_dict = result["data"]["metadata"]
            
            # Çıktıları Java formatına dönüştür
            java_outputs = {}
            for key, value in outputs_dict.items():
                if isinstance(value, torch.Tensor):
                    java_outputs[key] = convert_tensor_to_java(value)
                elif isinstance(value, np.ndarray):
                    java_outputs[key] = convert_tensor_to_java(torch.from_numpy(value))
                elif isinstance(value, (dict, list)):
                    if isinstance(value, dict):
                        java_outputs[key] = convert_dict_to_java_map(value)
                    else:
                        java_outputs[key] = convert_list_to_java_list(value)
                else:
                    # Primitive türler doğrudan kalsın
                    java_outputs[key] = value
            
            # Metadata'yı Java formatına dönüştür
            java_metadata = convert_dict_to_java_map(metadata_dict)
            
            # Sonuç paketini güncelle
            result["data"] = {
                "outputs": convert_dict_to_java_map(java_outputs),
                "metadata": java_metadata
            }
            
            # Tüm veri paketini Java Map'e dönüştür
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result 