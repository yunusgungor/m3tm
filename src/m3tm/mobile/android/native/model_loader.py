"""
M³TM v2.3 Model Yükleyici modülü.

Bu modül, TorchScript formatındaki M³TM modellerini yüklemek, 
mobil platformlarda çalıştırmak ve yönetmek için gerekli işlevleri sağlar.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json

import torch
import numpy as np

from .jni_helpers import (
    convert_tensor_to_java, 
    convert_java_to_tensor,
    convert_dict_to_java_map,
    JNIError,
    JNIRuntimeError,
    safe_jni_call
)

# Sabitler
DEFAULT_MODEL_CACHE_DIR = "model_cache"
TORCHSCRIPT_EXTENSION = ".pt"
MODEL_CONFIG_EXTENSION = ".json"
MODEL_INFO_FILE = "model_info.json"

class ModelLoadError(Exception):
    """Model yükleme sırasında oluşan hataları temsil eden istisna."""
    pass

class ModelNotFoundError(ModelLoadError):
    """Model dosyasının bulunamadığı durumlarda oluşan istisna."""
    pass

class ModelVersionError(ModelLoadError):
    """Model versiyonu uyumsuzlukları için istisna."""
    pass

class ModelLoader:
    """
    M³TM modellerini yüklemek, doğrulamak ve yönetmek için kullanılan sınıf.
    
    Bu sınıf, TorchScript formatındaki modelleri yüklemeyi, model meta verilerini
    işlemeyi ve modelin bellekte yönetimini kolaylaştıran yöntemler sağlar.
    """
    
    def __init__(
        self, 
        model_cache_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        ModelLoader sınıfını başlatır.
        
        Args:
            model_cache_dir: Model önbellek dizini. Belirtilmezse varsayılan kullanılır.
            device: Modelin yükleneceği cihaz ('cpu' veya 'cuda').
        """
        self.model_cache_dir = model_cache_dir or DEFAULT_MODEL_CACHE_DIR
        self.device = device
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        
        # Önbellek dizininin varlığını kontrol et ve oluştur
        cache_path = Path(self.model_cache_dir)
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Model önbellek dizini oluşturuldu: {self.model_cache_dir}")
        
        logging.info(f"ModelLoader başlatıldı: önbellek={self.model_cache_dir}, cihaz={self.device}")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Önbellek dizininde bulunan modellerin listesini döndürür.
        
        Returns:
            model_info_list: Her modelin meta verilerini içeren sözlükler listesi.
        """
        available_models = []
        cache_path = Path(self.model_cache_dir)
        
        for model_dir in [d for d in cache_path.iterdir() if d.is_dir()]:
            info_file = model_dir / MODEL_INFO_FILE
            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                        model_info['model_id'] = model_dir.name
                        available_models.append(model_info)
                except (json.JSONDecodeError, OSError) as e:
                    logging.warning(f"Model bilgisi okunamadı {info_file}: {e}")
        
        return available_models
    
    def load_model(
        self, 
        model_id: str, 
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Belirtilen model ID'sine sahip modeli yükler.
        
        Args:
            model_id: Yüklenecek modelin benzersiz tanımlayıcısı.
            version: İsteğe bağlı model versiyonu. Belirtilmezse en son versiyon kullanılır.
            
        Returns:
            loaded_model_info: Yüklenen model hakkında bilgi içeren sözlük.
            
        Raises:
            ModelNotFoundError: Model bulunamadığında.
            ModelLoadError: Model yüklenirken hata oluştuğunda.
            ModelVersionError: İstenen versiyon bulunamadığında.
        """
        # Model zaten yüklü mü kontrol et
        if model_id in self.loaded_models:
            logging.info(f"Model zaten yüklü: {model_id}")
            return self.loaded_models[model_id]
        
        # Model dizinini kontrol et
        model_dir = Path(self.model_cache_dir) / model_id
        if not model_dir.exists() or not model_dir.is_dir():
            raise ModelNotFoundError(f"Model dizini bulunamadı: {model_dir}")
        
        # Model bilgi dosyasını oku
        info_file = model_dir / MODEL_INFO_FILE
        if not info_file.exists():
            raise ModelLoadError(f"Model bilgi dosyası bulunamadı: {info_file}")
        
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ModelLoadError(f"Model bilgisi okunamadı: {e}")
        
        # Versiyon kontrolü ve seçimi
        model_version = version or model_info.get('latest_version')
        if not model_version:
            raise ModelVersionError("Model versiyonu belirtilmedi ve varsayılan versiyon bulunamadı")
        
        # Model dosyası yolunu oluştur
        model_file = model_dir / f"{model_id}_v{model_version}{TORCHSCRIPT_EXTENSION}"
        if not model_file.exists():
            raise ModelNotFoundError(f"Model dosyası bulunamadı: {model_file}")
        
        # Model yapılandırma dosyasını kontrol et
        config_file = model_dir / f"{model_id}_v{model_version}{MODEL_CONFIG_EXTENSION}"
        config_data = None
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Model yapılandırması okunamadı: {e}")
        
        # Modeli yükle
        try:
            model = torch.jit.load(str(model_file), map_location=self.device)
            model.eval()  # Çıkarım modunu ayarla
        except Exception as e:
            raise ModelLoadError(f"Model yüklenemedi ({model_file}): {str(e)}")
        
        # Yüklenen model bilgisini hazırla
        loaded_model = {
            "model_id": model_id,
            "version": model_version,
            "model": model,
            "info": model_info,
            "config": config_data,
            "file_path": str(model_file),
            "load_time": torch.cuda.Event().record() if self.device == "cuda" else None
        }
        
        # Modeli önbelleğe al
        self.loaded_models[model_id] = loaded_model
        logging.info(f"Model başarıyla yüklendi: {model_id} (v{model_version})")
        
        return {
            "model_id": model_id,
            "version": model_version,
            "info": model_info,
            "config": config_data
        }
    
    def unload_model(self, model_id: str) -> bool:
        """
        Belirtilen modeli bellekten kaldırır.
        
        Args:
            model_id: Kaldırılacak modelin benzersiz tanımlayıcısı.
            
        Returns:
            success: İşlemin başarı durumu.
        """
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logging.info(f"Model bellekten kaldırıldı: {model_id}")
            return True
        else:
            logging.warning(f"Kaldırılacak model bulunamadı: {model_id}")
            return False
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Belirtilen modelin meta verilerini döndürür.
        
        Args:
            model_id: Meta verileri alınacak modelin benzersiz tanımlayıcısı.
            
        Returns:
            metadata: Model meta verileri.
            
        Raises:
            ModelNotFoundError: Model bulunamadığında.
        """
        if model_id in self.loaded_models:
            model_data = self.loaded_models[model_id]
            return {
                "model_id": model_id,
                "version": model_data["version"],
                "info": model_data["info"],
                "config": model_data["config"],
                "is_loaded": True
            }
        
        # Model yüklü değilse, bilgi dosyasından meta verileri oku
        model_dir = Path(self.model_cache_dir) / model_id
        info_file = model_dir / MODEL_INFO_FILE
        
        if not info_file.exists():
            raise ModelNotFoundError(f"Model bilgisi bulunamadı: {info_file}")
        
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ModelLoadError(f"Model bilgisi okunamadı: {e}")
        
        return {
            "model_id": model_id,
            "version": model_info.get("latest_version", "unknown"),
            "info": model_info,
            "config": None,
            "is_loaded": False
        }
    
    def get_model(self, model_id: str) -> torch.jit.ScriptModule:
        """
        Yüklenen model nesnesini döndürür.
        
        Args:
            model_id: Alınacak modelin benzersiz tanımlayıcısı.
            
        Returns:
            model: PyTorch TorchScript modeli.
            
        Raises:
            ModelNotFoundError: Model bulunamadığında veya yüklenmediğinde.
        """
        if model_id not in self.loaded_models:
            raise ModelNotFoundError(f"Model yüklenmedi: {model_id}")
        
        return self.loaded_models[model_id]["model"]
    
    # JNI köprü işlevleri
    
    def jni_list_models(self) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek model listeleme işlevi.
        
        Returns:
            result: İşlem sonucu ve model listesi.
        """
        result = safe_jni_call(self.list_available_models)
        
        # Başarılı durumda, Java Map formatına dönüştür
        if result["status"] == "success":
            models_java_format = []
            for model in result["data"]:
                model_java = convert_dict_to_java_map(model)
                models_java_format.append(model_java)
            
            result["data"] = models_java_format
        
        return result
    
    def jni_load_model(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek model yükleme işlevi.
        
        Args:
            model_id: Yüklenecek modelin benzersiz tanımlayıcısı.
            version: İsteğe bağlı model versiyonu.
            
        Returns:
            result: İşlem sonucu ve yüklenen model bilgisi.
        """
        result = safe_jni_call(self.load_model, model_id, version)
        
        # Başarılı durumda, Java Map formatına dönüştür
        if result["status"] == "success":
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result
    
    def jni_unload_model(self, model_id: str) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek model kaldırma işlevi.
        
        Args:
            model_id: Kaldırılacak modelin benzersiz tanımlayıcısı.
            
        Returns:
            result: İşlem sonucu.
        """
        return safe_jni_call(self.unload_model, model_id)
    
    def jni_get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek model meta verisi alma işlevi.
        
        Args:
            model_id: Meta verileri alınacak modelin benzersiz tanımlayıcısı.
            
        Returns:
            result: İşlem sonucu ve model meta verileri.
        """
        result = safe_jni_call(self.get_model_metadata, model_id)
        
        # Başarılı durumda, Java Map formatına dönüştür
        if result["status"] == "success":
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result 