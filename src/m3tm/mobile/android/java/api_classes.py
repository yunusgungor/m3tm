"""
M³TM v2.3 Android SDK - Java API Sınıfları

Bu modül, Java API sınıflarını Python ile simüle eder.
Geliştirme ve test amacıyla kullanılır, gerçek cihazda JNI köprüsü
üzerinden Java sınıflarıyla değiştirilecektir.
"""

from typing import Dict, List, Optional, Any, Union, Callable
import logging
import json
import numpy as np
import torch
from pathlib import Path

from ..native import (
    ModelLoader, InferenceBridge, TrainingBridge,
    ModelNotFoundError, InferenceError, TrainingError,
    TrainingCallback as PyTrainingCallback
)

# Simüle edilmiş Java sınıfları (Python implementasyonu)

class M3TMException(Exception):
    """
    Java M3TMException sınıfının Python eşdeğeri.
    M³TM SDK'sıyla ilgili tüm istisnaların temel sınıfı.
    """
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause

class ModelException(M3TMException):
    """Model yükleme ve işleme sırasında oluşan hatalar için istisna sınıfı."""
    pass

class InferenceException(M3TMException):
    """Çıkarım işlemleri sırasında oluşan hatalar için istisna sınıfı."""
    pass

class TrainingException(M3TMException):
    """Eğitim işlemleri sırasında oluşan hatalar için istisna sınıfı."""
    pass

class TrainingCallback:
    """
    Java TrainingCallback arayüzünün Python eşdeğeri.
    Eğitim ilerlemesini izlemek için kullanılır.
    """
    def onBatchComplete(self, batch: int, metrics: Dict[str, Any]) -> None:
        """
        Her batch tamamlandığında çağrılır.
        
        Args:
            batch: Tamamlanan batch numarası.
            metrics: Batch metrikleri.
        """
        pass
    
    def onEpochComplete(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Her epoch tamamlandığında çağrılır.
        
        Args:
            epoch: Tamamlanan epoch numarası.
            metrics: Epoch metrikleri.
        """
        pass
    
    def onTrainingComplete(self, metrics: Dict[str, Any]) -> None:
        """
        Eğitim tamamlandığında çağrılır.
        
        Args:
            metrics: Final eğitim metrikleri.
        """
        pass

class TrainingCallbackAdapter(PyTrainingCallback):
    """
    Java TrainingCallback'i Python TrainingCallback'e dönüştüren adaptör sınıfı.
    """
    def __init__(self, java_callback: TrainingCallback):
        """
        TrainingCallbackAdapter sınıfını başlatır.
        
        Args:
            java_callback: Java stili geri çağrı nesnesi.
        """
        self.java_callback = java_callback
    
    def on_batch_complete(self, batch: int, metrics: Dict[str, Any]) -> None:
        """
        Her batch tamamlandığında çağrılır.
        
        Args:
            batch: Tamamlanan batch numarası.
            metrics: Batch metrikleri.
        """
        self.java_callback.onBatchComplete(batch, metrics)
    
    def on_epoch_complete(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Her epoch tamamlandığında çağrılır.
        
        Args:
            epoch: Tamamlanan epoch numarası.
            metrics: Epoch metrikleri.
        """
        self.java_callback.onEpochComplete(epoch, metrics)
    
    def on_training_complete(self, metrics: Dict[str, Any]) -> None:
        """
        Eğitim tamamlandığında çağrılır.
        
        Args:
            metrics: Final eğitim metrikleri.
        """
        self.java_callback.onTrainingComplete(metrics)

class ModelInfo:
    """
    Java ModelInfo sınıfının Python eşdeğeri.
    Model meta verilerini temsil eder.
    """
    def __init__(
        self,
        model_id: str,
        version: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        ModelInfo sınıfını başlatır.
        
        Args:
            model_id: Model benzersiz tanımlayıcısı.
            version: Model versiyonu.
            name: Model ismi (opsiyonel).
            description: Model açıklaması (opsiyonel).
            properties: Ek model özellikleri (opsiyonel).
        """
        self.model_id = model_id
        self.version = version
        self.name = name or model_id
        self.description = description or ""
        self.properties = properties or {}
    
    def getModelId(self) -> str:
        """Model ID'sini döndürür."""
        return self.model_id
    
    def getVersion(self) -> str:
        """Model versiyonunu döndürür."""
        return self.version
    
    def getName(self) -> str:
        """Model ismini döndürür."""
        return self.name
    
    def getDescription(self) -> str:
        """Model açıklamasını döndürür."""
        return self.description
    
    def getProperties(self) -> Dict[str, Any]:
        """Model özelliklerini döndürür."""
        return self.properties
    
    def toJson(self) -> str:
        """Model bilgisini JSON formatında döndürür."""
        info_dict = {
            "model_id": self.model_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "properties": self.properties
        }
        return json.dumps(info_dict)
    
    @classmethod
    def fromJson(cls, json_str: str) -> 'ModelInfo':
        """
        JSON formatından ModelInfo oluşturur.
        
        Args:
            json_str: JSON formatında model bilgisi.
            
        Returns:
            model_info: Oluşturulan ModelInfo nesnesi.
        """
        try:
            info_dict = json.loads(json_str)
            return cls(
                model_id=info_dict["model_id"],
                version=info_dict["version"],
                name=info_dict.get("name"),
                description=info_dict.get("description"),
                properties=info_dict.get("properties", {})
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ModelException(f"ModelInfo JSON'dan oluşturulamadı: {str(e)}")

class M3TMModel:
    """
    Java M3TMModel sınıfının Python eşdeğeri.
    Yüklenen modeli temsil eder ve çıkarım işlevleri sağlar.
    """
    def __init__(self, model_id: str, model_loader: ModelLoader):
        """
        M3TMModel sınıfını başlatır.
        
        Args:
            model_id: Model benzersiz tanımlayıcısı.
            model_loader: ModelLoader örneği.
        """
        self.model_id = model_id
        self.model_loader = model_loader
        self.inference_bridge = InferenceBridge(model_loader)
        
        # Model meta verilerini al
        try:
            metadata = model_loader.get_model_metadata(model_id)
            self.info = ModelInfo(
                model_id=model_id,
                version=metadata["version"],
                name=metadata["info"].get("name"),
                description=metadata["info"].get("description"),
                properties=metadata["info"]
            )
        except Exception as e:
            raise ModelException(f"Model bilgisi alınamadı: {str(e)}")
    
    def getModelInfo(self) -> ModelInfo:
        """Model bilgisini döndürür."""
        return self.info
    
    def processText(
        self, 
        text: str, 
        task: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Metin verisi üzerinde model çıkarımı yapar.
        
        Args:
            text: İşlenecek metin.
            task: İsteğe bağlı görev adı.
            options: Çıkarım seçenekleri.
            
        Returns:
            results: Çıkarım sonuçları.
            
        Raises:
            InferenceException: Çıkarım sırasında hata oluştuğunda.
        """
        try:
            result = self.inference_bridge.run_text_inference(
                self.model_id, text, task, options
            )
            return result
        except Exception as e:
            raise InferenceException(f"Metin işleme hatası: {str(e)}", e)
    
    def processImage(
        self,
        image_data: np.ndarray,
        task: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Görüntü verisi üzerinde model çıkarımı yapar.
        
        Args:
            image_data: İşlenecek görüntü (NumPy dizisi).
            task: İsteğe bağlı görev adı.
            options: Çıkarım seçenekleri.
            
        Returns:
            results: Çıkarım sonuçları.
            
        Raises:
            InferenceException: Çıkarım sırasında hata oluştuğunda.
        """
        try:
            # NumPy dizisini PyTorch tensörüne dönüştür
            image_tensor = torch.from_numpy(image_data)
            
            result = self.inference_bridge.run_image_inference(
                self.model_id, image_tensor, task, options
            )
            return result
        except Exception as e:
            raise InferenceException(f"Görüntü işleme hatası: {str(e)}", e)
    
    def processMultimodal(
        self,
        text: Optional[str] = None,
        image_data: Optional[np.ndarray] = None,
        task: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Metin ve görüntü verilerini birleştirerek çoklu-modalite çıkarımı yapar.
        
        Args:
            text: İşlenecek metin (opsiyonel).
            image_data: İşlenecek görüntü (NumPy dizisi, opsiyonel).
            task: İsteğe bağlı görev adı.
            options: Çıkarım seçenekleri.
            
        Returns:
            results: Çıkarım sonuçları.
            
        Raises:
            InferenceException: Çıkarım sırasında hata oluştuğunda.
        """
        try:
            # Görüntü verisini tensöre dönüştür (varsa)
            image_tensor = None
            if image_data is not None:
                image_tensor = torch.from_numpy(image_data)
            
            result = self.inference_bridge.run_multimodal_inference(
                self.model_id, text, image_tensor, task, options
            )
            return result
        except Exception as e:
            raise InferenceException(f"Çoklu-modalite işleme hatası: {str(e)}", e)
    
    def close(self) -> None:
        """
        Modeli kapatır ve kaynakları serbest bırakır.
        """
        try:
            self.model_loader.unload_model(self.model_id)
        except Exception as e:
            logging.warning(f"Model kapatılırken hata: {str(e)}")

class M3TMModelManager:
    """
    Java M3TMModelManager sınıfının Python eşdeğeri.
    Model yükleme, listeleme ve yönetimi için işlevler sağlar.
    """
    def __init__(self, model_cache_dir: Optional[str] = None):
        """
        M3TMModelManager sınıfını başlatır.
        
        Args:
            model_cache_dir: Model önbellek dizini (opsiyonel).
        """
        self.model_loader = ModelLoader(model_cache_dir=model_cache_dir)
    
    def loadModel(self, model_id: str, version: Optional[str] = None) -> M3TMModel:
        """
        Belirtilen modeli yükler.
        
        Args:
            model_id: Model benzersiz tanımlayıcısı.
            version: İsteğe bağlı model versiyonu.
            
        Returns:
            model: Yüklenen model.
            
        Raises:
            ModelException: Model yüklenemediğinde.
        """
        try:
            self.model_loader.load_model(model_id, version)
            return M3TMModel(model_id, self.model_loader)
        except Exception as e:
            raise ModelException(f"Model yüklenemedi: {str(e)}", e)
    
    def listAvailableModels(self) -> List[ModelInfo]:
        """
        Kullanılabilir modellerin listesini döndürür.
        
        Returns:
            models: Kullanılabilir modellerin listesi.
        """
        try:
            models_data = self.model_loader.list_available_models()
            model_infos = []
            
            for model_data in models_data:
                model_info = ModelInfo(
                    model_id=model_data["model_id"],
                    version=model_data.get("latest_version", "unknown"),
                    name=model_data.get("name"),
                    description=model_data.get("description"),
                    properties=model_data
                )
                model_infos.append(model_info)
            
            return model_infos
        except Exception as e:
            logging.error(f"Model listesi alınamadı: {str(e)}")
            return []
    
    def getModelInfo(self, model_id: str) -> ModelInfo:
        """
        Belirtilen model hakkında bilgi alır.
        
        Args:
            model_id: Model benzersiz tanımlayıcısı.
            
        Returns:
            model_info: Model bilgisi.
            
        Raises:
            ModelException: Model bilgisi alınamadığında.
        """
        try:
            metadata = self.model_loader.get_model_metadata(model_id)
            return ModelInfo(
                model_id=model_id,
                version=metadata["version"],
                name=metadata["info"].get("name"),
                description=metadata["info"].get("description"),
                properties=metadata["info"]
            )
        except Exception as e:
            raise ModelException(f"Model bilgisi alınamadı: {str(e)}", e)
    
    def isModelAvailable(self, model_id: str) -> bool:
        """
        Belirtilen modelin mevcut olup olmadığını kontrol eder.
        
        Args:
            model_id: Model benzersiz tanımlayıcısı.
            
        Returns:
            available: Modelin mevcut olup olmadığı.
        """
        try:
            self.model_loader.get_model_metadata(model_id)
            return True
        except:
            return False

class M3TMTrainingManager:
    """
    Java M3TMTrainingManager sınıfının Python eşdeğeri.
    Model eğitimi ve ince ayar işlemleri için işlevler sağlar.
    """
    def __init__(self, model_cache_dir: Optional[str] = None):
        """
        M3TMTrainingManager sınıfını başlatır.
        
        Args:
            model_cache_dir: Model önbellek dizini (opsiyonel).
        """
        self.model_loader = ModelLoader(model_cache_dir=model_cache_dir)
        self.training_bridge = TrainingBridge(self.model_loader)
    
    def createTrainingSession(
        self,
        model_id: str,
        training_config: Dict[str, Any]
    ) -> str:
        """
        Eğitim oturumu oluşturur.
        
        Args:
            model_id: Eğitilecek modelin benzersiz tanımlayıcısı.
            training_config: Eğitim yapılandırması.
            
        Returns:
            session_id: Oluşturulan oturum kimliği.
            
        Raises:
            TrainingException: Oturum oluşturulamadığında.
        """
        try:
            result = self.training_bridge.create_training_session(model_id, training_config)
            return result["session_id"]
        except Exception as e:
            raise TrainingException(f"Eğitim oturumu oluşturulamadı: {str(e)}", e)
    
    def startTraining(
        self,
        session_id: str,
        training_data: Dict[str, Any],
        callback: Optional[TrainingCallback] = None
    ) -> Dict[str, Any]:
        """
        Eğitim oturumunu başlatır.
        
        Args:
            session_id: Eğitim oturumu kimliği.
            training_data: Eğitim veri setleri.
            callback: Eğitim ilerlemesini bildirmek için isteğe bağlı callback.
            
        Returns:
            results: Eğitim sonuçları.
            
        Raises:
            TrainingException: Eğitim sırasında hata oluştuğunda.
        """
        try:
            # Java callback'i Python callback'e dönüştür
            py_callback = None
            if callback:
                py_callback = TrainingCallbackAdapter(callback)
            
            result = self.training_bridge.start_training(session_id, training_data, py_callback)
            return result
        except Exception as e:
            raise TrainingException(f"Eğitim başlatılamadı: {str(e)}", e)
    
    def getTrainingStatus(self, session_id: str) -> Dict[str, Any]:
        """
        Eğitim oturumunun durumunu alır.
        
        Args:
            session_id: Eğitim oturumu kimliği.
            
        Returns:
            status: Oturum durumu.
            
        Raises:
            TrainingException: Durum alınamadığında.
        """
        try:
            return self.training_bridge.get_training_status(session_id)
        except Exception as e:
            raise TrainingException(f"Eğitim durumu alınamadı: {str(e)}", e)
    
    def saveTrainedModel(
        self,
        session_id: str,
        output_path: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Eğitilmiş modeli kaydeder.
        
        Args:
            session_id: Eğitim oturumu kimliği.
            output_path: Kaydedilecek dizin (opsiyonel).
            model_name: Kaydedilecek model adı (opsiyonel).
            
        Returns:
            save_info: Kaydetme işlemi bilgisi.
            
        Raises:
            TrainingException: Model kaydedilemediğinde.
        """
        try:
            return self.training_bridge.save_trained_model(session_id, output_path, model_name)
        except Exception as e:
            raise TrainingException(f"Eğitilmiş model kaydedilemedi: {str(e)}", e) 