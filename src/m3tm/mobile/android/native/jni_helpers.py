"""
JNI yardımcı işlevleri modülü.

Bu modül, JNI (Java Native Interface) ile çalışmayı kolaylaştıran
yardımcı fonksiyonlar ve yardımcı sınıflar içerir.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import json
import os
import numpy as np
import torch

# JNI iletişiminde veri türü dönüşümleri için sabitler
JAVA_FLOAT_ARRAY = "float[]"
JAVA_INT_ARRAY = "int[]"
JAVA_STRING = "java.lang.String"
JAVA_BOOLEAN = "boolean"
JAVA_OBJECT = "java.lang.Object"
JAVA_MAP = "java.util.Map"
JAVA_LIST = "java.util.List"

# JNI hata türleri
class JNIError(Exception):
    """JNI işlemleri sırasında oluşan hatalar için temel istisna sınıfı."""
    pass

class JNITypeError(JNIError):
    """JNI veri türü dönüşümlerinde oluşan hatalar için istisna sınıfı."""
    pass

class JNIRuntimeError(JNIError):
    """JNI çalışma zamanı hatalar için istisna sınıfı."""
    pass

@dataclass
class JNIMethodSignature:
    """JNI metot imzasını tanımlayan sınıf."""
    method_name: str
    return_type: str
    parameter_types: List[str]
    
    def get_jni_signature(self) -> str:
        """JNI için metot imzasını döndürür."""
        params = "".join([_get_jni_type_signature(t) for t in self.parameter_types])
        return f"({params}){_get_jni_type_signature(self.return_type)}"

def _get_jni_type_signature(java_type: str) -> str:
    """Java türünden JNI tür imzasına dönüşüm."""
    type_map = {
        "int": "I",
        "float": "F",
        "boolean": "Z",
        "void": "V",
        "java.lang.String": "Ljava/lang/String;",
        "float[]": "[F",
        "int[]": "[I",
        "boolean[]": "[Z",
        "java.util.Map": "Ljava/util/Map;",
        "java.util.List": "Ljava/util/List;"
    }
    
    if java_type in type_map:
        return type_map[java_type]
    elif java_type.endswith("[]"):
        base_type = java_type[:-2]
        return f"[{_get_jni_type_signature(base_type)}"
    elif "." in java_type:
        return f"L{java_type.replace('.', '/')};"
    else:
        return java_type  # Bilinmeyen türler için doğrudan geçir
        
def convert_numpy_to_java(array: np.ndarray) -> Dict[str, Any]:
    """NumPy dizisini JNI üzerinden Java'ya aktarmak için dönüştürür."""
    if not isinstance(array, np.ndarray):
        raise JNITypeError(f"Girdi NumPy dizisi değil: {type(array)}")
    
    java_type = None
    if array.dtype == np.float32:
        java_type = JAVA_FLOAT_ARRAY
    elif array.dtype == np.int32:
        java_type = JAVA_INT_ARRAY
    else:
        # Otomatik dönüşüm dene
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
            java_type = JAVA_FLOAT_ARRAY
        elif np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.int32)
            java_type = JAVA_INT_ARRAY
        else:
            raise JNITypeError(f"Desteklenmeyen NumPy dizisi türü: {array.dtype}")
    
    # Java için düz bir dizi haline getir
    flattened_array = array.flatten().tolist()
    
    return {
        "java_type": java_type,
        "data": flattened_array,
        "shape": array.shape
    }

def convert_tensor_to_java(tensor: torch.Tensor) -> Dict[str, Any]:
    """PyTorch tensörünü JNI üzerinden Java'ya aktarmak için dönüştürür."""
    if not isinstance(tensor, torch.Tensor):
        raise JNITypeError(f"Girdi PyTorch tensörü değil: {type(tensor)}")
    
    # CPU'ya taşı ve NumPy'a çevir
    tensor_cpu = tensor.detach().cpu()
    return convert_numpy_to_java(tensor_cpu.numpy())

def convert_java_to_numpy(java_data: Dict[str, Any]) -> np.ndarray:
    """Java'dan gelen veriyi NumPy dizisine dönüştürür."""
    if not isinstance(java_data, dict) or "java_type" not in java_data or "data" not in java_data:
        raise JNITypeError(f"Geçersiz Java veri formatı: {java_data}")
    
    java_type = java_data["java_type"]
    data = java_data["data"]
    shape = java_data.get("shape", (len(data),))
    
    np_type = None
    if java_type == JAVA_FLOAT_ARRAY:
        np_type = np.float32
    elif java_type == JAVA_INT_ARRAY:
        np_type = np.int32
    else:
        raise JNITypeError(f"Desteklenmeyen Java dizisi türü: {java_type}")
    
    return np.array(data, dtype=np_type).reshape(shape)

def convert_java_to_tensor(java_data: Dict[str, Any]) -> torch.Tensor:
    """Java'dan gelen veriyi PyTorch tensörüne dönüştürür."""
    numpy_array = convert_java_to_numpy(java_data)
    return torch.from_numpy(numpy_array)

def convert_dict_to_java_map(dict_data: Dict[str, Any]) -> Dict[str, Any]:
    """Python sözlüğünü JNI üzerinden Java Map'ine dönüştürür."""
    result = {"java_type": JAVA_MAP, "entries": []}
    
    for key, value in dict_data.items():
        java_key = {"java_type": JAVA_STRING, "value": str(key)}
        
        # Değer türüne göre dönüşüm
        if isinstance(value, np.ndarray):
            java_value = convert_numpy_to_java(value)
        elif isinstance(value, torch.Tensor):
            java_value = convert_tensor_to_java(value)
        elif isinstance(value, dict):
            java_value = convert_dict_to_java_map(value)
        elif isinstance(value, list):
            java_value = convert_list_to_java_list(value)
        elif isinstance(value, (int, float, bool, str)):
            java_value = {"java_type": _get_python_to_java_type(value), "value": value}
        else:
            # Karmaşık nesneler için JSON serileştirme kullan
            java_value = {"java_type": JAVA_STRING, "value": json.dumps(str(value))}
        
        result["entries"].append({"key": java_key, "value": java_value})
    
    return result

def convert_list_to_java_list(list_data: List[Any]) -> Dict[str, Any]:
    """Python listesini JNI üzerinden Java List'ine dönüştürür."""
    result = {"java_type": JAVA_LIST, "items": []}
    
    for item in list_data:
        if isinstance(item, np.ndarray):
            java_item = convert_numpy_to_java(item)
        elif isinstance(item, torch.Tensor):
            java_item = convert_tensor_to_java(item)
        elif isinstance(item, dict):
            java_item = convert_dict_to_java_map(item)
        elif isinstance(item, list):
            java_item = convert_list_to_java_list(item)
        elif isinstance(item, (int, float, bool, str)):
            java_item = {"java_type": _get_python_to_java_type(item), "value": item}
        else:
            # Karmaşık nesneler için JSON serileştirme kullan
            java_item = {"java_type": JAVA_STRING, "value": json.dumps(str(item))}
        
        result["items"].append(java_item)
    
    return result

def _get_python_to_java_type(value: Any) -> str:
    """Python değerine karşılık gelen Java türünü döndürür."""
    if isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, str):
        return JAVA_STRING
    else:
        return JAVA_OBJECT

def format_jni_exception(e: Exception) -> Dict[str, Any]:
    """JNI istisnasını Java tarafına aktarmak için uygun formata dönüştürür."""
    return {
        "exception_type": e.__class__.__name__,
        "message": str(e),
        "stack_trace": "".join(logging.traceback.format_exception(None, e, e.__traceback__))
    }

def safe_jni_call(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """JNI çağrıları için güvenli bir sarmalayıcı sağlar."""
    try:
        result = func(*args, **kwargs)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        logging.exception(f"JNI çağrısı sırasında hata: {func.__name__}")
        return {
            "status": "error",
            "error": format_jni_exception(e)
        }

def check_jni_environment() -> Dict[str, Any]:
    """JNI ortam bilgilerini kontrol eder ve raporlar."""
    env_info = {
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "jni_available": False,
        "torch_jit_available": hasattr(torch, "jit") and hasattr(torch.jit, "load")
    }
    
    # JNI kullanılabilirliğini kontrol et (simüle ediliyor)
    env_info["jni_available"] = True
    
    return env_info 