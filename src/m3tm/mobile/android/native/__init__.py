"""
M³TM v2.3 - Android SDK Native Köprüsü

C++ ve JNI (Java Native Interface) modülleri için köprü işlevleri.
Bu modül, PyTorch C++ API'si ve Java/Kotlin API'si arasında iletişim sağlar.
"""

from .jni_helpers import (
    JNIError, 
    JNITypeError,
    JNIRuntimeError,
    JNIMethodSignature,
    convert_numpy_to_java,
    convert_tensor_to_java,
    convert_java_to_numpy,
    convert_java_to_tensor,
    convert_dict_to_java_map,
    convert_list_to_java_list,
    format_jni_exception,
    safe_jni_call,
    check_jni_environment
)

from .model_loader import (
    ModelLoader,
    ModelLoadError,
    ModelNotFoundError,
    ModelVersionError
)

from .inference_bridge import (
    InferenceBridge,
    InferenceError
)

from .training_bridge import (
    TrainingBridge,
    TrainingError,
    TrainingCallback,
    TrainingSessionExpiredError,
    TrainingSessionNotFoundError
)

__all__ = [
    # JNI Helpers
    "JNIError", 
    "JNITypeError",
    "JNIRuntimeError",
    "JNIMethodSignature",
    "convert_numpy_to_java",
    "convert_tensor_to_java",
    "convert_java_to_numpy",
    "convert_java_to_tensor",
    "convert_dict_to_java_map",
    "convert_list_to_java_list",
    "format_jni_exception",
    "safe_jni_call",
    "check_jni_environment",
    
    # Model Loader
    "ModelLoader",
    "ModelLoadError",
    "ModelNotFoundError",
    "ModelVersionError",
    
    # Inference Bridge
    "InferenceBridge",
    "InferenceError",
    
    # Training Bridge
    "TrainingBridge",
    "TrainingError",
    "TrainingCallback",
    "TrainingSessionExpiredError",
    "TrainingSessionNotFoundError"
] 