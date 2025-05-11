/**
 * M³TM v2.3 - Android SDK JNI Köprüsü
 * 
 * Java Native Interface köprüsü için C++ implementasyon dosyası.
 */

#include "m3tm_jni_bridge.h"
#include <iostream>
#include <stdexcept>

namespace m3tm {
namespace jni {

// Singleton instance
M3TMJniBridge& M3TMJniBridge::getInstance() {
    static M3TMJniBridge instance;
    return instance;
}

M3TMJniBridge::M3TMJniBridge() 
    : m_pythonModule(nullptr), m_javaVM(nullptr) {
}

M3TMJniBridge::~M3TMJniBridge() {
    shutdown();
}

bool M3TMJniBridge::initialize(JNIEnv* env, JavaVM* javaVM) {
    if (m_javaVM != nullptr) {
        // Zaten başlatılmış
        return true;
    }
    
    m_javaVM = javaVM;
    
    // Python'u başlat
    return initializePython();
}

void M3TMJniBridge::shutdown() {
    cleanupPython();
    m_javaVM = nullptr;
}

bool M3TMJniBridge::loadPythonModule(const std::string& modulePath) {
    if (!isPythonModuleLoaded()) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        // Python sys.path'e modül yolunu ekle
        PyObject* sysPath = PySys_GetObject("path");
        PyObject* moduleDir = PyUnicode_FromString(modulePath.c_str());
        PyList_Append(sysPath, moduleDir);
        Py_DECREF(moduleDir);
        
        // M3TM modülünü import et
        const char* moduleName = "m3tm.mobile.android";
        m_pythonModule = PyImport_ImportModule(moduleName);
        
        if (m_pythonModule == nullptr) {
            PyErr_Print();
            PyGILState_Release(gstate);
            return false;
        }
        
        PyGILState_Release(gstate);
        return true;
    }
    
    return true;
}

JNIEnv* M3TMJniBridge::getEnv() {
    JNIEnv* env;
    if (m_javaVM == nullptr) {
        return nullptr;
    }
    
    jint result = m_javaVM->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (result == JNI_EDETACHED) {
        // Thread JNI ortamına bağlı değil, bağla
        result = m_javaVM->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr);
        if (result != JNI_OK) {
            return nullptr;
        }
    } else if (result != JNI_OK) {
        return nullptr;
    }
    
    return env;
}

JavaVM* M3TMJniBridge::getJavaVM() {
    return m_javaVM;
}

bool M3TMJniBridge::isPythonModuleLoaded() const {
    return m_pythonModule != nullptr;
}

PyObject* M3TMJniBridge::getPythonModule() const {
    return m_pythonModule;
}

jobject M3TMJniBridge::callPythonMethod(const std::string& methodName, const std::vector<jobject>& args) {
    if (!isPythonModuleLoaded()) {
        throwJavaException("com/m3tm/sdk/M3TMException", "Python modülü yüklenmedi");
        return nullptr;
    }
    
    JNIEnv* env = getEnv();
    if (env == nullptr) {
        throwJavaException("com/m3tm/sdk/M3TMException", "JNI çevresi alınamadı");
        return nullptr;
    }
    
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    // Python fonksiyonunu al
    PyObject* pyFunc = PyObject_GetAttrString(m_pythonModule, methodName.c_str());
    if (pyFunc == nullptr) {
        PyErr_Print();
        PyGILState_Release(gstate);
        throwJavaException("com/m3tm/sdk/M3TMException", "Python fonksiyonu bulunamadı: " + methodName);
        return nullptr;
    }
    
    // Argümanları Python'a dönüştür
    PyObject* pyArgs = PyTuple_New(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        PyObject* pyArg = javaToPython(env, args[i]);
        if (pyArg == nullptr) {
            Py_DECREF(pyFunc);
            Py_DECREF(pyArgs);
            PyGILState_Release(gstate);
            throwJavaException("com/m3tm/sdk/M3TMException", "Argüman dönüştürülemedi: " + std::to_string(i));
            return nullptr;
        }
        PyTuple_SetItem(pyArgs, i, pyArg);
    }
    
    // Python fonksiyonunu çağır
    PyObject* pyResult = PyObject_CallObject(pyFunc, pyArgs);
    Py_DECREF(pyFunc);
    Py_DECREF(pyArgs);
    
    if (pyResult == nullptr) {
        PyErr_Print();
        PyGILState_Release(gstate);
        throwJavaException("com/m3tm/sdk/M3TMException", "Python fonksiyonu çağrısı başarısız: " + methodName);
        return nullptr;
    }
    
    // Sonucu Java'ya dönüştür
    jobject jResult = pythonToJava(env, pyResult);
    Py_DECREF(pyResult);
    
    PyGILState_Release(gstate);
    return jResult;
}

void M3TMJniBridge::throwJavaException(const std::string& exceptionClass, const std::string& message) {
    JNIEnv* env = getEnv();
    if (env == nullptr) {
        return;
    }
    
    jclass excClass = env->FindClass(exceptionClass.c_str());
    if (excClass == nullptr) {
        return;
    }
    
    env->ThrowNew(excClass, message.c_str());
}

bool M3TMJniBridge::initializePython() {
    if (Py_IsInitialized()) {
        return true;
    }
    
    Py_Initialize();
    if (!Py_IsInitialized()) {
        return false;
    }
    
    PyEval_InitThreads();
    m_gilState = PyGILState_Ensure();
    
    return true;
}

void M3TMJniBridge::cleanupPython() {
    if (m_pythonModule != nullptr) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_DECREF(m_pythonModule);
        m_pythonModule = nullptr;
        PyGILState_Release(gstate);
    }
    
    // Python interpreter'ı kapat (uygulama kapatılırken)
    if (Py_IsInitialized()) {
        PyGILState_Release(m_gilState);
        Py_Finalize();
    }
}

jobject M3TMJniBridge::pythonToJava(JNIEnv* env, PyObject* pyObj) {
    // String dönüşümü
    if (PyUnicode_Check(pyObj)) {
        const char* str = PyUnicode_AsUTF8(pyObj);
        jstring jStr = env->NewStringUTF(str);
        return jStr;
    }
    // Integer dönüşümü
    else if (PyLong_Check(pyObj)) {
        jlong value = PyLong_AsLong(pyObj);
        jclass cls = env->FindClass("java/lang/Long");
        jmethodID constructor = env->GetMethodID(cls, "<init>", "(J)V");
        return env->NewObject(cls, constructor, value);
    }
    // Float dönüşümü
    else if (PyFloat_Check(pyObj)) {
        jdouble value = PyFloat_AsDouble(pyObj);
        jclass cls = env->FindClass("java/lang/Double");
        jmethodID constructor = env->GetMethodID(cls, "<init>", "(D)V");
        return env->NewObject(cls, constructor, value);
    }
    // Boolean dönüşümü
    else if (PyBool_Check(pyObj)) {
        jboolean value = (pyObj == Py_True) ? JNI_TRUE : JNI_FALSE;
        jclass cls = env->FindClass("java/lang/Boolean");
        jmethodID constructor = env->GetMethodID(cls, "<init>", "(Z)V");
        return env->NewObject(cls, constructor, value);
    }
    // None dönüşümü
    else if (pyObj == Py_None) {
        return nullptr;
    }
    // Liste dönüşümü
    else if (PyList_Check(pyObj)) {
        jclass arrayListClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "()V");
        jmethodID arrayListAdd = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
        
        jobject arrayList = env->NewObject(arrayListClass, arrayListConstructor);
        Py_ssize_t size = PyList_Size(pyObj);
        
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item = PyList_GetItem(pyObj, i);
            jobject jItem = pythonToJava(env, item);
            env->CallBooleanMethod(arrayList, arrayListAdd, jItem);
            env->DeleteLocalRef(jItem);
        }
        
        return arrayList;
    }
    // Sözlük dönüşümü
    else if (PyDict_Check(pyObj)) {
        jclass hashMapClass = env->FindClass("java/util/HashMap");
        jmethodID hashMapConstructor = env->GetMethodID(hashMapClass, "<init>", "()V");
        jmethodID hashMapPut = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
        
        jobject hashMap = env->NewObject(hashMapClass, hashMapConstructor);
        PyObject* keys = PyDict_Keys(pyObj);
        Py_ssize_t size = PyList_Size(keys);
        
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* key = PyList_GetItem(keys, i);
            PyObject* value = PyDict_GetItem(pyObj, key);
            
            jobject jKey = pythonToJava(env, key);
            jobject jValue = pythonToJava(env, value);
            
            env->CallObjectMethod(hashMap, hashMapPut, jKey, jValue);
            
            env->DeleteLocalRef(jKey);
            env->DeleteLocalRef(jValue);
        }
        
        Py_DECREF(keys);
        return hashMap;
    }
    
    // Desteklenmeyen tip
    return nullptr;
}

PyObject* M3TMJniBridge::javaToPython(JNIEnv* env, jobject jObj) {
    if (jObj == nullptr) {
        Py_RETURN_NONE;
    }
    
    jclass cls = env->GetObjectClass(jObj);
    
    // String dönüşümü
    jclass stringClass = env->FindClass("java/lang/String");
    if (env->IsInstanceOf(jObj, stringClass)) {
        const char* str = env->GetStringUTFChars((jstring)jObj, nullptr);
        PyObject* pyStr = PyUnicode_FromString(str);
        env->ReleaseStringUTFChars((jstring)jObj, str);
        return pyStr;
    }
    
    // Integer dönüşümü
    jclass integerClass = env->FindClass("java/lang/Integer");
    if (env->IsInstanceOf(jObj, integerClass)) {
        jmethodID intValue = env->GetMethodID(integerClass, "intValue", "()I");
        jint value = env->CallIntMethod(jObj, intValue);
        return PyLong_FromLong(value);
    }
    
    // Long dönüşümü
    jclass longClass = env->FindClass("java/lang/Long");
    if (env->IsInstanceOf(jObj, longClass)) {
        jmethodID longValue = env->GetMethodID(longClass, "longValue", "()J");
        jlong value = env->CallLongMethod(jObj, longValue);
        return PyLong_FromLong(value);
    }
    
    // Float dönüşümü
    jclass floatClass = env->FindClass("java/lang/Float");
    if (env->IsInstanceOf(jObj, floatClass)) {
        jmethodID floatValue = env->GetMethodID(floatClass, "floatValue", "()F");
        jfloat value = env->CallFloatMethod(jObj, floatValue);
        return PyFloat_FromDouble(value);
    }
    
    // Double dönüşümü
    jclass doubleClass = env->FindClass("java/lang/Double");
    if (env->IsInstanceOf(jObj, doubleClass)) {
        jmethodID doubleValue = env->GetMethodID(doubleClass, "doubleValue", "()D");
        jdouble value = env->CallDoubleMethod(jObj, doubleValue);
        return PyFloat_FromDouble(value);
    }
    
    // Boolean dönüşümü
    jclass booleanClass = env->FindClass("java/lang/Boolean");
    if (env->IsInstanceOf(jObj, booleanClass)) {
        jmethodID booleanValue = env->GetMethodID(booleanClass, "booleanValue", "()Z");
        jboolean value = env->CallBooleanMethod(jObj, booleanValue);
        return PyBool_FromLong(value);
    }
    
    // Liste dönüşümü
    jclass listClass = env->FindClass("java/util/List");
    if (env->IsInstanceOf(jObj, listClass)) {
        jmethodID size = env->GetMethodID(listClass, "size", "()I");
        jmethodID get = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");
        
        jint listSize = env->CallIntMethod(jObj, size);
        PyObject* pyList = PyList_New(listSize);
        
        for (jint i = 0; i < listSize; ++i) {
            jobject item = env->CallObjectMethod(jObj, get, i);
            PyObject* pyItem = javaToPython(env, item);
            PyList_SetItem(pyList, i, pyItem);
            env->DeleteLocalRef(item);
        }
        
        return pyList;
    }
    
    // Map dönüşümü
    jclass mapClass = env->FindClass("java/util/Map");
    if (env->IsInstanceOf(jObj, mapClass)) {
        jmethodID entrySet = env->GetMethodID(mapClass, "entrySet", "()Ljava/util/Set;");
        jclass setClass = env->FindClass("java/util/Set");
        jmethodID iterator = env->GetMethodID(setClass, "iterator", "()Ljava/util/Iterator;");
        jclass iteratorClass = env->FindClass("java/util/Iterator");
        jmethodID hasNext = env->GetMethodID(iteratorClass, "hasNext", "()Z");
        jmethodID next = env->GetMethodID(iteratorClass, "next", "()Ljava/lang/Object;");
        jclass entryClass = env->FindClass("java/util/Map$Entry");
        jmethodID getKey = env->GetMethodID(entryClass, "getKey", "()Ljava/lang/Object;");
        jmethodID getValue = env->GetMethodID(entryClass, "getValue", "()Ljava/lang/Object;");
        
        jobject entrySetObj = env->CallObjectMethod(jObj, entrySet);
        jobject iteratorObj = env->CallObjectMethod(entrySetObj, iterator);
        
        PyObject* pyDict = PyDict_New();
        
        while (env->CallBooleanMethod(iteratorObj, hasNext)) {
            jobject entryObj = env->CallObjectMethod(iteratorObj, next);
            jobject keyObj = env->CallObjectMethod(entryObj, getKey);
            jobject valueObj = env->CallObjectMethod(entryObj, getValue);
            
            PyObject* pyKey = javaToPython(env, keyObj);
            PyObject* pyValue = javaToPython(env, valueObj);
            
            PyDict_SetItem(pyDict, pyKey, pyValue);
            
            Py_DECREF(pyKey);
            Py_DECREF(pyValue);
            
            env->DeleteLocalRef(keyObj);
            env->DeleteLocalRef(valueObj);
            env->DeleteLocalRef(entryObj);
        }
        
        env->DeleteLocalRef(iteratorObj);
        env->DeleteLocalRef(entrySetObj);
        
        return pyDict;
    }
    
    // Desteklenmeyen tip
    Py_RETURN_NONE;
}

// JNI fonksiyon implementasyonları

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    
    if (!M3TMJniBridge::getInstance().initialize(env, vm)) {
        return JNI_ERR;
    }
    
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    M3TMJniBridge::getInstance().shutdown();
}

// ModelManager API implementasyonları

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModelManager_nativeLoadModel(JNIEnv* env, jobject thiz, jstring modelId, jstring version) {
    // Python modülünün yüklü olduğundan emin ol
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    if (!bridge.isPythonModuleLoaded()) {
        std::string modulePath = "/data/data/com.m3tm.sdk/lib/python";
        if (!bridge.loadPythonModule(modulePath)) {
            bridge.throwJavaException("com/m3tm/sdk/ModelException", "Python modülü yüklenemedi");
            return nullptr;
        }
    }
    
    // Java argümanlarını hazırla
    std::vector<jobject> args;
    args.push_back(modelId);
    args.push_back(version);
    
    // Python metodunu çağır (model_loader.jni_load_model)
    return bridge.callPythonMethod("model_loader.jni_load_model", args);
}

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModelManager_nativeListAvailableModels(JNIEnv* env, jobject thiz) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    if (!bridge.isPythonModuleLoaded()) {
        std::string modulePath = "/data/data/com.m3tm.sdk/lib/python";
        if (!bridge.loadPythonModule(modulePath)) {
            bridge.throwJavaException("com/m3tm/sdk/ModelException", "Python modülü yüklenemedi");
            return nullptr;
        }
    }
    
    std::vector<jobject> args;
    return bridge.callPythonMethod("model_loader.jni_list_models", args);
}

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModelManager_nativeGetModelInfo(JNIEnv* env, jobject thiz, jstring modelId) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    if (!bridge.isPythonModuleLoaded()) {
        std::string modulePath = "/data/data/com.m3tm.sdk/lib/python";
        if (!bridge.loadPythonModule(modulePath)) {
            bridge.throwJavaException("com/m3tm/sdk/ModelException", "Python modülü yüklenemedi");
            return nullptr;
        }
    }
    
    std::vector<jobject> args;
    args.push_back(modelId);
    
    return bridge.callPythonMethod("model_loader.jni_get_model_metadata", args);
}

JNIEXPORT jboolean JNICALL Java_com_m3tm_sdk_M3TMModelManager_nativeIsModelAvailable(JNIEnv* env, jobject thiz, jstring modelId) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    if (!bridge.isPythonModuleLoaded()) {
        std::string modulePath = "/data/data/com.m3tm.sdk/lib/python";
        if (!bridge.loadPythonModule(modulePath)) {
            return JNI_FALSE;
        }
    }
    
    std::vector<jobject> args;
    args.push_back(modelId);
    
    jobject result = bridge.callPythonMethod("model_loader.jni_get_model_metadata", args);
    if (result == nullptr) {
        return JNI_FALSE;
    }
    
    env->DeleteLocalRef(result);
    return JNI_TRUE;
}

// Model API implementasyonları

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModel_nativeProcessText(JNIEnv* env, jobject thiz, jstring modelId, jstring text, jstring task, jobject options) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(modelId);
    args.push_back(text);
    args.push_back(task);
    args.push_back(options);
    
    return bridge.callPythonMethod("inference_bridge.jni_run_text_inference", args);
}

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModel_nativeProcessImage(JNIEnv* env, jobject thiz, jstring modelId, jobject imageData, jstring task, jobject options) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(modelId);
    args.push_back(imageData);
    args.push_back(task);
    args.push_back(options);
    
    return bridge.callPythonMethod("inference_bridge.jni_run_image_inference", args);
}

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMModel_nativeProcessMultimodal(JNIEnv* env, jobject thiz, jstring modelId, jstring text, jobject imageData, jstring task, jobject options) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(modelId);
    args.push_back(text);
    args.push_back(imageData);
    args.push_back(task);
    args.push_back(options);
    
    return bridge.callPythonMethod("inference_bridge.jni_run_multimodal_inference", args);
}

JNIEXPORT jboolean JNICALL Java_com_m3tm_sdk_M3TMModel_nativeCloseModel(JNIEnv* env, jobject thiz, jstring modelId) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(modelId);
    
    jobject result = bridge.callPythonMethod("model_loader.jni_unload_model", args);
    if (result == nullptr) {
        return JNI_FALSE;
    }
    
    // Sonuç başarılı mı kontrol et
    jclass mapClass = env->FindClass("java/util/Map");
    jmethodID get = env->GetMethodID(mapClass, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
    
    jstring statusKey = env->NewStringUTF("status");
    jobject statusObj = env->CallObjectMethod(result, get, statusKey);
    
    jclass stringClass = env->FindClass("java/lang/String");
    const char* status = env->GetStringUTFChars((jstring)statusObj, nullptr);
    
    bool success = std::string(status) == "success";
    
    env->ReleaseStringUTFChars((jstring)statusObj, status);
    env->DeleteLocalRef(statusKey);
    env->DeleteLocalRef(statusObj);
    env->DeleteLocalRef(result);
    
    return success ? JNI_TRUE : JNI_FALSE;
}

// TrainingManager API implementasyonları

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeCreateTrainingSession(JNIEnv* env, jobject thiz, jstring modelId, jobject trainingConfig) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(modelId);
    args.push_back(trainingConfig);
    
    return bridge.callPythonMethod("training_bridge.jni_create_training_session", args);
}

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeStartTraining(JNIEnv* env, jobject thiz, jstring sessionId, jobject trainingData, jobject callback) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(sessionId);
    args.push_back(trainingData);
    args.push_back(callback);
    
    return bridge.callPythonMethod("training_bridge.jni_start_training", args);
}

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeGetTrainingStatus(JNIEnv* env, jobject thiz, jstring sessionId) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(sessionId);
    
    return bridge.callPythonMethod("training_bridge.jni_get_training_status", args);
}

JNIEXPORT jobject JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeSaveTrainedModel(JNIEnv* env, jobject thiz, jstring sessionId, jstring outputPath, jstring modelName) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(sessionId);
    args.push_back(outputPath);
    args.push_back(modelName);
    
    return bridge.callPythonMethod("training_bridge.jni_save_trained_model", args);
}

JNIEXPORT jboolean JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeDeleteTrainingSession(JNIEnv* env, jobject thiz, jstring sessionId) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    args.push_back(sessionId);
    
    jobject result = bridge.callPythonMethod("training_bridge.jni_delete_training_session", args);
    if (result == nullptr) {
        return JNI_FALSE;
    }
    
    // Sonuç başarılı mı kontrol et
    jclass mapClass = env->FindClass("java/util/Map");
    jmethodID get = env->GetMethodID(mapClass, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
    
    jstring statusKey = env->NewStringUTF("status");
    jobject statusObj = env->CallObjectMethod(result, get, statusKey);
    
    jclass stringClass = env->FindClass("java/lang/String");
    const char* status = env->GetStringUTFChars((jstring)statusObj, nullptr);
    
    bool success = std::string(status) == "success";
    
    env->ReleaseStringUTFChars((jstring)statusObj, status);
    env->DeleteLocalRef(statusKey);
    env->DeleteLocalRef(statusObj);
    env->DeleteLocalRef(result);
    
    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL Java_com_m3tm_sdk_M3TMTrainingManager_nativeCleanExpiredSessions(JNIEnv* env, jobject thiz) {
    M3TMJniBridge& bridge = M3TMJniBridge::getInstance();
    
    std::vector<jobject> args;
    
    jobject result = bridge.callPythonMethod("training_bridge.jni_clean_expired_sessions", args);
    if (result == nullptr) {
        return 0;
    }
    
    // Sonuç başarılı mı kontrol et
    jclass mapClass = env->FindClass("java/util/Map");
    jmethodID get = env->GetMethodID(mapClass, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
    
    jstring statusKey = env->NewStringUTF("status");
    jobject statusObj = env->CallObjectMethod(result, get, statusKey);
    
    jclass stringClass = env->FindClass("java/lang/String");
    const char* status = env->GetStringUTFChars((jstring)statusObj, nullptr);
    
    bool success = std::string(status) == "success";
    
    if (!success) {
        env->ReleaseStringUTFChars((jstring)statusObj, status);
        env->DeleteLocalRef(statusKey);
        env->DeleteLocalRef(statusObj);
        env->DeleteLocalRef(result);
        return 0;
    }
    
    // Temizlenen oturum sayısını al
    jstring dataKey = env->NewStringUTF("data");
    jobject dataObj = env->CallObjectMethod(result, get, dataKey);
    
    jclass integerClass = env->FindClass("java/lang/Integer");
    jmethodID intValue = env->GetMethodID(integerClass, "intValue", "()I");
    
    jint cleanedCount = env->CallIntMethod(dataObj, intValue);
    
    env->ReleaseStringUTFChars((jstring)statusObj, status);
    env->DeleteLocalRef(statusKey);
    env->DeleteLocalRef(dataKey);
    env->DeleteLocalRef(statusObj);
    env->DeleteLocalRef(dataObj);
    env->DeleteLocalRef(result);
    
    return cleanedCount;
}

} // extern "C"

} // namespace jni
} // namespace m3tm 