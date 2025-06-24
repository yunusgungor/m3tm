# JNI (Java Native Interface) Android Development Documentation

## Overview
JNI enables Java code running in the Android Runtime (ART) to call and be called by native applications and libraries written in C/C++. This documentation covers JNI patterns found in AndroidX and best practices for mobile development.

## JNI Project Setup with CMake

### Basic JNI CMake Configuration
```cmake
# Minimum CMake version for Android projects
cmake_minimum_required(VERSION 3.22.1)

project("your_native_project")

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

# Define shared library for JNI
add_library(your_jni_lib SHARED
    # List C/C++ source files
    jni_wrapper.cpp
    native_implementation.cpp
)
```

### Finding Android NDK Libraries
```cmake
# Find required Android system libraries
find_library(android-lib android)
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)
find_library(opengl-lib GLESv2)
find_library(egl-lib EGL)
```

### Linking Android Libraries
```cmake
# Link Android system libraries
target_link_libraries(your_jni_lib 
    ${android-lib} 
    ${log-lib}
    ${jnigraphics-lib}
)

# Alternative linking patterns
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC log)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE dl)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC android)
```

### Advanced JNI Library Configuration
```cmake
# Complete JNI library setup example
cmake_minimum_required(VERSION 3.22.1)

project(camera_test_app_jni)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -DEGL_EGLEXT_PROTOTYPES")

add_library(
  opengl_renderer_jni SHARED
  jni_hooks.cpp
  opengl_renderer_jni.cpp
)

find_library(log-lib log)
find_library(android-lib android)
find_library(opengl-lib GLESv2)
find_library(egl-lib EGL)

target_link_libraries(opengl_renderer_jni 
    ${log-lib} 
    ${android-lib} 
    ${opengl-lib} 
    ${egl-lib}
)

# Optimize memory page alignment
target_link_options(
  opengl_renderer_jni
  PRIVATE
  "-Wl,-z,max-page-size=16384"
)
```

## JVMTI Integration

### JVMTI Header Configuration
```cmake
# Copy JVMTI header from JDK (NDK doesn't include it)
configure_file($ENV{JAVA_HOME}/include/jvmti.h external_jvmti/jvmti.h COPYONLY)

# Add JVMTI include directory
target_include_directories(your_jni_lib
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/external_jvmti)
```

### JNI with JVMTI Example
```cmake
add_library(
        compose_inspection_jni SHARED
        lambda_location_java_jni.cpp
)

# Configure JVMTI header
configure_file($ENV{JAVA_HOME}/include/jvmti.h external_jvmti/jvmti.h COPYONLY)

target_include_directories(compose_inspection_jni
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/external_jvmti)

target_link_libraries(compose_inspection_jni ${android-lib} ${log-lib})
target_link_options(compose_inspection_jni PRIVATE "-Wl,-z,max-page-size=16384")
```

## JNI Testing Integration

### GTest with JNI Configuration
```cmake
cmake_minimum_required(VERSION 3.22.1)

project(junit-gtest LANGUAGES CXX)

find_package(googletest REQUIRED CONFIG)

add_library(nativehelper INTERFACE)
target_include_directories(nativehelper INTERFACE ./)

add_library(junit-gtest STATIC gtest_wrapper.cpp)

target_link_libraries(junit-gtest
        PRIVATE
        googletest::gtest
        nativehelper
        PUBLIC
        -uJava_androidx_test_ext_junitgtest_GtestRunner_initialize
        -uJava_androidx_test_ext_junitgtest_GtestRunner_run
        -uJava_androidx_test_ext_junitgtest_GtestRunner_addTest
)
```

## Android Graphics Integration

### Surface and Graphics JNI
```cmake
# Camera testing surface JNI
add_library(
        testing_surface_jni SHARED
        jni_hooks.cpp
        surface_jni.cpp
)

find_library(android-lib android)
target_link_libraries(testing_surface_jni ${android-lib})
target_link_options(
        testing_surface_jni
        PRIVATE
        "-Wl,-z,max-page-size=16384"
)
```

### Image Processing JNI
```cmake
# Image processing with libyuv
add_library(
        image_processing_util_jni
        SHARED
        image_processing_util_jni.cc
)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)
find_library(android-lib android)
find_package(libyuv REQUIRED)

target_link_libraries(image_processing_util_jni 
    PRIVATE 
    ${log-lib} 
    ${android-lib} 
    ${jnigraphics-lib} 
    libyuv::yuv
)
```

## Multiplatform Native Integration

### AndroidX Multiplatform Native Compilation
```kotlin
// API for creating native compilation
AndroidXMultiplatformExtension.createNativeCompilation(
  // parameters for sources, includes, dependencies, Konan targets
): MultiTargetNativeCompilation

// Bundle native libraries
AndroidXMultiplatformExtension.addNativeLibrariesToJniLibs(
  compilation: MultiTargetNativeCompilation
)

AndroidXMultiplatformExtension.addNativeLibrariesToResources(
  compilation: MultiTargetNativeCompilation
)

// Cinterop integration
AndroidXMultiplatformExtension.createCinterop(
  compilation: MultiTargetNativeCompilation
)
```

## JNI Interface Patterns from AndroidX

### Native Canvas Integration
```kotlin
// AndroidCanvas JNI patterns
public final class AndroidCanvas_androidKt {
  method public static androidx.compose.ui.graphics.Canvas Canvas(android.graphics.Canvas c);
  method public static android.graphics.Canvas getNativeCanvas(androidx.compose.ui.graphics.Canvas);
}
```

### Native Paint Integration
```kotlin
public final class AndroidPaint implements androidx.compose.ui.graphics.Paint {
    ctor public AndroidPaint();
    ctor public AndroidPaint(android.graphics.Paint internalPaint);
    method public android.graphics.Paint asFrameworkPaint();
    // ... other native interop methods
}
```

### Native Path Integration
```kotlin
public final class AndroidPath implements androidx.compose.ui.graphics.Path {
  ctor public AndroidPath(optional android.graphics.Path internalPath);
  method public android.graphics.Path getInternalPath();
  // ... other native path operations
}
```

### EGL Handle Interface
```kotlin
public interface EGLHandle {
  method public long getNativeHandle();
  property public abstract long nativeHandle;
}
```

## JNI Method Naming Patterns

### Standard JNI Naming Convention
```cpp
// Pattern: Java_package_Class_method
JNIEXPORT return_type JNICALL
Java_com_example_MyClass_nativeMethod(JNIEnv *env, jobject thiz, ...)

// AndroidX examples with symbol exposure
-uJava_androidx_test_ext_junitgtest_GtestRunner_initialize
-uJava_androidx_test_ext_junitgtest_GtestRunner_run
-uJava_androidx_test_ext_junitgtest_GtestRunner_addTest
```

## Development Environment Setup

### Required NDK and CMake Installation
```bash
# Install specific NDK and CMake versions
sdkmanager --install "ndk;23.1.7779620"
sdkmanager --install "cmake;3.22.1"
```

### Gradle Integration for JVM Default Methods
```gradle
tasks.withType(KotlinCompile).configureEach {
    kotlinOptions {
        freeCompilerArgs += ["-Xjvm-default=all"]
    }
}
```

## Memory and Performance Optimization

### Linker Optimizations
```cmake
# Page size optimization for mobile
target_link_options(your_jni_lib
    PRIVATE
    "-Wl,-z,max-page-size=16384"
)
```

### Library Output Configuration
```cmake
# Organize output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
```

## Android-Specific JNI Patterns

### Activity Integration
```kotlin
@androidx.compose.ui.test.ExperimentalTestApi 
public sealed interface AndroidComposeUiTest<A extends androidx.activity.ComponentActivity> 
    extends androidx.compose.ui.test.ComposeUiTest {
  method public A? getActivity();
  property public abstract A? activity;
}
```

### Window Provider Interface
```kotlin
public interface DialogWindowProvider {
  method public android.view.Window getWindow();
  property public abstract android.view.Window window;
}
```

### SwipeRefreshLayout Integration
```java
public static interface SwipeRefreshLayout.OnRefreshListener {
  method public void onRefresh();
}
```

## Error Handling and Debugging

### Compiler Flags for Debugging
```cmake
# Strict error checking
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

# EGL prototypes for graphics debugging
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -DEGL_EGLEXT_PROTOTYPES")
```

### Library Detection
```cmake
# Validate library availability
find_library(android-lib android)
if(NOT android-lib)
    message(FATAL_ERROR "Android library not found")
endif()
```

## Best Practices for Mobile JNI

### 1. Memory Management
- Use RAII patterns in C++ code
- Properly manage JNI local/global references
- Implement proper cleanup in native destructors

### 2. Threading Considerations
- JNI calls must happen on the correct thread
- Use AttachCurrentThread for background threads
- Implement proper synchronization for shared resources

### 3. Performance Optimization
- Minimize JNI boundary crossings
- Cache method IDs and class references
- Use batch operations when possible
- Optimize memory alignment with linker flags

### 4. Error Handling
- Check for exceptions after each JNI call
- Implement proper error propagation
- Use appropriate logging mechanisms

### 5. Build System Integration
- Use consistent CMake patterns
- Properly configure NDK dependencies
- Ensure reproducible builds across environments

### 6. Testing Strategy
- Integrate with native testing frameworks (GTest)
- Test on multiple Android API levels
- Validate memory usage and performance
- Test threading scenarios thoroughly

## Integration with Mobile ML Frameworks

### TensorFlow Lite Integration
```cmake
# Example pattern for ML framework integration
find_package(tensorflow-lite REQUIRED)
target_link_libraries(your_jni_lib tensorflow-lite::tensorflowlite)
```

### Image Processing Pipelines
```cmake
# Combine JNI with image processing libraries
target_link_libraries(image_processing_jni 
    PRIVATE 
    ${jnigraphics-lib}
    libyuv::yuv
    opencv::opencv
)
```

This documentation provides comprehensive guidance for implementing robust JNI solutions in Android mobile applications, with patterns derived from AndroidX library implementations.
