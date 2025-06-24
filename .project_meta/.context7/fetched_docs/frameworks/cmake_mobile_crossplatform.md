# CMake Mobile and Cross-Platform Build Documentation

## Overview
CMake configuration patterns for mobile development, focusing on Android NDK and iOS cross-compilation for our mobile model project.

## Android NDK Cross-Compilation

### Basic Android Configuration
```cmake
# Via toolchain file
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 21)  # API level
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK /path/to/android-ndk)
set(CMAKE_ANDROID_STL_TYPE gnustl_static)
```

### Command Line Configuration
```bash
cmake ../src \
  -DCMAKE_SYSTEM_NAME=Android \
  -DCMAKE_SYSTEM_VERSION=21 \
  -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
  -DCMAKE_ANDROID_NDK=/path/to/android-ndk \
  -DCMAKE_ANDROID_STL_TYPE=c++_shared
```

### Advanced Android Build Settings
```cmake
# Android-specific build variables
set(CMAKE_ANDROID_ARCH armv7-a)
set(CMAKE_ANDROID_STL_TYPE c++_shared)
set(CMAKE_ANDROID_API_MIN 9)
set(CMAKE_ANDROID_API 15)
set(CMAKE_ANDROID_GUI 1)

# NDK version detection (CMake 3.20+)
set(CMAKE_ANDROID_NDK_VERSION)  # Auto-detected

# Default build type for Android (CMake 3.20+)
set(CMAKE_BUILD_TYPE RelWithDebInfo)  # Default for Android

# Platform level configuration
if(ANDROID AND NOT DEFINED ANDROID_PLATFORM_LEVEL AND NOT CMAKE_SYSTEM_VERSION EQUAL 1)
  set(ANDROID_PLATFORM_LEVEL "${CMAKE_SYSTEM_VERSION}")
endif()
```

### Android Toolchain Configuration
```cmake
# Toolchain prefix usage
${CMAKE_CXX_ANDROID_TOOLCHAIN_PREFIX}ld${CMAKE_CXX_ANDROID_TOOLCHAIN_SUFFIX}

# NDK toolchain version formats
# <major>.<minor>
# clang<major>.<minor>
# clang

# Deprecated headers support (CMake 3.9+)
set(CMAKE_ANDROID_NDK_DEPRECATED_HEADERS ON)
```

### Android Project Structure
```cmake
# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# Native library configuration
add_library(native_lib SHARED native_lib.c)

# Android executable with JNI
add_executable(android_app
  ${C_FILES}
  ${JAVA_FILES}
  ${RES_FILES}
  ${ANDROID_FILES}
)

# Target properties for Android
set_target_properties(android_app PROPERTIES
  ANDROID_SKIP_ANT_STEP 1
  ANDROID_PROGUARD 1
  ANDROID_PROGUARD_CONFIG_PATH proguard-android.txt
  ANDROID_SECURE_PROPS_PATH /secure/path
)

# Android library dependencies
set_property(TARGET android_app PROPERTY 
  ANDROID_NATIVE_LIB_DIRECTORIES $<TARGET_FILE_DIR:android_app>)
set_property(TARGET android_app PROPERTY 
  ANDROID_NATIVE_LIB_DEPENDENCIES $<TARGET_FILE_NAME:android_app>)
set_property(TARGET android_app PROPERTY 
  ANDROID_JAR_DIRECTORIES $<TARGET_FILE_DIR:native_lib>)
```

## iOS Cross-Compilation

### Basic iOS Configuration
```cmake
# Via toolchain file
set(CMAKE_SYSTEM_NAME iOS)
```

### Command Line iOS Configuration
```bash
# Basic iOS build
cmake .. -GXcode -DCMAKE_SYSTEM_NAME=iOS

# Universal binary with multiple architectures
cmake -S. -B_builds -GXcode \
      -DCMAKE_SYSTEM_NAME=iOS \
      "-DCMAKE_OSX_ARCHITECTURES=armv7;armv7s;arm64;i386;x86_64" \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=9.3 \
      -DCMAKE_INSTALL_PREFIX=`pwd`/_install \
      -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO \
      -DCMAKE_IOS_INSTALL_COMBINED=YES
```

### iOS Platform Settings
```cmake
# SDK and architecture configuration
set(CMAKE_OSX_SYSROOT iphoneos4.3)
set(CMAKE_OSX_ARCHITECTURES "armv6;armv7;i386")

# iOS-specific try-compile settings
if(IOS)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
endif()
```

### iOS Framework Linking
```cmake
# Link iOS frameworks
target_link_libraries(ios_app
  static_lib
  "-framework CoreGraphics"
  "-framework Foundation"
  "-framework UIKit"
)

# Apple framework linking (macOS/iOS)
if(APPLE)
  target_link_libraries(CMakeLib PUBLIC "-framework CoreFoundation")
  target_link_libraries(CMakeLib PUBLIC "-framework CoreServices")
endif()
```

### iOS Bundle Configuration
```cmake
# iOS app bundle
add_executable(ios_app MACOSX_BUNDLE 
  ${SOURCES} 
  ${HEADERS} 
  ${RESOURCES}
)

# Platform-specific function checks
if(APPLE)
  check_function_exists("mach_absolute_time" HAVE_MACH_ABSOLUTE_TIME)
endif()
```

### iOS Build Commands
```bash
# Build for iOS simulator
cmake --build ... -- -sdk iphonesimulator

# Check universal binary architectures
lipo -info _install/lib/libfoo.a

# Inspect iOS library version
otool -l _install/lib/libfoo.a | grep -A2 LC_VERSION_MIN_IPHONEOS
```

## Cross-Platform Toolchain Patterns

### Generic Cross-Compilation Template
```cmake
# Basic cross-compilation setup
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_SYSROOT /home/devel/rasp-pi-rootfs)
set(CMAKE_STAGING_PREFIX /home/devel/stage)

set(tools /home/devel/gcc-4.7-linaro-rpi-gnueabihf)
set(CMAKE_C_COMPILER ${tools}/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER ${tools}/bin/arm-linux-gnueabihf-g++)

# Find root path modes for cross-compilation
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

### Clang Cross-Compilation
```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(triple arm-linux-gnueabihf)

set(CMAKE_C_COMPILER clang)
set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_COMPILER_TARGET ${triple})
```

### Toolchain File Best Practices
```cmake
# Compiler specification
set(CMAKE_C_COMPILER /full/path/to/qcc --arg1 --arg2)

# Platform variables for try_compile
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES MY_CUSTOM_VARIABLE)

# Project with no languages (for toolchain-only projects)
project(Toolchain LANGUAGES NONE)
```

## Windows Mobile Platforms

### Windows Phone Configuration
```cmake
set(CMAKE_SYSTEM_NAME WindowsPhone)
set(CMAKE_SYSTEM_VERSION 8.1)
```

### Windows Store/Universal App Configuration
```cmake
# Windows Store (8.1)
set(CMAKE_SYSTEM_NAME WindowsStore)
set(CMAKE_SYSTEM_VERSION 8.1)

# Windows 10 Universal App
set(CMAKE_SYSTEM_NAME WindowsStore)
set(CMAKE_SYSTEM_VERSION 10.0)
```

## Build Optimization for Mobile

### Compiler Optimizations
```cmake
# Optimization flags for mobile
if(CMAKE_C_COMPILER_ID MATCHES "^(GNU|LCC)$")
  string(APPEND CMAKE_C_FLAGS " -O3")
endif()

# Disable optimizer for specific cases
if(CMAKE_C_COMPILER_ID STREQUAL "XL")
  set_property(TARGET target PROPERTY COMPILE_FLAGS "-qnooptimize")
elseif((CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "LCC") AND
       CMAKE_C_COMPILER_VERSION VERSION_LESS 3.4)
  set_property(TARGET target PROPERTY COMPILE_FLAGS "-O0")
endif()
```

### IPO (Interprocedural Optimization) Support
```cmake
# Check IPO support (CMake 3.9+)
include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if(result)
  set_property(TARGET target PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
```

### Mobile-Specific Considerations
```cmake
# Note on CMAKE_TRY_COMPILE_TARGET_TYPE
# Setting to STATIC_LIBRARY can interfere with function detection
# Commonly happens with ios.toolchain.cmake
if(NOT DEFINED CMAKE_TRY_COMPILE_TARGET_TYPE)
  # Let CMake decide based on platform
endif()
```

## Android.mk Integration

### Prebuilt Static Library Definition
```makefile
# Basic prebuilt library
include $(CLEAR_VARS)
LOCAL_MODULE := library_name
LOCAL_SRC_FILES := path/to/lib.a
LOCAL_HAS_CPP := true
include $(PREBUILT_STATIC_LIBRARY)

# Library with dependencies
include $(CLEAR_VARS)
LOCAL_MODULE := main_lib
LOCAL_SRC_FILES := path/to/main.a
LOCAL_CPP_FEATURES := rtti exceptions
LOCAL_STATIC_LIBRARIES := dep1 dep2 dep3
LOCAL_EXPORT_LDLIBS := -lm
LOCAL_HAS_CPP := true
include $(PREBUILT_STATIC_LIBRARY)
```

### Android.mk Import Configuration
```makefile
LOCAL_PATH := $(call my-dir)
_IMPORT_PREFIX := $(LOCAL_PATH)/../..
```

## ProGuard Configuration for Android

### Essential ProGuard Rules
```proguard
# Basic optimization settings
-dontoptimize
-dontpreverify
-dontskipnonpubliclibraryclasses

# Keep annotations
-keepattributes *Annotation*

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep view setters/getters
-keepclassmembers public class * extends android.view.View {
   void set*(***);
   *** get*();
}

# Keep Parcelable
-keep class * implements android.os.Parcelable {
  public static final android.os.Parcelable$Creator *;
}

# Suppress support library warnings
-dontwarn android.support.**

# Keep licensing service
-keep public class com.android.vending.licensing.ILicensingService
```

## Testing and Validation

### Symbol Existence Testing
```cmake
# Test with optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|LCC)$")
  string(APPEND CMAKE_CXX_FLAGS " -O3")
  check_cxx_symbol_exists(test_symbol "header.h" SYMBOL_EXISTS_O3)
  
  if(SYMBOL_EXISTS_O3)
    message(SEND_ERROR "False positive symbol detection with -O3")
  endif()
endif()
```

### Platform Feature Detection
```cmake
# Apple-specific feature detection
if(APPLE)
  # Check for platform-specific functions
  check_function_exists("mach_absolute_time" HAVE_MACH_ABSOLUTE_TIME)
  
  # Apple platform selection (CMake 3.31+)
  generate_apple_platform_selection_file(${CMAKE_CURRENT_BINARY_DIR}/platforms.cmake)
endif()
```

## Environment and Toolchain Detection

### Environment-Based Toolchain
```cmake
# Environment variable detection
set(ENV{CMAKE_TOOLCHAIN_FILE} "path/to/toolchain.cmake")

# Clear toolchain file
set(CMAKE_TOOLCHAIN_FILE "")
```

### Compiler Feature Handling
```cmake
# Handle AppleClang < 5.1 features
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND 
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
  list(REMOVE_ITEM CXX_non_features
    cxx_attribute_deprecated
    cxx_binary_literals
  )
endif()
```

## Key Mobile Development Constraints

1. **Target API Levels**: Ensure compatibility with minimum supported versions
2. **Architecture Support**: Plan for multiple architectures (arm64-v8a, x86_64)
3. **Library Dependencies**: Minimize external dependencies for smaller APK/IPA sizes
4. **Optimization Flags**: Balance performance vs. binary size
5. **Framework Integration**: Proper linking of platform-specific frameworks
6. **Cross-Compilation**: Robust toolchain configuration for host/target differences
7. **Testing**: Validate on actual devices, not just simulators
8. **Deployment**: Consider app store requirements and signing processes

## Best Practices for Mobile CMake

1. Use toolchain files for consistent cross-compilation
2. Test optimization flags thoroughly on target devices
3. Minimize try-compile operations for faster configuration
4. Validate symbol detection with different optimization levels
5. Use generator expressions for platform-specific configurations
6. Implement proper error handling for missing tools/SDKs
7. Document toolchain requirements and setup procedures
8. Test universal binaries for iOS compatibility
9. Verify ProGuard rules don't break functionality
10. Monitor binary sizes and startup performance

This documentation provides comprehensive guidance for setting up robust CMake build systems for mobile development across Android and iOS platforms.
