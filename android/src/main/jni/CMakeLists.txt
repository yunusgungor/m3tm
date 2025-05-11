cmake_minimum_required(VERSION 3.22.1)

project(m3tm_sdk_jni)

# C++17 aktif etme
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android loglama kütüphanesi
find_library(log-lib log)

# Ana kaynak dosya(ları)
set(SOURCES
    m3tm_jni.cpp
)

# Paylaşılan kütüphane
add_library(
    m3tm_sdk_jni
    SHARED
    ${SOURCES}
)

# Dahil edilecek header dizinleri
target_include_directories(
    m3tm_sdk_jni
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Bağımlı kütüphaneler
target_link_libraries(
    m3tm_sdk_jni
    ${log-lib}
)

# PyTorch'un etkinleştirilmesi için ilave ayarlar
# Not: Şu aşamada PyTorch aktif edilmiyor, ileride eklenecek

# Derleme flag'leri
target_compile_options(
    m3tm_sdk_jni
    PRIVATE
    -Wall
    -Werror
    -fexceptions
) 