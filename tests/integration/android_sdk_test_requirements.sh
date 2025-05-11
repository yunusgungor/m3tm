#!/bin/bash

# Android SDK testleri için gereksinimleri kontrol eden script

set -e

echo "Android SDK test gereksinimlerini kontrol ediliyor..."

# Android SDK ortam değişkenini kontrol et
if [ -z "$ANDROID_HOME" ]; then
    echo "HATA: ANDROID_HOME ortam değişkeni tanımlanmamış."
    echo "Lütfen Android SDK'nın yolunu ANDROID_HOME ortam değişkenine atayın."
    echo "Örnek: export ANDROID_HOME=/Users/username/Library/Android/sdk"
    exit 1
else
    echo "✓ ANDROID_HOME tanımlı: $ANDROID_HOME"
fi

# Android SDK Build Tools kontrolü
if [ ! -d "$ANDROID_HOME/build-tools" ]; then
    echo "HATA: Android SDK Build Tools bulunamadı."
    echo "Lütfen Android SDK Manager ile Build Tools'u yükleyin."
    exit 1
else
    echo "✓ Android SDK Build Tools mevcut"
    # Mevcut versiyonları listele
    echo "  Mevcut Build Tools versiyonları:"
    ls -1 "$ANDROID_HOME/build-tools" | grep -v "^\."
fi

# Android SDK Platform Tools kontrolü
if [ ! -d "$ANDROID_HOME/platform-tools" ]; then
    echo "HATA: Android SDK Platform Tools bulunamadı."
    echo "Lütfen Android SDK Manager ile Platform Tools'u yükleyin."
    exit 1
else
    echo "✓ Android SDK Platform Tools mevcut"
fi

# NDK kontrolü
if [ ! -d "$ANDROID_HOME/ndk" ] && [ ! -d "$ANDROID_NDK_HOME" ]; then
    echo "UYARI: Android NDK bulunamadı. JNI testleri için gerekli olabilir."
    echo "Lütfen Android SDK Manager ile NDK'yı yükleyin veya ANDROID_NDK_HOME tanımlayın."
else
    if [ -d "$ANDROID_NDK_HOME" ]; then
        echo "✓ Android NDK mevcut: $ANDROID_NDK_HOME"
    else
        echo "✓ Android NDK mevcut: $ANDROID_HOME/ndk"
        # Mevcut versiyonları listele
        echo "  Mevcut NDK versiyonları:"
        ls -1 "$ANDROID_HOME/ndk" | grep -v "^\."
    fi
fi

# Java kontrolü
if ! command -v javac &> /dev/null; then
    echo "HATA: Java Development Kit (JDK) bulunamadı."
    echo "Lütfen JDK'yı yükleyin ve PATH'e ekleyin."
    exit 1
else
    JAVA_VERSION=$(javac -version 2>&1 | head -n 1)
    echo "✓ Java Development Kit mevcut: $JAVA_VERSION"
fi

# Gradle kontrolü
GRADLE_WRAPPER="./android/gradlew"
if [ ! -f "$GRADLE_WRAPPER" ]; then
    echo "UYARI: Gradle Wrapper bulunamadı: $GRADLE_WRAPPER"
    echo "Gradle Wrapper'ı yenilemek için android/ dizininde 'gradle wrapper' komutunu çalıştırın."
else
    echo "✓ Gradle Wrapper mevcut"
    # Gradle versiyonunu kontrol et
    GRADLE_VERSION=$($GRADLE_WRAPPER -v 2>&1 | grep "Gradle " | head -n 1)
    echo "  $GRADLE_VERSION"
fi

# Test için örnek model varlığını kontrol et
TEST_MODEL_PATH="./model_checkpoints/test_model.pt"
if [ ! -f "$TEST_MODEL_PATH" ]; then
    echo "UYARI: Test modeli bulunamadı: $TEST_MODEL_PATH"
    echo "Android SDK testleri için örnek bir model dosyası gerekli olabilir."
else
    echo "✓ Test modeli mevcut: $TEST_MODEL_PATH"
fi

# Android SDK build kontrolü
echo "Android SDK derleme kontrolü yapılıyor..."
cd android
if [ -f "./gradlew" ]; then
    chmod +x ./gradlew
    ./gradlew clean > /dev/null 2>&1 || { echo "HATA: Gradle temizleme işlemi başarısız oldu."; exit 1; }
    echo "✓ Gradle temizleme işlemi başarılı"
    ./gradlew assembleDebug > /dev/null 2>&1 || { echo "HATA: Android SDK derlenemedi."; exit 1; }
    echo "✓ Android SDK derleme başarılı"
    cd ..
else
    echo "HATA: Android dizininde gradlew bulunamadı."
    cd ..
    exit 1
fi

echo "Tüm gereksinimler kontrol edildi. Android SDK testleri çalıştırılabilir."
exit 0 