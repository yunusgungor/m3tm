#!/bin/bash

# Test çalıştırma betiği

# xcodebuild yapılandırma bilgileri
PROJECT_NAME="M3TM"
SCHEME_NAME="M3TMTests"
DESTINATION="platform=iOS Simulator,name=iPhone 14,OS=16.2"

# Test uygulamasını oluştur ve testleri çalıştır
xcodebuild test -project "$PROJECT_NAME.xcodeproj" -scheme "$SCHEME_NAME" -destination "$DESTINATION" -configuration Debug

# Test sonuçlarını görüntüle
echo "Test sonuçları:"
cat /tmp/xcodebuild_test_result.log 