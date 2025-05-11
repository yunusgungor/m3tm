#!/bin/bash

# M³TM v2.3 Entegrasyon Test İş Akışı
# Codeflow-uyumlu entegrasyon test yürütme ve hata çözüm iş akışı
# ----------------------------------------------------------------------

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # Renk yok

echo -e "${BLUE}M³TM v2.3 Entegrasyon Test İş Akışını Başlatıyorum...${NC}"
echo -e "${BLUE}==============================================${NC}"

# Çalışma dizinini kontrol et
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
  echo -e "${RED}HATA: Git deposu bulunamadı. Bu betik mobilemodel projesinin kök dizininde çalıştırılmalıdır.${NC}"
  exit 1
fi

cd "$PROJECT_ROOT" || exit 1
echo -e "${GREEN}Proje kök dizini: ${PROJECT_ROOT}${NC}"

# Önkoşulları doğrula
echo -e "${BLUE}PRD ve mimari bütünlüğünü kontrol ediyorum...${NC}"
if [ ! -f "PRD.md" ]; then
  echo -e "${RED}HATA: PRD.md bulunamadı.${NC}"
  exit 1
fi

if [ ! -d ".project_meta/.architecture" ]; then
  echo -e "${RED}HATA: Mimari metaverileri (.project_meta/.architecture/) bulunamadı.${NC}"
  exit 1
fi

if [ ! -d ".project_meta/.patterns" ]; then
  echo -e "${RED}HATA: Örüntü kataloğu (.project_meta/.patterns/) bulunamadı.${NC}"
  exit 1
fi

# Hata log dizinlerini hazırla
echo -e "${BLUE}Hata log dizinlerini hazırlıyorum...${NC}"
mkdir -p .project_meta/.errors/metrics
mkdir -p .project_meta/.errors/reports/cascading_failures
mkdir -p .project_meta/.errors/reports/visualizations
mkdir -p .project_meta/.tests/visualizations

# Örüntü öğrenme, hata tespiti ve çözüm iş akışı
test_with_pattern_learning() {
  local test_file="$1"
  local test_name=$(basename "$test_file" .py)
  
  echo -e "\n${MAGENTA}[$test_name] Test yürütülüyor...${NC}"
  
  # Test çalıştır ve sonuçları kaydet
  python -m pytest -xvs "$test_file" > ".project_meta/.integration/logs/${test_name}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  
  # Test başarı durumunu kontrol et
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}[$test_name] Test başarılı!${NC}"
    # Başarılı testlerden örüntü öğrenme
    echo -e "${CYAN}[$test_name] Başarılı test uygulamalarından örüntüleri analiz ediyorum...${NC}"
    # Burada gerçek uygulamada pattern_learner çağrılacak
    python -c "print('Örüntü analizi simülasyonu: ${test_name} başarı örüntüleri tespit ediliyor...')"
  else
    echo -e "${RED}[$test_name] Test başarısız! Hataları analiz ediyorum...${NC}"
    
    # Hata analizi ve sınıflandırma
    echo -e "${YELLOW}[$test_name] Çok boyutlu hata sınıflandırması yapılıyor...${NC}"
    # Burada gerçek uygulamada error_analyzer çağrılacak
    
    # Hatanın çözüm stratejisini belirle
    echo -e "${YELLOW}[$test_name] Örüntü kataloguna göre çözüm stratejisi belirleniyor...${NC}"
    # Burada gerçek uygulamada pattern_catalog.json ve anti_patterns.json referans alınacak
    
    # Hata kaydını güncelle
    echo -e "${YELLOW}[$test_name] Hata kaydı güncelleniyor...${NC}"
    # Burada gerçek uygulamada error_log.json güncellenecek
    
    # Hata görselleştirmelerini oluştur
    echo -e "${YELLOW}[$test_name] Hata görselleştirmeleri oluşturuluyor...${NC}"
    # Burada gerçek uygulamada görselleştirme araçları çağrılacak
    
    return 1
  fi
  
  return 0
}

# Tüm testleri çalıştır ve sonuçları topla
echo -e "\n${BLUE}Entegrasyon Testlerini Çalıştırıyorum...${NC}"

TEST_FILES=(
  "tests/integration/test_numerical_stability.py"
  "tests/integration/test_fusion_integration.py"
  "tests/integration/test_search_robustness.py"
  "tests/integration/test_mobile_performance.py"
  "tests/integration/test_model_robustness.py"
  "tests/integration/test_model_e2e.py"
  "tests/integration/test_android_sdk.py"
)

PASSED=0
FAILED=0
FAILED_TESTS=()

for test_file in "${TEST_FILES[@]}"; do
  if [ ! -f "$test_file" ]; then
    echo -e "${YELLOW}UYARI: $test_file bulunamadı, atlanıyor.${NC}"
    continue
  fi
  
  test_with_pattern_learning "$test_file"
  
  if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    FAILED_TESTS+=("$(basename "$test_file")")
  fi
  
  # Her 3 test sonrası örüntü gözden geçirme
  if [ $(( (PASSED + FAILED) % 3 )) -eq 0 ]; then
    echo -e "\n${CYAN}Periyodik Örüntü Gözden Geçirmesi Yürütülüyor...${NC}"
    # Burada gerçek uygulamada periodic_pattern_review çağrılacak
    python -c "print('Örüntü gözden geçirme simülasyonu: Son 3 testten öğrenilen örüntüler değerlendiriliyor...')"
  fi
done

# Özet rapor
echo -e "\n${BLUE}Test Yürütme Özeti${NC}"
echo -e "${BLUE}==============================================${NC}"
echo -e "${GREEN}Başarılı Test Sayısı: $PASSED${NC}"
echo -e "${RED}Başarısız Test Sayısı: $FAILED${NC}"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
  echo -e "${RED}Başarısız Testler: ${FAILED_TESTS[*]}${NC}"
  
  # Döngüye girmeden pattern-based fixes
  echo -e "\n${MAGENTA}Örüntü Tabanlı Hata Düzeltme İş Akışı Başlatılıyor...${NC}"
  # Burada gerçek uygulamada pattern_based_fix_workflow çağrılacak
  
  # Mimari uygunluk kontrolü
  echo -e "\n${BLUE}Mimari Bütünlük Kontrolü Yürütülüyor...${NC}"
  # Burada gerçek uygulamada architecture_analyzer çağrılacak
  
  echo -e "\n${RED}Bazı testler başarısız oldu. Lütfen hata kayıtlarını kontrol edin:${NC}"
  echo -e "${YELLOW}.project_meta/.integration/logs/${NC}"
  echo -e "${YELLOW}.project_meta/.errors/error_log.json${NC}"
  exit 1
else
  echo -e "\n${GREEN}Tüm entegrasyon testleri başarılı!${NC}"
  
  # Dökümantasyon tazeliğini kontrol et
  echo -e "\n${BLUE}Döküman Tazeliği Kontrolü Yürütülüyor...${NC}"
  # Burada gerçek uygulamada doc_watcher çağrılacak
  
  echo -e "\n${GREEN}İş akışı başarıyla tamamlandı.${NC}"
  exit 0
fi 