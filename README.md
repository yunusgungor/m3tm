# M³TM v2.3 - Mobil Multi-Modal Modüler Transformer

Kullanıcı verilerinin gizliliğini koruyan, cihaz üzerinde çalışan ve kişiselleştirilebilir bir yapay zeka modeli.

## Proje Açıklaması

M³TM v2.3, kullanıcının kişisel verileriyle (resimler, notlar, belgeler vb.) tamamen cihaz üzerinde eğitilebilen, bu veriler üzerinde semantik arama yapabilen, kullanıcıya verilerini indirme imkanı sunan ve geliştiricilerin özel mobil uygulamalar oluşturabilmesi için bir SDK (Yazılım Geliştirme Kiti) sağlayan, gizlilik odaklı, gelişmiş bir multi-modal yapay zeka modelidir.

### Temel Özellikler

- Ultra-hafif, multi-modal çekirdek model
- Cihaz üzerinde eğitim ve kişiselleştirme
- Metin ve görüntü modaliteleri desteği
- Cihaz üzerinde semantik arama
- Veri indirme özellikleri
- Android ve iOS için SDK

## Kurulum

```bash
# Geliştirme ortamını kurmak için
pip install -e ".[dev]"

# Sadece kütüphaneyi kurmak için
pip install -e .
```

## Kullanım

```python
import m3tm

# Örnek kod yakında eklenecek
```

## Geliştirme

```bash
# Testleri çalıştırmak için
pytest

# Kod formatlamak için
black src tests
isort src tests
```

## Android SDK Testleri

Android SDK sarmalayıcısının test edilmesi için aşağıdaki adımları izleyin:

1. Gereksinimleri kontrol edin:

```bash
tests/integration/android_sdk_test_requirements.sh
```

2. Testleri çalıştırın:

```bash
tests/integration/run_android_tests.py
```

Özel parametreler:

- `--skip-requirements`: Gereksinim kontrolünü atlar
- `--skip-build`: Android SDK derlemeyi atlar
- `--filter=PATTERN`: Belirli testleri çalıştırmak için filtre uygular
- `-v, --verbose`: Ayrıntılı çıktı gösterir

Örnek kullanım:

```bash
# Tüm gereksinimleri kontrol et ve testleri çalıştır
tests/integration/run_android_tests.py

# Belirli testleri çalıştır
tests/integration/run_android_tests.py --filter=jni

# Gereksinim kontrolünü atlayarak ayrıntılı çıktı ile testleri çalıştır
tests/integration/run_android_tests.py --skip-requirements -v
```

Test sonuçları `test_results/android_sdk_test_results.json` dosyasına kaydedilir.

## Test Sonuçları

Android SDK testlerinin son çalıştırma sonuçları:

- Toplam test sayısı: 6
- Başarılı: 2
- Atlanan: 4
- Başarısız/Hatalı: 0

Testler kısmen çalıştırılabildi çünkü ANDROID_HOME ortam değişkeni tanımlanmamıştı. Sadece dosya yapısı ve story durumu doğrulanabildi.

Ayrıntılı test sonuçları `test_results/android_sdk_test_results.json` dosyasında bulunabilir.

## Lisans

© 2024 M³TM Team. Tüm hakları saklıdır. 