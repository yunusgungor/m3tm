import unittest
import subprocess
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

class TestAndroidSDK(unittest.TestCase):
    """Android SDK sarmalayıcısı için entegrasyon testleri."""
    
    @classmethod
    def setUpClass(cls):
        """Test ortamını hazırla."""
        # Android SDK dizinini belirleme
        cls.android_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'android')))
        
        # Örnek model dosyası yolu
        cls.test_model_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'model_checkpoints', 'test_model.pt')))
        
        # Test için gerekli ortam değişkenlerini kontrol et
        cls.android_home = os.environ.get('ANDROID_HOME')
        if not cls.android_home:
            print("NOT: ANDROID_HOME ortam değişkeni tanımlanmamış, bazı Android SDK testleri atlanacak.")
        
        # Test modeli kontrolü
        if not cls.test_model_path.exists():
            print(f"NOT: Test modeli bulunamadı: {cls.test_model_path}, model gerektiren testler atlanacak.")
    
    def test_android_sdk_files_exist(self):
        """Android SDK dosyalarının varlığını test et - bu test her zaman çalışabilir."""
        # Android dizini kontrolü
        self.assertTrue(self.android_dir.exists(), "Android dizini bulunamadı")
        
        # Temel yapı dosyalarını kontrol et
        self.assertTrue((self.android_dir / "build.gradle").exists(), "build.gradle bulunamadı")
        self.assertTrue((self.android_dir / "gradlew").exists(), "gradlew bulunamadı")
        
        # src dizini kontrolleri
        src_dir = self.android_dir / "src"
        self.assertTrue(src_dir.exists(), "src dizini bulunamadı")
        self.assertTrue((src_dir / "main").exists(), "src/main dizini bulunamadı")
        self.assertTrue((src_dir / "main" / "java").exists(), "src/main/java dizini bulunamadı")
        self.assertTrue((src_dir / "main" / "jni").exists(), "src/main/jni dizini bulunamadı")
        
        # JNI dosyası kontrolü
        self.assertTrue((src_dir / "main" / "jni" / "m3tm_jni.cpp").exists(), "m3tm_jni.cpp bulunamadı")
        self.assertTrue((src_dir / "main" / "jni" / "CMakeLists.txt").exists(), "CMakeLists.txt bulunamadı")
        
        # Java paket yapısı kontrolü
        java_dir = src_dir / "main" / "java" / "com" / "m3tm" / "sdk"
        self.assertTrue(java_dir.exists(), "Java SDK paketi bulunamadı")
        
        # Test için yeterli sayıda Java dosyasını kontrol et
        java_files = list(java_dir.glob("*.java"))
        self.assertGreaterEqual(len(java_files), 5, "Yeterli sayıda Java dosyası bulunamadı")
        
        # Örnek sınıfı kontrolü
        example_dir = java_dir / "example"
        self.assertTrue(example_dir.exists(), "Örnek kod dizini bulunamadı")
        self.assertTrue((example_dir / "M3TMExample.java").exists(), "M3TMExample.java bulunamadı")
    
    @unittest.skipIf(not os.environ.get('ANDROID_HOME'), "ANDROID_HOME ortam değişkeni tanımlanmamış")
    def test_jni_methods_existence(self):
        """JNI metotlarının varlığını kontrol et."""
        # JNI cpp dosyasını kontrol et ve gerekli metotların varlığını doğrula
        jni_path = self.android_dir / 'src' / 'main' / 'jni' / 'm3tm_jni.cpp'
        self.assertTrue(jni_path.exists(), "m3tm_jni.cpp dosyası bulunamadı")
        
        with open(jni_path, 'r') as f:
            jni_content = f.read()
        
        required_jni_methods = [
            "Java_com_m3tm_sdk_M3TMModelManager_nativeLoadModel",
            "Java_com_m3tm_sdk_M3TMModel_nativeTextInference",
            "Java_com_m3tm_sdk_M3TMModel_nativeImageInference",
            "Java_com_m3tm_sdk_M3TMModel_nativeMultimodalInference",
            "Java_com_m3tm_sdk_M3TMTrainingManager_nativeCreateTrainingSession",
            "Java_com_m3tm_sdk_M3TMTrainingManager_nativeTrainIteration"
        ]
        
        for method in required_jni_methods:
            self.assertIn(method, jni_content, f"JNI metodu bulunamadı: {method}")
    
    @unittest.skipIf(not os.environ.get('ANDROID_HOME'), "ANDROID_HOME ortam değişkeni tanımlanmamış")
    def test_android_java_api_structure(self):
        """Java API yapısını kontrol et."""
        # Gerekli Java sınıflarının varlığını kontrol et
        java_dir = self.android_dir / 'src' / 'main' / 'java' / 'com' / 'm3tm' / 'sdk'
        
        required_classes = [
            "M3TM.java",
            "M3TMModel.java",
            "M3TMModelManager.java",
            "M3TMTrainingManager.java",
            "ModelInfo.java",
            "M3TMException.java",
            "ModelException.java",
            "InferenceException.java",
            "TrainingException.java",
            "TrainingCallback.java"
        ]
        
        for class_file in required_classes:
            self.assertTrue((java_dir / class_file).exists(), f"{class_file} bulunamadı")
    
    @unittest.skipIf(not os.environ.get('ANDROID_HOME'), "ANDROID_HOME ortam değişkeni tanımlanmamış")
    def test_android_interface_methods(self):
        """Android arayüz metotlarını kontrol et."""
        # Temel API sınıflarının içeriğini kontrol et
        java_dir = self.android_dir / 'src' / 'main' / 'java' / 'com' / 'm3tm' / 'sdk'
        
        # M3TMModel sınıfını kontrol et
        with open(java_dir / "M3TMModel.java", 'r') as f:
            model_content = f.read()
        
        required_model_methods = [
            "public float[] textInference(",
            "public float[] imageInference(",
            "public float[] multimodalInference(",
            "public void close("
        ]
        
        for method in required_model_methods:
            self.assertIn(method, model_content, f"M3TMModel metodu bulunamadı: {method}")
        
        # M3TMModelManager sınıfını kontrol et
        with open(java_dir / "M3TMModelManager.java", 'r') as f:
            manager_content = f.read()
        
        required_manager_methods = [
            "public static M3TMModel loadModel(",
            "public static ModelInfo getModelInfo("
        ]
        
        for method in required_manager_methods:
            self.assertIn(method, manager_content, f"M3TMModelManager metodu bulunamadı: {method}")
        
        # M3TMTrainingManager sınıfını kontrol et
        with open(java_dir / "M3TMTrainingManager.java", 'r') as f:
            training_content = f.read()
        
        required_training_methods = [
            "public void createTrainingSession(",
            "public float trainIteration(",
            "public void setCallback(",
            "public void close("
        ]
        
        for method in required_training_methods:
            self.assertIn(method, training_content, f"M3TMTrainingManager metodu bulunamadı: {method}")
    
    @unittest.skipIf(not os.environ.get('ANDROID_HOME'), "ANDROID_HOME ortam değişkeni tanımlanmamış")
    def test_error_handling(self):
        """Hata yönetimi mekanizmalarını kontrol et."""
        # Exception sınıfları ve hata yönetimini kontrol et
        java_dir = self.android_dir / 'src' / 'main' / 'java' / 'com' / 'm3tm' / 'sdk'
        
        with open(java_dir / "M3TMException.java", 'r') as f:
            exception_content = f.read()
        
        self.assertIn("public class M3TMException extends Exception", exception_content)
        self.assertIn("getErrorCode()", exception_content)
        
        # Alt exception sınıflarını kontrol et
        exception_subclasses = ["ModelException", "InferenceException", "TrainingException"]
        for subclass in exception_subclasses:
            with open(java_dir / f"{subclass}.java", 'r') as f:
                subclass_content = f.read()
            
            self.assertIn(f"public class {subclass} extends M3TMException", subclass_content)
    
    def test_story_status(self):
        """Story 18'in durumunu ve test durumlarını kontrol et."""
        # story_18.json dosyasını oku
        story_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.project_meta', '.stories', 'story_18.json')))
        self.assertTrue(story_path.exists(), "Story 18 dosyası bulunamadı")
        
        with open(story_path, 'r') as f:
            story_data = json.load(f)
        
        # Temel bilgileri kontrol et
        self.assertEqual(story_data["story_id"], "story_18", "Yanlış story ID")
        self.assertEqual(story_data["title"], "Android SDK Sarmalayıcı İmplementasyonu", "Yanlış story başlığı")
        
        # Gerçekleştirilen görevleri kontrol et
        self.assertIn("completed_tasks", story_data, "Gerçekleştirilen görevler bulunamadı")
        self.assertGreaterEqual(len(story_data["completed_tasks"]), 1, "Gerçekleştirilen görevler boş olmamalı")
        
        # Test sonuçlarını kontrol et
        self.assertIn("test_results", story_data, "Test sonuçları bulunamadı")
        self.assertIn("coverage", story_data["test_results"], "Test kapsamı bulunamadı")
        self.assertIn("unit_tests", story_data["test_results"], "Birim test sonuçları bulunamadı")
        self.assertIn("integration_tests", story_data["test_results"], "Entegrasyon test sonuçları bulunamadı")
        
        # Şu andaki durumu kontrol et
        status = story_data["status"]
        self.assertIn(status, ["todo", "in_progress", "done"], "Geçersiz story durumu")
        print(f"Story 18 durumu: {status}")

if __name__ == '__main__':
    unittest.main() 