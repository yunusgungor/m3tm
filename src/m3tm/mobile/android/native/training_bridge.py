"""
M³TM v2.3 Eğitim Köprüsü modülü.

Bu modül, model eğitim işlemlerini Java/Kotlin tarafından 
çağrılabilir JNI işlevleri olarak sunar.
"""

import logging
import os
from pathlib import Path
import json
import time
import uuid
from typing import Dict, List, Optional, Union, Any, Callable

import torch
import numpy as np

from .jni_helpers import (
    convert_tensor_to_java,
    convert_java_to_tensor,
    convert_dict_to_java_map,
    convert_list_to_java_list,
    JNIError,
    JNIRuntimeError,
    safe_jni_call
)
from .model_loader import ModelLoader, ModelNotFoundError

# Sabitler
TRAINING_SESSIONS_DIR = "training_sessions"
MAX_INACTIVE_SESSION_AGE = 3600 * 24  # 24 saat (saniye cinsinden)

class TrainingError(Exception):
    """Eğitim işlemi sırasında oluşan hatalar için istisna sınıfı."""
    pass

class TrainingSessionExpiredError(TrainingError):
    """Eğitim oturumunun süresi dolduğunda fırlatılan istisna."""
    pass

class TrainingSessionNotFoundError(TrainingError):
    """Eğitim oturumu bulunamadığında fırlatılan istisna."""
    pass

class TrainingCallback:
    """
    Eğitim ilerleme geri çağrıları için arayüz sınıfı.
    Bu sınıf, eğitim sürecinin durumu hakkında bilgi sağlamak için kullanılır.
    """
    
    def on_batch_complete(self, batch: int, metrics: Dict[str, Any]) -> None:
        """
        Her batch tamamlandığında çağrılır.
        
        Args:
            batch: Tamamlanan batch numarası.
            metrics: Batch metrikleri.
        """
        pass
    
    def on_epoch_complete(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Her epoch tamamlandığında çağrılır.
        
        Args:
            epoch: Tamamlanan epoch numarası.
            metrics: Epoch metrikleri.
        """
        pass
    
    def on_training_complete(self, metrics: Dict[str, Any]) -> None:
        """
        Eğitim tamamlandığında çağrılır.
        
        Args:
            metrics: Final eğitim metrikleri.
        """
        pass

class TrainingBridge:
    """
    Model eğitim işlemlerini JNI üzerinden Java/Kotlin'e sunan köprü sınıfı.
    
    Bu sınıf, modelleri ince ayarlamak için eğitim oturumları oluşturmak, eğitim
    başlatmak, durumunu izlemek ve eğitilmiş modelleri kaydetmek için işlevler sağlar.
    """
    
    def __init__(self, model_loader: ModelLoader):
        """
        TrainingBridge sınıfını başlatır.
        
        Args:
            model_loader: Model yükleme ve yönetimi için ModelLoader örneği.
        """
        self.model_loader = model_loader
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Eğitim oturumları dizinini oluştur
        self.sessions_dir = Path(model_loader.model_cache_dir) / TRAINING_SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Varolan oturumları yükle
        self._load_active_sessions()
        
        logging.info("TrainingBridge başlatıldı")
    
    def create_training_session(
        self,
        model_id: str,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Yeni bir eğitim oturumu oluşturur.
        
        Args:
            model_id: Eğitilecek modelin benzersiz tanımlayıcısı.
            training_config: Eğitim yapılandırması.
            
        Returns:
            session_info: Oturum bilgisini içeren sözlük.
            
        Raises:
            ModelNotFoundError: Model bulunamadığında.
            TrainingError: Oturum oluşturulamadığında.
        """
        try:
            # Modelin var olduğunu doğrula
            model_info = self.model_loader.get_model_metadata(model_id)
            
            # Benzersiz oturum kimliği oluştur
            session_id = str(uuid.uuid4())
            
            # Oturum verilerini hazırla
            session_data = {
                "session_id": session_id,
                "model_id": model_id,
                "model_version": model_info["version"],
                "create_time": time.time(),
                "last_active_time": time.time(),
                "status": "created",
                "training_config": training_config,
                "metrics": {},
                "trained_model": None
            }
            
            # Oturum verilerini kaydet
            self.sessions[session_id] = session_data
            self._save_session(session_id)
            
            logging.info(f"Yeni eğitim oturumu oluşturuldu: {session_id} (model: {model_id})")
            
            return {
                "session_id": session_id,
                "model_id": model_id,
                "status": "created",
                "create_time": session_data["create_time"]
            }
            
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise TrainingError(f"Eğitim oturumu oluşturulamadı: {str(e)}")
    
    def start_training(
        self,
        session_id: str,
        training_data: Dict[str, Any],
        callback: Optional[TrainingCallback] = None
    ) -> Dict[str, Any]:
        """
        Eğitim oturumunu başlatır ve modeli eğitir.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            training_data: Eğitim verileri.
            callback: İlerlemeyi bildirmek için opsiyonel geri çağrı.
            
        Returns:
            training_result: Eğitim sonuçlarını içeren sözlük.
            
        Raises:
            TrainingSessionNotFoundError: Oturum bulunamadığında.
            TrainingError: Eğitim sırasında hata oluştuğunda.
        """
        try:
            # Oturumu doğrula
            session = self._get_validated_session(session_id)
            
            # Modeli yükle
            model_id = session["model_id"]
            model = self.model_loader.get_model(model_id)
            
            # Oturum durumunu güncelle
            session["status"] = "training"
            session["last_active_time"] = time.time()
            session["train_start_time"] = time.time()
            self._save_session(session_id)
            
            # Eğitim konfigürasyonunu al
            config = session["training_config"]
            num_epochs = config.get("num_epochs", 1)
            batch_size = config.get("batch_size", 8)
            learning_rate = config.get("learning_rate", 1e-5)
            
            logging.info(f"Eğitim başlatılıyor: {session_id} (model: {model_id}, epochs: {num_epochs})")
            
            try:
                # Modeli eğitim moduna getir
                model.train()
                
                # Optimizasyonu ayarla
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                # Eğitim verilerini hazırla
                train_data = self._prepare_training_data(training_data)
                
                # Epoch döngüsü
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    batch_count = 0
                    
                    # Batch döngüsü
                    for batch_idx, batch_data in enumerate(self._batch_data(train_data, batch_size)):
                        # Girişleri hazırla
                        inputs = self._prepare_batch_inputs(batch_data)
                        
                        # İleri geçiş
                        outputs = model(inputs)
                        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs
                        
                        # Geri yayılım
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # İlerlemeyi izle
                        batch_loss = loss.item()
                        epoch_loss += batch_loss
                        batch_count += 1
                        
                        # Batch tamamlandı geri çağrısı
                        if callback:
                            batch_metrics = {
                                "epoch": epoch + 1,
                                "batch": batch_idx + 1,
                                "loss": batch_loss,
                                "learning_rate": learning_rate
                            }
                            callback.on_batch_complete(batch_idx + 1, batch_metrics)
                    
                    # Epoch metrikleri
                    avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                    epoch_metrics = {
                        "epoch": epoch + 1,
                        "loss": avg_epoch_loss,
                        "learning_rate": learning_rate
                    }
                    
                    # Epoch metrikleri kaydet
                    if "epochs" not in session["metrics"]:
                        session["metrics"]["epochs"] = []
                    session["metrics"]["epochs"].append(epoch_metrics)
                    
                    # Epoch tamamlandı geri çağrısı
                    if callback:
                        callback.on_epoch_complete(epoch + 1, epoch_metrics)
                    
                    logging.info(f"Epoch {epoch+1}/{num_epochs} tamamlandı: kayıp = {avg_epoch_loss:.4f}")
                
                # Modeli değerlendirme moduna getir
                model.eval()
                
                # Eğitimi tamamla
                train_end_time = time.time()
                training_duration = train_end_time - session["train_start_time"]
                
                # Final metrikleri
                final_metrics = {
                    "num_epochs": num_epochs,
                    "total_batches": batch_count * num_epochs,
                    "training_duration": training_duration,
                    "final_loss": avg_epoch_loss,
                    "learning_rate": learning_rate
                }
                
                # Eğitilmiş modeli oturuma kaydet
                session["trained_model"] = model
                session["metrics"]["final"] = final_metrics
                session["status"] = "completed"
                session["last_active_time"] = time.time()
                session["train_end_time"] = train_end_time
                self._save_session(session_id)
                
                # Eğitim tamamlandı geri çağrısı
                if callback:
                    callback.on_training_complete(final_metrics)
                
                logging.info(f"Eğitim tamamlandı: {session_id} (süre: {training_duration:.2f}s)")
                
                return {
                    "session_id": session_id,
                    "status": "completed",
                    "training_duration": training_duration,
                    "metrics": final_metrics
                }
                
            except Exception as e:
                # Eğitim hatası
                session["status"] = "failed"
                session["error"] = str(e)
                session["last_active_time"] = time.time()
                self._save_session(session_id)
                
                logging.error(f"Eğitim hatası: {session_id} - {str(e)}")
                raise TrainingError(f"Eğitim sırasında hata: {str(e)}")
            
        except TrainingSessionNotFoundError:
            raise
        except Exception as e:
            raise TrainingError(f"Eğitim başlatılamadı: {str(e)}")
    
    def get_training_status(self, session_id: str) -> Dict[str, Any]:
        """
        Eğitim oturumunun durumunu alır.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            
        Returns:
            status_info: Oturum durumunu içeren sözlük.
            
        Raises:
            TrainingSessionNotFoundError: Oturum bulunamadığında.
        """
        try:
            # Oturumu doğrula
            session = self._get_validated_session(session_id)
            
            # Durum bilgisini hazırla
            status_info = {
                "session_id": session_id,
                "model_id": session["model_id"],
                "status": session["status"],
                "create_time": session["create_time"],
                "last_active_time": session["last_active_time"]
            }
            
            # Eğitim devam ediyorsa veya tamamlandıysa ek bilgiler ekle
            if session["status"] in ["training", "completed"]:
                # Eğitim metrikleri
                status_info["metrics"] = session["metrics"].get("final", {})
                
                # Eğitim süresi
                if "train_start_time" in session:
                    if "train_end_time" in session:
                        training_duration = session["train_end_time"] - session["train_start_time"]
                    else:
                        training_duration = time.time() - session["train_start_time"]
                    status_info["training_duration"] = training_duration
            
            # Hata varsa ekle
            if "error" in session:
                status_info["error"] = session["error"]
            
            return status_info
            
        except TrainingSessionNotFoundError:
            raise
    
    def save_trained_model(
        self,
        session_id: str,
        output_path: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Eğitilmiş modeli kaydeder.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            output_path: Kaydedilecek dizin (opsiyonel).
            model_name: Kaydedilecek model adı (opsiyonel).
            
        Returns:
            save_info: Kaydetme bilgisini içeren sözlük.
            
        Raises:
            TrainingSessionNotFoundError: Oturum bulunamadığında.
            TrainingError: Model kaydedilemediğinde veya eğitim tamamlanmadığında.
        """
        try:
            # Oturumu doğrula
            session = self._get_validated_session(session_id)
            
            # Eğitimin tamamlandığını kontrol et
            if session["status"] != "completed":
                raise TrainingError(f"Eğitim tamamlanmadı. Mevcut durum: {session['status']}")
            
            # Eğitilmiş modelin var olduğunu kontrol et
            if "trained_model" not in session or session["trained_model"] is None:
                raise TrainingError("Eğitilmiş model bulunamadı")
            
            # Model bilgilerini al
            model_id = session["model_id"]
            original_version = session["model_version"]
            
            # Yeni model adını ve versiyonunu belirle
            new_model_name = model_name or model_id
            new_version = f"{original_version}_ft_{int(time.time())}"  # ft = fine-tuned
            
            # Çıktı dizinini belirle
            model_dir = Path(output_path) if output_path else Path(self.model_loader.model_cache_dir) / new_model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Model dosya yollarını oluştur
            model_file = model_dir / f"{new_model_name}_v{new_version}.pt"
            config_file = model_dir / f"{new_model_name}_v{new_version}.json"
            info_file = model_dir / "model_info.json"
            
            # Model bilgilerini hazırla
            model_info = {
                "name": new_model_name,
                "description": f"Fine-tuned from {model_id} v{original_version}",
                "base_model": model_id,
                "base_version": original_version,
                "latest_version": new_version,
                "versions": [new_version],
                "fine_tuned": True,
                "fine_tune_date": time.time(),
                "training_config": session["training_config"],
                "training_metrics": session["metrics"]
            }
            
            # Model yapılandırmasını hazırla
            model_config = {
                "model_id": new_model_name,
                "version": new_version,
                "fine_tuned_from": {
                    "model_id": model_id,
                    "version": original_version
                },
                "training_config": session["training_config"]
            }
            
            try:
                # Modeli TorchScript olarak kaydet
                trained_model = session["trained_model"]
                traced_model = torch.jit.trace(trained_model, ())  # Input example will be needed here, this is a simplification
                torch.jit.save(traced_model, str(model_file))
                
                # Model bilgisi ve yapılandırmasını kaydet
                with open(info_file, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, ensure_ascii=False, indent=2)
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(model_config, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Eğitilmiş model kaydedildi: {new_model_name} v{new_version}")
                
                # Kaydetme bilgisini hazırla
                save_info = {
                    "model_id": new_model_name,
                    "version": new_version,
                    "file_path": str(model_file),
                    "config_path": str(config_file),
                    "info_path": str(info_file),
                    "fine_tuned_from": {
                        "model_id": model_id,
                        "version": original_version
                    }
                }
                
                return save_info
                
            except Exception as e:
                raise TrainingError(f"Model kaydedilemedi: {str(e)}")
            
        except TrainingSessionNotFoundError:
            raise
        except Exception as e:
            raise TrainingError(f"Eğitilmiş model kaydedilemedi: {str(e)}")
    
    def delete_training_session(self, session_id: str) -> bool:
        """
        Eğitim oturumunu siler.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            
        Returns:
            success: İşlemin başarı durumu.
        """
        try:
            # Oturumu doğrula
            self._get_validated_session(session_id)
            
            # Oturum dosyasını sil
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Bellekteki oturumu sil
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            logging.info(f"Eğitim oturumu silindi: {session_id}")
            return True
            
        except TrainingSessionNotFoundError:
            logging.warning(f"Silinecek oturum bulunamadı: {session_id}")
            return False
        except Exception as e:
            logging.error(f"Oturum silme hatası: {session_id} - {str(e)}")
            return False
    
    def clean_expired_sessions(self) -> int:
        """
        Süresi dolmuş oturumları temizler.
        
        Returns:
            cleaned_count: Temizlenen oturum sayısı.
        """
        current_time = time.time()
        expired_sessions = []
        
        # Süresi dolmuş oturumları bul
        for session_id, session in self.sessions.items():
            if (current_time - session["last_active_time"]) > MAX_INACTIVE_SESSION_AGE:
                expired_sessions.append(session_id)
        
        # Süresi dolmuş oturumları sil
        for session_id in expired_sessions:
            self.delete_training_session(session_id)
        
        logging.info(f"{len(expired_sessions)} süresi dolmuş oturum temizlendi")
        return len(expired_sessions)
    
    # JNI köprü işlevleri
    
    def jni_create_training_session(
        self,
        model_id: str,
        training_config_java: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek eğitim oturumu oluşturma işlevi.
        
        Args:
            model_id: Eğitilecek modelin benzersiz tanımlayıcısı.
            training_config_java: Java formatındaki eğitim yapılandırması.
            
        Returns:
            result: İşlem sonucu ve oturum bilgisi.
        """
        # Java yapılandırmasını Python sözlüğüne dönüştür (JNI implementasyonunda)
        training_config = training_config_java
        
        # Güvenli çağrı ile oturum oluştur
        result = safe_jni_call(self.create_training_session, model_id, training_config)
        
        # Başarılı durumda, çıktıları Java formatına dönüştür
        if result["status"] == "success":
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result
    
    def jni_start_training(
        self,
        session_id: str,
        training_data_java: Dict[str, Any],
        callback_java: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek eğitim başlatma işlevi.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            training_data_java: Java formatındaki eğitim verileri.
            callback_java: Java geri çağrı nesnesi (opsiyonel).
            
        Returns:
            result: İşlem sonucu ve eğitim sonuçları.
        """
        # Java verilerini Python yapılarına dönüştür (JNI implementasyonunda)
        training_data = training_data_java
        
        # Java geri çağrısını PyTrainingCallback'e dönüştürecek adaptör
        python_callback = None
        if callback_java:
            # Burada Java callback'i Python callback'e dönüştürecek kod olacak
            # (JNI implementasyonunda)
            pass
        
        # Güvenli çağrı ile eğitimi başlat
        result = safe_jni_call(self.start_training, session_id, training_data, python_callback)
        
        # Başarılı durumda, çıktıları Java formatına dönüştür
        if result["status"] == "success":
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result
    
    def jni_get_training_status(self, session_id: str) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek eğitim durumu alma işlevi.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            
        Returns:
            result: İşlem sonucu ve durum bilgisi.
        """
        # Güvenli çağrı ile durumu al
        result = safe_jni_call(self.get_training_status, session_id)
        
        # Başarılı durumda, çıktıları Java formatına dönüştür
        if result["status"] == "success":
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result
    
    def jni_save_trained_model(
        self,
        session_id: str,
        output_path: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek eğitilmiş model kaydetme işlevi.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            output_path: Kaydedilecek dizin (opsiyonel).
            model_name: Kaydedilecek model adı (opsiyonel).
            
        Returns:
            result: İşlem sonucu ve kaydetme bilgisi.
        """
        # Güvenli çağrı ile modeli kaydet
        result = safe_jni_call(self.save_trained_model, session_id, output_path, model_name)
        
        # Başarılı durumda, çıktıları Java formatına dönüştür
        if result["status"] == "success":
            result["data"] = convert_dict_to_java_map(result["data"])
        
        return result
    
    def jni_delete_training_session(self, session_id: str) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek eğitim oturumu silme işlevi.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            
        Returns:
            result: İşlem sonucu.
        """
        return safe_jni_call(self.delete_training_session, session_id)
    
    def jni_clean_expired_sessions(self) -> Dict[str, Any]:
        """
        JNI üzerinden çağrılabilecek süresi dolmuş oturumları temizleme işlevi.
        
        Returns:
            result: İşlem sonucu ve temizlenen oturum sayısı.
        """
        return safe_jni_call(self.clean_expired_sessions)
    
    # Yardımcı metotlar
    
    def _get_validated_session(self, session_id: str) -> Dict[str, Any]:
        """
        Oturumu doğrular ve döndürür.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
            
        Returns:
            session: Oturum verileri.
            
        Raises:
            TrainingSessionNotFoundError: Oturum bulunamadığında.
            TrainingSessionExpiredError: Oturumun süresi dolduğunda.
        """
        if session_id not in self.sessions:
            # Oturum dosyasını kontrol et
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session = json.load(f)
                        self.sessions[session_id] = session
                except (json.JSONDecodeError, OSError) as e:
                    raise TrainingSessionNotFoundError(f"Oturum dosyası okunamadı: {str(e)}")
            else:
                raise TrainingSessionNotFoundError(f"Oturum bulunamadı: {session_id}")
        
        session = self.sessions[session_id]
        
        # Oturumun süresinin dolup dolmadığını kontrol et
        current_time = time.time()
        if (current_time - session["last_active_time"]) > MAX_INACTIVE_SESSION_AGE:
            raise TrainingSessionExpiredError(f"Oturumun süresi doldu: {session_id}")
        
        # Son aktif zamanı güncelle
        session["last_active_time"] = current_time
        
        return session
    
    def _save_session(self, session_id: str) -> None:
        """
        Oturum verilerini dosyaya kaydeder.
        
        Args:
            session_id: Eğitim oturumunun benzersiz tanımlayıcısı.
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Eğitilmiş modeli geçici olarak çıkar (JSON'a dönüştürülemez)
        trained_model = session.pop("trained_model", None)
        
        try:
            # Oturum dosyasını kaydet
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, ensure_ascii=False, indent=2)
        finally:
            # Eğitilmiş modeli geri ekle
            if trained_model is not None:
                session["trained_model"] = trained_model
    
    def _load_active_sessions(self) -> None:
        """
        Aktif oturumları dosyalardan yükler.
        """
        if not self.sessions_dir.exists():
            return
        
        current_time = time.time()
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                    
                    # Oturumun süresinin dolup dolmadığını kontrol et
                    if "last_active_time" in session and (current_time - session["last_active_time"]) <= MAX_INACTIVE_SESSION_AGE:
                        session_id = session.get("session_id")
                        if session_id:
                            self.sessions[session_id] = session
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Oturum dosyası okunamadı {session_file}: {e}")
    
    def _prepare_training_data(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Eğitim verilerini modelin anlayacağı formatta hazırlar.
        
        Args:
            training_data: Ham eğitim verileri.
            
        Returns:
            processed_data: İşlenen eğitim verileri.
        """
        # Burada uygulama özelinde veri hazırlığı yapılacak
        # Bu örnek implementasyonda veriler doğrudan geçiriliyor
        return training_data
    
    def _batch_data(self, data: Dict[str, Any], batch_size: int):
        """
        Verileri batch'lere böler.
        
        Args:
            data: Eğitim verileri.
            batch_size: Batch boyutu.
            
        Yields:
            batch: Veri batch'i.
        """
        # Burada uygulama özelinde batch oluşturma yapılacak
        # Örnek olarak, basit bir liste üzerinde batch oluşturma:
        
        if "samples" in data and isinstance(data["samples"], list):
            samples = data["samples"]
            for i in range(0, len(samples), batch_size):
                yield {"samples": samples[i:i + batch_size]}
        else:
            # Tek bir batch olarak kabul et
            yield data
    
    def _prepare_batch_inputs(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Batch verilerini model girişlerine dönüştürür.
        
        Args:
            batch_data: Batch verileri.
            
        Returns:
            model_inputs: Model girişleri.
        """
        # Burada uygulama özelinde giriş hazırlığı yapılacak
        # Bu örnek implementasyonda veriler doğrudan geçiriliyor
        return batch_data 