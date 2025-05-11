"""
M³TM Veri İndirme Modülü

Bu modül, verimli ve kesintiye dayanıklı veri indirme özellikleri sağlar.
Büyük model dosyalarını ve veri setlerini parçalı, öncelikli ve asenkron 
olarak indirme yeteneği sunar.

Örüntü: FactoryMethod (PT-002), ConfigurationDataclass (PT-001)
"""

from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import os
import time
import hashlib
import asyncio
import threading
import logging
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
import json
import shutil
from datetime import datetime

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class DownloadStatus(Enum):
    """İndirme işleminin durumunu belirten enum sınıfı."""
    PENDING = auto()      # İndirme kuyruğunda bekliyor
    CONNECTING = auto()   # Bağlantı kuruluyor
    DOWNLOADING = auto()  # İndirme devam ediyor
    PAUSED = auto()       # Kullanıcı tarafından duraklatıldı
    COMPLETED = auto()    # Başarıyla tamamlandı
    FAILED = auto()       # Hata nedeniyle başarısız oldu
    CANCELED = auto()     # Kullanıcı tarafından iptal edildi
    VERIFYING = auto()    # Bütünlük doğrulaması yapılıyor


class DownloadPriority(Enum):
    """İndirme işleminin öncelik seviyesini belirten enum sınıfı."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class DownloadError(Exception):
    """İndirme işlemi sırasında oluşan hataları temsil eden sınıf."""
    
    def __init__(self, message: str, code: int = 0, retry_after: Optional[int] = None):
        """
        Args:
            message: Hata mesajı.
            code: Hata kodu.
            retry_after: Yeniden deneme için bekleme süresi (saniye cinsinden).
        """
        self.message = message
        self.code = code
        self.retry_after = retry_after
        super().__init__(message)


@dataclass
class DownloadConfig:
    """
    İndirme yöneticisi yapılandırması.
    
    Attributes:
        download_dir: İndirilen dosyaların kaydedileceği dizin.
        chunk_size: İndirme için parça boyutu (bayt cinsinden).
        max_retries: Başarısız indirmeler için maksimum yeniden deneme sayısı.
        retry_delay: Yeniden denemeler arasındaki bekleme süresi (saniye cinsinden).
        max_concurrent_downloads: Aynı anda maksimum indirme sayısı.
        connection_timeout: Bağlantı zaman aşımı süresi (saniye cinsinden).
        read_timeout: Okuma zaman aşımı süresi (saniye cinsinden).
        use_async: Asenkron indirme kullanılıp kullanılmayacağı.
        bandwidth_limit: Bant genişliği sınırı (bayt/saniye, None sınırsız).
        verify_downloads: İndirme tamamlandığında doğrulama yapılıp yapılmayacağı.
        temp_extension: Geçici dosya uzantısı.
        metadata_dir: İndirme meta verilerinin saklanacağı dizin.
    """
    
    download_dir: str = "./downloads"
    chunk_size: int = 1024 * 1024  # 1 MB
    max_retries: int = 3
    retry_delay: int = 5
    max_concurrent_downloads: int = 3
    connection_timeout: int = 30
    read_timeout: int = 30
    use_async: bool = True
    bandwidth_limit: Optional[int] = None
    verify_downloads: bool = True
    temp_extension: str = ".part"
    metadata_dir: Optional[str] = None
    
    def __post_init__(self):
        """Yapılandırmayı doğrular ve gerekli dizinleri oluşturur."""
        # İndirme dizinini oluştur
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Metaveri dizinini ayarla
        if self.metadata_dir is None:
            self.metadata_dir = os.path.join(self.download_dir, ".metadata")
        
        # Metaveri dizinini oluştur
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Asenkron indirme için aiohttp kontrolü
        if self.use_async and not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp kütüphanesi bulunamadı, senkron indirme kullanılacak")
            self.use_async = False
            
        # Geçersiz değerler için kontrol
        if self.chunk_size <= 0:
            raise ValueError("chunk_size sıfırdan büyük olmalıdır")
            
        if self.max_concurrent_downloads <= 0:
            raise ValueError("max_concurrent_downloads sıfırdan büyük olmalıdır")
            
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout sıfırdan büyük olmalıdır")


@dataclass
class DownloadTask:
    """
    Bir indirme görevini temsil eden sınıf.
    
    Attributes:
        url: İndirilecek dosyanın URL'si.
        destination: İndirilen dosyanın kaydedileceği yol.
        filename: İndirilen dosyanın adı (None ise URL'den çıkarılır).
        priority: İndirme önceliği.
        expected_size: Beklenen dosya boyutu (bayt cinsinden).
        expected_hash: Beklenen dosya hash değeri (doğrulama için).
        hash_algorithm: Hash algoritması ('md5', 'sha1', 'sha256').
        metadata: İndirme ile ilişkili ek meta veriler.
        callback: İlerleme güncellemeleri için geri çağırma fonksiyonu.
        task_id: Benzersiz görev kimliği.
        status: Mevcut indirme durumu.
        progress: İndirme ilerlemesi (0.0-1.0).
        downloaded_bytes: İndirilen bayt sayısı.
        total_bytes: Toplam bayt sayısı.
        error: Hata durumunda hata mesajı.
        created_at: Görevin oluşturulma zamanı.
        started_at: İndirmenin başlama zamanı.
        completed_at: İndirmenin tamamlanma zamanı.
        attempts: Yapılan deneme sayısı.
    """
    
    url: str
    destination: Optional[str] = None
    filename: Optional[str] = None
    priority: DownloadPriority = DownloadPriority.NORMAL
    expected_size: Optional[int] = None
    expected_hash: Optional[str] = None
    hash_algorithm: str = "sha256"
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable[[float, DownloadStatus, Optional[str]], None]] = None
    
    # Aşağıdaki alanlar sistem tarafından yönetilir
    task_id: str = field(default_factory=lambda: f"task_{int(time.time() * 1000)}")
    status: DownloadStatus = DownloadStatus.PENDING
    progress: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: Optional[int] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempts: int = 0
    
    def __post_init__(self):
        """Görev oluşturulduğunda gerçekleştirilecek işlemler."""
        # Eğer dosya adı belirtilmemişse URL'den çıkar
        if self.filename is None:
            parsed_url = urllib.parse.urlparse(self.url)
            self.filename = os.path.basename(parsed_url.path)
            if not self.filename:
                self.filename = f"download_{self.task_id}"
        
        # Eğer hedef belirtilmemişse varsayılan indirme dizinini kullan
        if self.destination is None:
            self.destination = os.path.join("./downloads", self.filename)
    
    def to_dict(self) -> Dict[str, Any]:
        """Görevi serileştirilebilir bir sözlüğe dönüştürür."""
        return {
            "url": self.url,
            "destination": self.destination,
            "filename": self.filename,
            "priority": self.priority.name,
            "expected_size": self.expected_size,
            "expected_hash": self.expected_hash,
            "hash_algorithm": self.hash_algorithm,
            "metadata": self.metadata,
            "task_id": self.task_id,
            "status": self.status.name,
            "progress": self.progress,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DownloadTask':
        """Serileştirilmiş bir sözlükten görev nesnesi oluşturur."""
        task = cls(
            url=data["url"],
            destination=data.get("destination"),
            filename=data.get("filename"),
            priority=DownloadPriority[data.get("priority", "NORMAL")],
            expected_size=data.get("expected_size"),
            expected_hash=data.get("expected_hash"),
            hash_algorithm=data.get("hash_algorithm", "sha256"),
            metadata=data.get("metadata", {})
        )
        
        # Sistem tarafından yönetilen alanları güncelle
        task.task_id = data.get("task_id", task.task_id)
        task.status = DownloadStatus[data.get("status", "PENDING")]
        task.progress = data.get("progress", 0.0)
        task.downloaded_bytes = data.get("downloaded_bytes", 0)
        task.total_bytes = data.get("total_bytes")
        task.error = data.get("error")
        task.created_at = data.get("created_at", time.time())
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.attempts = data.get("attempts", 0)
        
        return task


@dataclass
class DownloadResult:
    """
    Tamamlanan bir indirme işleminin sonucunu temsil eden sınıf.
    
    Attributes:
        task_id: İlişkili görev kimliği.
        success: İndirmenin başarılı olup olmadığı.
        destination: İndirilen dosyanın tam yolu.
        size: İndirilen dosyanın boyutu (bayt cinsinden).
        file_hash: İndirilen dosyanın hash değeri.
        error: Başarısız olursa hata mesajı.
        duration: İndirme süresi (saniye cinsinden).
        verified: Dosyanın doğrulanıp doğrulanmadığı.
    """
    
    task_id: str
    success: bool
    destination: str
    size: int
    file_hash: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0
    verified: bool = False


class DownloadManager:
    """
    İndirme işlemlerini yöneten ana sınıf.
    
    Bu sınıf, indirme kuyruğu yönetimi, önceliklendirme, asenkron indirme,
    kesintiye dayanıklılık ve ilerleme takibi gibi özellikleri sağlar.
    """
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        """
        DownloadManager sınıfını başlatır.
        
        Args:
            config: İndirme yöneticisi yapılandırması.
        """
        self.config = config or DownloadConfig()
        
        # İndirme kuyruğu ve görev durumları
        self._queue = []  # Bekleme kuyruğu
        self._active_tasks = {}  # task_id -> DownloadTask
        self._completed_tasks = {}  # task_id -> DownloadResult
        
        # Kuyruk kilidi ve olaylar
        self._queue_lock = threading.RLock()
        self._download_semaphore = threading.Semaphore(self.config.max_concurrent_downloads)
        
        # Çalışan iş parçacığı ve olay döngüsü
        self._worker_thread = None
        self._running = False
        self._event_loop = None
        
        # Asenkron indirme için
        if self.config.use_async:
            self._async_sessions = {}  # task_id -> aiohttp.ClientSession
        
        # Durdurulan indirmelerin takibi için
        self._checkpoint_timer = None
        self._checkpoint_interval = 60  # 60 saniyede bir kaydetme
        
        # İndirme durumunu geri yükle
        self._restore_state()
    
    def start(self):
        """İndirme yöneticisini başlatır ve iş parçacığını oluşturur."""
        with self._queue_lock:
            if self._running:
                return
            
            self._running = True
            
            # İş parçacığını başlat
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="DownloadManagerWorker"
            )
            self._worker_thread.start()
            
            # Kontrol noktası zamanlayıcısını başlat
            self._start_checkpoint_timer()
            
            logger.info("İndirme yöneticisi başlatıldı")
    
    def stop(self):
        """İndirme yöneticisini durdurur ve tüm aktif indirmeleri duraklatır."""
        with self._queue_lock:
            if not self._running:
                return
            
            self._running = False
            
            # Tüm aktif indirmeleri duraklat
            for task_id, task in list(self._active_tasks.items()):
                if task.status == DownloadStatus.DOWNLOADING:
                    self.pause(task_id)
            
            # Kontrol noktası zamanlayıcısını durdur
            if self._checkpoint_timer:
                self._checkpoint_timer.cancel()
                self._checkpoint_timer = None
            
            # Son bir kaydetme yap
            self._save_state()
            
            logger.info("İndirme yöneticisi durduruldu")
    
    def add_task(self, task: DownloadTask) -> str:
        """
        İndirme kuyruğuna yeni bir görev ekler.
        
        Args:
            task: İndirme görevi.
            
        Returns:
            str: Görev kimliği.
        """
        with self._queue_lock:
            # Görevin dosya yolunu tam yola dönüştür
            if task.destination:
                task.destination = os.path.abspath(task.destination)
            
            # Eğer görev zaten kuyrukta ise tekrar ekleme
            for existing_task in self._queue:
                if existing_task.task_id == task.task_id:
                    return task.task_id
            
            # Eğer görev zaten aktif ise tekrar ekleme
            if task.task_id in self._active_tasks:
                return task.task_id
            
            # Eğer görev zaten tamamlanmış ise tekrar ekleme
            if task.task_id in self._completed_tasks:
                return task.task_id
            
            # Kuyruğa ekle ve önceliğe göre sırala
            self._queue.append(task)
            self._sort_queue()
            
            # Durumu kaydet
            self._save_task_metadata(task)
            
            # Yönetici başlatılmamışsa başlat
            if not self._running:
                self.start()
            
            logger.info(f"İndirme görevi eklendi: {task.task_id} ({task.filename})")
            
            return task.task_id
    
    def add_url(self, 
               url: str, 
               destination: Optional[str] = None,
               filename: Optional[str] = None,
               priority: DownloadPriority = DownloadPriority.NORMAL,
               expected_hash: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None,
               callback: Optional[Callable] = None) -> str:
        """
        URL'den bir indirme görevi oluşturur ve kuyruğa ekler.
        
        Args:
            url: İndirilecek dosyanın URL'si.
            destination: İndirilen dosyanın kaydedileceği yol.
            filename: İndirilen dosyanın adı.
            priority: İndirme önceliği.
            expected_hash: Beklenen dosya hash değeri.
            metadata: İndirme ile ilişkili ek meta veriler.
            callback: İlerleme güncellemeleri için geri çağırma fonksiyonu.
            
        Returns:
            str: Görev kimliği.
        """
        task = DownloadTask(
            url=url,
            destination=destination,
            filename=filename,
            priority=priority,
            expected_hash=expected_hash,
            metadata=metadata or {},
            callback=callback
        )
        
        return self.add_task(task)
    
    def get_task(self, task_id: str) -> Optional[DownloadTask]:
        """
        Belirtilen kimliğe sahip görevi döndürür.
        
        Args:
            task_id: Görev kimliği.
            
        Returns:
            Optional[DownloadTask]: Görev nesnesi, bulunamazsa None.
        """
        with self._queue_lock:
            # Aktif görevlerde ara
            if task_id in self._active_tasks:
                return self._active_tasks[task_id]
            
            # Kuyrukta ara
            for task in self._queue:
                if task.task_id == task_id:
                    return task
            
            # Tamamlanan görevlerde ara
            if task_id in self._completed_tasks:
                # Tamamlanan görevler için sonuç nesnesinden bir görev oluştur
                result = self._completed_tasks[task_id]
                task = DownloadTask(
                    url="",  # Tamamlanan görevler için URL'yi saklamamış olabiliriz
                    destination=result.destination,
                    task_id=task_id
                )
                task.status = (DownloadStatus.COMPLETED if result.success 
                             else DownloadStatus.FAILED)
                task.progress = 1.0 if result.success else 0.0
                task.downloaded_bytes = result.size
                task.total_bytes = result.size
                task.error = result.error
                task.completed_at = time.time()  # Tam zamanı bilmiyoruz
                return task
            
            # Metaveri dosyasında ara
            metadata_path = os.path.join(self.config.metadata_dir, f"{task_id}.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    return DownloadTask.from_dict(data)
                except Exception as e:
                    logger.error(f"Görev metaverileri yüklenirken hata: {e}")
            
            return None 

    def get_status(self, task_id: str) -> Optional[DownloadStatus]:
        """
        Belirtilen görevin durumunu döndürür.
        
        Args:
            task_id: Görev kimliği.
            
        Returns:
            Optional[DownloadStatus]: Görev durumu, bulunamazsa None.
        """
        task = self.get_task(task_id)
        return task.status if task else None
    
    def get_progress(self, task_id: str) -> Optional[float]:
        """
        Belirtilen görevin ilerleme durumunu döndürür.
        
        Args:
            task_id: Görev kimliği.
            
        Returns:
            Optional[float]: İlerleme (0.0-1.0), bulunamazsa None.
        """
        task = self.get_task(task_id)
        return task.progress if task else None
    
    def get_result(self, task_id: str) -> Optional[DownloadResult]:
        """
        Belirtilen görevin sonucunu döndürür.
        
        Args:
            task_id: Görev kimliği.
            
        Returns:
            Optional[DownloadResult]: İndirme sonucu, bulunamazsa None.
        """
        with self._queue_lock:
            return self._completed_tasks.get(task_id)
    
    def is_running(self) -> bool:
        """İndirme yöneticisinin çalışıp çalışmadığını döndürür."""
        return self._running
    
    def get_queue_length(self) -> int:
        """Bekleyen görev sayısını döndürür."""
        with self._queue_lock:
            return len(self._queue)
    
    def get_active_tasks_count(self) -> int:
        """Aktif görev sayısını döndürür."""
        with self._queue_lock:
            return len(self._active_tasks)
    
    def get_all_tasks(self) -> List[DownloadTask]:
        """Tüm görevleri (bekleyen, aktif, tamamlanan) döndürür."""
        with self._queue_lock:
            all_tasks = []
            
            # Kuyrukta bekleyen görevler
            all_tasks.extend(self._queue)
            
            # Aktif görevler
            all_tasks.extend(self._active_tasks.values())
            
            # Tamamlanan görevler
            for task_id, result in self._completed_tasks.items():
                task = DownloadTask(
                    url="",  # Tamamlanan görevler için URL'yi saklamamış olabiliriz
                    destination=result.destination,
                    task_id=task_id
                )
                task.status = (DownloadStatus.COMPLETED if result.success 
                            else DownloadStatus.FAILED)
                task.progress = 1.0 if result.success else 0.0
                task.downloaded_bytes = result.size
                task.total_bytes = result.size
                task.error = result.error
                task.completed_at = time.time()  # Tam zamanı bilmiyoruz
                all_tasks.append(task)
            
            return all_tasks
    
    def cancel(self, task_id: str) -> bool:
        """
        Belirtilen görevi iptal eder.
        
        Args:
            task_id: Görev kimliği.
            
        Returns:
            bool: İşlem başarılıysa True, değilse False.
        """
        with self._queue_lock:
            # Kuyrukta mı kontrol et
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    # Kuyruktan çıkar
                    self._queue.pop(i)
                    
                    # Durumu güncelle
                    task.status = DownloadStatus.CANCELED
                    task.error = "Kullanıcı tarafından iptal edildi"
                    
                    # Tamamlanan görevlere ekle
                    self._completed_tasks[task_id] = DownloadResult(
                        task_id=task_id,
                        success=False,
                        destination=task.destination,
                        size=task.downloaded_bytes,
                        error="Kullanıcı tarafından iptal edildi"
                    )
                    
                    # Metaveri dosyasını güncelle
                    self._save_task_metadata(task)
                    
                    logger.info(f"İndirme görevi iptal edildi: {task_id}")
                    
                    return True
            
            # Aktif görevler arasında mı kontrol et
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                
                # Asenkron indirmeyi iptal et
                if self.config.use_async and task_id in self._async_sessions:
                    session = self._async_sessions[task_id]
                    asyncio.run_coroutine_threadsafe(
                        session.close(), self._event_loop
                    )
                    del self._async_sessions[task_id]
                
                # Durumu güncelle
                task.status = DownloadStatus.CANCELED
                task.error = "Kullanıcı tarafından iptal edildi"
                
                # Aktif görevlerden çıkar
                del self._active_tasks[task_id]
                
                # Semaforu serbest bırak
                self._download_semaphore.release()
                
                # Tamamlanan görevlere ekle
                self._completed_tasks[task_id] = DownloadResult(
                    task_id=task_id,
                    success=False,
                    destination=task.destination,
                    size=task.downloaded_bytes,
                    error="Kullanıcı tarafından iptal edildi"
                )
                
                # Geçici dosyayı temizle
                temp_file = f"{task.destination}{self.config.temp_extension}"
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.error(f"Geçici dosya silinirken hata: {e}")
                
                # Metaveri dosyasını güncelle
                self._save_task_metadata(task)
                
                logger.info(f"Aktif indirme görevi iptal edildi: {task_id}")
                
                return True
            
            # Görev bulunamadı
            return False
    
    def pause(self, task_id: str) -> bool:
        """
        Belirtilen görevi duraklatır.
        
        Args:
            task_id: Görev kimliği.
            
        Returns:
            bool: İşlem başarılıysa True, değilse False.
        """
        with self._queue_lock:
            # Kuyrukta mı kontrol et
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    # Durumu güncelle
                    task.status = DownloadStatus.PAUSED
                    
                    # Metaveri dosyasını güncelle
                    self._save_task_metadata(task)
                    
                    logger.info(f"Kuyrukta bekleyen görev duraklatıldı: {task_id}")
                    
                    return True
            
            # Aktif görevler arasında mı kontrol et
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                
                # Asenkron indirmeyi duraklat (iptal et ve ilerlemeyi kaydet)
                if self.config.use_async and task_id in self._async_sessions:
                    session = self._async_sessions[task_id]
                    asyncio.run_coroutine_threadsafe(
                        session.close(), self._event_loop
                    )
                    del self._async_sessions[task_id]
                
                # Durumu güncelle
                task.status = DownloadStatus.PAUSED
                
                # Aktif görevlerden çıkar
                del self._active_tasks[task_id]
                
                # Semaforu serbest bırak
                self._download_semaphore.release()
                
                # Kuyruğa geri ekle
                self._queue.append(task)
                self._sort_queue()
                
                # Metaveri dosyasını güncelle
                self._save_task_metadata(task)
                
                logger.info(f"Aktif indirme görevi duraklatıldı: {task_id}")
                
                return True
            
            # Görev bulunamadı
            return False
    
    def resume(self, task_id: str) -> bool:
        """
        Duraklatılmış bir görevi sürdürür.
        
        Args:
            task_id: Görev kimliği.
            
        Returns:
            bool: İşlem başarılıysa True, değilse False.
        """
        with self._queue_lock:
            # Kuyrukta mı kontrol et
            for task in self._queue:
                if task.task_id == task_id and task.status == DownloadStatus.PAUSED:
                    # Durumu güncelle
                    task.status = DownloadStatus.PENDING
                    
                    # Kuyruğu yeniden sırala
                    self._sort_queue()
                    
                    # Metaveri dosyasını güncelle
                    self._save_task_metadata(task)
                    
                    logger.info(f"İndirme görevi sürdürüldü: {task_id}")
                    
                    return True
            
            # Tamamlanmış görevlerde ara ve yeniden kuyruğa ekle
            result = self._completed_tasks.get(task_id)
            if result and not result.success:
                # Metaveri dosyasından tam görevi yüklemeye çalış
                metadata_path = os.path.join(self.config.metadata_dir, f"{task_id}.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            data = json.load(f)
                        task = DownloadTask.from_dict(data)
                        
                        # Durumu güncelle
                        task.status = DownloadStatus.PENDING
                        task.error = None
                        
                        # Tamamlanan görevlerden çıkar
                        del self._completed_tasks[task_id]
                        
                        # Kuyruğa ekle
                        self._queue.append(task)
                        self._sort_queue()
                        
                        # Metaveri dosyasını güncelle
                        self._save_task_metadata(task)
                        
                        logger.info(f"Başarısız görev yeniden başlatıldı: {task_id}")
                        
                        return True
                    except Exception as e:
                        logger.error(f"Görev metaverileri yüklenirken hata: {e}")
            
            # Görev bulunamadı veya uygun durumda değil
            return False
    
    def set_priority(self, task_id: str, priority: DownloadPriority) -> bool:
        """
        Belirtilen görevin önceliğini günceller.
        
        Args:
            task_id: Görev kimliği.
            priority: Yeni öncelik seviyesi.
            
        Returns:
            bool: İşlem başarılıysa True, değilse False.
        """
        with self._queue_lock:
            # Kuyrukta mı kontrol et
            for task in self._queue:
                if task.task_id == task_id:
                    # Önceliği güncelle
                    task.priority = priority
                    
                    # Kuyruğu yeniden sırala
                    self._sort_queue()
                    
                    # Metaveri dosyasını güncelle
                    self._save_task_metadata(task)
                    
                    logger.info(
                        f"Görev önceliği güncellendi: {task_id} -> {priority.name}"
                    )
                    
                    return True
            
            # Aktif görevler arasında mı kontrol et
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                
                # Önceliği güncelle
                task.priority = priority
                
                # Metaveri dosyasını güncelle
                self._save_task_metadata(task)
                
                logger.info(
                    f"Aktif görev önceliği güncellendi: {task_id} -> {priority.name}"
                )
                
                return True
            
            # Görev bulunamadı
            return False
    
    def clear_completed(self) -> int:
        """
        Tamamlanan görevleri temizler.
        
        Returns:
            int: Temizlenen görev sayısı.
        """
        with self._queue_lock:
            count = len(self._completed_tasks)
            self._completed_tasks.clear()
            return count
    
    def wait_for_completion(self, task_id: str, timeout: Optional[float] = None) -> Optional[DownloadResult]:
        """
        Belirtilen görevin tamamlanmasını bekler.
        
        Args:
            task_id: Görev kimliği.
            timeout: Zaman aşımı süresi (saniye cinsinden).
            
        Returns:
            Optional[DownloadResult]: İndirme sonucu, zaman aşımı olursa None.
        """
        start_time = time.time()
        
        while True:
            # Zaman aşımı kontrolü
            if timeout is not None and time.time() - start_time > timeout:
                return None
            
            # Görevi kontrol et
            with self._queue_lock:
                # Tamamlandı mı?
                if task_id in self._completed_tasks:
                    return self._completed_tasks[task_id]
                
                # Kuyrukta veya aktif mi?
                found = False
                for task in self._queue:
                    if task.task_id == task_id:
                        found = True
                        break
                
                if task_id in self._active_tasks:
                    found = True
                
                if not found:
                    # Görev bulunamadı
                    return None
            
            # Kısa bir süre bekle
            time.sleep(0.1)
    
    def wait_for_all(self, timeout: Optional[float] = None) -> bool:
        """
        Tüm görevlerin tamamlanmasını bekler.
        
        Args:
            timeout: Zaman aşımı süresi (saniye cinsinden).
            
        Returns:
            bool: Tüm görevler tamamlandıysa True, zaman aşımı olursa False.
        """
        start_time = time.time()
        
        while True:
            # Zaman aşımı kontrolü
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            # Kuyruğu ve aktif görevleri kontrol et
            with self._queue_lock:
                if not self._queue and not self._active_tasks:
                    return True
            
            # Kısa bir süre bekle
            time.sleep(0.1) 

    def _sort_queue(self):
        """Kuyruğu önceliğe göre sıralar."""
        self._queue.sort(key=lambda task: (
            -task.priority.value,  # Yüksek öncelikli görevler önce
            task.created_at  # Aynı öncelikli görevler arasında eski olanlar önce
        ))
    
    def _save_task_metadata(self, task: DownloadTask):
        """Görev metaverilerini dosyaya kaydeder."""
        metadata_path = os.path.join(self.config.metadata_dir, f"{task.task_id}.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(task.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Görev metaverileri kaydedilirken hata: {e}")
    
    def _save_state(self):
        """Yönetici durumunu kaydeder."""
        # Görevleri kaydet
        for task in self._queue:
            self._save_task_metadata(task)
        
        for task in self._active_tasks.values():
            self._save_task_metadata(task)
        
        # Durum dosyasını kaydet
        state_path = os.path.join(self.config.metadata_dir, "download_manager_state.json")
        try:
            state = {
                "queue": [task.task_id for task in self._queue],
                "active_tasks": list(self._active_tasks.keys()),
                "completed_tasks": {
                    task_id: {
                        "task_id": result.task_id,
                        "success": result.success,
                        "destination": result.destination,
                        "size": result.size,
                        "file_hash": result.file_hash,
                        "error": result.error,
                        "duration": result.duration,
                        "verified": result.verified
                    }
                    for task_id, result in self._completed_tasks.items()
                }
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Yönetici durumu kaydedilirken hata: {e}")
    
    def _restore_state(self):
        """Yönetici durumunu geri yükler."""
        state_path = os.path.join(self.config.metadata_dir, "download_manager_state.json")
        if not os.path.exists(state_path):
            return
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Görevleri yükle
            for task_id in state.get("queue", []):
                metadata_path = os.path.join(self.config.metadata_dir, f"{task_id}.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            data = json.load(f)
                        task = DownloadTask.from_dict(data)
                        self._queue.append(task)
                    except Exception as e:
                        logger.error(f"Görev metaverileri yüklenirken hata: {e}")
            
            # Sırala
            self._sort_queue()
            
            # Tamamlanan görevleri yükle
            for task_id, result_data in state.get("completed_tasks", {}).items():
                self._completed_tasks[task_id] = DownloadResult(
                    task_id=result_data.get("task_id", task_id),
                    success=result_data.get("success", False),
                    destination=result_data.get("destination", ""),
                    size=result_data.get("size", 0),
                    file_hash=result_data.get("file_hash"),
                    error=result_data.get("error"),
                    duration=result_data.get("duration", 0.0),
                    verified=result_data.get("verified", False)
                )
        except Exception as e:
            logger.error(f"Yönetici durumu geri yüklenirken hata: {e}")
    
    def _start_checkpoint_timer(self):
        """Düzenli aralıklarla durum kaydetme zamanlayıcısını başlatır."""
        if self._checkpoint_timer:
            self._checkpoint_timer.cancel()
        
        self._checkpoint_timer = threading.Timer(self._checkpoint_interval, self._checkpoint_task)
        self._checkpoint_timer.daemon = True
        self._checkpoint_timer.start()
    
    def _checkpoint_task(self):
        """Zamanlayıcı tarafından çağrılan kontrol noktası görevi."""
        with self._queue_lock:
            if not self._running:
                return
            
            self._save_state()
            
            # Zamanlayıcıyı yeniden başlat
            self._start_checkpoint_timer()
    
    def _worker_loop(self):
        """İş parçacığının ana döngüsü."""
        # Asenkron modda çalışıyorsa olay döngüsü oluştur
        if self.config.use_async:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        while self._running:
            try:
                # Semaforu al (eğer max_concurrent_downloads sınırına ulaşıldıysa bekler)
                self._download_semaphore.acquire()
                
                # Yönetici çalışmıyorsa çık
                if not self._running:
                    self._download_semaphore.release()
                    break
                
                # Kuyruktan sonraki görevi al
                task = None
                with self._queue_lock:
                    if self._queue:
                        task = self._queue.pop(0)
                
                # Görev yoksa semaforu serbest bırak ve bekle
                if not task:
                    self._download_semaphore.release()
                    time.sleep(0.1)
                    continue
                
                # Görevi aktif görevlere ekle
                with self._queue_lock:
                    self._active_tasks[task.task_id] = task
                
                # Görevi başlat
                if self.config.use_async:
                    # Asenkron indirme
                    future = asyncio.run_coroutine_threadsafe(
                        self._download_async(task), self._event_loop
                    )
                    
                    # İndirme sonucunu beklemeden devam et
                else:
                    # Senkron indirme (ayrı bir iş parçacığında çalıştır)
                    download_thread = threading.Thread(
                        target=self._download_sync,
                        args=(task,),
                        daemon=True,
                        name=f"DownloadThread-{task.task_id}"
                    )
                    download_thread.start()
            
            except Exception as e:
                logger.error(f"İş parçacığı döngüsünde hata: {e}")
                # Semaforu serbest bırak (eğer hata nedeniyle alındıysa)
                try:
                    self._download_semaphore.release()
                except ValueError:
                    pass
                time.sleep(1)  # Hatalardan sonra biraz bekle
        
        # Olay döngüsünü kapat
        if self.config.use_async and self._event_loop:
            for session in list(self._async_sessions.values()):
                asyncio.run_coroutine_threadsafe(session.close(), self._event_loop)
            
            self._event_loop.stop()
    
    async def _download_async(self, task: DownloadTask):
        """
        Asenkron indirme görevi.
        
        Args:
            task: İndirme görevi.
        """
        result = None
        
        try:
            # Durumu güncelle
            task.status = DownloadStatus.CONNECTING
            task.started_at = time.time()
            task.attempts += 1
            
            # İlerleme geri çağırma fonksiyonunu çağır
            if task.callback:
                task.callback(task.progress, task.status, None)
            
            # Hedef dizini oluştur
            os.makedirs(os.path.dirname(os.path.abspath(task.destination)), exist_ok=True)
            
            # Geçici dosya adı
            temp_file = f"{task.destination}{self.config.temp_extension}"
            
            # Mevcut indirme kontrolü
            resume_position = 0
            if os.path.exists(temp_file):
                resume_position = os.path.getsize(temp_file)
                logger.info(f"Mevcut indirmeye devam ediliyor: {task.destination} ({resume_position} bayt)")
            
            # aiohttp oturumu oluştur
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    connect=self.config.connection_timeout,
                    sock_read=self.config.read_timeout
                )
            )
            self._async_sessions[task.task_id] = session
            
            # İndirme başlat
            headers = {}
            if resume_position > 0:
                headers["Range"] = f"bytes={resume_position}-"
            
            async with session.get(task.url, headers=headers) as response:
                if not (200 <= response.status < 300 or response.status == 206):
                    raise DownloadError(
                        f"HTTP hatası: {response.status}",
                        code=response.status
                    )
                
                # Toplam boyutu al
                total_size = task.expected_size
                if "Content-Length" in response.headers:
                    try:
                        content_length = int(response.headers["Content-Length"])
                        if resume_position > 0:
                            total_size = resume_position + content_length
                        else:
                            total_size = content_length
                    except (ValueError, TypeError):
                        pass
                
                task.total_bytes = total_size
                
                # Hash hesaplayıcı oluştur
                hasher = None
                if self.config.verify_downloads and task.expected_hash:
                    hasher = hashlib.new(task.hash_algorithm)
                    
                    # Eğer devam ediyorsak, mevcut dosyanın hash'ini hesapla
                    if resume_position > 0:
                        with open(temp_file, 'rb') as f:
                            while chunk := f.read(self.config.chunk_size):
                                hasher.update(chunk)
                
                # Dosyayı aç (devam modunda)
                with open(temp_file, 'ab') as f:
                    # İndirme durumunu güncelle
                    task.status = DownloadStatus.DOWNLOADING
                    task.downloaded_bytes = resume_position
                    
                    if total_size:
                        task.progress = resume_position / total_size
                    
                    # İlerleme geri çağırma fonksiyonunu çağır
                    if task.callback:
                        task.callback(task.progress, task.status, None)
                    
                    # Verileri chunk olarak indir
                    start_time = time.time()
                    last_update_time = start_time
                    
                    async for chunk in response.content.iter_chunked(self.config.chunk_size):
                        # İşlemi durdur kontrolü
                        if task.status != DownloadStatus.DOWNLOADING:
                            raise DownloadError("İndirme duraklatıldı veya iptal edildi")
                        
                        # Bant genişliği sınırı kontrolü
                        if self.config.bandwidth_limit:
                            chunk_size = len(chunk)
                            expected_time = chunk_size / self.config.bandwidth_limit
                            elapsed = time.time() - last_update_time
                            if elapsed < expected_time:
                                await asyncio.sleep(expected_time - elapsed)
                        
                        # Chunk'ı yaz
                        f.write(chunk)
                        
                        # Hash güncelle
                        if hasher:
                            hasher.update(chunk)
                        
                        # İlerlemeyi güncelle
                        chunk_size = len(chunk)
                        task.downloaded_bytes += chunk_size
                        
                        if total_size:
                            task.progress = min(1.0, task.downloaded_bytes / total_size)
                        
                        # Belirli aralıklarla ilerleme bildirimi
                        current_time = time.time()
                        if current_time - last_update_time > 0.5:  # 500ms'de bir güncelle
                            last_update_time = current_time
                            
                            # İlerleme geri çağırma fonksiyonunu çağır
                            if task.callback:
                                task.callback(task.progress, task.status, None)
                            
                            # Metaveri dosyasını güncelle
                            self._save_task_metadata(task)
                
                # İndirme tamamlandı, doğrulama yap
                task.status = DownloadStatus.VERIFYING
                task.progress = 1.0
                
                # İlerleme geri çağırma fonksiyonunu çağır
                if task.callback:
                    task.callback(task.progress, task.status, None)
                
                # Hash doğrulama
                verified = True
                file_hash = None
                
                if hasher:
                    file_hash = hasher.hexdigest()
                    
                    if task.expected_hash and file_hash != task.expected_hash:
                        raise DownloadError(
                            f"Hash doğrulama hatası: Beklenen {task.expected_hash}, "
                            f"Alınan {file_hash}"
                        )
                
                # İndirme süresini hesapla
                end_time = time.time()
                duration = end_time - start_time
                
                # Geçici dosyayı hedef dosyaya taşı
                shutil.move(temp_file, task.destination)
                
                # İndirme tamamlandı
                task.status = DownloadStatus.COMPLETED
                task.completed_at = end_time
                
                # Başarılı sonuç oluştur
                result = DownloadResult(
                    task_id=task.task_id,
                    success=True,
                    destination=task.destination,
                    size=task.downloaded_bytes,
                    file_hash=file_hash,
                    duration=duration,
                    verified=verified
                )
                
                logger.info(f"İndirme tamamlandı: {task.destination} ({task.downloaded_bytes} bayt)")
        
        except asyncio.CancelledError:
            # İndirme iptal edildi
            logger.info(f"İndirme iptal edildi: {task.destination}")
            
            result = DownloadResult(
                task_id=task.task_id,
                success=False,
                destination=task.destination,
                size=task.downloaded_bytes,
                error="İndirme iptal edildi"
            )
        
        except DownloadError as e:
            # İndirme hatası
            logger.error(f"İndirme hatası: {task.destination} - {e}")
            
            # Yeniden deneme kontrolü
            if task.attempts < self.config.max_retries:
                # Görevi kuyruğa geri ekle
                with self._queue_lock:
                    task.status = DownloadStatus.PENDING
                    task.error = str(e)
                    self._queue.append(task)
                    self._sort_queue()
                
                logger.info(f"İndirme yeniden denenecek ({task.attempts}/{self.config.max_retries}): {task.destination}")
            else:
                # Maksimum yeniden deneme sayısına ulaşıldı
                task.status = DownloadStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                
                result = DownloadResult(
                    task_id=task.task_id,
                    success=False,
                    destination=task.destination,
                    size=task.downloaded_bytes,
                    error=str(e)
                )
                
                logger.error(f"İndirme başarısız (yeniden deneme limiti aşıldı): {task.destination}")
        
        except Exception as e:
            # Diğer hatalar
            logger.error(f"İndirme sırasında beklenmeyen hata: {task.destination} - {e}")
            
            task.status = DownloadStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            result = DownloadResult(
                task_id=task.task_id,
                success=False,
                destination=task.destination,
                size=task.downloaded_bytes,
                error=str(e)
            )
        
        finally:
            # Oturumu kapat
            if task.task_id in self._async_sessions:
                try:
                    await self._async_sessions[task.task_id].close()
                except Exception:
                    pass
                del self._async_sessions[task.task_id]
            
            # Görev durumunu güncelle
            with self._queue_lock:
                # Aktif görevlerden çıkar
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                
                # Sonucu kaydet
                if result:
                    self._completed_tasks[task.task_id] = result
                
                # Metaveri dosyasını güncelle
                self._save_task_metadata(task)
            
            # İlerleme geri çağırma fonksiyonunu son kez çağır
            if task.callback:
                task.callback(task.progress, task.status, task.error)
            
            # Semaforu serbest bırak
            self._download_semaphore.release()
    
    def _download_sync(self, task: DownloadTask):
        """
        Senkron indirme görevi.
        
        Args:
            task: İndirme görevi.
        """
        result = None
        
        try:
            # Durumu güncelle
            task.status = DownloadStatus.CONNECTING
            task.started_at = time.time()
            task.attempts += 1
            
            # İlerleme geri çağırma fonksiyonunu çağır
            if task.callback:
                task.callback(task.progress, task.status, None)
            
            # Hedef dizini oluştur
            os.makedirs(os.path.dirname(os.path.abspath(task.destination)), exist_ok=True)
            
            # Geçici dosya adı
            temp_file = f"{task.destination}{self.config.temp_extension}"
            
            # Mevcut indirme kontrolü
            resume_position = 0
            if os.path.exists(temp_file):
                resume_position = os.path.getsize(temp_file)
                logger.info(f"Mevcut indirmeye devam ediliyor: {task.destination} ({resume_position} bayt)")
            
            # HTTP isteği oluştur
            headers = {}
            if resume_position > 0:
                headers["Range"] = f"bytes={resume_position}-"
            
            request = urllib.request.Request(task.url, headers=headers)
            
            # İndirme başlat
            with urllib.request.urlopen(
                request, 
                timeout=max(self.config.connection_timeout, self.config.read_timeout)
            ) as response:
                # Toplam boyutu al
                total_size = task.expected_size
                if "Content-Length" in response.headers:
                    try:
                        content_length = int(response.headers["Content-Length"])
                        if resume_position > 0:
                            total_size = resume_position + content_length
                        else:
                            total_size = content_length
                    except (ValueError, TypeError):
                        pass
                
                task.total_bytes = total_size
                
                # Hash hesaplayıcı oluştur
                hasher = None
                if self.config.verify_downloads and task.expected_hash:
                    hasher = hashlib.new(task.hash_algorithm)
                    
                    # Eğer devam ediyorsak, mevcut dosyanın hash'ini hesapla
                    if resume_position > 0:
                        with open(temp_file, 'rb') as f:
                            while chunk := f.read(self.config.chunk_size):
                                hasher.update(chunk)
                
                # Dosyayı aç (devam modunda)
                with open(temp_file, 'ab') as f:
                    # İndirme durumunu güncelle
                    task.status = DownloadStatus.DOWNLOADING
                    task.downloaded_bytes = resume_position
                    
                    if total_size:
                        task.progress = resume_position / total_size
                    
                    # İlerleme geri çağırma fonksiyonunu çağır
                    if task.callback:
                        task.callback(task.progress, task.status, None)
                    
                    # Verileri chunk olarak indir
                    start_time = time.time()
                    last_update_time = start_time
                    
                    while True:
                        # İşlemi durdur kontrolü
                        if task.status != DownloadStatus.DOWNLOADING:
                            raise DownloadError("İndirme duraklatıldı veya iptal edildi")
                        
                        # Chunk oku
                        chunk = response.read(self.config.chunk_size)
                        if not chunk:
                            break
                        
                        # Bant genişliği sınırı kontrolü
                        if self.config.bandwidth_limit:
                            chunk_size = len(chunk)
                            expected_time = chunk_size / self.config.bandwidth_limit
                            elapsed = time.time() - last_update_time
                            if elapsed < expected_time:
                                time.sleep(expected_time - elapsed)
                        
                        # Chunk'ı yaz
                        f.write(chunk)
                        
                        # Hash güncelle
                        if hasher:
                            hasher.update(chunk)
                        
                        # İlerlemeyi güncelle
                        chunk_size = len(chunk)
                        task.downloaded_bytes += chunk_size
                        
                        if total_size:
                            task.progress = min(1.0, task.downloaded_bytes / total_size)
                        
                        # Belirli aralıklarla ilerleme bildirimi
                        current_time = time.time()
                        if current_time - last_update_time > 0.5:  # 500ms'de bir güncelle
                            last_update_time = current_time
                            
                            # İlerleme geri çağırma fonksiyonunu çağır
                            if task.callback:
                                task.callback(task.progress, task.status, None)
                            
                            # Metaveri dosyasını güncelle
                            self._save_task_metadata(task)
                
                # İndirme tamamlandı, doğrulama yap
                task.status = DownloadStatus.VERIFYING
                task.progress = 1.0
                
                # İlerleme geri çağırma fonksiyonunu çağır
                if task.callback:
                    task.callback(task.progress, task.status, None)
                
                # Hash doğrulama
                verified = True
                file_hash = None
                
                if hasher:
                    file_hash = hasher.hexdigest()
                    
                    if task.expected_hash and file_hash != task.expected_hash:
                        raise DownloadError(
                            f"Hash doğrulama hatası: Beklenen {task.expected_hash}, "
                            f"Alınan {file_hash}"
                        )
                
                # İndirme süresini hesapla
                end_time = time.time()
                duration = end_time - start_time
                
                # Geçici dosyayı hedef dosyaya taşı
                shutil.move(temp_file, task.destination)
                
                # İndirme tamamlandı
                task.status = DownloadStatus.COMPLETED
                task.completed_at = end_time
                
                # Başarılı sonuç oluştur
                result = DownloadResult(
                    task_id=task.task_id,
                    success=True,
                    destination=task.destination,
                    size=task.downloaded_bytes,
                    file_hash=file_hash,
                    duration=duration,
                    verified=verified
                )
                
                logger.info(f"İndirme tamamlandı: {task.destination} ({task.downloaded_bytes} bayt)")
        
        except DownloadError as e:
            # İndirme hatası
            logger.error(f"İndirme hatası: {task.destination} - {e}")
            
            # Yeniden deneme kontrolü
            if task.attempts < self.config.max_retries:
                # Görevi kuyruğa geri ekle
                with self._queue_lock:
                    task.status = DownloadStatus.PENDING
                    task.error = str(e)
                    self._queue.append(task)
                    self._sort_queue()
                
                logger.info(f"İndirme yeniden denenecek ({task.attempts}/{self.config.max_retries}): {task.destination}")
            else:
                # Maksimum yeniden deneme sayısına ulaşıldı
                task.status = DownloadStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
                
                result = DownloadResult(
                    task_id=task.task_id,
                    success=False,
                    destination=task.destination,
                    size=task.downloaded_bytes,
                    error=str(e)
                )
                
                logger.error(f"İndirme başarısız (yeniden deneme limiti aşıldı): {task.destination}")
        
        except Exception as e:
            # Diğer hatalar
            logger.error(f"İndirme sırasında beklenmeyen hata: {task.destination} - {e}")
            
            task.status = DownloadStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            result = DownloadResult(
                task_id=task.task_id,
                success=False,
                destination=task.destination,
                size=task.downloaded_bytes,
                error=str(e)
            )
        
        finally:
            # Görev durumunu güncelle
            with self._queue_lock:
                # Aktif görevlerden çıkar
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                
                # Sonucu kaydet
                if result:
                    self._completed_tasks[task.task_id] = result
                
                # Metaveri dosyasını güncelle
                self._save_task_metadata(task)
            
            # İlerleme geri çağırma fonksiyonunu son kez çağır
            if task.callback:
                task.callback(task.progress, task.status, task.error)
            
            # Semaforu serbest bırak
            self._download_semaphore.release()


class DownloadManagerFactory:
    """
    DownloadManager fabrika sınıfı.
    
    Örüntü: FactoryMethod (PT-002)
    """
    
    @staticmethod
    def create(
        download_dir: str = "./downloads",
        chunk_size: int = 1024 * 1024,
        max_retries: int = 3,
        max_concurrent_downloads: int = 3,
        use_async: bool = True,
        bandwidth_limit: Optional[int] = None,
        verify_downloads: bool = True
    ) -> DownloadManager:
        """
        Yeni bir DownloadManager örneği oluşturur.
        
        Args:
            download_dir: İndirilen dosyaların kaydedileceği dizin.
            chunk_size: İndirme için parça boyutu (bayt cinsinden).
            max_retries: Başarısız indirmeler için maksimum yeniden deneme sayısı.
            max_concurrent_downloads: Aynı anda maksimum indirme sayısı.
            use_async: Asenkron indirme kullanılıp kullanılmayacağı.
            bandwidth_limit: Bant genişliği sınırı (bayt/saniye, None sınırsız).
            verify_downloads: İndirme tamamlandığında doğrulama yapılıp yapılmayacağı.
            
        Returns:
            DownloadManager: İndirme yöneticisi.
        """
        config = DownloadConfig(
            download_dir=download_dir,
            chunk_size=chunk_size,
            max_retries=max_retries,
            max_concurrent_downloads=max_concurrent_downloads,
            use_async=use_async,
            bandwidth_limit=bandwidth_limit,
            verify_downloads=verify_downloads
        )
        
        return DownloadManager(config) 