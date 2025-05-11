"""
M³TM Veri İndirme Modülü Örneği

Bu örnek, M³TM veri indirme modülünün temel kullanımını gösterir.
"""

import sys
import os
import time
import logging
from typing import Dict, Any

# Modül yolunu ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from m3tm.download import (
    DownloadManagerFactory,
    DownloadStatus,
    DownloadPriority,
    DownloadTask
)

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# İlerleme bildirimi için callback fonksiyonu
def progress_callback(progress: float, status: DownloadStatus, error: str):
    """İndirme ilerlemesini ekrana yazdırır."""
    status_str = status.name if status else "UNKNOWN"
    progress_percent = int(progress * 100)
    
    error_str = f" - Hata: {error}" if error else ""
    
    print(f"\rİlerleme: [{progress_percent:3d}%] Durum: {status_str}{error_str}", end='')
    
    if status in [DownloadStatus.COMPLETED, DownloadStatus.FAILED, DownloadStatus.CANCELED]:
        print()  # Satır sonu


def main():
    """Örnek uygulama."""
    print("M³TM Veri İndirme Modülü Örneği")
    print("--------------------------------")
    
    # İndirme yöneticisi oluştur
    download_manager = DownloadManagerFactory.create(
        download_dir="./downloads",
        max_concurrent_downloads=2,
        verify_downloads=True
    )
    
    # İndirme yöneticisini başlat
    download_manager.start()
    
    try:
        # Örnek URL'ler
        urls = [
            # Küçük dosya (normal öncelik)
            "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md",
            
            # Orta boyutlu dosya (yüksek öncelik)
            "https://github.com/pytorch/data/raw/main/LICENSE",
            
            # Büyük dosya (düşük öncelik)
            "https://speed.hetzner.de/100MB.bin"
        ]
        
        # Görevleri ekle
        tasks = []
        
        # İlk görev: Normal öncelikli
        task_id1 = download_manager.add_url(
            url=urls[0],
            filename="pytorch_readme.md",
            callback=progress_callback
        )
        tasks.append(task_id1)
        print(f"Görev eklendi (normal öncelik): {task_id1}")
        
        # İkinci görev: Yüksek öncelikli
        task_id2 = download_manager.add_url(
            url=urls[1],
            filename="pytorch_license.txt",
            priority=DownloadPriority.HIGH,
            callback=progress_callback
        )
        tasks.append(task_id2)
        print(f"Görev eklendi (yüksek öncelik): {task_id2}")
        
        # İlk iki indirmenin tamamlanmasını bekle
        for task_id in tasks:
            result = download_manager.wait_for_completion(task_id)
            
            if result and result.success:
                print(f"İndirme başarıyla tamamlandı: {result.destination}")
                print(f"  Boyut: {result.size} bayt")
                print(f"  Süre: {result.duration:.2f} saniye")
                print(f"  Doğrulandı: {result.verified}")
                if result.file_hash:
                    print(f"  Hash: {result.file_hash}")
            else:
                error = result.error if result else "Bilinmeyen hata"
                print(f"İndirme başarısız: {error}")
        
        # Büyük dosya indirme (öncelik düşük)
        task_id3 = download_manager.add_url(
            url=urls[2],
            filename="large_file.bin",
            priority=DownloadPriority.LOW,
            callback=progress_callback
        )
        print(f"Görev eklendi (düşük öncelik): {task_id3}")
        
        # 3 saniye bekle ve durumu kontrol et
        time.sleep(3)
        
        # Durumu görüntüle
        task = download_manager.get_task(task_id3)
        if task:
            print("\nBüyük dosya indirme durumu:")
            print(f"  Durum: {task.status.name}")
            print(f"  İndirilen: {task.downloaded_bytes} / {task.total_bytes} bayt")
            print(f"  İlerleme: %{int(task.progress * 100)}")
        
        # İndirmeyi duraklat
        if download_manager.pause(task_id3):
            print("\nBüyük dosya indirmesi duraklatıldı")
            
            # 2 saniye bekle
            time.sleep(2)
            
            # İndirmeyi sürdür
            if download_manager.resume(task_id3):
                print("Büyük dosya indirmesi sürdürülüyor")
        
        # Biraz daha bekle
        time.sleep(3)
        
        # İndirmeyi iptal et
        if download_manager.cancel(task_id3):
            print("\nBüyük dosya indirmesi iptal edildi")
        
        # Tüm görevlerin durumunu listele
        print("\nTüm görevlerin durumu:")
        for task in download_manager.get_all_tasks():
            status = task.status.name
            progress = int(task.progress * 100)
            print(f"  {task.task_id}: {task.filename} - Durum: {status}, İlerleme: %{progress}")
        
    finally:
        # İndirme yöneticisini durdur
        download_manager.stop()
        print("\nİndirme yöneticisi durduruldu")


if __name__ == "__main__":
    main() 