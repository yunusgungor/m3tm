"""
M³TM Veri İndirme Modülü

Bu modül, M³TM modelinin mobil cihazlarda verimli ve kesintiye dayanıklı şekilde
veri indirme yeteneklerini sağlar. Büyük model dosyalarını ve veri setlerini
parçalı, öncelikli ve asenkron olarak indirme fonksiyonalitesini içerir.
"""

from .downloader import (
    DownloadConfig,
    DownloadStatus,
    DownloadPriority,
    DownloadManager,
    DownloadTask,
    DownloadResult,
    DownloadError,
    DownloadManagerFactory
)

__all__ = [
    'DownloadConfig',
    'DownloadStatus',
    'DownloadPriority',
    'DownloadManager',
    'DownloadTask',
    'DownloadResult',
    'DownloadError',
    'DownloadManagerFactory'
] 