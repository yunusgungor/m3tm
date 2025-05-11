"""
M³TM Arama İndeksleme Modülü

Bu modül, model çıktılarını verimli bir şekilde indeksleyen ve semantik arama
yapabilen SearchIndex sınıfını içerir. FAISS kütüphanesini kullanarak hem metin
hem de görüntü gömmelerini indeksler ve KNN araması gerçekleştirir.

Örüntü: ModelComposite (PT-003), FactoryMethod (PT-002), ConfigurationDataclass (PT-001)
"""

from typing import Optional, Dict, Any, Union, List, Tuple, Callable
import os
import json
import time
import threading
import logging
from pathlib import Path
from dataclasses import dataclass, field

import torch
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


@dataclass
class SearchIndexConfig:
    """
    Arama indeksi yapılandırması.
    
    Attributes:
        embedding_dim (int): Gömme vektörlerinin boyutu.
        index_type (str): İndeks türü ('flat', 'hnsw', 'ivf').
        metric_type (str): Benzerlik metriği ('l2', 'ip' [iç çarpım], 'cosine').
        max_index_size (int): İndeksin maksimum büyüklüğü (bellek yönetimi için).
        use_gpu (bool): GPU kullanılıp kullanılmayacağı.
        storage_path (str): İndeksin kalıcı depolamaya kaydedileceği dizin.
        compression_level (int): Vektör sıkıştırma seviyesi (0-255).
        hnsw_store_n (int): HNSW indeksi için komşu sayısı.
        ivf_nlist (int): IVF indeksi için liste sayısı.
        batch_size (int): Toplu indeksleme için parti boyutu.
    """
    
    embedding_dim: int = 128
    index_type: str = "flat"
    metric_type: str = "cosine"
    max_index_size: int = 1000000
    use_gpu: bool = False
    storage_path: Optional[str] = None
    compression_level: int = 0
    hnsw_store_n: int = 32
    ivf_nlist: int = 100
    batch_size: int = 128
    
    def __post_init__(self):
        """Yapılandırmayı doğrular ve gerekli ayarlamaları yapar."""
        # İndeks türünü kontrol et
        valid_index_types = ["flat", "hnsw", "ivf"]
        if self.index_type not in valid_index_types:
            raise ValueError(f"index_type '{self.index_type}' geçerli değil. "
                            f"Geçerli değerler: {valid_index_types}")
        
        # Metrik türünü kontrol et
        valid_metric_types = ["l2", "ip", "cosine"]
        if self.metric_type not in valid_metric_types:
            raise ValueError(f"metric_type '{self.metric_type}' geçerli değil. "
                            f"Geçerli değerler: {valid_metric_types}")
        
        # GPU kullanımını kontrol et
        if self.use_gpu and faiss is not None and not hasattr(faiss, 'StandardGpuResources'):
            logger.warning("GPU desteği ile derlenen FAISS bulunamadı. CPU kullanılacak.")
            self.use_gpu = False
        
        # Depolama yolunu kontrol et
        if self.storage_path is not None:
            self.storage_path = os.path.expanduser(self.storage_path)
            os.makedirs(self.storage_path, exist_ok=True)


class SearchIndex:
    """
    Vektör gömmeleri için arama indeksi sınıfı.
    
    Bu sınıf, model tarafından üretilen gömmeleri FAISS kütüphanesini kullanarak
    indeksler ve verimli bir şekilde benzerlik araması yapar.
    
    Örüntü: ModelComposite (PT-003)
    """
    
    def __init__(self, config: SearchIndexConfig):
        """
        SearchIndex sınıfını başlatır.
        
        Args:
            config: Arama indeksi yapılandırması.
        """
        if faiss is None:
            raise ImportError("Bu modülü kullanmak için FAISS kütüphanesi gereklidir. "
                              "pip install faiss-cpu veya pip install faiss-gpu ile kurabilirsiniz.")
        
        self.config = config
        self.index = None
        self.metadata = {}  # id -> metadata sözlüğü
        self.id_map = {}  # indeks -> id sözlüğü
        self.next_id = 0
        self._is_indexing = False
        self._indexing_progress = 0.0
        self._index_lock = threading.RLock()
        
        # İndeksi başlat
        self._create_index()
    
    def _create_index(self):
        """Yapılandırmaya göre FAISS indeksi oluşturur."""
        with self._index_lock:
            # Metrik türüne göre FAISS metriğini belirle
            if self.config.metric_type == "l2":
                metric = faiss.METRIC_L2
            elif self.config.metric_type == "ip":
                metric = faiss.METRIC_INNER_PRODUCT
            else:  # cosine
                metric = faiss.METRIC_INNER_PRODUCT  # Normalize edilmiş vektörler için iç çarpım = kosinüs benzerliği
            
            # İndeks türüne göre uygun FAISS indeksi oluştur
            if self.config.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.config.embedding_dim) if metric == faiss.METRIC_INNER_PRODUCT else \
                          faiss.IndexFlatL2(self.config.embedding_dim)
                
            elif self.config.index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(self.config.embedding_dim, self.config.hnsw_store_n, metric)
                
            elif self.config.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.config.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.config.embedding_dim, 
                                             self.config.ivf_nlist, metric)
                # IVF indeksi eğitim gerektirir, ancak bu aşamada veri olmadığından eğitim yapılmaz
            
            # Vektör sıkıştırma uygula (eğer belirtilmişse)
            if self.config.compression_level > 0:
                # ProductQuantizer kullanarak vektörleri sıkıştır
                # Bu, bellek kullanımını azaltır ancak kesinlik kaybına neden olabilir
                bits_per_component = min(8, max(4, self.config.compression_level))
                pq_index = faiss.IndexPQ(self.config.embedding_dim, 
                                       self.config.embedding_dim // 2,  # Number of subquantizers
                                       bits_per_component,
                                       metric)
                self.index = pq_index
            
            # GPU kullanımı (eğer belirtilmişse ve kullanılabilirse)
            if self.config.use_gpu and hasattr(faiss, 'StandardGpuResources'):
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    
    def add(self, 
            embeddings: Union[torch.Tensor, np.ndarray], 
            metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        İndekse gömme vektörleri ekler.
        
        Args:
            embeddings: Eklenecek gömme vektörleri [batch_size, embedding_dim].
            metadata_list: Her gömme için ilişkili meta veri listesi.
            
        Returns:
            List[int]: Eklenen gömmelere atanan ID'ler.
        """
        with self._index_lock:
            # Giriş veri türünü kontrol et ve numpy dizisine dönüştür
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            
            # Batch boyutunu al
            batch_size = embeddings.shape[0]
            
            # Metadata listesini kontrol et
            if metadata_list is None:
                metadata_list = [{}] * batch_size
            elif len(metadata_list) != batch_size:
                raise ValueError(f"metadata_list uzunluğu ({len(metadata_list)}) "
                               f"embeddings batch boyutu ({batch_size}) ile uyuşmuyor.")
            
            # Normalizasyon kontrolü (cosine metriği için gerekli)
            if self.config.metric_type == "cosine":
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Sıfır norm kontrolü
                zero_norm_indices = np.where(norms == 0)[0]
                if len(zero_norm_indices) > 0:
                    norms[zero_norm_indices] = 1.0
                    logger.warning(f"{len(zero_norm_indices)} vektörün normu sıfır, normalizasyon atlanıyor.")
                embeddings = embeddings / norms
            
            # IVF indeksi eğitim gerektirir ve henüz eğitilmediyse
            if self.config.index_type == "ivf" and not self.index.is_trained and batch_size > 0:
                if isinstance(self.index, faiss.IndexIVFFlat):
                    self.index.train(embeddings)
            
            # ID'leri hazırla
            ids = list(range(self.next_id, self.next_id + batch_size))
            self.next_id += batch_size
            
            # ID'lere metadata eşle
            for i, id_val in enumerate(ids):
                self.metadata[id_val] = metadata_list[i]
                self.id_map[id_val] = self.index.ntotal + i
            
            # Vektörleri indekse ekle
            self.index.add(embeddings)
            
            return ids
    
    def search(self, 
               query_embedding: Union[torch.Tensor, np.ndarray], 
               k: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Sorgu gömme vektörüne en benzer k sonucu döndürür.
        
        Args:
            query_embedding: Sorgu vektörü [embedding_dim] veya [batch_size, embedding_dim].
            k: Döndürülecek maksimum sonuç sayısı.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]: 
                (mesafe skorları, indeks değerleri, metadata listesi) üçlüsü.
        """
        with self._index_lock:
            # İndeks boşsa boş sonuç döndür
            if self.index.ntotal == 0:
                return np.array([]), np.array([]), []
            
            # Sorgu vektörünü uygun formata dönüştür
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.detach().cpu().numpy()
            
            # Tek bir vektör ise batch boyutu ekle
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Normalizasyon kontrolü (cosine metriği için gerekli)
            if self.config.metric_type == "cosine":
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Sorgu sonuç sayısını indeks boyutuna göre sınırla
            k = min(k, self.index.ntotal)
            
            # Arama yap
            distances, indices = self.index.search(query_embedding, k)
            
            # İndeksleri orijinal ID'lere dönüştür ve metadata'ları hazırla
            result_metadata = []
            result_ids = np.zeros_like(indices)
            
            # İndeks->ID dönüşümü ve metadata toplama
            for i in range(indices.shape[0]):
                for j in range(indices.shape[1]):
                    idx = indices[i, j]
                    # Geçersiz indeksleri kontrol et
                    if idx == -1:  # FAISS tarafından döndürülen geçersiz indeks
                        result_metadata.append({})
                        result_ids[i, j] = -1
                    else:
                        # İndeksi ID'ye çevir
                        id_val = next((id_key for id_key, index_val in self.id_map.items() 
                                    if index_val == idx), None)
                        if id_val is not None:
                            result_ids[i, j] = id_val
                            result_metadata.append(self.metadata.get(id_val, {}))
                        else:
                            result_ids[i, j] = -1
                            result_metadata.append({})
            
            # İç çarpım metriği için benzerlik skorlarını [0,1] aralığına dönüştür
            if self.config.metric_type in ["ip", "cosine"]:
                # FAISS, kosinüs uzaklığı yerine benzerlik döndürür
                # Maksimum benzerlik zaten 1'dir (normalize edilmiş vektörler için)
                distances = np.clip(distances, 0, 1)
            elif self.config.metric_type == "l2":
                # L2 mesafesi için, küçük değerler daha iyi benzerliği gösterir
                # Benzerliği [0,1] aralığına dönüştür (mesafe azaldıkça benzerlik artar)
                max_dist = np.max(distances) if distances.size > 0 else 1.0
                if max_dist > 0:
                    distances = 1.0 - distances / max_dist
            
            return distances, result_ids, result_metadata
    
    def batch_index(self, 
                    data_generator: Callable[[], Tuple[Union[torch.Tensor, np.ndarray], List[Dict[str, Any]]]],
                    total_items: int,
                    progress_callback: Optional[Callable[[float], None]] = None):
        """
        Büyük veri kümeleri için toplu indeksleme gerçekleştirir.
        
        Args:
            data_generator: Her çağrıldığında bir (embeddings, metadata_list) çifti döndüren fonksiyon.
            total_items: Toplam öğe sayısı (ilerleme hesaplaması için).
            progress_callback: İlerleme güncellemeleri için geri çağırma fonksiyonu.
        """
        self._is_indexing = True
        self._indexing_progress = 0.0
        
        try:
            items_processed = 0
            
            while items_processed < total_items:
                # Veri üreticiden bir parti al
                embeddings, metadata_list = data_generator()
                
                # Batch boyutunu al
                batch_size = embeddings.shape[0]
                
                # Partiyi indekse ekle
                self.add(embeddings, metadata_list)
                
                # İşlenen öğe sayısını güncelle
                items_processed += batch_size
                
                # İlerleme durumunu güncelle
                self._indexing_progress = min(1.0, items_processed / total_items)
                
                # İlerleme geri çağırma fonksiyonunu çağır
                if progress_callback:
                    progress_callback(self._indexing_progress)
        
        finally:
            self._is_indexing = False
            self._indexing_progress = 1.0
    
    def start_background_indexing(self, 
                               data_generator: Callable[[], Tuple[Union[torch.Tensor, np.ndarray], List[Dict[str, Any]]]],
                               total_items: int,
                               progress_callback: Optional[Callable[[float], None]] = None):
        """
        Arka planda indeksleme işlemi başlatır.
        
        Args:
            data_generator: Her çağrıldığında bir (embeddings, metadata_list) çifti döndüren fonksiyon.
            total_items: Toplam öğe sayısı (ilerleme hesaplaması için).
            progress_callback: İlerleme güncellemeleri için geri çağırma fonksiyonu.
        """
        if self._is_indexing:
            raise RuntimeError("Halihazırda bir indeksleme işlemi çalışıyor.")
        
        indexing_thread = threading.Thread(
            target=self.batch_index,
            args=(data_generator, total_items, progress_callback),
            daemon=True
        )
        indexing_thread.start()
        
        return indexing_thread
    
    def get_indexing_progress(self) -> float:
        """
        Mevcut indeksleme işleminin ilerlemesini döndürür.
        
        Returns:
            float: 0.0 ile 1.0 arasında bir ilerleme değeri.
        """
        return self._indexing_progress
    
    def is_indexing(self) -> bool:
        """
        İndeksleme işleminin devam edip etmediğini kontrol eder.
        
        Returns:
            bool: İndeksleme devam ediyorsa True, aksi halde False.
        """
        return self._is_indexing
    
    def save(self, path: Optional[str] = None):
        """
        İndeksi ve ilişkili verileri kalıcı depolamaya kaydeder.
        
        Args:
            path: İndeksin kaydedileceği dizin (None ise config'teki storage_path kullanılır).
        """
        with self._index_lock:
            # Kayıt yolunu belirle
            save_path = path or self.config.storage_path
            if save_path is None:
                raise ValueError("Kaydetme işlemi için path veya config.storage_path belirtilmelidir.")
            
            save_path = Path(save_path)
            os.makedirs(save_path, exist_ok=True)
            
            # İndeksi kaydet
            index_path = save_path / "faiss_index.bin"
            
            # GPU indeksi için CPU'ya dönüştürme
            if self.config.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_path))
            else:
                faiss.write_index(self.index, str(index_path))
            
            # Metadata ve ID haritasını kaydet
            metadata_path = save_path / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.metadata,
                    'id_map': self.id_map,
                    'next_id': self.next_id,
                    'config': self.config.__dict__
                }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SearchIndex':
        """
        Kaydedilmiş bir indeksi ve ilişkili verileri yükler.
        
        Args:
            path: İndeksin yükleneceği dizin.
            
        Returns:
            SearchIndex: Yüklenen indeks nesnesi.
        """
        load_path = Path(path)
        
        # Metadata ve yapılandırmayı yükle
        metadata_path = load_path / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Yapılandırma bilgilerini oluştur
        config_dict = data.get('config', {})
        config = SearchIndexConfig(**config_dict)
        
        # Yeni bir indeks nesnesi oluştur
        index_instance = cls(config)
        
        # Metadata ve ID haritasını güncelle
        index_instance.metadata = {int(k): v for k, v in data.get('metadata', {}).items()}
        index_instance.id_map = {int(k): v for k, v in data.get('id_map', {}).items()}
        index_instance.next_id = data.get('next_id', 0)
        
        # FAISS indeksini yükle
        index_path = load_path / "faiss_index.bin"
        loaded_index = faiss.read_index(str(index_path))
        
        # İndeksi GPU'ya taşı (eğer belirtilmişse)
        if config.use_gpu and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            loaded_index = faiss.index_cpu_to_gpu(res, 0, loaded_index)
        
        index_instance.index = loaded_index
        
        return index_instance
    
    def reset(self):
        """
        İndeksi sıfırlar ve tüm verileri temizler.
        """
        with self._index_lock:
            self.metadata = {}
            self.id_map = {}
            self.next_id = 0
            self._create_index()
    
    def get_size(self) -> int:
        """
        İndeksteki toplam öğe sayısını döndürür.
        
        Returns:
            int: İndeks büyüklüğü.
        """
        return self.index.ntotal if self.index else 0


class SearchIndexFactory:
    """
    Arama indeksi oluşturmak için fabrika sınıfı.
    
    Örüntü: FactoryMethod (PT-002)
    """
    
    @staticmethod
    def create(
        embedding_dim: int = 128,
        index_type: str = "flat",
        metric_type: str = "cosine",
        max_index_size: int = 1000000,
        use_gpu: bool = False,
        storage_path: Optional[str] = None,
        compression_level: int = 0
    ) -> SearchIndex:
        """
        Yeni bir SearchIndex örneği oluşturur.
        
        Args:
            embedding_dim: Gömme vektörlerinin boyutu (varsayılan: 128).
            index_type: İndeks türü ('flat', 'hnsw', 'ivf') (varsayılan: 'flat').
            metric_type: Benzerlik metriği ('l2', 'ip', 'cosine') (varsayılan: 'cosine').
            max_index_size: İndeksin maksimum büyüklüğü (varsayılan: 1000000).
            use_gpu: GPU kullanılıp kullanılmayacağı (varsayılan: False).
            storage_path: İndeksin kalıcı depolamaya kaydedileceği dizin (varsayılan: None).
            compression_level: Vektör sıkıştırma seviyesi (varsayılan: 0).
            
        Returns:
            SearchIndex: Arama indeksi örneği.
        """
        config = SearchIndexConfig(
            embedding_dim=embedding_dim,
            index_type=index_type,
            metric_type=metric_type,
            max_index_size=max_index_size,
            use_gpu=use_gpu,
            storage_path=storage_path,
            compression_level=compression_level
        )
        
        return SearchIndex(config) 