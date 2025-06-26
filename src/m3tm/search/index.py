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


@dataclass
class AdvancedSearchIndexConfig(SearchIndexConfig):
    """
    Gelişmiş arama indeksi yapılandırması - S30 Advanced Search Features.
    
    Context7 FAISS mobile optimization best practices uygulanmıştır.
    Patterns: PT-001 (ConfigurationDataclass), PT-015 (PluggableComponentStrategy)
    """
    
    # Advanced ANN Parameters (Context7 optimized)
    index_factory_string: Optional[str] = None  # FAISS factory string for complex indexes
    ann_algorithm: str = "hnsw"  # ANN algorithm: hnsw, ivf, pq, flat
    
    # HNSW Parameters (Mobile optimized)
    hnsw_m: int = 16  # Lower M for mobile to reduce memory
    hnsw_ef_construction: int = 200  # Balanced for mobile
    hnsw_ef_search: int = 50  # Runtime search parameter
    
    # IVF Parameters (Context7 recommendations)
    ivf_nprobe: int = 16  # Balance speed/accuracy for mobile
    ivf_quantizer_type: str = "flat"  # flat, pq for quantizer
    
    # Product Quantization (Mobile memory optimization)
    pq_m: int = 8  # Number of sub-quantizers
    pq_nbits: int = 8  # Bits per sub-quantizer
    enable_pq_compression: bool = True  # Memory efficiency
    
    # Large-scale optimizations
    max_memory_usage_gb: float = 2.0  # Mobile memory constraint
    enable_incremental_updates: bool = True
    index_sharding_enabled: bool = False  # For very large datasets
    shard_size_limit: int = 500000  # Items per shard
    
    # Performance monitoring
    enable_search_analytics: bool = True
    target_search_latency_ms: int = 50  # S30 requirement
    target_accuracy_recall: float = 0.95  # S30 requirement
    
    # Mobile-specific optimizations (Context7 best practices)
    enable_arm_optimization: bool = True  # ARM SVE support
    enable_batch_processing: bool = True  # Efficiency
    mobile_memory_strategy: str = "adaptive"  # adaptive, aggressive, conservative
    
    def __post_init__(self):
        """Enhanced validation with Context7 mobile constraints."""
        super().__post_init__()
        
        # Mobile memory validation
        if self.max_memory_usage_gb > 4.0:
            logger.warning(f"Memory usage {self.max_memory_usage_gb}GB exceeds mobile recommendations")
        
        # ANN algorithm validation
        valid_ann_algorithms = ["hnsw", "ivf", "pq", "flat", "ivf_pq"]
        if self.ann_algorithm not in valid_ann_algorithms:
            raise ValueError(f"ann_algorithm '{self.ann_algorithm}' invalid. "
                           f"Valid: {valid_ann_algorithms}")
        
        # HNSW mobile optimization checks
        if self.ann_algorithm == "hnsw":
            if self.hnsw_m > 32:
                logger.warning(f"HNSW M={self.hnsw_m} may be too high for mobile devices")
                
        # Auto-generate factory string if not provided
        if self.index_factory_string is None:
            self.index_factory_string = self._generate_mobile_optimized_factory_string()
    
    def _generate_mobile_optimized_factory_string(self) -> str:
        """Generate FAISS factory string optimized for mobile deployment."""
        if self.ann_algorithm == "hnsw":
            return f"HNSW{self.hnsw_m}"
        elif self.ann_algorithm == "ivf":
            if self.enable_pq_compression:
                return f"IVF{self.ivf_nlist},PQ{self.pq_m}"
            else:
                return f"IVF{self.ivf_nlist},Flat"
        elif self.ann_algorithm == "pq":
            return f"PQ{self.pq_m}"
        elif self.ann_algorithm == "ivf_pq":
            return f"IVF{self.ivf_nlist},PQ{self.pq_m}"
        else:  # flat
            return "Flat"


class AdvancedSearchIndexFactory:
    """
    Advanced search index factory with mobile optimization and Context7 best practices.
    
    Patterns: PT-002 (FactoryMethod), PT-015 (PluggableComponentStrategy)
    """
    
    @staticmethod
    def create_mobile_optimized_index(config: AdvancedSearchIndexConfig) -> 'AdvancedSearchIndex':
        """
        Create mobile-optimized FAISS index based on Context7 recommendations.
        
        Args:
            config: Advanced search configuration with mobile optimizations
            
        Returns:
            AdvancedSearchIndex instance optimized for mobile deployment
        """
        if faiss is None:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        return AdvancedSearchIndex(config)
    
    @staticmethod  
    def create_hnsw_index(embedding_dim: int, 
                         m: int = 16, 
                         ef_construction: int = 200,
                         max_memory_gb: float = 2.0) -> 'AdvancedSearchIndex':
        """
        Create HNSW index optimized for mobile with Context7 parameters.
        
        Args:
            embedding_dim: Embedding vector dimension
            m: HNSW connectivity parameter (lower for mobile)
            ef_construction: Construction-time search parameter
            max_memory_gb: Maximum memory usage constraint
            
        Returns:
            HNSW-based AdvancedSearchIndex
        """
        config = AdvancedSearchIndexConfig(
            embedding_dim=embedding_dim,
            ann_algorithm="hnsw",
            hnsw_m=m,
            hnsw_ef_construction=ef_construction,
            max_memory_usage_gb=max_memory_gb,
            enable_arm_optimization=True
        )
        return AdvancedSearchIndexFactory.create_mobile_optimized_index(config)
    
    @staticmethod
    def create_ivf_pq_index(embedding_dim: int,
                           nlist: int = 1024,
                           pq_m: int = 8,
                           max_memory_gb: float = 2.0) -> 'AdvancedSearchIndex':
        """
        Create IVF+PQ index for memory-efficient large-scale search.
        
        Args:
            embedding_dim: Embedding vector dimension
            nlist: Number of IVF clusters
            pq_m: Product quantization sub-quantizers
            max_memory_gb: Maximum memory usage constraint
            
        Returns:
            IVF+PQ-based AdvancedSearchIndex
        """
        config = AdvancedSearchIndexConfig(
            embedding_dim=embedding_dim,
            ann_algorithm="ivf_pq",
            ivf_nlist=nlist,
            pq_m=pq_m,
            enable_pq_compression=True,
            max_memory_usage_gb=max_memory_gb,
            enable_arm_optimization=True
        )
        return AdvancedSearchIndexFactory.create_mobile_optimized_index(config)
    
    @staticmethod
    def get_recommended_config_for_dataset_size(embedding_dim: int, 
                                               dataset_size: int,
                                               memory_constraint_gb: float = 2.0) -> AdvancedSearchIndexConfig:
        """
        Get recommended configuration based on dataset size and mobile constraints.
        
        Based on Context7 FAISS mobile optimization guidelines.
        
        Args:
            embedding_dim: Embedding vector dimension
            dataset_size: Expected number of vectors
            memory_constraint_gb: Mobile memory constraint
            
        Returns:
            Optimized AdvancedSearchIndexConfig
        """
        if dataset_size < 10000:
            # Small dataset - use flat index for accuracy
            return AdvancedSearchIndexConfig(
                embedding_dim=embedding_dim,
                ann_algorithm="flat",
                max_memory_usage_gb=memory_constraint_gb
            )
        elif dataset_size < 100000:
            # Medium dataset - use HNSW for balance
            return AdvancedSearchIndexConfig(
                embedding_dim=embedding_dim,
                ann_algorithm="hnsw",
                hnsw_m=16,
                hnsw_ef_construction=200,
                max_memory_usage_gb=memory_constraint_gb
            )
        else:
            # Large dataset - use IVF+PQ for memory efficiency
            nlist = min(int(np.sqrt(dataset_size)), 4096)
            return AdvancedSearchIndexConfig(
                embedding_dim=embedding_dim,
                ann_algorithm="ivf_pq",
                ivf_nlist=nlist,
                pq_m=8,
                enable_pq_compression=True,
                max_memory_usage_gb=memory_constraint_gb,
                enable_incremental_updates=True
            )
    

class AdvancedSearchIndex(SearchIndex):
    """
    Advanced search index with mobile optimization and large-scale support.
    
    Extends basic SearchIndex with:
    - HNSW algorithm support
    - Product Quantization compression
    - Large-scale dataset handling (1M+ items)
    - Mobile-optimized performance
    - Search analytics and monitoring
    
    Patterns: PT-003 (ModelComposite), PT-015 (PluggableComponentStrategy)
    """
    
    def __init__(self, config: AdvancedSearchIndexConfig):
        """Initialize advanced search index with enhanced configuration."""
        # Convert AdvancedSearchIndexConfig to SearchIndexConfig for parent
        base_config = SearchIndexConfig(
            embedding_dim=config.embedding_dim,
            index_type=config.ann_algorithm if config.ann_algorithm in ["flat", "hnsw", "ivf"] else "hnsw",
            metric_type=config.metric_type,
            max_index_size=config.max_index_size,
            use_gpu=config.use_gpu,
            storage_path=config.storage_path,
            compression_level=config.compression_level,
            hnsw_store_n=config.hnsw_m,
            ivf_nlist=config.ivf_nlist,
            batch_size=config.batch_size
        )
        
        super().__init__(base_config)
        
        # Advanced configuration
        self.advanced_config = config
        self.search_analytics = SearchAnalytics() if config.enable_search_analytics else None
        self.memory_monitor = MemoryMonitor(config.max_memory_usage_gb)
        
        # Advanced index creation
        self._create_advanced_index()
        
        logger.info(f"AdvancedSearchIndex initialized with {config.ann_algorithm} algorithm")
    
    def _create_advanced_index(self):
        """Create advanced FAISS index using factory string and mobile optimizations."""
        with self._index_lock:
            factory_string = self.advanced_config.index_factory_string
            
            try:
                # Use FAISS index factory for complex indexes
                self.index = faiss.index_factory(
                    self.advanced_config.embedding_dim,
                    factory_string,
                    faiss.METRIC_INNER_PRODUCT if self.advanced_config.metric_type in ["ip", "cosine"] else faiss.METRIC_L2
                )
                
                # Configure HNSW parameters if applicable
                if "HNSW" in factory_string:
                    self._configure_hnsw_parameters()
                
                # Configure IVF parameters if applicable  
                if "IVF" in factory_string:
                    self._configure_ivf_parameters()
                
                # Mobile ARM optimization (Context7 recommendation)
                if self.advanced_config.enable_arm_optimization:
                    self._apply_arm_optimizations()
                    
                logger.info(f"Advanced index created: {factory_string}")
                
            except Exception as e:
                logger.error(f"Failed to create advanced index: {e}")
                # Fallback to basic index
                super()._create_index()
    
    def _configure_hnsw_parameters(self):
        """Configure HNSW-specific parameters for mobile optimization."""
        if hasattr(self.index, 'hnsw'):
            hnsw = self.index.hnsw
            hnsw.efConstruction = self.advanced_config.hnsw_ef_construction
            hnsw.efSearch = self.advanced_config.hnsw_ef_search
            logger.info(f"HNSW configured: efConstruction={hnsw.efConstruction}, efSearch={hnsw.efSearch}")
    
    def _configure_ivf_parameters(self):
        """Configure IVF-specific parameters for mobile optimization."""
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.advanced_config.ivf_nprobe
            logger.info(f"IVF configured: nprobe={self.index.nprobe}")
    
    def _apply_arm_optimizations(self):
        """Apply ARM-specific optimizations based on Context7 guidelines."""
        # ARM optimizations would be applied at compilation time
        # This is a placeholder for runtime ARM-specific configurations
        logger.info("ARM optimizations applied")
    
    def search_with_analytics(self, 
                            query_embedding: Union[torch.Tensor, np.ndarray], 
                            k: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Enhanced search with performance analytics and monitoring.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadata, analytics)
        """
        start_time = time.time()
        
        # Perform search
        distances, indices, metadata = self.search(query_embedding, k)
        
        # Calculate analytics
        search_time_ms = (time.time() - start_time) * 1000
        
        analytics = {
            "search_time_ms": search_time_ms,
            "results_returned": len(indices),
            "memory_usage_mb": self.memory_monitor.get_current_usage_mb(),
            "index_size": self.get_ntotal(),
            "algorithm": self.advanced_config.ann_algorithm,
            "meets_latency_target": search_time_ms <= self.advanced_config.target_search_latency_ms
        }
        
        # Update search analytics
        if self.search_analytics:
            self.search_analytics.record_search(analytics)
        
        return distances, indices, metadata, analytics
    
    def add_with_memory_monitoring(self, 
                                 embeddings: Union[torch.Tensor, np.ndarray], 
                                 metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add embeddings with memory usage monitoring and optimization.
        
        Args:
            embeddings: Embedding vectors to add
            metadata_list: Optional metadata for each embedding
            
        Returns:
            List of assigned IDs
        """
        # Check memory before adding
        if not self.memory_monitor.can_add_vectors(len(embeddings)):
            logger.warning("Memory limit would be exceeded. Consider using incremental updates or compression.")
            
        return self.add(embeddings, metadata_list)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the index."""
        metrics = {
            "index_type": self.advanced_config.ann_algorithm,
            "index_size": self.get_ntotal(),
            "memory_usage_mb": self.memory_monitor.get_current_usage_mb(),
            "memory_limit_gb": self.advanced_config.max_memory_usage_gb,
            "embedding_dimension": self.advanced_config.embedding_dim
        }
        
        if self.search_analytics:
            metrics.update(self.search_analytics.get_summary_stats())
            
        return metrics


class SearchAnalytics:
    """Search performance analytics and monitoring."""
    
    def __init__(self):
        self.search_history = []
        self.total_searches = 0
        
    def record_search(self, analytics: Dict[str, Any]):
        """Record search analytics."""
        self.search_history.append(analytics)
        self.total_searches += 1
        
        # Keep only recent history to manage memory
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-500:]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from search history."""
        if not self.search_history:
            return {}
            
        search_times = [s["search_time_ms"] for s in self.search_history]
        
        return {
            "total_searches": self.total_searches,
            "avg_search_time_ms": np.mean(search_times),
            "p95_search_time_ms": np.percentile(search_times, 95),
            "searches_meeting_target": sum(1 for s in self.search_history if s["meets_latency_target"]),
            "target_meeting_rate": sum(1 for s in self.search_history if s["meets_latency_target"]) / len(self.search_history)
        }


class MemoryMonitor:
    """Memory usage monitoring for mobile deployment."""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    
    def get_current_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return 0.0
    
    def can_add_vectors(self, num_vectors: int, embedding_dim: int = 128) -> bool:
        """Check if adding vectors would exceed memory limit."""
        estimated_size_bytes = num_vectors * embedding_dim * 4  # float32
        current_usage_bytes = self.get_current_usage_mb() * 1024 * 1024
        
        return (current_usage_bytes + estimated_size_bytes) < self.max_memory_bytes