"""
M³TM Arama Servisi API

Bu modül, M³TM modelinin semantik arama yeteneklerine erişim sağlayan API'yi içerir.
Metin, görüntü ve çok modlu sorgular için arama fonksiyonları sunar.

Örüntü: FactoryMethod (PT-002), ConfigurationDataclass (PT-001)
"""

from typing import Optional, Dict, Any, Union, List, Tuple, Callable, TypeVar, Generic
from dataclasses import dataclass, field
import asyncio
import threading
import time
from enum import Enum
from pathlib import Path
import logging

import torch
import numpy as np

from m3tm.search.index import SearchIndex, SearchIndexConfig, SearchIndexFactory
from m3tm.search.projection import M3TMSearchProjection, SearchProjectionConfig, SearchProjectionFactory

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SearchResultType(Enum):
    """Arama sonucu türleri."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class SearchFilter:
    """
    Arama sonuçlarını filtrelemek için kullanılan filtre sınıfı.
    
    Attributes:
        metadata_filters (Dict[str, Any]): Metadata alanlarına göre filtreleme kriterleri.
        min_score (float): Minimum benzerlik skoru.
        result_type (Optional[SearchResultType]): Sonuç türü filtresi.
        date_range (Optional[Tuple[float, float]]): Tarih aralığı filtresi (timestamp olarak).
    """
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    min_score: float = 0.0
    result_type: Optional[SearchResultType] = None
    date_range: Optional[Tuple[float, float]] = None
    
    def apply(self, score: float, metadata: Dict[str, Any]) -> bool:
        """
        Filtreyi uygular ve sonucun filtreye uygun olup olmadığını kontrol eder.
        
        Args:
            score: Benzerlik skoru.
            metadata: Sonuç metadata'sı.
            
        Returns:
            bool: Sonuç filtreye uygunsa True, değilse False.
        """
        # Skor kontrolü
        if score < self.min_score:
            return False
        
        # Tür kontrolü
        if self.result_type is not None:
            result_type = metadata.get("type", "unknown")
            if result_type != self.result_type.value:
                return False
        
        # Tarih aralığı kontrolü
        if self.date_range is not None:
            timestamp = metadata.get("timestamp")
            if timestamp is None or not (self.date_range[0] <= timestamp <= self.date_range[1]):
                return False
        
        # Metadata filtresi kontrolü
        for key, value in self.metadata_filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        
        return True


@dataclass
class Pagination:
    """
    Arama sonuçlarını sayfalamak için kullanılan sınıf.
    
    Attributes:
        page (int): Sayfa numarası (0'dan başlar).
        page_size (int): Sayfa başına sonuç sayısı.
    """
    page: int = 0
    page_size: int = 10
    
    def apply(self, results: List[T]) -> List[T]:
        """
        Sayfalama uygular ve belirtilen sayfadaki sonuçları döndürür.
        
        Args:
            results: Tüm sonuçlar listesi.
            
        Returns:
            List[T]: Belirtilen sayfadaki sonuçlar.
        """
        start_idx = self.page * self.page_size
        end_idx = start_idx + self.page_size
        
        if start_idx >= len(results):
            return []
        
        return results[start_idx:end_idx]


@dataclass
class SearchResult(Generic[T]):
    """
    Tekil bir arama sonucu.
    
    Attributes:
        item: Sonuç öğesi.
        score (float): Benzerlik skoru.
        metadata (Dict[str, Any]): Sonuç metadata'sı.
        id (int): Sonuç ID'si.
    """
    item: T
    score: float
    metadata: Dict[str, Any]
    id: int


@dataclass
class SearchResults(Generic[T]):
    """
    Arama sonuçları koleksiyonu.
    
    Attributes:
        results (List[SearchResult[T]]): Sonuçlar listesi.
        total_count (int): Toplam sonuç sayısı (filtreleme ve sayfalama öncesi).
        query_time (float): Sorgu süresi (saniye).
        page (int): Mevcut sayfa numarası.
        total_pages (int): Toplam sayfa sayısı.
    """
    results: List[SearchResult[T]]
    total_count: int
    query_time: float
    page: int
    total_pages: int


@dataclass
class SearchServiceConfig:
    """
    Arama servisi yapılandırması.
    
    Attributes:
        index_config (SearchIndexConfig): Arama indeksi yapılandırması.
        projection_config (SearchProjectionConfig): Arama projeksiyonu yapılandırması.
        default_top_k (int): Varsayılan olarak döndürülecek sonuç sayısı.
        cache_projections (bool): Projeksiyonları önbelleğe alıp almama.
        enable_async (bool): Asenkron aramaları etkinleştirme.
        max_batch_size (int): Maksimum toplu işleme boyutu.
    """
    index_config: Optional[SearchIndexConfig] = None
    projection_config: Optional[SearchProjectionConfig] = None
    default_top_k: int = 10
    cache_projections: bool = True
    enable_async: bool = True
    max_batch_size: int = 32
    
    def __post_init__(self):
        """Varsayılan yapılandırmaları oluşturur."""
        if self.index_config is None:
            self.index_config = SearchIndexConfig()
        
        if self.projection_config is None:
            self.projection_config = SearchProjectionConfig(
                input_dim=512,  # Varsayılan model çıktı boyutu
                embedding_dim=self.index_config.embedding_dim
            )


class SearchService:
    """
    Semantik arama servisi.
    
    Bu sınıf, M³TM modelinin semantik arama yeteneklerine erişim sağlar.
    Metin, görüntü ve çok modlu sorgular için arama fonksiyonları sunar.
    
    Örüntü: FactoryMethod (PT-002)
    """
    
    def __init__(self, 
                 config: SearchServiceConfig,
                 model: Optional[torch.nn.Module] = None,
                 index: Optional[SearchIndex] = None,
                 projection: Optional[M3TMSearchProjection] = None):
        """
        SearchService sınıfını başlatır.
        
        Args:
            config: Arama servisi yapılandırması.
            model: İsteğe bağlı, önceden eğitilmiş M³TM modeli.
            index: İsteğe bağlı, önceden oluşturulmuş arama indeksi.
            projection: İsteğe bağlı, önceden oluşturulmuş arama projeksiyonu.
        """
        self.config = config
        self.model = model
        
        # İndeks oluştur veya kullan
        self.index = index if index is not None else SearchIndexFactory.create(
            embedding_dim=config.index_config.embedding_dim,
            index_type=config.index_config.index_type,
            metric_type=config.index_config.metric_type,
            use_gpu=config.index_config.use_gpu,
            storage_path=config.index_config.storage_path,
            compression_level=config.index_config.compression_level
        )
        
        # Projeksiyon oluştur veya kullan
        self.projection = projection if projection is not None else SearchProjectionFactory.create(
            input_dim=config.projection_config.input_dim,
            embedding_dim=config.projection_config.embedding_dim,
            dropout_rate=config.projection_config.dropout_rate,
            use_layer_norm=config.projection_config.use_layer_norm,
            layer_norm_eps=config.projection_config.layer_norm_eps
        )
        
        # Projeksiyon önbelleği
        self._projection_cache = {}
        self._cache_lock = threading.RLock()
    
    def search_by_text(self, 
                      text_query: str,
                      top_k: Optional[int] = None,
                      filter_params: Optional[SearchFilter] = None,
                      pagination: Optional[Pagination] = None) -> SearchResults:
        """
        Metin sorgusu ile arama yapar.
        
        Args:
            text_query: Metin sorgusu.
            top_k: Döndürülecek maksimum sonuç sayısı.
            filter_params: Filtreleme parametreleri.
            pagination: Sayfalama parametreleri.
            
        Returns:
            SearchResults: Arama sonuçları.
        """
        if self.model is None:
            raise ValueError("Metin araması için model gereklidir.")
        
        # Modeli kullanarak metin gömmesi oluştur
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_query)
        
        # Gömmeyi arama uzayına projekte et
        search_embedding = self._project_embedding(text_embedding, cache_key=f"text:{text_query}")
        
        # Arama yap
        return self._search_with_embedding(search_embedding, top_k, filter_params, pagination)
    
    def search_by_image(self,
                       image_tensor: torch.Tensor,
                       top_k: Optional[int] = None,
                       filter_params: Optional[SearchFilter] = None,
                       pagination: Optional[Pagination] = None) -> SearchResults:
        """
        Görüntü sorgusu ile arama yapar.
        
        Args:
            image_tensor: Görüntü tensörü [C, H, W] veya [B, C, H, W].
            top_k: Döndürülecek maksimum sonuç sayısı.
            filter_params: Filtreleme parametreleri.
            pagination: Sayfalama parametreleri.
            
        Returns:
            SearchResults: Arama sonuçları.
        """
        if self.model is None:
            raise ValueError("Görüntü araması için model gereklidir.")
        
        # Görüntü tensörünün boyutunu kontrol et ve düzenle
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        # Modeli kullanarak görüntü gömmesi oluştur
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
        
        # Gömmeyi arama uzayına projekte et
        # Görüntüler için önbellek anahtarı olarak hash kullan
        cache_key = f"image:{hash(image_tensor.cpu().numpy().tobytes())}"
        search_embedding = self._project_embedding(image_embedding, cache_key=cache_key)
        
        # Arama yap
        return self._search_with_embedding(search_embedding, top_k, filter_params, pagination)
    
    def search_multimodal(self,
                         text_query: Optional[str] = None,
                         image_tensor: Optional[torch.Tensor] = None,
                         top_k: Optional[int] = None,
                         filter_params: Optional[SearchFilter] = None,
                         pagination: Optional[Pagination] = None) -> SearchResults:
        """
        Çok modlu sorgu ile arama yapar.
        
        Args:
            text_query: Metin sorgusu.
            image_tensor: Görüntü tensörü [C, H, W] veya [B, C, H, W].
            top_k: Döndürülecek maksimum sonuç sayısı.
            filter_params: Filtreleme parametreleri.
            pagination: Sayfalama parametreleri.
            
        Returns:
            SearchResults: Arama sonuçları.
        """
        if self.model is None:
            raise ValueError("Çok modlu arama için model gereklidir.")
        
        if text_query is None and image_tensor is None:
            raise ValueError("En az bir sorgu türü (metin veya görüntü) belirtilmelidir.")
        
        # Metin ve görüntü gömmelerini oluştur
        text_embedding = None
        image_embedding = None
        
        with torch.no_grad():
            if text_query is not None:
                text_embedding = self.model.encode_text(text_query)
            
            if image_tensor is not None:
                # Görüntü tensörünün boyutunu kontrol et ve düzenle
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                
                image_embedding = self.model.encode_image(image_tensor)
        
        # Çok modlu gömmeyi oluştur (modelin füzyon mekanizmasını kullan)
        multimodal_embedding = self.model.fuse_embeddings(text_embedding, image_embedding)
        
        # Gömmeyi arama uzayına projekte et
        cache_key = None  # Çok modlu sorgular için önbellek kullanma
        search_embedding = self._project_embedding(multimodal_embedding, cache_key=cache_key)
        
        # Arama yap
        return self._search_with_embedding(search_embedding, top_k, filter_params, pagination)
    
    async def search_by_text_async(self,
                                 text_query: str,
                                 top_k: Optional[int] = None,
                                 filter_params: Optional[SearchFilter] = None,
                                 pagination: Optional[Pagination] = None) -> SearchResults:
        """
        Metin sorgusu ile asenkron arama yapar.
        
        Args:
            text_query: Metin sorgusu.
            top_k: Döndürülecek maksimum sonuç sayısı.
            filter_params: Filtreleme parametreleri.
            pagination: Sayfalama parametreleri.
            
        Returns:
            SearchResults: Arama sonuçları.
        """
        if not self.config.enable_async:
            raise RuntimeError("Asenkron aramalar devre dışı bırakılmış.")
        
        # Asenkron çalıştırma için bir iş parçacığı havuzu kullanılabilir
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.search_by_text(text_query, top_k, filter_params, pagination)
        )
        
        return result
    
    async def search_by_image_async(self,
                                  image_tensor: torch.Tensor,
                                  top_k: Optional[int] = None,
                                  filter_params: Optional[SearchFilter] = None,
                                  pagination: Optional[Pagination] = None) -> SearchResults:
        """
        Görüntü sorgusu ile asenkron arama yapar.
        
        Args:
            image_tensor: Görüntü tensörü [C, H, W] veya [B, C, H, W].
            top_k: Döndürülecek maksimum sonuç sayısı.
            filter_params: Filtreleme parametreleri.
            pagination: Sayfalama parametreleri.
            
        Returns:
            SearchResults: Arama sonuçları.
        """
        if not self.config.enable_async:
            raise RuntimeError("Asenkron aramalar devre dışı bırakılmış.")
        
        # Asenkron çalıştırma için bir iş parçacığı havuzu kullanılabilir
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.search_by_image(image_tensor, top_k, filter_params, pagination)
        )
        
        return result
    
    async def search_multimodal_async(self,
                                    text_query: Optional[str] = None,
                                    image_tensor: Optional[torch.Tensor] = None,
                                    top_k: Optional[int] = None,
                                    filter_params: Optional[SearchFilter] = None,
                                    pagination: Optional[Pagination] = None) -> SearchResults:
        """
        Çok modlu sorgu ile asenkron arama yapar.
        
        Args:
            text_query: Metin sorgusu.
            image_tensor: Görüntü tensörü [C, H, W] veya [B, C, H, W].
            top_k: Döndürülecek maksimum sonuç sayısı.
            filter_params: Filtreleme parametreleri.
            pagination: Sayfalama parametreleri.
            
        Returns:
            SearchResults: Arama sonuçları.
        """
        if not self.config.enable_async:
            raise RuntimeError("Asenkron aramalar devre dışı bırakılmış.")
        
        # Asenkron çalıştırma için bir iş parçacığı havuzu kullanılabilir
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self.search_multimodal(text_query, image_tensor, top_k, filter_params, pagination)
        )
        
        return result
    
    def add_item(self, 
                embedding: torch.Tensor,
                metadata: Dict[str, Any]) -> int:
        """
        İndekse bir öğe ekler.
        
        Args:
            embedding: Öğe gömmesi.
            metadata: Öğe metadata'sı.
            
        Returns:
            int: Eklenen öğenin ID'si.
        """
        # Gömmeyi arama uzayına projekte et
        search_embedding = self._project_embedding(embedding)
        
        # İndekse ekle
        ids = self.index.add(search_embedding, [metadata])
        
        return ids[0]
    
    def add_batch(self,
                 embeddings: torch.Tensor,
                 metadata_list: List[Dict[str, Any]]) -> List[int]:
        """
        İndekse toplu öğe ekler.
        
        Args:
            embeddings: Öğe gömmeleri [batch_size, embed_dim].
            metadata_list: Öğe metadata'ları listesi.
            
        Returns:
            List[int]: Eklenen öğelerin ID'leri.
        """
        # Gömmeleri arama uzayına projekte et
        search_embeddings = self._project_embedding(embeddings)
        
        # İndekse ekle
        ids = self.index.add(search_embeddings, metadata_list)
        
        return ids
    
    def add_text(self,
                text: str,
                metadata: Dict[str, Any]) -> int:
        """
        İndekse bir metin ekler.
        
        Args:
            text: Metin.
            metadata: Metadata.
            
        Returns:
            int: Eklenen metnin ID'si.
        """
        if self.model is None:
            raise ValueError("Metin eklemek için model gereklidir.")
        
        # Metni gömmeye dönüştür
        with torch.no_grad():
            embedding = self.model.encode_text(text)
        
        # Metadata'ya tür ekle
        if "type" not in metadata:
            metadata["type"] = SearchResultType.TEXT.value
        
        # İndekse ekle
        return self.add_item(embedding, metadata)
    
    def add_image(self,
                 image_tensor: torch.Tensor,
                 metadata: Dict[str, Any]) -> int:
        """
        İndekse bir görüntü ekler.
        
        Args:
            image_tensor: Görüntü tensörü [C, H, W].
            metadata: Metadata.
            
        Returns:
            int: Eklenen görüntünün ID'si.
        """
        if self.model is None:
            raise ValueError("Görüntü eklemek için model gereklidir.")
        
        # Görüntü tensörünün boyutunu kontrol et ve düzenle
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        # Görüntüyü gömmeye dönüştür
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
        
        # Metadata'ya tür ekle
        if "type" not in metadata:
            metadata["type"] = SearchResultType.IMAGE.value
        
        # İndekse ekle
        return self.add_item(embedding, metadata)
    
    def save(self, path: str):
        """
        Arama indeksini kaydeder.
        
        Args:
            path: Kayıt dizini.
        """
        self.index.save(path)
    
    @classmethod
    def load(cls, 
             path: str,
             model: Optional[torch.nn.Module] = None,
             projection: Optional[M3TMSearchProjection] = None) -> 'SearchService':
        """
        Kaydedilmiş bir arama indeksini yükler.
        
        Args:
            path: İndeks dizini.
            model: İsteğe bağlı, önceden eğitilmiş M³TM modeli.
            projection: İsteğe bağlı, önceden oluşturulmuş arama projeksiyonu.
            
        Returns:
            SearchService: Arama servisi.
        """
        # İndeksi yükle
        index = SearchIndex.load(path)
        
        # Yapılandırmayı oluştur
        config = SearchServiceConfig(
            index_config=index.config,
            projection_config=SearchProjectionConfig(
                input_dim=512,  # Varsayılan model çıktı boyutu
                embedding_dim=index.config.embedding_dim
            )
        )
        
        # Arama servisini oluştur
        return cls(config, model=model, index=index, projection=projection)
    
    def _project_embedding(self, 
                          embedding: torch.Tensor,
                          cache_key: Optional[str] = None) -> torch.Tensor:
        """
        Gömmeyi arama uzayına projekte eder.
        
        Args:
            embedding: Gömme.
            cache_key: Önbellek anahtarı.
            
        Returns:
            torch.Tensor: Arama uzayına projekte edilmiş gömme.
        """
        # Önbellekte varsa kullan
        if cache_key is not None and self.config.cache_projections:
            with self._cache_lock:
                if cache_key in self._projection_cache:
                    return self._projection_cache[cache_key]
        
        # Projeksiyon uygula
        search_embedding = self.projection(embedding, return_dict=False)
        
        # Önbelleğe ekle
        if cache_key is not None and self.config.cache_projections:
            with self._cache_lock:
                self._projection_cache[cache_key] = search_embedding
        
        return search_embedding
    
    def _search_with_embedding(self,
                              search_embedding: torch.Tensor,
                              top_k: Optional[int] = None,
                              filter_params: Optional[SearchFilter] = None,
                              pagination: Optional[Pagination] = None) -> SearchResults:
        """
        Arama gömme vektörü ile arama yapar.
        
        Args:
            search_embedding: Arama gömme vektörü.
            top_k: Döndürülecek maksimum sonuç sayısı.
            filter_params: Filtreleme parametreleri.
            pagination: Sayfalama parametreleri.
            
        Returns:
            SearchResults: Arama sonuçları.
        """
        # Varsayılan değerleri kullan
        if top_k is None:
            top_k = self.config.default_top_k
        
        if filter_params is None:
            filter_params = SearchFilter()
        
        if pagination is None:
            pagination = Pagination()
        
        # Arama başlangıç zamanı
        start_time = time.time()
        
        # Arama yap (filtreleme için daha fazla sonuç iste)
        distances, result_ids, result_metadata = self.index.search(
            search_embedding, 
            k=min(top_k * 4, self.index.get_size())  # Filtreleme için daha fazla sonuç iste
        )
        
        # Sonuçları oluştur
        results = []
        for i in range(len(result_ids[0])):
            rid = int(result_ids[0][i])
            if rid == -1:  # Geçersiz sonuç
                continue
                
            distance = float(distances[0][i])
            metadata = result_metadata[i]
            
            # Uzaklığı benzerlik skoruna dönüştür (metrik tipine göre)
            if self.index.config.metric_type == "cosine" or self.index.config.metric_type == "ip":
                # Cosine ve iç çarpım için, daha yüksek değer daha iyi benzerlik
                score = distance
            else:
                # L2 için, daha düşük değer daha iyi benzerlik, bu yüzden tersini al
                score = 1.0 / (1.0 + distance)  # 0 ile 1 arasında normalize et
            
            # Filtreleme uygula
            if filter_params.apply(score, metadata):
                results.append(SearchResult(
                    item=None,  # Öğe verisi şu anda yok, sadece ID ve metadata var
                    score=score,
                    metadata=metadata,
                    id=rid
                ))
        
        # Toplam sonuç sayısı
        total_count = len(results)
        
        # Sayfalama uygula
        page_results = pagination.apply(results)
        
        # Toplam sayfa sayısı
        total_pages = (total_count + pagination.page_size - 1) // pagination.page_size
        
        # Arama bitiş zamanı
        end_time = time.time()
        query_time = end_time - start_time
        
        # Sonuçları döndür
        return SearchResults(
            results=page_results,
            total_count=total_count,
            query_time=query_time,
            page=pagination.page,
            total_pages=total_pages
        )


class SearchServiceFactory:
    """
    Arama servisi oluşturmak için fabrika sınıfı.
    
    Örüntü: FactoryMethod (PT-002)
    """
    
    @staticmethod
    def create(
        model: Optional[torch.nn.Module] = None,
        embedding_dim: int = 128,
        index_type: str = "flat",
        metric_type: str = "cosine",
        use_gpu: bool = False,
        storage_path: Optional[str] = None,
        enable_async: bool = True
    ) -> SearchService:
        """
        Yeni bir SearchService örneği oluşturur.
        
        Args:
            model: İsteğe bağlı, önceden eğitilmiş M³TM modeli.
            embedding_dim: Arama gömme uzayının boyutu (varsayılan: 128).
            index_type: İndeks türü ('flat', 'hnsw', 'ivf') (varsayılan: 'flat').
            metric_type: Benzerlik metriği ('l2', 'ip', 'cosine') (varsayılan: 'cosine').
            use_gpu: GPU kullanılıp kullanılmayacağı (varsayılan: False).
            storage_path: İndeksin kalıcı depolamaya kaydedileceği dizin (varsayılan: None).
            enable_async: Asenkron aramaları etkinleştirme (varsayılan: True).
            
        Returns:
            SearchService: Arama servisi.
        """
        # İndeks yapılandırmasını oluştur
        index_config = SearchIndexConfig(
            embedding_dim=embedding_dim,
            index_type=index_type,
            metric_type=metric_type,
            use_gpu=use_gpu,
            storage_path=storage_path
        )
        
        # Projeksiyon yapılandırmasını oluştur
        projection_config = SearchProjectionConfig(
            input_dim=512,  # Varsayılan model çıktı boyutu
            embedding_dim=embedding_dim
        )
        
        # Servis yapılandırmasını oluştur
        config = SearchServiceConfig(
            index_config=index_config,
            projection_config=projection_config,
            enable_async=enable_async
        )
        
        # Arama servisini oluştur
        return SearchService(config, model=model) 