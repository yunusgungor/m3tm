"""
Arama Servisi Modülü

Bu modül, metin ve görüntü verilerinin karmaşık sorguları için yüksek seviyeli bir arama API'si sağlar.
"""

import torch
from typing import List, Dict, Any, Optional, Union

from .search_index import SearchIndex

class SearchService:
    """
    Arama servisi sınıfı.
    
    Bu sınıf, metin ve görüntü içeriklerinin semantik araması için yüksek seviyeli bir API sağlar.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Gömme vektörlerinin boyutu
        """
        self.embedding_dim = embedding_dim
        self.index = SearchIndex(embedding_dim)
        self.model = None  # Daha sonra model yüklenebilir
    
    def add_content(self, text: Optional[str] = None, image: Optional[torch.Tensor] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        İndekse yeni içerik ekler.
        
        Args:
            text: Metin içeriği (isteğe bağlı)
            image: Görüntü tensörü (isteğe bağlı)
            metadata: İlişkili metaveri
        
        Raises:
            ValueError: Ne metin ne de görüntü sağlanmadıysa
        """
        if text is None and image is None:
            raise ValueError("Either text or image must be provided")
        
        # Varsayılan metaveri
        if metadata is None:
            metadata = {}
        
        # Gömme hesapla
        embedding = self._get_embedding(text=text, image=image)
        
        # İndekse ekle
        self.index.add(embedding, metadata)
    
    def search(self, text: Optional[str] = None, image: Optional[torch.Tensor] = None, 
              top_k: int = 10) -> List[Dict[str, Any]]:
        """
        İndekste arama yapar.
        
        Args:
            text: Aranacak metin (isteğe bağlı)
            image: Aranacak görüntü (isteğe bağlı)
            top_k: Döndürülecek en benzer sonuç sayısı
            
        Returns:
            List[Dict]: En benzer öğelerin listesi (skor ve metaveri içerir)
            
        Raises:
            ValueError: Ne metin ne de görüntü sağlanmadıysa veya her ikisi de sağlandıysa
        """
        if text is None and image is None:
            raise ValueError("Either text or image must be provided")
        
        if text is not None and image is not None:
            raise ValueError("Only one of text or image should be provided")
        
        # Sorgu gömme hesapla
        query_embedding = self._get_embedding(text=text, image=image)
        
        # Arama yap
        results = self.index.search(query_embedding, top_k=top_k)
        
        return results
    
    def _get_embedding(self, text: Optional[str] = None, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Metin veya görüntü için gömme hesaplar.
        
        Args:
            text: Metin içeriği (isteğe bağlı)
            image: Görüntü tensörü (isteğe bağlı)
            
        Returns:
            torch.Tensor: Normalize edilmiş gömme vektörü
            
        Raises:
            ValueError: Model yüklenmemişse veya içerik işlenemiyorsa
        """
        # NOT: Bu yöntem gerçek uygulamada modeli kullanacaktır.
        # Şimdilik rastgele bir gömme oluşturacağız.
        
        # Gerçek uygulamada, model yüklenmiş olmalı
        if self.model is None:
            # Bu sadece test için, gerçek uygulamada daha iyi bir hata yönetimi gerekli
            # Rastgele bir gömme oluştur
            embedding = torch.randn(self.embedding_dim)
            # L2 normalizasyonu
            embedding = embedding / torch.norm(embedding, p=2)
            return embedding
        
        # Normalde model kullanılacaktır:
        # if text is not None:
        #     return self.model.get_text_embedding(text)
        # else:
        #     return self.model.get_image_embedding(image)
    
    def save_index(self, file_path: str) -> None:
        """
        Arama indeksini dosyaya kaydeder.
        
        Args:
            file_path: Kaydedilecek dosya yolu
        """
        self.index.save(file_path)
    
    def load_index(self, file_path: str) -> None:
        """
        Arama indeksini dosyadan yükler.
        
        Args:
            file_path: Yüklenecek dosya yolu
        """
        self.index.load(file_path) 