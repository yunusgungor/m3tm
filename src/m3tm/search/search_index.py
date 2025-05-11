"""
Arama İndeksi Modülü

Bu modül, arama gömmeleri için indeksleme ve arama işlevselliği sağlar.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional

class SearchIndex:
    """
    Arama indeksi sınıfı.
    
    Bu sınıf, arama gömmelerini saklar ve hızlı benzerlik araması yapabilir.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Gömme vektörlerinin boyutu
        """
        self.embedding_dim = embedding_dim
        self.embeddings = []  # List[torch.Tensor], her biri [embedding_dim] boyutunda
        self.metadata = []    # List[Dict], her gömme için ilişkili metaveri
    
    def add(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        İndekse gömme ve metaveri ekler.
        
        Args:
            embedding: Gömme vektörü [embedding_dim]
            metadata: İlişkili metaveri
        """
        # Gömme vektörünün doğru boyutta olduğunu kontrol et
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(f"Embedding must have shape [{self.embedding_dim}], got {embedding.shape}")
        
        # Normalize edilmiş olduğundan emin ol
        if not torch.allclose(torch.norm(embedding, p=2), torch.tensor(1.0), atol=1e-6):
            embedding = embedding / torch.norm(embedding, p=2)
        
        # Gömme ve metaveriyi sakla
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
    
    def search(self, query_embedding: torch.Tensor, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Benzerlik araması yapar.
        
        Args:
            query_embedding: Sorgu gömme vektörü [embedding_dim]
            top_k: Döndürülecek en benzer sonuç sayısı
            
        Returns:
            List[Dict]: En benzer öğelerin listesi (skor ve metaveri içerir)
        """
        if len(self.embeddings) == 0:
            return []
        
        # Normalize edilmiş olduğundan emin ol
        if not torch.allclose(torch.norm(query_embedding, p=2), torch.tensor(1.0), atol=1e-6):
            query_embedding = query_embedding / torch.norm(query_embedding, p=2)
        
        # Benzerlik skorlarını hesapla (nokta çarpımı)
        similarities = [torch.dot(query_embedding, emb).item() for emb in self.embeddings]
        
        # Skorlara göre sırala (azalan)
        indices = np.argsort(similarities)[::-1][:top_k]
        
        # Sonuçları hazırla
        results = []
        for idx in indices:
            results.append({
                "score": similarities[idx],
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def batch_search(self, query_embeddings: torch.Tensor, top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        Toplu benzerlik araması yapar.
        
        Args:
            query_embeddings: Sorgu gömme vektörleri [batch_size, embedding_dim]
            top_k: Her sorgu için döndürülecek en benzer sonuç sayısı
            
        Returns:
            List[List[Dict]]: Her sorgu için en benzer öğelerin listesi
        """
        if len(self.embeddings) == 0:
            return [[] for _ in range(query_embeddings.shape[0])]
        
        # Normalize edilmiş olduğundan emin ol
        norms = torch.norm(query_embeddings, p=2, dim=1, keepdim=True)
        query_embeddings = query_embeddings / norms
        
        # İndeksteki tüm gömmeleri birleştir
        index_embeddings = torch.stack(self.embeddings)  # [num_items, embedding_dim]
        
        # Toplu benzerlik hesapla
        # [batch_size, num_items]
        similarities = torch.matmul(query_embeddings, index_embeddings.t())
        
        # Her sorgu için en iyi sonuçları al
        results = []
        for i in range(similarities.shape[0]):
            sim_scores = similarities[i].tolist()
            indices = np.argsort(sim_scores)[::-1][:top_k]
            
            query_results = []
            for idx in indices:
                query_results.append({
                    "score": sim_scores[idx],
                    "metadata": self.metadata[idx]
                })
            
            results.append(query_results)
        
        return results
    
    def save(self, file_path: str) -> None:
        """
        İndeksi dosyaya kaydeder.
        
        Args:
            file_path: Kaydedilecek dosya yolu
        """
        # Gömme vektörlerini listeye dönüştür
        embeddings_list = [emb.tolist() for emb in self.embeddings]
        
        # İndeks verilerini hazırla
        data = {
            "embedding_dim": self.embedding_dim,
            "embeddings": embeddings_list,
            "metadata": self.metadata
        }
        
        # JSON olarak kaydet
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    def load(self, file_path: str) -> None:
        """
        İndeksi dosyadan yükler.
        
        Args:
            file_path: Yüklenecek dosya yolu
        """
        # JSON dosyasını oku
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Boyut kontrolü
        if data["embedding_dim"] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {data['embedding_dim']}")
        
        # Veriyi yükle
        self.embeddings = [torch.tensor(emb) for emb in data["embeddings"]]
        self.metadata = data["metadata"] 