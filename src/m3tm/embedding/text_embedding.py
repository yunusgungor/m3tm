"""
Metin Embedding Modülü

Bu modül, TokenizerConfig ve TextEmbeddingConfig yapılandırmalarını kullanarak 
metin tokenlerini vektör gösterimlerine dönüştüren PyTorch modüllerini içerir.
"""

import math
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextEmbeddingConfig, TokenizerConfig
from .tokenizer import SimpleTokenizer


class TextEmbedding(nn.Module):
    """
    Metin token ID'lerinden embedding vektörleri oluşturan PyTorch modülü.
    
    Bu modül, token ID'lerini karşılık gelen gömme vektörlerine dönüştürür.
    İsteğe bağlı olarak konum gömmesini (positional embedding) ekler ve
    bu gömmeleri normalize eder. Boyut düşürme desteği içerir.
    
    Örüntü: EmbeddingFactory
    """
    
    def __init__(self, config: TextEmbeddingConfig):
        """
        TextEmbedding modülünü başlatır.
        
        Args:
            config: Metin embedding yapılandırması
        """
        super().__init__()
        
        self.config = config
        self.tokenizer_config = config.tokenizer_config
        
        # Token embedding tablosu
        self.token_embedding = nn.Embedding(
            num_embeddings=self.tokenizer_config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.padding_idx
        )
        
        # Konum embedding'i (opsiyonel)
        self.position_embedding = None
        if config.use_position_embedding:
            self.position_embedding = nn.Embedding(
                num_embeddings=config.max_position_embeddings,
                embedding_dim=config.embed_dim
            )
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1))
            )
        
        # Embedding boyut düşürme projeksiyonu (opsiyonel)
        self.projection = None
        if config.use_embedding_projection and config.projection_dim is not None:
            self.projection = nn.Linear(config.embed_dim, config.projection_dim)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(
            config.projection_dim if config.use_embedding_projection and config.projection_dim is not None
            else config.embed_dim,
            eps=config.layer_norm_eps
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Model parametrelerini başlat
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Model ağırlıklarını başlatır."""
        # Token embedding tablosunu başlat
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Konum embedding'ini başlat (varsa)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # Projeksiyonu başlat (varsa)
        if self.projection is not None:
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.projection.bias)
        
        # LayerNorm'u başlat
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        TextEmbedding modülünün ileri geçişi.
        
        Args:
            input_ids: Token ID'lerini içeren tensör [batch_size, seq_length]
            attention_mask: Dikkat maskesi, padding'leri göz ardı etmek için [batch_size, seq_length]
            position_ids: Özel pozisyon ID'leri [batch_size, seq_length]
            return_dict: Çıktı sözlük formatında döndürülsün mü?
            
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: Embedding vektörleri [batch_size, seq_length, embed_dim]
            veya çıktı sözlüğü
        """
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        
        # Attention mask oluştur (belirtilmemişse)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
        
        # Token embedding'lerini al
        token_embeddings = self.token_embedding(input_ids)
        
        # Konum embedding'lerini al (kullanılıyorsa)
        if self.position_embedding is not None:
            # Position ID'leri oluştur (belirtilmemişse)
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            
            position_embeddings = self.position_embedding(position_ids)
            embeddings = token_embeddings + position_embeddings
        else:
            embeddings = token_embeddings
        
        # Projeksiyonu uygula (kullanılıyorsa)
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # Layer normalization ve dropout uygula
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Attention mask'ı embeddings ile çarp (opsiyonel)
        if attention_mask is not None:
            # Maskeden 3D tensor oluştur [batch_size, seq_length, 1]
            attention_mask = attention_mask.unsqueeze(-1)
            
            # Mask'ı embeddings'e uygula
            embeddings = embeddings * attention_mask
        
        if return_dict:
            return {
                "embeddings": embeddings,
                "attention_mask": attention_mask.squeeze(-1) if attention_mask is not None else None
            }
        else:
            return embeddings


class TextEmbeddingFactory:
    """
    Farklı metin embedding türleri oluşturmak için fabrika sınıfı.
    
    Bu sınıf, farklı yapılandırmalara göre metin embedding modülleri oluşturur.
    Standart metin embedding'i, özelleştirilmiş embedding'ler ve önceden eğitilmiş
    embedding modelleri destekler.
    
    Örüntü: EmbeddingFactory
    """
    
    @staticmethod
    def create_text_embedding(config: TextEmbeddingConfig) -> TextEmbedding:
        """
        Yapılandırmaya göre metin embedding modülü oluşturur.
        
        Args:
            config: Metin embedding yapılandırması
            
        Returns:
            TextEmbedding: Oluşturulan metin embedding modülü
        """
        return TextEmbedding(config)
    
    @staticmethod
    def create_compressed_embedding(
        config: TextEmbeddingConfig,
        compression_ratio: float = 0.5
    ) -> TextEmbedding:
        """
        Sıkıştırılmış metin embedding modülü oluşturur.
        
        Orijinal embedding boyutundan daha küçük bir boyuta sahip embedding oluşturur,
        ancak bilgi kaybını en aza indirmek için projeksiyonu otomatik yapılandırır.
        
        Args:
            config: Metin embedding yapılandırması
            compression_ratio: Sıkıştırma oranı (0-1 arası)
            
        Returns:
            TextEmbedding: Sıkıştırılmış metin embedding modülü
        """
        if compression_ratio <= 0 or compression_ratio >= 1:
            raise ValueError("compression_ratio must be between 0 and 1")
        
        # Yapılandırmayı kopyala ve modifiye et
        modified_config = TextEmbeddingConfig(
            embed_dim=config.embed_dim,
            tokenizer_config=config.tokenizer_config,
            padding_idx=config.padding_idx,
            use_position_embedding=config.use_position_embedding,
            max_position_embeddings=config.max_position_embeddings,
            dropout_rate=config.dropout_rate,
            layer_norm_eps=config.layer_norm_eps,
            use_embedding_projection=True,
            projection_dim=int(config.embed_dim * compression_ratio)
        )
        
        return TextEmbedding(modified_config) 