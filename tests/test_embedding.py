"""
TextEmbedding Modülü için Birim Testler

Bu modül, embedding modülünün doğru çalıştığını doğrulamak için testler içerir.
"""

import unittest
import torch
import tempfile
import os
from pathlib import Path

from m3tm.embedding.config import TokenizerConfig, TextEmbeddingConfig
from m3tm.embedding.tokenizer import SimpleTokenizer
from m3tm.embedding.text_embedding import TextEmbedding, TextEmbeddingFactory


class TestTokenizer(unittest.TestCase):
    """SimpleTokenizer sınıfı için test süiti."""
    
    def setUp(self):
        """Test için hazırlık."""
        self.config = TokenizerConfig(
            vocab_size=1000,
            max_seq_length=128,
            special_tokens={
                "<PAD>": 0,
                "<UNK>": 1,
                "<BOS>": 2,
                "<EOS>": 3,
            }
        )
        self.tokenizer = SimpleTokenizer(self.config)
        
        # Örnek metinler
        self.sample_texts = [
            "Bu bir test cümlesidir.",
            "Tokenizer'ın düzgün çalıştığını test etmek için yazılmıştır.",
            "Özel karakterler ve rakamlar da içerebilir: 123, $%&!",
        ]
        
        # Sözlük oluştur
        self.tokenizer.build_vocab(self.sample_texts)
    
    def test_init(self):
        """Tokenizer başlatma testleri."""
        self.assertEqual(self.tokenizer.config.vocab_size, 1000)
        self.assertEqual(self.tokenizer.config.max_seq_length, 128)
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.unk_token_id, 1)
    
    def test_encode_decode(self):
        """Encode ve decode işlemleri testleri."""
        for text in self.sample_texts:
            token_ids = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(token_ids)
            
            # Özel tokenler çıkarıldığından birebir aynı olmayabilir
            # Bu nedenle temel kelimeler mevcut mu kontrol ediyoruz
            for word in text.lower().split():
                # Noktalama işaretlerini temizle
                word = ''.join(c for c in word if c.isalnum())
                if word:  # Boş string değilse
                    self.assertIn(word, decoded.lower())
    
    def test_encode_batch(self):
        """Batch encode testleri."""
        batch_tokens = self.tokenizer.encode_batch(self.sample_texts)
        
        # Tüm dizilerin aynı uzunlukta olduğunu kontrol et (padding)
        lengths = [len(tokens) for tokens in batch_tokens]
        self.assertEqual(min(lengths), max(lengths))
    
    def test_encode_to_tensors(self):
        """Tensor dönüştürme testi."""
        tensor = self.tokenizer.encode_to_tensors(self.sample_texts)
        
        # Tensör boyutlarını kontrol et
        self.assertEqual(len(tensor.shape), 2)
        self.assertEqual(tensor.shape[0], len(self.sample_texts))
    
    def test_vocab_save_load(self):
        """Sözlük kaydetme ve yükleme testi."""
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Sözlüğü kaydet
            self.tokenizer.save_vocab(temp_path)
            
            # Yeni tokenizer oluştur
            new_tokenizer = SimpleTokenizer(self.config)
            
            # Sözlüğü yükle
            new_tokenizer.load_vocab(temp_path)
            
            # Sözlükleri karşılaştır
            self.assertEqual(len(self.tokenizer.vocab), len(new_tokenizer.vocab))
            for token, idx in self.tokenizer.vocab.items():
                self.assertEqual(new_tokenizer.vocab.get(token), idx)
        finally:
            os.unlink(temp_path)


class TestTextEmbedding(unittest.TestCase):
    """TextEmbedding sınıfı için test süiti."""
    
    def setUp(self):
        """Test için hazırlık."""
        self.tokenizer_config = TokenizerConfig(vocab_size=1000, max_seq_length=128)
        self.config = TextEmbeddingConfig(
            embed_dim=256,
            tokenizer_config=self.tokenizer_config,
            padding_idx=0,
            use_position_embedding=True,
            max_position_embeddings=128,
            dropout_rate=0.1
        )
        self.model = TextEmbedding(self.config)
        
        # Örnek girdi
        self.batch_size = 4
        self.seq_length = 16
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        self.attention_mask = torch.ones((self.batch_size, self.seq_length))
    
    def test_init(self):
        """Model başlatma testleri."""
        self.assertEqual(self.model.config.embed_dim, 256)
        self.assertEqual(self.model.tokenizer_config.vocab_size, 1000)
        self.assertIsNotNone(self.model.token_embedding)
        self.assertIsNotNone(self.model.position_embedding)
        self.assertIsNone(self.model.projection)
        self.assertIsNotNone(self.model.layer_norm)
        self.assertIsNotNone(self.model.dropout)
    
    def test_forward(self):
        """Forward pass testleri."""
        # Dict dönüşlü ileri geçiş
        output_dict = self.model(self.input_ids, self.attention_mask)
        
        # Çıktı türünü kontrol et
        self.assertIsInstance(output_dict, dict)
        self.assertIn("embeddings", output_dict)
        self.assertIn("attention_mask", output_dict)
        
        # Embeddings boyutunu kontrol et
        embeddings = output_dict["embeddings"]
        self.assertEqual(embeddings.shape, (self.batch_size, self.seq_length, self.config.embed_dim))
        
        # Tensor dönüşlü ileri geçiş
        embeddings = self.model(self.input_ids, self.attention_mask, return_dict=False)
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape, (self.batch_size, self.seq_length, self.config.embed_dim))
    
    def test_no_position_embedding(self):
        """Konum embedding'i olmadan test."""
        config = TextEmbeddingConfig(
            embed_dim=256,
            tokenizer_config=self.tokenizer_config,
            padding_idx=0,
            use_position_embedding=False,
            dropout_rate=0.1
        )
        model = TextEmbedding(config)
        
        output = model(self.input_ids, self.attention_mask)
        self.assertEqual(output["embeddings"].shape, (self.batch_size, self.seq_length, config.embed_dim))
    
    def test_projection(self):
        """Boyut düşürme projeksiyonu testi."""
        config = TextEmbeddingConfig(
            embed_dim=256,
            tokenizer_config=self.tokenizer_config,
            padding_idx=0,
            use_position_embedding=True,
            max_position_embeddings=128,
            dropout_rate=0.1,
            use_embedding_projection=True,
            projection_dim=128
        )
        model = TextEmbedding(config)
        
        output = model(self.input_ids, self.attention_mask)
        self.assertEqual(output["embeddings"].shape, (self.batch_size, self.seq_length, config.projection_dim))
    
    def test_embedding_factory(self):
        """TextEmbeddingFactory testi."""
        # Standart embedding
        model = TextEmbeddingFactory.create_text_embedding(self.config)
        self.assertIsInstance(model, TextEmbedding)
        
        # Sıkıştırılmış embedding
        compressed_model = TextEmbeddingFactory.create_compressed_embedding(
            self.config, compression_ratio=0.5
        )
        self.assertIsInstance(compressed_model, TextEmbedding)
        self.assertTrue(compressed_model.projection is not None)
        
        # Compressed output'un boyutunu kontrol et
        output = compressed_model(self.input_ids, self.attention_mask)
        self.assertEqual(output["embeddings"].shape, 
                         (self.batch_size, self.seq_length, int(self.config.embed_dim * 0.5)))


class TestEndToEnd(unittest.TestCase):
    """TextEmbedding modülü için uçtan uca testler."""
    
    def test_tokenizer_to_embedding(self):
        """Tokenizer'dan embedding'e uçtan uca test."""
        # Yapılandırmaları oluştur
        tokenizer_config = TokenizerConfig(vocab_size=1000, max_seq_length=128)
        embedding_config = TextEmbeddingConfig(
            embed_dim=256,
            tokenizer_config=tokenizer_config,
            padding_idx=0,
            use_position_embedding=True,
            max_position_embeddings=128,
            dropout_rate=0.1
        )
        
        # Tokenizer ve embedding'i oluştur
        tokenizer = SimpleTokenizer(tokenizer_config)
        embedding_model = TextEmbedding(embedding_config)
        
        # Örnek metinler
        texts = [
            "Bu bir test cümlesidir.",
            "TextEmbedding'in doğru çalıştığını test eden bir cümledir."
        ]
        
        # Metinleri tokenize et
        tokenizer.build_vocab(texts)
        input_tensor = tokenizer.encode_to_tensors(texts)
        
        # Embedding'e ilet
        output = embedding_model(input_tensor)
        
        # Çıktıyı kontrol et
        self.assertIsNotNone(output["embeddings"])
        self.assertEqual(output["embeddings"].shape[0], len(texts))
        self.assertEqual(output["embeddings"].shape[2], embedding_config.embed_dim)


if __name__ == "__main__":
    unittest.main() 