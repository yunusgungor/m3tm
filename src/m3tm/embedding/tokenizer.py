"""
Tokenizer Sınıfları ve Fonksiyonları

Bu modül, metin tokenizasyonu için sınıfları ve fonksiyonları içerir.
Metinleri token ID'lerine dönüştürmek için kullanılır.
Basit tokenizer ve ileri düzey tokenizer seçenekleri sunar.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Set, Union, Tuple
from collections import Counter
import torch

from .config import TokenizerConfig


class SimpleTokenizer:
    """
    Basit ve hızlı bir metin tokenizer.
    
    Bu tokenizer, metni kelimelerine ayırır ve her kelimeyi bir ID'ye dönüştürür.
    Kelime tabanlı tokenizasyon kullanarak, karakter veya alt-kelime tabanlı
    tokenizasyona göre daha hızlı çalışır. Mobil cihazlar için optimize edilmiştir.
    
    Örüntü: TokenProcessingStrategy
    """
    
    def __init__(self, config: TokenizerConfig):
        """
        SimpleTokenizer sınıfını başlatır.
        
        Args:
            config: Tokenizer yapılandırması
        """
        self.config = config
        
        # Sözlük ve ters sözlük
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        
        # Özel token sözlüğü
        self.special_tokens = config.special_tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            
        # Öntanımlı değerler
        self.pad_token_id = self.special_tokens.get("<PAD>", 0)
        self.unk_token_id = self.special_tokens.get("<UNK>", 1)
        self.bos_token_id = self.special_tokens.get("<BOS>", 2)
        self.eos_token_id = self.special_tokens.get("<EOS>", 3)
        
        # Tokenize için düzenli ifade
        self.pattern = re.compile(r'\S+')
    
    def _preprocess_text(self, text: str) -> str:
        """
        Metni tokenizasyon için ön işler.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            str: Ön işlenmiş metin
        """
        if self.config.lowercase:
            text = text.lower()
            
        if self.config.strip_accents:
            text = unicodedata.normalize('NFKD', text)
            text = ''.join([c for c in text if not unicodedata.combining(c)])
            
        return text
    
    def _split_text(self, text: str) -> List[str]:
        """
        Metni token dizisine ayırır.
        
        Args:
            text: Ayrılacak metin
            
        Returns:
            List[str]: Token dizisi
        """
        return self.pattern.findall(text)
    
    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """
        Metinlerden sözlük oluşturur.
        
        Args:
            texts: Sözlük oluşturmak için kullanılacak metin listesi
            min_freq: Bir kelimenin sözlüğe eklenmesi için gereken minimum frekans
        """
        # Kelime frekanslarını hesapla
        counter = Counter()
        for text in texts:
            preprocessed = self._preprocess_text(text)
            tokens = self._split_text(preprocessed)
            counter.update(tokens)
        
        # Min. frekansa göre filtrele
        filtered_tokens = [token for token, count in counter.items() if count >= min_freq]
        
        # Sözlük boyutunu kontrol et
        vocab_size = self.config.vocab_size
        available_slots = vocab_size - len(self.special_tokens)
        
        if len(filtered_tokens) > available_slots:
            # En sık kullanılan kelimeleri al
            most_common = [token for token, _ in counter.most_common(available_slots)]
            filtered_tokens = most_common
        
        # Sözlük oluştur
        next_idx = max(self.special_tokens.values()) + 1
        
        for token in filtered_tokens:
            if token not in self.vocab:
                self.vocab[token] = next_idx
                self.reverse_vocab[next_idx] = token
                next_idx += 1
    
    def save_vocab(self, file_path: str) -> None:
        """
        Sözlüğü dosyaya kaydeder.
        
        Args:
            file_path: Sözlüğün kaydedileceği dosya yolu
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
    
    def load_vocab(self, file_path: str) -> None:
        """
        Sözlüğü dosyadan yükler.
        
        Args:
            file_path: Sözlüğün yükleneceği dosya yolu
        """
        self.vocab = {}
        self.reverse_vocab = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idx_str = line.strip().split('\t')
                idx = int(idx_str)
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Metni token ID'lerine dönüştürür.
        
        Args:
            text: Tokenize edilecek metin
            add_special_tokens: Başlangıç ve bitiş tokenleri eklensin mi
            
        Returns:
            List[int]: Token ID'leri
        """
        preprocessed = self._preprocess_text(text)
        tokens = self._split_text(preprocessed)
        
        # Token'ları ID'lere dönüştür
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Özel tokenleri ekle
        if add_special_tokens:
            if self.config.add_bos_token:
                token_ids.insert(0, self.bos_token_id)
            if self.config.add_eos_token:
                token_ids.append(self.eos_token_id)
        
        # Maksimum uzunluğu kontrol et
        max_len = self.config.max_seq_length
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            # EOS token'ı sona ekle (eklenebiliyorsa ve isteniyorsa)
            if self.config.add_eos_token and add_special_tokens and token_ids[-1] != self.eos_token_id:
                token_ids[-1] = self.eos_token_id
                
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Token ID'lerini metne dönüştürür.
        
        Args:
            token_ids: Dönüştürülecek token ID'leri
            skip_special_tokens: Özel tokenleri atla
            
        Returns:
            str: Dönüştürülmüş metin
        """
        # Özel tokenleri filtrele
        if skip_special_tokens:
            special_ids = set(self.special_tokens.values())
            token_ids = [idx for idx in token_ids if idx not in special_ids]
        
        # ID'leri tokenlere dönüştür
        tokens = [self.reverse_vocab.get(idx, "<UNK>") for idx in token_ids]
        
        # Tokenleri birleştir
        return " ".join(tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True, 
                    padding: bool = True, truncation: bool = True) -> List[List[int]]:
        """
        Metin listesini token ID listelerine dönüştürür.
        
        Args:
            texts: Tokenize edilecek metin listesi
            add_special_tokens: Başlangıç ve bitiş tokenleri eklensin mi
            padding: Çıktı dizilerinin uzunluğu eşitlensin mi
            truncation: Uzun diziler kısaltılsın mı
            
        Returns:
            List[List[int]]: Token ID'lerinin listesi
        """
        batch_token_ids = []
        
        # Her metni tokenize et
        for text in texts:
            token_ids = self.encode(text, add_special_tokens=add_special_tokens)
            
            # Kısaltma
            if truncation and len(token_ids) > self.config.max_seq_length:
                token_ids = token_ids[:self.config.max_seq_length]
                # EOS token'ı sona ekle (eklenebiliyorsa ve isteniyorsa)
                if self.config.add_eos_token and add_special_tokens and token_ids[-1] != self.eos_token_id:
                    token_ids[-1] = self.eos_token_id
            
            batch_token_ids.append(token_ids)
        
        # Padding
        if padding:
            # En uzun diziyi bul
            max_len = max(len(ids) for ids in batch_token_ids)
            
            # Padding uygula
            for i, token_ids in enumerate(batch_token_ids):
                padding_len = max_len - len(token_ids)
                if padding_len > 0:
                    batch_token_ids[i] = token_ids + [self.pad_token_id] * padding_len
        
        return batch_token_ids
    
    def encode_to_tensors(self, texts: Union[str, List[str]], add_special_tokens: bool = True, 
                         padding: bool = True, truncation: bool = True) -> torch.Tensor:
        """
        Metni PyTorch tensörlerine dönüştürür.
        
        Args:
            texts: Tokenize edilecek metin veya metin listesi
            add_special_tokens: Başlangıç ve bitiş tokenleri eklensin mi
            padding: Çıktı dizilerinin uzunluğu eşitlensin mi
            truncation: Uzun diziler kısaltılsın mı
            
        Returns:
            torch.Tensor: Token ID'lerini içeren tensör
        """
        if isinstance(texts, str):
            texts = [texts]
        
        batch_token_ids = self.encode_batch(
            texts, 
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation
        )
        
        return torch.tensor(batch_token_ids, dtype=torch.long)
    
    def __len__(self) -> int:
        """
        Sözlük boyutunu döndürür.
        
        Returns:
            int: Sözlük boyutu
        """
        return len(self.vocab) 