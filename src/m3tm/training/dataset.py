"""
Veri Kümesi Modülü

Bu modül, M³TM modeli için veri kümesi sınıflarını içerir.
Metin ve görüntü verilerini işlemek için PyTorch Dataset sınıflarını sağlar.

Örüntüler:
- DatasetFactory (PT-007): Farklı veri kümesi türlerini oluşturmak için fabrika deseni
- DataPreprocessingPipeline (PT-010): Veri önişleme adımlarını modüler hale getirir
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from m3tm.config.model_config import DataConfig


class TextClassificationDataset(Dataset):
    """
    Metin sınıflandırma için veri kümesi sınıfı.
    
    Bu sınıf, metin verilerini ve etiketlerini işler ve model eğitimi için uygun
    formatta döndürür.
    
    Örüntüler:
    - DataPreprocessingPipeline (PT-010): Veri önişleme adımlarını modüler hale getirir
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Callable,
        max_length: int = 128,
        label_map: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None
    ):
        """
        TextClassificationDataset sınıfını başlatır.
        
        Args:
            texts: Metin listesi
            labels: Etiket listesi (sınıf indeksleri)
            tokenizer: Metinleri tokenize eden fonksiyon
            max_length: Maksimum token uzunluğu
            label_map: Etiket isimlerini indekslere eşleyen sözlük
            transform: Veri dönüşümü için fonksiyon
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or {}
        self.transform = transform
        
        # Veri doğrulama
        assert len(texts) == len(labels), "Metin ve etiket sayıları eşleşmiyor"
    
    def __len__(self) -> int:
        """
        Veri kümesinin uzunluğunu döndürür.
        
        Returns:
            int: Veri kümesi uzunluğu
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Belirtilen indeksteki veriyi döndürür.
        
        Args:
            idx: Veri indeksi
            
        Returns:
            Dict[str, torch.Tensor]: Tokenize edilmiş metin ve etiketi içeren sözlük
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Batch boyutunu kaldır
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Dönüşüm uygula
        if self.transform is not None:
            input_ids, attention_mask = self.transform(input_ids, attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }
    
    @classmethod
    def from_json(
        cls,
        json_path: Union[str, Path],
        tokenizer: Callable,
        text_field: str = "text",
        label_field: str = "label",
        max_length: int = 128,
        transform: Optional[Callable] = None
    ) -> "TextClassificationDataset":
        """
        JSON dosyasından veri kümesi oluşturur.
        
        Args:
            json_path: JSON dosya yolu
            tokenizer: Tokenizer fonksiyonu
            text_field: Metin alanının adı
            label_field: Etiket alanının adı
            max_length: Maksimum token uzunluğu
            transform: Veri dönüşümü için fonksiyon
            
        Returns:
            TextClassificationDataset: Oluşturulan veri kümesi
        """
        if isinstance(json_path, str):
            json_path = Path(json_path)
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = []
        labels = []
        label_set = set()
        
        for item in data:
            texts.append(item[text_field])
            labels.append(item[label_field])
            label_set.add(item[label_field])
        
        # Etiketleri sayısal indekslere dönüştür
        if isinstance(labels[0], str):
            label_map = {label: idx for idx, label in enumerate(sorted(label_set))}
            labels = [label_map[label] for label in labels]
        else:
            label_map = None
        
        return cls(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
            label_map=label_map,
            transform=transform
        )
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple["TextClassificationDataset", "TextClassificationDataset", "TextClassificationDataset"]:
        """
        Veri kümesini eğitim, doğrulama ve test kümelerine böler.
        
        Args:
            train_ratio: Eğitim kümesi oranı
            val_ratio: Doğrulama kümesi oranı
            test_ratio: Test kümesi oranı
            seed: Rastgele tohum
            
        Returns:
            Tuple[TextClassificationDataset, TextClassificationDataset, TextClassificationDataset]:
                Eğitim, doğrulama ve test kümeleri
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
            "Oranların toplamı 1.0 olmalıdır"
        
        # Veri indekslerini karıştır
        indices = list(range(len(self)))
        random.seed(seed)
        random.shuffle(indices)
        
        # Bölme noktalarını hesapla
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)
        
        # İndeksleri böl
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Alt kümeleri oluştur
        train_texts = [self.texts[i] for i in train_indices]
        train_labels = [self.labels[i] for i in train_indices]
        
        val_texts = [self.texts[i] for i in val_indices]
        val_labels = [self.labels[i] for i in val_indices]
        
        test_texts = [self.texts[i] for i in test_indices]
        test_labels = [self.labels[i] for i in test_indices]
        
        # Veri kümelerini oluştur
        train_dataset = TextClassificationDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label_map=self.label_map,
            transform=self.transform
        )
        
        val_dataset = TextClassificationDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label_map=self.label_map,
            transform=self.transform
        )
        
        test_dataset = TextClassificationDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label_map=self.label_map,
            transform=self.transform
        )
        
        return train_dataset, val_dataset, test_dataset


class DatasetFactory:
    """
    Veri kümesi oluşturmak için fabrika sınıfı.
    
    Bu sınıf, farklı veri kümesi türlerini oluşturmak için yöntemler sağlar.
    
    Örüntü: DatasetFactory (PT-007)
    """
    
    @staticmethod
    def create_text_classification_dataset(
        config: DataConfig,
        tokenizer: Callable,
        data_path: Optional[Union[str, Path]] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None
    ) -> TextClassificationDataset:
        """
        Metin sınıflandırma veri kümesi oluşturur.
        
        Args:
            config: Veri yapılandırması
            tokenizer: Tokenizer fonksiyonu
            data_path: Veri dosyası yolu (JSON)
            texts: Metin listesi (data_path belirtilmezse kullanılır)
            labels: Etiket listesi (data_path belirtilmezse kullanılır)
            transform: Veri dönüşümü için fonksiyon
            
        Returns:
            TextClassificationDataset: Oluşturulan veri kümesi
        """
        if data_path is not None:
            return TextClassificationDataset.from_json(
                json_path=data_path,
                tokenizer=tokenizer,
                text_field=config.text_field,
                label_field=config.label_field,
                max_length=config.max_length,
                transform=transform
            )
        else:
            assert texts is not None and labels is not None, \
                "data_path veya texts/labels belirtilmelidir"
            
            return TextClassificationDataset(
                texts=texts,
                labels=labels,
                tokenizer=tokenizer,
                max_length=config.max_length,
                transform=transform
            )
    
    @staticmethod
    def create_dataloaders(
        dataset: Dataset,
        batch_size: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Veri kümesinden DataLoader nesneleri oluşturur.
        
        Args:
            dataset: Veri kümesi
            batch_size: Batch boyutu
            train_ratio: Eğitim kümesi oranı
            val_ratio: Doğrulama kümesi oranı
            test_ratio: Test kümesi oranı
            seed: Rastgele tohum
            num_workers: İş parçacığı sayısı
            pin_memory: Belleği pinle
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Eğitim, doğrulama ve test DataLoader'ları
        """
        if hasattr(dataset, "split"):
            train_dataset, val_dataset, test_dataset = dataset.split(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed
            )
        else:
            # Veri kümesini manuel olarak böl
            indices = torch.randperm(len(dataset)).tolist()
            train_end = int(len(indices) * train_ratio)
            val_end = train_end + int(len(indices) * val_ratio)
            
            train_dataset = torch.utils.data.Subset(dataset, indices[:train_end])
            val_dataset = torch.utils.data.Subset(dataset, indices[train_end:val_end])
            test_dataset = torch.utils.data.Subset(dataset, indices[val_end:])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader 