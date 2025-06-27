"""
Gelişmiş Veri Yükleme ve Önişleme Pipeline'ı

Bu modül, M³TM modeli için kapsamlı veri yükleme, önişleme ve augmentation
işlemlerini gerçekleştiren sınıfları içerir.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

from ..config.model_config import DataConfig
from ..embedding.tokenizer import SimpleTokenizer


@dataclass
class DataPipelineConfig:
    """Veri pipeline konfigürasyonu."""
    # Dosya yolları
    data_dir: str
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Veri formatı
    data_format: str = "jsonl"  # jsonl, csv, parquet
    text_column: str = "text"
    label_column: str = "label"
    image_column: Optional[str] = None
    
    # Tokenization
    max_seq_length: int = 512
    tokenizer_type: str = "simple"
    vocab_size: int = 10000
    
    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    image_channels: int = 3
    normalize_images: bool = True
    
    # Augmentation
    text_augmentation: bool = False
    image_augmentation: bool = False
    augmentation_prob: float = 0.5
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True
    
    # Multimodal
    align_modalities: bool = False
    max_pairs: Optional[int] = None


class TextAugmentation:
    """Metin augmentation işlemleri."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Basit sinonim değiştirme (demo amaçlı)."""
        if random.random() > self.prob:
            return text
        
        # Basit sinonim sözlüğü (gerçek uygulamada daha kapsamlı olmalı)
        synonyms = {
            "good": ["great", "excellent", "wonderful"],
            "bad": ["terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"]
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms and random.random() < 0.3:
                words[i] = random.choice(synonyms[word.lower()])
        
        return " ".join(words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Rastgele kelime ekleme."""
        if random.random() > self.prob:
            return text
        
        words = text.split()
        if len(words) < 2:
            return text
        
        # Basit kelime listesi
        filler_words = ["very", "really", "quite", "extremely", "somewhat"]
        
        for _ in range(n):
            if random.random() < 0.2:
                pos = random.randint(0, len(words))
                words.insert(pos, random.choice(filler_words))
        
        return " ".join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Rastgele kelime silme."""
        if random.random() > self.prob:
            return text
        
        words = text.split()
        if len(words) <= 2:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        return " ".join(new_words) if new_words else text
    
    def __call__(self, text: str) -> str:
        """Rastgele augmentation uygula."""
        augmentations = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_deletion
        ]
        
        aug_func = random.choice(augmentations)
        return aug_func(text)


class ImageAugmentation:
    """Görüntü augmentation işlemleri."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), prob: float = 0.5):
        self.prob = prob
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.no_aug_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Augmentation uygula."""
        if random.random() < self.prob:
            return self.transform(image)
        else:
            return self.no_aug_transform(image)


class MultimodalDataset(Dataset):
    """Multimodal veri kümesi sınıfı."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        config: DataPipelineConfig,
        tokenizer: Optional[Callable] = None,
        text_augmentation: Optional[TextAugmentation] = None,
        image_augmentation: Optional[ImageAugmentation] = None,
        split: str = "train"
    ):
        self.data = data
        self.config = config
        self.split = split
        
        # Tokenizer
        if tokenizer is None:
            from ..embedding.config import TokenizerConfig
            tokenizer_config = TokenizerConfig(
                vocab_size=config.vocab_size,
                max_seq_length=config.max_seq_length
            )
            self.tokenizer = SimpleTokenizer(tokenizer_config)
            # Basit vocab oluştur
            all_texts = [item.get(config.text_column, "") for item in data]
            self.tokenizer.build_vocab(all_texts)
        else:
            self.tokenizer = tokenizer
        
        # Augmentation
        self.text_augmentation = text_augmentation if split == "train" else None
        self.image_augmentation = image_augmentation if split == "train" else None
        
        # Image transform (augmentation olmadan)
        if not self.image_augmentation:
            self.image_transform = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        result = {}
        
        # Text processing
        text = item.get(self.config.text_column, "")
        if self.text_augmentation and text:
            text = self.text_augmentation(text)
        
        if text:
            # Tokenize
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Padding/truncation
            if len(token_ids) > self.config.max_seq_length:
                token_ids = token_ids[:self.config.max_seq_length]
            else:
                token_ids.extend([self.tokenizer.pad_token_id] * (self.config.max_seq_length - len(token_ids)))
            
            result["input_ids"] = torch.tensor(token_ids, dtype=torch.long)
            result["attention_mask"] = torch.tensor(
                [1 if tid != self.tokenizer.pad_token_id else 0 for tid in token_ids],
                dtype=torch.long
            )
        
        # Image processing
        if self.config.image_column and self.config.image_column in item:
            image_path = item[self.config.image_column]
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    if self.image_augmentation:
                        result["pixel_values"] = self.image_augmentation(image)
                    else:
                        result["pixel_values"] = self.image_transform(image)
                except Exception as e:
                    logging.warning(f"Görüntü yüklenemedi {image_path}: {e}")
                    # Dummy image
                    result["pixel_values"] = torch.zeros(3, *self.config.image_size)
        
        # Label processing
        if self.config.label_column in item:
            label = item[self.config.label_column]
            if isinstance(label, str):
                # String label'ı integer'a çevir (basit mapping)
                label_map = getattr(self, 'label_map', {})
                if label not in label_map:
                    label_map[label] = len(label_map)
                    self.label_map = label_map
                label = label_map[label]
            result["labels"] = torch.tensor(label, dtype=torch.long)
        
        return result


class DataPipeline:
    """Ana veri pipeline sınıfı."""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cache dizini
        if config.cache_dir:
            os.makedirs(config.cache_dir, exist_ok=True)
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Veri dosyasını yükler."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {file_path}")
        
        self.logger.info(f"Veri yükleniyor: {file_path}")
        
        if self.config.data_format == "jsonl":
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        
        elif self.config.data_format == "csv":
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        
        elif self.config.data_format == "parquet":
            df = pd.read_parquet(file_path)
            return df.to_dict('records')
        
        else:
            raise ValueError(f"Desteklenmeyen veri formatı: {self.config.data_format}")
    
    def create_datasets(self) -> Tuple[MultimodalDataset, Optional[MultimodalDataset], Optional[MultimodalDataset]]:
        """Train, validation ve test veri setlerini oluşturur."""
        datasets = []
        
        # Augmentation nesneleri
        text_aug = TextAugmentation(self.config.augmentation_prob) if self.config.text_augmentation else None
        image_aug = ImageAugmentation(self.config.image_size, self.config.augmentation_prob) if self.config.image_augmentation else None
        
        # Train dataset
        if self.config.train_file:
            train_data = self.load_data(os.path.join(self.config.data_dir, self.config.train_file))
            train_dataset = MultimodalDataset(
                data=train_data,
                config=self.config,
                text_augmentation=text_aug,
                image_augmentation=image_aug,
                split="train"
            )
            datasets.append(train_dataset)
        else:
            datasets.append(None)
        
        # Validation dataset
        if self.config.val_file:
            val_data = self.load_data(os.path.join(self.config.data_dir, self.config.val_file))
            val_dataset = MultimodalDataset(
                data=val_data,
                config=self.config,
                tokenizer=datasets[0].tokenizer if datasets[0] else None,
                split="val"
            )
            datasets.append(val_dataset)
        else:
            datasets.append(None)
        
        # Test dataset
        if self.config.test_file:
            test_data = self.load_data(os.path.join(self.config.data_dir, self.config.test_file))
            test_dataset = MultimodalDataset(
                data=test_data,
                config=self.config,
                tokenizer=datasets[0].tokenizer if datasets[0] else None,
                split="test"
            )
            datasets.append(test_dataset)
        else:
            datasets.append(None)
        
        return tuple(datasets)
    
    def create_dataloaders(
        self,
        train_dataset: Optional[MultimodalDataset],
        val_dataset: Optional[MultimodalDataset],
        test_dataset: Optional[MultimodalDataset],
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """DataLoader nesnelerini oluşturur."""
        dataloaders = []
        
        for dataset, shuffle in [(train_dataset, True), (val_dataset, False), (test_dataset, False)]:
            if dataset:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    drop_last=shuffle  # Train için drop_last=True
                )
                dataloaders.append(loader)
            else:
                dataloaders.append(None)
        
        return tuple(dataloaders)
