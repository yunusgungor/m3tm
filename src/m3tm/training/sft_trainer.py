"""
Supervised Fine-Tuning (SFT) Trainer for M³TM

Bu modül, M³TM modeli için supervised fine-tuning implementasyonu sağlar.
SFT, modeli belirli görevler için instruction-following formatında eğitir.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from .base_trainer import BaseTrainer
from ..core.base_model import M3TMBaseModel


@dataclass
class SFTConfig:
    """SFT eğitim konfigürasyonu"""
    # Temel eğitim parametreleri
    learning_rate: float = 2e-5
    batch_size: int = 4
    epochs: int = 3
    max_seq_length: int = 512
    
    # SFT özel parametreleri
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    response_template: str = "### Response:\n"
    ignore_index: int = -100
    
    # Regularization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # Loss masking
    mask_instruction_tokens: bool = True  # Sadece response tokenlarında loss hesapla
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500


class SFTDataset(Dataset):
    """SFT için instruction-response dataset"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        config: SFTConfig,
        mode: str = "train"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SFTDataset oluşturuldu: {len(data)} örnek, mode: {mode}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tek bir eğitim örneği döndürür"""
        item = self.data[idx]
        
        # Instruction ve response'u formatla
        instruction = item.get("instruction", "")
        response = item.get("response", "")
        
        # Template'i uygula
        full_text = self.config.instruction_template.format(
            instruction=instruction,
            response=response
        )
        
        # SimpleTokenizer ile tokenize et
        token_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Truncation
        if len(token_ids) > self.config.max_seq_length:
            token_ids = token_ids[:self.config.max_seq_length]

        # Padding
        pad_length = self.config.max_seq_length - len(token_ids)
        if pad_length > 0:
            token_ids.extend([self.tokenizer.pad_token_id] * pad_length)

        # Attention mask oluştur
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in token_ids]

        # Tensor'e dönüştür
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Loss masking için labels oluştur
        labels = input_ids.clone()
        
        if self.config.mask_instruction_tokens:
            # Sadece response kısmında loss hesapla
            response_start = self._find_response_start(full_text, input_ids)
            if response_start > 0:
                labels[:response_start] = self.config.ignore_index
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": full_text  # Debug için
        }
    
    def _find_response_start(self, full_text: str, input_ids: torch.Tensor) -> int:
        """Response başlangıcını bulur"""
        try:
            response_marker = self.config.response_template
            response_start_char = full_text.find(response_marker)

            if response_start_char == -1:
                return 0

            # Character pozisyonunu token pozisyonuna çevir
            prefix_text = full_text[:response_start_char + len(response_marker)]
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)

            return len(prefix_tokens)
        except:
            return 0


class SFTTrainer(BaseTrainer):
    """Supervised Fine-Tuning Trainer"""
    
    def __init__(
        self,
        model: M3TMBaseModel,
        config: SFTConfig,
        tokenizer,
        save_dir: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir) if save_dir else Path("./sft_checkpoints")
        
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model'i device'a taşı
        self.model.to(self.device)
        
        # Checkpoint dizinini oluştur
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"SFTTrainer oluşturuldu - Device: {self.device}")
    
    def create_dataset(
        self,
        data_path: str,
        mode: str = "train"
    ) -> SFTDataset:
        """SFT dataset oluşturur"""
        # JSON Lines formatında veri yükle
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        return SFTDataset(data, self.tokenizer, self.config, mode)
    
    def train(
        self,
        train_dataset: SFTDataset,
        val_dataset: Optional[SFTDataset] = None
    ) -> Dict[str, Any]:
        """SFT eğitimini çalıştırır"""
        self.logger.info("SFT eğitimi başlıyor...")
        
        # DataLoader oluştur
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        # Optimizer ve scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Eğitim döngüsü
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # Training
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, global_step)
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                
                # Best model kaydet
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, train_loss, val_loss, is_best=True)
            
            # Regular checkpoint kaydet
            if (epoch + 1) % 1 == 0:  # Her epoch'ta kaydet
                self._save_checkpoint(epoch, train_loss, val_loss)
            
            global_step += len(train_loader)
        
        self.logger.info("SFT eğitimi tamamlandı!")
        
        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "total_steps": global_step
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        global_step: int
    ) -> float:
        """Tek epoch eğitim"""
        self.model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Batch'i device'a taşı
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            text_input = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            
            outputs = self.model(text_input=text_input)
            logits = outputs.get('logits')
            
            if logits is None:
                raise ValueError("Model logits döndürmüyor!")
            
            # Loss hesapla
            labels = batch['labels']
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.config.ignore_index
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Logging
            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Step {global_step + step + 1}: "
                    f"Loss = {loss.item():.4f}, "
                    f"Avg Loss = {avg_loss:.4f}, "
                    f"LR = {lr:.2e}"
                )
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validation epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Batch'i device'a taşı
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                text_input = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }
                
                outputs = self.model(text_input=text_input)
                logits = outputs.get('logits')
                
                # Loss hesapla
                labels = batch['labels']
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.config.ignore_index
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        is_best: bool = False
    ):
        """Checkpoint kaydet"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model kaydedildi: {best_path}")
