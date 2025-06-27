"""
Gemini-based Knowledge Distillation for M³TM

Bu modül, Gemini-2.5-Flash'ı teacher model olarak kullanarak
M³TM modelini knowledge distillation ile eğitir.
"""

import os
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .base_trainer import BaseTrainer
from .gemini_integration import GeminiClient, GeminiConfig
from ..core.base_model import M3TMBaseModel


@dataclass
class DistillationConfig:
    """Knowledge distillation konfigürasyonu"""
    # Temel eğitim parametreleri
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 3
    max_seq_length: int = 128
    
    # Distillation parametreleri
    temperature: float = 4.0  # Softmax temperature
    alpha: float = 0.7  # Hard target vs soft target balance
    beta: float = 0.3   # Student loss vs distillation loss balance
    
    # Gemini parametreleri
    gemini_temperature: float = 0.3  # Gemini için düşük temperature (daha tutarlı)
    max_teacher_samples: int = 1000  # Teacher'dan alınacak maksimum örnek
    
    # Regularization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500


class GeminiDistillationDataset:
    """Gemini teacher outputs ile distillation dataset"""
    
    def __init__(
        self,
        prompts: List[str],
        gemini_client: GeminiClient,
        tokenizer,
        config: DistillationConfig,
        cache_path: Optional[str] = None
    ):
        self.prompts = prompts
        self.gemini_client = gemini_client
        self.tokenizer = tokenizer
        self.config = config
        self.cache_path = Path(cache_path) if cache_path else None
        
        self.logger = logging.getLogger(__name__)
        
        # Teacher outputs'u yükle veya üret
        self.teacher_outputs = self._load_or_generate_teacher_outputs()
        
        self.logger.info(f"DistillationDataset oluşturuldu: {len(self.teacher_outputs)} örnek")
    
    def _load_or_generate_teacher_outputs(self) -> List[Dict[str, Any]]:
        """Teacher outputs'u cache'den yükle veya Gemini'den üret"""
        
        # Cache'den yüklemeyi dene
        if self.cache_path and self.cache_path.exists():
            self.logger.info(f"Teacher outputs cache'den yükleniyor: {self.cache_path}")
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cached_data = [json.loads(line) for line in f]
                
                if len(cached_data) >= len(self.prompts):
                    return cached_data[:len(self.prompts)]
                else:
                    self.logger.warning("Cache eksik, yeniden üretiliyor...")
            except Exception as e:
                self.logger.error(f"Cache yükleme hatası: {e}")
        
        # Gemini'den teacher outputs üret
        return self._generate_teacher_outputs()
    
    def _generate_teacher_outputs(self) -> List[Dict[str, Any]]:
        """Gemini'den teacher outputs üret"""
        self.logger.info("Gemini'den teacher outputs üretiliyor...")
        
        system_prompt = """Sen bir uzman eğitim asistanısın. Verilen prompt'lara en iyi şekilde cevap ver.
        
Kurallar:
1. Doğru ve detaylı bilgi ver
2. Açık ve anlaşılır ol
3. Örnekler kullan
4. Yapılandırılmış cevaplar ver"""

        teacher_outputs = []
        
        # Batch'ler halinde işle
        batch_size = 5
        for i in range(0, len(self.prompts), batch_size):
            batch_prompts = self.prompts[i:i + batch_size]
            
            for prompt in batch_prompts:
                try:
                    # Gemini'den response al
                    response = self.gemini_client.generate_text(
                        prompt, 
                        system_prompt
                    )
                    
                    teacher_outputs.append({
                        "prompt": prompt,
                        "teacher_response": response,
                        "teacher_logits": None  # API'den logits alamayız
                    })
                    
                except Exception as e:
                    self.logger.error(f"Teacher output üretim hatası: {e}")
                    # Boş response ekle
                    teacher_outputs.append({
                        "prompt": prompt,
                        "teacher_response": "",
                        "teacher_logits": None
                    })
            
            self.logger.info(f"İşlenen: {min(i + batch_size, len(self.prompts))}/{len(self.prompts)}")
        
        # Cache'e kaydet
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                for item in teacher_outputs:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.logger.info(f"Teacher outputs cache'e kaydedildi: {self.cache_path}")
        
        return teacher_outputs
    
    def __len__(self) -> int:
        return len(self.teacher_outputs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tek bir distillation örneği döndürür"""
        item = self.teacher_outputs[idx]
        
        prompt = item["prompt"]
        teacher_response = item["teacher_response"]
        
        # Student için input (prompt)
        student_input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # Teacher target (prompt + response)
        full_text = prompt + " " + teacher_response
        teacher_target_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        
        # Truncation ve padding
        max_len = self.config.max_seq_length
        
        if len(student_input_ids) > max_len:
            student_input_ids = student_input_ids[:max_len]
        if len(teacher_target_ids) > max_len:
            teacher_target_ids = teacher_target_ids[:max_len]
        
        # Padding
        student_pad_len = max_len - len(student_input_ids)
        teacher_pad_len = max_len - len(teacher_target_ids)
        
        if student_pad_len > 0:
            student_input_ids.extend([self.tokenizer.pad_token_id] * student_pad_len)
        if teacher_pad_len > 0:
            teacher_target_ids.extend([self.tokenizer.pad_token_id] * teacher_pad_len)
        
        # Attention masks
        student_attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 
                                 for token_id in student_input_ids]
        teacher_attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 
                                 for token_id in teacher_target_ids]
        
        return {
            "student_input_ids": torch.tensor(student_input_ids, dtype=torch.long),
            "student_attention_mask": torch.tensor(student_attention_mask, dtype=torch.long),
            "teacher_target_ids": torch.tensor(teacher_target_ids, dtype=torch.long),
            "teacher_attention_mask": torch.tensor(teacher_attention_mask, dtype=torch.long),
            "prompt": prompt,
            "teacher_response": teacher_response
        }


class GeminiDistillationTrainer(BaseTrainer):
    """Gemini-based Knowledge Distillation Trainer"""
    
    def __init__(
        self,
        model: M3TMBaseModel,
        config: DistillationConfig,
        gemini_client: GeminiClient,
        tokenizer,
        save_dir: Optional[str] = None
    ):
        # BaseTrainer için config oluştur
        from .base_trainer import BaseTrainingConfig
        base_config = BaseTrainingConfig(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            max_seq_length=config.max_seq_length,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            gradient_clip=config.gradient_clip,
            device='auto',
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True
        )
        
        super().__init__(model, base_config, save_dir)
        
        self.distillation_config = config
        self.gemini_client = gemini_client
        self.tokenizer = tokenizer
        
        self.logger.info("GeminiDistillationTrainer oluşturuldu")
    
    def create_dataset(
        self,
        data_path: str,
        mode: str = "train"
    ) -> GeminiDistillationDataset:
        """Distillation dataset oluşturur"""
        # Prompts'u yükle
        prompts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'prompt' in data:
                    prompts.append(data['prompt'])
                elif 'instruction' in data:
                    prompts.append(data['instruction'])
                else:
                    # İlk string field'ı kullan
                    for key, value in data.items():
                        if isinstance(value, str):
                            prompts.append(value)
                            break
        
        # Cache path
        cache_path = f"cache/gemini_teacher_outputs_{mode}.jsonl"
        
        return GeminiDistillationDataset(
            prompts=prompts,
            gemini_client=self.gemini_client,
            tokenizer=self.tokenizer,
            config=self.distillation_config,
            cache_path=cache_path
        )
    
    def train(
        self,
        train_dataset: GeminiDistillationDataset,
        val_dataset: Optional[GeminiDistillationDataset] = None
    ) -> Dict[str, Any]:
        """Knowledge distillation eğitimini çalıştırır"""
        self.logger.info("Gemini knowledge distillation eğitimi başlıyor...")
        
        # DataLoader oluştur
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Multiprocessing sorununu önlemek için 0
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0  # Multiprocessing sorununu önlemek için 0
            )
        
        # Optimizer ve scheduler
        optimizer = self.create_optimizer()
        total_steps = len(train_loader) * self.config.epochs
        scheduler = self.create_scheduler(optimizer, total_steps)
        
        # Eğitim döngüsü
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # Training
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler)
            
            # Validation
            val_metrics = None
            if val_loader:
                val_metrics = self._validate_epoch(val_loader)
                
                val_loss = val_metrics.get('val_loss', float('inf'))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(self.save_dir / "best_model.pt")
            
            # Regular checkpoint
            self.save_model(self.save_dir / f"checkpoint_epoch_{epoch}.pt")
            
            self.current_epoch = epoch
        
        self.logger.info("Knowledge distillation eğitimi tamamlandı!")
        
        return {
            "final_train_metrics": train_metrics,
            "final_val_metrics": val_metrics,
            "best_val_loss": best_val_loss
        }
    
    def _train_epoch(self, train_loader, optimizer, scheduler) -> Dict[str, float]:
        """Tek epoch distillation eğitimi"""
        self.model.train()
        
        total_loss = 0.0
        total_distill_loss = 0.0
        total_student_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Batch'i device'a taşı
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Student forward pass
            student_text_input = {
                'input_ids': batch['student_input_ids'],
                'attention_mask': batch['student_attention_mask']
            }
            
            student_outputs = self.model(text_input=student_text_input)
            student_logits = student_outputs.get('logits')
            
            if student_logits is None:
                raise ValueError("Model logits döndürmüyor!")
            
            # Teacher targets (hard targets)
            teacher_targets = batch['teacher_target_ids']
            
            # Student loss (hard targets)
            student_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                teacher_targets.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            
            # Distillation loss (soft targets)
            # Teacher'dan soft targets alamadığımız için sadece hard targets kullanıyoruz
            distillation_loss = torch.tensor(0.0, device=self.device)
            
            # Combined loss
            total_batch_loss = (
                self.distillation_config.alpha * student_loss +
                (1 - self.distillation_config.alpha) * distillation_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_batch_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            optimizer.step()
            scheduler.step()
            
            # Metrics
            total_loss += total_batch_loss.item()
            total_student_loss += student_loss.item()
            total_distill_loss += distillation_loss.item()
            
            # Logging
            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                avg_student_loss = total_student_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                
                self.log_training_progress(
                    step + 1,
                    len(train_loader),
                    avg_loss,
                    lr,
                    {
                        "student_loss": avg_student_loss,
                        "distill_loss": total_distill_loss / (step + 1)
                    }
                )
        
        return {
            "avg_loss": total_loss / len(train_loader),
            "avg_student_loss": total_student_loss / len(train_loader),
            "avg_distill_loss": total_distill_loss / len(train_loader)
        }
    
    def _validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validation epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_student_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Student forward pass
                student_text_input = {
                    'input_ids': batch['student_input_ids'],
                    'attention_mask': batch['student_attention_mask']
                }
                
                student_outputs = self.model(text_input=student_text_input)
                student_logits = student_outputs.get('logits')
                
                # Student loss
                teacher_targets = batch['teacher_target_ids']
                student_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    teacher_targets.view(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
                
                total_loss += student_loss.item()
                total_student_loss += student_loss.item()
        
        metrics = {
            "val_loss": total_loss / len(val_loader),
            "val_student_loss": total_student_loss / len(val_loader)
        }
        
        self.logger.info(f"Validation - Loss: {metrics['val_loss']:.4f}")
        
        return metrics
