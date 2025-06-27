"""
Group Relative Policy Optimization (GRPO) Trainer for M³TM

Bu modül, M³TM modeli için GRPO implementasyonu sağlar.
GRPO, preference learning ve reinforcement learning kullanarak
modeli insan tercihlerine göre optimize eder.
"""

import os
import logging
import json
import math
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
class GRPOConfig:
    """GRPO eğitim konfigürasyonu"""
    # Temel eğitim parametreleri
    learning_rate: float = 1e-5
    batch_size: int = 2
    epochs: int = 1
    max_seq_length: int = 512
    
    # GRPO özel parametreleri
    beta: float = 0.1  # KL penalty coefficient
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO clip range
    
    # Group relative optimization
    group_size: int = 4  # Comparison group size
    relative_threshold: float = 0.1  # Relative preference threshold
    
    # Value function
    use_value_function: bool = True
    value_loss_coef: float = 0.5
    
    # Regularization
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # Model architecture
    hidden_size: int = 512  # Model hidden size

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500


class RewardModel(nn.Module):
    """Reward model for preference learning"""
    
    def __init__(self, base_model: M3TMBaseModel, config: GRPOConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Reward head
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else 512
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Value head (for advantage estimation)
        if config.use_value_function:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            )
    
    def forward(
        self,
        text_input: Dict[str, torch.Tensor],
        return_values: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Base model features
        outputs = self.base_model(text_input=text_input, return_dict=True)
        features = outputs.get('fused_features')
        
        if features is None:
            # Fallback to text features
            features = outputs.get('text_features')
        
        if features is None:
            raise ValueError("Model features bulunamadı!")
        
        # Pool features (take mean over sequence)
        if features.dim() == 3:  # [batch, seq, hidden]
            features = features.mean(dim=1)
        
        # Compute rewards
        rewards = self.reward_head(features).squeeze(-1)
        
        result = {'rewards': rewards}
        
        # Compute values if requested
        if return_values and self.config.use_value_function:
            values = self.value_head(features).squeeze(-1)
            result['values'] = values
        
        return result


class GRPODataset(Dataset):
    """GRPO için preference dataset"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        config: GRPOConfig,
        mode: str = "train"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"GRPODataset oluşturuldu: {len(data)} örnek, mode: {mode}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Tek bir preference örneği döndürür"""
        item = self.data[idx]
        
        # Prompt ve responses
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        # SimpleTokenizer ile tokenize
        def tokenize_text(text):
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)

            # Truncation
            if len(token_ids) > self.config.max_seq_length:
                token_ids = token_ids[:self.config.max_seq_length]

            # Padding
            pad_length = self.config.max_seq_length - len(token_ids)
            if pad_length > 0:
                token_ids.extend([self.tokenizer.pad_token_id] * pad_length)

            # Attention mask oluştur
            attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in token_ids]

            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
            }
        
        # Prompt + chosen response
        chosen_text = prompt + " " + chosen
        chosen_tokens = tokenize_text(chosen_text)
        
        # Prompt + rejected response
        rejected_text = prompt + " " + rejected
        rejected_tokens = tokenize_text(rejected_text)
        
        return {
            "chosen": chosen_tokens,
            "rejected": rejected_tokens,
            "prompt": prompt,
            "chosen_text": chosen_text,
            "rejected_text": rejected_text
        }


class GRPOTrainer(BaseTrainer):
    """Group Relative Policy Optimization Trainer"""
    
    def __init__(
        self,
        model: M3TMBaseModel,
        config: GRPOConfig,
        tokenizer,
        save_dir: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir) if save_dir else Path("./grpo_checkpoints")
        
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reward model oluştur
        self.reward_model = RewardModel(model, config)
        self.reward_model.to(self.device)
        
        # Reference model (frozen copy for KL penalty)
        self.ref_model = RewardModel(model, config)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Checkpoint dizinini oluştur
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"GRPOTrainer oluşturuldu - Device: {self.device}")
    
    def create_dataset(
        self,
        data_path: str,
        mode: str = "train"
    ) -> GRPODataset:
        """GRPO dataset oluşturur"""
        # JSON Lines formatında preference data yükle
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        return GRPODataset(data, self.tokenizer, self.config, mode)
    
    def train(
        self,
        train_dataset: GRPODataset,
        val_dataset: Optional[GRPODataset] = None
    ) -> Dict[str, Any]:
        """GRPO eğitimini çalıştırır"""
        self.logger.info("GRPO eğitimi başlıyor...")
        
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
            self.reward_model.parameters(),
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
        best_val_reward = float('-inf')
        global_step = 0
        
        for epoch in range(self.config.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # Training
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler, global_step)
            
            # Validation
            val_metrics = None
            if val_loader:
                val_metrics = self._validate_epoch(val_loader)
                
                # Best model kaydet
                val_reward = val_metrics.get('avg_reward', float('-inf'))
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
            
            # Regular checkpoint kaydet
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            global_step += len(train_loader)
        
        self.logger.info("GRPO eğitimi tamamlandı!")
        
        return {
            "final_train_metrics": train_metrics,
            "final_val_metrics": val_metrics,
            "best_val_reward": best_val_reward,
            "total_steps": global_step
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        global_step: int
    ) -> Dict[str, float]:
        """Tek epoch eğitim"""
        self.reward_model.train()
        
        total_loss = 0.0
        total_reward_loss = 0.0
        total_value_loss = 0.0
        total_kl_penalty = 0.0
        
        for step, batch in enumerate(train_loader):
            # Batch'i device'a taşı
            chosen = {k: v.to(self.device) for k, v in batch['chosen'].items()}
            rejected = {k: v.to(self.device) for k, v in batch['rejected'].items()}
            
            # Forward pass - chosen responses
            chosen_outputs = self.reward_model(chosen, return_values=True)
            chosen_rewards = chosen_outputs['rewards']
            
            # Forward pass - rejected responses
            rejected_outputs = self.reward_model(rejected, return_values=True)
            rejected_rewards = rejected_outputs['rewards']
            
            # Reference model rewards (for KL penalty)
            with torch.no_grad():
                ref_chosen_outputs = self.ref_model(chosen, return_values=False)
                ref_rejected_outputs = self.ref_model(rejected, return_values=False)
                ref_chosen_rewards = ref_chosen_outputs['rewards']
                ref_rejected_rewards = ref_rejected_outputs['rewards']
            
            # Preference loss (Bradley-Terry model)
            reward_diff = chosen_rewards - rejected_rewards
            preference_loss = -F.logsigmoid(reward_diff).mean()
            
            # KL penalty
            kl_chosen = F.kl_div(
                F.log_softmax(chosen_rewards, dim=-1),
                F.softmax(ref_chosen_rewards, dim=-1),
                reduction='batchmean'
            )
            kl_rejected = F.kl_div(
                F.log_softmax(rejected_rewards, dim=-1),
                F.softmax(ref_rejected_rewards, dim=-1),
                reduction='batchmean'
            )
            kl_penalty = (kl_chosen + kl_rejected) / 2
            
            # Value loss (if using value function)
            value_loss = 0.0
            if self.config.use_value_function:
                chosen_values = chosen_outputs.get('values')
                rejected_values = rejected_outputs.get('values')
                
                if chosen_values is not None and rejected_values is not None:
                    # Simple value loss - values should predict rewards
                    value_loss = (
                        F.mse_loss(chosen_values, chosen_rewards.detach()) +
                        F.mse_loss(rejected_values, rejected_rewards.detach())
                    ) / 2
            
            # Total loss
            loss = (
                preference_loss +
                self.config.beta * kl_penalty +
                self.config.value_loss_coef * value_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.reward_model.parameters(),
                    self.config.gradient_clip
                )
            
            optimizer.step()
            scheduler.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_reward_loss += preference_loss.item()
            total_value_loss += value_loss if isinstance(value_loss, float) else value_loss.item()
            total_kl_penalty += kl_penalty.item()
            
            # Logging
            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                avg_reward_loss = total_reward_loss / (step + 1)
                avg_value_loss = total_value_loss / (step + 1)
                avg_kl = total_kl_penalty / (step + 1)
                lr = scheduler.get_last_lr()[0]
                
                self.logger.info(
                    f"Step {global_step + step + 1}: "
                    f"Loss = {loss.item():.4f}, "
                    f"Reward Loss = {preference_loss.item():.4f}, "
                    f"Value Loss = {avg_value_loss:.4f}, "
                    f"KL = {kl_penalty.item():.4f}, "
                    f"LR = {lr:.2e}"
                )
        
        return {
            "avg_loss": total_loss / len(train_loader),
            "avg_reward_loss": total_reward_loss / len(train_loader),
            "avg_value_loss": total_value_loss / len(train_loader),
            "avg_kl_penalty": total_kl_penalty / len(train_loader)
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation epoch"""
        self.reward_model.eval()
        
        total_reward_diff = 0.0
        total_accuracy = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                chosen = {k: v.to(self.device) for k, v in batch['chosen'].items()}
                rejected = {k: v.to(self.device) for k, v in batch['rejected'].items()}
                
                # Forward pass
                chosen_outputs = self.reward_model(chosen, return_values=False)
                rejected_outputs = self.reward_model(rejected, return_values=False)
                
                chosen_rewards = chosen_outputs['rewards']
                rejected_rewards = rejected_outputs['rewards']
                
                # Metrics
                reward_diff = chosen_rewards - rejected_rewards
                accuracy = (reward_diff > 0).float().mean()
                
                total_reward_diff += reward_diff.mean().item()
                total_accuracy += accuracy.item()
                num_samples += 1
        
        metrics = {
            "avg_reward_diff": total_reward_diff / num_samples,
            "accuracy": total_accuracy / num_samples,
            "avg_reward": total_reward_diff / num_samples  # For best model selection
        }
        
        self.logger.info(
            f"Validation - Reward Diff: {metrics['avg_reward_diff']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """Checkpoint kaydet"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.reward_model.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
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
