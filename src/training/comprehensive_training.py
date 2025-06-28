"""
Comprehensive Training Configuration and Orchestrator for Production.
"""

import os
import yaml
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
import datasets
import json
from .checkpoint_manager import EnhancedSFTTrainer, ProductionCheckpointManager

logger = logging.getLogger(__name__)

@dataclass
class ComprehensiveTrainingConfig:
    """Production-ready training configuration."""
    
    # Model Configuration
    model_name: str = "microsoft/DialoGPT-small"
    tokenizer_name: Optional[str] = None
    max_length: int = 512
    
    # Training Configuration
    output_dir: str = "./training_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Data Configuration
    train_data_path: str = "./data/expanded/sft_train_expanded.jsonl"
    eval_data_path: str = "./data/expanded/sft_val_expanded.jsonl"
    
    # Production Configuration
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Performance Configuration
    gradient_accumulation_steps: int = 4
    fp16: bool = False  # Disabled due to Half precision compatibility issues
    dataloader_num_workers: int = 2
    
    # Additional configurations
    remove_unused_columns: bool = False
    report_to: List[str] = field(default_factory=lambda: ["none"])
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "ComprehensiveTrainingConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('training', {}))
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            logging_steps=self.logging_steps,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=self.remove_unused_columns,
            report_to=self.report_to,
        )


class TrainingOrchestrator:
    """Production-ready training orchestrator."""
    
    def __init__(self, config: ComprehensiveTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup(self):
        """Initialize tokenizer, model, and trainer."""
        logger.info(f"Setting up training with model: {self.config.model_name}")
        
        # Setup tokenizer with Context7 best practices for attention mask
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Context7 critical fix: Ensure pad_token_id != eos_token_id
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            logger.info(f"🔧 Fixing pad_token conflict: pad_token_id={self.tokenizer.pad_token_id}, eos_token_id={self.tokenizer.eos_token_id}")
            
            # Context7 recommended approach: Use distinct pad token
            if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None and self.tokenizer.unk_token_id != self.tokenizer.eos_token_id:
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
                self.tokenizer.pad_token = self.tokenizer.unk_token
                logger.info(f"✅ Set pad_token to unk_token: {self.tokenizer.pad_token_id}")
            else:
                # Add a special pad token as per Context7 examples
                try:
                    self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
                    logger.info(f"✅ Added new pad token: {self.tokenizer.pad_token_id}")
                except:
                    # Fallback: use a different safe ID
                    self.tokenizer.pad_token_id = 0 if self.tokenizer.eos_token_id != 0 else 1
                    self.tokenizer.pad_token = '<pad>'
                    logger.info(f"✅ Set fallback pad_token_id: {self.tokenizer.pad_token_id}")
            
            # Final verification they're different
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id + 1 if self.tokenizer.eos_token_id < 50000 else self.tokenizer.eos_token_id - 1
                logger.info(f"✅ Final verification - pad_token_id: {self.tokenizer.pad_token_id}")
            
        # Setup model with explicit float32 (avoid Half precision issues)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Force float32 to avoid Half precision errors
        )
        
        # Resize token embeddings if needed (Context7 best practice)
        original_vocab_size = self.model.config.vocab_size
        self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"📝 Resized embeddings: {original_vocab_size} -> {len(self.tokenizer)}")
        
        logger.info("✅ Setup completed successfully with Context7 fixes")
        
    def load_dataset(self, data_path: str):
        """Load and prepare dataset."""
        logger.info(f"Loading dataset from: {data_path}")
        
        # Load JSONL data
        data_lines = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data_lines.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        # Convert to HuggingFace dataset
        dataset = datasets.Dataset.from_list(data_lines)
        
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset
        
    def prepare_trainer(self):
        """Prepare the SFT trainer."""
        # Load datasets
        train_dataset = self.load_dataset(self.config.train_data_path)
        eval_dataset = self.load_dataset(self.config.eval_data_path)
        
        # Setup training arguments
        training_args = self.config.to_training_arguments()
        
        # Create enhanced trainer with checkpoint management
        self.trainer = EnhancedSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.max_length,
        )
        
        logger.info("Trainer prepared successfully")
        
    def train(self):
        """Run the training process."""
        if self.trainer is None:
            raise RuntimeError("Trainer not prepared. Call prepare_trainer() first.")
            
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Training completed successfully")
        
    def run_complete_training(self):
        """Run the complete training pipeline."""
        try:
            self.setup()
            self.prepare_trainer()
            self.train()
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False


def create_production_training_config(
    model_name: str = "microsoft/DialoGPT-small",
    output_dir: str = "./production_training_output",
    **kwargs
) -> ComprehensiveTrainingConfig:
    """Create a production-ready training configuration."""
    
    return ComprehensiveTrainingConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        warmup_steps=50,
        save_steps=250,
        eval_steps=250,
        logging_steps=50,
        gradient_accumulation_steps=8,
        fp16=False,  # Disabled due to Half precision compatibility issues
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=["none"],
        **kwargs
    )


def run_production_training(config_path: Optional[str] = None, **config_kwargs):
    """Run production training with given configuration."""
    
    if config_path and os.path.exists(config_path):
        config = ComprehensiveTrainingConfig.from_yaml(config_path)
    else:
        config = create_production_training_config(**config_kwargs)
    
    orchestrator = TrainingOrchestrator(config)
    return orchestrator.run_complete_training()


if __name__ == "__main__":
    # Example usage
    success = run_production_training(
        model_name="microsoft/DialoGPT-small",
        output_dir="./test_training_output",
        num_train_epochs=1
    )
    print(f"Training {'succeeded' if success else 'failed'}")
