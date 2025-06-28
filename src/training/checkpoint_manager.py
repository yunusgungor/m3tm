"""
Checkpoint management and recovery system for production training.
"""

import os
import torch
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import pickle

logger = logging.getLogger(__name__)


class ProductionCheckpointManager:
    """Enhanced checkpoint manager with reliable model saving."""
    
    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, model, tokenizer, trainer_state, step: int, **kwargs):
        """Save complete checkpoint with model weights."""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model weights (the missing pytorch_model.bin)
            logger.info(f"Saving model weights to {checkpoint_dir}")
            model.save_pretrained(checkpoint_dir, safe_serialization=False)
            
            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Save trainer state
            trainer_state_path = checkpoint_dir / "trainer_state.json"
            with open(trainer_state_path, 'w') as f:
                json.dump(trainer_state, f, indent=2)
            
            # Save additional metadata
            metadata = {
                'step': step,
                'timestamp': str(torch.cuda.Event().record() if torch.cuda.is_available() else 'cpu'),
                'model_size_mb': self._get_model_size(model),
                **kwargs
            }
            
            metadata_path = checkpoint_dir / "checkpoint_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Verify checkpoint integrity
            if self._verify_checkpoint(checkpoint_dir):
                logger.info(f"✅ Checkpoint {step} saved successfully")
                self._cleanup_old_checkpoints()
                return True
            else:
                logger.error(f"❌ Checkpoint {step} verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint {step}: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str, model_class=None):
        """Load checkpoint with verification."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not self._verify_checkpoint(checkpoint_dir):
            raise ValueError(f"Invalid or corrupted checkpoint: {checkpoint_dir}")
        
        try:
            # Load model
            if model_class:
                model = model_class.from_pretrained(checkpoint_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            
            # Load trainer state
            trainer_state_path = checkpoint_dir / "trainer_state.json"
            trainer_state = None
            if trainer_state_path.exists():
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
            
            logger.info(f"✅ Checkpoint loaded from {checkpoint_dir}")
            return model, tokenizer, trainer_state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_dir}: {e}")
            raise
    
    def _verify_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Verify checkpoint integrity."""
        required_files = [
            "pytorch_model.bin",  # This is the critical missing file
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (checkpoint_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.warning(f"Missing checkpoint files: {missing_files}")
            return False
        
        return True
    
    def _get_model_size(self, model) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoints = list(self.output_dir.glob("checkpoint-*"))
        if len(checkpoints) <= self.save_total_limit:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.save_total_limit]:
            try:
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        
        for checkpoint_dir in self.output_dir.glob("checkpoint-*"):
            try:
                step = int(checkpoint_dir.name.split("-")[1])
                integrity = self._verify_checkpoint(checkpoint_dir)
                
                metadata_path = checkpoint_dir / "checkpoint_metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                checkpoints.append({
                    'step': step,
                    'path': str(checkpoint_dir),
                    'integrity': integrity,
                    'metadata': metadata
                })
            except Exception as e:
                logger.warning(f"Failed to process checkpoint {checkpoint_dir}: {e}")
        
        return sorted(checkpoints, key=lambda x: x['step'])


class EnhancedSFTTrainer:
    """Enhanced SFT Trainer with proper checkpoint management."""
    
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, training_args, **kwargs):
        from trl import SFTTrainer
        
        self.checkpoint_manager = ProductionCheckpointManager(
            training_args.output_dir, 
            training_args.save_total_limit
        )
        
        # Initialize base trainer
        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            **kwargs
        )
        
        # Override save methods
        self._original_save_model = self.trainer.save_model
        self.trainer.save_model = self._enhanced_save_model
        
    def _enhanced_save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Enhanced model saving with proper checkpoint management."""
        output_dir = output_dir or self.trainer.args.output_dir
        
        # Use our checkpoint manager
        step = self.trainer.state.global_step
        success = self.checkpoint_manager.save_checkpoint(
            model=self.trainer.model,
            tokenizer=self.trainer.tokenizer,
            trainer_state=self.trainer.state.to_dict(),
            step=step,
            eval_results=getattr(self.trainer, 'eval_results', {}),
        )
        
        if success:
            logger.info(f"Enhanced checkpoint saved at step {step}")
        else:
            logger.error(f"Failed to save enhanced checkpoint at step {step}")
            # Fallback to original method
            self._original_save_model(output_dir, _internal_call)
    
    def train(self, **kwargs):
        """Run training with enhanced checkpointing."""
        return self.trainer.train(**kwargs)
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save final model."""
        return self._enhanced_save_model(output_dir)
    
    def __getattr__(self, name):
        """Delegate other attributes to the base trainer."""
        return getattr(self.trainer, name)


def fix_checkpoint_system():
    """Fix common checkpoint issues in the training system."""
    
    # Ensure proper model saving configuration
    import transformers
    
    # Monkey patch to ensure pytorch_model.bin is always saved
    original_save_pretrained = transformers.PreTrainedModel.save_pretrained
    
    def enhanced_save_pretrained(self, save_directory, **kwargs):
        # Force safe_serialization to False to create pytorch_model.bin
        kwargs['safe_serialization'] = False
        return original_save_pretrained(self, save_directory, **kwargs)
    
    transformers.PreTrainedModel.save_pretrained = enhanced_save_pretrained
    
    logger.info("✅ Checkpoint system fixed - pytorch_model.bin will be saved")


# Apply fix when module is imported
fix_checkpoint_system()
