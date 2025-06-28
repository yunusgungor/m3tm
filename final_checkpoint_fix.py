"""
Final checkpoint fix - Force PyTorch model saving and attention mask handling
Based on Context7 PyTorch and Transformers best practices
"""

import os
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_attention_mask_tokenizer():
    """Fix attention mask and tokenizer pad token issues - Context7 best practices"""
    
    def patch_tokenizer_init():
        """Patch tokenizer to fix pad_token conflicts using Context7 guidance"""
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizer
            
            # Store original from_pretrained
            if not hasattr(AutoTokenizer, '_original_from_pretrained'):
                AutoTokenizer._original_from_pretrained = AutoTokenizer.from_pretrained
            
            def enhanced_from_pretrained(cls, *args, **kwargs):
                tokenizer = AutoTokenizer._original_from_pretrained(*args, **kwargs)
                
                # Context7 critical fix: Ensure pad_token_id != eos_token_id
                if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
                    logger.info(f"🔧 Fixing pad_token conflict: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
                    
                    # Context7 recommended approach: Use distinct pad token
                    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None and tokenizer.unk_token_id != tokenizer.eos_token_id:
                        tokenizer.pad_token_id = tokenizer.unk_token_id
                        tokenizer.pad_token = tokenizer.unk_token
                        logger.info(f"✅ Set pad_token to unk_token: {tokenizer.pad_token_id}")
                    else:
                        # Add a special pad token as per Context7 examples
                        try:
                            tokenizer.add_special_tokens({'pad_token': '<pad>'})
                            logger.info(f"✅ Added new pad token: {tokenizer.pad_token_id}")
                        except:
                            # Fallback: use a different safe ID
                            tokenizer.pad_token_id = 0 if tokenizer.eos_token_id != 0 else 1
                            tokenizer.pad_token = '<pad>'
                            logger.info(f"✅ Set fallback pad_token_id: {tokenizer.pad_token_id}")
                    
                    # Ensure they're still different after fix
                    if tokenizer.pad_token_id == tokenizer.eos_token_id:
                        tokenizer.pad_token_id = tokenizer.eos_token_id + 1 if tokenizer.eos_token_id < 50000 else tokenizer.eos_token_id - 1
                        logger.info(f"✅ Final fix - pad_token_id: {tokenizer.pad_token_id}")
                
                return tokenizer
            
            AutoTokenizer.from_pretrained = classmethod(enhanced_from_pretrained)
            logger.info("✅ Context7-based tokenizer pad_token fix applied")
            
        except ImportError as e:
            logger.warning(f"Could not patch tokenizer: {e}")
    
    patch_tokenizer_init()


def fix_model_attention_mask():
    """Fix model attention mask handling - Context7 Transformers best practices"""
    
    try:
        from transformers.modeling_utils import PreTrainedModel
        
        # Store original forward method
        if not hasattr(PreTrainedModel, '_original_forward'):
            PreTrainedModel._original_forward = PreTrainedModel.forward
        
        def enhanced_forward(self, *args, **kwargs):
            # Context7 guidance: Always ensure attention_mask is provided when input_ids present
            if 'input_ids' in kwargs and 'attention_mask' not in kwargs:
                input_ids = kwargs['input_ids']
                if hasattr(input_ids, 'shape') and hasattr(input_ids, 'device'):
                    # Create attention mask based on Context7 examples
                    # 1 for real tokens, 0 for padding tokens
                    if hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
                        attention_mask = (input_ids != self.config.pad_token_id).long()
                    else:
                        # Fallback: assume 0 is padding
                        attention_mask = (input_ids != 0).long()
                    
                    kwargs['attention_mask'] = attention_mask
                    logger.debug(f"🔧 Auto-generated attention_mask: shape={attention_mask.shape}, non_zero={attention_mask.sum().item()}")
            
            return PreTrainedModel._original_forward(self, *args, **kwargs)
        
        # Apply the patch
        PreTrainedModel.forward = enhanced_forward
        logger.info("✅ Context7-based model attention_mask fix applied")
        
    except ImportError as e:
        logger.warning(f"Could not patch model forward: {e}")


def force_pytorch_model_save():
    """Force all models to save as pytorch_model.bin"""
    
    # Monkey patch PreTrainedModel to always save pytorch_model.bin
    from transformers import PreTrainedModel
    
    original_save_pretrained = PreTrainedModel.save_pretrained
    
    def enhanced_save_pretrained(self, save_directory, **kwargs):
        # Always force safe_serialization=False to create pytorch_model.bin
        kwargs['safe_serialization'] = False
        logger.info(f"🔧 Forcing pytorch_model.bin save to {save_directory}")
        result = original_save_pretrained(self, save_directory, **kwargs)
        
        # Verify the file was created
        pytorch_model_path = Path(save_directory) / "pytorch_model.bin"
        if pytorch_model_path.exists():
            logger.info(f"✅ pytorch_model.bin successfully saved ({pytorch_model_path.stat().st_size / 1024 / 1024:.2f}MB)")
        else:
            logger.error(f"❌ pytorch_model.bin NOT saved to {save_directory}")
            
        return result
    
    PreTrainedModel.save_pretrained = enhanced_save_pretrained
    logger.info("✅ Enhanced model saving applied - pytorch_model.bin will be forced")


def force_trainer_save():
    """Force SFTTrainer to save model properly - Context7 PyTorch checkpoint best practices"""
    
    try:
        from trl import SFTTrainer
        
        if not hasattr(SFTTrainer, '_original_save_model'):
            SFTTrainer._original_save_model = SFTTrainer.save_model
        
        def enhanced_save_model(self, output_dir=None, _internal_call=False):
            output_dir = output_dir or self.args.output_dir
            logger.info(f"🔧 Enhanced SFTTrainer saving to {output_dir} using Context7 PyTorch practices")
            
            # Call original save first
            result = SFTTrainer._original_save_model(self, output_dir, _internal_call)
            
            # Context7 best practice: Comprehensive checkpoint saving
            if hasattr(self, 'model') and self.model is not None:
                try:
                    # 1. Force pytorch_model.bin save (Context7 guidance)
                    self.model.save_pretrained(output_dir, safe_serialization=False)
                    
                    # 2. Save comprehensive checkpoint following PyTorch examples
                    checkpoint_path = Path(output_dir) / "training_checkpoint.pt"
                    checkpoint_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'epoch': getattr(self.state, 'epoch', 0),
                        'global_step': getattr(self.state, 'global_step', 0),
                        'best_metric': getattr(self.state, 'best_metric', None),
                        'log_history': getattr(self.state, 'log_history', []),
                        'training_args': vars(self.args) if hasattr(self, 'args') else {}
                    }
                    
                    # Add optimizer state (Context7 PyTorch examples)
                    if hasattr(self, 'optimizer') and self.optimizer is not None:
                        checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                        
                    # Add scheduler state if available
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                        checkpoint_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
                        
                    # Save using PyTorch's recommended method
                    torch.save(checkpoint_dict, checkpoint_path)
                    logger.info(f"✅ Context7-style comprehensive checkpoint saved to {checkpoint_path}")
                    
                    # 3. Create a minimal state dict save for loading compatibility
                    state_dict_path = Path(output_dir) / "pytorch_model_state.pt"
                    torch.save(self.model.state_dict(), state_dict_path)
                    
                    # Verify all files exist with sizes
                    files_to_check = [
                        ("pytorch_model.bin", Path(output_dir) / "pytorch_model.bin"),
                        ("training_checkpoint.pt", checkpoint_path),
                        ("pytorch_model_state.pt", state_dict_path)
                    ]
                    
                    for name, path in files_to_check:
                        if path.exists():
                            size_mb = path.stat().st_size / 1024 / 1024
                            logger.info(f"✅ {name} verified: {size_mb:.2f}MB")
                        else:
                            logger.warning(f"❌ {name} missing at {path}")
                        
                except Exception as e:
                    logger.warning(f"Enhanced save failed: {e}")
                    
            return result
        
        SFTTrainer.save_model = enhanced_save_model
        logger.info("✅ Context7-based enhanced SFTTrainer saving applied")
        
    except ImportError:
        logger.warning("SFTTrainer not available for patching")


def patch_sft_trainer_checkpoint_saving():
    """
    Enhanced SFTTrainer patching for Context7-compatible checkpoint saving.
    Ensures pytorch_model.bin is always created during training checkpoints.
    """
    try:
        from trl import SFTTrainer
        import torch
        from pathlib import Path
        import logging
        import os
        
        logger = logging.getLogger(__name__)
        logger.info("🔧 Applying Context7-based SFTTrainer checkpoint patch...")
        
        # Store original methods for fallback
        original_save_model = getattr(SFTTrainer, 'save_model', None)
        
        # CRITICAL: Also patch _save_checkpoint method which is used during training
        original_save_checkpoint = getattr(SFTTrainer, '_save_checkpoint', None)
        if not original_save_checkpoint:
            # Fallback to parent class method
            try:
                from transformers import Trainer
                original_save_checkpoint = getattr(Trainer, '_save_checkpoint', None)
            except:
                original_save_checkpoint = None
        
        def enhanced_save_checkpoint(self, checkpoint_folder, trial=None):
            """
            Enhanced _save_checkpoint method with Context7 PyTorch checkpoint best practices.
            This is called during training when save_steps is reached.
            """
            # Call original method first to get the basic checkpoint structure
            result = None
            if original_save_checkpoint:
                try:
                    result = original_save_checkpoint(self, checkpoint_folder, trial)
                except Exception as e:
                    logger.warning(f"Original _save_checkpoint failed: {e}")
                
            try:
                if hasattr(self, 'model') and self.model is not None:
                    logger.info(f"🚀 Enhanced _save_checkpoint to {checkpoint_folder}")
                    
                    # Ensure checkpoint folder exists
                    os.makedirs(checkpoint_folder, exist_ok=True)
                    
                    # 1. CRITICAL: Force pytorch_model.bin creation during training checkpoints
                    pytorch_model_path = Path(checkpoint_folder) / "pytorch_model.bin"
                    torch.save(self.model.state_dict(), pytorch_model_path)
                    
                    # Verify the file was created and has reasonable size
                    if pytorch_model_path.exists():
                        size_mb = pytorch_model_path.stat().st_size / 1024 / 1024
                        logger.info(f"✅ Training checkpoint pytorch_model.bin: {size_mb:.2f}MB")
                    else:
                        logger.error(f"❌ Failed to create pytorch_model.bin in {checkpoint_folder}")
                    
                    # 2. Force model.save_pretrained to ensure all standard files exist
                    try:
                        self.model.save_pretrained(checkpoint_folder, safe_serialization=False)
                        logger.info(f"✅ Model save_pretrained completed for {checkpoint_folder}")
                    except Exception as e:
                        logger.warning(f"Model save_pretrained failed: {e}")
                    
                    # 3. Save additional checkpoint metadata for training resume
                    checkpoint_metadata = {
                        'model_state_dict': self.model.state_dict(),
                        'global_step': getattr(self.state, 'global_step', 0),
                        'epoch': getattr(self.state, 'epoch', 0),
                        'training_args': vars(self.args) if hasattr(self, 'args') else {}
                    }
                    
                    # Add optimizer state if available (important for resume)
                    if hasattr(self, 'optimizer') and self.optimizer is not None:
                        try:
                            checkpoint_metadata['optimizer_state_dict'] = self.optimizer.state_dict()
                        except:
                            pass
                        
                    # Add scheduler state if available
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                        try:
                            checkpoint_metadata['scheduler_state_dict'] = self.lr_scheduler.state_dict()
                        except:
                            pass
                        
                    # Save comprehensive training checkpoint
                    training_checkpoint_path = Path(checkpoint_folder) / "training_checkpoint.pt"
                    torch.save(checkpoint_metadata, training_checkpoint_path)
                    logger.info(f"✅ Training metadata checkpoint saved")
                    
                    # 4. Create tokenizer files if missing
                    try:
                        if hasattr(self, 'processing_class') and self.processing_class is not None:
                            self.processing_class.save_pretrained(checkpoint_folder)
                            logger.info(f"✅ Tokenizer saved to checkpoint")
                        elif hasattr(self, 'tokenizer') and self.tokenizer is not None:
                            self.tokenizer.save_pretrained(checkpoint_folder)
                            logger.info(f"✅ Tokenizer saved to checkpoint")
                    except Exception as e:
                        logger.warning(f"Tokenizer save failed: {e}")
                        
                except Exception as e:
                    logger.warning(f"Enhanced _save_checkpoint failed: {e}")
                    
            return result
        
        # Apply the critical patch
        if original_save_checkpoint:
            SFTTrainer._save_checkpoint = enhanced_save_checkpoint
            logger.info("✅ Context7-based enhanced SFTTrainer._save_checkpoint applied")
        else:
            logger.warning("⚠️ Could not find _save_checkpoint method to patch")
            
        logger.info("✅ Context7-based enhanced SFTTrainer checkpoint saving applied")
        
    except ImportError:
        logger.warning("TRL not available, skipping SFTTrainer checkpoint patch")
    except Exception as e:
        logger.error(f"Failed to apply SFTTrainer checkpoint patch: {e}")


def apply_all_checkpoint_fixes():
    """Apply all checkpoint and attention mask fixes"""
    logger.info("🔧 Applying comprehensive checkpoint and attention mask fixes...")
    
    # Apply attention mask fixes first
    fix_attention_mask_tokenizer()
    fix_model_attention_mask()
    
    # Apply checkpoint fixes
    force_pytorch_model_save()
    force_trainer_save()
    
    # CRITICAL: Apply SFTTrainer checkpoint saving patch
    patch_sft_trainer_checkpoint_saving()
    
    # Set environment variables
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['HF_SAVE_SERIALIZATION'] = 'pytorch'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
    
    logger.info("✅ All fixes applied: attention_mask + checkpoints + SFTTrainer patches")


if __name__ == "__main__":
    apply_all_checkpoint_fixes()
    print("✅ Checkpoint and attention mask fixes applied successfully")
