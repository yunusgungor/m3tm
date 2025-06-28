"""
Final checkpoint fix - Force PyTorch model saving
"""

import os
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Force SFTTrainer to save model properly"""
    
    try:
        from trl import SFTTrainer
        
        original_save_model = SFTTrainer.save_model
        
        def enhanced_save_model(self, output_dir=None, _internal_call=False):
            output_dir = output_dir or self.args.output_dir
            logger.info(f"🔧 Enhanced SFTTrainer saving to {output_dir}")
            
            # Call original save
            result = original_save_model(self, output_dir, _internal_call)
            
            # Force additional save with our method
            if hasattr(self, 'model') and self.model is not None:
                try:
                    self.model.save_pretrained(output_dir, safe_serialization=False)
                    logger.info(f"✅ Additional model save completed to {output_dir}")
                except Exception as e:
                    logger.warning(f"Additional save failed: {e}")
                    
            return result
        
        SFTTrainer.save_model = enhanced_save_model
        logger.info("✅ Enhanced SFTTrainer saving applied")
        
    except ImportError:
        logger.warning("SFTTrainer not available for patching")


def apply_all_checkpoint_fixes():
    """Apply all checkpoint fixes"""
    logger.info("🔧 Applying comprehensive checkpoint fixes...")
    
    force_pytorch_model_save()
    force_trainer_save()
    
    # Set environment variables
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['HF_SAVE_SERIALIZATION'] = 'pytorch'
    
    logger.info("✅ All checkpoint fixes applied")


if __name__ == "__main__":
    apply_all_checkpoint_fixes()
    print("✅ Checkpoint fixes applied successfully")
