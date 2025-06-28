#!/usr/bin/env python3
"""
Hızlı checkpoint testi - Minimum veri ile checkpoint fix'ini test et
"""

import os
import sys
import time
import json
import torch
import logging
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_checkpoint_test():
    """Ultra hızlı checkpoint testi - 5 örnek, 3 step"""
    
    logger.info("🚀 Hızlı checkpoint testi başlatılıyor...")
    
    try:
        # Context7 fix'lerini uygula
        logger.info("🔧 Checkpoint fix'leri uygulanıyor...")
        try:
            from final_checkpoint_fix import apply_all_checkpoint_fixes
            apply_all_checkpoint_fixes()
            logger.info("✅ Fix'ler uygulandı")
        except Exception as fix_error:
            logger.warning(f"Fix uygulanamadı: {fix_error}")
        
        # Gerekli importlar
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        import json
        
        # Model yükle
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Model yükleniyor: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Ultra minimal veri - sadece 5 örnek
        minimal_data = [
            {"text": "### İnsan: Python'da nasıl merhaba yazdırırım?\n### Asistan: print('Merhaba Dünya!')"},
            {"text": "### İnsan: Liste nasıl oluştururum?\n### Asistan: my_list = []"},
            {"text": "### İnsan: Döngü nasıl yazarım?\n### Asistan: for i in range(10): print(i)"},
            {"text": "### İnsan: Fonksiyon nasıl tanımlarım?\n### Asistan: def my_function(): pass"},
            {"text": "### İnsan: Değişken nasıl atarım?\n### Asistan: x = 5"}
        ]
        
        dataset = Dataset.from_list(minimal_data)
        logger.info(f"Minimal dataset hazırlandı: {len(minimal_data)} örnek")
        
        # Ultra hızlı training config
        output_dir = "./temp_quick_test"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,  # En küçük batch
            learning_rate=1e-4,
            max_length=128,  # Kısa sequence
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            logging_steps=1,
            save_strategy="steps",
            save_steps=2,  # Her 2 step'te checkpoint
            eval_strategy="no",
            report_to=[],
            max_steps=3,  # Sadece 3 step
            dataset_text_field="text",
            packing=False,
            save_safetensors=False,
            save_total_limit=None,
            load_best_model_at_end=False,
            dataloader_num_workers=0,
            gradient_checkpointing=False
        )
        
        # Trainer oluştur
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer
        )
        
        logger.info("⚡ Ultra hızlı training başlatılıyor (3 step)...")
        start_time = time.time()
        
        # Train
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training tamamlandı: {training_time:.2f}s")
        
        # Checkpoint kontrolü
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
        logger.info(f"Oluşturulan checkpoint sayısı: {len(checkpoints)}")
        
        if len(checkpoints) == 0:
            logger.error("❌ HATA: Hiç checkpoint oluşturulmadı!")
            return False
        
        # En son checkpoint'i kontrol et
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
        checkpoint_path = os.path.join(output_dir, latest_checkpoint)
        
        logger.info(f"Son checkpoint kontrol ediliyor: {latest_checkpoint}")
        
        # Dosya kontrolü
        checkpoint_files = os.listdir(checkpoint_path)
        required_files = ['pytorch_model.bin', 'config.json', 'training_args.bin']
        
        missing_files = [f for f in required_files if f not in checkpoint_files]
        
        if missing_files:
            logger.error(f"❌ EKSIK DOSYALAR: {missing_files}")
            logger.info(f"Mevcut dosyalar: {checkpoint_files}")
            
            # pytorch_model.bin özel kontrolü
            pytorch_model_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
            if os.path.exists(pytorch_model_path):
                size_mb = os.path.getsize(pytorch_model_path) / 1024 / 1024
                logger.info(f"pytorch_model.bin bulundu: {size_mb:.2f}MB")
            else:
                logger.error("❌ pytorch_model.bin bulunamadı!")
            
            return False
        
        # Dosya boyutları
        for file in required_files:
            file_path = os.path.join(checkpoint_path, file)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / 1024 / 1024
                logger.info(f"✅ {file}: {size_mb:.2f}MB")
        
        # Hızlı inference testi
        logger.info("🔄 Hızlı inference testi...")
        test_input = "Python'da liste nasıl oluştururum?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Inference testi: {response[:50]}...")
        
        logger.info("✅ BAŞARILI - Hızlı checkpoint testi geçti!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test başarısız: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = quick_checkpoint_test()
    if success:
        print("✅ Hızlı test BAŞARILI - Checkpoint fix çalışıyor!")
    else:
        print("❌ Hızlı test BAŞARISIZ - Fix'ler düzeltilmeli!")
    
    sys.exit(0 if success else 1)
