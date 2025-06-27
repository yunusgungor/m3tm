"""
SFT ve GRPO Eğitim Örneği

Bu script, M³TM modelini SFT ve GRPO ile eğitmek için örnek kod sağlar.
"""

import os
import sys
import json
import logging
from pathlib import Path

# M³TM modüllerini import et
sys.path.append('src')

import torch
from m3tm.config.model_config import M3TMConfig
from m3tm.core.base_model import M3TMBaseModel
from m3tm.embedding.tokenizer import SimpleTokenizer
from m3tm.training.sft_trainer import SFTTrainer, SFTConfig
from m3tm.training.grpo_trainer import GRPOTrainer, GRPOConfig


def setup_logging():
    """Logging setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def create_sample_sft_data():
    """SFT için örnek veri oluştur"""
    sft_data = [
        {
            "instruction": "Aşağıdaki metni özetle:",
            "response": "Bu metin, yapay zeka teknolojilerinin gelişimi hakkında bilgi veriyor."
        },
        {
            "instruction": "Bu görüntüde ne görüyorsun?",
            "response": "Görüntüde bir kedi ve köpek birlikte oynuyor."
        },
        {
            "instruction": "Python'da liste nasıl oluşturulur?",
            "response": "Python'da liste oluşturmak için köşeli parantez kullanılır: my_list = [1, 2, 3]"
        },
        {
            "instruction": "Türkiye'nin başkenti nedir?",
            "response": "Türkiye'nin başkenti Ankara'dır."
        },
        {
            "instruction": "Makine öğrenmesi nedir?",
            "response": "Makine öğrenmesi, bilgisayarların verilerden öğrenerek tahminler yapmasını sağlayan bir yapay zeka dalıdır."
        }
    ]
    
    # Train/val split
    train_data = sft_data[:4]
    val_data = sft_data[4:]
    
    # JSON Lines formatında kaydet
    os.makedirs('data', exist_ok=True)
    
    with open('data/sft_train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open('data/sft_val.jsonl', 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("SFT veri dosyaları oluşturuldu: data/sft_train.jsonl, data/sft_val.jsonl")


def create_sample_grpo_data():
    """GRPO için örnek preference veri oluştur"""
    grpo_data = [
        {
            "prompt": "Python'da döngü nasıl yazılır?",
            "chosen": "Python'da for döngüsü şu şekilde yazılır:\n\nfor i in range(10):\n    print(i)\n\nBu kod 0'dan 9'a kadar sayıları yazdırır.",
            "rejected": "Python'da döngü var ama nasıl yazıldığını bilmiyorum."
        },
        {
            "prompt": "Yapay zeka nedir?",
            "chosen": "Yapay zeka (AI), makinelerin insan benzeri düşünme ve öğrenme yeteneklerini simüle etmesini sağlayan teknoloji dalıdır. Makine öğrenmesi, derin öğrenme ve doğal dil işleme gibi alt alanları vardır.",
            "rejected": "Yapay zeka robotlar demek."
        },
        {
            "prompt": "Türkiye'nin en büyük şehri hangisidir?",
            "chosen": "Türkiye'nin en büyük şehri İstanbul'dur. Yaklaşık 15 milyon nüfusu ile hem Türkiye'nin hem de Avrupa'nın en kalabalık şehridir.",
            "rejected": "Ankara en büyük şehir."
        }
    ]
    
    # Train/val split
    train_data = grpo_data[:2]
    val_data = grpo_data[2:]
    
    # JSON Lines formatında kaydet
    os.makedirs('data', exist_ok=True)
    
    with open('data/grpo_train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open('data/grpo_val.jsonl', 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("GRPO veri dosyaları oluşturuldu: data/grpo_train.jsonl, data/grpo_val.jsonl")


def run_sft_training():
    """SFT eğitimini çalıştır"""
    print("\n=== SFT Eğitimi Başlıyor ===")
    
    # Model ve tokenizer oluştur
    config = M3TMConfig.get_tiny_config()
    model = M3TMBaseModel(config)
    
    # Basit tokenizer oluştur
    from m3tm.embedding.config import TokenizerConfig
    tokenizer_config = TokenizerConfig(vocab_size=1000, max_seq_length=512)
    tokenizer = SimpleTokenizer(tokenizer_config)
    
    # Basit vocab oluştur
    sample_texts = [
        "Python programlama dili",
        "Yapay zeka teknolojisi",
        "Makine öğrenmesi algoritmaları",
        "Derin öğrenme modelleri"
    ]
    tokenizer.build_vocab(sample_texts)
    
    # SFT config - model config ile uyumlu boyutlar
    sft_config = SFTConfig(
        learning_rate=2e-5,
        batch_size=2,
        epochs=2,
        max_seq_length=config.text_config.max_seq_len,  # Model config'den al
        logging_steps=1,
        eval_steps=2
    )
    
    # SFT trainer oluştur
    sft_trainer = SFTTrainer(
        model=model,
        config=sft_config,
        tokenizer=tokenizer,
        save_dir="./sft_checkpoints"
    )
    
    # Dataset oluştur
    train_dataset = sft_trainer.create_dataset('data/sft_train.jsonl', mode='train')
    val_dataset = sft_trainer.create_dataset('data/sft_val.jsonl', mode='val')
    
    # Model bilgilerini logla
    sft_trainer.log_model_info()
    
    # Eğitimi çalıştır
    try:
        results = sft_trainer.train(train_dataset, val_dataset)
        print(f"SFT Eğitimi tamamlandı! Sonuçlar: {results}")
        return True
    except Exception as e:
        print(f"SFT Eğitimi hatası: {e}")
        return False


def run_grpo_training():
    """GRPO eğitimini çalıştır"""
    print("\n=== GRPO Eğitimi Başlıyor ===")
    
    # Model ve tokenizer oluştur
    config = M3TMConfig.get_tiny_config()
    model = M3TMBaseModel(config)
    
    # Basit tokenizer oluştur
    from m3tm.embedding.config import TokenizerConfig
    tokenizer_config = TokenizerConfig(vocab_size=1000, max_seq_length=512)
    tokenizer = SimpleTokenizer(tokenizer_config)
    
    # Basit vocab oluştur
    sample_texts = [
        "Python programlama dili",
        "Yapay zeka teknolojisi", 
        "Makine öğrenmesi algoritmaları",
        "Derin öğrenme modelleri"
    ]
    tokenizer.build_vocab(sample_texts)
    
    # GRPO config - model config ile uyumlu boyutlar
    grpo_config = GRPOConfig(
        learning_rate=1e-5,
        batch_size=1,  # Küçük batch size
        epochs=1,
        max_seq_length=config.text_config.max_seq_len,  # Model config'den al
        beta=0.1,
        logging_steps=1,
        eval_steps=2,
        hidden_size=config.fusion_config.output_dim  # Model'den hidden size al
    )
    
    # GRPO trainer oluştur
    grpo_trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        tokenizer=tokenizer,
        save_dir="./grpo_checkpoints"
    )
    
    # Dataset oluştur
    train_dataset = grpo_trainer.create_dataset('data/grpo_train.jsonl', mode='train')
    val_dataset = grpo_trainer.create_dataset('data/grpo_val.jsonl', mode='val')
    
    # Model bilgilerini logla
    grpo_trainer.log_model_info()
    
    # Eğitimi çalıştır
    try:
        results = grpo_trainer.train(train_dataset, val_dataset)
        print(f"GRPO Eğitimi tamamlandı! Sonuçlar: {results}")
        return True
    except Exception as e:
        print(f"GRPO Eğitimi hatası: {e}")
        return False


def run_sequential_training():
    """Önce SFT, sonra GRPO ile sıralı eğitim"""
    print("\n=== Sıralı Eğitim: SFT → GRPO ===")
    
    # 1. SFT ile başla
    sft_success = run_sft_training()
    
    if not sft_success:
        print("SFT eğitimi başarısız, GRPO'ya geçilmiyor")
        return False
    
    print("\nSFT tamamlandı, GRPO'ya geçiliyor...")
    
    # 2. SFT'den sonra GRPO
    # Gerçek uygulamada SFT'den eğitilmiş modeli yüklersiniz
    grpo_success = run_grpo_training()
    
    if grpo_success:
        print("\n🎉 Sıralı eğitim başarıyla tamamlandı!")
        print("Model önce SFT ile instruction-following öğrendi,")
        print("sonra GRPO ile insan tercihlerine göre optimize edildi.")
        return True
    else:
        print("GRPO eğitimi başarısız")
        return False


def main():
    """Ana fonksiyon"""
    setup_logging()
    
    print("M³TM SFT ve GRPO Eğitim Örneği")
    print("=" * 50)
    
    # Örnek veri oluştur
    create_sample_sft_data()
    create_sample_grpo_data()
    
    # Eğitim seçenekleri
    print("\nEğitim seçenekleri:")
    print("1. Sadece SFT")
    print("2. Sadece GRPO") 
    print("3. Sıralı eğitim (SFT → GRPO)")
    
    choice = input("\nSeçiminizi yapın (1-3): ").strip()
    
    if choice == "1":
        run_sft_training()
    elif choice == "2":
        run_grpo_training()
    elif choice == "3":
        run_sequential_training()
    else:
        print("Geçersiz seçim, sıralı eğitim çalıştırılıyor...")
        run_sequential_training()


if __name__ == "__main__":
    main()
