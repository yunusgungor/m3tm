#!/usr/bin/env python3
"""
Veri Genişletme ve Synthetic Veri Üretimi Scripti
Bu script, mevcut yüksek kaliteli verileri kullanarak büyük eğitim veri setleri oluşturur.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataExpander:
    """Veri genişletme ve synthetic veri üretimi için yardımcı sınıf"""
    
    def __init__(self, base_data_dir: str = "/Users/yunusgungor/work/mobilemodel/data"):
        self.base_data_dir = Path(base_data_dir)
        self.expanded_data_dir = self.base_data_dir / "expanded"
        self.expanded_data_dir.mkdir(exist_ok=True)
        
        # Template'ler ve varyasyonlar için pattern'lar
        self.instruction_templates = [
            "{original_instruction}",
            "{original_instruction} Detaylı açıklama ile anlatır mısınız?",
            "{original_instruction} Pratik örneklerle açıklayın.",
            "{original_instruction} Kod örnekleri ile birlikte açıklayın.",
            "{original_instruction} Başlangıç seviyesi için açıklayın.",
            "{original_instruction} İleri seviye için detayları nelerdir?",
            "Mobil uygulama geliştirme bağlamında, {original_instruction}",
            "Python/AI/ML perspektifinden, {original_instruction}",
            "Performans optimizasyonu açısından, {original_instruction}",
            "En iyi uygulamalar ile birlikte, {original_instruction}"
        ]
        
        self.response_prefixes = [
            "Harika bir soru! ",
            "Mükemmel bir konu! ",
            "Çok önemli bir konu, ",
            "Bu konuda detaylı bilgi vereyim. ",
            "Açıklayayım: ",
            "Şu şekilde açıklayabilirim: ",
            "Bu konu hakkında kapsamlı bilgi: ",
            "Detayları ile açıklayacağım: "
        ]
    
    def load_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """JSONL dosyasını yükler"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
            logger.info(f"{file_path} dosyasından {len(data)} satır yüklendi")
        except Exception as e:
            logger.error(f"{file_path} dosyası yüklenirken hata: {e}")
        return data
    
    def save_jsonl_file(self, data: List[Dict[str, Any]], file_path: Path):
        """Veriyi JSONL formatında kaydeder"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"{len(data)} satır {file_path} dosyasına kaydedildi")
        except Exception as e:
            logger.error(f"{file_path} dosyasına yazılırken hata: {e}")
    
    def create_variations(self, original_data: Dict[str, Any], num_variations: int = 3) -> List[Dict[str, Any]]:
        """Bir veri örneğinden varyasyonlar oluşturur"""
        variations = []
        
        if 'instruction' in original_data and 'response' in original_data:
            # SFT formatı için varyasyonlar
            original_instruction = original_data['instruction']
            original_response = original_data['response']
            
            for i in range(num_variations):
                # Instruction varyasyonları
                template = random.choice(self.instruction_templates)
                new_instruction = template.format(original_instruction=original_instruction)
                
                # Response varyasyonları (prefix ekle)
                response_prefix = random.choice(self.response_prefixes)
                new_response = response_prefix + original_response
                
                variation = {
                    'instruction': new_instruction,
                    'response': new_response
                }
                
                # Orijinal metadata'yı koru
                for key in original_data:
                    if key not in ['instruction', 'response']:
                        variation[key] = original_data[key]
                
                # Kalite skorunu biraz düşür (çünkü synthetic)
                if 'quality_score' in variation:
                    variation['quality_score'] = max(0.6, variation['quality_score'] - 0.1)
                
                variations.append(variation)
                
        elif 'conversation' in original_data:
            # Multi-turn formatı için varyasyonlar
            for i in range(num_variations):
                variation = original_data.copy()
                # Conversation'ı değiştir
                if len(variation['conversation']) >= 2:
                    # İlk user message'ı varyasyonla
                    first_user_msg = variation['conversation'][0]['content']
                    template = random.choice(self.instruction_templates)
                    new_first_msg = template.format(original_instruction=first_user_msg)
                    
                    new_conversation = variation['conversation'].copy()
                    new_conversation[0]['content'] = new_first_msg
                    
                    # Assistant response'a prefix ekle
                    if len(new_conversation) > 1 and new_conversation[1]['role'] == 'assistant':
                        prefix = random.choice(self.response_prefixes)
                        new_conversation[1]['content'] = prefix + new_conversation[1]['content']
                    
                    variation['conversation'] = new_conversation
                
                variations.append(variation)
        
        return variations
    
    def expand_sft_data(self, target_size: int = 1000):
        """SFT verilerini genişletir"""
        logger.info("SFT verileri genişletiliyor...")
        
        # Mevcut kaliteli verileri yükle
        source_files = [
            self.base_data_dir / "optimized" / "ai_ml_high_quality.jsonl",
            self.base_data_dir / "optimized" / "programming_high_quality.jsonl",
            self.base_data_dir / "gemini" / "sft_train.jsonl",
            self.base_data_dir / "sft_train.jsonl"
        ]
        
        all_data = []
        for file_path in source_files:
            if file_path.exists():
                data = self.load_jsonl_file(file_path)
                all_data.extend(data)
        
        logger.info(f"Toplam {len(all_data)} orijinal veri yüklendi")
        
        # Varyasyonlar oluştur
        expanded_data = []
        expanded_data.extend(all_data)  # Orijinal verileri de ekle
        
        while len(expanded_data) < target_size and len(all_data) > 0:
            # Rastgele bir veriyi seç
            source_item = random.choice(all_data)
            # Varyasyonlar oluştur
            variations = self.create_variations(source_item, num_variations=3)
            expanded_data.extend(variations)
        
        # Hedef boyuta kırp
        expanded_data = expanded_data[:target_size]
        
        # Kaydet
        output_file = self.expanded_data_dir / "sft_train_expanded.jsonl"
        self.save_jsonl_file(expanded_data, output_file)
        
        # Validation set oluştur (%10)
        val_size = len(expanded_data) // 10
        val_data = expanded_data[:val_size]
        val_file = self.expanded_data_dir / "sft_val_expanded.jsonl"
        self.save_jsonl_file(val_data, val_file)
        
        logger.info(f"SFT verisi genişletildi: {len(expanded_data)} eğitim, {len(val_data)} validasyon")
        
        return output_file, val_file
    
    def expand_grpo_data(self, target_size: int = 500):
        """GRPO verilerini genişletir"""
        logger.info("GRPO verileri genişletiliyor...")
        
        # GRPO formatı için source verileri yükle
        source_files = [
            self.base_data_dir / "gemini" / "grpo_train.jsonl",
            self.base_data_dir / "grpo_train.jsonl"
        ]
        
        all_data = []
        for file_path in source_files:
            if file_path.exists():
                data = self.load_jsonl_file(file_path)
                all_data.extend(data)
        
        # GRPO verisi az ise, SFT verisinden dönüştür
        if len(all_data) < 10:
            logger.info("GRPO verisi az, SFT verisinden dönüştürülüyor...")
            sft_files = [
                self.base_data_dir / "optimized" / "ai_ml_high_quality.jsonl",
                self.base_data_dir / "optimized" / "programming_high_quality.jsonl"
            ]
            
            for file_path in sft_files:
                if file_path.exists():
                    sft_data = self.load_jsonl_file(file_path)
                    for item in sft_data:
                        if 'instruction' in item and 'response' in item:
                            # SFT'den GRPO formatına dönüştür
                            grpo_item = {
                                'prompt': item['instruction'],
                                'chosen': item['response'],
                                'rejected': 'Bu soruyu cevaplayamıyorum.' # Basit rejected response
                            }
                            # Metadata'yı koru
                            for key in item:
                                if key not in ['instruction', 'response']:
                                    grpo_item[key] = item[key]
                            all_data.append(grpo_item)
        
        logger.info(f"Toplam {len(all_data)} orijinal GRPO verisi hazırlandı")
        
        # Varyasyonlar oluştur
        expanded_data = []
        expanded_data.extend(all_data)
        
        while len(expanded_data) < target_size and len(all_data) > 0:
            source_item = random.choice(all_data)
            
            # GRPO için özel varyasyonlar
            if 'prompt' in source_item:
                for i in range(2):
                    template = random.choice(self.instruction_templates)
                    new_prompt = template.format(original_instruction=source_item['prompt'])
                    
                    variation = source_item.copy()
                    variation['prompt'] = new_prompt
                    
                    # Chosen response'a prefix ekle
                    if 'chosen' in variation:
                        prefix = random.choice(self.response_prefixes)
                        variation['chosen'] = prefix + variation['chosen']
                    
                    expanded_data.append(variation)
        
        # Hedef boyuta kırp
        expanded_data = expanded_data[:target_size]
        
        # Kaydet
        output_file = self.expanded_data_dir / "grpo_train_expanded.jsonl"
        self.save_jsonl_file(expanded_data, output_file)
        
        # Validation set
        val_size = len(expanded_data) // 10
        val_data = expanded_data[:val_size]
        val_file = self.expanded_data_dir / "grpo_val_expanded.jsonl"
        self.save_jsonl_file(val_data, val_file)
        
        logger.info(f"GRPO verisi genişletildi: {len(expanded_data)} eğitim, {len(val_data)} validasyon")
        
        return output_file, val_file

def main():
    """Ana fonksiyon"""
    expander = DataExpander()
    
    logger.info("=== Veri Genişletme İşlemi Başlatılıyor ===")
    
    # SFT verilerini genişlet
    sft_train_file, sft_val_file = expander.expand_sft_data(target_size=1000)
    
    # GRPO verilerini genişlet
    grpo_train_file, grpo_val_file = expander.expand_grpo_data(target_size=500)
    
    logger.info("=== Veri Genişletme Tamamlandı ===")
    logger.info(f"SFT Eğitim: {sft_train_file}")
    logger.info(f"SFT Validasyon: {sft_val_file}")
    logger.info(f"GRPO Eğitim: {grpo_train_file}")
    logger.info(f"GRPO Validasyon: {grpo_val_file}")
    
    # İstatistikler
    logger.info("\n=== Veri İstatistikleri ===")
    for file_path in [sft_train_file, sft_val_file, grpo_train_file, grpo_val_file]:
        if file_path.exists():
            with open(file_path, 'r') as f:
                count = sum(1 for line in f if line.strip())
            logger.info(f"{file_path.name}: {count} satır")

if __name__ == "__main__":
    main()
