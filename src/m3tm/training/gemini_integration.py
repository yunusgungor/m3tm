"""
Gemini-2.5-Flash API Integration for M³TM Training

Bu modül, Google'ın Gemini-2.5-Flash modelini M³TM eğitimi için kullanmak üzere
API entegrasyonu sağlar. Knowledge distillation, synthetic data generation ve
reward modeling için kullanılabilir.
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google GenerativeAI kütüphanesi bulunamadı. 'pip install google-generativeai' ile yükleyin.")


@dataclass
class GeminiConfig:
    """Gemini API konfigürasyonu"""
    api_key: Optional[str] = None
    model_name: str = "gemini-2.0-flash"  # Gemini-2.5-Flash henüz mevcut değilse
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_day: int = 1000
    
    # Safety settings
    safety_settings: Dict[str, str] = None
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")
        
        if self.safety_settings is None:
            self.safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", 
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }


class GeminiClient:
    """Gemini API client with rate limiting and error handling"""
    
    def __init__(self, config: GeminiConfig):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI kütüphanesi gerekli")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # API key kontrolü
        if not self.config.api_key:
            raise ValueError("GEMINI_API_KEY environment variable gerekli")
        
        # API'yi yapılandır
        genai.configure(api_key=self.config.api_key)
        
        # Model'i başlat
        generation_config = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=generation_config,
            safety_settings=self.config.safety_settings
        )
        
        # Rate limiting
        self.request_times = []
        self.daily_requests = 0
        self.last_reset_day = time.time() // 86400  # Day number
        
        self.logger.info(f"GeminiClient başlatıldı - Model: {self.config.model_name}")
    
    def _check_rate_limits(self):
        """Rate limit kontrolü"""
        current_time = time.time()
        current_day = current_time // 86400
        
        # Günlük reset
        if current_day > self.last_reset_day:
            self.daily_requests = 0
            self.last_reset_day = current_day
        
        # Günlük limit kontrolü
        if self.daily_requests >= self.config.requests_per_day:
            raise Exception(f"Günlük request limiti aşıldı: {self.config.requests_per_day}")
        
        # Dakikalık limit kontrolü
        minute_ago = current_time - 60
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        if len(self.request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            self.logger.warning(f"Rate limit, {sleep_time:.1f} saniye bekleniyor...")
            time.sleep(sleep_time)
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """Metin üretimi"""
        self._check_rate_limits()
        
        # System prompt varsa ekle
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        for attempt in range(max_retries):
            try:
                self.request_times.append(time.time())
                self.daily_requests += 1
                
                response = self.model.generate_content(full_prompt)
                
                if response.text:
                    return response.text.strip()
                else:
                    self.logger.warning(f"Boş response alındı, attempt {attempt + 1}")
                    
            except Exception as e:
                self.logger.error(f"Gemini API hatası (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        raise Exception("Gemini API'den response alınamadı")
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        batch_size: int = 5
    ) -> List[str]:
        """Batch metin üretimi"""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                try:
                    result = self.generate_text(prompt, system_prompt)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch item failed: {e}")
                    batch_results.append("")  # Empty result for failed items
            
            results.extend(batch_results)
            
            # Batch'ler arası kısa bekleme
            if i + batch_size < len(prompts):
                time.sleep(1)
        
        return results


class SyntheticDataGenerator:
    """Gemini kullanarak synthetic data üretimi"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        self.logger = logging.getLogger(__name__)
    
    def generate_sft_data(
        self,
        topics: List[str],
        num_samples_per_topic: int = 10,
        language: str = "Turkish"
    ) -> List[Dict[str, str]]:
        """SFT için instruction-response pairs üretir"""
        
        system_prompt = f"""Sen bir eğitim verisi üretim asistanısın. Verilen konular için kaliteli instruction-response çiftleri oluştur.

Kurallar:
1. {language} dilinde yaz
2. Instructions açık ve spesifik olmalı
3. Responses doğru, detaylı ve yararlı olmalı
4. Çeşitli zorluk seviyelerinde sorular sor
5. JSON formatında döndür

Format:
{{"instruction": "...", "response": "..."}}"""

        sft_data = []
        
        for topic in topics:
            self.logger.info(f"Konu için veri üretiliyor: {topic}")
            
            prompt = f"""Konu: {topic}

Bu konu için {num_samples_per_topic} farklı instruction-response çifti oluştur.
Çeşitli türlerde sorular sor: açıklama, örnek, karşılaştırma, uygulama, analiz.

ÖNEMLI: Response'lar kısa ve öz olsun (maksimum 200 kelime).
Özel karakterler ve uzun metinlerden kaçın.

Her bir çift için JSON formatında döndür:
{{"instruction": "soru veya görev", "response": "kısa ve yararlı cevap"}}

Çiftleri liste halinde döndür. Sadece JSON döndür, başka açıklama yapma."""

            try:
                response = self.client.generate_text(prompt, system_prompt)
                
                # JSON parse etmeye çalış
                try:
                    # Response'u temizle ve JSON parse et
                    cleaned_response = response.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]

                    # JSON array olarak parse etmeye çalış
                    if not cleaned_response.startswith('['):
                        # Tek bir JSON object ise array'e çevir
                        cleaned_response = f"[{cleaned_response}]"

                    parsed_data = json.loads(cleaned_response)

                    if isinstance(parsed_data, list):
                        sft_data.extend(parsed_data)
                    else:
                        sft_data.append(parsed_data)

                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON parse hatası: {e}")
                    # JSON parse başarısız olursa, basit format ile devam et
                    try:
                        # Basit instruction-response formatı oluştur
                        simple_item = {
                            "instruction": f"{topic} hakkında bilgi verin",
                            "response": response[:500] + "..." if len(response) > 500 else response
                        }
                        sft_data.append(simple_item)
                        self.logger.info(f"Basit format ile eklendi: {topic}")
                    except:
                        self.logger.error(f"Basit format da başarısız: {topic}")

            except Exception as e:
                self.logger.error(f"Veri üretim hatası: {e}")
        
        self.logger.info(f"Toplam {len(sft_data)} SFT örneği üretildi")
        return sft_data
    
    def generate_grpo_data(
        self,
        prompts: List[str],
        language: str = "Turkish"
    ) -> List[Dict[str, str]]:
        """GRPO için preference data üretir"""
        
        system_prompt = f"""Sen bir preference data üretim asistanısın. Verilen prompt'lar için bir iyi (chosen) ve bir kötü (rejected) response üret.

Kurallar:
1. {language} dilinde yaz
2. Chosen response: doğru, detaylı, yararlı, iyi yapılandırılmış
3. Rejected response: eksik, yanlış, belirsiz veya yararsız
4. Fark açık olmalı ama çok abartılı olmamalı
5. JSON formatında döndür

Format:
{{"prompt": "...", "chosen": "iyi response", "rejected": "kötü response"}}"""

        grpo_data = []
        
        for prompt in prompts:
            self.logger.info(f"Prompt için preference data üretiliyor: {prompt[:50]}...")
            
            generation_prompt = f"""Prompt: {prompt}

Bu prompt için bir iyi (chosen) ve bir kötü (rejected) response üret.

JSON formatında döndür:
{{"prompt": "{prompt}", "chosen": "kaliteli ve yararlı response", "rejected": "düşük kaliteli response"}}"""

            try:
                response = self.client.generate_text(generation_prompt, system_prompt)
                
                # JSON parse et
                try:
                    cleaned_response = response.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                    
                    parsed_data = json.loads(cleaned_response)
                    grpo_data.append(parsed_data)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON parse hatası: {e}")
                    
            except Exception as e:
                self.logger.error(f"Preference data üretim hatası: {e}")
        
        self.logger.info(f"Toplam {len(grpo_data)} GRPO örneği üretildi")
        return grpo_data
    
    def save_data(self, data: List[Dict], output_path: str):
        """Veriyi JSON Lines formatında kaydet"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Veri kaydedildi: {output_path} ({len(data)} örnek)")


class GeminiRewardModel:
    """Gemini'yi GRPO için reward model olarak kullanma"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        self.logger = logging.getLogger(__name__)
    
    def score_responses(
        self,
        prompt: str,
        responses: List[str],
        criteria: str = "doğruluk, yararlılık, açıklık"
    ) -> List[float]:
        """Response'ları skorlar (0-10 arası)"""
        
        system_prompt = f"""Sen bir response kalite değerlendirme asistanısın. Verilen prompt ve response'ları şu kriterlere göre 0-10 arası skorla:

Kriterler: {criteria}

Skorlama:
- 0-3: Çok kötü (yanlış, zararlı, alakasız)
- 4-5: Kötü (eksik, belirsiz)
- 6-7: Orta (kabul edilebilir ama geliştirilmeli)
- 8-9: İyi (doğru, yararlı, açık)
- 10: Mükemmel (çok doğru, çok yararlı, çok açık)

Sadece sayısal skor döndür, açıklama yapma."""

        scores = []
        
        for i, response in enumerate(responses):
            scoring_prompt = f"""Prompt: {prompt}

Response: {response}

Bu response'u 0-10 arası skorla. Sadece sayı döndür."""

            try:
                score_text = self.client.generate_text(scoring_prompt, system_prompt)
                
                # Sayısal skoru çıkar
                try:
                    score = float(score_text.strip())
                    score = max(0, min(10, score))  # 0-10 arası sınırla
                    scores.append(score)
                except ValueError:
                    self.logger.warning(f"Geçersiz skor: {score_text}")
                    scores.append(5.0)  # Varsayılan skor
                    
            except Exception as e:
                self.logger.error(f"Skorlama hatası: {e}")
                scores.append(5.0)  # Varsayılan skor
        
        return scores
    
    def compare_responses(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> float:
        """İki response'u karşılaştırır (-1 ile 1 arası)"""
        
        system_prompt = """Sen bir response karşılaştırma asistanısın. İki response'u karşılaştır ve hangisinin daha iyi olduğunu belirle.

Çıktı formatı:
- Response A çok daha iyiyse: 1.0
- Response A biraz daha iyiyse: 0.5
- Eşitlerse: 0.0
- Response B biraz daha iyiyse: -0.5
- Response B çok daha iyiyse: -1.0

Sadece sayı döndür."""

        comparison_prompt = f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Hangi response daha iyi? -1.0 ile 1.0 arası skor ver."""

        try:
            score_text = self.client.generate_text(comparison_prompt, system_prompt)
            
            try:
                score = float(score_text.strip())
                score = max(-1, min(1, score))  # -1 ile 1 arası sınırla
                return score
            except ValueError:
                self.logger.warning(f"Geçersiz karşılaştırma skoru: {score_text}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Karşılaştırma hatası: {e}")
            return 0.0
