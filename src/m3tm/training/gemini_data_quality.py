"""
Gemini Data Quality Enhancement Module

Bu modül, yüksek kaliteli eğitim verisi üretmek için
domain-specific prompt templates, multi-turn conversations
ve quality validation özellikleri sağlar.
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class DataQualityConfig:
    """Data quality konfigürasyonu"""
    min_response_length: int = 50
    max_response_length: int = 1000
    quality_threshold: float = 0.7
    diversity_threshold: float = 0.8
    language: str = "Turkish"
    domain_focus: Optional[str] = None


class PromptTemplate(ABC):
    """Base prompt template class"""
    
    @abstractmethod
    def generate_instruction(self, context: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def generate_system_prompt(self) -> str:
        pass
    
    @abstractmethod
    def validate_response(self, instruction: str, response: str) -> float:
        pass


class ProgrammingPromptTemplate(PromptTemplate):
    """Programming/coding specific prompts"""
    
    def __init__(self, language: str = "Turkish"):
        self.language = language
        self.programming_languages = [
            "Python", "JavaScript", "Java", "C++", "Go", "Rust", 
            "TypeScript", "C#", "PHP", "Ruby", "Swift", "Kotlin"
        ]
        self.topics = [
            "veri yapıları", "algoritmalar", "web geliştirme", "API tasarımı",
            "veritabanı", "testing", "debugging", "performance optimization",
            "design patterns", "clean code", "version control", "deployment"
        ]
    
    def generate_instruction(self, context: Dict[str, Any]) -> str:
        lang = random.choice(self.programming_languages)
        topic = random.choice(self.topics)
        
        templates = [
            f"{lang} ile {topic} konusunda bir örnek kod yazın ve açıklayın.",
            f"{lang}'da {topic} nasıl implement edilir? Kod örneği ile gösterin.",
            f"{topic} için {lang} best practices nelerdir?",
            f"{lang} ile {topic} yaparken dikkat edilmesi gereken noktalar nelerdir?",
            f"{lang}'da {topic} ile ilgili bir problem çözün ve adım adım açıklayın."
        ]
        
        return random.choice(templates)
    
    def generate_system_prompt(self) -> str:
        return f"""Sen bir uzman yazılım geliştirici ve eğitmensin. {self.language} dilinde yanıt ver.

Kurallar:
1. Kod örnekleri net ve çalışır durumda olmalı
2. Açıklamalar teknik ama anlaşılır olmalı
3. Best practices'i vurgula
4. Yaygın hataları belirt
5. Pratik örnekler kullan

Format:
- Kısa açıklama
- Kod örneği (syntax highlighting ile)
- Detaylı açıklama
- İpuçları ve best practices"""
    
    def validate_response(self, instruction: str, response: str) -> float:
        score = 0.0
        
        # Code block kontrolü
        if "```" in response:
            score += 0.3
        
        # Açıklama kalitesi
        if len(response) > 200:
            score += 0.2
        
        # Teknik terimler
        technical_terms = ["function", "class", "variable", "method", "algorithm", "data structure"]
        found_terms = sum(1 for term in technical_terms if term.lower() in response.lower())
        score += min(0.3, found_terms * 0.1)
        
        # Best practices mention
        if any(word in response.lower() for word in ["best practice", "öneril", "dikkat", "hata"]):
            score += 0.2
        
        return min(1.0, score)


class AIMLPromptTemplate(PromptTemplate):
    """AI/ML specific prompts"""
    
    def __init__(self, language: str = "Turkish"):
        self.language = language
        self.topics = [
            "makine öğrenmesi", "deep learning", "neural networks", "NLP",
            "computer vision", "reinforcement learning", "data preprocessing",
            "model evaluation", "feature engineering", "hyperparameter tuning",
            "transfer learning", "ensemble methods", "clustering", "classification"
        ]
        self.frameworks = ["TensorFlow", "PyTorch", "Scikit-learn", "Keras", "Hugging Face"]
    
    def generate_instruction(self, context: Dict[str, Any]) -> str:
        topic = random.choice(self.topics)
        framework = random.choice(self.frameworks)
        
        templates = [
            f"{topic} nedir ve nasıl çalışır? Pratik örnekle açıklayın.",
            f"{framework} kullanarak {topic} nasıl implement edilir?",
            f"{topic} için hangi algoritmalar kullanılır? Karşılaştırın.",
            f"{topic} projesinde karşılaşılan yaygın problemler ve çözümleri nelerdir?",
            f"{topic} modelinin performansını nasıl değerlendiririz?"
        ]
        
        return random.choice(templates)
    
    def generate_system_prompt(self) -> str:
        return f"""Sen bir AI/ML uzmanı ve araştırmacısısın. {self.language} dilinde yanıt ver.

Kurallar:
1. Teorik bilgiyi pratik örneklerle destekle
2. Matematiksel kavramları basit terimlerle açıkla
3. Kod örnekleri ekle (Python tercih et)
4. Güncel teknolojileri ve best practices'i kullan
5. Yaygın hataları ve pitfall'ları belirt

Format:
- Kavramsal açıklama
- Matematiksel formül (gerekirse)
- Kod örneği
- Pratik uygulama önerileri
- İleri okuma kaynakları"""
    
    def validate_response(self, instruction: str, response: str) -> float:
        score = 0.0
        
        # AI/ML terminoloji
        ml_terms = ["model", "algorithm", "training", "validation", "accuracy", "loss", "gradient"]
        found_terms = sum(1 for term in ml_terms if term.lower() in response.lower())
        score += min(0.4, found_terms * 0.05)
        
        # Code example
        if "```python" in response or "import" in response:
            score += 0.3
        
        # Mathematical content
        if any(symbol in response for symbol in ["=", "+", "-", "*", "/", "^", "∑", "∫"]):
            score += 0.2
        
        # Practical advice
        if any(word in response.lower() for word in ["öneri", "dikkat", "pratik", "uygulama"]):
            score += 0.1
        
        return min(1.0, score)


class BusinessPromptTemplate(PromptTemplate):
    """Business/entrepreneurship specific prompts"""
    
    def __init__(self, language: str = "Turkish"):
        self.language = language
        self.topics = [
            "startup", "business plan", "marketing", "sales", "finance",
            "leadership", "management", "strategy", "innovation", "digital transformation",
            "customer experience", "brand building", "investment", "scaling"
        ]
    
    def generate_instruction(self, context: Dict[str, Any]) -> str:
        topic = random.choice(self.topics)
        
        templates = [
            f"{topic.title()} alanında başarılı olmak için hangi stratejiler kullanılır?",
            f"Küçük bir işletme için {topic} nasıl planlanır ve uygulanır?",
            f"{topic} konusunda yaygın hatalar nelerdir ve nasıl önlenir?",
            f"{topic} için güncel trendler ve gelecek öngörüleri nelerdir?",
            f"{topic} alanında case study örneği verin ve analiz edin."
        ]
        
        return random.choice(templates)
    
    def generate_system_prompt(self) -> str:
        return f"""Sen bir deneyimli business consultant ve girişimcisin. {self.language} dilinde yanıt ver.

Kurallar:
1. Pratik ve uygulanabilir öneriler ver
2. Gerçek case study'ler ve örnekler kullan
3. Sayısal veriler ve metrikler ekle
4. Risk faktörlerini belirt
5. Adım adım action plan'lar oluştur

Format:
- Problem/durum analizi
- Stratejik yaklaşım
- Uygulama adımları
- Başarı metrikleri
- Risk yönetimi
- Örnek case study"""
    
    def validate_response(self, instruction: str, response: str) -> float:
        score = 0.0
        
        # Business terminology
        business_terms = ["strategy", "market", "customer", "revenue", "profit", "growth", "ROI"]
        found_terms = sum(1 for term in business_terms if term.lower() in response.lower())
        score += min(0.3, found_terms * 0.05)
        
        # Actionable advice
        if any(word in response.lower() for word in ["adım", "plan", "strateji", "öneri", "uygula"]):
            score += 0.3
        
        # Metrics/numbers
        if any(char.isdigit() for char in response) or "%" in response:
            score += 0.2
        
        # Case study/example
        if any(word in response.lower() for word in ["örnek", "case", "şirket", "deneyim"]):
            score += 0.2
        
        return min(1.0, score)


class MultiTurnConversationGenerator:
    """Multi-turn conversation generator"""
    
    def __init__(self, prompt_template: PromptTemplate, max_turns: int = 5):
        self.prompt_template = prompt_template
        self.max_turns = max_turns
        self.logger = logging.getLogger(__name__)
    
    def generate_conversation(
        self, 
        initial_context: Dict[str, Any],
        gemini_client
    ) -> List[Dict[str, str]]:
        """Multi-turn conversation üret"""
        conversation = []
        context = initial_context.copy()
        
        # İlk instruction
        instruction = self.prompt_template.generate_instruction(context)
        system_prompt = self.prompt_template.generate_system_prompt()
        
        try:
            response = gemini_client.generate_text(instruction, system_prompt)
            conversation.append({
                "role": "user",
                "content": instruction
            })
            conversation.append({
                "role": "assistant", 
                "content": response
            })
            
            # Follow-up questions
            for turn in range(1, self.max_turns):
                follow_up = self._generate_follow_up(conversation, context)
                if not follow_up:
                    break
                
                response = gemini_client.generate_text(follow_up, system_prompt)
                conversation.append({
                    "role": "user",
                    "content": follow_up
                })
                conversation.append({
                    "role": "assistant",
                    "content": response
                })
                
        except Exception as e:
            self.logger.error(f"Conversation generation error: {e}")
        
        return conversation
    
    def _generate_follow_up(
        self, 
        conversation: List[Dict[str, str]], 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Follow-up question üret"""
        if len(conversation) < 2:
            return None
        
        last_response = conversation[-1]["content"]
        
        follow_up_templates = [
            "Bu konuda daha detaylı bilgi verebilir misiniz?",
            "Pratik bir örnek verebilir misiniz?",
            "Bu yaklaşımın avantajları ve dezavantajları nelerdir?",
            "Alternatif yöntemler var mı?",
            "Bu konuda yaygın hatalar nelerdir?",
            "Başlangıç seviyesindeki biri için önerileriniz nelerdir?",
            "İleri seviye için hangi konulara odaklanmalı?",
            "Bu alanda güncel trendler nelerdir?"
        ]
        
        return random.choice(follow_up_templates)


class QualityValidator:
    """Data quality validation"""
    
    def __init__(self, config: DataQualityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_instruction_response_pair(
        self, 
        instruction: str, 
        response: str,
        prompt_template: Optional[PromptTemplate] = None
    ) -> Dict[str, Any]:
        """Instruction-response pair'ini validate et"""
        
        validation_result = {
            "is_valid": True,
            "quality_score": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        # Length validation
        if len(response) < self.config.min_response_length:
            validation_result["issues"].append("Response çok kısa")
            validation_result["is_valid"] = False
        
        if len(response) > self.config.max_response_length:
            validation_result["issues"].append("Response çok uzun")
            validation_result["suggestions"].append("Response'u kısaltın")
        
        # Content quality
        quality_score = 0.0
        
        # Basic quality checks
        if response.strip():
            quality_score += 0.2
        
        if len(response.split()) > 10:
            quality_score += 0.2
        
        if any(char in response for char in ".!?"):
            quality_score += 0.1
        
        # Template-specific validation
        if prompt_template:
            template_score = prompt_template.validate_response(instruction, response)
            quality_score += template_score * 0.5
        
        validation_result["quality_score"] = min(1.0, quality_score)
        
        # Quality threshold check
        if validation_result["quality_score"] < self.config.quality_threshold:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Quality score düşük")
        
        return validation_result
    
    def validate_dataset(self, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """Tüm dataset'i validate et"""
        total_items = len(dataset)
        valid_items = 0
        quality_scores = []
        all_issues = []
        
        for item in dataset:
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            
            validation = self.validate_instruction_response_pair(instruction, response)
            
            if validation["is_valid"]:
                valid_items += 1
            
            quality_scores.append(validation["quality_score"])
            all_issues.extend(validation["issues"])
        
        return {
            "total_items": total_items,
            "valid_items": valid_items,
            "validity_rate": valid_items / total_items if total_items > 0 else 0.0,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "common_issues": self._get_common_issues(all_issues),
            "recommendations": self._get_recommendations(all_issues)
        }
    
    def _get_common_issues(self, issues: List[str]) -> Dict[str, int]:
        """Yaygın sorunları say"""
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _get_recommendations(self, issues: List[str]) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        if "Response çok kısa" in issues:
            recommendations.append("Daha detaylı ve açıklayıcı response'lar üretin")
        
        if "Quality score düşük" in issues:
            recommendations.append("Prompt template'leri gözden geçirin ve iyileştirin")
        
        return recommendations
