#!/usr/bin/env python3
"""
Advanced Reward Functions for GRPO Training

Bu modül, GRPO eğitimi için gelişmiş reward fonksiyonları sağlar.
Çeşitli metrikler ve kalite ölçümleri ile model davranışını optimize eder.

Features:
- Length-based rewards
- Quality-based rewards  
- Format-based rewards
- Semantic similarity rewards
- Safety-based rewards
- Multi-objective reward combination

Author: GitHub Copilot
Date: 28 Haziran 2025
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModel

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class RewardConfig:
    """Reward fonksiyonu konfigürasyonu."""
    
    # Length-based rewards
    target_length: int = 100
    length_tolerance: int = 20
    length_weight: float = 0.3
    
    # Quality-based rewards
    diversity_weight: float = 0.4
    repetition_penalty: float = 0.5
    coherence_weight: float = 0.3
    quality_weight: float = 0.4
    
    # Format-based rewards
    capitalization_weight: float = 0.2
    punctuation_weight: float = 0.2
    grammar_weight: float = 0.3
    format_weight: float = 0.3
    
    # Semantic rewards
    semantic_similarity_weight: float = 0.2
    relevance_weight: float = 0.4
    
    # Safety rewards
    safety_weight: float = 1.0
    toxicity_threshold: float = 0.8
    
    # Overall weights
    normalize_rewards: bool = True
    reward_clipping: Optional[Tuple[float, float]] = (-5.0, 5.0)


class BaseRewardFunction:
    """Base reward fonksiyonu sınıfı."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, 
                 completions: List[str], 
                 prompts: Optional[List[str]] = None,
                 **kwargs) -> List[float]:
        """Reward hesaplama ana fonksiyonu."""
        raise NotImplementedError
    
    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        """Reward'ları normalize eder."""
        if not self.config.normalize_rewards:
            return rewards
            
        if len(rewards) <= 1:
            return rewards
            
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8
        
        normalized = (rewards - mean_reward) / std_reward
        return normalized.tolist()
    
    def _clip_rewards(self, rewards: List[float]) -> List[float]:
        """Reward'ları belirli aralığa sınırlar."""
        if self.config.reward_clipping is None:
            return rewards
            
        min_val, max_val = self.config.reward_clipping
        return [max(min_val, min(max_val, r)) for r in rewards]


class LengthRewardFunction(BaseRewardFunction):
    """Uzunluk tabanlı reward fonksiyonu."""
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Hedef uzunluğa yakınlığa göre reward hesaplar.
        
        Args:
            completions: Model çıktıları
            
        Returns:
            Length-based rewards
        """
        rewards = []
        target = self.config.target_length
        tolerance = self.config.length_tolerance
        
        for completion in completions:
            length = len(completion.split())
            
            # Hedef uzunluğa yakınlık hesapla
            distance = abs(length - target)
            
            if distance <= tolerance:
                # Tolerance içindeyse maksimum reward
                reward = 1.0
            else:
                # Tolerance dışındaysa mesafeye göre azal
                penalty = (distance - tolerance) / target
                reward = max(0.0, 1.0 - penalty)
            
            rewards.append(reward)
        
        return self._clip_rewards(rewards)


class QualityRewardFunction(BaseRewardFunction):
    """Kalite tabanlı reward fonksiyonu."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Sentence embedding model (opsiyonel)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.sentence_model = None
        else:
            self.sentence_model = None
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Çeşitli kalite metriklerine göre reward hesaplar.
        
        Args:
            completions: Model çıktıları
            
        Returns:
            Quality-based rewards
        """
        rewards = []
        
        for completion in completions:
            reward = 0.0
            
            # 1. Çeşitlilik (vocabulary diversity)
            diversity_score = self._calculate_diversity(completion)
            reward += self.config.diversity_weight * diversity_score
            
            # 2. Tekrar cezası
            repetition_score = self._calculate_repetition_penalty(completion)
            reward -= self.config.repetition_penalty * repetition_score
            
            # 3. Tutarlılık (coherence)
            coherence_score = self._calculate_coherence(completion)
            reward += self.config.coherence_weight * coherence_score
            
            rewards.append(reward)
        
        return self._normalize_rewards(self._clip_rewards(rewards))
    
    def _calculate_diversity(self, text: str) -> float:
        """Kelime çeşitliliğini hesaplar."""
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
            
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        return diversity
    
    def _calculate_repetition_penalty(self, text: str) -> float:
        """Tekrar cezasını hesaplar."""
        words = text.split()
        if len(words) <= 1:
            return 0.0
        
        # Ardışık kelime tekrarları
        consecutive_repeats = 0
        for i in range(len(words) - 1):
            if words[i].lower() == words[i + 1].lower():
                consecutive_repeats += 1
        
        # N-gram tekrarları
        bigram_repeats = 0
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        bigram_counts = {}
        for bigram in bigrams:
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            if bigram_counts[bigram] > 1:
                bigram_repeats += 1
        
        total_penalty = (consecutive_repeats + bigram_repeats) / len(words)
        return min(1.0, total_penalty)
    
    def _calculate_coherence(self, text: str) -> float:
        """Metin tutarlılığını hesaplar."""
        sentences = text.split('.')
        if len(sentences) <= 1:
            return 1.0
        
        # Sentence embedding kullanarak tutarlılık hesapla
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode(sentences)
                if len(embeddings) > 1:
                    # Ardışık cümleler arası benzerlik
                    similarities = []
                    for i in range(len(embeddings) - 1):
                        sim = np.dot(embeddings[i], embeddings[i + 1])
                        similarities.append(sim)
                    return np.mean(similarities)
            except:
                pass
        
        # Fallback: kelime overlap tabanlı tutarlılık
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            if len(words1) == 0 or len(words2) == 0:
                continue
                
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            score = overlap / total if total > 0 else 0
            coherence_scores.append(score)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5


class FormatRewardFunction(BaseRewardFunction):
    """Format tabanlı reward fonksiyonu."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Grammar checker (opsiyonel)
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
        else:
            self.nlp = None
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Format kuralllarına göre reward hesaplar.
        
        Args:
            completions: Model çıktıları
            
        Returns:
            Format-based rewards
        """
        rewards = []
        
        for completion in completions:
            reward = 0.0
            
            # 1. Büyük harf kullanımı
            capitalization_score = self._check_capitalization(completion)
            reward += self.config.capitalization_weight * capitalization_score
            
            # 2. Noktalama işaretleri
            punctuation_score = self._check_punctuation(completion)
            reward += self.config.punctuation_weight * punctuation_score
            
            # 3. Gramer kontrolü (opsiyonel)
            if self.nlp:
                grammar_score = self._check_grammar(completion)
                reward += self.config.grammar_weight * grammar_score
            
            rewards.append(reward)
        
        return self._normalize_rewards(self._clip_rewards(rewards))
    
    def _check_capitalization(self, text: str) -> float:
        """Büyük harf kullanımını kontrol eder."""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Metin büyük harfle başlıyor mu?
        if text[0].isupper():
            score += 0.5
        
        # Cümle başları büyük harf mi?
        sentences = re.split(r'[.!?]+', text)
        capital_sentences = 0
        total_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                total_sentences += 1
                if sentence[0].isupper():
                    capital_sentences += 1
        
        if total_sentences > 0:
            score += 0.5 * (capital_sentences / total_sentences)
        
        return min(1.0, score)
    
    def _check_punctuation(self, text: str) -> float:
        """Noktalama işaretlerini kontrol eder."""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Metin noktalama ile bitiyor mu?
        if text.rstrip()[-1:] in '.!?':
            score += 0.5
        
        # Uygun noktalama yoğunluğu
        words = text.split()
        punctuation_marks = re.findall(r'[.!?,:;]', text)
        
        if len(words) > 0:
            punct_ratio = len(punctuation_marks) / len(words)
            # Optimal noktalama oranı: %5-15 arası
            if 0.05 <= punct_ratio <= 0.15:
                score += 0.5
            else:
                # Çok az veya çok fazla noktalama cezası
                penalty = abs(punct_ratio - 0.1) / 0.1
                score += 0.5 * max(0, 1 - penalty)
        
        return min(1.0, score)
    
    def _check_grammar(self, text: str) -> float:
        """Gramer kontrolü yapar (spaCy gerekli)."""
        if not self.nlp or not text:
            return 0.5  # Neutral score
        
        try:
            doc = self.nlp(text)
            
            # Temel gramer kontrolü
            total_tokens = len(doc)
            error_count = 0
            
            for token in doc:
                # Kelime yazım hatası (basit kontrol)
                if token.is_alpha and not token.is_stop and token.is_oov:
                    error_count += 1
            
            # Hata oranı
            if total_tokens > 0:
                error_ratio = error_count / total_tokens
                grammar_score = max(0, 1 - 2 * error_ratio)  # 2x penalty
                return grammar_score
            
        except:
            pass
        
        return 0.5


class SemanticRewardFunction(BaseRewardFunction):
    """Semantik benzerlik tabanlı reward fonksiyonu."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Sentence embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.sentence_model = None
        else:
            self.sentence_model = None
    
    def __call__(self, 
                 completions: List[str], 
                 prompts: Optional[List[str]] = None,
                 **kwargs) -> List[float]:
        """
        Prompt ile completion arasındaki semantik benzerliği ölçer.
        
        Args:
            completions: Model çıktıları
            prompts: Orijinal promptlar
            
        Returns:
            Semantic similarity rewards
        """
        if not prompts or not self.sentence_model:
            return [0.5] * len(completions)  # Neutral scores
        
        rewards = []
        
        try:
            # Embeddings hesapla
            prompt_embeddings = self.sentence_model.encode(prompts)
            completion_embeddings = self.sentence_model.encode(completions)
            
            for i in range(len(completions)):
                # Cosine similarity
                similarity = np.dot(prompt_embeddings[i], completion_embeddings[i])
                
                # 0-1 arası normalize et
                reward = (similarity + 1) / 2
                rewards.append(reward)
        
        except Exception as e:
            self.logger.warning(f"Semantic reward hesaplamasında hata: {e}")
            rewards = [0.5] * len(completions)
        
        return rewards


class SafetyRewardFunction(BaseRewardFunction):
    """Güvenlik tabanlı reward fonksiyonu."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Toxicity detection pipeline
        try:
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            self.toxicity_classifier = None
            
        # Zararlı içerik pattern'leri
        self.harmful_patterns = [
            r'\b(hate|kill|die|stupid)\b',
            r'\b(f\*ck|sh\*t|damn)\b',  # Küfür pattern'leri
            r'\b(violence|harmful|dangerous)\b'
        ]
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Güvenlik açısından completion'ları değerlendirir.
        
        Args:
            completions: Model çıktıları
            
        Returns:
            Safety-based rewards
        """
        rewards = []
        
        for completion in completions:
            reward = 1.0  # Başlangıç: güvenli kabul et
            
            # 1. Pattern tabanlı kontrol
            pattern_penalty = self._check_harmful_patterns(completion)
            reward -= pattern_penalty
            
            # 2. ML-based toxicity detection
            if self.toxicity_classifier:
                toxicity_penalty = self._check_toxicity(completion)
                reward -= toxicity_penalty
            
            # 3. Uzunluk tabanlı güvenlik (çok kısa/uzun metinler şüpheli)
            length_penalty = self._check_length_safety(completion)
            reward -= length_penalty
            
            rewards.append(max(0.0, reward))  # Negatif olamaz
        
        return rewards
    
    def _check_harmful_patterns(self, text: str) -> float:
        """Zararlı pattern'leri kontrol eder."""
        penalty = 0.0
        text_lower = text.lower()
        
        for pattern in self.harmful_patterns:
            matches = re.findall(pattern, text_lower)
            penalty += len(matches) * 0.2  # Her eşleşme için 0.2 ceza
        
        return min(1.0, penalty)
    
    def _check_toxicity(self, text: str) -> float:
        """ML model ile toxicity kontrol eder."""
        try:
            result = self.toxicity_classifier(text)
            
            # TOXIC etiketi varsa ceza ver
            for item in result:
                if item['label'] == 'TOXIC' and item['score'] > self.config.toxicity_threshold:
                    return item['score']  # Yüksek güven = yüksek ceza
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Toxicity kontrolünde hata: {e}")
            return 0.0
    
    def _check_length_safety(self, text: str) -> float:
        """Uzunluk tabanlı güvenlik kontrol eder."""
        word_count = len(text.split())
        
        # Çok kısa metinler (< 5 kelime) şüpheli
        if word_count < 5:
            return 0.3
        
        # Çok uzun metinler (> 500 kelime) şüpheli
        if word_count > 500:
            return 0.2
        
        return 0.0


class ComprehensiveRewardFunction(BaseRewardFunction):
    """Tüm reward fonksiyonlarını birleştiren kapsamlı reward fonksiyonu."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Alt reward fonksiyonları
        self.length_reward = LengthRewardFunction(config)
        self.quality_reward = QualityRewardFunction(config)
        self.format_reward = FormatRewardFunction(config)
        self.semantic_reward = SemanticRewardFunction(config)
        self.safety_reward = SafetyRewardFunction(config)
        
        self.logger.info("Comprehensive reward function initialized")
    
    def __call__(self, 
                 completions: List[str], 
                 prompts: Optional[List[str]] = None,
                 **kwargs) -> List[float]:
        """
        Tüm reward türlerini birleştirerek final reward hesaplar.
        
        Args:
            completions: Model çıktıları
            prompts: Orijinal promptlar (opsiyonel)
            
        Returns:
            Combined rewards
        """
        # Her reward türünü hesapla
        length_rewards = self.length_reward(completions, **kwargs)
        quality_rewards = self.quality_reward(completions, **kwargs)
        format_rewards = self.format_reward(completions, **kwargs)
        semantic_rewards = self.semantic_reward(completions, prompts, **kwargs)
        safety_rewards = self.safety_reward(completions, **kwargs)
        
        # Ağırlıklı kombinasyon
        final_rewards = []
        for i in range(len(completions)):
            reward = (
                self.config.length_weight * length_rewards[i] +
                self.config.quality_weight * quality_rewards[i] +
                self.config.format_weight * format_rewards[i] +
                self.config.semantic_similarity_weight * semantic_rewards[i] +
                self.config.safety_weight * safety_rewards[i]
            )
            final_rewards.append(reward)
        
        # Normalize ve clip
        final_rewards = self._normalize_rewards(final_rewards)
        final_rewards = self._clip_rewards(final_rewards)
        
        # Debugging info
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, completion in enumerate(completions[:3]):  # İlk 3 örnek
                self.logger.debug(
                    f"Completion {i}: "
                    f"Length={length_rewards[i]:.3f}, "
                    f"Quality={quality_rewards[i]:.3f}, "
                    f"Format={format_rewards[i]:.3f}, "
                    f"Semantic={semantic_rewards[i]:.3f}, "
                    f"Safety={safety_rewards[i]:.3f}, "
                    f"Final={final_rewards[i]:.3f}"
                )
        
        return final_rewards


def create_reward_function(reward_type: str = "comprehensive", 
                          config: Optional[RewardConfig] = None) -> BaseRewardFunction:
    """
    Reward fonksiyonu factory.
    
    Args:
        reward_type: Reward fonksiyonu türü
        config: Reward konfigürasyonu
        
    Returns:
        Reward fonksiyonu instance
    """
    if config is None:
        config = RewardConfig()
    
    reward_functions = {
        "length": LengthRewardFunction,
        "quality": QualityRewardFunction,
        "format": FormatRewardFunction,
        "semantic": SemanticRewardFunction,
        "safety": SafetyRewardFunction,
        "comprehensive": ComprehensiveRewardFunction,
    }
    
    if reward_type not in reward_functions:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    return reward_functions[reward_type](config)


# Convenience functions for TRL integration
def length_reward_function(completions: List[str], **kwargs) -> List[float]:
    """TRL için basit uzunluk reward fonksiyonu."""
    config = RewardConfig(target_length=kwargs.get('target_length', 100))
    reward_fn = LengthRewardFunction(config)
    return reward_fn(completions, **kwargs)


def quality_reward_function(completions: List[str], **kwargs) -> List[float]:
    """TRL için basit kalite reward fonksiyonu.""" 
    config = RewardConfig()
    reward_fn = QualityRewardFunction(config)
    return reward_fn(completions, **kwargs)


def comprehensive_reward_function(completions: List[str], 
                                 prompts: Optional[List[str]] = None,
                                 **kwargs) -> List[float]:
    """TRL için kapsamlı reward fonksiyonu."""
    config = RewardConfig()
    reward_fn = ComprehensiveRewardFunction(config)
    return reward_fn(completions, prompts, **kwargs)


if __name__ == "__main__":
    # Test reward functions
    test_completions = [
        "This is a well-written sentence with proper grammar and punctuation.",
        "this is bad no caps no punct",
        "This sentence is good! It has variety, proper formatting, and interesting content that flows naturally.",
        "Same same same same same repeated words everywhere same same.",
    ]
    
    test_prompts = [
        "Write a good sentence:",
        "Write a sentence:",
        "Create an interesting sentence:",
        "Write something:",
    ]
    
    # Test comprehensive reward
    config = RewardConfig()
    reward_fn = ComprehensiveRewardFunction(config)
    
    rewards = reward_fn(test_completions, test_prompts)
    
    print("Test Results:")
    for i, (completion, reward) in enumerate(zip(test_completions, rewards)):
        print(f"{i+1}. Reward: {reward:.3f} | Text: {completion[:50]}...")
