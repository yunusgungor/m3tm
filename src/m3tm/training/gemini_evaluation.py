"""
Gemini Training Evaluation & Monitoring Module

Bu modül, model performansını değerlendirmek ve eğitim sürecini
monitor etmek için comprehensive evaluation metrics sağlar.
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from collections import defaultdict, deque


@dataclass
class EvaluationConfig:
    """Evaluation konfigürasyonu"""
    metrics: List[str] = field(default_factory=lambda: [
        "bleu", "rouge", "semantic_similarity", "coherence", "relevance"
    ])
    reference_model: Optional[str] = None
    human_evaluation: bool = False
    automated_evaluation: bool = True
    save_results: bool = True
    results_dir: str = "evaluation_results"


class TextMetrics:
    """Text quality metrics"""
    
    @staticmethod
    def calculate_bleu(reference: str, candidate: str) -> float:
        """BLEU score hesapla (basit implementasyon)"""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if not cand_words:
            return 0.0
        
        # 1-gram precision
        ref_counts = {}
        for word in ref_words:
            ref_counts[word] = ref_counts.get(word, 0) + 1
        
        cand_counts = {}
        for word in cand_words:
            cand_counts[word] = cand_counts.get(word, 0) + 1
        
        matches = 0
        for word, count in cand_counts.items():
            matches += min(count, ref_counts.get(word, 0))
        
        precision = matches / len(cand_words) if cand_words else 0.0
        
        # Brevity penalty (basit)
        bp = min(1.0, len(cand_words) / len(ref_words)) if ref_words else 0.0
        
        return bp * precision
    
    @staticmethod
    def calculate_rouge_l(reference: str, candidate: str) -> float:
        """ROUGE-L score hesapla"""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if not ref_words or not cand_words:
            return 0.0
        
        # Longest Common Subsequence
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(ref_words, cand_words)
        
        if lcs_len == 0:
            return 0.0
        
        precision = lcs_len / len(cand_words)
        recall = lcs_len / len(ref_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_semantic_similarity(text1: str, text2: str) -> float:
        """Semantic similarity (basit word overlap)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def calculate_coherence(text: str) -> float:
        """Text coherence score"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent
        
        # Basit coherence: sentence'lar arası word overlap
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            similarity = TextMetrics.calculate_semantic_similarity(
                sentences[i], sentences[i + 1]
            )
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    @staticmethod
    def calculate_relevance(instruction: str, response: str) -> float:
        """Instruction-response relevance"""
        return TextMetrics.calculate_semantic_similarity(instruction, response)


class ModelEvaluator:
    """Model performance evaluator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        if config.save_results:
            self.results_dir = Path(config.results_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self._init_results_db()
    
    def _init_results_db(self):
        """Results database'ini başlat"""
        self.db_path = self.results_dir / "evaluation_results.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    model_name TEXT,
                    dataset_name TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluations(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_metric ON evaluations(model_name, metric_name)
            """)
    
    def evaluate_responses(
        self,
        instructions: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        model_name: str = "gemini_model",
        dataset_name: str = "test_set"
    ) -> Dict[str, Any]:
        """Response'ları evaluate et"""
        
        if len(instructions) != len(responses):
            raise ValueError("Instructions ve responses sayısı eşit olmalı")
        
        if references and len(references) != len(responses):
            raise ValueError("References ve responses sayısı eşit olmalı")
        
        results = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": time.time(),
            "total_samples": len(responses),
            "metrics": {}
        }
        
        # Metric calculations
        for metric in self.config.metrics:
            metric_scores = []
            
            for i, (instruction, response) in enumerate(zip(instructions, responses)):
                reference = references[i] if references else None
                
                if metric == "bleu" and reference:
                    score = TextMetrics.calculate_bleu(reference, response)
                elif metric == "rouge" and reference:
                    score = TextMetrics.calculate_rouge_l(reference, response)
                elif metric == "semantic_similarity" and reference:
                    score = TextMetrics.calculate_semantic_similarity(reference, response)
                elif metric == "coherence":
                    score = TextMetrics.calculate_coherence(response)
                elif metric == "relevance":
                    score = TextMetrics.calculate_relevance(instruction, response)
                else:
                    continue  # Skip unsupported metrics
                
                metric_scores.append(score)
            
            if metric_scores:
                results["metrics"][metric] = {
                    "mean": np.mean(metric_scores),
                    "std": np.std(metric_scores),
                    "min": np.min(metric_scores),
                    "max": np.max(metric_scores),
                    "scores": metric_scores
                }
        
        # Save results
        if self.config.save_results:
            self._save_results(results)
        
        self.logger.info(f"Evaluation completed: {model_name} on {dataset_name}")
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Results'ları database'e kaydet"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for metric_name, metric_data in results["metrics"].items():
                    conn.execute("""
                        INSERT INTO evaluations 
                        (timestamp, model_name, dataset_name, metric_name, metric_value, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        results["timestamp"],
                        results["model_name"],
                        results["dataset_name"],
                        metric_name,
                        metric_data["mean"],
                        json.dumps(metric_data)
                    ))
        except Exception as e:
            self.logger.error(f"Results save error: {e}")
    
    def compare_models(
        self,
        model_names: List[str],
        dataset_name: Optional[str] = None,
        time_window_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """Model'leri karşılaştır"""
        
        if not self.config.save_results:
            raise ValueError("Model comparison requires save_results=True")
        
        comparison_results = {
            "models": model_names,
            "dataset_name": dataset_name,
            "metrics": {},
            "rankings": {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query conditions
                conditions = ["model_name IN ({})".format(','.join(['?'] * len(model_names)))]
                params = model_names.copy()
                
                if dataset_name:
                    conditions.append("dataset_name = ?")
                    params.append(dataset_name)
                
                if time_window_hours:
                    cutoff_time = time.time() - (time_window_hours * 3600)
                    conditions.append("timestamp > ?")
                    params.append(cutoff_time)
                
                query = f"""
                    SELECT model_name, metric_name, AVG(metric_value) as avg_score
                    FROM evaluations 
                    WHERE {' AND '.join(conditions)}
                    GROUP BY model_name, metric_name
                """
                
                cursor = conn.execute(query, params)
                results = cursor.fetchall()
                
                # Organize results
                for model_name, metric_name, avg_score in results:
                    if metric_name not in comparison_results["metrics"]:
                        comparison_results["metrics"][metric_name] = {}
                    comparison_results["metrics"][metric_name][model_name] = avg_score
                
                # Calculate rankings
                for metric_name, model_scores in comparison_results["metrics"].items():
                    sorted_models = sorted(
                        model_scores.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    comparison_results["rankings"][metric_name] = [
                        {"model": model, "score": score, "rank": i + 1}
                        for i, (model, score) in enumerate(sorted_models)
                    ]
        
        except Exception as e:
            self.logger.error(f"Model comparison error: {e}")
        
        return comparison_results
    
    def get_performance_trends(
        self,
        model_name: str,
        metric_name: str,
        time_window_hours: float = 24.0
    ) -> Dict[str, Any]:
        """Performance trend'lerini al"""
        
        if not self.config.save_results:
            raise ValueError("Performance trends require save_results=True")
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, metric_value 
                    FROM evaluations 
                    WHERE model_name = ? AND metric_name = ? AND timestamp > ?
                    ORDER BY timestamp
                """, (model_name, metric_name, cutoff_time))
                
                results = cursor.fetchall()
                
                if not results:
                    return {"error": "No data found"}
                
                timestamps = [r[0] for r in results]
                values = [r[1] for r in results]
                
                # Calculate trend
                if len(values) > 1:
                    trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                    trend_direction = "improving" if trend_slope > 0 else "declining"
                else:
                    trend_slope = 0.0
                    trend_direction = "stable"
                
                return {
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "time_window_hours": time_window_hours,
                    "data_points": len(values),
                    "latest_score": values[-1],
                    "average_score": np.mean(values),
                    "trend_slope": trend_slope,
                    "trend_direction": trend_direction,
                    "timestamps": timestamps,
                    "values": values
                }
        
        except Exception as e:
            self.logger.error(f"Performance trends error: {e}")
            return {"error": str(e)}


class TrainingMonitor:
    """Training process monitor"""
    
    def __init__(self, model_name: str, log_interval: int = 100):
        self.model_name = model_name
        self.log_interval = log_interval
        
        # Metrics tracking
        self.training_metrics = defaultdict(list)
        self.validation_metrics = defaultdict(list)
        self.step_times = deque(maxlen=1000)
        
        # Current state
        self.current_step = 0
        self.current_epoch = 0
        self.start_time = time.time()
        
        self.logger = logging.getLogger(__name__)
    
    def log_step(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        learning_rate: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Training step'i logla"""
        self.current_step = step
        self.current_epoch = epoch
        
        # Record metrics
        self.training_metrics["loss"].append(train_loss)
        self.training_metrics["learning_rate"].append(learning_rate)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.training_metrics[key].append(value)
        
        # Step timing
        current_time = time.time()
        if len(self.step_times) > 0:
            step_duration = current_time - self.step_times[-1]
        else:
            step_duration = 0.0
        self.step_times.append(current_time)
        
        # Periodic logging
        if step % self.log_interval == 0:
            self._log_progress(step, epoch, train_loss, learning_rate, step_duration)
    
    def log_validation(
        self,
        epoch: int,
        val_loss: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Validation results'ları logla"""
        self.validation_metrics["loss"].append(val_loss)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.validation_metrics[key].append(value)
        
        self.logger.info(f"Validation Epoch {epoch} - Loss: {val_loss:.4f}")
    
    def _log_progress(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        learning_rate: float,
        step_duration: float
    ):
        """Progress'i logla"""
        elapsed_time = time.time() - self.start_time
        
        # Calculate averages
        recent_losses = self.training_metrics["loss"][-self.log_interval:]
        avg_loss = np.mean(recent_losses) if recent_losses else train_loss
        
        recent_times = list(self.step_times)[-self.log_interval:]
        if len(recent_times) > 1:
            avg_step_time = (recent_times[-1] - recent_times[0]) / (len(recent_times) - 1)
        else:
            avg_step_time = step_duration
        
        self.logger.info(
            f"Step {step} (Epoch {epoch}) - "
            f"Loss: {train_loss:.4f} (avg: {avg_loss:.4f}) - "
            f"LR: {learning_rate:.2e} - "
            f"Time: {avg_step_time:.2f}s/step - "
            f"Elapsed: {elapsed_time/60:.1f}min"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Training summary'si al"""
        total_time = time.time() - self.start_time
        
        summary = {
            "model_name": self.model_name,
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "total_time_minutes": total_time / 60,
            "steps_per_second": self.current_step / total_time if total_time > 0 else 0,
        }
        
        # Training metrics summary
        if self.training_metrics["loss"]:
            summary["training"] = {
                "final_loss": self.training_metrics["loss"][-1],
                "best_loss": min(self.training_metrics["loss"]),
                "loss_improvement": (
                    self.training_metrics["loss"][0] - self.training_metrics["loss"][-1]
                    if len(self.training_metrics["loss"]) > 1 else 0.0
                )
            }
        
        # Validation metrics summary
        if self.validation_metrics["loss"]:
            summary["validation"] = {
                "final_loss": self.validation_metrics["loss"][-1],
                "best_loss": min(self.validation_metrics["loss"]),
            }
        
        return summary
