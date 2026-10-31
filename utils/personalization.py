"""
personalization.py
==================
Confidence-Aware Calibration, Personal Vocabulary, and User Feedback Learning.

Features:
1. Top-3 Candidate Ranking & Dynamic Confidence Gating.
2. Personal Vocabulary & Custom Dictionary:
   - Stores user-defined names, acronyms, and custom workplace terminology.
   - Persists to 'data/personal_vocabulary.json'.
3. User Correction & Adaptive Calibration Engine:
   - Logs user corrections (when user overrides a prediction or edits transcript).
   - Computes local confusion priors to adaptively calibrate inference probabilities.
   - Persists to 'data/user_corrections.json'.
4. Transcript History Manager with Undo / Redo support.
"""

from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
import time
import numpy as np


@dataclass
class CandidatePrediction:
    """A single candidate prediction with label and confidence."""
    label: str
    confidence: float
    margin: float = 0.0


@dataclass
class ConfidenceReport:
    """Evaluates whether a prediction meets confidence thresholds."""
    top1: CandidatePrediction
    top3: List[CandidatePrediction]
    is_confident: bool
    is_uncertain: bool
    requires_user_clarification: bool
    suggested_alternatives: List[str]


class PersonalVocabularyManager:
    """Manages user-added custom words, names, and domain phrases."""

    def __init__(self, storage_path: str = "data/personal_vocabulary.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.custom_words: List[str] = []
        self.custom_phrases: List[str] = []
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.custom_words = data.get("words", [])
                    self.custom_phrases = data.get("phrases", [])
            except Exception:
                self.custom_words = []
                self.custom_phrases = []

    def save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "words": sorted(list(set(self.custom_words))),
                    "phrases": sorted(list(set(self.custom_phrases))),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=4)
        except Exception as e:
            print(f"[!] Warning: Could not save personal vocabulary: {e}")

    def add_custom_word(self, word: str) -> bool:
        clean = word.strip().upper()
        if clean and clean not in self.custom_words:
            self.custom_words.append(clean)
            self.save()
            return True
        return False

    def remove_custom_word(self, word: str) -> bool:
        clean = word.strip().upper()
        if clean in self.custom_words:
            self.custom_words.remove(clean)
            self.save()
            return True
        return False

    def get_all_vocabulary(self) -> List[str]:
        return sorted(list(set(self.custom_words)))

    def get_all_custom_words(self) -> List[str]:
        return sorted(list(set(self.custom_words)))


class UserCorrectionTracker:
    """
    Logs user overrides and computes calibration weight priors to
    adapt future predictions to the specific user's signing style.
    """

    def __init__(self, storage_path: str = "data/user_corrections.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.corrections_log: List[Dict[str, Any]] = []
        self.confusion_matrix: Dict[str, Dict[str, int]] = {}
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.corrections_log = data.get("log", [])
                    self.confusion_matrix = data.get("confusion", {})
            except Exception:
                self.corrections_log = []
                self.confusion_matrix = {}

    def save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "log": self.corrections_log[-200:],  # Keep recent 200
                    "confusion": self.confusion_matrix,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=4)
        except Exception as e:
            print(f"[!] Warning: Could not save user corrections: {e}")

    def record_correction(
        self,
        predicted_sign: str,
        corrected_sign: str,
        context: str = "MANUAL_OVERRIDE"
    ):
        """Records an event where user corrected predicted_sign to corrected_sign."""
        pred = predicted_sign.strip().upper()
        corr = corrected_sign.strip().upper()
        if not pred or not corr or pred == corr:
            return

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted": pred,
            "corrected": corr,
            "context": context
        }
        self.corrections_log.append(entry)

        if pred not in self.confusion_matrix:
            self.confusion_matrix[pred] = {}
        self.confusion_matrix[pred][corr] = self.confusion_matrix[pred].get(corr, 0) + 1

        self.save()

    def get_calibration_boost(self, predicted_class: str, candidate_class: str) -> float:
        """
        Returns a calibrated probability boost multiplier [1.0 to 1.3]
        if the user historically corrects predicted_class to candidate_class.
        """
        pred = predicted_class.strip().upper()
        cand = candidate_class.strip().upper()
        if pred in self.confusion_matrix and cand in self.confusion_matrix[pred]:
            count = self.confusion_matrix[pred][cand]
            # Logarithmic bounded boost up to 30%
            return 1.0 + min(0.30, 0.08 * np.log1p(count))
        return 1.0


def evaluate_confidence(
    top_candidates: List[Tuple[str, float]],
    confidence_threshold: float = 0.45,
    margin_threshold: float = 0.04
) -> ConfidenceReport:
    """
    Evaluates top prediction candidates against confidence and margin thresholds.
    """
    if not top_candidates:
        dummy = CandidatePrediction("Uncertain Gesture", 0.0)
        return ConfidenceReport(dummy, [dummy], False, True, True, [])

    top1 = CandidatePrediction(top_candidates[0][0], top_candidates[0][1])
    top3 = [CandidatePrediction(c[0], c[1]) for c in top_candidates[:3]]

    margin = 0.0
    if len(top3) > 1:
        margin = top3[0].confidence - top3[1].confidence
        top1.margin = margin

    is_confident = (top1.confidence >= confidence_threshold) and (margin >= margin_threshold)
    is_uncertain = not is_confident
    requires_user_clarification = is_uncertain and (top1.confidence >= 0.25)

    alternatives = [c.label for c in top3 if c.label != top1.label]

    return ConfidenceReport(
        top1=top1,
        top3=top3,
        is_confident=is_confident,
        is_uncertain=is_uncertain,
        requires_user_clarification=requires_user_clarification,
        suggested_alternatives=alternatives
    )
