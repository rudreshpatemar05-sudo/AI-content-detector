from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    AI_LABEL,
    BOT_THRESHOLD,
    CALIBRATOR_PATH,
    HUMAN_LABEL,
    HUMAN_THRESHOLD,
    MIN_WORDS_FOR_DETECTION,
    MODEL_PATH,
    HIGHLIGHT_WINDOW_SENTENCES,
    HIGHLIGHT_WINDOW_STEP,
    HIGHLIGHT_WINDOW_WORDS,
    PROBABILITY_DISPLAY_MAX,
    PROBABILITY_DISPLAY_MIN,
)
from src.utils import logger


def _missing_model_error(path: Path) -> FileNotFoundError:
    return FileNotFoundError(f"TF-IDF model not found at {path}. Run python src/train_tfidf.py first.")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise _missing_model_error(MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    classifier = model.named_steps.get("clf")
    if classifier is None or list(classifier.classes_) != [HUMAN_LABEL, AI_LABEL]:
        raise ValueError("The saved TF-IDF model class order is not [0=human, 1=AI]. Retrain with train_tfidf.py.")
    return model


@lru_cache(maxsize=1)
def load_calibrator():
    if not CALIBRATOR_PATH.exists():
        raise _missing_model_error(CALIBRATOR_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    if list(calibrator.classes_) != [HUMAN_LABEL, AI_LABEL]:
        raise ValueError("The saved probability calibrator class order is not [0=human, 1=AI]. Retrain with train_tfidf.py.")
    return calibrator


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", str(text).strip()) if part.strip()]


def _ai_probabilities(texts: list[str]) -> np.ndarray:
    model = load_model()
    raw_ai_probabilities = model.predict_proba(texts)[:, 1]
    calibrator = load_calibrator()
    calibrated = calibrator.predict_proba(raw_ai_probabilities.reshape(-1, 1))[:, 1]
    return calibrated


def _label(probability: float) -> str:
    if probability >= BOT_THRESHOLD:
        return "likely_ai"
    if probability <= HUMAN_THRESHOLD:
        return "likely_human"
    return "uncertain"


def predict(text: str) -> dict[str, float | int | str]:
    """Score the full passage as one input and return calibrated AI likelihood."""
    cleaned = _clean(text)
    word_count = len(cleaned.split())
    if word_count < MIN_WORDS_FOR_DETECTION:
        logger.warning("Text is shorter than %d words; detection may be unreliable.", MIN_WORDS_FOR_DETECTION)
    probability = float(_ai_probabilities([cleaned])[0])
    displayed_probability = float(np.clip(probability, PROBABILITY_DISPLAY_MIN, PROBABILITY_DISPLAY_MAX))
    return {"ai_probability": displayed_probability, "label": _label(probability), "word_count": word_count}


def _sentence_groups(sentences: list[str]) -> list[tuple[int, int, list[str]]]:
    groups: list[tuple[int, int, list[str]]] = []
    seen: set[tuple[int, int]] = set()
    start = 0
    while start < len(sentences):
        end = min(start + HIGHLIGHT_WINDOW_SENTENCES, len(sentences))
        while end < len(sentences) and sum(len(s.split()) for s in sentences[start:end]) < HIGHLIGHT_WINDOW_WORDS:
            end += 1
        while start > 0 and sum(len(s.split()) for s in sentences[start:end]) < HIGHLIGHT_WINDOW_WORDS:
            start -= 1
        key = (start, end)
        if key not in seen:
            groups.append((start, end, sentences[start:end]))
            seen.add(key)
        if end >= len(sentences):
            break
        start = max(start + 1, start + HIGHLIGHT_WINDOW_STEP)
    return groups


def predict_sentences(text: str) -> list[dict[str, float | int | str]]:
    """Return rough sentence hints using neighboring sentences as context."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(text)) if part.strip()]
    if not sentences:
        return []
    groups = _sentence_groups(sentences)
    group_texts = [" ".join(group[2]) for group in groups]
    probabilities = _ai_probabilities(group_texts)
    results: list[dict[str, float | int | str]] = []
    for (start, end, group), probability_value in zip(groups, probabilities):
        probability = float(probability_value)
        results.append({
            "sentence": " ".join(group),
            "start_sentence": start + 1,
            "end_sentence": end,
            "word_count": sum(len(sentence.split()) for sentence in group),
            "ai_probability": float(np.clip(probability, PROBABILITY_DISPLAY_MIN, PROBABILITY_DISPLAY_MAX)),
            "label": _label(probability),
            "hint": "Rough hint from an overlapping passage window; noisier than the overall score",
        })
    return results


if __name__ == "__main__":
    sample = (
        "This report explains how regular exercise supports heart health and general wellbeing. "
        "It also gives practical steps for maintaining a healthy lifestyle over time."
    )
    print(predict(sample))
