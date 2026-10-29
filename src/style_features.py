from __future__ import annotations

import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


_WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_CONTRACTION_RE = re.compile(r"\b\w+(?:['’][a-z]+)\b", re.IGNORECASE)
_FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}


def extract_style_features(text: str) -> list[float]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    words = _WORD_RE.findall(normalized)
    lower_words = [word.lower() for word in words]
    sentences = [sentence for sentence in _SENTENCE_RE.split(normalized) if sentence.strip()]
    lengths = [len(_WORD_RE.findall(sentence)) for sentence in sentences]
    avg_sentence_length = float(np.mean(lengths)) if lengths else 0.0
    sentence_variance = float(np.var(lengths)) if lengths else 0.0
    type_token_ratio = len(set(lower_words)) / len(lower_words) if lower_words else 0.0
    punctuation_rate = sum(char in ".,;:!?—–-()[]{}\"'" for char in normalized) / max(len(normalized), 1)
    contraction_rate = len(_CONTRACTION_RE.findall(normalized)) / max(len(words), 1)
    average_word_length = float(np.mean([len(word) for word in words])) if words else 0.0
    first_person_rate = sum(word in _FIRST_PERSON for word in lower_words) / max(len(words), 1)
    return [
        avg_sentence_length,
        sentence_variance,
        type_token_ratio,
        punctuation_rate,
        contraction_rate,
        average_word_length,
        first_person_rate,
    ]


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Cheap, interpretable seven-column stylometric transformer for text rows."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return csr_matrix(np.asarray([extract_style_features(text) for text in X], dtype=np.float64))
