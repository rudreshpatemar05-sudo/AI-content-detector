from __future__ import annotations

import json
import random
import re
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CALIBRATION_BINS, CALIBRATION_C, CALIBRATION_PRECISION_TARGET,
    CALIBRATOR_PATH, CHAR_MAX_FEATURES, CHAR_MIN_DF, CHAR_NGRAM_RANGE,
    EVASION_TEST_PATH, LOGISTIC_C_VALUES, LOGISTIC_MAX_ITER, MODEL_PATH,
    PROBABILITY_DISPLAY_MAX, PROBABILITY_DISPLAY_MIN, ROBUST_AUGMENT_FRACTION,
    ROBUST_AUG_CALIBRATOR_PATH, ROBUST_AUG_MODEL_PATH,
    ROBUST_NOAUG_CALIBRATOR_PATH, ROBUST_NOAUG_MODEL_PATH, RESULTS_DIR, SEED,
    STYLE_FEATURE_COUNT, TOP_FEATURE_COUNT, TRAIN_PATH, VAL_PATH,
    V1_CALIBRATOR_PATH, V1_MODEL_PATH, WORD_MAX_FEATURES, WORD_MIN_DF, WORD_NGRAM_RANGE,
)
from src.style_features import StyleFeatures
from src.utils import set_seed


def humanize(text: str) -> str:
    value = str(text)
    # Shorten stock constructions while preserving the original claim.
    shorten = [
        (r"\bdue to the fact that\b", "because"),
        (r"\bin order to\b", "to"),
        (r"\ba large number of\b", "many"),
        (r"\bthe majority of\b", "most"),
        (r"\bhas the ability to\b", "can"),
        (r"\bplays a crucial role in\b", "helps with"),
        (r"\bis widely regarded as\b", "is"),
        (r"\bit is important to\b", "remember to"),
        (r"\bhowever\b", "but"),
        (r"\bmoreover\b", "also"),
        (r"\badditionally\b", "also"),
        (r"\bin conclusion\b", "overall"),
    ]
    for pattern, replacement in shorten:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    contractions = [
        (r"\bit is\b", "it's"), (r"\bthey are\b", "they're"),
        (r"\bwe are\b", "we're"), (r"\bdo not\b", "don't"),
        (r"\bdoes not\b", "doesn't"), (r"\bcannot\b", "can't"),
        (r"\bwill not\b", "won't"), (r"\bhave not\b", "haven't"),
        (r"\bhas not\b", "hasn't"), (r"\bis not\b", "isn't"),
    ]
    for pattern, replacement in contractions:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def augment_ai_training(train: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ai = train[train["label"].astype(int) == 1]
    sample_count = int(round(len(ai) * ROBUST_AUGMENT_FRACTION))
    sample = ai.sample(n=sample_count, random_state=SEED).copy()
    sample["text"] = sample["text"].map(humanize)
    sample["group_id"] = sample["group_id"].astype(str) + "::augmented"
    sample["source"] = sample["source"].astype(str) + "::augmented"
    augmented = pd.concat([train, sample], ignore_index=True)
    return augmented.sample(frac=1, random_state=SEED).reset_index(drop=True), sample_count


def build_features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=WORD_NGRAM_RANGE, sublinear_tf=True, min_df=WORD_MIN_DF, max_features=WORD_MAX_FEATURES)),
        ("char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAM_RANGE, sublinear_tf=True, min_df=CHAR_MIN_DF, max_features=CHAR_MAX_FEATURES)),
        ("style", Pipeline([("features", StyleFeatures()), ("scale", StandardScaler(with_mean=False))])),
    ], n_jobs=1)


def build_classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(C=c_value, max_iter=LOGISTIC_MAX_ITER, class_weight="balanced", random_state=SEED)


def select_thresholds(y: np.ndarray, p: np.ndarray) -> dict[str, float | dict[str, float]]:
    human_threshold = 0.0
    for threshold in np.sort(np.unique(p[p <= 0.5]))[::-1]:
        chosen = p <= threshold
        if chosen.any() and float(np.mean(y[chosen] == 0)) >= CALIBRATION_PRECISION_TARGET:
            human_threshold = float(threshold)
            break
    ai_threshold = 1.0
    for threshold in np.sort(np.unique(p[p >= 0.5])):
        chosen = p >= threshold
        if chosen.any() and float(np.mean(y[chosen] == 1)) >= CALIBRATION_PRECISION_TARGET:
            ai_threshold = float(threshold)
            break
    human_mask, ai_mask = p <= human_threshold, p >= ai_threshold
    return {
        "human_threshold": human_threshold,
        "ai_threshold": ai_threshold,
        "validation_purity": {
            "human": float(np.mean(y[human_mask] == 0)) if human_mask.any() else 0.0,
            "ai": float(np.mean(y[ai_mask] == 1)) if ai_mask.any() else 0.0,
        },
        "uncertain_band_width": ai_threshold - human_threshold,
        "target_precision": CALIBRATION_PRECISION_TARGET,
    }


def save_reliability(y: np.ndarray, p: np.ndarray, name: str) -> None:
    observed, predicted = calibration_curve(y, p, n_bins=CALIBRATION_BINS, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="#777", label="Perfect calibration")
    ax.plot(predicted, observed, marker="o", label=name)
    ax.set(xlabel="Mean predicted AI probability", ylabel="Observed AI rate", title=f"Validation calibration: {name}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"reliability_{name}.png", dpi=180)
    plt.close(fig)


def save_word_features(features: FeatureUnion, classifier: LogisticRegression, path: Path) -> None:
    vectorizer = features.named_transformers["word"]
    names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0][:len(names)]
    path.write_text(json.dumps({
        "human": [{"word": str(names[i]), "coefficient": float(coefs[i])} for i in np.argsort(coefs)[:TOP_FEATURE_COUNT]],
        "ai": [{"word": str(names[i]), "coefficient": float(coefs[i])} for i in np.argsort(coefs)[-TOP_FEATURE_COUNT:][::-1]],
    }, indent=2), encoding="utf-8")


def fit_variant(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    name: str,
    model_path: Path,
    calibrator_path: Path,
    c_values: tuple[float, ...] = LOGISTIC_C_VALUES,
) -> tuple[dict[str, object], Path, Path]:
    x_train = train["text"].fillna("").astype(str).tolist()
    y_train = train["label"].astype(int).to_numpy()
    x_val = validation["text"].fillna("").astype(str).tolist()
    y_val = validation["label"].astype(int).to_numpy()
    features = build_features()
    print(f"[{name}] fit features train={len(x_train)} val={len(x_val)}", flush=True)
    xtr = features.fit_transform(x_train)
    xv = features.transform(x_val)
    tuning = []
    classifiers = {}
    for c_value in c_values:
        clf = build_classifier(float(c_value))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            clf.fit(xtr, y_train)
        if captured:
            print(f"[{name}] C={c_value} warnings={len(captured)}: {captured[0].message}", flush=True)
        val_probabilities = clf.predict_proba(xv)[:, 1]
        f1 = float(f1_score(y_val, val_probabilities >= 0.5, zero_division=0))
        tuning.append({
            "C": float(c_value),
            "validation_f1": f1,
            "validation_log_loss": float(log_loss(y_val, val_probabilities, labels=[0, 1])),
        })
        classifiers[float(c_value)] = clf
        print(f"[{name}] C={c_value:g} validation F1={f1:.6f}", flush=True)
    best = max(tuning, key=lambda row: (row["validation_f1"], -row["C"]))
    classifier = classifiers[best["C"]]
    pipeline = Pipeline([("features", features), ("clf", classifier)])
    raw = pipeline.predict_proba(x_val)[:, 1]
    calibrator = LogisticRegression(C=CALIBRATION_C, max_iter=LOGISTIC_MAX_ITER, random_state=SEED)
    calibrator.fit(raw.reshape(-1, 1), y_val)
    calibrated = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    thresholds = select_thresholds(y_val, calibrated)
    save_reliability(y_val, raw, f"{name}_before")
    save_reliability(y_val, calibrated, f"{name}_after")
    report = {
        "model": name,
        "best_C": float(best["C"]),
        "c_tuning": tuning,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "calibration": "Platt sigmoid fit to validation predictions",
        "validation": {
            "brier_before": float(brier_score_loss(y_val, raw)),
            "brier_after": float(brier_score_loss(y_val, calibrated)),
            "log_loss_before": float(log_loss(y_val, raw, labels=[0, 1])),
            "log_loss_after": float(log_loss(y_val, calibrated, labels=[0, 1])),
            "precision_after_at_0.5": float(precision_score(y_val, calibrated >= 0.5, zero_division=0)),
            "recall_after_at_0.5": float(recall_score(y_val, calibrated >= 0.5, zero_division=0)),
            "f1_after_at_0.5": float(f1_score(y_val, calibrated >= 0.5, zero_division=0)),
            "thresholds": thresholds,
        },
    }
    joblib.dump(pipeline, model_path, compress=3)
    joblib.dump(calibrator, calibrator_path, compress=3)
    save_word_features(features, classifier, RESULTS_DIR / f"top_features_{name}.json")
    return report, model_path, calibrator_path


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VAL_PATH)
    if EVASION_TEST_PATH.exists():
        evasion_texts = set(pd.read_csv(EVASION_TEST_PATH)["text"].dropna().astype(str))
        assert not evasion_texts.intersection(set(train.text.astype(str))), "Evasion text leaked into train."
        assert not evasion_texts.intersection(set(validation.text.astype(str))), "Evasion text leaked into validation."
    no_aug_report, _, _ = fit_variant(train, validation, "robust_noaug", ROBUST_NOAUG_MODEL_PATH, ROBUST_NOAUG_CALIBRATOR_PATH)
    augmented, augmented_count = augment_ai_training(train)
    if EVASION_TEST_PATH.exists():
        evasion_texts = set(pd.read_csv(EVASION_TEST_PATH)["text"].dropna().astype(str))
        assert not evasion_texts.intersection(set(augmented.text.astype(str))), "Evasion text leaked into augmented training data."
    aug_report, _, _ = fit_variant(augmented, validation, "robust_aug", ROBUST_AUG_MODEL_PATH, ROBUST_AUG_CALIBRATOR_PATH)

    # Record v1's validation calibration before/after with the same untouched validation data.
    v1 = joblib.load(V1_MODEL_PATH)
    v1_calibrator = joblib.load(V1_CALIBRATOR_PATH)
    x_val = validation.text.fillna("").astype(str).tolist()
    y_val = validation.label.astype(int).to_numpy()
    v1_raw = v1.predict_proba(x_val)[:, 1]
    v1_post = v1_calibrator.predict_proba(v1_raw.reshape(-1, 1))[:, 1]
    save_reliability(y_val, v1_raw, "v1_before")
    save_reliability(y_val, v1_post, "v1_after")
    reports = {
        "augmentation": {
            "fraction_of_ai_train_rows_augmented": ROBUST_AUGMENT_FRACTION,
            "added_rows": augmented_count,
            "rules": "shorten common constructions; replace formal connectives; contract common phrases; simplify/remove specified stock wording",
            "scope": "AI-labelled training rows only",
        },
        "v1_validation": {
            "brier_before": float(brier_score_loss(y_val, v1_raw)),
            "brier_after": float(brier_score_loss(y_val, v1_post)),
            "log_loss_before": float(log_loss(y_val, v1_raw, labels=[0, 1])),
            "log_loss_after": float(log_loss(y_val, v1_post, labels=[0, 1])),
        },
        "variants": [no_aug_report, aug_report],
    }
    (RESULTS_DIR / "robust_training.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
