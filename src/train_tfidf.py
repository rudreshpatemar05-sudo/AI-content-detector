from __future__ import annotations

import json
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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CALIBRATION_BINS,
    CALIBRATION_C,
    CALIBRATOR_PATH,
    CHAR_MAX_FEATURES,
    CHAR_MIN_DF,
    CHAR_NGRAM_RANGE,
    HUMAN_THRESHOLD,
    LOGISTIC_C_VALUES,
    LOGISTIC_MAX_ITER,
    MODEL_PATH,
    RELIABILITY_PLOT_PATH,
    RESULTS_DIR,
    SEED,
    TARGET_AI_PRECISION,
    TARGET_HUMAN_PURITY,
    TOP_FEATURES_PATH,
    TOP_FEATURE_COUNT,
    TRAIN_PATH,
    TUNING_RESULTS_PATH,
    VAL_PATH,
    WORD_MAX_FEATURES,
    WORD_MIN_DF,
    WORD_NGRAM_RANGE,
    BOT_THRESHOLD,
)
from src.utils import logger, set_seed


def build_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=WORD_NGRAM_RANGE,
                    sublinear_tf=True,
                    min_df=WORD_MIN_DF,
                    max_features=WORD_MAX_FEATURES,
                ),
            ),
            (
                "char_wb",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=CHAR_NGRAM_RANGE,
                    sublinear_tf=True,
                    min_df=CHAR_MIN_DF,
                    max_features=CHAR_MAX_FEATURES,
                ),
            ),
        ],
        n_jobs=1,
    )


def build_classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        max_iter=LOGISTIC_MAX_ITER,
        class_weight="balanced",
        random_state=SEED,
    )


def select_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | dict[str, float]]:
    human_threshold = float(HUMAN_THRESHOLD)
    bot_threshold = float(BOT_THRESHOLD)

    human_candidates = np.unique(probabilities[probabilities <= 0.5])
    human_candidates = np.sort(human_candidates)[::-1]
    for threshold in human_candidates:
        selected = probabilities <= threshold
        if selected.any() and float(np.mean(labels[selected] == 0)) >= TARGET_HUMAN_PURITY:
            human_threshold = float(threshold)
            break

    ai_candidates = np.unique(probabilities[probabilities >= 0.5])
    ai_candidates = np.sort(ai_candidates)
    for threshold in ai_candidates:
        selected = probabilities >= threshold
        if selected.any() and float(np.mean(labels[selected] == 1)) >= TARGET_AI_PRECISION:
            bot_threshold = float(threshold)
            break

    human_selected = probabilities <= human_threshold
    ai_selected = probabilities >= bot_threshold
    human_purity = float(np.mean(labels[human_selected] == 0)) if human_selected.any() else 0.0
    ai_precision = float(np.mean(labels[ai_selected] == 1)) if ai_selected.any() else 0.0
    return {
        "human_threshold": human_threshold,
        "ai_threshold": bot_threshold,
        "validation_purity": {"human": human_purity, "ai": ai_precision},
    }


def save_top_word_features(features: FeatureUnion, classifier: LogisticRegression) -> None:
    # FeatureUnion exposes fitted transformers via named_transformers in
    # supported sklearn versions; the trailing-underscore spelling is not
    # available in all versions.
    word_vectorizer = features.named_transformers["word"]
    word_names = word_vectorizer.get_feature_names_out()
    word_coefficients = classifier.coef_[0][: len(word_names)]
    human_indices = np.argsort(word_coefficients)[:TOP_FEATURE_COUNT]
    ai_indices = np.argsort(word_coefficients)[-TOP_FEATURE_COUNT:][::-1]
    result = {
        "human": [
            {"word": str(word_names[index]), "coefficient": float(word_coefficients[index])}
            for index in human_indices
        ],
        "ai": [
            {"word": str(word_names[index]), "coefficient": float(word_coefficients[index])}
            for index in ai_indices
        ],
    }
    TOP_FEATURES_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Top human-associated words:")
    for item in result["human"]:
        print(f"  {item['word']}: {item['coefficient']:.4f}")
    print("Top AI-associated words:")
    for item in result["ai"]:
        print(f"  {item['word']}: {item['coefficient']:.4f}")


def save_reliability_plot(labels: np.ndarray, probabilities: np.ndarray) -> None:
    observed, predicted = calibration_curve(labels, probabilities, n_bins=CALIBRATION_BINS, strategy="uniform")
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect calibration")
    axis.plot(predicted, observed, marker="o", color="#176b5b", label="Platt calibrated")
    axis.set(xlabel="Mean predicted AI probability", ylabel="Observed AI rate", title="Validation reliability")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(RELIABILITY_PLOT_PATH, dpi=180)
    plt.close(figure)


def save_config_thresholds(thresholds: dict[str, object]) -> None:
    config_path = ROOT / "config.py"
    config_text = config_path.read_text(encoding="utf-8")
    config_text = re.sub(r"(?m)^HUMAN_THRESHOLD\s*=.*$", f"HUMAN_THRESHOLD = {float(thresholds['human_threshold']):.12g}", config_text)
    config_text = re.sub(r"(?m)^BOT_THRESHOLD\s*=.*$", f"BOT_THRESHOLD = {float(thresholds['ai_threshold']):.12g}", config_text)
    config_path.write_text(config_text, encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VAL_PATH)
    for name, frame in (("train", train), ("validation", validation)):
        if not {"text", "label", "dataset", "group_id"}.issubset(frame.columns):
            raise ValueError(f"{name} CSV is missing expected columns: {list(frame.columns)!r}")
    x_train = train["text"].fillna("").astype(str).tolist()
    y_train = train["label"].astype(int).to_numpy()
    x_val = validation["text"].fillna("").astype(str).tolist()
    y_val = validation["label"].astype(int).to_numpy()
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
        raise ValueError("Train and validation must both contain labels 0 (human) and 1 (AI).")

    logger.info("Fitting word and character features on %d training rows.", len(x_train))
    features = build_features()
    x_train_features = features.fit_transform(x_train)
    x_val_features = features.transform(x_val)
    logger.info("Sparse feature matrix: train=%s, validation=%s", x_train_features.shape, x_val_features.shape)

    tuning: list[dict[str, float]] = []
    classifiers: dict[float, LogisticRegression] = {}
    for c_value in LOGISTIC_C_VALUES:
        classifier = build_classifier(float(c_value))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            classifier.fit(x_train_features, y_train)
        if captured:
            logger.warning("C=%s emitted %d warning(s); first: %s", c_value, len(captured), captured[0].message)
        classes = list(classifier.classes_)
        if classes != [0, 1]:
            raise ValueError(f"Unexpected class order for C={c_value}: {classes!r}; expected [0, 1].")
        predictions = classifier.predict(x_val_features)
        val_f1 = float(f1_score(y_val, predictions, pos_label=1, zero_division=0))
        tuning.append({"C": float(c_value), "validation_f1": val_f1})
        classifiers[float(c_value)] = classifier
        print(f"C={c_value:g} validation F1={val_f1:.6f}")

    best_result = max(tuning, key=lambda result: (result["validation_f1"], -result["C"]))
    best_c = best_result["C"]
    best_classifier = classifiers[best_c]
    pipeline = Pipeline([("features", features), ("clf", best_classifier)])
    if list(pipeline.named_steps["clf"].classes_) != [0, 1]:
        raise ValueError(f"Unexpected class order in final classifier: {pipeline.named_steps['clf'].classes_!r}")

    raw_val_probabilities = pipeline.predict_proba(x_val)[:, 1]
    calibrator = LogisticRegression(C=CALIBRATION_C, max_iter=LOGISTIC_MAX_ITER, random_state=SEED)
    calibrator.fit(raw_val_probabilities.reshape(-1, 1), y_val)
    if list(calibrator.classes_) != [0, 1]:
        raise ValueError(f"Unexpected calibrator class order: {calibrator.classes_!r}")
    calibrated_val = calibrator.predict_proba(raw_val_probabilities.reshape(-1, 1))[:, 1]

    thresholds = select_thresholds(y_val, calibrated_val)
    save_config_thresholds(thresholds)
    val_predictions = (calibrated_val >= 0.5).astype(int)
    val_summary = {
        "rows": len(y_val),
        "precision": float(precision_score(y_val, val_predictions, zero_division=0)),
        "recall": float(recall_score(y_val, val_predictions, zero_division=0)),
        "f1": float(f1_score(y_val, val_predictions, zero_division=0)),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    joblib.dump(calibrator, CALIBRATOR_PATH, compress=3)
    save_top_word_features(features, best_classifier)
    save_reliability_plot(y_val, calibrated_val)
    report = {
        "best_C": best_c,
        "c_tuning": tuning,
        "calibration": "Platt sigmoid fit to held-out validation predictions",
        "validation_calibrated_metrics_at_0.5": val_summary,
        "thresholds": thresholds,
        "pipeline_classes": [int(value) for value in pipeline.classes_],
        "calibrator_classes": [int(value) for value in calibrator.classes_],
        "train_rows": len(y_train),
        "validation_rows": len(y_val),
    }
    TUNING_RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected C={best_c:g}; verified classes_={pipeline.named_steps['clf'].classes_.tolist()}")
    print(f"Human threshold={thresholds['human_threshold']:.6f}; AI threshold={thresholds['ai_threshold']:.6f}")
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved calibrator: {CALIBRATOR_PATH}")
    print(f"Saved calibration plot: {RELIABILITY_PLOT_PATH}")
    logger.info("Training complete. Results: %s", TUNING_RESULTS_PATH)


if __name__ == "__main__":
    main()
