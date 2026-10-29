from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    BOT_THRESHOLD, CALIBRATION_C, CALIBRATOR_PATH, CALIBRATION_PRECISION_TARGET,
    CHAR_MAX_FEATURES, CHAR_MIN_DF, CHAR_NGRAM_RANGE, EVASION_TEST_PATH,
    HELDOUT_CUSTOM_PATH, HELDOUT_WIKICSAI_PATH, LOGISTIC_C_VALUES,
    LOGISTIC_MAX_ITER, MODEL_PATH, PROBABILITY_DISPLAY_MAX, RAID_LOGISTIC_C_VALUES,
    PROBABILITY_DISPLAY_MIN, RESULTS_DIR, ROBUST_AUG_CALIBRATOR_PATH,
    ROBUST_AUGMENT_FRACTION, ROBUST_AUG_MODEL_PATH,
    ROBUST_NOAUG_CALIBRATOR_PATH, ROBUST_NOAUG_MODEL_PATH,
    RAID_AUG_CALIBRATOR_PATH, RAID_AUG_MODEL_PATH, RAID_NOAUG_CALIBRATOR_PATH,
    RAID_NOAUG_MODEL_PATH, RAID_SAMPLE_PATH, SEED, STYLE_FEATURE_COUNT,
    TEST_PATH, TRAIN_PATH, VAL_PATH, V1_CALIBRATOR_PATH, V1_METRICS_PATH,
    V1_MODEL_PATH, WORD_MAX_FEATURES, WORD_MIN_DF, WORD_NGRAM_RANGE,
)
from src.train_robust import select_thresholds


def score_model(model_path: Path, calibrator_path: Path, frame: pd.DataFrame) -> np.ndarray:
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    raw = model.predict_proba(frame["text"].fillna("").astype(str).tolist())[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def metric_row(model_name: str, split_name: str, frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, object]:
    labels = frame["label"].astype(int).to_numpy()
    predicted = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "model": model_name,
        "split": split_name,
        "rows": len(frame),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "brier": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    heldout = pd.read_csv(HELDOUT_WIKICSAI_PATH)
    raid_sample = pd.read_csv(RAID_SAMPLE_PATH) if RAID_SAMPLE_PATH.exists() else pd.DataFrame()
    raid_validation = raid_sample[raid_sample.raid_partition == "validation"].copy() if not raid_sample.empty else pd.DataFrame()
    raid_test = raid_sample[raid_sample.raid_partition == "test"].copy() if not raid_sample.empty else pd.DataFrame()
    evasion = pd.read_csv(EVASION_TEST_PATH)
    required = {"text", "label", "kind"}
    if not required.issubset(evasion.columns):
        raise ValueError(f"Evasion test must contain {sorted(required)}; found {list(evasion.columns)}")
    if not set(evasion.label.astype(int)).issubset({0, 1}):
        raise ValueError("Evasion labels must be 0 or 1.")
    evasion_texts = set(evasion.text.fillna("").astype(str))
    for path in (TRAIN_PATH, VAL_PATH):
        frame = pd.read_csv(path)
        assert not evasion_texts.intersection(set(frame.text.fillna("").astype(str))), f"Evasion data leaked into {path.name}."

    splits: list[tuple[str, pd.DataFrame]] = [("combined_test", test)]
    for dataset in ("hc3", "gpt_wiki"):
        subset = test[test.dataset == dataset].reset_index(drop=True)
        if not subset.empty:
            splits.append((f"{dataset}_test", subset))
    splits.append(("heldout_hc3_wiki_csai", heldout))
    if HELDOUT_CUSTOM_PATH.exists():
        splits.append(("heldout_custom", pd.read_csv(HELDOUT_CUSTOM_PATH)))
    if not raid_test.empty:
        splits.append(("raid_source_test", raid_test))
    splits.append(("evasion_overall", evasion))
    for kind, group in evasion.groupby("kind", sort=True):
        splits.append((f"evasion_{kind}", group.reset_index(drop=True)))

    models = [
        ("v1", V1_MODEL_PATH, V1_CALIBRATOR_PATH),
        ("robust_noaug", ROBUST_NOAUG_MODEL_PATH, ROBUST_NOAUG_CALIBRATOR_PATH),
        ("robust_aug", ROBUST_AUG_MODEL_PATH, ROBUST_AUG_CALIBRATOR_PATH),
    ]
    for name, model_path, calibrator_path in (
        ("raid_noaug", RAID_NOAUG_MODEL_PATH, RAID_NOAUG_CALIBRATOR_PATH),
        ("raid_aug", RAID_AUG_MODEL_PATH, RAID_AUG_CALIBRATOR_PATH),
    ):
        if model_path.exists() and calibrator_path.exists():
            models.append((name, model_path, calibrator_path))
    for name, model_path, calibrator_path in models:
        if not model_path.exists() or not calibrator_path.exists():
            raise FileNotFoundError(f"Missing {name} model/calibrator: {model_path}, {calibrator_path}")

    rows: list[dict[str, object]] = []
    detailed: dict[str, dict[str, dict[str, object]]] = {}
    probabilities_by_model: dict[str, dict[str, np.ndarray]] = {}
    validation_metrics: dict[str, dict[str, object]] = {}
    validation_calibrators: dict[str, LogisticRegression] = {}
    validation_thresholds: dict[str, dict[str, object]] = {}
    validation_frames = {"original_validation": validation}
    if not raid_validation.empty:
        validation_frames["raid_validation"] = raid_validation
    for model_name, model_path, calibrator_path in models:
        detailed[model_name] = {}
        probabilities_by_model[model_name] = {}
        candidate_model = joblib.load(model_path)
        combined_validation_text = pd.concat(
            [frame["text"].fillna("").astype(str) for frame in validation_frames.values()],
            ignore_index=True,
        ).tolist()
        combined_validation_labels = np.concatenate([
            frame["label"].astype(int).to_numpy() for frame in validation_frames.values()
        ])
        raw_combined_validation = candidate_model.predict_proba(combined_validation_text)[:, 1]
        evaluation_calibrator = LogisticRegression(C=CALIBRATION_C, max_iter=LOGISTIC_MAX_ITER)
        evaluation_calibrator.fit(raw_combined_validation.reshape(-1, 1), combined_validation_labels)
        validation_calibrators[model_name] = evaluation_calibrator
        calibrated_combined_validation = evaluation_calibrator.predict_proba(
            raw_combined_validation.reshape(-1, 1)
        )[:, 1]
        validation_thresholds[model_name] = select_thresholds(
            combined_validation_labels, calibrated_combined_validation
        )
        validation_offset = 0
        validation_results = {}
        for validation_name, validation_frame in validation_frames.items():
            validation_texts = validation_frame["text"].fillna("").astype(str).tolist()
            raw_probabilities = candidate_model.predict_proba(validation_texts)[:, 1]
            validation_probabilities = evaluation_calibrator.predict_proba(
                raw_probabilities.reshape(-1, 1)
            )[:, 1]
            validation_results[validation_name] = metric_row(
                model_name, validation_name, validation_frame, validation_probabilities
            )
        validation_metrics[model_name] = {
            **validation_results,
            "selection_f1": float(np.mean([result["f1"] for result in validation_results.values()])),
        }
        # Score each base set once; HC3/GPT subsets and per-kind evasion rows
        # reuse the corresponding probabilities without another vectorization pass.
        base_probabilities = {
            name: evaluation_calibrator.predict_proba(
                candidate_model.predict_proba(frame["text"].fillna("").astype(str).tolist())[:, 1].reshape(-1, 1)
            )[:, 1]
            for name, frame in (
                ("combined_test", test),
                ("heldout_hc3_wiki_csai", heldout),
                ("evasion_overall", evasion),
            )
        }
        if not raid_test.empty:
            raw_raid_test = candidate_model.predict_proba(raid_test["text"].fillna("").astype(str).tolist())[:, 1]
            base_probabilities["raid_source_test"] = evaluation_calibrator.predict_proba(
                raw_raid_test.reshape(-1, 1)
            )[:, 1]
        if HELDOUT_CUSTOM_PATH.exists():
            custom_frame = next(frame for name, frame in splits if name == "heldout_custom")
            raw_custom = candidate_model.predict_proba(custom_frame["text"].fillna("").astype(str).tolist())[:, 1]
            base_probabilities["heldout_custom"] = evaluation_calibrator.predict_proba(
                raw_custom.reshape(-1, 1)
            )[:, 1]
        for split_name, frame in splits:
            if split_name in base_probabilities:
                probabilities = base_probabilities[split_name]
            elif split_name in ("hc3_test", "gpt_wiki_test"):
                mask = test.dataset.eq(split_name.removesuffix("_test")).to_numpy()
                probabilities = base_probabilities["combined_test"][mask]
            elif split_name.startswith("evasion_"):
                kind = split_name.removeprefix("evasion_")
                mask = evasion.kind.eq(kind).to_numpy()
                probabilities = base_probabilities["evasion_overall"][mask]
            else:
                raise ValueError(f"No cached probabilities for evaluation split {split_name!r}")
            result = metric_row(model_name, split_name, frame, probabilities)
            detailed[model_name][split_name] = result
            probabilities_by_model[model_name][split_name] = probabilities
            rows.append({key: result[key] for key in ("model", "split", "rows", "accuracy", "precision", "recall", "f1", "roc_auc", "brier")})
            print(f"{model_name}/{split_name}: {json.dumps(result, sort_keys=True)}", flush=True)

    # Original and RAID validation scores determine selection; test scores are report-only.
    selected = max(validation_metrics, key=lambda name: float(validation_metrics[name]["selection_f1"]))
    _, selected_model_path, selected_calibrator_path = next(
        item for item in models if item[0] == selected
    )
    shutil.copy2(selected_model_path, MODEL_PATH)
    joblib.dump(validation_calibrators[selected], CALIBRATOR_PATH, compress=3)

    # Adopt thresholds from the shared validation calibration protocol.
    training_report = json.loads((RESULTS_DIR / "robust_training.json").read_text(encoding="utf-8"))
    chosen_thresholds = validation_thresholds[selected]
    if selected == "v1":
        selected_training = json.loads((RESULTS_DIR / "tuning_results.json").read_text(encoding="utf-8"))
        selected_best_c = selected_training["best_C"]
        selected_train_rows = selected_training["train_rows"]
    elif selected.startswith("raid_"):
        fitted_pipeline = joblib.load(selected_model_path)
        selected_best_c = float(fitted_pipeline.named_steps["clf"].C)
        raid_train_rows = int((raid_sample.raid_partition == "train").sum()) if not raid_sample.empty else 0
        selected_train_rows = len(pd.read_csv(TRAIN_PATH)) + raid_train_rows
        selected_training = {
            "model": selected,
            "best_C": selected_best_c,
            "train_rows": selected_train_rows,
            "validation_selection": validation_metrics[selected],
        }
    else:
        chosen_report = next(item for item in training_report["variants"] if item["model"] == selected)
        selected_training = chosen_report
        selected_best_c = chosen_report["best_C"]
        selected_train_rows = chosen_report["train_rows"]
    config_path = ROOT / "config.py"
    config = config_path.read_text(encoding="utf-8")
    config = re.sub(r"(?m)^HUMAN_THRESHOLD\s*=.*$", f"HUMAN_THRESHOLD = {float(chosen_thresholds['human_threshold']):.12g}", config)
    config = re.sub(r"(?m)^BOT_THRESHOLD\s*=.*$", f"BOT_THRESHOLD = {float(chosen_thresholds['ai_threshold']):.12g}", config)
    config_path.write_text(config, encoding="utf-8")

    selected_config = {
        "selected_model": selected,
        "selection_basis": "highest mean calibrated F1 at probability 0.5 across original and RAID validation sets",
        "seed": SEED,
        "label_mapping": {"0": "human", "1": "AI"},
        "artifacts": {
            "model": str(MODEL_PATH.relative_to(ROOT)),
            "calibrator": str(CALIBRATOR_PATH.relative_to(ROOT)),
        },
        "training": {
            "model_type": "TF-IDF FeatureUnion + LogisticRegression",
            "train_rows": int(selected_train_rows),
            "validation_rows": int(sum(len(frame) for frame in validation_frames.values())),
            "best_C": float(selected_best_c),
            "searched_C_values": list(RAID_LOGISTIC_C_VALUES if selected.startswith("raid_") else LOGISTIC_C_VALUES),
            "class_weight": "balanced",
            "max_iter": LOGISTIC_MAX_ITER,
            "word_features": {
                "ngram_range": list(WORD_NGRAM_RANGE),
                "min_df": WORD_MIN_DF,
                "max_features": WORD_MAX_FEATURES,
            },
            "character_features": {
                "ngram_range": list(CHAR_NGRAM_RANGE),
                "min_df": CHAR_MIN_DF,
                "max_features": CHAR_MAX_FEATURES,
            },
            "style_feature_count": STYLE_FEATURE_COUNT if selected != "v1" else 0,
            "ai_augmentation_fraction": ROBUST_AUGMENT_FRACTION if selected in ("robust_aug", "raid_aug") else 0.0,
        },
        "calibration": {
            "method": "Platt scaling with LogisticRegression",
            "C": CALIBRATION_C,
        },
        "thresholds": chosen_thresholds,
        "validation_metrics": validation_metrics[selected],
        "scikit_learn_version": sklearn.__version__,
    }
    (RESULTS_DIR / "selected_training_config.json").write_text(
        json.dumps(selected_config, indent=2), encoding="utf-8"
    )

    error_rows: list[dict[str, object]] = []
    for split_name, frame in splits:
        probabilities = probabilities_by_model[selected][split_name]
        labels = frame["label"].astype(int).to_numpy()
        predictions = (probabilities >= 0.5).astype(int)
        for index in np.flatnonzero(labels != predictions):
            row = frame.iloc[int(index)]
            error_rows.append({
                "model": selected,
                "split": split_name,
                "error_type": "false_positive" if predictions[index] == 1 else "false_negative",
                "actual_label": int(labels[index]),
                "predicted_label": int(predictions[index]),
                "ai_probability": float(probabilities[index]),
                "dataset": str(row.get("dataset", "")),
                "source": str(row.get("source", "")),
                "word_count": len(str(row.get("text", "")).split()),
                "text": str(row.get("text", "")),
            })
    pd.DataFrame(error_rows).to_csv(RESULTS_DIR / "error_analysis.csv", index=False)

    combined_acc = detailed[selected]["combined_test"]["accuracy"]
    held_acc = detailed[selected]["heldout_hc3_wiki_csai"]["accuracy"]
    evasion_acc = detailed[selected]["evasion_overall"]["accuracy"]
    summary = {
        "selected_model": selected,
        "selection_basis": "highest mean calibrated F1 at probability 0.5 across original and RAID validation sets; test splits are evaluation-only",
        "validation_selection": validation_metrics,
        "accuracy_drop_combined_to_wikicsai": combined_acc - held_acc,
        "accuracy_drop_combined_to_evasion": combined_acc - evasion_acc,
        "results": detailed,
    }
    (RESULTS_DIR / "robust_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "comparison.csv", index=False)
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    table = "| Model | Split | Rows | Accuracy | Precision | Recall | F1 | ROC-AUC |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for model_name, _, _ in models:
        for split_name in ("combined_test", "hc3_test", "gpt_wiki_test", "heldout_hc3_wiki_csai", "raid_source_test", "evasion_overall", "evasion_ai_original", "evasion_ai_humanized"):
            result = detailed[model_name].get(split_name)
            if result is None:
                continue
            auc = "—" if result["roc_auc"] is None else f"{float(result['roc_auc']):.4f}"
            table += f"| {model_name} | {split_name} | {result['rows']} | {result['accuracy']:.4f} | {result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} | {auc} |\n"
    table_pattern = re.compile(
        r"(?m)^\| Model \| Split \| Rows \| Accuracy \| Precision \| Recall \| F1 \| ROC-AUC \|\r?\n(?:^\|.*\r?\n?)+"
    )
    readme, table_count = table_pattern.subn(table.rstrip() + "\n", readme, count=1)
    if table_count != 1:
        raise ValueError("Could not locate the results table in README.md; evaluation did not update documentation.")

    selected_validation = validation_metrics[selected]
    selected_test = detailed[selected]["combined_test"]
    selected_heldout = detailed[selected]["heldout_hc3_wiki_csai"]
    selected_evasion = detailed[selected]["evasion_overall"]
    purity = chosen_thresholds.get("validation_purity", {})
    summary = (
        f"The app model is `{selected}`, selected by highest mean calibrated F1 at 0.5 across "
        f"original and RAID validation sets ({float(selected_validation['selection_f1']):.4f}). "
        f"On the untouched combined test set it achieved "
        f"{float(selected_test['accuracy']):.4f} accuracy and {float(selected_test['f1']):.4f} F1; "
        f"on held-out HC3 `wiki_csai` it achieved {float(selected_heldout['accuracy']):.4f} accuracy "
        f"and {float(selected_heldout['f1']):.4f} F1. The held-out HC3 corpus is related to training."
    )
    if "raid_source_test" in detailed[selected]:
        raid_test_result = detailed[selected]["raid_source_test"]
        summary += (
            f" On the source-held-out RAID sample it achieved {float(raid_test_result['accuracy']):.4f} "
            f"accuracy and {float(raid_test_result['f1']):.4f} F1 ({raid_test_result['rows']} rows)."
        )
    threshold_summary = (
        f"Validation thresholds are Likely human ≤{float(chosen_thresholds['human_threshold']):.2%} "
        f"AI probability and Likely AI ≥{float(chosen_thresholds['ai_threshold']):.2%}. "
        f"Validation human purity was {float(purity.get('human', 0.0)):.2%} and AI precision was "
        f"{float(purity.get('ai', 0.0)):.2%}. Per-model calibration and thresholds use combined validation predictions."
    )
    evasion_summary = (
        f"The selected model scored {selected_evasion['accuracy']:.1%} on the supplied evasion set "
        f"({selected_evasion['rows']} examples). This sample is too small to estimate general evasion "
        "performance; paraphrased AI text may still be missed."
    )
    report_match = re.search(r"(?m)^The (?:selected|app) model is ", readme)
    if report_match is None:
        raise ValueError("Could not locate the selected-model summary in README.md.")
    report_start = report_match.start()
    report_end = readme.index("\n\nConfusion matrices", report_start)
    readme = readme[:report_start] + summary + "\n\n" + threshold_summary + readme[report_end:]
    evasion_match = re.search(r"(?m)^The (?:v1|selected) model scored ", readme)
    if evasion_match is None:
        raise ValueError("Could not locate the evasion summary in README.md.")
    evasion_start = evasion_match.start()
    evasion_end = readme.index("\n\n## Limitations", evasion_start)
    readme = readme[:evasion_start] + evasion_summary + readme[evasion_end:]
    readme_path.write_text(readme, encoding="utf-8")

    for split_name, frame in splits:
        result = detailed[selected][split_name]
        cm = np.asarray(result["confusion_matrix"])
        (RESULTS_DIR / f"{split_name}_confusion_matrix.csv").write_text(
            "actual_human,predicted_human,predicted_ai\n" +
            f"human,{cm[0,0]},{cm[0,1]}\nAI,{cm[1,0]},{cm[1,1]}\n", encoding="utf-8")
    plot_splits = ["combined_test", "hc3_test", "gpt_wiki_test", "heldout_hc3_wiki_csai", "evasion_overall"]
    if "raid_source_test" in detailed[selected]:
        plot_splits.append("raid_source_test")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_splits))
    width = min(0.25, 0.8 / len(models))
    for index, (model_name, _, _) in enumerate(models):
        values = [detailed[model_name][name]["accuracy"] for name in plot_splits]
        ax.bar(x + (index - (len(models) - 1) / 2) * width, values, width, label=model_name)
    ax.set_xticks(x, plot_splits, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "comparison.png", dpi=180)
    plt.close(fig)

    for split_name in plot_splits:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = np.asarray(detailed[selected][split_name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Human", "AI"], yticklabels=["Human", "AI"], xlabel="Predicted", ylabel="Actual", title=f"{selected}: {split_name}")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{split_name}_confusion_matrix.png", dpi=180)
        plt.close(fig)
    print(f"Selected final model: {selected}", flush=True)
    print(f"Selected held-out F1: {detailed[selected]['heldout_hc3_wiki_csai']['f1']:.6f}", flush=True)
    print(f"Selected evasion accuracy drop: {detailed[selected]['combined_test']['accuracy'] - detailed[selected]['evasion_overall']['accuracy']:.6f}", flush=True)


if __name__ == "__main__":
    main()
