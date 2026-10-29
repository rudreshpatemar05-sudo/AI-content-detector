from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    RAID_AUG_CALIBRATOR_PATH,
    RAID_AUG_MODEL_PATH,
    RAID_NOAUG_CALIBRATOR_PATH,
    RAID_NOAUG_MODEL_PATH,
    RAID_LOGISTIC_C_VALUES,
    RAID_SAMPLE_PATH,
    RESULTS_DIR,
    SEED,
    TRAIN_PATH,
    VAL_PATH,
)
from src.train_robust import augment_ai_training, fit_variant
from src.utils import set_seed


def main() -> None:
    set_seed(SEED)
    if not RAID_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"RAID sample not found: {RAID_SAMPLE_PATH}. Run src/prepare_raid_sample.py first.")

    base_train = pd.read_csv(TRAIN_PATH)
    base_validation = pd.read_csv(VAL_PATH)
    raid = pd.read_csv(RAID_SAMPLE_PATH)
    required = {"text", "label", "group_id", "raid_partition"}
    if not required.issubset(raid.columns):
        raise ValueError(f"RAID sample is missing columns: {sorted(required - set(raid.columns))}")

    raid_train = raid[raid.raid_partition == "train"].copy()
    raid_validation = raid[raid.raid_partition == "validation"].copy()
    if raid_train.empty or raid_validation.empty:
        raise ValueError("RAID sample must contain train and validation source groups.")

    train_groups = set(raid_train.group_id)
    validation_groups = set(raid_validation.group_id)
    test_groups = set(raid.loc[raid.raid_partition == "test", "group_id"])
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise AssertionError("RAID source groups overlap between train, validation, and test.")

    training = pd.concat([base_train, raid_train[base_train.columns]], ignore_index=True)
    validation = pd.concat([base_validation, raid_validation[base_validation.columns]], ignore_index=True)
    if set(training.label.astype(int)) != {0, 1} or set(validation.label.astype(int)) != {0, 1}:
        raise ValueError("Combined training and validation data must contain both labels.")

    noaug_report, _, _ = fit_variant(
        training,
        validation,
        "raid_noaug",
        RAID_NOAUG_MODEL_PATH,
        RAID_NOAUG_CALIBRATOR_PATH,
        RAID_LOGISTIC_C_VALUES,
    )
    augmented, augmented_count = augment_ai_training(training)
    aug_report, _, _ = fit_variant(
        augmented,
        validation,
        "raid_aug",
        RAID_AUG_MODEL_PATH,
        RAID_AUG_CALIBRATOR_PATH,
        RAID_LOGISTIC_C_VALUES,
    )

    y = validation.label.astype(int).to_numpy()
    result = {
        "dataset": "original train/validation plus source-matched RAID train/validation",
        "seed": SEED,
        "base_train_rows": int(len(base_train)),
        "raid_train_rows": int(len(raid_train)),
        "base_validation_rows": int(len(base_validation)),
        "raid_validation_rows": int(len(raid_validation)),
        "combined_validation_rows": int(len(validation)),
        "augmentation_added_rows": int(augmented_count),
        "selection_metric": "calibrated validation F1 at probability 0.5; reported on combined validation only",
        "variants": [noaug_report, aug_report],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "raid_training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()