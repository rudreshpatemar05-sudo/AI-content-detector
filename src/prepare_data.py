from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CUSTOM_TEST_RAW_PATH,
    DATA_MANIFEST_PATH,
    GPT_WIKI_CSV_MEMBER,
    GPT_WIKI_MAX_ROWS,
    GPT_WIKI_ZIP_PATH,
    HC3_DATASET_REPO,
    HELDOUT_CUSTOM_PATH,
    HELDOUT_WIKICSAI_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_HC3_PATH,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from src.utils import logger, set_seed


DATA_COLUMNS = ["dataset", "source", "group_id", "text", "label"]


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def is_english_text(text: str, ascii_ratio_threshold: float = 0.85) -> bool:
    if not text:
        return False
    allowed = sum(
        ch.isascii() and (ch.isalpha() or ch.isspace() or ch.isdigit() or ch in ".,!?;:'\"()[]{}-/\\")
        for ch in text
    )
    return allowed / max(len(text), 1) >= ascii_ratio_threshold


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(DATA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    result = frame[DATA_COLUMNS].copy()
    result["text"] = result["text"].map(clean_text)
    result["group_id"] = result["group_id"].astype(str)
    result["label"] = pd.to_numeric(result["label"], errors="coerce")
    result = result[result["label"].isin([0, 1]) & result["group_id"].notna()].copy()
    result["label"] = result["label"].astype(int)
    result = result[result["text"].str.split().str.len() >= 10]
    result = result[result["text"].map(is_english_text)]
    return result.reset_index(drop=True)


def download_hc3_jsonl() -> Path:
    if RAW_HC3_PATH.exists():
        return RAW_HC3_PATH
    cached_path = hf_hub_download(
        repo_id=HC3_DATASET_REPO,
        repo_type="dataset",
        filename="all.jsonl",
    )
    return Path(cached_path)


def load_hc3() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with download_hc3_jsonl().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = record.get("id", record.get("qid", record.get("question_id", line_number)))
            source = str(record.get("source") or "unknown")
            for answer in record.get("human_answers") or []:
                rows.append({
                    "dataset": "hc3",
                    "source": source,
                    "group_id": str(question_id),
                    "text": answer,
                    "label": 0,
                })
            for answer in record.get("chatgpt_answers") or []:
                rows.append({
                    "dataset": "hc3",
                    "source": source,
                    "group_id": str(question_id),
                    "text": answer,
                    "label": 1,
                })
    return clean_frame(pd.DataFrame(rows, columns=DATA_COLUMNS))


def load_gpt_wiki() -> pd.DataFrame:
    if not GPT_WIKI_ZIP_PATH.exists():
        raise FileNotFoundError(f"GPT-wiki archive not found: {GPT_WIKI_ZIP_PATH}")
    with zipfile.ZipFile(GPT_WIKI_ZIP_PATH) as archive:
        if GPT_WIKI_CSV_MEMBER not in archive.namelist():
            raise ValueError(
                f"Expected {GPT_WIKI_CSV_MEMBER!r} in {GPT_WIKI_ZIP_PATH.name}; "
                f"found {archive.namelist()!r}"
            )
        source = archive.open(GPT_WIKI_CSV_MEMBER)
        raw = pd.read_csv(source, usecols=["id", "wiki_intro", "generated_intro"])

    expected_columns = {"id", "wiki_intro", "generated_intro"}
    if not expected_columns.issubset(raw.columns):
        raise ValueError(f"GPT-wiki CSV columns do not match the inspected mapping: {list(raw.columns)!r}")
    logger.info("GPT-wiki loaded: shape=%s; columns=%s", raw.shape, list(raw.columns))
    logger.info("GPT-wiki sample rows:\n%s", raw.head(2).to_string(index=False, max_colwidth=240))

    pair_count = max(1, GPT_WIKI_MAX_ROWS // 2)
    if len(raw) > pair_count:
        raw = raw.sample(n=pair_count, random_state=SEED)

    rows: list[dict[str, Any]] = []
    for row in raw.itertuples(index=False):
        group_id = str(row.id)
        rows.append({"dataset": "gpt_wiki", "source": "wikipedia", "group_id": group_id, "text": row.wiki_intro, "label": 0})
        rows.append({"dataset": "gpt_wiki", "source": "gpt_generated", "group_id": group_id, "text": row.generated_intro, "label": 1})
    return clean_frame(pd.DataFrame(rows, columns=DATA_COLUMNS))


def load_custom_holdout() -> pd.DataFrame:
    if not CUSTOM_TEST_RAW_PATH.exists():
        return pd.DataFrame(columns=DATA_COLUMNS)
    raw = pd.read_csv(CUSTOM_TEST_RAW_PATH)
    if not {"text", "label"}.issubset(raw.columns):
        raise ValueError(
            f"{CUSTOM_TEST_RAW_PATH.name} must contain text,label columns; found {list(raw.columns)!r}"
        )
    frame = pd.DataFrame({
        "dataset": "custom",
        "source": CUSTOM_TEST_RAW_PATH.stem,
        "group_id": [f"custom:{index}" for index in range(len(raw))],
        "text": raw["text"],
        "label": raw["label"],
    })
    return clean_frame(frame)


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    conflicting_texts = frame.groupby("text")["label"].nunique()
    conflicting_texts = set(conflicting_texts[conflicting_texts > 1].index)
    if conflicting_texts:
        logger.warning("Dropping %d texts with conflicting labels.", len(conflicting_texts))
        frame = frame[~frame["text"].isin(conflicting_texts)]
    return frame.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)


def split_by_group(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Dataset-prefix the IDs so an HC3 question ID cannot collide with a GPT-wiki row ID.
    frame = frame.copy()
    frame["_split_group"] = frame["dataset"].astype(str) + "::" + frame["group_id"].astype(str)
    unique_groups = frame["_split_group"].drop_duplicates().to_frame()
    train_idx, remainder_idx = next(
        GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEED).split(
            unique_groups, groups=unique_groups["_split_group"]
        )
    )
    remainder = unique_groups.iloc[remainder_idx].reset_index(drop=True)
    val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, train_size=1 / 3, random_state=SEED + 1).split(
            remainder, groups=remainder["_split_group"]
        )
    )
    train_groups = set(unique_groups.iloc[train_idx]["_split_group"])
    val_groups = set(remainder.iloc[val_idx]["_split_group"])
    test_groups = set(remainder.iloc[test_idx]["_split_group"])
    parts = []
    for group_set in (train_groups, val_groups, test_groups):
        parts.append(frame[frame["_split_group"].isin(group_set)].drop(columns="_split_group").reset_index(drop=True))
    return parts[0], parts[1], parts[2]


def balance_train(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame["label"].value_counts()
    if not {0, 1}.issubset(set(counts.index)):
        raise ValueError(f"Training split must contain both labels 0 and 1; found {counts.to_dict()}")
    per_class = int(counts.min())
    balanced = pd.concat(
        [frame[frame["label"] == label].sample(n=per_class, random_state=SEED) for label in (0, 1)],
        ignore_index=True,
    )
    return balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)


def print_counts(name: str, frame: pd.DataFrame) -> None:
    labels = frame["label"].value_counts().sort_index().to_dict() if not frame.empty else {}
    datasets = frame["dataset"].value_counts().sort_index().to_dict() if not frame.empty else {}
    logger.info("%s rows=%d | labels=%s | datasets=%s", name, len(frame), labels, datasets)


def main() -> None:
    set_seed(SEED)
    hc3 = load_hc3()
    heldout_wiki = deduplicate(hc3[hc3["source"] == "wiki_csai"].copy())
    hc3_mixed = hc3[hc3["source"] != "wiki_csai"].copy()
    gpt_wiki = load_gpt_wiki()
    custom = load_custom_holdout()

    heldout_texts = set(heldout_wiki["text"])
    if not custom.empty:
        heldout_texts.update(custom["text"])
    mixed = pd.concat([hc3_mixed, gpt_wiki], ignore_index=True)
    mixed = deduplicate(mixed)
    mixed = mixed[~mixed["text"].isin(heldout_texts)].reset_index(drop=True)

    train, val, test = split_by_group(mixed)
    train = balance_train(train)
    assert not set(heldout_wiki["text"]).intersection(train["text"]), "Held-out wiki_csai text leaked into train."
    assert not set(heldout_wiki["text"]).intersection(val["text"]), "Held-out wiki_csai text leaked into validation."
    if not custom.empty:
        assert not set(custom["text"]).intersection(train["text"]), "Custom holdout text leaked into train."
        assert not set(custom["text"]).intersection(val["text"]), "Custom holdout text leaked into validation."

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train[DATA_COLUMNS].to_csv(TRAIN_PATH, index=False)
    val[DATA_COLUMNS].to_csv(VAL_PATH, index=False)
    test[DATA_COLUMNS].to_csv(TEST_PATH, index=False)
    heldout_wiki[DATA_COLUMNS].to_csv(HELDOUT_WIKICSAI_PATH, index=False)
    if not custom.empty:
        custom[DATA_COLUMNS].to_csv(HELDOUT_CUSTOM_PATH, index=False)
    elif HELDOUT_CUSTOM_PATH.exists():
        HELDOUT_CUSTOM_PATH.unlink()

    print_counts("train_balanced", train)
    print_counts("validation", val)
    print_counts("combined_test", test)
    print_counts("heldout_wikicsai", heldout_wiki)
    if not custom.empty:
        print_counts("heldout_custom", custom)
    manifests = {
        "train_balanced": train,
        "validation": val,
        "combined_test": test,
        "heldout_wikicsai": heldout_wiki,
    }
    if not custom.empty:
        manifests["heldout_custom"] = custom
    manifest_splits = {}
    for name, frame in manifests.items():
        manifest_splits[name] = {
            "rows": int(len(frame)),
            "label_counts": {str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()},
            "dataset_counts": {str(key): int(value) for key, value in frame["dataset"].value_counts().sort_index().items()},
        }
    DATA_MANIFEST_PATH.write_text(
        json.dumps({
            "sources": {"hc3": "loaded", "gpt_wiki_intro": "loaded_from_local_archive"},
            "held_out_dataset": "HC3 wiki_csai",
            "splits": manifest_splits,
            "grouped_split_seed": SEED,
            "labels": {"0": "human", "1": "AI"},
        }, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved train/val/test and holdouts under %s", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
