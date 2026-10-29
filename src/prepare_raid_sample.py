from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    RAID_SAMPLE_MANIFEST_PATH,
    RAID_SAMPLE_PATH,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    HELDOUT_WIKICSAI_PATH,
)
from src.prepare_data import DATA_COLUMNS, clean_frame, clean_text


RAID_REVISION = "5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1"
RAID_SHARD_TEMPLATE = (
    "https://huggingface.co/datasets/liamdugan/raid/resolve/"
    f"{RAID_REVISION}/raid/partial-train/{{shard:04d}}.parquet"
)
SHARDS_WITH_HUMAN_ROWS = (0, 2, 3, 5, 6, 7, 8, 9)
ROW_GROUPS_PER_SHARD = 4
RAID_GENERATORS = (
    "chatgpt", "cohere", "cohere-chat", "gpt2", "gpt3", "gpt4",
    "llama-chat", "mistral", "mistral-chat", "mpt", "mpt-chat",
)
HELDOUT_AI_MODELS = ("gpt2", "mpt")
ROW_COLUMNS = ["source_id", "model", "domain", "attack", "generation"]


def _read_row_group(shard: int, row_group: int) -> pd.DataFrame:
    url = RAID_SHARD_TEMPLATE.format(shard=shard)
    with fsspec.open(url, "rb", block_size=65_536, cache_type="readahead").open() as remote:
        parquet = pq.ParquetFile(remote)
        table = parquet.read_row_group(row_group, columns=ROW_COLUMNS)
    return table.to_pandas()


def _sample_records() -> tuple[pd.DataFrame, dict[str, object]]:
    rng = random.Random(SEED)
    human_by_source: dict[str, dict[str, object]] = {}
    selected_groups: dict[str, list[int]] = {}

    for shard in SHARDS_WITH_HUMAN_ROWS:
        url = RAID_SHARD_TEMPLATE.format(shard=shard)
        with fsspec.open(url, "rb", block_size=65_536, cache_type="readahead").open() as remote:
            parquet = pq.ParquetFile(remote)
            model_column = parquet.schema_arrow.get_field_index("model")
            human_groups = []
            for row_group in range(parquet.metadata.num_row_groups):
                stats = parquet.metadata.row_group(row_group).column(model_column).statistics
                if stats and "human" in (stats.min, stats.max):
                    human_groups.append(row_group)

        count = min(ROW_GROUPS_PER_SHARD, len(human_groups))
        selected = sorted(rng.sample(human_groups, count))
        selected_groups[str(shard)] = selected
        for row_group in selected:
            frame = _read_row_group(shard, row_group)
            for row in frame[frame["model"] == "human"].to_dict(orient="records"):
                source_id = str(row["source_id"])
                human_by_source.setdefault(source_id, row)

    human_domains = {str(row["domain"]) for row in human_by_source.values()}
    ai_candidates: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for shard in range(10):
        url = RAID_SHARD_TEMPLATE.format(shard=shard)
        with fsspec.open(url, "rb", block_size=65_536, cache_type="readahead").open() as remote:
            parquet = pq.ParquetFile(remote)
            model_column = parquet.schema_arrow.get_field_index("model")
            domain_column = parquet.schema_arrow.get_field_index("domain")
            for row_group in range(parquet.metadata.num_row_groups):
                model_stats = parquet.metadata.row_group(row_group).column(model_column).statistics
                domain_stats = parquet.metadata.row_group(row_group).column(domain_column).statistics
                if not model_stats or not domain_stats:
                    continue
                domain = str(domain_stats.min)
                if domain_stats.min != domain_stats.max or domain not in human_domains:
                    continue
                for model_name in RAID_GENERATORS:
                    if model_stats.min == model_name or model_stats.max == model_name:
                        ai_candidates[(model_name, domain)].append((shard, row_group))

    ai_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(ai_candidates):
        candidates = ai_candidates[key]
        rng.shuffle(candidates)
        for shard, row_group in candidates:
            frame = _read_row_group(shard, row_group)
            matched = frame[
                (frame["model"] == key[0])
                & frame["source_id"].astype(str).isin(human_by_source)
            ]
            if not matched.empty:
                for row in matched.to_dict(orient="records"):
                    ai_by_source[str(row["source_id"])].append(row)
                break

    paired_rows: list[dict[str, object]] = []
    model_counts: Counter[str] = Counter()
    source_ids = list(set(human_by_source) & set(ai_by_source))
    rng.shuffle(source_ids)
    for source_id in source_ids:
        human = human_by_source[source_id]
        ai_rows = ai_by_source[source_id]
        if not ai_rows:
            continue

        least_used = min(model_counts[str(row["model"])] for row in ai_rows)
        model_options = sorted({str(row["model"]) for row in ai_rows if model_counts[str(row["model"])] == least_used})
        selected_model = rng.choice(model_options)
        generated = rng.choice([row for row in ai_rows if str(row["model"]) == selected_model])
        model_counts[selected_model] += 1
        domain = str(generated["domain"])
        pair_id = f"raid:{source_id}"
        paired_rows.extend([
            {
                "dataset": "raid",
                "source": f"raid_human:{domain}",
                "group_id": pair_id,
                "text": human["generation"],
                "label": 0,
                "model": "human",
                "domain": str(human["domain"]),
                "attack": str(human["attack"]),
            },
            {
                "dataset": "raid",
                "source": f"raid_{generated['model']}:{domain}",
                "group_id": pair_id,
                "text": generated["generation"],
                "label": 1,
                "model": str(generated["model"]),
                "domain": domain,
                "attack": str(generated["attack"]),
            },
        ])

    if not paired_rows:
        raise ValueError("The selected RAID row groups contained no source-matched human/AI pairs.")

    frame = pd.DataFrame(paired_rows)
    frame["text"] = frame["text"].map(clean_text)
    frame["_normalized_text"] = frame["text"].map(lambda text: " ".join(text.split()).casefold())
    seen_texts: set[str] = set()
    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH, HELDOUT_WIKICSAI_PATH):
        if path.exists():
            existing = clean_frame(pd.read_csv(path))["text"]
            seen_texts.update(existing.map(lambda text: " ".join(text.split()).casefold()))
    frame = frame[~frame["_normalized_text"].isin(seen_texts)].drop_duplicates(
        subset="_normalized_text", keep="first"
    )

    metadata = frame.set_index("_normalized_text")[["model", "domain", "attack"]]
    cleaned = clean_frame(frame[DATA_COLUMNS])
    cleaned["_normalized_text"] = cleaned["text"].map(lambda text: " ".join(text.split()).casefold())
    for column in ("model", "domain", "attack"):
        cleaned[column] = cleaned["_normalized_text"].map(metadata[column])
    frame = cleaned.drop(columns="_normalized_text")

    labels_per_group = frame.groupby("group_id")["label"].nunique()
    valid_groups = labels_per_group[labels_per_group == 2].index
    frame = frame[frame["group_id"].isin(valid_groups)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid paired RAID examples remain after cleaning and deduplication.")

    test_groups = set(frame.loc[frame["model"].isin(HELDOUT_AI_MODELS), "group_id"])
    development_groups = frame.loc[~frame["group_id"].isin(test_groups), ["group_id"]].drop_duplicates().reset_index(drop=True)
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=SEED).split(
            development_groups, groups=development_groups["group_id"]
        )
    )
    group_partitions = {
        "train": set(development_groups.iloc[train_idx]["group_id"]),
        "validation": set(development_groups.iloc[val_idx]["group_id"]),
        "test": test_groups,
    }
    for name, group_ids in group_partitions.items():
        frame.loc[frame["group_id"].isin(group_ids), "raid_partition"] = name

    split_groups = {name: set(group["group_id"]) for name, group in frame.groupby("raid_partition")}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_groups.get(left, set()) & split_groups.get(right, set()):
            raise AssertionError(f"RAID source groups overlap between {left} and {right}.")
    train_models = set(frame.loc[frame["raid_partition"] == "train", "model"]) - {"human"}
    validation_models = set(frame.loc[frame["raid_partition"] == "validation", "model"]) - {"human"}
    test_models = set(frame.loc[frame["raid_partition"] == "test", "model"]) - {"human"}
    if test_models & (train_models | validation_models):
        raise AssertionError("A held-out RAID model appears in train or validation.")

    manifest = {
        "dataset": "liamdugan/raid",
        "license": "MIT",
        "revision": RAID_REVISION,
        "seed": SEED,
        "sampling": {
            "shards": list(SHARDS_WITH_HUMAN_ROWS),
            "row_groups_per_shard": ROW_GROUPS_PER_SHARD,
            "source_matched_rows_per_group": 2,
            "ai_generators": sorted(model_counts),
            "heldout_ai_models": list(HELDOUT_AI_MODELS),
            "selection": "one human and one model-balanced AI generation per matched source_id; held-out generators reserved for test",
        },
        "selected_row_groups": selected_groups,
        "rows": int(len(frame)),
        "source_groups": int(frame["group_id"].nunique()),
        "model_counts": {str(key): int(value) for key, value in frame["model"].value_counts().items()},
        "domain_counts": {str(key): int(value) for key, value in frame["domain"].value_counts().items()},
        "partitions": {
            str(name): {
                "rows": int(len(group)),
                "source_groups": int(group["group_id"].nunique()),
                "label_counts": {str(key): int(value) for key, value in group["label"].value_counts().sort_index().items()},
            }
            for name, group in frame.groupby("raid_partition")
        },
    }
    return frame, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repartition-existing",
        action="store_true",
        help="Repartition the saved sample without fetching RAID again.",
    )
    args = parser.parse_args()
    if args.repartition_existing:
        if not RAID_SAMPLE_PATH.exists() or not RAID_SAMPLE_MANIFEST_PATH.exists():
            raise FileNotFoundError("An existing RAID sample and manifest are required for offline repartitioning.")
        frame = pd.read_csv(RAID_SAMPLE_PATH)
        manifest = json.loads(RAID_SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
        test_groups = set(frame.loc[frame["model"].isin(HELDOUT_AI_MODELS), "group_id"])
        development_groups = frame.loc[
            ~frame["group_id"].isin(test_groups), ["group_id"]
        ].drop_duplicates().reset_index(drop=True)
        if not test_groups or development_groups.empty:
            raise ValueError("The saved sample cannot provide both held-out and development model groups.")
        train_idx, val_idx = next(
            GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=SEED).split(
                development_groups, groups=development_groups["group_id"]
            )
        )
        partitions = {
            "train": set(development_groups.iloc[train_idx]["group_id"]),
            "validation": set(development_groups.iloc[val_idx]["group_id"]),
            "test": test_groups,
        }
        frame["raid_partition"] = ""
        for name, group_ids in partitions.items():
            frame.loc[frame["group_id"].isin(group_ids), "raid_partition"] = name
        model_sets = {
            name: set(part.loc[part["label"] == 1, "model"])
            for name, part in frame.groupby("raid_partition")
        }
        if model_sets.get("test", set()) != set(HELDOUT_AI_MODELS):
            raise ValueError(f"Expected held-out AI models {HELDOUT_AI_MODELS}; found {model_sets.get('test')}.")
        if model_sets["test"] & (model_sets["train"] | model_sets["validation"]):
            raise AssertionError("A held-out AI model appears in train or validation.")
        manifest["sampling"]["heldout_ai_models"] = list(HELDOUT_AI_MODELS)
        manifest["sampling"]["selection"] = "existing sample repartitioned by generator; source groups remain grouped"
        manifest["partitions"] = {
            str(name): {
                "rows": int(len(part)),
                "source_groups": int(part["group_id"].nunique()),
                "label_counts": {
                    str(key): int(value)
                    for key, value in part["label"].value_counts().sort_index().items()
                },
                "ai_models": sorted(str(model) for model in model_sets[name]),
            }
            for name, part in frame.groupby("raid_partition")
        }
    else:
        frame, manifest = _sample_records()
    RAID_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAID_SAMPLE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAID_SAMPLE_PATH, index=False)
    RAID_SAMPLE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved RAID sample: {RAID_SAMPLE_PATH}")


if __name__ == "__main__":
    main()