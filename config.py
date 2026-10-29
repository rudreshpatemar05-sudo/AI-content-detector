from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
APP_DIR = PROJECT_ROOT / "app"

RAW_HC3_PATH = RAW_DATA_DIR / "all.jsonl"
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
HELDOUT_WIKICSAI_PATH = PROCESSED_DATA_DIR / "heldout_wikicsai.csv"
HELDOUT_CUSTOM_PATH = PROCESSED_DATA_DIR / "heldout_custom.csv"

MODEL_PATH = MODELS_DIR / "tfidf_lr.joblib"
CALIBRATOR_PATH = MODELS_DIR / "tfidf_calibrator.joblib"
GPT_WIKI_ZIP_PATH = RAW_DATA_DIR / "GPT-wiki-intro.csv.zip"
GPT_WIKI_CSV_MEMBER = "GPT-wiki-intro.csv"
CUSTOM_TEST_RAW_PATH = RAW_DATA_DIR / "my_test.csv"
EVASION_TEST_PATH = RAW_DATA_DIR / "evasion_test.csv"
EVASION_TEST_USE = "evaluation_only"
RAID_SAMPLE_PATH = RAW_DATA_DIR / "raid_sample.csv"
RAID_SAMPLE_MANIFEST_PATH = RESULTS_DIR / "raid_sample_manifest.json"
ROBUST_NOAUG_MODEL_PATH = MODELS_DIR / "tfidf_lr_robust_noaug.joblib"
ROBUST_AUG_MODEL_PATH = MODELS_DIR / "tfidf_lr_robust_aug.joblib"
ROBUST_NOAUG_CALIBRATOR_PATH = MODELS_DIR / "tfidf_lr_robust_noaug_calibrator.joblib"
ROBUST_AUG_CALIBRATOR_PATH = MODELS_DIR / "tfidf_lr_robust_aug_calibrator.joblib"
RAID_NOAUG_MODEL_PATH = MODELS_DIR / "tfidf_lr_raid_noaug.joblib"
RAID_AUG_MODEL_PATH = MODELS_DIR / "tfidf_lr_raid_aug.joblib"
RAID_NOAUG_CALIBRATOR_PATH = MODELS_DIR / "tfidf_lr_raid_noaug_calibrator.joblib"
RAID_AUG_CALIBRATOR_PATH = MODELS_DIR / "tfidf_lr_raid_aug_calibrator.joblib"
V1_MODEL_PATH = MODELS_DIR / "tfidf_lr_v1.joblib"
V1_CALIBRATOR_PATH = MODELS_DIR / "tfidf_calibrator_v1.joblib"
V1_METRICS_PATH = RESULTS_DIR / "v1_metrics.json"

SEED = 42
HUMAN_LABEL = 0
AI_LABEL = 1
SENTENCE_GROUP_SIZE = 3
GPT_WIKI_MAX_ROWS = 30_000
WORD_NGRAM_RANGE = (1, 2)
WORD_MIN_DF = 2
WORD_MAX_FEATURES = 200_000
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_FEATURES = 200_000
LOGISTIC_C_VALUES = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50)
RAID_LOGISTIC_C_VALUES = (0.5, 5, 20, 50)
LOGISTIC_MAX_ITER = 2_000
CALIBRATION_C = 1.0
TARGET_HUMAN_PURITY = 0.90
TARGET_AI_PRECISION = 0.90
TOP_FEATURE_COUNT = 20
CALIBRATION_BINS = 10
ROBUST_AUGMENT_FRACTION = 0.20
CALIBRATION_PRECISION_TARGET = 0.98
PROBABILITY_DISPLAY_MIN = 0.01
PROBABILITY_DISPLAY_MAX = 0.99
STYLE_FEATURE_COUNT = 7
HIGHLIGHT_WINDOW_WORDS = 40
HIGHLIGHT_WINDOW_SENTENCES = 4
HIGHLIGHT_WINDOW_STEP = 2

BOT_THRESHOLD = 0.813246213291
HUMAN_THRESHOLD = 0.499810127632
MIN_WORDS_FOR_DETECTION = 30

# Data source config
HC3_DATASET_REPO = "Hello-SimpleAI/HC3"
GPT_WIKI_DATASET_REPO = "aadityaubhat/GPT-wiki-intro"
DATA_MANIFEST_PATH = RESULTS_DIR / "data_manifest.json"
TOP_FEATURES_PATH = RESULTS_DIR / "top_features.json"
RELIABILITY_PLOT_PATH = RESULTS_DIR / "calibration_reliability.png"
COMPARISON_PLOT_PATH = RESULTS_DIR / "comparison.png"
TUNING_RESULTS_PATH = RESULTS_DIR / "tuning_results.json"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, APP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

