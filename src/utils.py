from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR))


logger = logging.getLogger("ai_content_detector")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_files_exist(paths: Sequence[str | Path], message: str) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{message} Missing: {', '.join(missing)}")


def require_project_files(required_files: Iterable[str | Path], caller: str) -> None:
    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"{caller} requires {Path(file_path).name}. Run the previous stage first: "
                f"python src/prepare_data.py"
            )
