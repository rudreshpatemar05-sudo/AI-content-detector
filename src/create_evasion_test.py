"""Create the explicitly evaluation-only test cases supplied by the user."""

import csv
import sys

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVASION_TEST_PATH


ROWS = [
    {
        "text": "The Amazon rainforest is a vast tropical rainforest located in South America, covering approximately 5.5 million square kilometers across nine countries, including Brazil, Peru, and Colombia. It is widely regarded as one of the most biodiverse regions on Earth, providing habitat for millions of species of plants, animals, and insects. The forest plays a crucial role in regulating the global climate by absorbing large quantities of carbon dioxide and releasing oxygen. However, deforestation driven by agriculture, logging, and infrastructure development poses a significant threat to its long-term survival.",
        "label": 1,
        "kind": "ai_original",
    },
    {
        "text": "The Amazon covers roughly 5.5 million square kilometers and stretches across nine countries, with most of it in Brazil. Scientists still haven't catalogued everything living there, and every few months a new frog or beetle turns up. The forest also stores a lot of carbon, so its loss affects the whole planet's climate, not just South America. Most of the damage comes from cattle ranching, soy farms and new roads, and recovery is slow because the soil is poorer than it looks.",
        "label": 1,
        "kind": "ai_humanized",
    },
]


def main() -> None:
    EVASION_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVASION_TEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "kind"])
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"Created evaluation-only file: {EVASION_TEST_PATH}")
    print(f"Rows: {len(ROWS)}; kinds: {{'ai_original': 1, 'ai_humanized': 1}}; labels: {{1: {len(ROWS)}}}")


if __name__ == "__main__":
    main()
