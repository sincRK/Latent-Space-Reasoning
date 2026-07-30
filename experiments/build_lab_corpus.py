"""Build unlabeled lab Befundstexte JSONL for Qwen DAPT stage 2.

Train source (b) only — bcc + cobra + textbausteine, ⊥ labels.

Usage (from Latent-Space-Reasoning/):
  python experiments/build_lab_corpus.py --output ../data/lab_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
from icd11_integration import ensure_icd11_on_path

ICD11_ROOT = ensure_icd11_on_path()

from labgraph.data.lab_corpus_loader import (  # noqa: E402
    default_lab_corpus_paths,
    load_lab_corpus,
)

DEFAULT_OUT = ICD11_ROOT / "data" / "lab_corpus.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build lab Befundstexte JSONL for DAPT")
    p.add_argument(
        "--paths",
        nargs="*",
        default=[str(p) for p in default_lab_corpus_paths(ICD11_ROOT)],
        help="CSV paths (default: bcc + cobra + textbausteine)",
    )
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--min-chars", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    texts = load_lab_corpus(args.paths)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for text in texts:
            if len(text) < args.min_chars:
                continue
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    print(f"✓ Wrote {n} lab texts → {out_path}")


if __name__ == "__main__":
    main()
