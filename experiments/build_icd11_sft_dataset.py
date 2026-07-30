"""Build multi-format ICD-11 catalog SFT JSONL for Qwen QLoRA stage 1.

Train source (a) only — category_strings* catalog, ⊥ benchmark pairings.
Reuses nie_paper catalog guard + lab corpus path conventions.

Usage (from Latent-Space-Reasoning/):
  python experiments/build_icd11_sft_dataset.py \\
    --csv ../notebooks/data/category_strings_oct2_with_definitions.csv \\
    --output ../data/icd11_catalog_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
from icd11_integration import ensure_icd11_on_path, icd11_coding_root

ICD11_ROOT = ensure_icd11_on_path()

from labgraph.data.catalog_loader import (  # noqa: E402
    _branch_path_text,
    _catalog_description,
    _is_benchmark_csv,
    _parse_list_field,
)

SYSTEM_MSG = "Answer to the best of your ability."
DEFAULT_CSV = ICD11_ROOT / "notebooks" / "data" / "category_strings_oct2_with_definitions.csv"
DEFAULT_OUT = ICD11_ROOT / "data" / "icd11_catalog_sft.jsonl"

CODE_SYSTEMS = {
    "icd11": {
        "label": "ICD-11 MMS",
        "default_csv": DEFAULT_CSV,
        "default_out": DEFAULT_OUT,
    },
    "icd10": {
        "label": "ICD-10-GM",
        "default_csv": ICD11_ROOT / "notebooks" / "data" / "icd10_category_strings.csv",
        "default_out": ICD11_ROOT / "data" / "icd10_catalog_sft.jsonl",
    },
}


def _parent_key(code: str) -> str:
    if "." in code:
        return code.rsplit(".", 1)[0]
    return code[:4] if len(code) >= 4 else code


def _chat_record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def build_sft_records(df, *, seed: int, max_sibling_pairs: int, code_label: str) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    by_parent: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for _, row in df.iterrows():
        code = str(row["chosen_category"]).strip()
        if not code or code == "nan":
            continue

        desc = _catalog_description(row)
        branch = _branch_path_text(row)
        titles = _parse_list_field(row.get("leaf_titles_list"))

        if desc:
            records.append(
                _chat_record(
                    f"What is the {code_label} code for this clinical description?\n\n{desc}\n\n"
                    "Reply with only the code.",
                    code,
                )
            )
            records.append(
                _chat_record(
                    f"Describe {code_label} code {code} in clinical terms.",
                    desc,
                )
            )

        if branch and branch != desc:
            records.append(
                _chat_record(
                    f"What {code_label} leaf code matches this hierarchy path?\n\n{branch}\n\n"
                    "Reply with only the code.",
                    code,
                )
            )

        for title in titles:
            title = title.strip()
            if len(title) < 3:
                continue
            records.append(
                _chat_record(
                    f"Map this {code_label} synonym or short title to its code:\n\n{title}\n\n"
                    "Reply with only the code.",
                    code,
                )
            )

        label_text = desc or branch or " | ".join(titles) or code
        by_parent[_parent_key(code)].append((code, label_text, branch or desc or code))

    sibling_added = 0
    for _parent, members in by_parent.items():
        if len(members) < 2:
            continue
        rng.shuffle(members)
        for i in range(0, len(members) - 1, 2):
            if sibling_added >= max_sibling_pairs:
                break
            (code_a, text_a, _), (code_b, text_b, _) = members[i], members[i + 1]
            if code_a == code_b:
                continue
            options = [code_a, code_b]
            rng.shuffle(options)
            opt_str = ", ".join(options)
            records.append(
                _chat_record(
                    f"Which {code_label} code best matches this description?\n\n"
                    f"{text_a}\n\n"
                    f"Choose one of: {opt_str}\n"
                    "Reply with only the code.",
                    code_a,
                )
            )
            sibling_added += 1
        if sibling_added >= max_sibling_pairs:
            break

    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build catalog SFT JSONL for QLoRA")
    p.add_argument("--taxonomy", default="icd11", choices=sorted(CODE_SYSTEMS))
    p.add_argument("--csv", default="")
    p.add_argument("--output", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-sibling-pairs",
        type=int,
        default=2000,
        help="cap sibling-discrimination examples",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = CODE_SYSTEMS[args.taxonomy]
    csv_path = Path(args.csv or spec["default_csv"])
    out_path = Path(args.output or spec["default_out"])
    code_label = spec["label"]

    import pandas as pd

    df = pd.read_csv(csv_path)
    if _is_benchmark_csv(df, csv_path):
        raise ValueError(f"Refusing benchmark CSV for SFT: {csv_path}")

    records = build_sft_records(
        df,
        seed=args.seed,
        max_sibling_pairs=args.max_sibling_pairs,
        code_label=code_label,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(records)} SFT records → {out_path}")


if __name__ == "__main__":
    main()
