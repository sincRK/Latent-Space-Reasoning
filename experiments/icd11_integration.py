"""Resolve icd11_coding repo root for LSR medical-coding experiments.

Layout (default):
  icd11_coding/                    ← this repo (benchmark, data, nie_paper)
  icd11_coding/Latent-Space-Reasoning/   ← LSR submodule on branch ``icd11``

Override with env ``ICD11_CODING_ROOT`` when LSR is checked out elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LSR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ICD11_ROOT = LSR_ROOT.parent


def icd11_coding_root() -> Path:
    raw = os.environ.get("ICD11_CODING_ROOT", "").strip()
    root = Path(raw).resolve() if raw else DEFAULT_ICD11_ROOT.resolve()
    if not (root / "icd11_coding").is_dir() and not (root / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"icd11_coding repo not found at {root}. "
            "Clone icd11_coding and set ICD11_CODING_ROOT, or nest LSR at "
            "icd11_coding/Latent-Space-Reasoning/ (git submodule on branch icd11)."
        )
    marker = root / "icd11_coding" / "benchmark"
    if marker.is_dir():
        return root
    if (root / "icd11_coding" / "__init__.py").exists() or (root / "pyproject.toml").exists():
        return root
    raise FileNotFoundError(
        f"{root} does not look like icd11_coding (missing icd11_coding/benchmark). "
        f"Set ICD11_CODING_ROOT to the repo root."
    )


def ensure_icd11_on_path(root: Path | None = None) -> Path:
    """Insert icd11_coding package root on sys.path; return resolved root."""
    repo = root or icd11_coding_root()
    path = str(repo)
    if path not in sys.path:
        sys.path.insert(0, path)
    nie = repo / "nie_paper"
    if nie.is_dir() and str(nie) not in sys.path:
        sys.path.insert(0, str(nie))
    return repo
