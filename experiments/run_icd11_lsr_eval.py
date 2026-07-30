"""ICD-11 benchmark eval with Qwen3 + latent-space perturbation.

Runs the oct20 ICD-11 bench (293 cases, 251 scored) through:
  1. baseline — greedy zero-shot (no soft prefix)
  2. perturb@K — random embedding prefix (LSR random_noise) + plurality vote

Matches LSR sensitivity setup: 2 soft tokens scaled to embedding RMS.

Usage (from Latent-Space-Reasoning/):
  python experiments/run_icd11_lsr_eval.py --mode baseline --limit 5
  python experiments/run_icd11_lsr_eval.py --mode perturb --n-seeds 10
  python experiments/run_icd11_lsr_eval.py --mode both --output ../results/icd11_lsr_qwen3_4b_8bit.json
  python experiments/run_icd11_lsr_eval.py --mode both --adapter ../checkpoints/qwen_icd11_lora_dapt

Requires: pip install -e ".[quant]" in this repo + icd11_coding on PYTHONPATH.
Train: build_icd11_sft_dataset.py → train_qwen_coding_lora.py --stage sft → build_lab_corpus.py → train --stage dapt
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
LSR_ROOT = EXPERIMENTS_DIR.parent

from icd11_integration import ensure_icd11_on_path, icd11_coding_root  # noqa: E402

ICD11_ROOT = icd11_coding_root()
ensure_icd11_on_path(ICD11_ROOT)

sys.path.insert(0, str(LSR_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from harness import auto_calibrate
from latent_reasoning.core.encoder import LLMEncoder
from run_latent_sensitivity import decode_with_raw_soft_prompt, run_zero_shot, safe_print
from strip_thinking_tokens import strip_thinking

from icd11_coding.benchmark.icd11 import build_icd11_oct20_benchmark, is_scorable_label
from icd11_coding.benchmark.icd10 import load_icd10_benchmark
from icd11_coding.benchmark.metrics import match_any_predicted, normalize_code, primary_label


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_QUANT = "8bit"

TAXONOMY = {
    "icd11": {
        "benchmark_name": "icd11_oct20",
        "default_benchmark": ICD11_ROOT / "notebooks" / "data" / "icd11_oct20_benchmark.csv",
        "default_output": ICD11_ROOT / "results" / "icd11_lsr_eval.json",
        "load_cases": lambda path: build_icd11_oct20_benchmark(path),
        "prompt_template": """\
Assign the single most appropriate ICD-11 MMS code for this pathology report.

Report:
{text}

Reply with only the ICD-11 MMS code (example: 2F21.0). No explanation.""",
        "label_pattern": r"ICD[- ]?11[^:]*:\s*([0-9A-Z][0-9A-Z.]+)",
        "code_system": "ICD-11 MMS",
        "example_code": "2F21.0",
    },
    "icd10": {
        "benchmark_name": "icd10_testdatensatz",
        "default_benchmark": ICD11_ROOT / "notebooks" / "data" / "icd10_benchmark.csv",
        "default_output": ICD11_ROOT / "results" / "icd10_lsr_eval.json",
        "load_cases": lambda path: load_icd10_benchmark(benchmark_csv=path),
        "prompt_template": """\
Assign the single most appropriate ICD-10-GM code for this pathology report.

Report:
{text}

Reply with only the ICD-10-GM code (example: C43.1). No explanation.""",
        "label_pattern": r"ICD[- ]?10[^:]*:\s*([A-Z][0-9][0-9A-Z.]*)",
        "code_system": "ICD-10-GM",
        "example_code": "C43.1",
    },
}

ICD11_CODE_RE = re.compile(
    r"\b("
    r"[0-9][0-9A-Z]{1,4}(?:\.[0-9A-Z]+)*"
    r"|[A-Z][0-9]{2}[0-9A-Z]*(?:\.[0-9A-Z]+)*"
    r")\b",
    re.IGNORECASE,
)

_ICD10_CODE_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b")

# ICD-10 / ICD-O / ICD-11 label variants in report text (incl. unicode hyphen/space).
_ICD_LABEL = r"ICD(?:[-\s‑–—]?O|O)?(?:[-\s‑–—]?10|10)?"
_ICD10_BODY = r"[A-Z]\s*\d[\d.]*"
_ICDO_MORPH = r"M\s*\d{4}/\d"

USER_PROMPT_TEMPLATE = TAXONOMY["icd11"]["prompt_template"]


def get_taxonomy(name: str) -> dict:
    key = name.lower()
    if key not in TAXONOMY:
        raise ValueError(f"Unknown taxonomy {name!r}; choose from {list(TAXONOMY)}")
    return TAXONOMY[key]


def format_user_prompt(text: str, *, taxonomy: dict | None = None) -> str:
    template = (taxonomy or TAXONOMY["icd11"])["prompt_template"]
    return template.format(text=text.strip())


def extract_code(response: str, *, taxonomy: dict | None = None) -> str:
    """Parse taxonomy code from model output."""
    tax = taxonomy or TAXONOMY["icd11"]
    cleaned, _ = strip_thinking(response)
    text = cleaned.strip()
    if not text:
        return ""

    labeled = re.search(tax["label_pattern"], text, re.IGNORECASE)
    if labeled:
        return normalize_code(labeled.group(1))

    first_line = text.splitlines()[0].strip()
    if tax["benchmark_name"].startswith("icd10"):
        icd10_match = _ICD10_CODE_RE.search(first_line)
        if icd10_match:
            return normalize_code(icd10_match.group(1))
        matches = _ICD10_CODE_RE.findall(text)
        if matches:
            return normalize_code(max(matches, key=len))
        return normalize_code(first_line)

    if re.fullmatch(r"[0-9A-Z][0-9A-Z.]{2,12}", first_line, re.IGNORECASE):
        return normalize_code(first_line)

    tail = first_line.split(":")[-1].strip()
    if re.fullmatch(r"[0-9A-Z][0-9A-Z.]{2,12}", tail, re.IGNORECASE):
        return normalize_code(tail)

    matches = ICD11_CODE_RE.findall(text)
    if not matches:
        return normalize_code(first_line)
    return normalize_code(max(matches, key=len))


def extract_icd11_code(response: str) -> str:
    return extract_code(response, taxonomy=TAXONOMY["icd11"])


@dataclass
class CaseResult:
    case_id: str
    scorable: bool
    labels: tuple[str, ...]
    predicted: str
    exact_match: bool
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ICD11_ROOT))
    except ValueError:
        return str(path)


def remove_codes_from_text(text: str) -> str:
    """Strip ICD-10 / ICD-O / ICD-11 code mentions from report text."""
    cleaned = text

    # Parenthetical blocks that cite a classification code.
    cleaned = re.sub(
        rf"\([^)]*?(?:{_ICD_LABEL}|ICDO)[^)]*?\)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Inline labelled codes, e.g. ICD10:C44.9 or ICD-10: D 23.9.
    cleaned = re.sub(
        rf"(?:{_ICD_LABEL})\s*:?\s*(?:{_ICD10_BODY}|{_ICDO_MORPH})",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Bare morphology codes, e.g. ICDO M 9540/0.
    cleaned = re.sub(
        rf"ICDO?\s*{_ICDO_MORPH}",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return cleaned.strip()


def prepare_report_text(text: str, *, remove_codes: bool) -> tuple[str, str]:
    raw = text.strip()
    if not remove_codes:
        return raw, raw
    return remove_codes_from_text(raw), raw


def _unwrap_zero_shot(result: str | tuple) -> tuple[str, str]:
    if isinstance(result, tuple):
        resp, raw = result[0], result[1] if len(result) > 1 else result[0]
        return resp, raw
    return result, result


def make_random_noise_soft_prompt(
    *,
    num_soft_tokens: int,
    embed_dim: int,
    target_rms: float,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    sp = torch.randn(1, num_soft_tokens, embed_dim, generator=gen)
    current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
    return sp * (target_rms / current_rms)


def plurality_vote(codes: list[str]) -> tuple[str, dict[str, int]]:
    normalized = [normalize_code(c) for c in codes if normalize_code(c)]
    if not normalized:
        return "", {}
    counts = Counter(normalized)
    winner, _ = counts.most_common(1)[0]
    return winner, dict(counts)


def score_prediction(predicted: str, labels: tuple[str, ...]) -> bool:
    return match_any_predicted(predicted, labels)


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    scored = [r for r in results if r.scorable]
    correct = sum(1 for r in scored if r.exact_match)
    n_scored = len(scored)
    return {
        "n_cases": len(results),
        "n_scored": n_scored,
        "n_skipped": len(results) - n_scored,
        "accuracy": (correct / n_scored) if n_scored else 0.0,
        "n_correct": correct,
    }


def run_baseline_case(
    encoder: LLMEncoder,
    prompt: str,
    *,
    max_new_tokens: int,
    enable_thinking: bool,
    taxonomy: dict | None = None,
) -> tuple[str, str]:
    result = run_zero_shot(
        encoder,
        prompt,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )
    resp, raw = _unwrap_zero_shot(result)
    return extract_code(resp, taxonomy=taxonomy), raw


def run_perturb_case(
    encoder: LLMEncoder,
    prompt: str,
    *,
    cal: dict[str, float | int],
    num_soft_tokens: int,
    n_seeds: int,
    seed_base: int,
    max_new_tokens: int,
    enable_thinking: bool,
    taxonomy: dict | None = None,
) -> tuple[str, list[str], list[str], dict[str, int]]:
    embed_dim = int(cal["embed_dim"])
    target_rms = float(cal["embedding_rms"])
    per_seed_codes: list[str] = []
    per_seed_raw: list[str] = []

    for seed_idx in range(n_seeds):
        sp = make_random_noise_soft_prompt(
            num_soft_tokens=num_soft_tokens,
            embed_dim=embed_dim,
            target_rms=target_rms,
            seed=seed_base + seed_idx,
        )
        raw_text, _ = decode_with_raw_soft_prompt(
            encoder,
            sp,
            prompt,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        per_seed_raw.append(raw_text)
        per_seed_codes.append(extract_code(raw_text, taxonomy=taxonomy))

    voted, counts = plurality_vote(per_seed_codes)
    return voted, per_seed_codes, per_seed_raw, counts


def evaluate_mode(
    *,
    mode: str,
    encoder: LLMEncoder,
    cal: dict[str, float | int],
    cases: list,
    args: argparse.Namespace,
    prior: dict[str, Any] | None = None,
) -> tuple[list[CaseResult], dict[str, Any]]:
    prior = prior or {}
    done_ids = set(prior.get("completed_case_ids", []))
    results: list[CaseResult] = list(prior.get("case_results", []))

    taxonomy = get_taxonomy(args.taxonomy)

    for idx, case in enumerate(cases):
        if case.case_id in done_ids:
            continue

        labels = case.labels
        scorable = is_scorable_label(primary_label(labels))
        report_text, raw_text = prepare_report_text(
            case.text, remove_codes=args.remove_codes,
        )
        user_prompt = format_user_prompt(report_text, taxonomy=taxonomy)

        t0 = time.time()
        meta: dict[str, Any] = {"mode": mode, "latency_s": 0.0}
        if args.remove_codes:
            meta["input_text"] = report_text
            meta["input_text_raw"] = raw_text

        if mode == "baseline":
            predicted, raw = run_baseline_case(
                encoder,
                user_prompt,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                taxonomy=taxonomy,
            )
            meta["raw_response"] = raw
        elif mode == "perturb":
            predicted, seed_codes, seed_raw, vote_counts = run_perturb_case(
                encoder,
                user_prompt,
                cal=cal,
                num_soft_tokens=args.num_soft_tokens,
                n_seeds=args.n_seeds,
                seed_base=args.seed_base,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                taxonomy=taxonomy,
            )
            meta["seed_codes"] = seed_codes
            meta["seed_raw"] = seed_raw
            meta["vote_counts"] = vote_counts
            meta["raw_response"] = seed_raw[0] if seed_raw else ""
        else:
            raise ValueError(f"unknown mode: {mode}")

        meta["latency_s"] = round(time.time() - t0, 2)
        hit = score_prediction(predicted, labels) if scorable else False

        result = CaseResult(
            case_id=case.case_id,
            scorable=scorable,
            labels=labels,
            predicted=predicted,
            exact_match=hit,
            raw_response=str(meta.get("raw_response", "")),
            metadata=meta,
        )
        results.append(result)
        done_ids.add(case.case_id)

        status = "SKIP" if not scorable else ("OK" if hit else "MISS")
        safe_print(
            f"[{mode}] {idx + 1}/{len(cases)} {case.case_id} "
            f"pred={predicted or '?'} gt={primary_label(labels)} [{status}]"
        )

        if args.checkpoint and (idx + 1) % args.checkpoint_every == 0:
            _save_checkpoint(args.checkpoint, mode, args, cal, results, done_ids)

    summary = summarize_results(results)
    summary["mode"] = mode
    return results, summary


def _save_checkpoint(
    path: Path,
    mode: str,
    args: argparse.Namespace,
    cal: dict,
    results: list[CaseResult],
    done_ids: set[str],
) -> None:
    payload = {
        "mode": mode,
        "model": args.model,
        "quantization": args.quantization,
        "calibration": cal,
        "completed_case_ids": sorted(done_ids),
        "case_results": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "case_results" in data:
        data["case_results"] = [
            CaseResult(
                case_id=r["case_id"],
                scorable=r["scorable"],
                labels=tuple(r["labels"]),
                predicted=r["predicted"],
                exact_match=r["exact_match"],
                raw_response=r.get("raw_response", ""),
                metadata=r.get("metadata", {}),
            )
            for r in data["case_results"]
        ]
    return data


def load_lora_adapter(encoder: LLMEncoder, adapter_path: str) -> None:
    from peft import PeftModel

    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {path}")
    print(f"Loading LoRA adapter: {path}")
    encoder.model = PeftModel.from_pretrained(encoder.model, str(path))
    encoder.model.eval()


def build_report(
    *,
    args: argparse.Namespace,
    cal: dict,
    summaries: dict[str, dict],
    results_by_mode: dict[str, list[CaseResult]],
    taxonomy: dict,
) -> dict[str, Any]:
    return {
        "benchmark": taxonomy["benchmark_name"],
        "taxonomy": args.taxonomy,
        "benchmark_csv": repo_relative(Path(args.benchmark)),
        "model": args.model,
        "adapter": args.adapter or None,
        "quantization": args.quantization,
        "num_soft_tokens": args.num_soft_tokens,
        "n_seeds": args.n_seeds,
        "enable_thinking": args.enable_thinking,
        "remove_codes": args.remove_codes,
        "calibration": cal,
        "summaries": summaries,
        "predictions": {
            mode: [asdict(r) for r in rows]
            for mode, rows in results_by_mode.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICD bench eval with LSR perturbation")
    parser.add_argument(
        "--taxonomy",
        default="icd11",
        choices=sorted(TAXONOMY),
        help="icd11 (oct20) or icd10 (Testdatensatz)",
    )
    parser.add_argument("--benchmark", default="", help="override benchmark CSV path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quantization", default=DEFAULT_QUANT, choices=["none", "4bit", "8bit"])
    parser.add_argument("--mode", default="both", choices=["baseline", "perturb", "both"])
    parser.add_argument("--n-seeds", type=int, default=10, help="perturbation seeds for plurality@K")
    parser.add_argument("--num-soft-tokens", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=2024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--remove-codes",
        action="store_true",
        help="strip ICD-10 / ICD-O / ICD-11 code mentions from report text before prompt",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all cases")
    parser.add_argument("--output", default="", help="JSON report path (taxonomy default if empty)")
    parser.add_argument("--checkpoint", default="", help="resume JSON path (defaults to output + .ckpt)")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--adapter",
        default="",
        help="PEFT LoRA dir from train_qwen_coding_lora.py (stage 1 or 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = get_taxonomy(args.taxonomy)
    if not args.benchmark:
        args.benchmark = str(taxonomy["default_benchmark"])
    if not args.output:
        args.output = str(taxonomy["default_output"])

    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else output_path.with_suffix(".ckpt.json")

    cases = taxonomy["load_cases"](benchmark_path)
    if args.limit > 0:
        cases = cases[: args.limit]

    n_scored = sum(1 for c in cases if is_scorable_label(primary_label(c.labels)))
    print(f"{args.taxonomy} bench: {len(cases)} cases ({n_scored} scored)")
    print(f"Model: {args.model} ({args.quantization})")
    print(f"Mode: {args.mode}")
    if args.remove_codes:
        print("Input: code mentions stripped from report text")

    quant = None if args.quantization == "none" else args.quantization
    print("\nLoading encoder...")
    encoder = LLMEncoder(model_name=args.model, quantization=quant)
    if args.adapter:
        load_lora_adapter(encoder, args.adapter)
    cal = auto_calibrate(encoder)
    print(
        f"Calibration: embed_dim={cal['embed_dim']}, "
        f"embedding_rms={cal['embedding_rms']:.5f}, device={encoder._device}"
    )

    modes = ["baseline", "perturb"] if args.mode == "both" else [args.mode]
    summaries: dict[str, dict] = {}
    results_by_mode: dict[str, list[CaseResult]] = {}

    for mode in modes:
        print(f"\n{'=' * 60}\nMODE: {mode}\n{'=' * 60}")
        prior = _load_checkpoint(ckpt_path) if ckpt_path.exists() else {}
        if prior.get("mode") not in (None, mode):
            prior = {}
        results, summary = evaluate_mode(
            mode=mode,
            encoder=encoder,
            cal=cal,
            cases=cases,
            args=args,
            prior=prior if prior.get("mode") == mode else None,
        )
        results_by_mode[mode] = results
        summaries[mode] = summary
        print(
            f"\n{mode}: accuracy={summary['accuracy']:.1%} "
            f"({summary['n_correct']}/{summary['n_scored']} scored)"
        )

    report = build_report(
        args=args,
        cal=cal,
        summaries=summaries,
        results_by_mode=results_by_mode,
        taxonomy=taxonomy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {output_path}")

    if ckpt_path.exists():
        ckpt_path.unlink()

    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
