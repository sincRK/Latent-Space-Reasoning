#!/usr/bin/env bash
# LSR ICD-11 track — full pipeline overview
#
# Prereqs (run once, outside this script):
#   conda activate icd11_coding
#   cd Latent-Space-Reasoning && pip install -e ".[quant,train]"
#   cd .. && pip install -e .
#
# Usage:
#   ./experiments/run_icd11_lsr_pipe.sh              # print stages
#   ./experiments/run_icd11_lsr_pipe.sh all          # eval-pre → build → train → eval-post
#   ./experiments/run_icd11_lsr_pipe.sh eval-pre     # baseline + perturb@10 (pre-train)
#   ./experiments/run_icd11_lsr_pipe.sh build-data   # SFT + lab JSONL
#   ./experiments/run_icd11_lsr_pipe.sh train          # QLoRA stage 1 + 2
#   ./experiments/run_icd11_lsr_pipe.sh eval-post      # baseline + perturb@10 (post-train)
#   ./experiments/run_icd11_lsr_pipe.sh smoke          # 3-case baseline sanity check
#
# Override paths via env, e.g.:
#   RESULTS_DIR=/tmp/lsr_results ./experiments/run_icd11_lsr_pipe.sh eval-pre

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LSR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${ICD11_CODING_ROOT:-$(cd "${LSR_ROOT}/.." && pwd)}"

cd "${LSR_ROOT}"

# --- config (override with env) ---
MODEL="${MODEL:-Qwen/Qwen3-4B}"
QUANT="${QUANT:-8bit}"
N_SEEDS="${N_SEEDS:-10}"
CATALOG_CSV="${CATALOG_CSV:-${REPO_ROOT}/notebooks/data/category_strings_oct2_with_definitions.csv}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
SFT_JSONL="${SFT_JSONL:-${DATA_DIR}/icd11_catalog_sft.jsonl}"
LAB_JSONL="${LAB_JSONL:-${DATA_DIR}/lab_corpus.jsonl}"
ADAPTER_SFT="${ADAPTER_SFT:-${CKPT_DIR}/qwen_icd11_lora}"
ADAPTER_DAPT="${ADAPTER_DAPT:-${CKPT_DIR}/qwen_icd11_lora_dapt}"

step() { echo ""; echo "=== [$*] ==="; }

stage_smoke() {
  step "smoke: baseline × 3 cases"
  python experiments/run_icd11_lsr_eval.py \
    --model "${MODEL}" \
    --quantization "${QUANT}" \
    --mode baseline \
    --limit 3 \
    --output "${RESULTS_DIR}/icd11_lsr_smoke.json"
}

stage_eval_pre() {
  mkdir -p "${RESULTS_DIR}"
  step "eval-pre: baseline (zero-shot, 8-bit)"
  python experiments/run_icd11_lsr_eval.py \
    --model "${MODEL}" \
    --quantization "${QUANT}" \
    --mode baseline \
    --output "${RESULTS_DIR}/icd11_lsr_baseline.json"

  step "eval-pre: perturb@${N_SEEDS} + plurality vote"
  python experiments/run_icd11_lsr_eval.py \
    --model "${MODEL}" \
    --quantization "${QUANT}" \
    --mode perturb \
    --n-seeds "${N_SEEDS}" \
    --output "${RESULTS_DIR}/icd11_lsr_perturb${N_SEEDS}.json"

  step "eval-pre: both modes → single JSON"
  python experiments/run_icd11_lsr_eval.py \
    --model "${MODEL}" \
    --quantization "${QUANT}" \
    --mode both \
    --n-seeds "${N_SEEDS}" \
    --output "${RESULTS_DIR}/icd11_lsr_qwen3_4b_8bit.json"
}

stage_build_data() {
  mkdir -p "${DATA_DIR}"
  step "build-data: ICD-11 catalog multi-format SFT JSONL"
  python experiments/build_icd11_sft_dataset.py \
    --csv "${CATALOG_CSV}" \
    --output "${SFT_JSONL}"

  step "build-data: unlabeled lab Befundstexte JSONL"
  python experiments/build_lab_corpus.py \
    --output "${LAB_JSONL}"
}

stage_train() {
  mkdir -p "${CKPT_DIR}"
  step "train stage 1: catalog SFT (QLoRA)"
  python experiments/train_qwen_coding_lora.py \
    --stage sft \
    --model "${MODEL}" \
    --quantization "${QUANT}" \
    --dataset "${SFT_JSONL}" \
    --output "${ADAPTER_SFT}"

  step "train stage 2: lab Befund DAPT (QLoRA, resume stage 1)"
  python experiments/train_qwen_coding_lora.py \
    --stage dapt \
    --model "${MODEL}" \
    --quantization "${QUANT}" \
    --dataset "${LAB_JSONL}" \
    --resume "${ADAPTER_SFT}" \
    --output "${ADAPTER_DAPT}"
}

stage_eval_post() {
  mkdir -p "${RESULTS_DIR}"
  step "eval-post: baseline + perturb@${N_SEEDS} with finetuned adapter"
  python experiments/run_icd11_lsr_eval.py \
    --model "${MODEL}" \
    --quantization "${QUANT}" \
    --mode both \
    --n-seeds "${N_SEEDS}" \
    --adapter "${ADAPTER_DAPT}" \
    --output "${RESULTS_DIR}/icd11_lsr_finetuned.json"
}

print_overview() {
  cat <<EOF
LSR ICD-11 pipeline (from ${LSR_ROOT})

  smoke       → baseline, 3 cases
  eval-pre    → baseline + perturb@${N_SEEDS} (pre-train)
  build-data  → ${SFT_JSONL} + ${LAB_JSONL}
  train       → ${ADAPTER_SFT} → ${ADAPTER_DAPT}
  eval-post   → baseline + perturb@${N_SEEDS} + adapter
  all         → eval-pre → build-data → train → eval-post

Install (once, outside pipe):
  conda activate icd11_coding
  cd Latent-Space-Reasoning && pip install -e ".[quant,train]"
  cd .. && pip install -e .

EOF
}

CMD="${1:-overview}"
case "${CMD}" in
  overview|help|-h|--help) print_overview ;;
  smoke)      stage_smoke ;;
  eval-pre)   stage_eval_pre ;;
  build-data) stage_build_data ;;
  train)      stage_train ;;
  eval-post)  stage_eval_post ;;
  all)
    stage_eval_pre
    stage_build_data
    stage_train
    stage_eval_post
    step "done"
    echo "Results: ${RESULTS_DIR}/"
    echo "Adapter:  ${ADAPTER_DAPT}"
    ;;
  *)
    echo "Unknown stage: ${CMD}" >&2
    print_overview
    exit 1
    ;;
esac
