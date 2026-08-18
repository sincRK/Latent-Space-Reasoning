#!/usr/bin/env bash
# ICD-10 LSR track — Qwen3-4B baseline + perturb@10 on Testdatensatz.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LSR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${ICD11_CODING_ROOT:-$(cd "${LSR_ROOT}/.." && pwd)}"

cd "${LSR_ROOT}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
QUANT="${QUANT:-8bit}"
N_SEEDS="${N_SEEDS:-10}"
CATALOG_CSV="${CATALOG_CSV:-${REPO_ROOT}/notebooks/data/icd10_category_strings.csv}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints_icd10}"
SFT_JSONL="${SFT_JSONL:-${DATA_DIR}/icd10_catalog_sft.jsonl}"
LAB_JSONL="${LAB_JSONL:-${DATA_DIR}/lab_corpus.jsonl}"
ADAPTER_SFT="${ADAPTER_SFT:-${CKPT_DIR}/qwen_icd10_lora}"
ADAPTER_DAPT="${ADAPTER_DAPT:-${CKPT_DIR}/qwen_icd10_lora_dapt}"
BENCH_CSV="${BENCH_CSV:-${REPO_ROOT}/notebooks/data/icd10_benchmark.csv}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
bench_args() {
  if [[ -f "${BENCH_CSV}" ]]; then
    printf '%s\n' --benchmark "${BENCH_CSV}"
  fi
}
step() { echo ""; echo "=== [$*] ==="; }
stage_smoke() {
  python experiments/run_icd11_lsr_eval.py --taxonomy icd10 --mode baseline --limit 3 \
    $(bench_args) \
    --output "${RESULTS_DIR}/icd10_lsr_smoke${OUTPUT_SUFFIX}.json"
}
stage_eval_pre() {
  mkdir -p "${RESULTS_DIR}"
  python experiments/run_icd11_lsr_eval.py --taxonomy icd10 --mode both --n-seeds "${N_SEEDS}" \
    $(bench_args) \
    --output "${RESULTS_DIR}/icd10_lsr_qwen3_4b_8bit${OUTPUT_SUFFIX}.json"
}
stage_build_data() {
  mkdir -p "${DATA_DIR}"
  python experiments/build_icd11_sft_dataset.py --taxonomy icd10 --csv "${CATALOG_CSV}" --output "${SFT_JSONL}"
  python experiments/build_lab_corpus.py --output "${LAB_JSONL}"
}
stage_train() {
  python experiments/train_qwen_coding_lora.py --stage sft --dataset "${SFT_JSONL}" --output "${ADAPTER_SFT}"
  python experiments/train_qwen_coding_lora.py --stage dapt --dataset "${LAB_JSONL}" \
    --resume "${ADAPTER_SFT}" --output "${ADAPTER_DAPT}"
}
stage_eval_post() {
  python experiments/run_icd11_lsr_eval.py --taxonomy icd10 --mode both --n-seeds "${N_SEEDS}" \
    $(bench_args) \
    --adapter "${ADAPTER_DAPT}" --output "${RESULTS_DIR}/icd10_lsr_finetuned${OUTPUT_SUFFIX}.json"
}
STAGE="${1:-}"
case "${STAGE}" in
  smoke) stage_smoke ;;
  eval-pre) stage_eval_pre ;;
  build-data) stage_build_data ;;
  train) stage_train ;;
  eval-post) stage_eval_post ;;
  all) stage_eval_pre; stage_build_data; stage_train; stage_eval_post ;;
  ""|-h|--help)
    echo "Usage: $0 [smoke|eval-pre|build-data|train|eval-post|all]" ;;
  *) echo "Unknown: ${STAGE}" >&2; exit 1 ;;
esac
