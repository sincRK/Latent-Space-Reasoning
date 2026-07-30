# LSR × icd11_coding — medical coding track

This branch adds **ICD-10/ICD-11 benchmark eval and QLoRA training** on top of upstream Latent-Space-Reasoning. It **requires** the [`icd11_coding`](https://github.com/) repo for benchmarks, data paths, and shared scoring.

## Layout

```
icd11_coding/                          # main repo — pip install -e .
├── notebooks/data/                    # benches, catalogs, lab CSVs
├── icd11_coding/benchmark/            # EvalReport, loaders, metrics
├── nie_paper/                         # catalog_loader (SFT build)
├── scripts/run_icd11_lsr_pipe.sh      # thin wrapper → this repo
└── Latent-Space-Reasoning/            # **this repo**, branch icd11 (submodule)
    └── experiments/
        ├── run_icd11_lsr_eval.py      # baseline + perturb@10
        ├── run_icd11_lsr_pipe.sh
        ├── run_icd10_lsr_pipe.sh
        ├── build_icd11_sft_dataset.py
        ├── build_lab_corpus.py
        └── train_qwen_coding_lora.py
```

## One-time setup (from icd11_coding root)

```bash
conda activate icd11_coding
./scripts/setup_lsr.sh          # submodule init + pip install both packages
```

Or manually:

```bash
git submodule update --init --recursive Latent-Space-Reasoning
cd Latent-Space-Reasoning && git checkout icd11
pip install -e ".[quant,train]"
cd .. && pip install -e .
```

## Run

From **icd11_coding** (recommended):

```bash
./scripts/run_icd11_lsr_pipe.sh smoke
./scripts/run_icd10_lsr_pipe.sh smoke
```

From **Latent-Space-Reasoning** (paths default to parent icd11_coding):

```bash
./experiments/run_icd11_lsr_pipe.sh eval-pre
```

## Env overrides

| variable | default | purpose |
|---|---|---|
| `ICD11_CODING_ROOT` | parent of LSR | icd11_coding repo root if not nested |
| `CATALOG_CSV`, `RESULTS_DIR`, `CKPT_DIR` | under icd11_coding | pipe scripts |

## Upstream vs this branch

- **`main`** — general LSR (latent perturbation, sensitivity, etc.); no ICD deps
- **`icd11`** — ICD experiments only; imports `icd11_coding.benchmark`, `nie_paper.labgraph`

Do not merge `icd11` into upstream without making `icd11_coding` an optional extra; keep medical coding on this branch.

## Docs (icd11_coding)

- `ai-artifacts/docs/LSR_ICD11.md`
- `ai-artifacts/docs/LSR_ICD10.md`
- `ai-artifacts/docs/REPO_LAYOUT.md`
