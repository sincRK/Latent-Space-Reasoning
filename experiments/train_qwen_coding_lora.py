"""Two-stage QLoRA for ICD-11 coding track (LSR / LabGraph train parity).

Stage 1 (sft): multi-format catalog JSONL from build_icd11_sft_dataset.py
Stage 2 (dapt): unlabeled lab JSONL from build_lab_corpus.py

Usage (from Latent-Space-Reasoning/, CUDA + bitsandbytes required):
  python experiments/build_icd11_sft_dataset.py
  python experiments/train_qwen_coding_lora.py --stage sft \\
    --dataset ../data/icd11_catalog_sft.jsonl --output ../checkpoints/qwen_icd11_lora

  python experiments/build_lab_corpus.py
  python experiments/train_qwen_coding_lora.py --stage dapt \\
    --dataset ../data/lab_corpus.jsonl \\
    --resume ../checkpoints/qwen_icd11_lora --output ../checkpoints/qwen_icd11_lora_dapt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen3-4B"
SYSTEM_MSG = "Answer to the best of your ability."


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_chat(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    parts = []
    for msg in messages:
        parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
    return "\n\n".join(parts)


def build_sft_dataset(tokenizer, path: Path) -> Dataset:
    texts: list[str] = []
    for row in load_jsonl(path):
        messages = row.get("messages")
        if not messages:
            continue
        texts.append(format_chat(tokenizer, messages))
    return Dataset.from_dict({"text": texts})


def build_dapt_dataset(path: Path) -> Dataset:
    texts = [row["text"] for row in load_jsonl(path) if row.get("text")]
    return Dataset.from_dict({"text": texts})


def load_base_model(model_name: str, quant: str):
    bnb_config = None
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quant == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    kwargs = {"trust_remote_code": True}
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def attach_lora(model, *, resume: str | None):
    if resume:
        print(f"Loading adapter from {resume}")
        return PeftModel.from_pretrained(model, resume, is_trainable=True)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lora)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA train for ICD-11 LSR track")
    p.add_argument("--stage", required=True, choices=["sft", "dapt"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--quantization", default="8bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--dataset", required=True, help="JSONL from build_* scripts")
    p.add_argument("--output", required=True, help="adapter output dir")
    p.add_argument("--resume", default="", help="existing LoRA dir (required for dapt stage 2)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant = None if args.quantization == "none" else args.quantization
    model, tokenizer = load_base_model(args.model, quant or "none")
    model = attach_lora(model, resume=args.resume or None)
    model.print_trainable_parameters()

    if args.stage == "sft":
        dataset = build_sft_dataset(tokenizer, dataset_path)
    else:
        if not args.resume:
            raise ValueError("dapt stage requires --resume pointing to stage-1 adapter")
        dataset = build_dapt_dataset(dataset_path)

    print(f"Stage {args.stage}: {len(dataset)} training rows from {dataset_path.name}")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        max_steps=args.max_steps,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"✓ Saved adapter → {output_dir}")


if __name__ == "__main__":
    main()
