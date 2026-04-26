#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5 Instruct on `qwen25_finetune_data.jsonl` (multi-turn JSON strokes).

Install (example):
  pip install torch transformers "trl>=0.16" peft accelerate datasets bitsandbytes
  (TRL 0.16+ uses SFTConfig(max_length=...); older TRL used max_seq_length — this script supports both.)

  Qwen2.5-7B + LoRA SFT (batch 1, grad accum 8, ~2k context, this script) — approx VRAM:
    --base_quant 8bit  (default on CUDA)  ≈ 16–22 GB
    --base_quant 4bit                    ≈ 12–18 GB  (tightest; still OK for ~20GB)
    --base_quant none  (bf16/fp16 base)   ≈ 24–40 GB  (often needs A100-40/48/80G or 2× GPU)
  OOM: lower --max_seq_length, or use --base_quant 4bit, or
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Run from repo root:
  PYTHONPATH=. python datagen_sft/train_qwen_lora.py

Or with a local JSONL path / output dir:
  PYTHONPATH=. python datagen_sft/train_qwen_lora.py \\
    --dataset datagen_sft/qwen25_finetune_data.jsonl \\
    --output_dir outputs/qwen25-handwriting-lora \\
    --model_name Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dotenv import load_dotenv

# Repo root on path (package learn_handwriting lives at root)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _dataset_stats(path: str) -> None:
    from collections import Counter

    tasks: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            md = row.get("metadata") or {}
            tasks[str(md.get("task", "?"))] += 1
            chars[str(md.get("target_character", "?"))] += 1
    print(f"Dataset rows: {n}")
    print(f"  By task: {dict(tasks)}")
    print(f"  Distinct letters: {len(chars)} (min/max count per letter: {min(chars.values())}/{max(chars.values())})")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="LoRA SFT for handwriting stroke policy (Qwen Instruct).")
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "qwen25_finetune_data.jsonl"),
        help="JSONL with `messages` (+ optional `metadata`) per line",
    )
    parser.add_argument(
        "--model_name",
        default=os.environ.get("FT_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        help="Base instruct model id (HF hub or local path)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.environ.get("FT_OUTPUT_DIR", os.path.join(_ROOT, "outputs", "qwen25-handwriting-lora")),
    )
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Token cap for SFT (longer = much more VRAM). 4096+ often OOMs 7B on 20–24GB without QLoRA.",
    )
    parser.add_argument(
        "--base_quant",
        choices=("8bit", "4bit", "none"),
        default=None,
        help="How the frozen base is stored: 8bit (default on CUDA; QLoRA + bitsandbytes), 4bit (lowest VRAM), "
        "none=full bf16/fp16 (highest quality headroom, needs a large GPU). MPS/CPU: forced to none.",
    )
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--merge_and_save",
        default=None,
        help="If set, path to save merged full weights after training (needs extra disk + RAM).",
    )
    parser.add_argument("--hub_model_id", default=None, help="Hugging Face repo ID to push to.")
    parser.add_argument("--hub_private", action="store_true", help="Whether to make the HF repo private.")
    parser.add_argument("--hub_revision", default="main", help="The branch/revision to push to.")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    _dataset_stats(args.dataset)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import inspect

    from trl import SFTTrainer, SFTConfig

    # TRL: max_seq_length on SFTConfig was renamed to max_length (0.16+) and the old name removed in 1.x
    _sft_init = inspect.signature(SFTConfig.__init__)
    if "max_length" in _sft_init.parameters:
        _max_len_kw = {"max_length": args.max_seq_length}
    elif "max_seq_length" in _sft_init.parameters:
        _max_len_kw = {"max_seq_length": args.max_seq_length}
    else:
        raise SystemExit(
            "SFTConfig has neither max_length nor max_seq_length; install a supported trl, e.g. pip install -U 'trl>=0.16'"
        )

    # --- Device Discovery ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n" + "="*40)
    print(f"DEVICE DISCOVERY")
    print(f"Detected device: {device.upper()}")
    if device == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  BF16 Support: {torch.cuda.is_bf16_supported()}")
    elif device == "mps":
        print(f"  Apple Silicon GPU detected (MPS)")
    else:
        print(f"  WARNING: No GPU (CUDA or MPS) detected. Training on CPU will be VERY slow.")
    print("="*40 + "\n")
    # ------------------------

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw = load_dataset("json", data_files=args.dataset, split="train")

    def add_text(batch: dict) -> dict:
        # `apply_chat_template` expects list of message dicts per row when batched
        texts = []
        for msgs in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    to_drop = [c for c in raw.column_names if c != "text"]
    ds = raw.map(
        add_text,
        batched=True,
        remove_columns=to_drop,
        desc="Formatting chat template",
    )

    if args.base_quant is None:
        args.base_quant = "8bit" if device == "cuda" else "none"
    if args.base_quant in ("4bit", "8bit") and device != "cuda":
        print("4/8-bit QLoRA needs CUDA; loading in bf16/fp32 instead (--base_quant none).")
        args.base_quant = "none"

    # Dtype selection (used for non-quantized loads and as quant compute dtype)
    if device == "cuda":
        torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    elif device == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if args.base_quant in ("4bit", "8bit"):
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "QLoRA (4/8-bit) needs bitsandbytes: pip install bitsandbytes  "
                "Or use full precision: --base_quant none (needs more VRAM)."
            ) from e
        if args.base_quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
            print("Loading base in 4-bit NF4 (QLoRA) — tightest VRAM")
        else:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Loading base in 8-bit (QLoRA) — default; LoRA still trained in fp16/bf16")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        print("Loading base in full bf16/fp16 (no quant) — use a 24–40GB+ class GPU for 7B + LoRA SFT")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = (device == "cuda" and not use_bf16) or (device == "mps")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        **_max_len_kw,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private,
        hub_token=os.environ.get("HF_TOKEN"),
        hub_strategy="end",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if args.hub_model_id:
        print(f"Pushing to Hugging Face: {args.hub_model_id} (revision: {args.hub_revision})")
        trainer.push_to_hub(revision=args.hub_revision)
        
    print(f"LoRA adapter saved to {args.output_dir}")

    if args.merge_and_save:
        from peft import PeftModel

        print("Merging LoRA into base weights…")
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        merged = PeftModel.from_pretrained(base, args.output_dir)
        merged = merged.merge_and_unload()
        os.makedirs(args.merge_and_save, exist_ok=True)
        merged.save_pretrained(args.merge_and_save)
        tokenizer.save_pretrained(args.merge_and_save)
        print(f"Merged model saved to {args.merge_and_save}")


if __name__ == "__main__":
    main()
