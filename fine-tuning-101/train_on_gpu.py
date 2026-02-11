#!/usr/bin/env python3
"""
Train Gemma 1B LoRA adapter for entity resolution on GPU.

Standalone script — zero dependency on shared/ modules. Downloads dataset
from HF Hub, trains with bfloat16 + LoRA, and pushes adapter to HF Hub.

Usage (RunPod A4000):
    python train_on_gpu.py
    python train_on_gpu.py --epochs 5 --lr 1e-4 --batch-size 8
    python train_on_gpu.py --no-push  # train without pushing adapter
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-3-1b-it"
DATASET_REPO = "abicyclerider/entity-resolution-pairs"
ADAPTER_REPO = "abicyclerider/gemma-1b-entity-resolution-lora"


def main():
    parser = argparse.ArgumentParser(description="Train Gemma 1B LoRA for entity resolution")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM, slower)")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing adapter to Hub")
    args = parser.parse_args()

    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        use_bf16 = torch.cuda.is_bf16_supported()
    elif torch.backends.mps.is_available():
        device = "mps"
        gpu_name = "Apple MPS"
        use_bf16 = False
        print(f"Device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        gpu_name = "CPU"
        use_bf16 = False
        print("Warning: No GPU detected, training will be slow")

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using dtype: {dtype}")

    # Load dataset from Hub
    print(f"\nLoading dataset from {DATASET_REPO}...")
    dataset = load_dataset(DATASET_REPO)
    print(f"  Train: {len(dataset['train'])}, Eval: {len(dataset['eval'])}, Test: {len(dataset['test'])}")

    # Load model and tokenizer
    print(f"\nLoading {MODEL_ID} in {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
    )
    print(f"Model loaded on: {model.device}")
    print(f"Parameters: {model.num_parameters():,}")

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config — GPU-optimized
    eff_batch = args.batch_size * args.grad_accum
    output_dir = "./output/entity-resolution-gpu"

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=False,
        dataloader_num_workers=2 if device == "cuda" else 0,
        dataloader_pin_memory=device == "cuda",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
    )

    steps_per_epoch = len(dataset["train"]) // eff_batch
    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} grad_accum = {eff_batch} effective")
    print(f"  Steps/epoch: ~{steps_per_epoch}, Total: ~{steps_per_epoch * args.epochs}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Learning rate: {args.lr}")
    print(f"  bf16: {use_bf16}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")

    # Train
    print(f"\nStarting training...")
    train_result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Final training loss: {train_result.training_loss:.4f}")

    # Save adapter locally
    adapter_path = f"{output_dir}/best-adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to: {adapter_path}")

    # Push to Hub
    if not args.no_push:
        print(f"\nPushing adapter to {ADAPTER_REPO} (private)...")
        model.push_to_hub(ADAPTER_REPO, private=True)
        tokenizer.push_to_hub(ADAPTER_REPO, private=True)
        print("Done! Adapter available on HF Hub.")
    else:
        print("\n--no-push specified, skipping upload.")


if __name__ == "__main__":
    main()
