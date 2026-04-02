"""Train the Hebrew G2P classifier model.

Example:
    uv run src/train.py \
        --train-dataset dataset/knesset_vox_new_asr_split.tsv \
        --eval-dataset dataset/gt_alignment.jsonl \
        --output-dir outputs/g2p-classifier

Multi-GPU:
    accelerate launch src/train.py \
        --train-dataset dataset/knesset_vox_new_asr_split.tsv \
        --eval-dataset dataset/gt_alignment.jsonl \
        --output-dir outputs/g2p-classifier
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from checkpoint import save_checkpoint
from constants import IGNORE_INDEX, MAX_LEN
from data_align import align_sentence, strip_nikud
from data_tokenize import process_sentence
from eval import evaluate
from model import G2PModel
from optimizer import cosine_lr_lambda, parameter_groups
from constants import TOKENIZER_PATH
from tokenization import load_tokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AlignmentDataset(Dataset):
    """
    Lazy dataset that works with two formats:
      - JSONL  (.jsonl): pre-aligned, one JSON object per line
      - Raw TSV (.tsv/.txt): hebrew_with_nikud<TAB>ipa — aligned on-the-fly
    """

    def __init__(self, path: str):
        self.is_jsonl = path.endswith(".jsonl")
        self.tokenizer = load_tokenizer(TOKENIZER_PATH)
        with open(path, encoding="utf-8") as f:
            raw = [l for l in f.readlines() if l.strip()]
        if not self.is_jsonl:
            # Pre-filter: keep only lines that have Hebrew characters (skip headers/empty)
            raw = [l for l in raw if any("\u05d0" <= c <= "\u05ea" for c in l.split("\t")[0])]
        self.lines = raw
        print(f"Loaded {len(self.lines):,} lines from {path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        line = self.lines[idx]

        if self.is_jsonl:
            obj = json.loads(line)
            hebrew, alignment = next(iter(obj.items()))
            ref_ipa = "".join(chunk for _, chunk in alignment)
        else:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                return self[idx + 1]  # skip malformed
            hebrew = strip_nikud(parts[0])
            ref_ipa = parts[1].strip()
            alignment = align_sentence(hebrew, ref_ipa)
            if alignment is None:
                return self[idx + 1]  # skip failed alignment

        record = process_sentence(hebrew, alignment, tokenizer)
        if record is None:
            return self[idx + 1]
        record["text"] = hebrew
        record["ref_ipa"] = ref_ipa
        return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Hebrew G2P classifier model")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--eval-dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-encoder-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after this many optimizer steps (-1 = no limit)")
    parser.add_argument("--init-from-checkpoint", type=str, default=None)
    parser.add_argument("--init-weights-only", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--flash-attention", action="store_true", default=False)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class ClassifierDataCollator:
    """Pad classifier dataset features to the same length within a batch."""

    pad_id: int = 0
    ignore_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask = [], []
        consonant_labels, vowel_labels, stress_labels = [], [], []

        texts, ref_ipas = [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            consonant_labels.append(list(f["consonant_labels"]) + [self.ignore_id] * pad)
            vowel_labels.append(list(f["vowel_labels"]) + [self.ignore_id] * pad)
            stress_labels.append(list(f["stress_labels"]) + [self.ignore_id] * pad)
            texts.append(f.get("text", ""))
            ref_ipas.append(f.get("ref_ipa", ""))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "consonant_labels": torch.tensor(consonant_labels, dtype=torch.long),
            "vowel_labels": torch.tensor(vowel_labels, dtype=torch.long),
            "stress_labels": torch.tensor(stress_labels, dtype=torch.long),
            "texts": texts,
            "ref_ipas": ref_ipas,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project="hebrew-g2p-classifier", config=vars(args), mode=args.wandb_mode)

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    train_dataset = AlignmentDataset(args.train_dataset)
    eval_dataset = AlignmentDataset(args.eval_dataset)

    collator = ClassifierDataCollator()
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator, num_workers=4, pin_memory=pin_memory)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=pin_memory)

    model = G2PModel(flash_attention=args.flash_attention)

    if args.init_from_checkpoint:
        from safetensors.torch import load_file
        state = load_file(str(Path(args.init_from_checkpoint) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.init_from_checkpoint}")

    if args.freeze_encoder_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        if accelerator.is_main_process:
            print("Encoder frozen.")

    optimizer = torch.optim.AdamW(
        parameter_groups(model, args.encoder_lr, args.head_lr, args.weight_decay)
    )

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    if args.max_steps > 0:
        total_opt_steps = min(total_opt_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, args.warmup_steps, total_opt_steps),
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    opt_step = 0
    if args.init_from_checkpoint and not args.init_weights_only:
        state_path = Path(args.init_from_checkpoint) / "train_state.json"
        if state_path.exists():
            saved = json.loads(state_path.read_text())
            opt_step = saved["step"]
            for _ in range(opt_step):
                scheduler.step()
            if accelerator.is_main_process:
                print(f"Resumed from step {opt_step}")

    global_step = opt_step * args.gradient_accumulation_steps
    optimizer.zero_grad()
    best_wer = float("inf")
    no_improve_count = 0
    stop_training = False

    for epoch in range(math.ceil(args.epochs)):
        if stop_training:
            break
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True, disable=not accelerator.is_main_process)

        for batch in pbar:
            if opt_step >= total_opt_steps:
                break

            if args.freeze_encoder_steps > 0 and global_step == args.freeze_encoder_steps:
                for p in accelerator.unwrap_model(model).encoder.parameters():
                    p.requires_grad_(True)
                if accelerator.is_main_process:
                    print(f"\n[step {opt_step}] Encoder unfrozen.")

            batch.pop("texts", None)
            batch.pop("ref_ipas", None)
            with accelerator.autocast():
                out = model(**batch)

            scaled_loss = out["loss"] / args.gradient_accumulation_steps
            accelerator.backward(scaled_loss)
            epoch_loss_sum += out["loss"].item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = epoch_loss_sum / epoch_steps
                pbar.set_postfix(
                    step=opt_step,
                    loss=f"{train_loss:.4f}",
                    enc_lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    head_lr=f"{optimizer.param_groups[2]['lr']:.2e}",
                )

                if accelerator.is_main_process:
                    if opt_step % args.logging_steps == 0:
                        wandb.log({
                            "train_loss": train_loss,
                            "lr_encoder": optimizer.param_groups[0]["lr"],
                            "lr_head": optimizer.param_groups[2]["lr"],
                            "epoch": epoch,
                        }, step=opt_step)

                    if opt_step % args.save_steps == 0:
                        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16, tokenizer, MAX_LEN)
                        wandb.log({k: v for k, v in metrics.items() if k not in ("refs", "hyps")}, step=opt_step)
                        print(f"\n[step {opt_step}] acc={metrics.get('acc', float('nan')):.4f}  wer={metrics.get('wer', float('nan')):.4f}  cer={metrics.get('cer', float('nan')):.4f}  consonant={metrics['consonant_acc']:.4f}  vowel={metrics['vowel_acc']:.4f}  stress={metrics['stress_acc']:.4f}  loss={metrics['eval_loss']:.4f}")
                        if "refs" in metrics:
                            for i, (ref, hyp) in enumerate(zip(metrics["refs"][:3], metrics["hyps"][:3]), 1):
                                print(f"  {i}. GT:   {ref}")
                                print(f"     Pred: {hyp}")
                        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics.get("acc", metrics["mean_acc"]), args.save_total_limit)
                        if metrics.get("wer", float("inf")) < best_wer:
                            best_wer = metrics["wer"]
                            no_improve_count = 0
                            best_ckpt_dir = output_dir / "checkpoint-best"
                            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
                            from safetensors.torch import save_file
                            save_file(accelerator.unwrap_model(model).state_dict(), str(best_ckpt_dir / "model.safetensors"))
                            (best_ckpt_dir / "train_state.json").write_text(json.dumps({"step": opt_step, **metrics}))
                            print(f"  [checkpoint-best updated at step {opt_step}] [patience: {no_improve_count}/{args.early_stopping_patience}]")
                        else:
                            no_improve_count += 1
                            print(f"  [patience: {no_improve_count}/{args.early_stopping_patience}]")
                            if no_improve_count >= args.early_stopping_patience:
                                print(f"[step {opt_step}] Early stopping triggered (best wer={best_wer:.4f})")
                                stop_training = True
                                break

    if accelerator.is_main_process:
        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16, tokenizer, MAX_LEN)
        wandb.log({k: v for k, v in metrics.items() if k not in ("refs", "hyps")})
        print(f"\nFinal: consonant_acc={metrics['consonant_acc']:.4f}  vowel_acc={metrics['vowel_acc']:.4f}  stress_acc={metrics['stress_acc']:.4f}  mean_acc={metrics['mean_acc']:.4f}  wer={metrics.get('wer', float('nan')):.4f}  cer={metrics.get('cer', float('nan')):.4f}")
        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["mean_acc"], args.save_total_limit)
        wandb.finish()


if __name__ == "__main__":
    main()
