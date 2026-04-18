"""Train the Hebrew nikud classifier model.

Example:
    uv run src/train.py \
        --train-dataset dataset/train_nikud.txt \
        --eval-dataset  dataset/eval_nikud.txt \
        --output-dir    outputs/nikud-classifier

Training data format: one sentence per line, vocalized Hebrew (with nikud).
TSV files are also accepted; only the first column is used.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from jiwer import cer, wer

from constants import (
    IGNORE_INDEX,
    ID_TO_NIKUD,
    ID_TO_SHIN,
    MAT_LECT_TOKEN,
    SHIN_LETTER,
    MAX_LEN,
    is_hebrew_letter,
    strip_nikud,
)
from align_data import extract_nikud
from infer import _decode
from model import HebrewNikudClassifier
from tokenization import load_encoder_tokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NikudDataset(Dataset):
    """
    Lazy dataset that reads vocalized Hebrew text (one sentence per line).
    TSV files use only the first column.
    """

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(self.lines):,} lines from {path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokenizer = load_encoder_tokenizer()
        vocalized = self.lines[idx].split("\t", 1)[0]
        bare = strip_nikud(vocalized)

        pairs = extract_nikud(vocalized)
        if not pairs:
            return self[idx + 1]

        from prepare_tokens import build_token_labels
        record = build_token_labels(bare, pairs, vocalized, tokenizer)
        if record is None:
            return self[idx + 1]

        return {
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "nikud_labels": record["nikud_labels"],
            "shin_labels": record["shin_labels"],
            "text": bare,
            "ref": vocalized,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class NikudDataCollator:
    pad_id: int = 0
    ignore_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask = [], []
        nikud_labels, shin_labels = [], []
        texts, refs = [], []

        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            nikud_labels.append(list(f["nikud_labels"]) + [self.ignore_id] * pad)
            shin_labels.append(list(f["shin_labels"]) + [self.ignore_id] * pad)
            texts.append(f.get("text", ""))
            refs.append(f.get("ref", ""))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "nikud_labels": torch.tensor(nikud_labels, dtype=torch.long),
            "shin_labels": torch.tensor(shin_labels, dtype=torch.long),
            "texts": texts,
            "refs": refs,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Hebrew nikud classifier model")
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
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--init-from-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model, output_dir: Path, step: int, metrics: dict, save_total_limit: int):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, **metrics}))
    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.name != "checkpoint-best"],
        key=lambda p: int(p.name.split("-")[1]),
    )
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def _decode_labels(
    text: str,
    offset_mapping: list,
    nikud_labels: list[int],
    shin_labels: list[int],
) -> str:
    """Reconstruct vocalized Hebrew from ground-truth label IDs."""
    result: list[str] = []
    prev_char_end = 0

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start > prev_char_end:
            result.append(text[prev_char_end:start])
        if end - start != 1:
            if end > start:
                prev_char_end = end
            continue
        char = text[start:end]
        prev_char_end = end
        if not is_hebrew_letter(char):
            result.append(char)
            continue

        if char == SHIN_LETTER:
            sl = int(shin_labels[tok_idx]) if tok_idx < len(shin_labels) else IGNORE_INDEX
            shin_mark = ID_TO_SHIN.get(sl, "") if sl != IGNORE_INDEX else ""
            result.append(char + shin_mark)
        else:
            result.append(char)

        nl = int(nikud_labels[tok_idx]) if tok_idx < len(nikud_labels) else IGNORE_INDEX
        if nl != IGNORE_INDEX:
            nikud = ID_TO_NIKUD.get(nl, "")
            if nikud and nikud != MAT_LECT_TOKEN:
                result.append(nikud)

    if prev_char_end < len(text):
        result.append(text[prev_char_end:])
    return "".join(result)


def evaluate(model, eval_loader, device, fp16: bool, tokenizer) -> dict:
    model.eval()
    total_loss = 0.0
    total_nikud_acc = 0.0
    total_shin_acc = 0.0
    refs_all, hyps_all = [], []
    n_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            texts = batch.pop("texts")
            refs = batch.pop("refs")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch)
            total_loss += out["loss"].item()
            total_nikud_acc += compute_accuracy(out["nikud_logits"], batch["nikud_labels"])
            total_shin_acc += compute_accuracy(out["shin_logits"], batch["shin_labels"])
            n_batches += 1

            n_labels = batch["nikud_labels"].cpu().tolist()
            s_labels = batch["shin_labels"].cpu().tolist()

            for i, text in enumerate(texts):
                enc = tokenizer(text, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
                offset_mapping = enc["offset_mapping"]
                ref = _decode_labels(text, offset_mapping, n_labels[i], s_labels[i])
                hyp = _decode(
                    text=text,
                    offset_mapping=offset_mapping,
                    nikud_logits=out["nikud_logits"][i],
                    shin_logits=out["shin_logits"][i],
                )
                refs_all.append(ref)
                hyps_all.append(hyp)

    model.train()
    mean_wer = sum(wer(r, h) for r, h in zip(refs_all, hyps_all)) / max(len(refs_all), 1)
    mean_cer = sum(cer(r, h) for r, h in zip(refs_all, hyps_all)) / max(len(refs_all), 1)
    return {
        "eval_loss": total_loss / n_batches,
        "nikud_acc": total_nikud_acc / n_batches,
        "shin_acc": total_shin_acc / n_batches,
        "cer": mean_cer,
        "wer": mean_wer,
        "acc": 1 - mean_wer,
        "refs": refs_all,
        "hyps": hyps_all,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="hebrew-nikud-classifier", config=vars(args), mode=args.wandb_mode)

    encoder_tokenizer = load_encoder_tokenizer()
    train_dataset = NikudDataset(args.train_dataset)
    eval_dataset = NikudDataset(args.eval_dataset)

    collator = NikudDataCollator()
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator, num_workers=4, pin_memory=pin_memory)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=pin_memory)

    model = HebrewNikudClassifier().to(device)

    if args.init_from_checkpoint:
        from safetensors.torch import load_file
        state = load_file(str(Path(args.init_from_checkpoint) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {args.init_from_checkpoint}")

    if args.freeze_encoder_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print("Encoder frozen.")

    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.encoder_lr, args.head_lr, args.weight_decay)
    )

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    if args.max_steps > 0:
        total_opt_steps = min(total_opt_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, args.warmup_steps, total_opt_steps),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    global_step = 0
    opt_step = 0
    optimizer.zero_grad()
    best_wer = float("inf")
    no_improve_count = 0
    stop_training = False

    for epoch in range(math.ceil(args.epochs)):
        if stop_training:
            break
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True)

        for batch in pbar:
            if opt_step >= total_opt_steps:
                break

            if args.freeze_encoder_steps > 0 and global_step == args.freeze_encoder_steps:
                for p in model.encoder.parameters():
                    p.requires_grad_(True)
                print(f"\n[step {opt_step}] Encoder unfrozen.")

            batch.pop("texts", None)
            batch.pop("refs", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=args.fp16):
                out = model(**batch)

            scaled_loss = out["loss"] / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            epoch_loss_sum += out["loss"].item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
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

                if opt_step % args.save_steps == 0:
                    metrics = evaluate(model, eval_loader, device, args.fp16, encoder_tokenizer)
                    print(f"\n[step {opt_step}] nikud_acc: {metrics['nikud_acc']:.4f}  shin_acc: {metrics['shin_acc']:.4f}  CER: {metrics['cer']:.4f}  WER: {metrics['wer']:.4f}  loss: {metrics['eval_loss']:.4f}")
                    for i, (ref, hyp) in enumerate(zip(metrics["refs"][:3], metrics["hyps"][:3]), 1):
                        print(f"  {i}. GT:   {ref}")
                        print(f"     Pred: {hyp}")
                    save_checkpoint(model, output_dir, opt_step, {k: v for k, v in metrics.items() if k not in ("refs", "hyps")}, args.save_total_limit)
                    if metrics["wer"] < best_wer:
                        best_wer = metrics["wer"]
                        no_improve_count = 0
                        best_ckpt_dir = output_dir / "checkpoint-best"
                        best_ckpt_dir.mkdir(parents=True, exist_ok=True)
                        from safetensors.torch import save_file
                        save_file(model.state_dict(), str(best_ckpt_dir / "model.safetensors"))
                        (best_ckpt_dir / "train_state.json").write_text(json.dumps({"step": opt_step, **{k: v for k, v in metrics.items() if k not in ("refs", "hyps")}}))
                        print(f"  [checkpoint-best updated at step {opt_step}]")
                    else:
                        no_improve_count += 1
                        print(f"  [patience: {no_improve_count}/{args.early_stopping_patience}]")
                        if no_improve_count >= args.early_stopping_patience:
                            print(f"[step {opt_step}] Early stopping triggered.")
                            stop_training = True
                            break

    metrics = evaluate(model, eval_loader, device, args.fp16, encoder_tokenizer)
    print(f"\nFinal: nikud_acc: {metrics['nikud_acc']:.4f}  shin_acc: {metrics['shin_acc']:.4f}  CER: {metrics['cer']:.4f}  WER: {metrics['wer']:.4f}  loss: {metrics['eval_loss']:.4f}")
    for i, (ref, hyp) in enumerate(zip(metrics["refs"][:3], metrics["hyps"][:3]), 1):
        print(f"  {i}. GT:   {ref}")
        print(f"     Pred: {hyp}")
    save_checkpoint(model, output_dir, opt_step, {k: v for k, v in metrics.items() if k not in ("refs", "hyps")}, args.save_total_limit)


if __name__ == "__main__":
    main()
