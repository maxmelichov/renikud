"""Train the Hebrew G2P classifier model.

Example:
    uv run src/train.py \
        --train-dataset dataset/.cache/classifier-train \
        --eval-dataset dataset/.cache/classifier-val \
        --output-dir outputs/g2p-classifier
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from jiwer import cer, wer
from constants import IGNORE_INDEX, ID_TO_CONSONANT, ID_TO_VOWEL, STRESS_YES, is_hebrew_letter, MAX_LEN
from infer import _decode, build_tokenizer_vocab
from model import HebrewG2PClassifier
from tokenization import load_encoder_tokenizer


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
    parser.add_argument("--wandb-mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--early-stopping-patience", type=int, default=40, help="Stop if metric does not improve for this many eval intervals (40 × 500 steps = 20K steps)")
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

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask = [], []
        consonant_labels, vowel_labels, stress_labels = [], [], []

        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            consonant_labels.append(list(f["consonant_labels"]) + [self.ignore_id] * pad)
            vowel_labels.append(list(f["vowel_labels"]) + [self.ignore_id] * pad)
            stress_labels.append(list(f["stress_labels"]) + [self.ignore_id] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "consonant_labels": torch.tensor(consonant_labels, dtype=torch.long),
            "vowel_labels": torch.tensor(vowel_labels, dtype=torch.long),
            "stress_labels": torch.tensor(stress_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model, output_dir: Path, step: int, acc: float, save_total_limit: int):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "acc": acc}))
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Per-token accuracy ignoring IGNORE_INDEX positions."""
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def _decode_labels(text: str, offset_mapping: list, consonant_labels, vowel_labels, stress_labels) -> str:
    """Decode per-token label IDs into an IPA string using the same format as infer._decode."""
    result = []
    prev_char_end = 0
    label_idx = 0
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
            if not (char == "'" and start > 0 and text[start - 1] in "גזצץ"):
                result.append(char)
            continue
        if tok_idx >= len(consonant_labels):
            break
        c = int(consonant_labels[tok_idx])
        v = int(vowel_labels[tok_idx])
        s = int(stress_labels[tok_idx])
        if c == IGNORE_INDEX:
            continue
        consonant = ID_TO_CONSONANT.get(c, "∅")
        vowel = ID_TO_VOWEL.get(v, "∅")
        chunk = ""
        if consonant != "∅":
            chunk += consonant
        if s == STRESS_YES:
            chunk += "ˈ"
        if vowel != "∅":
            chunk += vowel
        result.append(chunk)
    if prev_char_end < len(text):
        result.append(text[prev_char_end:])
    return "".join(result)


def evaluate(model, eval_loader, device, fp16: bool, tokenizer, max_len: int) -> dict:
    model.eval()
    total_loss = 0.0
    refs, hyps = [], []
    vocab = build_tokenizer_vocab(tokenizer)

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch)
            total_loss += out["loss"].item()

            texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            c_labels = batch["consonant_labels"].cpu().tolist()
            v_labels = batch["vowel_labels"].cpu().tolist()
            s_labels = batch["stress_labels"].cpu().tolist()

            for i, text in enumerate(texts):
                seq_len = int(batch["attention_mask"][i].sum().item())
                enc = tokenizer(text, truncation=True, max_length=seq_len, return_offsets_mapping=True)
                offset_mapping = enc["offset_mapping"]
                refs.append(_decode_labels(text, offset_mapping, c_labels[i], v_labels[i], s_labels[i]))
                hyps.append(_decode(
                    text=text,
                    offset_mapping=offset_mapping,
                    consonant_logits=out["consonant_logits"][i],
                    vowel_logits=out["vowel_logits"][i],
                    stress_logits=out["stress_logits"][i],
                ))

    model.train()
    mean_wer = sum(wer(r, h) for r, h in zip(refs, hyps)) / len(refs)
    mean_cer = sum(cer(r, h) for r, h in zip(refs, hyps)) / len(refs)
    return {
        "eval_loss": total_loss / len(eval_loader),
        "cer": mean_cer,
        "wer": mean_wer,
        "acc": 1 - mean_wer,
        "refs": refs,
        "hyps": hyps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="hebrew-g2p-classifier", config=vars(args), mode=args.wandb_mode)

    train_dataset = load_from_disk(args.train_dataset)
    eval_dataset = load_from_disk(args.eval_dataset)

    collator = ClassifierDataCollator()
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)

    encoder_tokenizer = load_encoder_tokenizer()
    model = HebrewG2PClassifier().to(device)

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
                    metrics = evaluate(model, eval_loader, device, args.fp16, encoder_tokenizer, MAX_LEN)
                    print(f"\n[step {opt_step}] CER: {metrics['cer']:.4f}  WER: {metrics['wer']:.4f}  Acc: {metrics['acc']:.1%}  loss: {metrics['eval_loss']:.4f}")
                    for i, (ref, hyp) in enumerate(zip(metrics["refs"][:3], metrics["hyps"][:3]), 1):
                        print(f"  {i}. GT:   {ref}")
                        print(f"     Pred: {hyp}")
                    save_checkpoint(model, output_dir, opt_step, metrics["wer"], args.save_total_limit)
                    if metrics["wer"] < best_wer:
                        best_wer = metrics["wer"]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= args.early_stopping_patience:
                            print(f"[step {opt_step}] Early stopping: WER has not improved for {args.early_stopping_patience} evals (best={best_wer:.4f})")
                            stop_training = True
                            break

    metrics = evaluate(model, eval_loader, device, args.fp16, encoder_tokenizer, MAX_LEN)
    print(f"\nFinal: CER: {metrics['cer']:.4f}  WER: {metrics['wer']:.4f}  Acc: {metrics['acc']:.1%}  loss: {metrics['eval_loss']:.4f}")
    for i, (ref, hyp) in enumerate(zip(metrics["refs"][:3], metrics["hyps"][:3]), 1):
        print(f"  {i}. GT:   {ref}")
        print(f"     Pred: {hyp}")
    save_checkpoint(model, output_dir, opt_step, metrics["wer"], args.save_total_limit)


if __name__ == "__main__":
    main()
