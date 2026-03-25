import os
import random
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import jiwer

from config import TrainArgs


@dataclass
class TrainingLine:
    vocalized: str      # The phonemes (target)
    unvocalized: str    # The Hebrew text (input)


@dataclass 
class MetricsResult:
    wer: float
    cer: float
    wer_accuracy: float
    cer_accuracy: float
    val_loss: float


def read_lines(data_dir: str, max_context_length: int, max_lines: Optional[int] = None) -> List[TrainingLine]:
    """Read tab-separated text files and return TrainingLine objects"""
    lines = []
    
    for file_path in Path(data_dir).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                    
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    continue
                    
                hebrew_text, phonemes = parts
                
                # Simple length filtering
                if len(hebrew_text) > max_context_length or len(phonemes) > max_context_length:
                    continue
                
                lines.append(TrainingLine(
                    vocalized=phonemes,    # Target (phonemes)  
                    unvocalized=hebrew_text  # Input (Hebrew text)
                ))
                
                if max_lines and len(lines) >= max_lines:
                    return lines
    
    return lines


def prepare_indices(lines: List[TrainingLine], val_split: float, split_seed: int) -> Tuple[List[int], List[int]]:
    """Split data indices into train and validation"""
    random.seed(split_seed)
    indices = list(range(len(lines)))
    random.shuffle(indices)
    
    val_size = int(len(lines) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    return train_indices, val_indices


def save_train_metadata(val_indices, lines, val_split, split_seed, max_lines, data_dir, ckpt_dir, best_model_info=None, last_model_info=None, wandb_info=None):
    """Save training metadata to checkpoint directory"""
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Get current UTC date in yyyy-mm-dd format
    utc_date = datetime.utcnow().strftime("%Y-%m-%d")
    
    # Get absolute paths of all files in data directory
    data_path = Path(data_dir)
    data_files = [str(file.absolute()) for file in data_path.glob("*.txt")]
    data_files.sort()  # Sort for consistency
    
    metadata = {
        "training_date_utc": utc_date,
        "data_files": data_files,
        "val_split": val_split,
        "split_seed": split_seed,
        "max_lines": max_lines,
        "data_dir": data_dir,
        "total_lines": len(lines),
        "val_indices": val_indices
    }
    
    # Add wandb information if provided
    if wandb_info:
        metadata["wandb"] = wandb_info
    
    # Add best and last model information if provided
    if best_model_info:
        metadata["best_model"] = best_model_info
    
    if last_model_info:
        metadata["last_model"] = last_model_info
    
    with open(os.path.join(ckpt_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def update_metadata_with_models(ckpt_dir, best_model_info, last_model_info):
    """Update existing metadata.json with best and last model information"""
    metadata_path = os.path.join(ckpt_dir, "metadata.json")
    
    # Read existing metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Update with model information
    if best_model_info:
        metadata["best_model"] = best_model_info
    
    if last_model_info:
        metadata["last_model"] = last_model_info
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def prepare_lines(args: TrainArgs, wandb_info=None) -> Tuple[List[TrainingLine], List[TrainingLine]]:
    """Higher level function that reads lines, splits them, and saves metadata"""
    # Read all lines
    print("📖🔍 Reading lines from dataset...")
    lines = read_lines(args.data_dir, args.max_context_length, args.max_lines)

    # Prepare train/val split
    train_indices, val_indices = prepare_indices(lines, args.val_split, args.split_seed)

    # Create train and validation datasets
    train_lines = [lines[i] for i in train_indices]
    val_lines = [lines[i] for i in val_indices]

    # Save metadata
    save_train_metadata(
        val_indices, lines, args.val_split, args.split_seed, args.max_lines, args.data_dir, args.ckpt_dir, wandb_info=wandb_info
    )

    # Print samples
    print("🛤️ Train samples:")
    for i in train_lines[:3]:
        print(f"\t{i.vocalized} | {i.unvocalized}")

    print("🧪 Validation samples:")
    for i in val_lines[:3]:
        print(f"\t{i.vocalized} | {i.unvocalized}")

    print(
        f"✅ Loaded {len(train_lines)} training lines and {len(val_lines)} validation lines."
    )

    return train_lines, val_lines


def calculate_wer_cer_metrics(
    predictions: List[str], ground_truth: List[str], val_loss: float = 0.0
) -> MetricsResult:
    """
    Calculate WER and CER metrics from predictions and ground truth.

    Args:
        predictions: List of predicted text strings
        ground_truth: List of ground truth text strings
        val_loss: Validation loss (optional, defaults to 0.0)

    Returns:
        MetricsResult containing WER, CER, and accuracy metrics
    """
    # Calculate WER and CER using jiwer
    wer = jiwer.wer(ground_truth, predictions)
    cer = jiwer.cer(ground_truth, predictions)

    # Handle the case where jiwer returns a dict instead of float
    if isinstance(wer, dict):
        wer = float(wer.get("wer", 0.0))
    if isinstance(cer, dict):
        cer = float(cer.get("cer", 0.0))

    # Calculate accuracies as percentages (1 - error_rate) * 100
    wer_accuracy = (1 - wer) * 100
    cer_accuracy = (1 - cer) * 100

    return MetricsResult(
        wer=wer,
        cer=cer,
        wer_accuracy=wer_accuracy,
        cer_accuracy=cer_accuracy,
        val_loss=val_loss,
    )


def log_metrics(
    metrics: MetricsResult,
    predictions: List[str],
    ground_truth: List[str],
    phase: str = "val",
) -> None:
    """
    Log metrics and examples to console.

    Args:
        metrics: MetricsResult containing the calculated metrics
        predictions: List of predicted text strings
        ground_truth: List of ground truth text strings
        phase: Phase identifier ("train" or "val")
    """
    # Log metrics to console
    print(f"📊 {phase.upper()} Metrics:")
    print(f"  Loss: {metrics.val_loss:.4f}")
    print(f"  WER: {metrics.wer:.4f} (Accuracy: {metrics.wer_accuracy:.2f}%)")
    print(f"  CER: {metrics.cer:.4f} (Accuracy: {metrics.cer_accuracy:.2f}%)")

    # Log random text examples to console
    num_examples = min(3, len(ground_truth))
    if num_examples > 0:
        random_indices = random.sample(range(len(ground_truth)), num_examples)
        print(f"\n📝 {phase.upper()} Examples:")
        for i, idx in enumerate(random_indices):
            print(f"  Example {i + 1}:")
            print(f"    Source:    {ground_truth[idx]}")
            print(f"    Predicted: {predictions[idx]}")
