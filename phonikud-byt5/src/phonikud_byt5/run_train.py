#!/usr/bin/env python3

import os
import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer, Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback


class SilentProgressCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass
from torch.utils.data import Dataset
import wandb
import random
from tqdm import tqdm

from config import TrainArgs
from utils import prepare_lines, calculate_wer_cer_metrics, log_metrics, TrainingLine, update_metadata_with_models


class BestLastModelCallback(TrainerCallback):
    """Custom callback to save best and last models during training"""
    
    def __init__(self, ckpt_dir, tokenizer, val_lines, use_wandb=False, patience=40):
        self.ckpt_dir = ckpt_dir
        self.tokenizer = tokenizer
        self.val_lines = val_lines
        self.use_wandb = use_wandb
        self.best_eval_loss = float('inf')
        self.best_wer = float('inf')
        self.best_step = 0
        self.last_eval_loss = None
        self.last_step = 0
        self.patience = patience
        self.no_improve_count = 0
    
    def _compute_wer_cer(self, model, device, batch_size=16):
        model.eval()
        predictions, ground_truth = [], []
        with torch.no_grad():
            for i in range(0, len(self.val_lines), batch_size):
                batch = self.val_lines[i:i + batch_size]
                inputs = self.tokenizer(
                    [line.unvocalized for line in batch],
                    return_tensors='pt', padding=True, truncation=True, max_length=512
                ).to(device)
                output_ids = model.generate(**inputs, max_length=512)
                for j, line in enumerate(batch):
                    pred = self.tokenizer.decode(output_ids[j], skip_special_tokens=True)
                    predictions.append(pred)
                    ground_truth.append(line.vocalized)
        return calculate_wer_cer_metrics(predictions, ground_truth), predictions, ground_truth

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when metrics are logged - captures eval_loss"""
        if not state.is_world_process_zero:
            return
        if logs and 'eval_loss' in logs:
            current_eval_loss = logs['eval_loss']
            current_step = state.global_step

            try:
                device = next(model.parameters()).device
                metrics, predictions, ground_truth = self._compute_wer_cer(model, device)
                import json as _json
                metrics_dict = {
                    "step": current_step,
                    "eval_loss": current_eval_loss,
                    "cer": metrics.cer,
                    "wer": metrics.wer,
                    "acc": metrics.wer_accuracy / 100,
                }

                # Save last model (always)
                last_model_path = f"{self.ckpt_dir}/last_model"
                model.save_pretrained(last_model_path)
                self.tokenizer.save_pretrained(last_model_path)
                open(f"{last_model_path}/train_state.json", "w").write(_json.dumps(metrics_dict, indent=2))

                self.last_eval_loss = current_eval_loss
                self.last_step = current_step

                # Save best model (if improved by WER)
                is_best = metrics.wer < self.best_wer
                if is_best:
                    self.best_eval_loss = current_eval_loss
                    self.best_wer = metrics.wer
                    self.best_step = current_step
                    self.no_improve_count = 0

                    best_model_path = f"{self.ckpt_dir}/checkpoint-best"
                    model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    open(f"{best_model_path}/train_state.json", "w").write(_json.dumps(metrics_dict, indent=2))
                else:
                    self.no_improve_count += 1

                marker = " ✓ new best" if is_best else f"  patience: {self.no_improve_count}/{self.patience}"
                tqdm.write(f"\n[step {current_step}] loss: {current_eval_loss:.4f}  CER: {metrics.cer:.4f}  WER: {metrics.wer:.4f}  Acc: {metrics.wer_accuracy:.1f}%{marker}")
                for idx in random.sample(range(len(self.val_lines)), min(3, len(self.val_lines))):
                    tqdm.write(f"  Src:  {self.val_lines[idx].unvocalized}")
                    tqdm.write(f"  GT:   {ground_truth[idx]}")
                    tqdm.write(f"  Pred: {predictions[idx]}")


                # Update metadata
                best_model_info = {
                    "path": "checkpoint-best",
                    "eval_loss": self.best_eval_loss,
                    "step": self.best_step
                }
                
                last_model_info = {
                    "path": "last_model", 
                    "eval_loss": self.last_eval_loss,
                    "step": self.last_step
                }
                
                # Add wandb info to metadata if available
                if self.use_wandb and hasattr(wandb, 'run') and wandb.run is not None:
                    wandb_url = f"https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}"
                    wandb_info = {
                        "run_id": wandb.run.id,
                        "run_name": wandb.run.name,
                        "project": wandb.run.project,
                        "entity": wandb.run.entity,
                        "url": wandb_url,
                    }
                    best_model_info["wandb"] = wandb_info
                    last_model_info["wandb"] = wandb_info
                
                update_metadata_with_models(self.ckpt_dir, best_model_info, last_model_info)
                    
            except Exception as e:
                print(f"❌ ERROR in callback: {e}")
                import traceback
                traceback.print_exc()


class HebrewG2PDataset(Dataset):
    def __init__(self, lines, tokenizer, max_length=512):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx]
        
        # Tokenize input (Hebrew text)
        inputs = self.tokenizer(
            line.unvocalized,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (phonemes)
        targets = self.tokenizer(
            line.vocalized,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }


def main():
    args = TrainArgs().parse_args()
    
    print(f"🚀 Starting ByT5 Hebrew G2P Training")
    print(f"Data dir: {args.data_dir}")
    print(f"Model: {args.model_name}")
    
    # Initialize wandb if not disabled
    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config={
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_context_length": args.max_context_length,
                "val_split": args.val_split,
                "eval_steps": args.eval_steps,
            }
        )
        report_to = ["wandb"]
        print(f"📊 Wandb initialized: {args.wandb_mode} mode")
    else:
        report_to = []
        print("📊 Wandb disabled")
    
    # Prepare wandb info if enabled
    wandb_info = None
    if args.wandb_mode != "disabled" and hasattr(wandb, 'run') and wandb.run is not None:
        wandb_url = f"https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}"
        wandb_info = {
            "run_id": wandb.run.id,
            "run_name": wandb.run.name,
            "project": wandb.run.project,
            "entity": wandb.run.entity,
            "url": wandb_url,
        }
    
    # Load data
    train_lines, val_lines = prepare_lines(args, wandb_info=wandb_info)
    
    # Initialize model and tokenizer
    print(f"📦 Loading {args.model_name}...")
    tokenizer = ByT5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Set up device with fallback
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    model = model.to(device)
    
    # Create datasets
    train_dataset = HebrewG2PDataset(train_lines, tokenizer, args.max_context_length)
    val_dataset = HebrewG2PDataset(val_lines, tokenizer, args.max_context_length)
    
    # Setup training
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=args.ckpt_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=args.logging_steps,
        log_level="error",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",  # Disable automatic checkpoint saving
        load_best_model_at_end=False,  # We handle this in our callback
        metric_for_best_model="eval_loss",
        report_to=report_to,
    )
    
    # Initialize callback for saving best/last models
    callback = BestLastModelCallback(args.ckpt_dir, tokenizer, val_lines, use_wandb=(args.wandb_mode != "disabled"))
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(SilentProgressCallback)
    
    # Start training
    print("🏋️ Starting training...")
    result = trainer.train()
    
    print(f"✅ Training complete!")
    print(f"🏆 Best model: {callback.best_eval_loss:.4f} (step {callback.best_step})")
    print(f"📦 Last model: {callback.last_eval_loss:.4f} (step {callback.last_step})")
    print(f"📁 Models saved in: {args.ckpt_dir}/checkpoint-best and {args.ckpt_dir}/last_model")
    print(f"📋 Metadata continuously updated during training")
    
    # Close wandb if it was initialized
    if args.wandb_mode != "disabled":
        wandb.finish()
        print("📊 Wandb session completed")


if __name__ == "__main__":
    main()
