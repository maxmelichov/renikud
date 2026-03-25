#!/usr/bin/env python3

from typing import Literal
import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
from tqdm import tqdm
from tap import Tap

from utils import read_lines, calculate_wer_cer_metrics, TrainingLine


class EvalArgs(Tap):
    model_path: str = "checkpoints/final_model"  # Path to trained model directory  
    data_dir: str = "data"    # Path to test data
    max_lines: int = 1000  # Limit for testing
    max_context_length: int = 512
    batch_size: int = 8
    device: Literal["mps", "cuda", "cpu"] = "cuda"  # Device to use: "mps", "cuda", "cpu"
    
    # For single prediction
    text: str = None  # Single Hebrew text to predict


def predict_batch(model, tokenizer, texts, max_length=512, device=None):
    """Predict phonemes for a batch of Hebrew texts"""
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # Move inputs to device if specified
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            repetition_penalty=1.1,  # Gentler repetition penalty
        )
    
    # Decode predictions
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return predictions


def evaluate_model(args: EvalArgs):
    """Evaluate trained model on test data"""
    print(f"📦 Loading model from {args.model_path}...")
    
    # Load model and tokenizer
    tokenizer = ByT5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    
    # Set up device with fallback
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU") 
        args.device = "cpu"
    
    device_obj = torch.device(args.device)
    print(f"🔧 Using device: {device_obj}")
    model = model.to(device_obj)
    model.eval()
    
    print(f"📖 Loading test data from {args.data_dir}...")
    
    # Load test data (using same format as training)
    test_lines = read_lines(args.data_dir, args.max_context_length, args.max_lines)
    
    print(f"🧪 Evaluating on {len(test_lines)} samples...")
    
    # Prepare data for evaluation
    hebrew_texts = [line.unvocalized for line in test_lines]
    ground_truth = [line.vocalized for line in test_lines]
    
    # Predict in batches
    all_predictions = []
    
    for i in tqdm(range(0, len(hebrew_texts), args.batch_size), desc="Predicting"):
        batch_texts = hebrew_texts[i:i+args.batch_size]
        batch_predictions = predict_batch(model, tokenizer, batch_texts, args.max_context_length, device_obj)
        all_predictions.extend(batch_predictions)
    
    # Calculate metrics
    metrics = calculate_wer_cer_metrics(all_predictions, ground_truth)
    
    # Print results
    print(f"\n📊 Evaluation Results:")
    print(f"WER: {metrics.wer:.4f} ({metrics.wer_accuracy:.2f}% accuracy)")
    print(f"CER: {metrics.cer:.4f} ({metrics.cer_accuracy:.2f}% accuracy)")
    
    # Show examples
    print(f"\n📝 Examples:")
    for i in range(min(5, len(test_lines))):
        print(f"Input:     {hebrew_texts[i]}")
        print(f"Target:    {ground_truth[i]}")
        print(f"Predicted: {all_predictions[i]}")
        print("---")
    
    return metrics


def predict_single(args: EvalArgs):
    """Predict phonemes for a single Hebrew text"""
    print(f"📦 Loading model from {args.model_path}...")
    
    # Load model and tokenizer
    tokenizer = ByT5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    
    # Set up device with fallback
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU") 
        args.device = "cpu"
    
    device_obj = torch.device(args.device)
    print(f"🔧 Using device: {device_obj}")
    model = model.to(device_obj)
    model.eval()
    
    # Predict
    prediction = predict_batch(model, tokenizer, [args.text], args.max_context_length, device_obj)[0]
    
    print(f"Input: {args.text}")
    print(f"Predicted phonemes: {prediction}")
    
    return prediction


def main():
    args = EvalArgs().parse_args()
    
    if args.text:
        # Single prediction mode
        predict_single(args)
    else:
        # Evaluation mode
        evaluate_model(args)


if __name__ == "__main__":
    main()
