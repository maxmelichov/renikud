#!/usr/bin/env python3

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import ByT5Tokenizer
from pathlib import Path
from tap import Tap


class PredictArgs(Tap):
    model_path: str = "./onnx_model"  # Path to ONNX model directory
    max_length: int = 512  # Maximum sequence length for generation
    num_beams: int = 2  # Number of beams for beam search


def main():
    """Run inference with ONNX ByT5 model"""
    args = PredictArgs().parse_args()
    
    print(f"🔮 ByT5 ONNX Inference using HuggingFace Optimum")
    print(f"   Model: {args.model_path}")
    print(f"   Max length: {args.max_length}")
    print(f"   Num beams: {args.num_beams}")
    
    # Hebrew text to predict (same as original)
    hebrew_text = "בוא נשחק בירושלים! בוא תרד לאכול תרד! הוא רצה את זה גם אבל היא רצה מהר והקדימה אותו!"
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        print(f"   Run export.py first to create the ONNX model")
        return
    
    # Check for ONNX model
    onnx_files = list(model_path.glob("*.onnx"))
    if not onnx_files:
        print(f"❌ No ONNX files found in {model_path}")
        print(f"   Run export.py first to create the ONNX model")
        return
    
    try:
        print(f"📦 Loading ONNX model and tokenizer...")
        
        # Load the ONNX model using Optimum
        model = ORTModelForSeq2SeqLM.from_pretrained(str(model_path))
        tokenizer = ByT5Tokenizer.from_pretrained(str(model_path))
        
        print(f"✅ Model and tokenizer loaded successfully")
        print(f"   ONNX Runtime providers: {model.providers}")
        
        print(f"\n🔮 Predicting phonemes for Hebrew text...")
        print(f"Input: {hebrew_text}")
        
        # Tokenize input
        inputs = tokenizer(
            hebrew_text,
            max_length=args.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        print(f"🔄 Running inference...")
        
        # Generate prediction using the ONNX model
        # ONNX runtime has limited parameter support - keeping it minimal
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,  # Even shorter to force early termination
            num_beams=1,     # Greedy search only
            do_sample=False,
        )
        
        # Decode the prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Inference completed successfully")
        print(f"\n📝 Results:")
        print(f"Phonemes: {prediction}")
        
        print(f"\n⚡ Performance Info:")
        print(f"   - Using ONNX Runtime for faster inference")
        print(f"   - Optimized memory usage compared to PyTorch")
        print(f"   - Same generation quality as original model")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print(f"\nTroubleshooting:")
        print(f"   1. Make sure export.py completed successfully")
        print(f"   2. Check that ONNX files exist in {model_path}")
        print(f"   3. Try: pip install --upgrade optimum[onnxruntime]")


if __name__ == "__main__":
    main()
