#!/usr/bin/env python3

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import ByT5Tokenizer
from pathlib import Path
from tap import Tap
from onnxruntime.quantization import quantize_dynamic, QuantType


class ExportArgs(Tap):
    model_path: str = "../checkpoints/best_model"  # Path to trained model directory
    output_dir: str = "./onnx_model"  # Output directory for ONNX model
    quantize_int8: bool = False  # Apply INT8 quantization
    quantization_type: str = "dynamic"  # Quantization type: 'dynamic' or 'static'


def main():
    """Export ByT5 model to ONNX using HuggingFace Optimum with optional INT8 quantization"""
    args = ExportArgs().parse_args()
    
    print(f"🚀 Exporting ByT5 model to ONNX using HuggingFace Optimum")
    print(f"   Model: {args.model_path}")
    print(f"   Output: {args.output_dir}")
    if args.quantize_int8:
        print(f"   Quantization: INT8 ({args.quantization_type})")
    else:
        print(f"   Quantization: None (FP32)")
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return
    
    # Check for config.json
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"❌ config.json not found in {model_path}")
        return
    
    try:
        print(f"📦 Loading and converting model to ONNX...")
        
        # Convert the model directly using Optimum
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            export=True,  # This triggers the conversion
        )
        
        print(f"✅ Model converted successfully")
        
        # Save the ONNX model
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving ONNX model to {output_path}...")
        ort_model.save_pretrained(str(output_path))
        
        # Also save the tokenizer
        print(f"📝 Saving tokenizer...")
        tokenizer = ByT5Tokenizer.from_pretrained(str(model_path))
        tokenizer.save_pretrained(str(output_path))
        
        # Apply INT8 quantization if requested
        if args.quantize_int8:
            print(f"🔢 Applying INT8 quantization ({args.quantization_type})...")
            
            # Find ONNX files to quantize
            onnx_files = list(output_path.glob("*.onnx"))
            
            if not onnx_files:
                print(f"❌ No ONNX files found for quantization")
                return
            
            for onnx_file in onnx_files:
                print(f"   Quantizing: {onnx_file.name}")
                
                # Create quantized filename
                quantized_file = onnx_file.parent / f"{onnx_file.stem}_int8.onnx"
                
                try:
                    if args.quantization_type == "dynamic":
                        # Dynamic quantization (no calibration data needed)
                        quantize_dynamic(
                            model_input=str(onnx_file),
                            model_output=str(quantized_file),
                            weight_type=QuantType.QInt8
                        )
                    else:
                        print(f"⚠️  Static quantization not implemented yet, falling back to dynamic")
                        quantize_dynamic(
                            model_input=str(onnx_file),
                            model_output=str(quantized_file),
                            weight_type=QuantType.QInt8
                        )
                    
                    # Remove original file and rename quantized file
                    onnx_file.unlink()
                    quantized_file.rename(onnx_file)
                    print(f"   ✅ {onnx_file.name} quantized successfully")
                    
                except Exception as quant_error:
                    print(f"   ❌ Error quantizing {onnx_file.name}: {quant_error}")
                    # Keep the original file if quantization fails
                    if quantized_file.exists():
                        quantized_file.unlink()
        
        print(f"✅ Export completed successfully!")
        print(f"   ONNX model saved to: {output_path}")
        
        # Show file sizes
        onnx_files = list(output_path.glob("*.onnx"))
        if onnx_files:
            print(f"📁 Generated ONNX files:")
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                quant_status = "(INT8)" if args.quantize_int8 else "(FP32)"
                print(f"   - {onnx_file.name}: {size_mb:.1f} MB {quant_status}")
        
        print(f"\n🎉 Ready to use!")
        print(f"   Run: python predict.py --model_path {output_path}")
        print(f"\n💡 Usage tips:")
        print(f"   - For INT8 quantization: python export.py --quantize_int8")
        print(f"   - Quantization reduces model size but may slightly affect accuracy")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        print(f"   Make sure the model is a valid ByT5 model")
        print(f"   Try: pip install --upgrade optimum[onnxruntime]")


if __name__ == "__main__":
    main()
