# ByT5 ONNX Export Tool

Quick tool to export ByT5 models to ONNX format with optional INT8 quantization.

## Export Options

```bash
# 1. Regular FP32 export (default)
python export.py --model_path ../checkpoints/best_model --output_dir ./onnx_model

# 2. INT8 dynamic quantization (smaller, faster)  
python export.py --model_path ../checkpoints/best_model --output_dir ./onnx_model --quantize_int8

# 3. INT8 static quantization (future - falls back to dynamic for now)
python export.py --quantize_int8 --quantization_type static
```

## Run the exported model
```bash
python predict.py --model_path ./onnx_model
```

INT8 models are ~4x smaller but may have slight accuracy loss. Perfect for deployment! 🚀
