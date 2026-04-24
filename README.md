# Best Classifier on Vox Knesset

This repository stores the best checkpoint of our Hebrew G2P classifier model.

## Files

- `best Classifier on vox knesset/model.safetensors`  
  The trained model weights.

- `best Classifier on vox knesset/train_state.json`  
  The saved training step and validation metrics for this best checkpoint.

- `best Classifier on vox knesset/README.md`  
  A short explanation of this checkpoint folder.

## What the weights do

The model predicts, for each Hebrew letter:

1. consonant
2. vowel (nikud)
3. stress

These 3 predictions are combined into the final IPA pronunciation output.
