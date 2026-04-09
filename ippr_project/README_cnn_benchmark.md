# CNN Benchmark: AI vs Real Image Classification

This document describes the benchmark notebook at [ippr_project/cnn_benchmark.ipynb](ippr_project/cnn_benchmark.ipynb).

## Goal

Train and evaluate a deep CNN to classify images into:
- fake (AI-generated)
- real

Dataset source:
- Hugging Face dataset repo: ShreyashDhoot/AI_vs_Real

## Notebook Features

The notebook includes:
- Hugging Face data loading with retry logic for rate limits.
- Train/validation/test split creation.
- Class imbalance analysis and class weights.
- Deep CNN with residual connections.
- Focal Loss for imbalance-robust optimization.
- Learning-rate warmup plus ReduceLROnPlateau.
- PR-AUC-focused monitoring and early stopping.
- Threshold tuning (not fixed at 0.5).
- Comprehensive evaluation and precision-recall visualization.

## Model Architecture

High-level architecture in the benchmark notebook:

```text
Input (160x160x3)
   |
   +--> Data pipeline preprocessing:
   |      - normalize to [0,1]
   |      - random flip
   |      - random brightness/contrast
   |
   +--> Conv Block 1: Conv(16) -> BN -> Conv(16) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
   |
   +--> Residual Block 2:
   |      Main: Conv(32) -> BN -> Conv(32) -> BN
   |      Skip: 1x1 Conv(32)
   |      Add -> ReLU -> MaxPool -> Dropout(0.30)
   |
   +--> Residual Block 3:
   |      Main: Conv(64) -> BN -> Conv(64) -> BN
   |      Skip: 1x1 Conv(64)
   |      Add -> ReLU -> MaxPool -> Dropout(0.35)
   |
   +--> Residual Block 4:
   |      Main: Conv(128) -> BN -> Conv(128) -> BN
   |      Skip: 1x1 Conv(128)
   |      Add -> ReLU -> MaxPool -> Dropout(0.40)
   |
   +--> GlobalAveragePooling
   +--> Dense(128) + BN + Dropout(0.45)
   +--> Dense(64) + BN + Dropout(0.35)
   +--> Dense(1, sigmoid)
```

## Loss, Optimization, and Scheduling

- Loss: custom binary Focal Loss (alpha=0.25, gamma=2.0).
- Optimizer: Adam.
- Class weights: computed from training labels and passed into fit.
- LR schedule:
  - warmup in early epochs,
  - then adaptive reduction via ReduceLROnPlateau.
- Early stopping monitors validation PR-AUC.

## Data Pipeline Notes

The training dataset uses repeat with explicit step counts to avoid generator exhaustion:
- train_steps_per_epoch = ceil(train_size / batch_size)
- validation_steps = ceil(val_size / batch_size)
- test_steps = ceil(test_size / batch_size)

This avoids the warning:
- "Your input ran out of data; interrupting training"

## Evaluation Protocol

Evaluation is focused on imbalance-aware metrics:
- PR-AUC
- Precision
- Recall
- F1-score
- ROC-AUC
- Balanced accuracy
- MCC
- Specificity and sensitivity

Threshold selection:
- Searches thresholds from 0.10 to 0.94.
- Picks threshold that maximizes F1-score on predictions.
- Reports comparison against default threshold 0.5.

## How To Run

1. Open [ippr_project/cnn_benchmark.ipynb](ippr_project/cnn_benchmark.ipynb).
2. Run cells in order from top to bottom.
3. If needed, set your Hugging Face token in the first cell.
4. Train in Cell 5 and evaluate in Cells 7-9.

## Dependencies

- tensorflow
- numpy
- matplotlib
- scikit-learn
- datasets
- huggingface_hub

Example install command:

```bash
pip install tensorflow numpy matplotlib scikit-learn datasets huggingface_hub
```

## Output Artifacts

Model save cell writes:
- deep_dense_cnn_ai_vs_real.keras

## Troubleshooting

1. Keras preprocessing compatibility errors
- Augmentation is already handled in tf.data preprocessing to avoid keras namespace issues.

2. Input ran out of data warning
- Use the updated cells that define repeat and explicit steps_per_epoch.

3. Low precision with decent accuracy
- Keep focal loss, class weights, and threshold tuning enabled.
- Use PR-AUC and precision-recall curves as primary diagnostics.
