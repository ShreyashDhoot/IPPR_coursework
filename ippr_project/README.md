# Tandem Model: Real vs AI Image Classification

This project uses a dual-branch architecture to classify an image as:
- Real
- AI-generated

Implementation notebook:
- `ippr_project/tandam_model.ipynb`

Dataset:
- Hugging Face: `ShreyashDhoot/Ai-vs_Real`

---

## 1) Model Overview

The network has two branches:
- A deep CNN branch for visual feature learning from RGB images.
- A texture branch that computes GLCM features and passes them through a small DNN.

The two branches are fused exactly where the CNN fully connected head starts.

### ASCII Architecture Diagram

```text
Input Image (RGB, 224x224)
           |
           +------------------------------+
           |                              |
           v                              v
   [CNN Branch]                    [Texture Branch]
ConvBlock(3->32)                   Grayscale + Quantization (64 levels)
ConvBlock(32->64)                  GLCM at distance=1, 8 angles:
ConvBlock(64->128)                 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5
ConvBlock(128->256)                |
Conv(256->512, 3x3) + BN + ReLU    v
AdaptiveAvgPool(1x1)               12 features per angle
Flatten -> 512-d                    (12 x 8 = 96 features total)
           |                        |
           |                   MLP: 96 -> 128 -> 64
           |                              |
           +-------------- Concatenate ----+
                          (512 + 64 = 576)
                                   |
                          FC: 576 -> 256 -> 2
                                   |
                               Softmax logits
```

---

## 2) GLCM Feature Design

For each of the 8 directions, 12 texture descriptors are computed:
1. Contrast
2. Dissimilarity
3. Homogeneity
4. ASM
5. Energy
6. Correlation
7. Entropy
8. Mean(i)
9. Mean(j)
10. Variance(i)
11. Cluster Shade
12. Cluster Prominence

Total texture feature vector size:
- 12 x 8 = 96

---

## 3) Trainable Parameter Count

Parameter count for the current implementation in `tandam_model.ipynb`:

### A) CNN branch
- Conv/BN stack total: 2,353,888

### B) Texture MLP branch
- Linear(96 -> 128): 12,416
- Linear(128 -> 64): 8,256
- Texture branch total: 20,672

### C) Fusion classifier
- Linear(576 -> 256): 147,712
- Linear(256 -> C): 257 x C

Where C is the number of classes.

### Total parameters (general)
- Total = 2,522,272 + 257 x C

### Total parameters for Real vs AI (C=2)
- Total trainable parameters = 2,522,786

All parameters are trainable in the current notebook setup.

---

## 4) Data Pipeline

1. Load dataset from Hugging Face using datasets.load_dataset.
2. Detect label column automatically (label/labels/target/class fallback).
3. Build train/validation/test splits if not fully provided by dataset.
4. For each sample:
   - CNN input: RGB image resized to 224x224 and normalized.
   - Texture input: grayscale quantized image -> GLCM -> 96-d feature vector.

---

## 5) Training Setup (Current Notebook)

- Framework: PyTorch
- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Epochs: 10 (default)
- Batch size: 32
- Best model selection: highest validation accuracy

After training, the model is evaluated on test data with:
- Confusion matrix
- Classification report

Saved weights file:
- `ai_vs_real_tandem_cnn_glcm.pth`

---

## 6) How To Run

Open and run:
- `ippr_project/tandam_model.ipynb`

If dependencies are missing, install:
- torch
- torchvision
- datasets
- scikit-image
- scikit-learn
- tqdm

---

## 7) Notes

- If the dataset split format changes, the notebook has fallback split logic.
- If class count is not 2, the final layer automatically adapts to detected class count.
- Input image size and GLCM settings can be tuned for speed/accuracy trade-offs.
