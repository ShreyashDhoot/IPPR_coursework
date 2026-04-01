# Experiment 6 - MNIST CNN Digit Recognition

Notebook: experiment_6.ipynb

## Aim
Build and train a small Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset, then evaluate model performance with accuracy, precision, recall, and confusion matrix analysis.

## Dataset
- Source: TensorFlow Keras built-in MNIST dataset (`tf.keras.datasets.mnist`)
- Classes: 10 digits (0 to 9)
- Image size: 28 x 28 grayscale

## Workflow
1. Import required libraries (`tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
2. Load MNIST train/test split.
3. Normalize pixel values to [0, 1].
4. Expand dimensions to `(28, 28, 1)` for CNN input.
5. Define and compile a compact CNN.
6. Train with validation split.
7. Evaluate on test data.
8. Compute precision and recall for each class.
9. Generate confusion matrix.
10. Plot training/validation accuracy, per-class precision/recall bars, and confusion matrix heatmap.

## CNN Architecture
- Conv2D: 32 filters, 3x3, ReLU
- MaxPooling2D: 2x2
- Conv2D: 64 filters, 3x3, ReLU
- MaxPooling2D: 2x2
- Flatten
- Dense: 64, ReLU
- Dropout: 0.3
- Dense: 10, Softmax

Loss and optimization:
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`

## Metrics and Visualizations
The notebook reports and plots:
- Test Accuracy
- Macro Precision
- Macro Recall
- Accuracy per epoch (train vs validation)
- Precision and Recall per digit class (0 to 9)
- Confusion Matrix heatmap
- Sample prediction grid (true vs predicted labels)

## Requirements
Install packages in the notebook kernel if needed:

```bash
pip install tensorflow scikit-learn seaborn matplotlib numpy
```

## How To Run
1. Open experiment_6.ipynb.
2. Run all cells from top to bottom.
3. Wait for training to complete.
4. Review metric values in output and plots for performance interpretation.

## Outcome
This experiment demonstrates a complete deep learning workflow for multi-class image classification on MNIST and introduces practical evaluation using both aggregate and class-wise metrics.
