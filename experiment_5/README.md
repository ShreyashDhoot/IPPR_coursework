# Experiment 5

Notebook: experiment_5.ipynb

## Aim
Explore image segmentation methods including region growing, connected-component merging, and clustering.

## Covered Work
- Region-growing style segmentation using:
  - seed point initialization
  - 8-neighborhood traversal (BFS using deque)
  - threshold-based inclusion criterion
- Binary segmentation with thresholding and connected components.
- Region adjacency detection and mean-intensity-based region merging (union-find).
- K-means based grayscale segmentation (K = 3).

## Libraries Used
- cv2
- numpy
- matplotlib
- requests (imported)
- collections.deque

## Notes
- Uses absolute path to input image from experiment_1.
- Seed and threshold values strongly affect region-growing output.

## How To Run
1. Open experiment_5.ipynb.
2. Install dependencies (opencv-python, numpy, matplotlib).
3. Make sure referenced image path is valid.
4. Run cells in order and tune seed/threshold if needed.
