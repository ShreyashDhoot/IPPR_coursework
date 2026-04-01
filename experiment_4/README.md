# Experiment 4 - Edge Detection

Notebook: EdgeDetection.ipynb

## Aim
Perform and compare edge detection methods, then study edge linking and boundary representation techniques.

## Covered Work
- Synthetic test image generation tailored for different detectors.
- Part A: gradient and Laplacian detectors.
  - Sobel
  - Prewitt
  - Roberts
  - Laplacian (4-connected and 8-connected variants)
  - side-by-side detector comparison
- Part B: edge linking and representation.
  - Canny vs Laplacian under noise
  - LoG (Marr-Hildreth) with varying sigma
  - contour extraction
  - convex hull
  - probabilistic Hough line transform
  - Harris corner detection
- Included conclusion and theoretical Q&A (Sobel vs Laplacian on given matrix patch).

## Libraries Used
- cv2
- numpy
- matplotlib

## How To Run
1. Open EdgeDetection.ipynb.
2. Install dependencies (opencv-python, numpy, matplotlib).
3. Run all cells from top to bottom.

## Output Expectation
The notebook is self-contained and mostly uses synthetic images, making it easy to reproduce without external image files.
