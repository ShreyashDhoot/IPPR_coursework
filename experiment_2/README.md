# Experiment 2

Notebook: experiment_2.ipynb

## Aim
Study histogram equalization methods for contrast enhancement in grayscale and color images.

## Covered Work
- Histogram equalization on grayscale image using cv2.equalizeHist.
- Histogram plotting and comparison of original vs equalized outputs.
- Color equalization approach 1:
  - split RGB channels
  - equalize each channel independently
  - merge and visualize results
- Custom histogram equalization function implemented from first principles:
  - histogram and PDF computation
  - CDF mapping
  - intensity remapping to produce enhanced image

## Libraries Used
- cv2
- numpy
- matplotlib
- pandas (imported)

## Notes
- Input paths are absolute (for example, D:\\ippr\\experiment_2\\low_contrast_img.jpg).
- For portability, use relative paths where possible.

## How To Run
1. Open experiment_2.ipynb.
2. Install dependencies (opencv-python, numpy, matplotlib).
3. Verify input images exist at configured paths.
4. Run all cells in order.
