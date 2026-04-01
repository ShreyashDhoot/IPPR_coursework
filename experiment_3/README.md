# Experiment 3

Notebook: experiment_3.ipynb

## Aim
Apply spatial filtering and analyze denoising behavior under Gaussian noise.

## Covered Work
- Average filtering with multiple mask sizes (3, 7, 9, 16, 32).
- Gaussian noise insertion into grayscale image.
- Noise reduction using:
  - median filter
  - Gaussian filter
- Sharpening using a 3x3 high-boost style kernel with filter2D.

## Libraries Used
- cv2
- numpy
- matplotlib
- pandas (imported)

## Notes
- Notebook references an image from experiment_1 via absolute path.
- The markdown mentions salt-and-pepper noise objective; current implementation primarily demonstrates Gaussian noise filtering.

## How To Run
1. Open experiment_3.ipynb.
2. Install dependencies (opencv-python, numpy, matplotlib).
3. Ensure referenced image path is valid.
4. Execute cells sequentially.
