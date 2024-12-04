# Quadrilateral Processing and Scaling Script

This step processes an input image to detect and fit a quadrilateral(In our case: Monochrome Mask), scales its corners, and outputs a resized warped quadrilateral image.

## Features
- **Mask Preprocessing**: Applies Gaussian blur and morphological operations to refine the binary mask.
- **Contour Detection**: Finds and fits the largest contour to a quadrilateral shape.
- **Quadrilateral Fitting**: Scales the detected quadrilateral corners and computes a perspective transformation matrix.
- **Output**: Saves the processed mask and the resized warped quadrilateral image.

## Requirements
- Python 3.6+
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- `enhanced_quadrilateral_fitter` from `Step_4_Morphological_Adaption`

## Note:
- Please check the quadrilateral_fitter/enhanced_quadrilateral_fitter.py for enhanced features and functions.
