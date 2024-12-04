# Image Stitching Module

This module provides functionality for stitching two custom images together (not necessarily panorama image patches).

## Classes
### `Image_Stitching`
A class that handles the image stitching process.

---

## Methods

### `__init__(self)`
- Initializes the `Image_Stitching` class with default parameters.

### `registration(self, img1, img2)`
- Registers two images by finding keypoints and computing the homography matrix.

### `create_mask(self, img1, img2, version)`
- Creates a mask for blending two images smoothly.

### `blending(self, img1, img2)`
- Blends two images together to create a single stitched image.

### `main(argv1, argv2)`
- Reads two images from file paths and creates a stitched image.

---

## Usage
Run the script with two image file paths as arguments to generate a stitched image.

### Example
```bash
python run_Image_Stitching.py '/path/to/image1.jpg' '/path/to/image2.jpg'
