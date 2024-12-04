# Enhanced Semantic-SAM Mask Processing
## Overview

- Processes images using the Semantic SAM framework.
- Generates masks and applies semantic class predictions.
- Includes functions for:
   - Mask combination
   - Cropping
   - Filtering
- Creates refined outputs.

---

## Installation and Setup

1. **Install Required Dependencies**:
   Make sure the following libraries are installed:
   - `torch`
   - `clip`
   - `Pillow` (PIL)
   - `numpy`
   - `opencv-python`
   - `torchvision`
   - Any additional dependencies in `utils.semantic_sam`

   Install these via pip:
   ```bash
   pip install torch torchvision numpy pillow opencv-python

## Note
We define 'top_k_masks' parameter in the script to determine the number of masks to retain and process after the initial mask generation step. It helps filter and prioritize the masks based on their relevance (e.g., area) before further steps.