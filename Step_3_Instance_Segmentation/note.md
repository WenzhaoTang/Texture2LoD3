# Semantic SAM Mask Processing

## Overview
This script processes images using the Semantic SAM framework, generating masks and applying semantic class predictions to create refined outputs. The script includes functions for mask combination, cropping, and filtering.

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
The 'top_k_masks' parameter in the script determines the number of masks to retain and process after the initial mask generation step. It helps filter and prioritize the masks based on their relevance (e.g., area) before further steps.