# Panorama Downloader Script

This script downloads Google Street View panoramas along a specified route and saves metadata about each downloaded panorama.

## Prerequisites

Ensure the following Python packages are installed:
- `os` (standard library)
- `numpy`
- `pandas`
- `math` (standard library)
- `streetlevel` (for Street View API)

Install missing dependencies using `pip`:
```bash
pip install numpy pandas streetlevel

## How to Use

### Step 1: Modify Configuration
Update the `config` dictionary in the script to specify your parameters.

#### Config Parameters:

- **start**:
  Latitude and longitude of the starting point.
  Example:
  ```python
  "start": (48.14985834142777, 11.568646905269516)
  "end": (48.149701975155594, 11.56927631875078)
  "num_points": 40
  "output_dir": "output_tum_v3"
  "metadata_file": "gsv_metadata_v3.csv"
  "location_label": "Munich, Bavaria"
