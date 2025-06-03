# download_metadata.py

Download Google Street View panoramas and metadata along a line between two GPS coordinates.

## Prerequisites
this script is inspired by [StreetLevel](https://github.com/sk-zk/streetlevel), In order to download Google Street View panoramas and metadata, you need to install required packages:
```bash
pip install streetlevel
```

## Configuration

Edit the `start_lat`, `start_lon`, `end_lat`, and `end_lon` variables at the top of `download_metadata.py`. You can also adjust the number of samples (`np.linspace`) and output paths directly in the script.

## Usage

Run the script from the repository root:
```bash
python download_metadata/download_metadata.py
```

By default, panoramas are saved to `output_tum/` and metadata is written to `gsv_metadata.csv` in the current directory.
