# Clustering Model Surfaces

## Prerequisites

Ensure you have the following Python packages installed:
- `numpy`
- `scikit-learn`
- `shapely`
- `scipy`
- `open3d`

You can install them using pip:
```bash
pip install numpy scikit-learn shapely scipy open3d
```

## Steps

1. **Prepare your OBJ file**: Ensure you have a valid LoD3 OBJ file that you want to process.

2. **Run the script**: Use the following command to run the script, replacing `<path_to_obj_file>` and `<output_directory>` with your actual file path and desired output directory.
    ```bash
    python run_normals_cluster_retriangulate.py <path_to_obj_file> <output_directory>
    ```

3. **Check the output**: The script will generate clustered OBJ files in the specified output directory and visualize the clusters with quadrilaterals.