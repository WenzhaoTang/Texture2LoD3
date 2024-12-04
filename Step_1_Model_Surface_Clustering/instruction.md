# Instructions for Clustering Model Surfaces

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

1. **Prepare your OBJ file**: Ensure you have a valid OBJ file that you want to process.

2. **Run the script**: Use the following command to run the script, replacing `<path_to_obj_file>` and `<output_directory>` with your actual file path and desired output directory.
    ```bash
    python script.py <path_to_obj_file> <output_directory>
    ```

3. **Check the output**: The script will generate clustered OBJ files in the specified output directory and visualize the clusters with quadrilaterals.

## Script Overview

- **parse_obj(file_path)**: Parses the OBJ file to extract vertices and faces.
- **compute_normals(vertices, faces)**: Computes normals for the faces.
- **cluster_normals(normals, eps=0.1, min_samples=1)**: Clusters the normals using DBSCAN.
- **select_quadrilateral_corners(vertices, faces, cluster_faces)**: Selects quadrilateral corners for each cluster.
- **save_clusters_as_obj(vertices, faces, clusters, output_dir, normals)**: Saves each cluster as a separate OBJ file.
- **visualize_clusters_with_quadrilaterals(vertices, faces, clusters, output_dir)**: Visualizes the clusters with quadrilaterals.

Follow these instructions to successfully cluster and visualize your model surfaces.