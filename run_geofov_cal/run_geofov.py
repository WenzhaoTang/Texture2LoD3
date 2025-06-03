import os
import pandas as pd
from geofov_utils import (
    load_buildings,
    run_fov_processing,
    plot_building_rays
)

def main():
    # Guide paths
    meta_csv = 'data/demo_data/gsv_metadata_demo.csv' # update
    buildings_dir = 'data/citygml_v2'
    output_csv = 'data/geofov_v2.csv' # update
    crs = 'EPSG:25832'
    buffer_dist = 50
    output_dir = 'set/your/own/output/dir'

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(meta_csv)
    buildings = load_buildings(buildings_dir)
    if not buildings:
        print("No building data, exiting.")
        return

    print("Starting FOV computation...")
    run_fov_processing(df, buildings, crs, buffer_dist, output_csv)
    print("Plotting building rays...")
    plot_building_rays(output_csv, buildings_dir, crs, output_dir)
    print("Done.")

if __name__ == '__main__':
    main()
