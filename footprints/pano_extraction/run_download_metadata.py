import os
import numpy as np
import pandas as pd
import math
from streetlevel import streetview

# Configuration for start and end coordinates
config = {
    "start": (48.14985834142777, 11.568646905269516),
    "end": (48.149701975155594, 11.56927631875078),
    "num_points": 40,  # Number of interpolated points
    "output_dir": "output_tum_v3",
    "metadata_file": "gsv_metadata_v3.csv",
    "location_label": "Munich, Bavaria"
}

start_lat, start_lon = config["start"]
end_lat, end_lon = config["end"]
num_points = config["num_points"]
output_dir = config["output_dir"]
metadata_file = config["metadata_file"]
location_label = config["location_label"]

latitudes = np.linspace(start_lat, end_lat, num_points)
longitudes = np.linspace(start_lon, end_lon, num_points)

os.makedirs(output_dir, exist_ok=True)

metadata_list = []

downloaded_pano_ids = set()

for lat, lon in zip(latitudes, longitudes):
    try:
        pano = streetview.find_panorama(lat, lon)
        if pano.id not in downloaded_pano_ids:
            filename = os.path.join(output_dir, f"{pano.id}.jpg")
            streetview.download_panorama(pano, filename)
            downloaded_pano_ids.add(pano.id)
            print(f"Downloaded panorama {pano.id} at location {lat}, {lon} to {filename}")
            
            metadata_list.append({
                'pid': pano.id,
                'lat': lat,
                'lng': lon,
                'heading': pano.heading,
                'degree': (360 * pano.heading) / (2 * math.pi),  # Convert radians to degrees
                'date': pano.date,
                'location': location_label
            })
        else:
            print(f"Skipped downloading panorama {pano.id} at location {lat}, {lon} (already downloaded)")
    except Exception as e:
        print(f"Failed to download panorama at location {lat}, {lon}: {e}")

df = pd.DataFrame(metadata_list)
df.to_csv(metadata_file, index_label='index')
