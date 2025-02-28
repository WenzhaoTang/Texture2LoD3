import os
import numpy as np
import pandas as pd
import math
from streetlevel import streetview

def download_panoramas(start_coords, end_coords, output_dir, num_points=40, location=None):
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords

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

                degree = math.degrees(pano.heading)

                metadata_list.append({
                    'pid': pano.id,
                    'lat': lat,
                    'lng': lon,
                    'heading': pano.heading,
                    'degree': degree,
                    'date': pano.date,
                    'location': location
                })
            else:
                print(f"Skipped panorama {pano.id} at location {lat}, {lon} (already downloaded)")
        except Exception as e:
            print(f"Failed to download panorama at location {lat}, {lon}: {e}")

    df = pd.DataFrame(metadata_list)
    df.to_csv('gsv_metadata.csv', index_label='index')

if __name__ == '__main__':
    start_coords = (48.14832898674328, 11.565430441215184)
    end_coords = (48.147932547764434, 11.566883558177446)
    output_dir = "output"
    download_panoramas(start_coords, end_coords, output_dir, num_points=40, location="Munich, Bavaria")
