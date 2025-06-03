import os
import numpy as np
import pandas as pd
import math
from streetlevel import streetview

# Example metadata entry:
# {'pid': '1GOMlCg2FacDnCO2BADLlA', 'lat': 48.14795019438397, 'lng': 11.56683288779934, 'heading': 1.9705131042820878, 'degree': 112.9020843505859, 'date': '2023-09', 'location': 'Munich, Bavaria'}

# Example Coordinates
start_lat, start_lon = 48.14968893075899, 11.5693990761673
end_lat, end_lon = 48.14875023979149, 11.568787791142347

# interpolate 40 points between start and end
latitudes = np.linspace(start_lat, end_lat, 40)
longitudes = np.linspace(start_lon, end_lon, 40)

output_dir = "output_tum"
os.makedirs(output_dir, exist_ok=True)

metadata_list = []
downloaded_pano_ids = set()

for i, (lat, lon) in enumerate(zip(latitudes, longitudes), start=1):
    try:
        pano = streetview.find_panorama(lat, lon)
        if pano.id not in downloaded_pano_ids:
            filename = os.path.join(output_dir, f"{pano.id}.jpg")
            streetview.download_panorama(pano, filename)
            downloaded_pano_ids.add(pano.id)
            print(f"Downloaded panorama {pano.id} at ({lat:.6f}, {lon:.6f}) to {filename}")

            degree = math.degrees(pano.heading)
            metadata_list.append({
                'pid': pano.id,
                'lat': lat,
                'lng': lon,
                'heading': pano.heading,
                'degree': degree,
                'date': pano.date,
                'location': "Munich, Bavaria"
            })
        else:
            print(f"Skipped {pano.id} at ({lat:.6f}, {lon:.6f}) — already downloaded")
    except Exception as e:
        print(f"Failed at ({lat:.6f}, {lon:.6f}): {e}")

# save metadata
df = pd.DataFrame(metadata_list)
df.to_csv('gsv_metadata.csv', index_label='index')
