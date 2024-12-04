import pandas as pd
import os, sys
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
import json
from groundingdino.util.inference import load_model
from pano_dino_process_v1 import process_images
from pano_to_building import save_bounding_boxes_as_images

model = load_model("Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py", 
                   "groundingdino_swinb_cogcoor.pth")

geo_fov = pd.read_csv(f'data/02_geofov/geofov_new.csv')
geo_fov['left_angle_image'] = (geo_fov['left_angle_geo'] - geo_fov['degree'] + 180) % 360
geo_fov['right_angle_image'] = (geo_fov['right_angle_geo'] - geo_fov['degree'] + 180) % 360

image_dir = f'data/01_img'
out_dir = f'output/pano_dino'
os.makedirs(out_dir, exist_ok= True)

process_images(
    geo_fov=geo_fov,
    DATA_FOLDER=image_dir,
    out_dir=out_dir,
    TEXT_PROMPT= "building",
    BOX_THRESHOLD=0.25,
    model=model,
    adjust_ag=10 / 360
)

all_results = pd.read_csv(os.path.join(out_dir, 'detected_buildings.csv'))
df = save_bounding_boxes_as_images(all_results, image_dir, out_dir)
df.to_csv(os.path.join(out_dir, 'image_urls.csv'))