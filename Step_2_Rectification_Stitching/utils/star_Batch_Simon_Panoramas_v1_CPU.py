import os
import glob
import json
import time
import multiprocessing

# Third-party imports
import numpy as np
from PIL import Image
import skimage.io
import matplotlib.pyplot as plt

# Local module imports
from default_params import default_params
from Panos.Pano_rectification import simon_rectification
from Panos.Pano_project import project_face, stitch_tiles, render_imgs, project_facade_for_refine
from Panos.Pano_new_pano import create_new_panorama, draw_new_panorama
from Panos.Pano_visualization import (
    R_heading, draw_all_vp_and_hl_color, draw_all_vp_and_hl_bi,
    draw_zenith_on_top_color, draw_zenith_on_top_bi, draw_sphere_zenith,
    R_roll, R_pitch
)
from Panos.Pano_zp_hvp import calculate_consensus_zp
from Panos.Pano_consensus_vis import (
    draw_consensus_zp_hvps, draw_consensus_rectified_sphere,
    draw_center_hvps_rectified_sphere, draw_center_hvps_on_panorams
)
from Panos.Pano_histogram import calculate_histogram
import Pano_hvp

# Configuration Parameters
PLOT_REDUNDANT = False
SAVE_DIRECTLY = True

ROOT_DIR = 'Pano_new'
COUNTRY_CITY = 'TUM'
NEW_COUNT = 3
TMP_COUNT = str(NEW_COUNT)

IMG_FOLDER = os.path.join(ROOT_DIR, COUNTRY_CITY, 'img_tum_v3/')
INTER_DIR = os.path.join(ROOT_DIR, 'Pano_hl_z_vp_v3/')
RENDERING_OUTPUT_FOLDER = os.path.join(ROOT_DIR, COUNTRY_CITY, 'Rendering_tum_v3/')
THREAD_NUM = 1
TASK = 'tum_building_v3/'
THREAD_DIR = f"{THREAD_NUM}/"

# Ensure rendering output directory exists
os.makedirs(RENDERING_OUTPUT_FOLDER, exist_ok=True)

def clean_tmp_folder(tmp_folder):
    """Remove all .jpg files from the temporary folder."""
    removelist = glob.glob(os.path.join(tmp_folder, '*.jpg'))
    for file_path in removelist:
        os.remove(file_path)

def process_image(im_path):
    """Process a single image for panorama analysis."""
    print(im_path)
    
    # Open image
    im = Image.open(im_path)
    rendering_img_base = os.path.join(
        RENDERING_OUTPUT_FOLDER, 
        os.path.splitext(os.path.basename(im_path))[0]
    )
    
    # Setup temporary folder
    tmp_folder = os.path.join(ROOT_DIR, COUNTRY_CITY, 'tmp', TASK, THREAD_DIR)
    os.makedirs(tmp_folder, exist_ok=True)
    # clean_tmp_folder(tmp_folder)
    
    # Render images from panorama
    panorama_img = skimage.io.imread(im_path)
    tilelist = render_imgs(panorama_img, tmp_folder, SAVE_DIRECTLY)


    ########### save each tile directly to the tmp folder ###########
    if not SAVE_DIRECTLY:
        tilelist = sorted(glob.glob(os.path.join(tmp_folder, '*.jpg')))
    else:
        for i, tile in enumerate(tilelist):
            try:
                tile_image = Image.fromarray(tile)  # Ensure tile is a valid NumPy array
                tile_save_path = os.path.join(tmp_folder, f"tile_{i:03d}.jpg")
                tile_image.save(tile_save_path)
                print(f"Saved tile: {tile_save_path}")
            except Exception as e:
                print(f"Error saving tile {i}: {e}")

    # Verify if the files exist
    print(f"Files in tmp_folder: {os.listdir(tmp_folder)}")
    ########### save each tile directly to the tmp folder ###########
    
    if not SAVE_DIRECTLY:
        tilelist = sorted(glob.glob(os.path.join(tmp_folder, '*.jpg')))
    
    # Initialize lists to collect rectification results
    hl, hvps, hvp_groups = [], [], []
    z, z_group, ls = [], [], []
    z_homo, hvp_homo, ls_homo = [], [], []
    
    # Rectify each tile
    for i, tile in enumerate(tilelist):
        rectification = simon_rectification(tile, i, INTER_DIR, ROOT_DIR, TMP_COUNT)
        (tmp_hl, tmp_hvps, tmp_hvp_groups, tmp_z, tmp_z_group, 
         tmp_ls, tmp_z_homo, tmp_hvp_homo, tmp_ls_homo, params) = rectification
         
        hl.append(tmp_hl)
        hvps.append(tmp_hvps)
        hvp_groups.append(tmp_hvp_groups)
        z.append(tmp_z)
        z_group.append(tmp_z_group)
        ls.append(tmp_ls)
        z_homo.append(tmp_z_homo)
        hvp_homo.append(tmp_hvp_homo)
        ls_homo.append(tmp_ls_homo)
    
    # Clean temporary folder after rectification
    # clean_tmp_folder(tmp_folder)
    
    # Calculate zenith points
    zenith_points = np.array([
        R_heading(np.pi / 2 * (i - 1)).dot(zenith) 
        for i, zenith in enumerate(z_homo)
    ])
    hv_points = [
        (R_heading(np.pi / 2 * (i - 1)).dot(hv_p.T)).T 
        for i, hv_p in enumerate(hvp_homo)
    ]
    
    if PLOT_REDUNDANT:
        draw_all_vp_and_hl_color(zenith_points, hv_points, im.copy(), ROOT_DIR)
        draw_all_vp_and_hl_bi(zenith_points, hv_points, im.copy(), ROOT_DIR)
        draw_sphere_zenith(zenith_points, hv_points, ROOT_DIR)
    
    # Calculate consensus zenith point
    zenith_consensus, best_zenith = calculate_consensus_zp(zenith_points, method='svd')
    zenith_consensus_org = np.array([
        R_heading(-np.pi / 2 * (i - 1)).dot(zenith) 
        for i, zenith in enumerate(zenith_consensus)
    ])
    
    # Obtain params instance
    params_instance = default_params()
    
    # Get consensus horizontal vanishing points (HVPs)
    result_list = [
        Pano_hvp.get_all_hvps(ls_homo[i], zenith_consensus_org[i], params_instance)
        for i in range(len(zenith_consensus_org))
    ]
    hvps_consensus_org = [result for result in result_list]
    hvps_consensus_uni = [
        (R_heading(np.pi / 2 * (i - 1)).dot(hv_p.T)).T 
        for i, hv_p in enumerate(hvps_consensus_org)
    ]
    
    if PLOT_REDUNDANT:
        draw_consensus_zp_hvps(best_zenith, hvps_consensus_uni, im.copy(), ROOT_DIR)
    
    # Calculate pitch and roll
    pitch = np.arctan(best_zenith[2] / best_zenith[1])
    roll = -np.arctan(best_zenith[0] / (np.sign(best_zenith[1]) * 
                     np.hypot(best_zenith[1], best_zenith[2])))
    
    # Rectify HVPs based on pitch and roll
    hvps_consensus_rectified = [
        R_roll(-roll).dot(R_pitch(-pitch).dot(vp.T)).T 
        for vp in hvps_consensus_uni
    ]
    
    if PLOT_REDUNDANT:
        draw_consensus_rectified_sphere(hvps_consensus_rectified, ROOT_DIR)
    
    # Calculate histogram of rectified HVPs
    final_hvps_rectified = calculate_histogram(hvps_consensus_rectified, ROOT_DIR, PLOT_REDUNDANT)
    
    if PLOT_REDUNDANT:
        draw_center_hvps_rectified_sphere(np.array(final_hvps_rectified), ROOT_DIR)
        draw_center_hvps_on_panorams(
            best_zenith, 
            np.array(final_hvps_rectified), 
            im.copy(), 
            pitch, 
            roll, 
            ROOT_DIR
        )
        # Optionally draw the new panorama
        new_pano_path = create_new_panorama(im_path, pitch, roll, ROOT_DIR)
        draw_new_panorama(new_pano_path, np.array(final_hvps_rectified), ROOT_DIR)
    
    # Render images from panoramas
    project_facade_for_refine(
        np.array(final_hvps_rectified), 
        im.copy(), 
        pitch, 
        roll, 
        im_path, 
        ROOT_DIR, 
        tmp_folder, 
        rendering_img_base, 
        TMP_COUNT
    )
    print("Processing complete for:", im_path)

def main():
    """Main function to process all images."""
    image_list = sorted(glob.glob(os.path.join(IMG_FOLDER, '*.jpg')))
    
    for im_path in image_list:
        process_image(im_path)
    
    print("All images have been processed successfully.")

if __name__ == "__main__":
    main()
