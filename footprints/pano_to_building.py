from PIL import Image
import os
import cv2
import pandas as pd
import ast
import numpy as np

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


def get_perspective(original_image, FOV, THETA, PHI, height, width):
    img = np.array(original_image)
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #

    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], np.float32)
    K_inv = np.linalg.inv(K)

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz)
    XY = lonlat2XY(lonlat, shape=img.shape).astype(np.float32)
    persp = cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    new_img_pil = Image.fromarray(persp)
    return new_img_pil

            ##########################################################################################################

def save_bounding_boxes_as_images(all_results, image_dir, out_dir):
    building_dir = os.path.join(out_dir, f"detected_building")
    all_data = []

    if not os.path.exists(building_dir):
            os.makedirs(building_dir)

    for _, row in all_results.iterrows():
        pid = row['pid']
        building_id = row['building_id']
        boxes = row['boxes']
        boxes = ast.literal_eval(boxes)

        img_path = os.path.join(image_dir, f'{pid}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            continue 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape  # Correct way to get dimensions
        
        # for idx, box in enumerate(boxes):
        #     # Assuming the box is [x_min, y_min, x_max, y_max] format
        #     # x_center, y_center, box_width, box_height = box[:4]
        #     # width, height = img.size
        # for idx, box in enumerate(boxes):
        x_center, y_center, box_width, box_height = map(float, boxes[:4])

        theta = int((x_center - 0.5) * 360)
        phi = int((1 - y_center  - 0.5) * 180)
        FOV = int(box_width * 360)

        cropped_image = get_perspective(img,
                                        FOV=FOV,
                                        THETA=theta,
                                        PHI=phi,
                                        height=box_height * height * 1.1,
                                        width=box_width  * width * 1.1)
        
        building_url = os.path.join(building_dir,
                                        f"{pid}_bdid_{building_id}.png")
        # cv2.imwrite(building_url, cropped_image)
        cropped_image.save(building_url)

        all_data.append({
            "pid": pid,
            "image_url": img_path,
            "building_id": building_id,
            "building_url": building_url,
        })
        
    df = pd.DataFrame(all_data)

    return df