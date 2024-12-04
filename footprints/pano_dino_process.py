import os
import cv2
import csv
import io
import math
import warnings
from PIL import Image
from groundingdino.util.inference import load_image, predict
import random
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def generate_random_color():
    """Generate a random RGB color."""
    return [random.randint(0, 255) for _ in range(3)]

def check_id(save_folder):
    """Retrieve the set of IDs from filenames in the specified folder."""
    ids = set()
    for name in os.listdir(save_folder):
        if not name.startswith('.'):
            ids.add(os.path.splitext(name)[0])
    return ids

def filter_boxes_based_on_conditions(results, cropped_image):
    """Filter boxes to keep the one that contains the center and has the highest confidence."""
    img_width = cropped_image.shape[1]
    img_height = cropped_image.shape[0]
    central_x = img_width / 2

    # Convert relative box coordinates to absolute pixel coordinates
    results['boxes'] = [
        [
            (box[0] - 0.5 * box[2]) * img_width,
            (box[1] - 0.5 * box[3]) * img_height,
            (box[0] + 0.5 * box[2]) * img_width,
            (box[1] + 0.5 * box[3]) * img_height
        ]
        for box in results['boxes']
    ]

    # Keep boxes that contain the central x-coordinate
    keep_indices = [
        i for i, box in enumerate(results['boxes'])
        if box[0] <= central_x <= box[2]
    ]

    # If multiple boxes, keep the one with the highest confidence (logit)
    if len(keep_indices) >= 2:
        filtered_logits = [results['logits'][i] for i in keep_indices]
        max_logit_index = filtered_logits.index(max(filtered_logits))
        keep_indices = [keep_indices[max_logit_index]]
    
    filtered_boxes = [results['boxes'][i] for i in keep_indices]
    filtered_logits = [results['logits'][i] for i in keep_indices]

    return filtered_boxes, filtered_logits

def process_images(geo_fov, DATA_FOLDER, out_dir, TEXT_PROMPT, BOX_THRESHOLD, model, adjust_ag):
    TEXT_THRESHOLD = 0.3
    annotated_dir = os.path.join(out_dir, 'annotated_image')
    os.makedirs(annotated_dir, exist_ok=True)

    geo_fov['pid'] = geo_fov['pid'].astype(str)
    pid_csv = set(geo_fov['pid'].unique())
    already_id = check_id(annotated_dir)
    pid_all = check_id(DATA_FOLDER)
    remaining_pid = (pid_csv & pid_all) - already_id
    print(f'Total images: {len(pid_all)}, Unfinished: {len(pid_all - already_id)}')

    for pid in tqdm(remaining_pid, desc='Processing images', dynamic_ncols=True, leave=False):
        img_path = os.path.join(DATA_FOLDER, f'{pid}.jpg')
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            # print('No image found for:', pid)
            continue

        img_geo_fov = geo_fov[geo_fov['pid'] == pid]

        for _, row in img_geo_fov.iterrows():
            building_id = row['building_id']
            left_bd = row['left_angle_image']
            right_bd = row['right_angle_image']
            color = generate_random_color()

            img_height, img_width = img.shape[:2]
            if math.isnan(left_bd) or math.isnan(right_bd):
                continue
            x_left = max(int(left_bd / 360 * img_width - adjust_ag * img_width), 0)
            x_right = min(int(right_bd / 360 * img_width + adjust_ag * img_width), img_width)

            if x_right <= x_left:
                # print("Invalid crop dimensions for building_id:", building_id)
                continue

            cropped_image = img[:, x_left:x_right]
            pil_image = Image.fromarray(cropped_image)
            image_io = io.BytesIO()
            pil_image.save(image_io, format='PNG')
            image_io.seek(0)
            _, image = load_image(image_io)

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

            results = {
                'boxes': boxes.tolist(),
                'logits': logits.tolist(),
                'phrases': phrases
            }

            filtered_boxes, filtered_logits = filter_boxes_based_on_conditions(results, cropped_image)
            if not filtered_boxes:
                continue

            # Adjust boxes to the original image coordinates
            adjusted_boxes = [
                [box[0] + x_left, box[1], box[2] + x_left, box[3]]
                for box in filtered_boxes
            ]

            x1, y1, x2, y2 = map(int, adjusted_boxes[0][:4])
            conf = filtered_logits[0]

            # Draw rectangle and text annotation
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f'ID: {building_id}, Conf: {conf:.2f}'
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Normalize bounding box coordinates
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            cx_normalized = cx / img_width
            cy_normalized = cy / img_height
            w_normalized = w / img_width
            h_normalized = h / img_height

            normalized_box = [cx_normalized, cy_normalized, w_normalized, h_normalized, conf]

            result_row = {
                'pid': pid,
                'building_id': building_id,
                'boxes': normalized_box
            }

            # Write results to CSV
            file_path = os.path.join(out_dir, 'detected_buildings.csv')
            file_exists = os.path.isfile(file_path)
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(result_row.keys())
                writer.writerow(result_row.values())

        save_path = os.path.join(annotated_dir, f'{pid}.jpg')
        cv2.imwrite(save_path, img)