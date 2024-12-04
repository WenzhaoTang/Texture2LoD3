import os
import sys
from PIL import Image, ImageOps
import numpy as np
from utils.semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator
import torch
import clip
from torchvision import transforms
import cv2 

def apply_mask_and_crop(original_image, mask_array, padding=10):
    mask = Image.fromarray((mask_array * 255).astype(np.uint8))
    bbox = mask.getbbox()
    if bbox is None:
        return None

    if isinstance(original_image, Image.Image):
        width, height = original_image.size
    elif isinstance(original_image, np.ndarray):
        height, width = original_image.shape[:2]
    else:
        raise TypeError("Unsupported image type")

    left = max(bbox[0] - padding, 0)
    upper = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, width)
    lower = min(bbox[3] + padding, height)

    if isinstance(original_image, Image.Image):
        cropped_image = original_image.crop((left, upper, right, lower))
    else:
        cropped_image = original_image[upper:lower, left:right]

    mask_cropped = mask.crop((left, upper, right, lower))
    mask_cropped_np = np.array(mask_cropped) / 255.0
    mask_3c = np.stack([mask_cropped_np] * 3, axis=-1)

    if isinstance(cropped_image, Image.Image):
        cropped_image_np = np.array(cropped_image)
    else:
        cropped_image_np = cropped_image

    masked_image_np = cropped_image_np * mask_3c
    masked_image = Image.fromarray(masked_image_np.astype(np.uint8))
    return masked_image

def combine_masks(masks_list, height, width):
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in masks_list:
        combined_mask = np.logical_or(combined_mask, mask['segmentation'])
    return combined_mask.astype(np.uint8)

def subtract_masks(base_mask, subtract_mask):
    return np.logical_and(base_mask, np.logical_not(subtract_mask)).astype(np.uint8)

def apply_final_mask(original_image, final_mask):
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image

    final_mask_3c = np.stack([final_mask] * 3, axis=-1)
    masked_image_np = original_np * final_mask_3c
    return Image.fromarray(masked_image_np.astype(np.uint8))

def remove_small_artefacts(mask, min_size=500):
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    cleaned_mask = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 255
    cleaned_mask_binary = (cleaned_mask > 0).astype(np.uint8)
    return cleaned_mask_binary

def generate_white_black_mask_from_mask(mask_array, output_path):
    binary_mask = (mask_array * 255).astype(np.uint8)
    binary_mask_image = Image.fromarray(binary_mask)
    binary_mask_image.save(output_path)
    print(f"White-Black mask saved to {output_path}")

def main():
    image_path = 'examples_2/dfw7DepxqMV1d2xfCDCv2Q_VP_0_1.jpg'
    ckpt_path = 'ckpts/swinl_only_sam_many2many.pth'
    save_path = '/home/tang/code/Semantic-SAM/vis_v3'

    model_type = 'L'
    top_k_masks = 20 # Update
    padding = 10
    CONFIDENCE_THRESHOLD = 0.1

    min_artefact_size = 500
    mask_threshold = 128

    class_names = [
        'facade of a building',
        'building eave',
        'clear sky',
        'tree',
        'road',
        'car',
        'building',
        'person',
        'window',
        'door'
    ]

    os.makedirs(save_path, exist_ok=True)

    try:
        original_image, input_image = prepare_image(image_pth=image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    try:
        mask_generator = SemanticSamAutomaticMaskGenerator(
            build_semantic_sam(model_type=model_type, ckpt=ckpt_path)
        )
    except Exception as e:
        print(f"Error initializing Semantic SAM: {e}")
        sys.exit(1)

    print("Generating masks...")
    masks = mask_generator.generate(input_image)
    print(f"Number of masks generated: {len(masks)}")

    masks = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)[:top_k_masks]
    print(f"Number of masks after filtering: {len(masks)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        model_clip, preprocess_clip = clip.load("ViT-L/14", device=device)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        sys.exit(1)

    text_prompts = [f"a photograph of {c}" for c in class_names]
    text_tokens = clip.tokenize(text_prompts).to(device)

    facade_masks = []
    eave_masks = []

    print(f"Saving results to {save_path}...")
    for idx, mask in enumerate(masks):
        mask_array = mask.get('segmentation', None)
        if mask_array is None:
            print(f"Mask {idx}: No segmentation found. Skipping.")
            continue

        masked_image = apply_mask_and_crop(original_image, mask_array, padding=padding)
        if masked_image is None:
            print(f"Mask {idx}: Empty mask. Skipping.")
            continue

        try:
            input_image_clip = preprocess_clip(masked_image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Mask {idx}: Error in preprocessing for CLIP: {e}. Skipping.")
            continue

        with torch.no_grad():
            image_features = model_clip.encode_image(input_image_clip)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            text_features = model_clip.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(3)

        if values[0].item() >= CONFIDENCE_THRESHOLD:
            predicted_class = class_names[indices[0].item()]
            confidence = values[0].item()
        else:
            predicted_class = 'unknown'
            confidence = values[0].item()

        print(f"Mask {idx}: Predicted class: {predicted_class} with confidence {confidence:.4f}")

        sanitized_class = predicted_class.replace(' ', '_')
        output_path = os.path.join(save_path, f"mask_{idx}_{sanitized_class}.png")
        try:
            masked_image.save(output_path)
            print(f"Saving mask {idx} to {output_path}")
        except Exception as e:
            print(f"Mask {idx}: Error saving image: {e}")

        mask['predicted_class'] = predicted_class
        mask['confidence'] = confidence

        if predicted_class == 'facade of a building':
            facade_masks.append(mask)
        elif predicted_class == 'building eave':
            eave_masks.append(mask)

    if isinstance(original_image, Image.Image):
        width, height = original_image.size
    elif isinstance(original_image, np.ndarray):
        height, width = original_image.shape[:2]
    else:
        raise TypeError("Unsupported image type for original_image")

    if not facade_masks:
        print("No facade masks found. Exiting.")
        sys.exit(0)

    print("Combining facade masks...")
    facade_combined_mask = combine_masks(facade_masks, height, width)

    print("Combining eave masks...")
    if eave_masks:
        eave_combined_mask = combine_masks(eave_masks, height, width)
    else:
        eave_combined_mask = np.zeros((height, width), dtype=np.uint8)
        print("No eave masks found. Proceeding without eave subtraction.")

    print("Subtracting eave masks from facade masks...")
    facade_without_eave_mask = subtract_masks(facade_combined_mask, eave_combined_mask)

    print("Filtering small floating artefacts from the mask...")
    facade_cleaned_mask = remove_small_artefacts(facade_without_eave_mask, min_size=min_artefact_size)
    print("Small artefacts removed.")

    white_black_mask_path = os.path.join(save_path, "facade_without_eave_mask.png")
    try:
        generate_white_black_mask_from_mask(facade_cleaned_mask, white_black_mask_path)
    except Exception as e:
        print(f"Error generating white-black mask: {e}")

    print("Applying final mask to the original image...")
    final_image = apply_final_mask(original_image, facade_cleaned_mask)

    final_output_path = os.path.join(save_path, "facade_without_eave.png")
    try:
        final_image.save(final_output_path)
        print(f"Facade without eave saved to {final_output_path}")
    except Exception as e:
        print(f"Error saving final facade image: {e}")

    print("Creating overlay image for visualization...")
    try:
        overlay = original_image.copy() if isinstance(original_image, Image.Image) else Image.fromarray(original_image)
        overlay_np = np.array(overlay)
        final_mask_binary = facade_cleaned_mask > 0
        overlay_np[final_mask_binary] = [0, 255, 0]
        overlay_image = Image.fromarray(overlay_np)
        overlay_output_path = os.path.join(save_path, "facade_without_eave_overlay.png")
        overlay_image.save(overlay_output_path)
        print(f"Overlay image saved to {overlay_output_path}")
    except Exception as e:
        print(f"Error creating overlay image: {e}")

    try:
        plot_results(masks, original_image, save_path=save_path)
        print("Visualized mask overlays saved.")
    except Exception as e:
        print(f"Error visualizing results: {e}")

    print("All results saved successfully.")

if __name__ == "__main__":
    main()
