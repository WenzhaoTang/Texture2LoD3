import os
import sys
import torch
import clip
import numpy as np
import cv2
from PIL import Image, ImageOps
from torchvision import transforms
from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator

def resize_mask_to_original(mask, original_image):
    if isinstance(original_image, Image.Image):
        target_size = original_image.size
    elif isinstance(original_image, np.ndarray):
        height, width = original_image.shape[:2]
        target_size = (width, height)
    else:
        raise TypeError("Unsupported type for original_image")
    
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img_resized = mask_img.resize(target_size, resample=Image.NEAREST)
    resized_mask = (np.array(mask_img_resized) > 127).astype(np.uint8)
    return resized_mask

def apply_mask_full(original_image, mask_array):
    mask = Image.fromarray((mask_array * 255).astype(np.uint8))
    if isinstance(original_image, Image.Image):
        orig_size = original_image.size
    elif isinstance(original_image, np.ndarray):
        orig_size = (original_image.shape[1], original_image.shape[0])
    else:
        raise TypeError("Unsupported type for original_image")
    
    if mask.size != orig_size:
        mask = mask.resize(orig_size, resample=Image.NEAREST)
    mask_np = np.array(mask) / 255.0
    mask_3c = np.stack([mask_np] * 3, axis=-1)

    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image

    if original_np.shape[:2] != mask_np.shape[:2]:
        raise ValueError("Mask and image sizes do not match after resizing.")
    masked_image_np = original_np * mask_3c
    return Image.fromarray(masked_image_np.astype(np.uint8))

def combine_masks(masks_list, target_size, original_image):
    if isinstance(original_image, Image.Image):
        width, height = original_image.size
    elif isinstance(original_image, np.ndarray):
        height, width = original_image.shape[:2]
    else:
        raise TypeError("Unsupported type for original_image")
    
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in masks_list:
        mask_seg = mask.get('segmentation', None)
        if mask_seg is None:
            continue
        mask_seg = np.array(mask_seg)
        if mask_seg.ndim < 2:
            continue
        if mask_seg.shape[0] != height or mask_seg.shape[1] != width:
            mask_seg = resize_mask_to_original(mask_seg, original_image)
        combined_mask = np.logical_or(combined_mask, mask_seg)
    return combined_mask.astype(np.uint8)

def subtract_masks(base_mask, subtract_mask):
    return np.logical_and(base_mask, np.logical_not(subtract_mask)).astype(np.uint8)

def apply_final_mask(original_image, final_mask):
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image

    if final_mask.shape != original_np.shape[:2]:
        final_mask = resize_mask_to_original(final_mask, original_image)
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

def smooth_mask(mask, kernel_size=5, iterations=1):
    if mask.max() <= 1:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    smooth = (opened > 127).astype(np.uint8)
    return smooth

def generate_white_black_mask_from_mask(mask_array, output_path):
    binary_mask = (mask_array * 255).astype(np.uint8)
    binary_mask_image = Image.fromarray(binary_mask)
    binary_mask_image.save(output_path)
    print(f"White-black mask saved to {output_path}")

def create_semantic_map(masks, original_image, class_color_dict):
    if isinstance(original_image, Image.Image):
        width, height = original_image.size
    elif isinstance(original_image, np.ndarray):
        height, width = original_image.shape[:2]
        original_image_np = original_image
    else:
        raise TypeError("Unsupported type for original_image")
    
    semantic_map = np.zeros((height, width, 3), dtype=np.uint8)

    for mask_info in masks:
        mask_array = mask_info.get('segmentation')
        if mask_array is None:
            continue
        if mask_array.shape[0] != height or mask_array.shape[1] != width:
            mask_array = resize_mask_to_original(mask_array, original_image)
        predicted_class = mask_info.get('predicted_class', 'unknown')
        color = class_color_dict.get(predicted_class, (255, 255, 255))
        semantic_map[mask_array == 1] = color
    
    return Image.fromarray(semantic_map)

def main():
    image_path = 'run_tex_extraction/dfw7DepxqMV1d2xfCDCv2Q_VP_0_1.jpg'
    ckpt_path = 'ckpts/swinl_only_sam_many2many.pth' # need to be manually downloaded
    save_path = 'run_tex_extraction/output' # update
    os.makedirs(save_path, exist_ok=True)

    model_type = 'L'
    top_k_masks = 127
    CONFIDENCE_THRESHOLD = 0.05
    min_artefact_size = 200

    class_names = [
        'building facade',
        'building eave',
        'clear sky',
        'tree',
        'road',
        'car',
        'person',
        'window',
        'door'
    ]

    class_color_dict = {
        'building facade': (255, 179, 186),
        'building eave': (255, 223, 186),
        'clear sky': (255, 255, 186),
        'tree': (186, 255, 201),
        'road': (186, 225, 255),
        'car': (255, 201, 255),
        'person': (204, 255, 229),
        'window': (255, 204, 229),
        'door': (229, 204, 255),
        'unknown': (220, 220, 220)
    }

    try:
        full_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    try:
        _, input_image = prepare_image(image_pth=image_path)
    except Exception as e:
        print(f"Error in prepare_image: {e}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_image = input_image.to(device)

    try:
        model = build_semantic_sam(model_type=model_type, ckpt=ckpt_path)
        model = model.to(device)
        mask_generator = SemanticSamAutomaticMaskGenerator(model)
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    print("Generating masks...")
    try:
        masks = mask_generator.generate(input_image)
    except Exception as e:
        print(f"Error during mask generation: {e}")
        sys.exit(1)
    masks = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)[:top_k_masks]

    try:
        model_clip, preprocess_clip = clip.load("ViT-L/14", device=device)
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        sys.exit(1)

    text_prompts = [f"a photograph of {c}" for c in class_names]
    text_tokens = clip.tokenize(text_prompts).to(device)

    facade_masks = []
    eave_masks = []

    print(f"Saving results to {save_path}...")
    for idx, mask in enumerate(masks):
        mask_array = mask.get('segmentation', None)
        if mask_array is None or np.array(mask_array).ndim < 2:
            continue

        try:
            masked_image = apply_mask_full(full_image, np.array(mask_array))
        except Exception as e:
            print(f"Mask {idx}: Error applying mask: {e}")
            continue

        try:
            input_image_clip = preprocess_clip(masked_image).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Mask {idx}: Error preprocessing for CLIP: {e}")
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

        print(f"Mask {idx}: Predicted {predicted_class} ({confidence:.4f})")
        mask['predicted_class'] = predicted_class
        mask['confidence'] = confidence

        # 只保留 building facade，其它一律命名为 others
        if predicted_class == 'building facade':
            file_label = 'building_facade'
        else:
            file_label = 'others'
        output_path = os.path.join(save_path, f"mask_{idx}_{file_label}.png")
        try:
            masked_image.save(output_path)
            print(f"Saved mask {idx} to {output_path}")
        except Exception as e:
            print(f"Mask {idx}: Error saving image: {e}")

        if predicted_class == 'building facade':
            facade_masks.append(mask)
            upscaled_mask = resize_mask_to_original(np.array(mask_array), full_image)
            upscaled_mask_path = os.path.join(save_path, f"mask_{idx}_facade_upscaled.png")
            try:
                generate_white_black_mask_from_mask(upscaled_mask, upscaled_mask_path)
            except Exception as e:
                print(f"Mask {idx}: Error saving upscaled mask: {e}")
        elif predicted_class == 'building eave':
            eave_masks.append(mask)

    semantic_map_pil = create_semantic_map(masks, full_image, class_color_dict)
    semantic_map_path = os.path.join(save_path, "full_size_semantic_map.png")
    semantic_map_pil.save(semantic_map_path)
    print(f"Saved semantic map to {semantic_map_path}")

    if isinstance(full_image, Image.Image):
        width, height = full_image.size
    elif isinstance(full_image, np.ndarray):
        height, width = full_image.shape[:2]
    else:
        raise TypeError("Unsupported type for original_image")

    print("Combining facade masks...")
    facade_combined_mask = combine_masks(facade_masks, (width, height), full_image)

    print("Combining eave masks...")
    if eave_masks:
        eave_combined_mask = combine_masks(eave_masks, (width, height), full_image)
    else:
        eave_combined_mask = np.zeros((height, width), dtype=np.uint8)
        print("No eave masks found.")

    print("Subtracting eave from facade...")
    facade_without_eave_mask = subtract_masks(facade_combined_mask, eave_combined_mask)

    print("Removing small artefacts...")
    facade_cleaned_mask = remove_small_artefacts(facade_without_eave_mask, min_size=min_artefact_size)

    print("Smoothing mask...")
    facade_smoothed_mask = smooth_mask(facade_cleaned_mask, kernel_size=5, iterations=1)

    white_black_mask_path = os.path.join(save_path, "facade_without_eave_mask.png")
    try:
        generate_white_black_mask_from_mask(facade_smoothed_mask, white_black_mask_path)
    except Exception as e:
        print(f"Error saving white-black mask: {e}")

    print("Applying final mask...")
    final_image = apply_final_mask(full_image, facade_smoothed_mask)
    final_output_path = os.path.join(save_path, "facade_without_eave.png")
    try:
        final_image.save(final_output_path)
        print(f"Saved facade-only image to {final_output_path}")
    except Exception as e:
        print(f"Error saving final image: {e}")

    print("Creating overlay visualization...")
    try:
        overlay = full_image.copy()
        overlay_np = np.array(overlay)
        final_mask_binary = facade_smoothed_mask > 0
        overlay_np[final_mask_binary] = [0, 255, 0]
        overlay_image = Image.fromarray(overlay_np)
        overlay_output_path = os.path.join(save_path, "facade_without_eave_overlay.png")
        overlay_image.save(overlay_output_path)
        print(f"Saved overlay to {overlay_output_path}")
    except Exception as e:
        print(f"Error creating overlay: {e}")

    print("Visualizing with plot_results...")
    try:
        plot_results(masks, full_image, save_path=save_path)
        print("Visualization saved.")
    except Exception as e:
        print(f"Error in plot_results: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
