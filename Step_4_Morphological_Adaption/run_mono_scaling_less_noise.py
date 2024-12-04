import cv2
import numpy as np
import argparse
from quadrilateral_fitter.my_quadrilateral_fitter import QuadrilateralFitter
import matplotlib.pyplot as plt

def print_detected_corners(corners):
    print("Detected Corners of the Quadrilateral:")
    for i, corner in enumerate(corners):
        print(f"Corner {i + 1}: {corner}")

def preprocess_mask(binary_mask):
    blurred_mask = cv2.GaussianBlur(binary_mask, (25, 25), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        convex_mask = np.zeros_like(binary_mask)
        cv2.drawContours(convex_mask, [hull], -1, 255, thickness=cv2.FILLED)
        return convex_mask
    else:
        return opened_mask

def main(args):
    image = cv2.imread(args.image_path)
    gray_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
    processed_mask = preprocess_mask(binary_mask)
    cv2.imwrite(args.processed_mask_path, processed_mask)
    print(f"Processed mask saved to: {args.processed_mask_path}")

    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        noisy_corners = np.squeeze(largest_contour)
        if len(noisy_corners) >= 4:
            fitter = QuadrilateralFitter(polygon=noisy_corners)
            fitted_quadrilateral = np.array(fitter.fit(), dtype=np.int32)
            print_detected_corners(fitted_quadrilateral)
            scaled_corners = fitted_quadrilateral * args.scaling_factor
            print("\nScaled Corners of the Quadrilateral (after multiplication):")
            for i, corner in enumerate(scaled_corners):
                print(f"Corner {i + 1}: {corner}")
            original_image = cv2.imread(args.original_image_path)
            height, width, _ = original_image.shape
            target_corners = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            scaled_corners = np.array(scaled_corners, dtype=np.float32)
            perspective_matrix = cv2.getPerspectiveTransform(scaled_corners, target_corners)
            warped_image = cv2.warpPerspective(original_image, perspective_matrix, (width, height))
            resized_image = cv2.resize(warped_image, (args.final_width, args.final_height), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(args.output_resized_path, resized_image)
            print(f"\nResized warped quadrilateral saved to: {args.output_resized_path}")
            plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            plt.title("Resized Warped Quadrilateral")
            plt.axis("off")
            plt.show()
        else:
            print("The contour does not have enough points to fit a quadrilateral.")
    else:
        print("No valid contours found in the image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and scale quadrilateral in an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("processed_mask_path", type=str, help="Path to save the processed mask.")
    parser.add_argument("original_image_path", type=str, help="Path to the original image.")
    parser.add_argument("output_resized_path", type=str, help="Path to save the resized warped quadrilateral.")
    parser.add_argument("--scaling_factor", type=float, default=2.0516, help="Scaling factor for the quadrilateral.")
    parser.add_argument("--final_width", type=int, default=2048, help="Final width of the resized image.")
    parser.add_argument("--final_height", type=int, default=1313, help="Final height of the resized image.")
    args = parser.parse_args()
    main(args)
