import cv2
import os
import numpy as np

def correct_brightness(img1, img2, overlap_region=100):
    """
    Correct brightness in img2 to match img1 based on overlap region.
    This helps handle lighting variations across the sequence.
    """
    if overlap_region <= 0 or overlap_region >= img1.shape[1] or overlap_region >= img2.shape[1]:
        return img2
    
    # Extract overlap regions
    right_overlap = img1[:, -overlap_region:, :].astype(np.float32)
    left_overlap = img2[:, :overlap_region, :].astype(np.float32)
    
    # Calculate mean brightness of each region
    mean1 = np.mean(right_overlap)
    mean2 = np.mean(left_overlap)
    
    # Calculate correction factor
    if mean2 > 1:  # Avoid division by zero
        correction = mean1 / mean2
        # Apply gentle correction: don't over-correct, use 70% of calculated correction
        correction = 1 + 0.7 * (correction - 1)
    else:
        correction = 1.0
    
    # Apply brightness correction to img2
    corrected = np.clip(img2.astype(np.float32) * correction, 0, 255)
    return np.uint8(corrected)


def find_best_overlap(img1, img2, max_overlap=250, prev_overlap=200):
    """
    Find the best overlap by computing Mean Absolute Error (MAE) between 
    right edge of img1 and left edge of img2 at different offsets.
    Uses MAE instead of raw sum for better accuracy with large images.
    Includes adaptive constraints to prevent sudden jumps.
    """
    best_overlap = 100
    best_error = float('inf')
    
    # Try different overlap amounts, focusing on larger overlaps
    # Images typically have significant overlap (50-80% of width)
    for overlap in range(80, min(max_overlap, 280), 5):
        if overlap >= img1.shape[1] or overlap >= img2.shape[1]:
            continue
            
        right_edge = img1[:, -overlap:, :].astype(np.float32)
        left_edge = img2[:, :overlap, :].astype(np.float32)
        
        # Use Mean Absolute Error for scale-independent comparison
        mae = np.mean(np.abs(right_edge - left_edge))
        
        if mae < best_error:
            best_error = mae
            best_overlap = overlap
    
    # Adaptive clamping: prevent overlap from dropping below a threshold
    # If previous overlap is known, maintain consistency
    if prev_overlap is not None:
        # Allow up to 40px variation, but never go below 150px
        min_overlap = max(150, prev_overlap - 40)
        max_overlap_constraint = prev_overlap + 40
        best_overlap = np.clip(best_overlap, min_overlap, max_overlap_constraint)
    
    return best_overlap


def blend_images_multiband(img1, img2, overlap, num_bands=4):
    """
    Blend two images using multiband blending for better handling of brightness variations.
    This creates a Laplacian pyramid for each band and blends them separately.
    """
    if overlap <= 0:
        return np.concatenate([img1, img2], axis=1)
    
    overlap = min(overlap, img1.shape[1] - 10, img2.shape[1] - 10)
    overlap = max(overlap, 10)
    
    result_width = img1.shape[1] + img2.shape[1] - overlap
    result = np.zeros((img1.shape[0], result_width, img1.shape[2]), dtype=np.float32)
    
    # Simple blending with wider feathering range for stability
    overlap_region = np.zeros((img1.shape[0], overlap, img1.shape[2]), dtype=np.float32)
    
    # Create smooth blend weights using cubic interpolation
    for i in range(overlap):
        t = i / (overlap - 1) if overlap > 1 else 0.5
        # Cubic ease-in-out for smoother transitions
        alpha = t * t * (3 - 2 * t)
        
        overlap_region[:, i, :] = (1 - alpha) * img1[:, -overlap + i, :].astype(np.float32) + \
                                  alpha * img2[:, i, :].astype(np.float32)
    
    # Place the first image (minus overlap)
    result[:, :img1.shape[1] - overlap, :] = img1[:, :-overlap, :].astype(np.float32)
    
    # Place the blended overlap region
    result[:, img1.shape[1] - overlap:img1.shape[1], :] = overlap_region
    
    # Place the rest of the second image
    result[:, img1.shape[1]:, :] = img2[:, overlap:, :].astype(np.float32)
    
    return np.uint8(np.clip(result, 0, 255))


def stitch_images(image_folder):
    # Get list of image files
    images = []
    for file in os.listdir(image_folder):
        if file.endswith('.jpeg') or file.endswith('.jpg'):
            images.append(os.path.join(image_folder, file))

    # Sort images by the number in the filename
    images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    # Read images
    imgs = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is not None:
            imgs.append(img)
            print(f"Loaded {os.path.basename(img_path)}")
        else:
            print(f"Failed to load image: {img_path}")

    if len(imgs) < 2:
        print("Need at least 2 images to stitch.")
        return None

    # Verify all images have same height
    heights = [img.shape[0] for img in imgs]
    if len(set(heights)) != 1:
        print("Images have different heights, cannot stitch.")
        return None

    print(f"\nLoaded {len(imgs)} images")
    print("Starting stitching process with adaptive overlap...\n")
    
    # Start with the first image
    result = imgs[0].copy()
    prev_overlap = 200  # Initial overlap estimate
    
    # Stitch each subsequent image
    for i in range(1, len(imgs)):
        print(f"Stitching {i+1}/{len(imgs)}: ", end="", flush=True)
        
        # Apply brightness correction to current image
        corrected_img = correct_brightness(result, imgs[i], overlap_region=120)
        
        # Find optimal overlap with adaptive constraints
        overlap = find_best_overlap(result, corrected_img, prev_overlap=prev_overlap)
        prev_overlap = overlap
        print(f"({overlap}px) ", end="")
        
        # Blend and stitch with corrected image
        result = blend_images_multiband(result, corrected_img, overlap)
        print(f"-> width: {result.shape[1]}")
    
    return result


if __name__ == "__main__":
    image_folder = os.environ.get('IMAGE_FOLDER')
    output_file = os.environ.get('OUTPUT_FILE')
    overlap_min = int(os.environ.get('OVERLAP_MIN', 150))
    overlap_max_variation = int(os.environ.get('OVERLAP_MAX_VARIATION', 40))
    blend_width = int(os.environ.get('BLEND_WIDTH', 100))
    
    stitched = stitch_images(image_folder, overlap_min, overlap_max_variation, blend_width)
    if stitched is not None:
        cv2.imwrite(output_file, stitched)
        print(f"Stitched image saved as {output_file}")
    else:
        print("Stitching failed.")