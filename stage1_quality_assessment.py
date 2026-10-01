"""
Stage 1: Image Quality Assessment & Enhancement
- Checks blur, illumination, and field of view
- Enhances gradeable images with CLAHE + denoising
- Flags ungradeable images for recapture
"""

import cv2
import numpy as np


def check_blur(image, threshold=100.0):
    """Returns (is_sharp, score). Higher score = sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold, score


def check_illumination(image, low=40, high=220):
    """Returns (is_well_lit, mean_brightness)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    return low <= mean_brightness <= high, mean_brightness


def check_field_of_view(image, min_coverage=0.35):
    """
    Checks how much of the image is covered by the circular retinal
    field of view (vs black background), by thresholding brightness.
    Returns (is_full_fov, coverage_ratio).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coverage = np.count_nonzero(mask) / mask.size
    return coverage >= min_coverage, coverage


def assess_quality(image, blur_threshold=100.0, illum_range=(40, 220), fov_min=0.35):
    """
    Runs all three checks. Returns a dict with pass/fail per check
    and an overall 'gradeable' boolean.
    """
    is_sharp, blur_score = check_blur(image, blur_threshold)
    is_lit, brightness = check_illumination(image, *illum_range)
    is_full_fov, coverage = check_field_of_view(image, fov_min)

    gradeable = is_sharp and is_lit and is_full_fov

    return {
        "gradeable": gradeable,
        "blur": {"pass": is_sharp, "score": round(float(blur_score), 2)},
        "illumination": {"pass": is_lit, "score": round(float(brightness), 2)},
        "field_of_view": {"pass": is_full_fov, "score": round(float(coverage), 3)},
    }


def enhance_image(image, clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
    """
    Applies CLAHE on the green channel (best contrast for retinal
    vessels/lesions) plus mild denoising. Returns enhanced BGR image.
    """
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    g_enhanced = clahe.apply(g)
    enhanced = cv2.merge((b, g_enhanced, r))
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    return enhanced


def process_image(image_path, save_path=None):
    """
    Full Stage 1 pipeline for a single image.
    Returns (result_dict, enhanced_image_or_None).
    If ungradeable, enhanced_image_or_None is None and result_dict
    tells you exactly why (which check failed).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    quality = assess_quality(image)

    if not quality["gradeable"]:
        return quality, None

    enhanced = enhance_image(image)

    if save_path:
        cv2.imwrite(save_path, enhanced)

    return quality, enhanced


if __name__ == "__main__":
    # Example usage — replace with a real image path to test
    test_path = "/content/drive/MyDrive/p1/data/sample_fundus.jpg"
    result, enhanced_img = process_image(test_path, save_path="/content/enhanced_sample.jpg")
    print(result)
    if enhanced_img is not None:
        print("Image passed quality check and was enhanced.")
    else:
        print("Image failed quality check — recapture needed.")
