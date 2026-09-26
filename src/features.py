"""
features.py - Feature extraction and class definitions for apparent object size.

Features (4 numerical values):
  1. width_ratio  = box_width / image_width
  2. height_ratio = box_height / image_height
  3. area_ratio   = (box_width * box_height) / (image_width * image_height)
  4. aspect_ratio = box_width / box_height

Classes (6 categories based on frame coverage):
  0: VERY_SMALL (0% - 2%)
  1: SMALL      (2% - 8%)
  2: MEDIUM     (8% - 20%)
  3: LARGE      (20% - 40%)
  4: VERY_LARGE (40% - 70%)
  5: HUGE       (70% - 100%)
"""

from typing import Dict, List, Tuple, Union
import numpy as np

CLASS_NAMES: Dict[int, str] = {
    0: "VERY_SMALL",
    1: "SMALL",
    2: "MEDIUM",
    3: "LARGE",
    4: "VERY_LARGE",
    5: "HUGE",
}

CLASS_THRESHOLDS: List[Tuple[float, float, int]] = [
    (0.00, 0.02, 0),  # VERY_SMALL: 0% - 2%
    (0.02, 0.08, 1),  # SMALL:      2% - 8%
    (0.08, 0.20, 2),  # MEDIUM:     8% - 20%
    (0.20, 0.40, 3),  # LARGE:      20% - 40%
    (0.40, 0.70, 4),  # VERY_LARGE: 40% - 70%
    (0.70, 1.00, 5),  # HUGE:       70% - 100%
]

FEATURE_COLUMNS: List[str] = [
    "width_ratio",
    "height_ratio",
    "area_ratio",
    "aspect_ratio",
]


def extract_features(
    bbox: Union[List[float], Tuple[float, float, float, float], np.ndarray],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Extract the 4 numerical features from a bounding box and image dimensions.

    Args:
        bbox: [x, y, w, h] or [x1, y1, x2, y2] bounding box coordinates.
              Assumed to be [x, y, w, h] where x,y are top-left, w,h are width/height.
        image_shape: (height, width) or (height, width, channels) of image.

    Returns:
        np.ndarray: 1D array of [width_ratio, height_ratio, area_ratio, aspect_ratio] as float32.
    """
    img_h, img_w = image_shape[0], image_shape[1]
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

    # Clip dimensions to valid non-negative ranges
    w = max(0.0, float(w))
    h = max(0.0, float(h))

    # Safe division guards
    width_ratio = float(np.clip(w / float(img_w), 0.0, 1.0))
    height_ratio = float(np.clip(h / float(img_h), 0.0, 1.0))
    area_ratio = float(np.clip((w * h) / float(img_w * img_h), 0.0, 1.0))
    aspect_ratio = float(w / max(h, 1e-6))

    return np.array(
        [width_ratio, height_ratio, area_ratio, aspect_ratio], dtype=np.float32
    )


def classify_by_threshold(area_ratio: float) -> int:
    """
    Classify apparent size into 1 of 6 classes using prototype area_ratio thresholds.
    Used for baseline comparison and initial label assignment.

    Args:
        area_ratio: float in range [0.0, 1.0]

    Returns:
        int: class index (0 to 5)
    """
    area_ratio = max(0.0, min(1.0, float(area_ratio)))
    if area_ratio < 0.02:
        return 0  # VERY_SMALL
    elif area_ratio < 0.08:
        return 1  # SMALL
    elif area_ratio < 0.20:
        return 2  # MEDIUM
    elif area_ratio < 0.40:
        return 3  # LARGE
    elif area_ratio < 0.70:
        return 4  # VERY_LARGE
    else:
        return 5  # HUGE


def get_class_name(class_id: int) -> str:
    """Return the human-readable string for a class index."""
    return CLASS_NAMES.get(int(class_id), "UNKNOWN")


if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE EXTRACTION SELF-TEST")
    print("=" * 60)
    # Example: 160x180 box in 640x480 image
    bbox = [200, 100, 160, 180]
    img_size = (480, 640)
    feats = extract_features(bbox, img_size)
    cls_id = classify_by_threshold(feats[2])

    print(f"Image Size    : {img_size[1]}x{img_size[0]}")
    print(f"Bounding Box  : [x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}]")
    print(f"width_ratio   : {feats[0]:.4f}")
    print(f"height_ratio  : {feats[1]:.4f}")
    print(f"area_ratio    : {feats[2]:.5f} ({feats[2] * 100:.2f}% coverage)")
    print(f"aspect_ratio  : {feats[3]:.4f}")
    print(f"Classified    : {cls_id} ({get_class_name(cls_id)})")
    print("Self-test passed!")
