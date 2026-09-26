"""
collect_data.py - Dataset collector for object apparent size classifier.

Supports:
1. Live camera interactive collection with NanoDet detector and keypress tagging.
2. Synthetic generator generating balanced, realistic multi-object datasets with
   diverse aspect ratios and coverage tiers for immediate training & testing.
3. Strict object-level metadata recording ('object_name', 'object_type') to enforce
   object-wise train/test splitting (Section 11).
"""

from typing import List, Optional
import argparse
import os
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from src.detect import NanoDetDetector
from src.features import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    classify_by_threshold,
    extract_features,
    get_class_name,
)

DATA_PATH = "data/features.csv"


def append_sample_to_csv(
    features: np.ndarray,
    label: int,
    object_name: str,
    object_type: str,
    image_name: str = "",
    csv_path: str = DATA_PATH,
) -> None:
    """Append a single record to the dataset CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = {
        "width_ratio": round(float(features[0]), 5),
        "height_ratio": round(float(features[1]), 5),
        "area_ratio": round(float(features[2]), 5),
        "aspect_ratio": round(float(features[3]), 5),
        "label": int(label),
        "class_name": get_class_name(label),
        "object_name": object_name,
        "object_type": object_type,
        "image_name": image_name,
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    df.to_csv(csv_path, mode="a", index=False, header=header)


def generate_synthetic_dataset(
    num_samples_per_class: int = 100,
    csv_path: str = DATA_PATH,
    seed: int = 42,
) -> None:
    """
    Generate a balanced multi-object dataset across the 6 apparent size classes.
    Objects have realistic aspect-ratio profiles (e.g. tall bottles vs wide keyboards).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Object archetypes with realistic aspect ratio (width / height) distributions
    archetypes = {
        # Training candidate objects
        "bottle_1": {"type": "bottle", "ar_mean": 0.35, "ar_std": 0.05, "split_role": "train"},
        "bottle_2": {"type": "bottle", "ar_mean": 0.40, "ar_std": 0.06, "split_role": "train"},
        "cup_blue": {"type": "cup", "ar_mean": 0.85, "ar_std": 0.10, "split_role": "train"},
        "cup_white": {"type": "cup", "ar_mean": 0.90, "ar_std": 0.10, "split_role": "train"},
        "book_A": {"type": "book", "ar_mean": 0.72, "ar_std": 0.08, "split_role": "train"},
        "phone_A": {"type": "cell phone", "ar_mean": 0.50, "ar_std": 0.05, "split_role": "train"},
        "keyboard": {"type": "keyboard", "ar_mean": 2.60, "ar_std": 0.30, "split_role": "train"},
        "box_cardboard": {"type": "box", "ar_mean": 1.10, "ar_std": 0.15, "split_role": "train"},
        "mouse_black": {"type": "mouse", "ar_mean": 0.65, "ar_std": 0.08, "split_role": "train"},
        # Unseen test objects (Section 11 requirement: split by object)
        "bottle_unseen": {"type": "bottle", "ar_mean": 0.32, "ar_std": 0.04, "split_role": "test"},
        "shoe_unseen": {"type": "shoe", "ar_mean": 1.70, "ar_std": 0.20, "split_role": "test"},
        "book_B_unseen": {"type": "book", "ar_mean": 0.68, "ar_std": 0.07, "split_role": "test"},
        "mug_unseen": {"type": "cup", "ar_mean": 0.82, "ar_std": 0.09, "split_role": "test"},
        "ball_unseen": {"type": "sports ball", "ar_mean": 1.00, "ar_std": 0.05, "split_role": "test"},
    }

    # Coverage tiers for the 6 classes
    tiers = [
        (0, 0.002, 0.020),  # VERY_SMALL (0% - 2%)
        (1, 0.020, 0.080),  # SMALL      (2% - 8%)
        (2, 0.080, 0.200),  # MEDIUM     (8% - 20%)
        (3, 0.200, 0.400),  # LARGE      (20% - 40%)
        (4, 0.400, 0.700),  # VERY_LARGE (40% - 70%)
        (5, 0.700, 0.950),  # HUGE       (70% - 95%)
    ]

    records = []
    image_w, image_h = 640.0, 480.0
    total_image_area = image_w * image_h

    for label_id, min_cov, max_cov in tiers:
        for _ in range(num_samples_per_class):
            obj_name = random.choice(list(archetypes.keys()))
            arch = archetypes[obj_name]

            # Sample area ratio within tier bounds with slight margin noise
            area_ratio = np.random.uniform(min_cov, max_cov)
            target_box_area = area_ratio * total_image_area

            # Sample aspect ratio (w / h) around archetype mean
            aspect_ratio = float(np.clip(np.random.normal(arch["ar_mean"], arch["ar_std"]), 0.2, 3.5))

            # Solve: w * h = area, w / h = ar  => h = sqrt(area / ar), w = ar * h
            box_h = np.sqrt(target_box_area / aspect_ratio)
            box_w = aspect_ratio * box_h

            # Clip to image bounds
            box_w = min(box_w, image_w * 0.98)
            box_h = min(box_h, image_h * 0.98)

            # Recompute accurate ratios
            actual_w_ratio = box_w / image_w
            actual_h_ratio = box_h / image_h
            actual_area_ratio = (box_w * box_h) / total_image_area
            actual_ar = box_w / max(box_h, 1e-6)

            records.append({
                "width_ratio": round(float(actual_w_ratio), 5),
                "height_ratio": round(float(actual_h_ratio), 5),
                "area_ratio": round(float(actual_area_ratio), 5),
                "aspect_ratio": round(float(actual_ar), 5),
                "label": int(label_id),
                "class_name": CLASS_NAMES[label_id],
                "object_name": obj_name,
                "object_type": arch["type"],
                "image_name": f"synth_{obj_name}_{label_id}_{random.randint(1000, 9999)}.jpg",
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} samples across {len(tiers)} classes in '{csv_path}'.")
    print(df["class_name"].value_counts())
    print("\nObjects distribution:")
    print(df["object_name"].value_counts())


def run_live_collection(
    camera_id: int = 0,
    model_path: str = "models/detector/nanodet.onnx",
    csv_path: str = DATA_PATH,
    current_object: str = "item_1",
) -> None:
    """Interactive camera collector with live NanoDet detection and keypress logging."""
    detector = NanoDetDetector(model_path=model_path)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print("=" * 60)
    print("LIVE DATASET COLLECTOR")
    print("Controls:")
    print("  [SPACE] / [ENTER]: Save current detection (auto-label)")
    print("  [0] - [5]        : Save with manual class override (0=V_SMALL...5=HUGE)")
    print("  [N]              : Enter new object name")
    print("  [Q] / [ESC]      : Quit collection")
    print("=" * 60)

    count_saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        det = detector.detect_single(frame)
        display = frame.copy()

        features = None
        suggested_class = None

        if det:
            features = extract_features(det["bbox"], (h, w))
            area_ratio = features[2]
            suggested_class = classify_by_threshold(area_ratio)
            s_name = get_class_name(suggested_class)

            extra = f"Cov: {area_ratio * 100:.1f}% | Size: {s_name}"
            display = detector.draw_detection(display, det, extra_text=extra)

        # Overlay UI instructions
        cv2.putText(
            display,
            f"Object: {current_object} | Saved: {count_saved}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        if det and suggested_class is not None:
            cv2.putText(
                display,
                f"Press [SPACE] to save as {get_class_name(suggested_class)}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display,
                "No confident object detected",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Object Size Dataset Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), 27]:
            break
        elif key == ord("n"):
            # Console prompt for object name
            new_name = input("Enter new object name (e.g., bottle_red, mug_ceramic): ").strip()
            if new_name:
                current_object = new_name
                print(f"Current object set to: {current_object}")
        elif key in [32, 13]:  # SPACE or ENTER
            if det is not None and features is not None:
                append_sample_to_csv(
                    features=features,
                    label=suggested_class,
                    object_name=current_object,
                    object_type=det["class_name"],
                    csv_path=csv_path,
                )
                count_saved += 1
                print(f"Saved sample #{count_saved}: {det['class_name']} ({get_class_name(suggested_class)})")
        elif ord("0") <= key <= ord("5"):
            override_class = key - ord("0")
            if det is not None and features is not None:
                append_sample_to_csv(
                    features=features,
                    label=override_class,
                    object_name=current_object,
                    object_type=det["class_name"],
                    csv_path=csv_path,
                )
                count_saved += 1
                print(f"Saved sample #{count_saved}: {det['class_name']} ({get_class_name(override_class)}) [MANUAL]")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collection complete. Total samples recorded: {count_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect dataset for apparent size classifier")
    parser.add_argument("--camera", type=int, default=-1, help="Camera index for live collection")
    parser.add_argument("--object", type=str, default="object_1", help="Current object name for live mode")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic multi-object dataset")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Samples per class for synthetic generation")
    parser.add_argument("--csv", type=str, default=DATA_PATH, help="Output CSV path")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset(
            num_samples_per_class=args.samples_per_class,
            csv_path=args.csv,
        )
    elif args.camera >= 0:
        run_live_collection(
            camera_id=args.camera,
            csv_path=args.csv,
            current_object=args.object,
        )
    else:
        # Default: generate balanced 600-sample dataset across classes and objects
        print("No mode specified. Generating balanced starter dataset...")
        generate_synthetic_dataset(
            num_samples_per_class=100,
            csv_path=args.csv,
        )
