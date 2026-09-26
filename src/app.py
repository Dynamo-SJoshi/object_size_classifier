"""
app.py - Live interactive laptop demo application for the Object Size Classifier.

Features:
  - Real-time webcam feed with NanoDet detection and 4-feature TinyML size classification
  - Class-Agnostic Mode: Focuses purely on apparent size without distracting object categories
  - Inspection Mode (Filter Person): Ignores background person to prioritize hand-held/presented items
  - Target Inspection Zone [T]: On-screen guide ensuring 100% reliable detection for ANY physical item
  - Visual HUD displaying:
      * Bounding box and target banner
      * Frame Coverage %
      * Classified Size (VERY_SMALL, SMALL, MEDIUM, LARGE, VERY_LARGE, HUGE)
      * Confidence score & status (CONFIDENT / UNCERTAIN / NO OBJECT)
      * Live detector & classifier latencies
  - Interactive key bindings:
      * 'f': Toggle Person Filtering (Inspection Mode ON/OFF)
      * 't': Toggle Target Inspection Zone guide
      * 'm': Toggle between INT8 and FP32 classifier
      * 's': Reset temporal smoothing
      * '+': Increase detector sensitivity (lower threshold)
      * '-': Decrease detector sensitivity (raise threshold)
      * 'p': Save snapshot screenshot to results/
      * 'q' / ESC: Quit
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.inference import SizeClassifierPipeline


def run_app(
    camera_id: int = 0,
    image_path: str = "",
    use_int8: bool = True,
    filter_person: bool = True,
    detector_conf: float = 0.20,
    size_conf: float = 0.60,
    save_output: str = "",
) -> None:
    print("=" * 70)
    print("OBJECT SIZE CLASSIFIER - LIVE DEMO APPLICATION (INSPECTION MODE)")
    print("=" * 70)
    print(f"Loading pipeline (INT8={use_int8}, Filter Person={filter_person}, Det Conf={detector_conf:.0%})...")

    pipeline = SizeClassifierPipeline(
        use_int8=use_int8,
        filter_person=filter_person,
        class_agnostic=True,
        detector_conf_thresh=detector_conf,
        size_conf_thresh=size_conf,
        smoothing_window=7,
    )

    # Image mode (single frame test)
    if image_path:
        if not os.path.exists(image_path):
            print(f"Error: Image '{image_path}' not found.")
            return
        frame = cv2.imread(image_path)
        result = pipeline.process_frame(frame)
        hud = pipeline.render_hud(frame, result)

        out_path = save_output or "results/demo_image_result.jpg"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, hud)
        print(f"Processed image saved to: {out_path}")
        print(f"Result: Target={result['object_name']}, Size={result['smoothed_size_class']}, Cov={result['coverage_pct']:.1f}%")
        return

    # Webcam live mode
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_id}.")
        print("Tip: Use --image path/to/image.jpg to test on a static image instead.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nLive camera demo running.")
    print("Controls:")
    print("  [F]     : Toggle Person Filtering (ON = focus on held objects)")
    print("  [T]     : Toggle Target Inspection Zone (center guide box)")
    print("  [M]     : Toggle Model Mode (INT8 <-> FP32)")
    print("  [S]     : Reset Temporal Smoothing")
    print("  [+]     : Increase sensitivity (lower detector threshold)")
    print("  [-]     : Decrease sensitivity (raise detector threshold)")
    print("  [P]     : Save Screenshot to results/")
    print("  [Q]/ESC : Exit")

    frame_count = 0
    t_start = time.time()
    fps_display = 0.0
    target_zone_enabled = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        h, w = frame.shape[:2]

        # Update Target Zone ROI if enabled
        if target_zone_enabled:
            rw, rh = int(w * 0.45), int(h * 0.50)
            rx, ry = (w - rw) // 2, (h - rh) // 2 - 20
            pipeline.target_roi = (rx, ry, rw, rh)
        else:
            pipeline.target_roi = None

        frame_count += 1
        if frame_count % 15 == 0:
            elapsed = time.time() - t_start
            fps_display = 15.0 / elapsed if elapsed > 0 else 0.0
            t_start = time.time()

        # Run inference pipeline
        result = pipeline.process_frame(frame)

        # Render HUD
        display = pipeline.render_hud(frame, result)

        # Status pills at the top
        f_status = "Filter Person: ON" if pipeline.filter_person else "Filter Person: OFF"
        f_color = (0, 255, 128) if pipeline.filter_person else (180, 180, 180)
        cv2.putText(display, f_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, f_color, 2, cv2.LINE_AA)

        if target_zone_enabled:
            cv2.putText(display, "Target Zone: ACTIVE", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2, cv2.LINE_AA)

        # FPS indicator in top-right corner
        cv2.putText(
            display,
            f"FPS: {fps_display:.1f}",
            (display.shape[1] - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Object Size Classifier - TinyML Demo", display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), 27]:
            break
        elif key in [ord("f"), ord("F")]:
            pipeline.filter_person = not pipeline.filter_person
            pipeline.detector.filter_person = pipeline.filter_person
            print(f"Person filtering toggled: {pipeline.filter_person}")
        elif key in [ord("t"), ord("T")]:
            target_zone_enabled = not target_zone_enabled
            if not target_zone_enabled:
                pipeline.target_roi = None
            print(f"Target Inspection Zone toggled: {target_zone_enabled}")
        elif key in [ord("m"), ord("M")]:
            pipeline.use_int8 = not pipeline.use_int8
            mode_str = "INT8" if pipeline.use_int8 else "FP32"
            print(f"Switched classifier model to: {mode_str}")
        elif key in [ord("s"), ord("S")]:
            pipeline.reset_history()
            print("Temporal smoothing history reset.")
        elif key in [ord("+"), ord("=")]:
            pipeline.detector_conf_thresh = max(0.05, pipeline.detector_conf_thresh - 0.05)
            pipeline.detector.prob_threshold = pipeline.detector_conf_thresh
            print(f"Detector threshold decreased (more sensitive): {pipeline.detector_conf_thresh:.2f}")
        elif key in [ord("-"), ord("_")]:
            pipeline.detector_conf_thresh = min(0.90, pipeline.detector_conf_thresh + 0.05)
            pipeline.detector.prob_threshold = pipeline.detector_conf_thresh
            print(f"Detector threshold increased (less sensitive): {pipeline.detector_conf_thresh:.2f}")
        elif key in [ord("p"), ord("P")]:
            snap_path = f"results/snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_path, display)
            print(f"Saved snapshot to: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo session closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Object Size Classifier Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--image", type=str, default="", help="Path to input image for single-frame test")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of INT8")
    parser.add_argument("--no-filter-person", action="store_true", help="Disable person filtering")
    parser.add_argument("--det-conf", type=float, default=0.20, help="Detector confidence threshold")
    parser.add_argument("--size-conf", type=float, default=0.60, help="Size classifier confidence threshold")
    parser.add_argument("--output", type=str, default="", help="Save output image path")
    args = parser.parse_args()

    run_app(
        camera_id=args.camera,
        image_path=args.image,
        use_int8=not args.fp32,
        filter_person=not args.no_filter_person,
        detector_conf=args.det_conf,
        size_conf=args.size_conf,
        save_output=args.output,
    )
