"""
app.py - Live interactive laptop demo with Figure-Ground Object Differentiator
and Strict Person Suppression.

Features:
  - Universal Object Size Inspection: Differentiates between object and background.
    Works for sleeping masks, earbud cases, tools, cards, fruits, fabrics, and ANY physical object.
  - Strict Person Suppression: Ensures human body/head is NEVER selected as an object.
  - Target Inspection Zone [T]: GrabCut shrink-wrap around whatever item is presented.
  - Background Calibration [B]: Press [B] to calibrate the empty background.
    Only new objects held in front of the camera are isolated.
  - Click-to-Target: Click on any object with the mouse to target and measure it!
  - Controls:
      * [B]     : Calibrate / Reset Background reference
      * [T]     : Toggle Target Inspection Zone
      * [F]     : Toggle Person Filtering (Strict Suppression)
      * [M]     : Toggle between INT8 and FP32 model
      * [S]     : Reset temporal smoothing
      * [P]     : Save screenshot snapshot
      * [Q]/ESC : Exit
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

clicked_point = None

def on_mouse_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def run_app(
    camera_id: int = 0,
    image_path: str = "",
    use_int8: bool = True,
    filter_person: bool = True,
    detector_conf: float = 0.20,
    size_conf: float = 0.60,
    save_output: str = "",
) -> None:
    global clicked_point
    print("=" * 70)
    print("OBJECT SIZE CLASSIFIER - STRICT PERSON SUPPRESSION DEMO")
    print("=" * 70)

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
        h, w = frame.shape[:2]

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
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Object Size Classifier - Person Filtered"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    print("\nLive Camera Demo Running.")
    print("=" * 55)
    print("HOW TO MEASURE ANY ARBITRARY OBJECT:")
    print("  - Hold the object inside the Target Zone [T].")
    print("  - Or press [B] to calibrate the background.")
    print("  - Or CLICK on any object to target it immediately!")
    print("=" * 55)
    print("Keybindings:")
    print("  [B]     : Calibrate Background reference snapshot")
    print("  [T]     : Toggle Target Zone guide (Default: ON)")
    print("  [F]     : Toggle Person Filtering (Default: ON)")
    print("  [M]     : Toggle INT8 <-> FP32 model")
    print("  [S]     : Reset Temporal Smoothing & Target Lock")
    print("  [P]     : Save Screenshot")
    print("  [Q]/ESC : Exit")

    frame_count = 0
    t_start = time.time()
    fps_display = 0.0
    target_zone_enabled = True
    target_center = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Handle mouse click to set inspection target
        if clicked_point is not None:
            cx, cy = clicked_point
            target_center = (cx, cy)
            target_zone_enabled = True
            clicked_point = None

        # Update Target Zone ROI
        if target_zone_enabled:
            rw, rh = int(w * 0.35), int(h * 0.45)
            if target_center is not None:
                cx, cy = target_center
                rx = max(0, min(cx - rw // 2, w - rw))
                ry = max(0, min(cy - rh // 2, h - rh))
            else:
                # Default position: Left-center (where held items usually appear)
                rx = int(w * 0.12)
                ry = int(h * 0.15)
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

        # Top Status Indicators
        f_status = "Person Filter: ON (Strict)" if pipeline.filter_person else "Person Filter: OFF"
        f_color = (0, 255, 128) if pipeline.filter_person else (180, 180, 180)
        cv2.putText(display, f_status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, f_color, 2, cv2.LINE_AA)

        if pipeline.bg_frame is not None:
            cv2.putText(display, "BG: CALIBRATED", (280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

        # FPS indicator
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

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), 27]:
            break
        elif key in [ord("b"), ord("B")]:
            pipeline.calibrate_background(frame)
            print("Background calibrated successfully!")
        elif key in [ord("t"), ord("T")]:
            target_zone_enabled = not target_zone_enabled
            if not target_zone_enabled:
                pipeline.target_roi = None
                target_center = None
            print(f"Target Zone toggled: {target_zone_enabled}")
        elif key in [ord("f"), ord("F")]:
            pipeline.filter_person = not pipeline.filter_person
            pipeline.detector.filter_person = pipeline.filter_person
            print(f"Person filtering toggled: {pipeline.filter_person}")
        elif key in [ord("m"), ord("M")]:
            pipeline.use_int8 = not pipeline.use_int8
            print(f"Switched model: {'INT8' if pipeline.use_int8 else 'FP32'}")
        elif key in [ord("s"), ord("S")]:
            pipeline.reset_history()
            target_center = None
            print("Reset tracking & smoothing history.")
        elif key in [ord("p"), ord("P")]:
            snap_path = f"results/snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snap_path, display)
            print(f"Saved snapshot to: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo session closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Size Classifier - Strict Person Filtered")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--image", type=str, default="", help="Path to input image for single-frame test")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of INT8")
    parser.add_argument("--output", type=str, default="", help="Save output image path")
    args = parser.parse_args()

    run_app(
        camera_id=args.camera,
        image_path=args.image,
        use_int8=not args.fp32,
        save_output=args.output,
    )
