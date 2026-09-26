"""
inference.py - Unified inference pipeline with strict person suppression,
figure-ground differentiation, temporal smoothing, and TinyML size classification.
"""

from typing import Deque, Dict, List, Optional, Tuple, Union
from collections import deque
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import joblib
import numpy as np

from src.detect import NanoDetDetector
from src.features import CLASS_NAMES, extract_features, get_class_name
from src.models import TinyNeuralNetwork


class SizeClassifierPipeline:
    def __init__(
        self,
        detector_model_path: str = "models/detector/nanodet.onnx",
        size_model_path: str = "models/size_classifier/tiny_nn.joblib",
        use_int8: bool = True,
        filter_person: bool = True,
        class_agnostic: bool = True,
        detector_conf_thresh: float = 0.20,
        size_conf_thresh: float = 0.60,
        smoothing_window: int = 7,
    ):
        self.detector = NanoDetDetector(
            model_path=detector_model_path,
            prob_threshold=detector_conf_thresh,
            filter_person=filter_person,
        )
        if not os.path.exists(size_model_path):
            print(f"Size model not found at '{size_model_path}'. Running train.py to initialize models...")
            from src.train import train_and_evaluate
            train_and_evaluate()

        self.size_model: TinyNeuralNetwork = joblib.load(size_model_path)
        self.use_int8 = use_int8
        self.filter_person = filter_person
        self.class_agnostic = class_agnostic
        self.detector_conf_thresh = detector_conf_thresh
        self.size_conf_thresh = size_conf_thresh

        # Calibrated background frame
        self.bg_frame: Optional[np.ndarray] = None

        # Target Inspection Zone
        self.target_roi: Optional[Tuple[int, int, int, int]] = None

        # Temporal smoothing deque buffer
        self.smoothing_window = smoothing_window
        self.history: Deque[int] = deque(maxlen=smoothing_window)
        self.prob_history: Deque[np.ndarray] = deque(maxlen=smoothing_window)

    def calibrate_background(self, frame: np.ndarray) -> None:
        """Capture the current scene as the reference background."""
        self.bg_frame = frame.copy()
        self.reset_history()

    def reset_history(self) -> None:
        """Reset temporal smoothing buffer."""
        self.history.clear()
        self.prob_history.clear()

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Union[str, float, int, list, bool, None]]:
        """
        Process a single image frame through the pipeline.
        Strictly suppresses human body detections from ever becoming target objects.
        """
        t0 = time.perf_counter()
        img_h, img_w = frame.shape[:2]

        t_det_start = time.perf_counter()

        # Step 1: Detect person in frame for strict spatial suppression
        all_raw_dets = self.detector.detect_all(frame)
        person_boxes = []
        if self.filter_person:
            for d in all_raw_dets:
                if d["class_id"] == 0 or d["class_name"] == "person":
                    if d["confidence"] > 0.35:
                        person_boxes.append(d["bbox"])

        det = None

        # Step 2: Target Inspection Zone (GrabCut) or Calibrated Background Subtraction
        if self.target_roi is not None or self.bg_frame is not None:
            det = self.detector.segment_figure_ground(
                frame,
                target_roi=self.target_roi,
                bg_frame=self.bg_frame,
                person_boxes=person_boxes,
                filter_person=self.filter_person,
                max_coverage=0.30,  # Any object > 30% coverage is rejected as human body
            )

        # Step 3: Fallback to NanoDet detector (with strict person filtering)
        if det is None:
            det = self.detector.detect_single(
                frame,
                strategy="highest_confidence",
                filter_person=self.filter_person,
                person_boxes=person_boxes,
            )

        t_det_ms = (time.perf_counter() - t_det_start) * 1000.0

        # Step 4: Strict Sanity Guard - Ensure det is NOT the user
        if det is not None and self.filter_person:
            bx, by, bw, bh = det["bbox"]
            cov = (bw * bh) / (img_w * img_h)
            # Rejection check 1: > 30% of screen is human body
            if cov > 0.30:
                det = None
            # Rejection check 2: Overlap with person head/torso
            elif person_boxes:
                for (px, py, pw, ph) in person_boxes:
                    ix1 = max(bx, px)
                    iy1 = max(by, py)
                    ix2 = min(bx + bw, px + pw)
                    iy2 = min(by + bh, py + ph)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if (inter / (bw * bh)) > 0.40 and cov > 0.15:
                            det = None
                            break

        if det is None or det["confidence"] < self.detector_conf_thresh:
            self.history.append(-1)
            return {
                "status": "NO_OBJECT",
                "object_name": "None",
                "detector_conf": 0.0,
                "coverage_pct": 0.0,
                "raw_size_class": "None",
                "smoothed_size_class": "No target object detected",
                "size_conf": 0.0,
                "bbox": None,
                "features": None,
                "det_time_ms": t_det_ms,
                "cls_time_ms": 0.0,
                "total_time_ms": (time.perf_counter() - t0) * 1000.0,
                "is_confident": False,
            }

        # 5. Feature Extraction (4 numerical features)
        bbox = det["bbox"]
        features = extract_features(bbox, (img_h, img_w))
        coverage_pct = float(features[2] * 100.0)

        # 6. Size Classification via Tiny NN
        t_cls_start = time.perf_counter()
        feats_2d = features.reshape(1, -1)
        if self.use_int8:
            raw_class_idx = int(self.size_model.predict_int8(feats_2d)[0])
            probs = self.size_model.predict_proba_int8(feats_2d)[0]
        else:
            raw_class_idx = int(self.size_model.predict_fp32(feats_2d)[0])
            probs = self.size_model.predict_proba_fp32(feats_2d)[0]
        t_cls_ms = (time.perf_counter() - t_cls_start) * 1000.0

        # 7. Temporal Smoothing
        self.history.append(raw_class_idx)
        self.prob_history.append(probs)

        avg_probs = np.mean(self.prob_history, axis=0)
        smoothed_class_idx = int(np.argmax(avg_probs))
        smoothed_conf = float(avg_probs[smoothed_class_idx])

        # 8. Confidence Gating
        if smoothed_conf < self.size_conf_thresh:
            status = "UNCERTAIN"
            display_size = f"UNCERTAIN ({get_class_name(smoothed_class_idx)}?)"
            is_confident = False
        else:
            status = "CONFIDENT"
            display_size = get_class_name(smoothed_class_idx)
            is_confident = True

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": status,
            "object_name": "OBJECT",
            "detector_conf": det["confidence"],
            "coverage_pct": coverage_pct,
            "raw_size_class": get_class_name(raw_class_idx),
            "smoothed_size_class": display_size,
            "size_conf": smoothed_conf,
            "bbox": bbox,
            "features": features.tolist(),
            "det_time_ms": t_det_ms,
            "cls_time_ms": t_cls_ms,
            "total_time_ms": total_ms,
            "is_confident": is_confident,
        }

    def render_hud(
        self,
        frame: np.ndarray,
        result: Dict[str, Union[str, float, int, list, bool, None]],
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw Target Inspection Zone box if enabled
        if self.target_roi is not None:
            rx, ry, rw, rh = self.target_roi
            guide_color = (0, 200, 255)
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), guide_color, 1)
            c_len = 15
            cv2.line(out, (rx, ry), (rx + c_len, ry), guide_color, 3)
            cv2.line(out, (rx, ry), (rx, ry + c_len), guide_color, 3)
            cv2.line(out, (rx + rw, ry), (rx + rw - c_len, ry), guide_color, 3)
            cv2.line(out, (rx + rw, ry), (rx + rw, ry + c_len), guide_color, 3)
            cv2.line(out, (rx, ry + rh), (rx + c_len, ry + rh), guide_color, 3)
            cv2.line(out, (rx, ry + rh), (rx, ry + rh - c_len), guide_color, 3)
            cv2.line(out, (rx + rw, ry + rh), (rx + rw - c_len, ry + rh), guide_color, 3)
            cv2.line(out, (rx + rw, ry + rh), (rx + rw, ry + rh - c_len), guide_color, 3)
            cv2.putText(out, "TARGET ZONE (Hold item here)", (rx + 5, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, guide_color, 1)

        # Draw bounding box if object is present
        if result["bbox"] is not None:
            x, y, bw, bh = [int(v) for v in result["bbox"]]
            color = (0, 255, 0) if result["is_confident"] else (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 3)
            cv2.putText(
                out,
                f"OBJECT ({result['coverage_pct']:.1f}% cov)",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        # Bottom Telemetry Overlay
        card_h = 135
        card_y = h - card_h
        overlay = out.copy()
        cv2.rectangle(overlay, (0, card_y), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)
        cv2.line(out, (0, card_y), (w, card_y), (60, 60, 60), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (240, 240, 240)
        accent_color = (0, 230, 255)
        green = (50, 230, 50)
        orange = (0, 165, 255)
        red = (80, 80, 255)

        # Left Column: Object & Size Info
        if result["status"] == "NO_OBJECT":
            cv2.putText(out, "No target object detected", (20, card_y + 40), font, 0.75, red, 2)
            cv2.putText(out, "Hold item in Target Zone [T] or present object to camera", (20, card_y + 75), font, 0.50, (170, 170, 170), 1)
        else:
            cv2.putText(out, "Target:   OBJECT", (20, card_y + 30), font, 0.65, text_color, 2)
            cv2.putText(out, f"Coverage: {result['coverage_pct']:.1f}%", (20, card_y + 60), font, 0.65, accent_color, 2)

            size_color = green if result["is_confident"] else orange
            cv2.putText(out, f"Size:     {result['smoothed_size_class']}", (20, card_y + 90), font, 0.75, size_color, 2)
            cv2.putText(out, f"Confidence: {result['size_conf']:.1%}", (20, card_y + 118), font, 0.55, (200, 200, 200), 1)

        # Right Column: Architecture & Benchmark Telemetry
        model_mode = "INT8 TinyML" if self.use_int8 else "FP32 TinyML"
        bg_status = "Calibrated" if self.bg_frame is not None else "Auto"
        p_filter_status = "ON (Strict)" if self.filter_person else "OFF"
        r_x = max(w - 320, w // 2)
        cv2.putText(out, f"Person Filter: {p_filter_status} [F]", (r_x, card_y + 28), font, 0.48, (180, 180, 180), 1)
        cv2.putText(out, f"Classifier:    {model_mode}", (r_x, card_y + 52), font, 0.48, (180, 180, 180), 1)
        cv2.putText(out, f"Det Latency:   {result['det_time_ms']:.1f} ms", (r_x, card_y + 76), font, 0.48, (180, 180, 180), 1)
        cv2.putText(out, f"Cls Latency:   {result['cls_time_ms']:.3f} ms", (r_x, card_y + 100), font, 0.48, green, 1)

        return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Object Size Classifier Pipeline on an image")
    parser.add_argument("--image", type=str, default="data/raw/sample.png", help="Path to input image")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 Tiny NN instead of INT8")
    parser.add_argument("--output", type=str, default="results/demo_image_result.jpg", help="Path to save HUD output")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found at '{args.image}'")
        sys.exit(1)

    frame = cv2.imread(args.image)
    pipeline = SizeClassifierPipeline(use_int8=not args.fp32)
    result = pipeline.process_frame(frame)

    print("=" * 60)
    print("OBJECT SIZE CLASSIFIER INFERENCE RESULT")
    print("=" * 60)
    print(f"Status           : {result['status']}")
    print(f"Target           : {result['object_name']}")
    print(f"Coverage         : {result['coverage_pct']:.2f}%")
    print(f"Raw Size Class   : {result['raw_size_class']}")
    print(f"Smoothed Class   : {result['smoothed_size_class']}")
    print(f"Confidence       : {result['size_conf']:.1%}")
    print(f"Detector Latency : {result['det_time_ms']:.2f} ms")
    print(f"Classifier Latency: {result['cls_time_ms']:.4f} ms")
    print("=" * 60)

    hud = pipeline.render_hud(frame, result)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, hud)
    print(f"Visualization saved to: {args.output}")
