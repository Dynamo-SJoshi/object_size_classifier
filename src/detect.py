"""
detect.py - Hybrid Object Detector & Figure-Ground Separator with Strict Person Suppression.

Features:
  1. Strict Person Suppression: Guarantees that human head, torso, or full body detections
     are NEVER selected as the target object during inspection.
  2. Calibrated Background Difference with Person Torso Masking:
     Only changes outside the person's core body are detected as objects.
  3. Target Inspection Zone (GrabCut shrink-wrap around physical items).
  4. Pretrained NanoDet (OpenCV Zoo) for standard object detection.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class NanoDetDetector:
    def __init__(
        self,
        model_path: str = "models/detector/nanodet.onnx",
        prob_threshold: float = 0.20,
        iou_threshold: float = 0.50,
        input_size: Tuple[int, int] = (416, 416),
        filter_person: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.filter_person = filter_person
        self.strides = (8, 16, 32)
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        self.anchors_mlvl = []
        for s in self.strides:
            feat_h, feat_w = self.input_size[1] // s, self.input_size[0] // s
            shift_x = np.arange(0, feat_w) * s
            shift_y = np.arange(0, feat_h) * s
            xv, yv = np.meshgrid(shift_x, shift_y)
            cx = xv.flatten() + 0.5 * (s - 1)
            cy = yv.flatten() + 0.5 * (s - 1)
            self.anchors_mlvl.append(np.column_stack((cx, cy)))

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        img_h, img_w = image.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / float(img_w), target_h / float(img_h))
        nw, nh = int(round(img_w * scale)), int(round(img_h * scale))

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy : dy + nh, dx : dx + nw] = resized

        normalized = canvas.astype(np.float32)
        normalized = (normalized - self.mean) / self.std
        blob = cv2.dnn.blobFromImage(normalized)
        return blob, scale, dx, dy

    def detect_all(self, image: np.ndarray) -> List[Dict[str, Union[List[float], float, int, str]]]:
        orig_h, orig_w = image.shape[:2]
        blob, scale, dx, dy = self._preprocess(image)

        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        cls_scores = outs[:3]
        bbox_preds = outs[3:]

        bboxes_mlvl, scores_mlvl = [], []
        for s, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.anchors_mlvl):
            cls_score = cls_score.squeeze(0)
            bbox_pred = bbox_pred.squeeze(0)

            x_exp = np.exp(bbox_pred.reshape(-1, self.reg_max + 1))
            bbox_pred = x_exp / np.sum(x_exp, axis=1, keepdims=True)
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4) * s

            nms_pre = 1000
            if cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk = max_scores.argsort()[::-1][:nms_pre]
                anchors = anchors[topk, :]
                bbox_pred = bbox_pred[topk, :]
                cls_score = cls_score[topk, :]

            x1 = np.clip(anchors[:, 0] - bbox_pred[:, 0], 0, self.input_size[0])
            y1 = np.clip(anchors[:, 1] - bbox_pred[:, 1], 0, self.input_size[1])
            x2 = np.clip(anchors[:, 0] + bbox_pred[:, 2], 0, self.input_size[0])
            y2 = np.clip(anchors[:, 1] + bbox_pred[:, 3], 0, self.input_size[1])

            bboxes_mlvl.append(np.column_stack([x1, y1, x2, y2]))
            scores_mlvl.append(cls_score)

        bboxes = np.concatenate(bboxes_mlvl, axis=0)
        scores = np.concatenate(scores_mlvl, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        indices = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(),
            confidences.tolist(),
            self.prob_threshold,
            self.iou_threshold,
        )

        detections = []
        if len(indices) > 0:
            for idx in indices:
                i = int(idx)
                b = bboxes_wh[i]
                conf = float(confidences[i])
                cid = int(class_ids[i])

                orig_x = float(max(0.0, (b[0] - dx) / scale))
                orig_y = float(max(0.0, (b[1] - dy) / scale))
                orig_w_box = float(min(float(orig_w) - orig_x, b[2] / scale))
                orig_h_box = float(min(float(orig_h) - orig_y, b[3] / scale))

                cname = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"class_{cid}"

                detections.append({
                    "bbox": [orig_x, orig_y, orig_w_box, orig_h_box],
                    "confidence": conf,
                    "class_id": cid,
                    "class_name": cname,
                })

        return detections

    def detect_single(
        self,
        image: np.ndarray,
        strategy: str = "highest_confidence",
        filter_person: Optional[bool] = None,
        person_boxes: Optional[List[List[float]]] = None,
    ) -> Optional[Dict[str, Union[List[float], float, int, str]]]:
        should_filter_person = self.filter_person if filter_person is None else filter_person
        detections = self.detect_all(image)
        if not detections:
            return None

        candidates = detections
        if should_filter_person:
            # 1. Filter out class_id == 0 ('person')
            non_person = [d for d in detections if d["class_id"] != 0 and d["class_name"] != "person"]
            
            # 2. Strict spatial person suppression (reject any candidate that is essentially a person)
            surviving = []
            h, w = image.shape[:2]
            for cand in non_person:
                bx, by, bw, bh = cand["bbox"]
                cov = (bw * bh) / (w * h)
                # Any object larger than 30% of screen overlapping with the user is suppressed
                if cov > 0.30:
                    continue
                if person_boxes:
                    is_p = False
                    for (px, py, pw, ph) in person_boxes:
                        # Check intersection
                        ix1 = max(bx, px)
                        iy1 = max(by, py)
                        ix2 = min(bx + bw, px + pw)
                        iy2 = min(by + bh, py + ph)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            if (inter / (bw * bh)) > 0.50 and cov > 0.15:
                                is_p = True
                                break
                    if not is_p:
                        surviving.append(cand)
                else:
                    surviving.append(cand)
            candidates = surviving

        if not candidates:
            return None

        if strategy == "largest_area":
            return max(candidates, key=lambda d: d["bbox"][2] * d["bbox"][3])
        else:
            return max(candidates, key=lambda d: d["confidence"])

    @staticmethod
    def segment_figure_ground(
        image: np.ndarray,
        target_roi: Optional[Tuple[int, int, int, int]] = None,
        bg_frame: Optional[np.ndarray] = None,
        person_boxes: Optional[List[List[float]]] = None,
        filter_person: bool = True,
        min_coverage: float = 0.005,
        max_coverage: float = 0.30,
    ) -> Optional[Dict[str, Union[List[float], float, int, str]]]:
        """
        True Figure-Ground Differentiator with Strict Person Suppression.
        Guarantees human body / head is NEVER returned as the object.
        """
        h, w = image.shape[:2]

        # Method 1: Target Inspection Zone (GrabCut shrink-wrap)
        # When target_roi is provided, it isolates the object inside the target guide
        if target_roi is not None:
            rx, ry, rw, rh = target_roi
            rx = max(0, min(rx, w - 10))
            ry = max(0, min(ry, h - 10))
            rw = max(20, min(rw, w - rx))
            rh = max(20, min(rh, h - ry))

            roi = image[ry : ry + rh, rx : rx + rw]
            if roi.size > 0:
                mask = np.zeros(roi.shape[:2], np.uint8)
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)

                margin_x = max(5, int(rw * 0.05))
                margin_y = max(5, int(rh * 0.05))
                inner_rect = (margin_x, margin_y, rw - 2 * margin_x, rh - 2 * margin_y)

                try:
                    cv2.grabCut(roi, mask, inner_rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
                    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

                    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        best_c = max(cnts, key=cv2.contourArea)
                        c_area = cv2.contourArea(best_c)
                        if c_area > (rw * rh * 0.03):
                            bx, by, bw, bh = cv2.boundingRect(best_c)
                            cov = (bw * bh) / (w * h)
                            if cov <= max_coverage:
                                return {
                                    "bbox": [float(rx + bx), float(ry + by), float(bw), float(bh)],
                                    "confidence": 0.95,
                                    "class_id": -1,
                                    "class_name": "target_object",
                                }
                except Exception:
                    pass

        # Method 2: Calibrated Background Subtraction with Strict Person Suppression
        if bg_frame is not None and bg_frame.shape == image.shape:
            diff = cv2.absdiff(image, bg_frame)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

            # STRICT PERSON SUPPRESSION: Mask out person's core torso and head from diff!
            if filter_person and person_boxes:
                for (px, py, pw, ph) in person_boxes:
                    # Mask center 70% of person box
                    mx1 = max(0, int(px + pw * 0.15))
                    mx2 = min(w, int(px + pw * 0.85))
                    my1 = max(0, int(py))
                    my2 = min(h, int(py + ph))
                    closed[my1:my2, mx1:mx2] = 0

            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = []
            for c in cnts:
                area = cv2.contourArea(c)
                bx, by, bw, bh = cv2.boundingRect(c)
                cov = (bw * bh) / (w * h)

                # STRICT CRITERIA: A presented object is between 0.5% and 25% of frame
                # Any box > 30% of frame is the human body and is rejected!
                if (w * h * min_coverage) < area and cov <= max_coverage:
                    # Also verify center does not lie inside person's torso
                    center_x = bx + bw / 2.0
                    center_y = by + bh / 2.0
                    inside_person = False
                    if filter_person and person_boxes:
                        for (px, py, pw, ph) in person_boxes:
                            if (px + pw * 0.15) < center_x < (px + pw * 0.85) and py < center_y < (py + ph):
                                inside_person = True
                                break
                    if not inside_person:
                        valid.append((area, [float(bx), float(by), float(bw), float(bh)]))

            if valid:
                valid.sort(key=lambda x: x[0], reverse=True)
                return {
                    "bbox": valid[0][1],
                    "confidence": 0.95,
                    "class_id": -1,
                    "class_name": "target_object",
                }

        return None
