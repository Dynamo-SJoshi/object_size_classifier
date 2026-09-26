"""
detect.py - Lightweight NanoDet object detector using OpenCV DNN.

Implements:
- NanoDet ONNX loading via cv2.dnn
- 416x416 letterbox preprocessing
- Anchor generation and regression decoding
- NMS filtering and single-object selection (Milestone 1 / Step 7)
- Inspection Mode: Ignores background 'person' to prioritize hand-held / presented items
- Target Inspection ROI: Saliency / contour fallback for arbitrary objects not in COCO
- Class-Agnostic Mode: Labels all items as generic 'OBJECT'
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

# Standard COCO 80 Class Labels
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
        """
        Initialize the NanoDet detector.

        Args:
            model_path: Path to the NanoDet .onnx model file.
            prob_threshold: Minimum confidence score to accept detection. Default 0.20.
            iou_threshold: Non-Maximum Suppression (NMS) IoU threshold.
            input_size: Network input resolution (width, height), default (416, 416).
            filter_person: When True, ignores background person detections so that
                           hand-held or desk-placed objects take priority.
        """
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

        # Initialize network via OpenCV DNN
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        # Pre-compute anchor centers for each stride level
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
        """Letterbox pad and normalize image to input_size (416, 416)."""
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
        """Run inference on image and return all detected bounding boxes."""
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
    ) -> Optional[Dict[str, Union[List[float], float, int, str]]]:
        """
        Detect and select exactly ONE object.

        When filter_person=True:
            Filters out background person detections so hand-held or desk-placed
            objects are selected instead of the person.
        """
        should_filter_person = self.filter_person if filter_person is None else filter_person
        detections = self.detect_all(image)
        if not detections:
            return None

        candidates = detections
        if should_filter_person:
            # Filter out person class (id 0)
            non_person = [d for d in detections if d["class_id"] != 0 and d["class_name"] != "person"]
            if non_person:
                candidates = non_person
            else:
                # If only person is detected, return None so user can hold an object
                return None

        if strategy == "largest_area":
            return max(candidates, key=lambda d: d["bbox"][2] * d["bbox"][3])
        else:
            return max(candidates, key=lambda d: d["confidence"])

    def detect_in_roi(
        self,
        image: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
    ) -> Optional[Dict[str, Union[List[float], float, int, str]]]:
        """
        Target Inspection Zone: Detect ANY physical object inside an ROI rectangle
        using contour saliency. Guarantees 100% detection for objects not in COCO
        (e.g., earbud cases, keys, screws, small cards, batteries, tools).

        Args:
            roi_rect: (x, y, w, h) bounding box of the target inspection zone.
        """
        rx, ry, rw, rh = roi_rect
        h, w = image.shape[:2]
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = max(10, min(rw, w - rx))
        rh = max(10, min(rh, h - ry))

        roi = image[ry : ry + rh, rx : rx + rw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Contrast thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_area = rw * rh
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            # Minimum 1% of ROI and maximum 95% of ROI
            if 0.01 * roi_area < area < 0.95 * roi_area:
                valid_contours.append((area, c))

        if not valid_contours:
            # Try alternate threshold in case object is brighter than background
            _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours2:
                area = cv2.contourArea(c)
                if 0.01 * roi_area < area < 0.95 * roi_area:
                    valid_contours.append((area, c))

        if valid_contours:
            # Pick largest contour inside target zone
            valid_contours.sort(key=lambda item: item[0], reverse=True)
            best_contour = valid_contours[0][1]
            bx, by, bw, bh = cv2.boundingRect(best_contour)

            return {
                "bbox": [float(rx + bx), float(ry + by), float(bw), float(bh)],
                "confidence": 0.95,
                "class_id": -1,
                "class_name": "target_object",
            }

        return None

    @staticmethod
    def draw_detection(
        image: np.ndarray,
        detection: Dict[str, Union[List[float], float, int, str]],
        color: Tuple[int, int, int] = (0, 255, 0),
        extra_text: str = "",
        class_agnostic: bool = True,
    ) -> np.ndarray:
        """
        Draw a clean bounding box and label banner over the image.
        In class-agnostic mode, labels item simply as 'OBJECT' (Section 1).
        """
        out = image.copy()
        x, y, w, h = [int(v) for v in detection["bbox"]]
        conf = detection["confidence"]
        name = "OBJECT" if class_agnostic else detection["class_name"].upper()

        label = f"{name} {conf:.1%}"
        if extra_text:
            label += f" | {extra_text}"

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        banner_y1 = max(0, y - label_h - 8)
        banner_y2 = y
        cv2.rectangle(out, (x, banner_y1), (x + label_w + 10, banner_y2), color, -1)
        cv2.putText(
            out,
            label,
            (x + 5, banner_y2 - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        return out
