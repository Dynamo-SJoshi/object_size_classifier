"""
benchmark.py - Benchmark laptop runtime performance, memory footprint, and model sizes.

Measures (Section 18):
  - Image input resolution
  - Detector inference time (mean, std, min, max)
  - Size classifier inference time (FP32 vs INT8 vs Decision Tree)
  - Total latency & FPS
  - Process RAM usage (MB)
  - Model disk footprint (NanoDet vs Tiny NN)
  - Saves to results/benchmark.csv
"""

from typing import Dict, List
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import joblib
import numpy as np
import pandas as pd
import psutil

from src.detect import NanoDetDetector
from src.features import extract_features
from src.models import TinyNeuralNetwork


def benchmark_pipeline(
    image_path: str = "data/raw/test_cam.jpg",
    num_warmup: int = 10,
    num_runs: int = 100,
    save_csv: str = "results/benchmark.csv",
) -> pd.DataFrame:
    print("=" * 70)
    print("OBJECT SIZE CLASSIFIER - SYSTEM BENCHMARK SUITE")
    print("=" * 70)

    # 1. Load test image
    if os.path.exists(image_path):
        frame = cv2.imread(image_path)
    else:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_h, img_w = frame.shape[:2]
    print(f"Test image resolution: {img_w}x{img_h}")

    # 2. Measure Model File Sizes
    nanodet_path = "models/detector/nanodet.onnx"
    nanodet_int8_path = "models/detector/nanodet_int8.onnx"
    dt_path = "models/size_classifier/decision_tree.joblib"
    nn_path = "models/size_classifier/tiny_nn.joblib"
    c_header_path = "embedded/model_data.h"
    c_source_path = "embedded/model_data.cc"

    sizes_kb = {
        "NanoDet ONNX (FP32)": os.path.getsize(nanodet_path) / 1024.0 if os.path.exists(nanodet_path) else 0,
        "NanoDet ONNX (INT8)": os.path.getsize(nanodet_int8_path) / 1024.0 if os.path.exists(nanodet_int8_path) else 0,
        "Decision Tree (joblib)": os.path.getsize(dt_path) / 1024.0 if os.path.exists(dt_path) else 0,
        "Tiny NN (joblib)": os.path.getsize(nn_path) / 1024.0 if os.path.exists(nn_path) else 0,
        "Embedded C Header (h)": os.path.getsize(c_header_path) / 1024.0 if os.path.exists(c_header_path) else 0,
        "Embedded C Source (cc)": os.path.getsize(c_source_path) / 1024.0 if os.path.exists(c_source_path) else 0,
    }

    print("\n--- Model Footprints on Disk ---")
    for name, sz in sizes_kb.items():
        print(f"  {name:<28}: {sz:>8.2f} KB")

    # 3. Benchmark NanoDet Detector
    detector = NanoDetDetector(model_path=nanodet_path)
    print("\nWarming up detector...")
    for _ in range(num_warmup):
        detector.detect_single(frame)

    print(f"Benchmarking detector over {num_runs} runs...")
    det_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        det = detector.detect_single(frame)
        det_times.append((time.perf_counter() - t0) * 1000.0)

    det_mean = np.mean(det_times)
    det_std = np.std(det_times)

    # 4. Benchmark Feature Extraction
    sample_bbox = det["bbox"] if det else [100.0, 100.0, 200.0, 200.0]
    feat_times = []
    for _ in range(num_runs * 10):
        t0 = time.perf_counter()
        feats = extract_features(sample_bbox, (img_h, img_w))
        feat_times.append((time.perf_counter() - t0) * 1000.0)
    feat_mean = np.mean(feat_times)

    # 5. Benchmark Classifiers
    tiny_nn: TinyNeuralNetwork = joblib.load(nn_path)
    dt = joblib.load(dt_path)
    sample_input = feats.reshape(1, -1)

    # FP32 NN
    fp32_times = []
    for _ in range(num_runs * 10):
        t0 = time.perf_counter()
        _ = tiny_nn.predict_fp32(sample_input)
        fp32_times.append((time.perf_counter() - t0) * 1000.0)
    fp32_mean = np.mean(fp32_times)

    # INT8 NN
    int8_times = []
    for _ in range(num_runs * 10):
        t0 = time.perf_counter()
        _ = tiny_nn.predict_int8(sample_input)
        int8_times.append((time.perf_counter() - t0) * 1000.0)
    int8_mean = np.mean(int8_times)

    # Decision Tree
    dt_times = []
    for _ in range(num_runs * 10):
        t0 = time.perf_counter()
        _ = dt.predict(sample_input)
        dt_times.append((time.perf_counter() - t0) * 1000.0)
    dt_mean = np.mean(dt_times)

    # 6. End-to-End Latency & RAM
    total_latency_ms = det_mean + feat_mean + int8_mean
    fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # 7. Summary Table
    rows = [
        {"Component": "Image Input", "Resolution / Metric": f"{img_w}x{img_h}", "Mean Latency (ms)": 0.0, "FPS": "-", "Footprint (KB)": "-"},
        {"Component": "NanoDet (Detector)", "Resolution / Metric": "416x416 Input", "Mean Latency (ms)": round(det_mean, 3), "FPS": round(1000.0 / det_mean, 1), "Footprint (KB)": round(sizes_kb["NanoDet ONNX (FP32)"], 1)},
        {"Component": "Feature Extractor (4 feats)", "Resolution / Metric": "4 Ratios", "Mean Latency (ms)": round(feat_mean, 4), "FPS": "-", "Footprint (KB)": "-"},
        {"Component": "Tiny NN (FP32)", "Resolution / Metric": "4->8->8->6", "Mean Latency (ms)": round(fp32_mean, 4), "FPS": round(1000.0 / fp32_mean, 1), "Footprint (KB)": round(sizes_kb["Tiny NN (joblib)"], 2)},
        {"Component": "Tiny NN (INT8)", "Resolution / Metric": "4->8->8->6 INT8", "Mean Latency (ms)": round(int8_mean, 4), "FPS": round(1000.0 / int8_mean, 1), "Footprint (KB)": round(sizes_kb["Embedded C Source (cc)"], 2)},
        {"Component": "Decision Tree (Model 1)", "Resolution / Metric": "max_depth=5", "Mean Latency (ms)": round(dt_mean, 4), "FPS": round(1000.0 / dt_mean, 1), "Footprint (KB)": round(sizes_kb["Decision Tree (joblib)"], 2)},
        {"Component": "End-to-End Pipeline (INT8)", "Resolution / Metric": "Total Inference", "Mean Latency (ms)": round(total_latency_ms, 2), "FPS": round(fps, 1), "Footprint (KB)": f"RAM: {ram_mb:.1f} MB"},
    ]

    df_bench = pd.DataFrame(rows)
    print("\n--- Benchmark Results ---")
    print(df_bench.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"KEY ARCHITECTURAL INSIGHT (Section 18):")
    detector_pct = (det_mean / total_latency_ms) * 100.0
    classifier_pct = (int8_mean / total_latency_ms) * 100.0
    print(f"  Detector Latency   : {det_mean:.2f} ms ({detector_pct:.1f}% of compute)")
    print(f"  Classifier Latency : {int8_mean:.4f} ms ({classifier_pct:.3f}% of compute)")
    print(f"  => The size classifier is essentially free; object detection dominates computation!")
    print("=" * 70)

    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    df_bench.to_csv(save_csv, index=False)
    print(f"\nSaved benchmark table to: {save_csv}")
    return df_bench


if __name__ == "__main__":
    benchmark_pipeline()
