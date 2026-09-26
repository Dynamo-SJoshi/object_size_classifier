"""
train.py - Model training, INT8 quantization, and embedded C-array export.

Models trained and compared:
  1. Baseline: Area-ratio threshold rule
  2. Model 1: Decision Tree (max_depth=5)
  3. Model 2: Tiny Neural Network (4 -> Dense(8) -> Dense(8) -> Dense(6))
     - FP32 Model
     - INT8 Quantized Model

Key Methodology:
  - Strict Object-Wise Split: Trains on a subset of objects and evaluates on UNSEEN objects
    to ensure the network learns apparent size rather than memorizing object appearance.
  - INT8 Quantization: Quantizes FP32 weights into int8 with calibration scale factors.
  - Embedded Export: Generates standalone C headers (model_data.h / model_data.cc).
"""

from typing import Dict, List, Tuple
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.features import CLASS_NAMES, FEATURE_COLUMNS, classify_by_threshold
from src.models import BaselineAreaClassifier, TinyNeuralNetwork


def export_c_arrays(tiny_nn: TinyNeuralNetwork, output_dir: str = "embedded") -> Tuple[str, str]:
    """
    Export the INT8 quantized model weights and inference function to standalone C files.
    """
    os.makedirs(output_dir, exist_ok=True)
    w = tiny_nn.int8_weights
    W1_q = w["W1_q"]  # (4, 8)
    W2_q = w["W2_q"]  # (8, 8)
    W3_q = w["W3_q"]  # (8, 6)

    h_content = f"""// model_data.h - Auto-generated embedded size classifier header
#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define NUM_INPUT_FEATURES 4
#define LAYER1_UNITS 8
#define LAYER2_UNITS 8
#define NUM_CLASSES 6

// Layer 1 weights (4 -> 8) and scale
extern const int8_t W1_q[4][8];
extern const float s_W1;
extern const float b1[8];

// Layer 2 weights (8 -> 8) and scale
extern const int8_t W2_q[8][8];
extern const float s_W2;
extern const float b2[8];

// Layer 3 weights (8 -> 6) and scale
extern const int8_t W3_q[8][6];
extern const float s_W3;
extern const float b3[6];

// Standalone C inference function
int predict_size_class(const float features[NUM_INPUT_FEATURES]);
void predict_size_probabilities(const float features[NUM_INPUT_FEATURES], float probs[NUM_CLASSES]);

#ifdef __cplusplus
}}
#endif

#endif // MODEL_DATA_H_
"""

    def format_2d(arr: np.ndarray) -> str:
        rows = []
        for row in arr:
            row_str = ", ".join(f"{int(v)}" for v in row)
            rows.append(f"    {{{row_str}}}")
        return "{\n" + ",\n".join(rows) + "\n}"

    def format_1d(arr: np.ndarray) -> str:
        return "{" + ", ".join(f"{float(v):.6f}f" for v in arr) + "}"

    cc_content = f"""// model_data.cc - Auto-generated embedded size classifier implementation
#include "model_data.h"
#include <math.h>

const float s_W1 = {w['s_W1']:.8f}f;
const float s_W2 = {w['s_W2']:.8f}f;
const float s_W3 = {w['s_W3']:.8f}f;

const int8_t W1_q[4][8] = {format_2d(W1_q)};
const float b1[8] = {format_1d(w['b1'])};

const int8_t W2_q[8][8] = {format_2d(W2_q)};
const float b2[8] = {format_1d(w['b2'])};

const int8_t W3_q[8][6] = {format_2d(W3_q)};
const float b3[6] = {format_1d(w['b3'])};

int predict_size_class(const float features[NUM_INPUT_FEATURES]) {{
    float probs[NUM_CLASSES];
    predict_size_probabilities(features, probs);
    int best_cls = 0;
    float max_p = probs[0];
    for (int i = 1; i < NUM_CLASSES; ++i) {{
        if (probs[i] > max_p) {{
            max_p = probs[i];
            best_cls = i;
        }}
    }}
    return best_cls;
}}

void predict_size_probabilities(const float features[NUM_INPUT_FEATURES], float probs[NUM_CLASSES]) {{
    // Layer 1: 4 -> 8 (ReLU)
    float h1[LAYER1_UNITS];
    for (int j = 0; j < LAYER1_UNITS; ++j) {{
        float sum = b1[j];
        for (int i = 0; i < NUM_INPUT_FEATURES; ++i) {{
            sum += features[i] * ((float)W1_q[i][j] * s_W1);
        }}
        h1[j] = sum > 0.0f ? sum : 0.0f; // ReLU
    }}

    // Layer 2: 8 -> 8 (ReLU)
    float h2[LAYER2_UNITS];
    for (int j = 0; j < LAYER2_UNITS; ++j) {{
        float sum = b2[j];
        for (int i = 0; i < LAYER1_UNITS; ++i) {{
            sum += h1[i] * ((float)W2_q[i][j] * s_W2);
        }}
        h2[j] = sum > 0.0f ? sum : 0.0f; // ReLU
    }}

    // Layer 3: 8 -> 6 (Logits)
    float logits[NUM_CLASSES];
    float max_l = -1e9f;
    for (int j = 0; j < NUM_CLASSES; ++j) {{
        float sum = b3[j];
        for (int i = 0; i < LAYER2_UNITS; ++i) {{
            sum += h2[i] * ((float)W3_q[i][j] * s_W3);
        }}
        logits[j] = sum;
        if (sum > max_l) max_l = sum;
    }}

    // Softmax
    float exp_sum = 0.0f;
    for (int j = 0; j < NUM_CLASSES; ++j) {{
        probs[j] = expf(logits[j] - max_l);
        exp_sum += probs[j];
    }}
    for (int j = 0; j < NUM_CLASSES; ++j) {{
        probs[j] /= exp_sum;
    }}
}}
"""

    h_path = os.path.join(output_dir, "model_data.h")
    cc_path = os.path.join(output_dir, "model_data.cc")
    with open(h_path, "w") as f:
        f.write(h_content)
    with open(cc_path, "w") as f:
        f.write(cc_content)
    return h_path, cc_path


def split_by_object(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset STRICTLY by object_name (Section 11 requirement).
    Ensures test objects were NEVER seen during training.
    """
    unique_objects = df["object_name"].unique()
    np.random.seed(seed)
    shuffled_objects = np.random.permutation(unique_objects)

    # Objects with '_unseen' in their name are explicitly prioritized for testing
    explicit_test = [o for o in shuffled_objects if "unseen" in o]
    other_objects = [o for o in shuffled_objects if "unseen" not in o]

    n_test = max(len(explicit_test), int(len(unique_objects) * test_ratio))
    test_objects = set(explicit_test + other_objects[: max(0, n_test - len(explicit_test))])
    train_objects = set([o for o in unique_objects if o not in test_objects])

    train_df = df[df["object_name"].isin(train_objects)].copy()
    test_df = df[df["object_name"].isin(test_objects)].copy()
    return train_df, test_df


def train_and_evaluate(csv_path: str = "data/features.csv") -> None:
    print("=" * 70)
    print("OBJECT SIZE CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"Dataset not found at '{csv_path}'. Run src/collect_data.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Total dataset samples: {len(df)}")
    print(f"Unique objects: {df['object_name'].nunique()} ({list(df['object_name'].unique())})")

    # 1. Object-wise Train / Test Split
    train_df, test_df = split_by_object(df)
    print("\n--- Object-Wise Split (Section 11) ---")
    print(f"Training Objects ({train_df['object_name'].nunique()}): {list(train_df['object_name'].unique())}")
    print(f"Testing Objects  ({test_df['object_name'].nunique()}): {list(test_df['object_name'].unique())}")
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values

    # 2. Baseline Model
    baseline = BaselineAreaClassifier()
    y_pred_base = baseline.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base, average="weighted")
    print(f"\n[Baseline Threshold]  Accuracy: {acc_base:.4f} | F1: {f1_base:.4f}")

    # 3. Model 1: Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt, average="weighted")
    print(f"[Decision Tree]       Accuracy: {acc_dt:.4f} | F1: {f1_dt:.4f}")

    # 4. Model 2: Tiny Neural Network (FP32 & INT8)
    tiny_nn = TinyNeuralNetwork(random_state=42)
    tiny_nn.fit(X_train, y_train)

    y_pred_fp32 = tiny_nn.predict_fp32(X_test)
    acc_fp32 = accuracy_score(y_test, y_pred_fp32)
    f1_fp32 = f1_score(y_test, y_pred_fp32, average="weighted")
    print(f"[Tiny NN - FP32]      Accuracy: {acc_fp32:.4f} | F1: {f1_fp32:.4f}")

    y_pred_int8 = tiny_nn.predict_int8(X_test)
    acc_int8 = accuracy_score(y_test, y_pred_int8)
    f1_int8 = f1_score(y_test, y_pred_int8, average="weighted")
    agreement = np.mean(y_pred_fp32 == y_pred_int8)
    print(f"[Tiny NN - INT8]      Accuracy: {acc_int8:.4f} | F1: {f1_int8:.4f} (Agreement with FP32: {agreement:.1%})")

    # 5. Save Models
    models_dir = "models/size_classifier"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(dt, os.path.join(models_dir, "decision_tree.joblib"))
    joblib.dump(tiny_nn, os.path.join(models_dir, "tiny_nn.joblib"))

    # Save weights as NPZ
    np.savez(
        os.path.join(models_dir, "tiny_nn_int8_weights.npz"),
        **tiny_nn.int8_weights,
    )

    # 6. Export Embedded C Arrays
    h_path, cc_path = export_c_arrays(tiny_nn, output_dir="embedded")
    print(f"\nEmbedded C headers generated:")
    print(f"  - {h_path} ({os.path.getsize(h_path)} bytes)")
    print(f"  - {cc_path} ({os.path.getsize(cc_path)} bytes)")

    # 7. Save Accuracy Summary
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "accuracy.txt")
    with open(summary_path, "w") as f:
        f.write("OBJECT SIZE CLASSIFIER - MODEL COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset Total: {len(df)} samples | Train: {len(train_df)} | Test: {len(test_df)}\n")
        f.write(f"Test Objects (Strict Unseen): {list(test_df['object_name'].unique())}\n\n")
        f.write(f"{'Model':<24} | {'Accuracy':<10} | {'F1-Score':<10}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Baseline Threshold':<24} | {acc_base:<10.4f} | {f1_base:<10.4f}\n")
        f.write(f"{'Decision Tree':<24} | {acc_dt:<10.4f} | {f1_dt:<10.4f}\n")
        f.write(f"{'Tiny NN (FP32)':<24} | {acc_fp32:<10.4f} | {f1_fp32:<10.4f}\n")
        f.write(f"{'Tiny NN (INT8)':<24} | {acc_int8:<10.4f} | {f1_int8:<10.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"INT8 to FP32 Agreement: {agreement:.2%}\n")
    print(f"\nAccuracy summary saved to: {summary_path}")


if __name__ == "__main__":
    train_and_evaluate()
