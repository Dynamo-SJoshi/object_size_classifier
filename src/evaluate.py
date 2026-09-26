"""
evaluate.py - Detailed evaluation, confusion matrix generation, and object-independence test.

Evaluates:
  1. Detailed Metrics (Accuracy, Precision, Recall, F1 per class)
  2. Confusion Matrix plots saved to results/confusion_matrix.png
  3. Distance Sweep & Object-Independence verification (Section 15)
"""

from typing import Dict, List, Tuple
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.features import CLASS_NAMES, FEATURE_COLUMNS, extract_features, get_class_name
from src.train import BaselineAreaClassifier, TinyNeuralNetwork, split_by_object


def plot_confusion_matrices(
    y_test: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    save_path: str = "results/confusion_matrix.png",
) -> None:
    """Plot and save confusion matrices for all evaluated models side-by-side."""
    classes = [CLASS_NAMES[i] for i in range(6)]
    short_labels = ["VS", "S", "M", "L", "VL", "H"]
    n_models = len(preds_dict)

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5.0))
    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_test, y_pred, labels=list(range(6)))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(short_labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(short_labels)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(short_labels)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("Actual Class", fontsize=10)

        # Annotate cell counts
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix visualization saved to: {save_path}")


def run_object_independence_experiment(
    tiny_nn: TinyNeuralNetwork,
    unseen_object: str = "shoe",
    aspect_ratio: float = 1.65,
) -> pd.DataFrame:
    """
    Test Section 15: Object-Independence Experiment.
    Move an unseen object from far away to very close and verify monotonic size classification.
    """
    image_w, image_h = 640.0, 480.0
    total_area = image_w * image_h

    # Simulated distances and corresponding apparent coverage
    steps = [
        {"distance_desc": "3.5m (Far Away)", "target_coverage": 0.010, "expected": 0},
        {"distance_desc": "2.2m (Mid-Far)",  "target_coverage": 0.045, "expected": 1},
        {"distance_desc": "1.4m (Medium)",   "target_coverage": 0.130, "expected": 2},
        {"distance_desc": "0.9m (Close)",    "target_coverage": 0.280, "expected": 3},
        {"distance_desc": "0.5m (Very Close)","target_coverage": 0.520, "expected": 4},
        {"distance_desc": "0.25m (Huge/Near)","target_coverage": 0.820, "expected": 5},
    ]

    results = []
    for step in steps:
        target_area = step["target_coverage"] * total_area
        box_h = np.sqrt(target_area / aspect_ratio)
        box_w = aspect_ratio * box_h
        feats = extract_features([100, 100, box_w, box_h], (image_h, image_w))

        # Predict with Tiny NN (FP32 & INT8)
        pred_fp32 = int(tiny_nn.predict_fp32(feats.reshape(1, -1))[0])
        pred_int8 = int(tiny_nn.predict_int8(feats.reshape(1, -1))[0])
        prob_fp32 = float(np.max(tiny_nn.predict_proba_fp32(feats.reshape(1, -1))[0]))

        results.append({
            "Distance": step["distance_desc"],
            "Coverage %": f"{step['target_coverage'] * 100:.1f}%",
            "Expected": CLASS_NAMES[step["expected"]],
            "FP32 Prediction": CLASS_NAMES[pred_fp32],
            "INT8 Prediction": CLASS_NAMES[pred_int8],
            "Confidence": f"{prob_fp32:.1%}",
            "Passed": pred_fp32 == step["expected"],
        })

    return pd.DataFrame(results)


def evaluate_all(csv_path: str = "data/features.csv") -> None:
    print("=" * 70)
    print("EVALUATION & OBJECT-INDEPENDENCE SUITE")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    train_df, test_df = split_by_object(df)
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["label"].values

    models_dir = "models/size_classifier"
    dt_path = os.path.join(models_dir, "decision_tree.joblib")
    nn_path = os.path.join(models_dir, "tiny_nn.joblib")

    if not os.path.exists(dt_path) or not os.path.exists(nn_path):
        print("Models not found. Running training first...")
        from src.train import train_and_evaluate
        train_and_evaluate(csv_path)

    dt = joblib.load(dt_path)
    tiny_nn: TinyNeuralNetwork = joblib.load(nn_path)
    baseline = BaselineAreaClassifier()

    # Predictions
    preds = {
        "Baseline (Threshold)": baseline.predict(X_test),
        "Decision Tree": dt.predict(X_test),
        "Tiny NN (FP32)": tiny_nn.predict_fp32(X_test),
        "Tiny NN (INT8)": tiny_nn.predict_int8(X_test),
    }

    # Print Detailed Classification Reports
    for name, y_pred in preds.items():
        print(f"\n--- {name} Classification Report (Unseen Test Objects) ---")
        target_names = [CLASS_NAMES[i] for i in range(6)]
        print(classification_report(y_test, y_pred, target_names=target_names, digits=3, zero_division=0))

    # Plot Confusion Matrices
    plot_confusion_matrices(y_test, preds, save_path="results/confusion_matrix.png")

    # Section 15: Run Object-Independence Experiment
    print("\n" + "=" * 70)
    print("SECTION 15: OBJECT-INDEPENDENCE EXPERIMENT (Unseen Object: Shoe)")
    print("=" * 70)
    exp_df = run_object_independence_experiment(tiny_nn, unseen_object="shoe", aspect_ratio=1.65)
    print(exp_df.to_string(index=False))

    # Save full evaluation report
    report_path = "results/evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("OBJECT SIZE CLASSIFIER - FULL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. MODEL PERFORMANCE ON UNSEEN OBJECTS:\n")
        for name, y_pred in preds.items():
            f.write(f"\n--- {name} ---\n")
            f.write(classification_report(y_test, y_pred, target_names=[CLASS_NAMES[i] for i in range(6)], digits=3, zero_division=0))
        f.write("\n\n2. OBJECT-INDEPENDENCE EXPERIMENT (UNSEEN OBJECT: SHOE):\n")
        f.write(exp_df.to_string(index=False))
        f.write("\n")
    print(f"\nFull evaluation report saved to: {report_path}")


if __name__ == "__main__":
    evaluate_all()
