"""
models.py - Model definitions for the Object Size Classifier.

Contains:
  1. BaselineAreaClassifier (Fixed threshold heuristic)
  2. TinyNeuralNetwork (4 -> 8 -> 8 -> 6 MLP with FP32 and INT8 support)
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.features import CLASS_NAMES, classify_by_threshold


class BaselineAreaClassifier:
    """Baseline heuristic classifier using fixed area_ratio coverage thresholds."""

    def __init__(self):
        self.name = "Baseline Threshold"

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X[:, 2] is area_ratio
        return np.array([classify_by_threshold(x[2]) for x in X], dtype=np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        probs = np.zeros((len(X), 6), dtype=np.float32)
        for i, p in enumerate(preds):
            probs[i, p] = 1.0
        return probs


class TinyNeuralNetwork:
    """
    Tiny 4 -> 8 -> 8 -> 6 Neural Network with FP32 and INT8 quantization support.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(8, 8),
            activation="relu",
            solver="lbfgs",
            max_iter=3000,
            random_state=random_state,
        )
        self.is_fitted = False
        self.int8_weights = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TinyNeuralNetwork":
        self.mlp.fit(X, y)
        self.is_fitted = True
        self._quantize_int8(X)
        return self

    def predict_fp32(self, X: np.ndarray) -> np.ndarray:
        """Run pure-NumPy FP32 inference."""
        W1, W2, W3 = self.mlp.coefs_
        b1, b2, b3 = self.mlp.intercepts_

        h1 = np.maximum(0.0, np.dot(X, W1) + b1)
        h2 = np.maximum(0.0, np.dot(h1, W2) + b2)
        logits = np.dot(h2, W3) + b3
        return np.argmax(logits, axis=1)

    def predict_proba_fp32(self, X: np.ndarray) -> np.ndarray:
        """Softmax probabilities in FP32."""
        W1, W2, W3 = self.mlp.coefs_
        b1, b2, b3 = self.mlp.intercepts_

        h1 = np.maximum(0.0, np.dot(X, W1) + b1)
        h2 = np.maximum(0.0, np.dot(h1, W2) + b2)
        logits = np.dot(h2, W3) + b3
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def _quantize_int8(self, X_calibration: np.ndarray) -> None:
        """
        Calibrate and quantize weights and biases to symmetric INT8.
        Computes scale factors: S_w = max(|W|) / 127.0
        """
        W1, W2, W3 = self.mlp.coefs_
        b1, b2, b3 = self.mlp.intercepts_

        s_W1 = float(np.max(np.abs(W1)) / 127.0)
        s_W2 = float(np.max(np.abs(W2)) / 127.0)
        s_W3 = float(np.max(np.abs(W3)) / 127.0)

        W1_q = np.clip(np.round(W1 / s_W1), -128, 127).astype(np.int8)
        W2_q = np.clip(np.round(W2 / s_W2), -128, 127).astype(np.int8)
        W3_q = np.clip(np.round(W3 / s_W3), -128, 127).astype(np.int8)

        self.int8_weights = {
            "W1_q": W1_q,
            "s_W1": s_W1,
            "b1": b1.astype(np.float32),
            "W2_q": W2_q,
            "s_W2": s_W2,
            "b2": b2.astype(np.float32),
            "W3_q": W3_q,
            "s_W3": s_W3,
            "b3": b3.astype(np.float32),
        }

    def predict_int8(self, X: np.ndarray) -> np.ndarray:
        """Simulate INT8 quantized inference."""
        w = self.int8_weights
        W1 = w["W1_q"] * w["s_W1"]
        W2 = w["W2_q"] * w["s_W2"]
        W3 = w["W3_q"] * w["s_W3"]

        h1 = np.maximum(0.0, np.dot(X, W1) + w["b1"])
        h2 = np.maximum(0.0, np.dot(h1, W2) + w["b2"])
        logits = np.dot(h2, W3) + w["b3"]
        return np.argmax(logits, axis=1)

    def predict_proba_int8(self, X: np.ndarray) -> np.ndarray:
        w = self.int8_weights
        W1 = w["W1_q"] * w["s_W1"]
        W2 = w["W2_q"] * w["s_W2"]
        W3 = w["W3_q"] * w["s_W3"]

        h1 = np.maximum(0.0, np.dot(X, W1) + w["b1"])
        h2 = np.maximum(0.0, np.dot(h1, W2) + w["b2"])
        logits = np.dot(h2, W3) + w["b3"]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)
