"""
tests/test_pipeline.py - Comprehensive Unit & Integration Tests for Object Size Classifier.
Uses Python's standard unittest library (zero external test dependencies required).
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    classify_by_threshold,
    extract_features,
    get_class_name,
)
from src.models import BaselineAreaClassifier, TinyNeuralNetwork
from src.inference import SizeClassifierPipeline


class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features_values(self):
        # 100x200 box in 400x500 image (w=400, h=500)
        bbox = [50, 60, 100, 200]
        img_size = (500, 400)  # (h, w)
        feats = extract_features(bbox, img_size)

        self.assertEqual(len(feats), 4)
        # width_ratio = 100 / 400 = 0.25
        self.assertAlmostEqual(feats[0], 0.25, places=5)
        # height_ratio = 200 / 500 = 0.40
        self.assertAlmostEqual(feats[1], 0.40, places=5)
        # area_ratio = (100 * 200) / (400 * 500) = 0.10
        self.assertAlmostEqual(feats[2], 0.10, places=5)
        # aspect_ratio = 100 / 200 = 0.50
        self.assertAlmostEqual(feats[3], 0.50, places=5)

    def test_threshold_classification_classes(self):
        # 0: VERY_SMALL (< 2%)
        self.assertEqual(classify_by_threshold(0.01), 0)
        self.assertEqual(get_class_name(0), "VERY_SMALL")

        # 1: SMALL (2% - 8%)
        self.assertEqual(classify_by_threshold(0.05), 1)
        self.assertEqual(get_class_name(1), "SMALL")

        # 2: MEDIUM (8% - 20%)
        self.assertEqual(classify_by_threshold(0.12), 2)
        self.assertEqual(get_class_name(2), "MEDIUM")

        # 3: LARGE (20% - 40%)
        self.assertEqual(classify_by_threshold(0.30), 3)
        self.assertEqual(get_class_name(3), "LARGE")

        # 4: VERY_LARGE (40% - 70%)
        self.assertEqual(classify_by_threshold(0.55), 4)
        self.assertEqual(get_class_name(4), "VERY_LARGE")

        # 5: HUGE (>= 70%)
        self.assertEqual(classify_by_threshold(0.85), 5)
        self.assertEqual(get_class_name(5), "HUGE")


class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.uniform(0.01, 0.99, size=(60, 4)).astype(np.float32)
        self.y = np.array([classify_by_threshold(x[2]) for x in self.X], dtype=np.int32)

    def test_baseline_classifier(self):
        model = BaselineAreaClassifier()
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y))
        self.assertTrue(np.all((preds >= 0) & (preds < 6)))

    def test_tiny_neural_network_fp32_int8(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiny = TinyNeuralNetwork(max_iter=100, random_state=42)
            tiny.fit(self.X, self.y)

        # 2D batch inference
        preds_fp32 = tiny.predict_fp32(self.X[:5])
        preds_int8 = tiny.predict_int8(self.X[:5])
        self.assertEqual(len(preds_fp32), 5)
        self.assertEqual(len(preds_int8), 5)

        # 1D single vector inference
        single_fp32 = tiny.predict_fp32(self.X[0])
        single_int8 = tiny.predict_int8(self.X[0])
        self.assertTrue(isinstance(int(single_fp32), int))
        self.assertTrue(isinstance(int(single_int8), int))

        # Probabilities sum to 1.0
        probs_fp32 = tiny.predict_proba_fp32(self.X[:3])
        probs_int8 = tiny.predict_proba_int8(self.X[:3])
        self.assertTrue(np.allclose(np.sum(probs_fp32, axis=1), 1.0))
        self.assertTrue(np.allclose(np.sum(probs_int8, axis=1), 1.0))


class TestInferencePipeline(unittest.TestCase):
    def test_pipeline_on_synthetic_frame(self):
        # Create a synthetic image with a distinct centered rectangle
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        frame[165:315, 245:395] = 20

        pipeline = SizeClassifierPipeline(use_int8=True, filter_person=False)
        result = pipeline.process_frame(frame)

        self.assertIn("status", result)
        self.assertIn("coverage_pct", result)
        self.assertIn("smoothed_size_class", result)
        self.assertIn("det_time_ms", result)
        self.assertIn("cls_time_ms", result)

        hud = pipeline.render_hud(frame, result)
        self.assertEqual(hud.shape, frame.shape)
        self.assertEqual(hud.dtype, np.uint8)


class TestEmbeddedArtifacts(unittest.TestCase):
    def test_c_headers_exist(self):
        root = Path(__file__).resolve().parent.parent
        h_file = root / "embedded" / "model_data.h"
        cc_file = root / "embedded" / "model_data.cc"
        self.assertTrue(h_file.exists(), "embedded/model_data.h missing")
        self.assertTrue(cc_file.exists(), "embedded/model_data.cc missing")
        self.assertGreater(h_file.stat().st_size, 200)
        self.assertGreater(cc_file.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
