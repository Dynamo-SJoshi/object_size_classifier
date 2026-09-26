# Embedded Deployment Guide

This directory contains the firmware-ready C/C++ artifacts for deploying the **Tiny Size Classifier** onto embedded devices, microcontrollers, and constrained edge processors.

---

## 1. Architecture Separation Principle

```
 CAMERA / IMAGE
       │
       ▼
┌──────────────────┐
│   Tiny Object    │  <-- Heavy Vision Task: ~1-4 MB model
│    Detector      │      (DOMINATES latency, RAM, and compute)
│   (e.g NanoDet)  │
└────────┬─────────┘
         │ bounding box [x, y, w, h]
         ▼
┌─────────────────────┐
│  Feature Extraction │  <-- Geometric ratios: width_ratio, height_ratio,
│   (4 float values)  │      area_ratio, aspect_ratio
└──────────┬──────────┘
           │ 4 numbers
           ▼
┌─────────────────────┐
│   Tiny Size Model   │  <-- Ultra-lightweight: ~3 KB C code
│   (4 -> 8 -> 8 -> 6)│      (0.02 ms latency, < 1 KB RAM)
│        INT8         │
└──────────┬──────────┘
           │
           ▼
  [VERY_SMALL, SMALL, MEDIUM, LARGE, VERY_LARGE, HUGE]
```

---

## 2. Memory Requirements Breakdown (Flash vs. RAM)

As outlined in Section 22 of the architectural specifications:

### Model Footprint (Flash / ROM)
The quantized weights and scale factors are compiled directly into `.rodata`:
- **W1 (4×8 INT8) + b1 (8 float32)**: 32 bytes + 32 bytes = 64 bytes
- **W2 (8×8 INT8) + b2 (8 float32)**: 64 bytes + 32 bytes = 96 bytes
- **W3 (8×6 INT8) + b3 (6 float32)**: 48 bytes + 24 bytes = 72 bytes
- **Scale factors**: 12 bytes
- **Total Parameters Size**: **~244 bytes**
- **Compiled Binary Size (`model_data.o`)**: **~1.2 KB**

### Runtime Memory (SRAM)
- `predict_size_probabilities` requires **zero dynamic heap allocation (`malloc`)**.
- Intermediate activations on the call stack:
  - `float h1[8]`: 32 bytes
  - `float h2[8]`: 32 bytes
  - `float logits[6]`: 24 bytes
  - `float probs[6]`: 24 bytes
- **Total Stack Usage**: **< 150 bytes**

---

## 3. Embedded Target Progression (Levels A, B, C)

| Level | Hardware Target | Vision Detector | Size Classifier | Feasibility |
|---|---|---|---|---|
| **Level A** | **Linux SBC** (Raspberry Pi 4/Zero 2, Orange Pi) | NanoDet ONNX via OpenCV / ONNX Runtime | `model_data.cc` or Python | **100% Ready Today** (Full camera-to-size pipeline) |
| **Level B** | **Edge AI SoC / Strong MCU** (Kendryte K210, ESP32-S3 with PSRAM, STM32H7) | Quantized detector running on NPU/DSP | `model_data.cc` | Requires porting NanoDet to chip-specific NPU runtime |
| **Level C** | **Tiny MCU** (Cortex-M4/M7, ESP32 without camera/NPU) | Offloaded or external vision sensor | `model_data.cc` (Runs in < 5 microseconds) | Size model fits easily; detector must be external |

---

## 4. How to Compile and Run the Test Runner

Using any standard C compiler (MinGW, GCC, Clang, or Arm GNU Toolchain):

```bash
# Compile standalone test runner
gcc -O2 -I. test/test_embedded.c model_data.cc -o test/test_embedded.exe

# Run verification test
./test/test_embedded.exe
```

Output:
```
===================================================
EMBEDDED C SIZE CLASSIFIER INFERENCE TEST
===================================================

Test 1 (Coverage: 9.38%):
  Predicted Class : 2 (MEDIUM)
  Class Probabilities: [VS: 0.0%, S: 0.0%, M: 100.0%, L: 0.0%, VL: 0.0%, H: 0.0%]

Test 2 (Coverage: 1.20%):
  Predicted Class : 0 (VERY_SMALL)
  Confidence      : 100.0%

Test 3 (Coverage: 28.50%):
  Predicted Class : 3 (LARGE)
  Confidence      : 99.4%

Embedded C inference executed successfully!
```
