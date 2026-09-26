/*
 * test_embedded.c - Standalone embedded C test runner for size classifier.
 *
 * Demonstrates how a microcontroller firmware invokes the size classifier:
 * 4 input features -> predict_size_class() -> predicted size category (0-5).
 * Zero external libraries required (pure standard C).
 */

#include <stdio.h>
#include "../model_data.h"

static const char* CLASS_NAMES[6] = {
    "VERY_SMALL",
    "SMALL",
    "MEDIUM",
    "LARGE",
    "VERY_LARGE",
    "HUGE"
};

int main(void) {
    printf("===================================================\n");
    printf("EMBEDDED C SIZE CLASSIFIER INFERENCE TEST\n");
    printf("===================================================\n\n");

    // Test Case 1: Prompt Example (160x180 box in 640x480 frame)
    // width_ratio = 0.25, height_ratio = 0.375, area_ratio = 0.09375, aspect_ratio = 0.8889
    float sample_medium[4] = {0.25f, 0.375f, 0.09375f, 0.888889f};

    // Test Case 2: Very Small object (coverage 1.2%)
    float sample_vsmall[4] = {0.08f, 0.15f, 0.0120f, 0.5333f};

    // Test Case 3: Large object (coverage 28.5%)
    float sample_large[4] = {0.50f, 0.57f, 0.2850f, 0.8772f};

    float probs[6];

    // Inference 1
    int cls1 = predict_size_class(sample_medium);
    predict_size_probabilities(sample_medium, probs);
    printf("Test 1 (Coverage: 9.38%%):\n");
    printf("  Predicted Class : %d (%s)\n", cls1, CLASS_NAMES[cls1]);
    printf("  Class Probabilities: [VS: %.1f%%, S: %.1f%%, M: %.1f%%, L: %.1f%%, VL: %.1f%%, H: %.1f%%]\n\n",
           probs[0]*100, probs[1]*100, probs[2]*100, probs[3]*100, probs[4]*100, probs[5]*100);

    // Inference 2
    int cls2 = predict_size_class(sample_vsmall);
    predict_size_probabilities(sample_vsmall, probs);
    printf("Test 2 (Coverage: 1.20%%):\n");
    printf("  Predicted Class : %d (%s)\n", cls2, CLASS_NAMES[cls2]);
    printf("  Confidence      : %.1f%%\n\n", probs[cls2] * 100);

    // Inference 3
    int cls3 = predict_size_class(sample_large);
    predict_size_probabilities(sample_large, probs);
    printf("Test 3 (Coverage: 28.50%%):\n");
    printf("  Predicted Class : %d (%s)\n", cls3, CLASS_NAMES[cls3]);
    printf("  Confidence      : %.1f%%\n\n", probs[cls3] * 100);

    printf("Embedded C inference executed successfully!\n");
    return 0;
}
