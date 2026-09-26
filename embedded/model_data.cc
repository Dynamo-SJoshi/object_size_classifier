// model_data.cc - Auto-generated embedded size classifier implementation
#include "model_data.h"
#include <math.h>

const float s_W1 = 0.12734237f;
const float s_W2 = 0.08579412f;
const float s_W3 = 0.13583334f;

const int8_t W1_q[4][8] = {
    {-1, 75, 11, 86, 66, -4, -30, 4},
    {1, -2, 12, 91, 48, -3, -2, -4},
    {-2, 36, 5, 127, 41, -4, -12, -1},
    {0, 3, -18, -35, 52, -5, 39, -4}
};
const float b1[8] = {-0.615110f, 0.056262f, -2.397608f, -13.090212f, 4.013677f, -0.568978f, 5.394166f, -0.084637f};

const int8_t W2_q[8][8] = {
    {-5, 0, -7, 6, -3, 2, -3, 0},
    {35, -12, 127, 3, -66, 16, -39, -38},
    {58, -4, -20, -39, -7, 0, -37, -3},
    {3, -12, 75, 30, -26, 42, 1, -36},
    {-35, -5, 22, 26, -6, 10, -16, -10},
    {5, 2, -2, -6, -3, -2, 3, 2},
    {57, 0, -54, -77, -21, -19, -67, 6},
    {0, -1, -7, -6, -7, 2, -3, 0}
};
const float b2[8] = {14.827366f, -0.675421f, 5.841774f, -6.274712f, -3.360149f, -7.022214f, -5.009784f, -2.630088f};

const int8_t W3_q[8][6] = {
    {75, 66, 19, -26, -51, -71},
    {4, 0, 3, 4, -2, -4},
    {-127, -23, 26, 39, 44, 39},
    {-42, 5, -2, 16, 22, -6},
    {9, 8, -14, -10, -3, 17},
    {-104, -35, -31, 53, 37, 73},
    {7, 14, -13, -13, 0, -3},
    {-2, -3, -4, 2, -6, 20}
};
const float b3[6] = {4.921513f, 12.240489f, 3.447745f, -2.376771f, -4.680367f, -13.428244f};

int predict_size_class(const float features[NUM_INPUT_FEATURES]) {
    float probs[NUM_CLASSES];
    predict_size_probabilities(features, probs);
    int best_cls = 0;
    float max_p = probs[0];
    for (int i = 1; i < NUM_CLASSES; ++i) {
        if (probs[i] > max_p) {
            max_p = probs[i];
            best_cls = i;
        }
    }
    return best_cls;
}

void predict_size_probabilities(const float features[NUM_INPUT_FEATURES], float probs[NUM_CLASSES]) {
    // Layer 1: 4 -> 8 (ReLU)
    float h1[LAYER1_UNITS];
    for (int j = 0; j < LAYER1_UNITS; ++j) {
        float sum = b1[j];
        for (int i = 0; i < NUM_INPUT_FEATURES; ++i) {
            sum += features[i] * ((float)W1_q[i][j] * s_W1);
        }
        h1[j] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    // Layer 2: 8 -> 8 (ReLU)
    float h2[LAYER2_UNITS];
    for (int j = 0; j < LAYER2_UNITS; ++j) {
        float sum = b2[j];
        for (int i = 0; i < LAYER1_UNITS; ++i) {
            sum += h1[i] * ((float)W2_q[i][j] * s_W2);
        }
        h2[j] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    // Layer 3: 8 -> 6 (Logits)
    float logits[NUM_CLASSES];
    float max_l = -1e9f;
    for (int j = 0; j < NUM_CLASSES; ++j) {
        float sum = b3[j];
        for (int i = 0; i < LAYER2_UNITS; ++i) {
            sum += h2[i] * ((float)W3_q[i][j] * s_W3);
        }
        logits[j] = sum;
        if (sum > max_l) max_l = sum;
    }

    // Softmax
    float exp_sum = 0.0f;
    for (int j = 0; j < NUM_CLASSES; ++j) {
        probs[j] = expf(logits[j] - max_l);
        exp_sum += probs[j];
    }
    for (int j = 0; j < NUM_CLASSES; ++j) {
        probs[j] /= exp_sum;
    }
}
