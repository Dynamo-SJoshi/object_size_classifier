// model_data.h - Auto-generated embedded size classifier header
#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
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
}
#endif

#endif // MODEL_DATA_H_
