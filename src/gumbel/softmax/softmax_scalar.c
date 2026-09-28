#include "softmax_scalar.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void softmax_scalar_f32(
    float* output,
    const float* input,
    size_t count
) {
    if (!output || !input || count == 0) return;

    float max_val = input[0];
    for (size_t i = 1; i < count; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < count; i++) {
        output[i] *= inv_sum;
    }
}

void softmax_scalar_f64(
    double* output,
    const double* input,
    size_t count
) {
    if (!output || !input || count == 0) return;

    double max_val = input[0];
    for (size_t i = 1; i < count; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    double inv_sum = 1.0 / sum;
    for (size_t i = 0; i < count; i++) {
        output[i] *= inv_sum;
    }
}

void softmax_batch_f32(
    float* output,
    const float* input,
    size_t batch_size,
    size_t class_count
) {
    if (!output || !input) return;

    for (size_t b = 0; b < batch_size; b++) {
        softmax_scalar_f32(
            output + b * class_count,
            input + b * class_count,
            class_count
        );
    }
}

void softmax_batch_f64(
    double* output,
    const double* input,
    size_t batch_size,
    size_t class_count
) {
    if (!output || !input) return;

    for (size_t b = 0; b < batch_size; b++) {
        softmax_scalar_f64(
            output + b * class_count,
            input + b * class_count,
            class_count
        );
    }
}

softmax_state_f32* softmax_state_create(size_t count) {
    if (count == 0) return NULL;

    softmax_state_f32* state = (softmax_state_f32*)malloc(sizeof(softmax_state_f32));
    if (!state) return NULL;

    state->logits = (float*)aligned_alloc(32, count * sizeof(float));
    state->probs = (float*)aligned_alloc(32, count * sizeof(float));
    state->count = count;

    if (!state->logits || !state->probs) {
        free(state->logits);
        free(state->probs);
        free(state);
        return NULL;
    }

    memset(state->logits, 0, count * sizeof(float));
    memset(state->probs, 0, count * sizeof(float));

    return state;
}

void softmax_state_free(softmax_state_f32* state) {
    if (state) {
        free(state->logits);
        free(state->probs);
        free(state);
    }
}

void softmax_state_compute(softmax_state_f32* state) {
    if (!state) return;
    softmax_scalar_f32(state->probs, state->logits, state->count);
}
