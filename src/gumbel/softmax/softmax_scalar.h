#ifndef BSH_GUMBEL_SOFTMAX_SCALAR_H
#define BSH_GUMBEL_SOFTMAX_SCALAR_H

#include <stddef.h>

void softmax_scalar_f32(
    float* output,
    const float* input,
    size_t count
);

void softmax_scalar_f64(
    double* output,
    const double* input,
    size_t count
);

void softmax_batch_f32(
    float* output,
    const float* input,
    size_t batch_size,
    size_t class_count
);

void softmax_batch_f64(
    double* output,
    const double* input,
    size_t batch_size,
    size_t class_count
);

typedef struct {
    float* logits;
    float* probs;
    size_t count;
} softmax_state_f32;

softmax_state_f32* softmax_state_create(size_t count);
void softmax_state_free(softmax_state_f32* state);
void softmax_state_compute(softmax_state_f32* state);

#endif
