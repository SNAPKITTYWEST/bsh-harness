#ifndef BSH_GUMBEL_SCALAR_H
#define BSH_GUMBEL_SCALAR_H

#include "../rng/xorshift.h"
#include <stddef.h>

float gumbel_sample_f32(xorshift128plus_t* rng);
double gumbel_sample_f64(xorshift128plus_t* rng);

void gumbel_fill_f32(xorshift128plus_t* rng, float* data, size_t count);
void gumbel_fill_f64(xorshift128plus_t* rng, double* data, size_t count);

typedef struct {
    float temperature;
    int hard;
} gumbel_softmax_config_t;

gumbel_softmax_config_t* gumbel_config_create(float temperature, int hard);
void gumbel_config_free(gumbel_softmax_config_t* cfg);

void gumbel_softmax_f32(
    float* output,
    const float* logits,
    const float* gumbel_noise,
    size_t count,
    float temperature,
    int hard
);

void gumbel_softmax_f32_batch(
    float* output,
    const float* logits,
    xorshift128plus_t* rng,
    size_t batch_size,
    size_t class_count,
    float temperature,
    int hard
);

void gumbel_softmax_gradient_f32(
    float* grad_logits,
    const float* grad_output,
    const float* probs,
    size_t count
);

typedef struct {
    float* logits;
    float* noise;
    float* output;
    float* soft_output;
    size_t count;
    float temperature;
    int hard;
} gumbel_state_f32;

gumbel_state_f32* gumbel_state_create(size_t count, float temperature, int hard);
void gumbel_state_free(gumbel_state_f32* state);
void gumbel_state_forward(gumbel_state_f32* state, xorshift128plus_t* rng);
void gumbel_state_backward(gumbel_state_f32* state, const float* grad_output);

#endif
