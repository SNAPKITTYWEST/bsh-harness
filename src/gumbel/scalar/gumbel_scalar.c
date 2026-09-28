#include "gumbel_scalar.h"
#include "../softmax/softmax_scalar.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

float gumbel_sample_f32(xorshift128plus_t* rng) {
    if (!rng) return 0.0f;

    float u = rng_uniform_f32(rng);
    return -logf(-logf(u));
}

double gumbel_sample_f64(xorshift128plus_t* rng) {
    if (!rng) return 0.0;

    double u = rng_uniform_f64(rng);
    return -log(-log(u));
}

void gumbel_fill_f32(xorshift128plus_t* rng, float* data, size_t count) {
    if (!rng || !data || count == 0) return;
    for (size_t i = 0; i < count; i++) {
        data[i] = gumbel_sample_f32(rng);
    }
}

void gumbel_fill_f64(xorshift128plus_t* rng, double* data, size_t count) {
    if (!rng || !data || count == 0) return;
    for (size_t i = 0; i < count; i++) {
        data[i] = gumbel_sample_f64(rng);
    }
}

gumbel_softmax_config_t* gumbel_config_create(float temperature, int hard) {
    gumbel_softmax_config_t* cfg = (gumbel_softmax_config_t*)malloc(sizeof(gumbel_softmax_config_t));
    if (cfg) {
        cfg->temperature = temperature > 0.0f ? temperature : 1.0f;
        cfg->hard = hard;
    }
    return cfg;
}

void gumbel_config_free(gumbel_softmax_config_t* cfg) {
    if (cfg) free(cfg);
}

void gumbel_softmax_f32(
    float* output,
    const float* logits,
    const float* gumbel_noise,
    size_t count,
    float temperature,
    int hard
) {
    if (!output || !logits || !gumbel_noise || count == 0) return;
    if (temperature <= 0.0f) return;

    float* z = (float*)malloc(count * sizeof(float));
    if (!z) return;

    float max_z = -1e6f;
    for (size_t i = 0; i < count; i++) {
        z[i] = (logits[i] + gumbel_noise[i]) / temperature;
        if (z[i] > max_z) max_z = z[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        output[i] = expf(z[i] - max_z);
        sum += output[i];
    }

    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < count; i++) {
        output[i] *= inv_sum;
    }

    if (hard) {
        size_t max_idx = 0;
        float max_val = output[0];
        for (size_t i = 1; i < count; i++) {
            if (output[i] > max_val) {
                max_val = output[i];
                max_idx = i;
            }
        }

        memset(output, 0, count * sizeof(float));
        output[max_idx] = 1.0f;
    }

    free(z);
}

void gumbel_softmax_f32_batch(
    float* output,
    const float* logits,
    xorshift128plus_t* rng,
    size_t batch_size,
    size_t class_count,
    float temperature,
    int hard
) {
    if (!output || !logits || !rng) return;

    float* noise = (float*)malloc(class_count * sizeof(float));
    if (!noise) return;

    for (size_t b = 0; b < batch_size; b++) {
        gumbel_fill_f32(rng, noise, class_count);
        gumbel_softmax_f32(
            output + b * class_count,
            logits + b * class_count,
            noise,
            class_count,
            temperature,
            hard
        );
    }

    free(noise);
}

void gumbel_softmax_gradient_f32(
    float* grad_logits,
    const float* grad_output,
    const float* probs,
    size_t count
) {
    if (!grad_logits || !grad_output || !probs || count == 0) return;

    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += grad_output[i] * probs[i];
    }

    for (size_t i = 0; i < count; i++) {
        grad_logits[i] = probs[i] * (grad_output[i] - sum);
    }
}

gumbel_state_f32* gumbel_state_create(size_t count, float temperature, int hard) {
    if (count == 0) return NULL;

    gumbel_state_f32* state = (gumbel_state_f32*)malloc(sizeof(gumbel_state_f32));
    if (!state) return NULL;

    state->logits = (float*)aligned_alloc(32, count * sizeof(float));
    state->noise = (float*)aligned_alloc(32, count * sizeof(float));
    state->output = (float*)aligned_alloc(32, count * sizeof(float));
    state->soft_output = (float*)aligned_alloc(32, count * sizeof(float));
    state->count = count;
    state->temperature = temperature > 0.0f ? temperature : 1.0f;
    state->hard = hard;

    if (!state->logits || !state->noise || !state->output || !state->soft_output) {
        gumbel_state_free(state);
        return NULL;
    }

    memset(state->logits, 0, count * sizeof(float));
    memset(state->noise, 0, count * sizeof(float));
    memset(state->output, 0, count * sizeof(float));
    memset(state->soft_output, 0, count * sizeof(float));

    return state;
}

void gumbel_state_free(gumbel_state_f32* state) {
    if (state) {
        free(state->logits);
        free(state->noise);
        free(state->output);
        free(state->soft_output);
        free(state);
    }
}

void gumbel_state_forward(gumbel_state_f32* state, xorshift128plus_t* rng) {
    if (!state || !rng) return;

    gumbel_fill_f32(rng, state->noise, state->count);

    memcpy(state->soft_output, state->output, state->count * sizeof(float));
    gumbel_softmax_f32(
        state->soft_output,
        state->logits,
        state->noise,
        state->count,
        state->temperature,
        0
    );

    gumbel_softmax_f32(
        state->output,
        state->logits,
        state->noise,
        state->count,
        state->temperature,
        state->hard
    );
}

void gumbel_state_backward(gumbel_state_f32* state, const float* grad_output) {
    if (!state || !grad_output) return;

    float* grad_logits = (float*)malloc(state->count * sizeof(float));
    if (!grad_logits) return;

    gumbel_softmax_gradient_f32(
        grad_logits,
        grad_output,
        state->soft_output,
        state->count
    );

    free(grad_logits);
}
