#include "xorshift.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

xorshift128plus_t* rng_create(uint64_t seed) {
    xorshift128plus_t* rng = (xorshift128plus_t*)malloc(sizeof(xorshift128plus_t));
    if (!rng) return NULL;
    rng_seed(rng, seed);
    return rng;
}

void rng_free(xorshift128plus_t* rng) {
    if (rng) free(rng);
}

void rng_seed(xorshift128plus_t* rng, uint64_t seed) {
    if (!rng) return;
    rng->state[0] = seed ^ 0x9e3779b97f4a7c15ULL;
    rng->state[1] = seed + 0xbf58476d1ce4e5b9ULL;
    for (int i = 0; i < 10; i++) {
        rng_next_u64(rng);
    }
}

uint64_t rng_next_u64(xorshift128plus_t* rng) {
    if (!rng) return 0;

    uint64_t s1 = rng->state[0];
    uint64_t s0 = rng->state[1];
    uint64_t result = s0 + s1;

    s1 ^= s1 << 23;
    s1 ^= s1 >> 17;
    s1 ^= s0 ^ (s0 >> 26);

    rng->state[0] = s0;
    rng->state[1] = s1;

    return result;
}

uint32_t rng_next_u32(xorshift128plus_t* rng) {
    return (uint32_t)(rng_next_u64(rng) >> 32);
}

float rng_uniform_f32(xorshift128plus_t* rng) {
    if (!rng) return 0.5f;

    uint32_t u = rng_next_u32(rng);
    float f = (float)(u >> 8) * (1.0f / 16777216.0f);

    while (f == 0.0f || f == 1.0f) {
        u = rng_next_u32(rng);
        f = (float)(u >> 8) * (1.0f / 16777216.0f);
    }

    return f;
}

double rng_uniform_f64(xorshift128plus_t* rng) {
    if (!rng) return 0.5;

    uint64_t u = rng_next_u64(rng);
    double d = (double)(u >> 11) * (1.0 / 9007199254740992.0);

    while (d == 0.0 || d == 1.0) {
        u = rng_next_u64(rng);
        d = (double)(u >> 11) * (1.0 / 9007199254740992.0);
    }

    return d;
}

void rng_fill_f32(xorshift128plus_t* rng, float* data, size_t count) {
    if (!rng || !data) return;
    for (size_t i = 0; i < count; i++) {
        data[i] = rng_uniform_f32(rng);
    }
}

void rng_fill_f64(xorshift128plus_t* rng, double* data, size_t count) {
    if (!rng || !data) return;
    for (size_t i = 0; i < count; i++) {
        data[i] = rng_uniform_f64(rng);
    }
}
