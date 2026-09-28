#ifndef BSH_GUMBEL_RNG_XORSHIFT_H
#define BSH_GUMBEL_RNG_XORSHIFT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    uint64_t state[2];
} xorshift128plus_t;

xorshift128plus_t* rng_create(uint64_t seed);
void rng_free(xorshift128plus_t* rng);
void rng_seed(xorshift128plus_t* rng, uint64_t seed);

uint64_t rng_next_u64(xorshift128plus_t* rng);
uint32_t rng_next_u32(xorshift128plus_t* rng);

float rng_uniform_f32(xorshift128plus_t* rng);
double rng_uniform_f64(xorshift128plus_t* rng);

void rng_fill_f32(xorshift128plus_t* rng, float* data, size_t count);
void rng_fill_f64(xorshift128plus_t* rng, double* data, size_t count);

#endif
