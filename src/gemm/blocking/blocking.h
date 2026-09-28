#ifndef BSH_GEMM_BLOCKING_H
#define BSH_GEMM_BLOCKING_H

#include "../memory/allocator.h"
#include <stddef.h>

#define MC 64
#define NC 128
#define KC 64

typedef struct {
    size_t mc, nc, kc;
} block_config_t;

block_config_t* blocking_config_create(size_t mc, size_t nc, size_t kc);
void blocking_config_free(block_config_t* cfg);

void gemm_blocked_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta,
    const block_config_t* cfg
);

void pack_A_f32(
    float* packed,
    const matrix_t* A,
    size_t k_start, size_t k_end,
    size_t m_start, size_t m_end
);

void pack_B_f32(
    float* packed,
    const matrix_t* B,
    size_t k_start, size_t k_end,
    size_t n_start, size_t n_end
);

void microkernel_f32(
    float* C,
    const float* packed_A,
    const float* packed_B,
    size_t mc, size_t nc, size_t kc,
    size_t ldc
);

#endif
