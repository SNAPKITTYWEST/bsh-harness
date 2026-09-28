#ifndef BSH_GEMM_NEON_H
#define BSH_GEMM_NEON_H

#include "../memory/allocator.h"

void gemm_neon_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
);

void microkernel_neon_4x4(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
);

void microkernel_neon_8x8(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
);

#endif
