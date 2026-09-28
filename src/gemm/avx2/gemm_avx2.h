#ifndef BSH_GEMM_AVX2_H
#define BSH_GEMM_AVX2_H

#include "../memory/allocator.h"

void gemm_avx2_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
);

void microkernel_avx2_4x8(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
);

void microkernel_avx2_8x8(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
);

#endif
