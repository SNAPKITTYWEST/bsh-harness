#ifndef BSH_GEMM_AVX512_H
#define BSH_GEMM_AVX512_H

#include "../memory/allocator.h"

void gemm_avx512_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
);

void microkernel_avx512_4x16(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
);

void microkernel_avx512_16x16(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
);

#endif
