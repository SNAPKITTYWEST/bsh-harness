#ifndef BSH_GEMM_SCALAR_H
#define BSH_GEMM_SCALAR_H

#include "../memory/allocator.h"

void gemm_scalar_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
);

void gemm_scalar_f64(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    double alpha,
    double beta
);

typedef struct {
    size_t M, N, K;
    float alpha, beta;
    float max_error;
    float mean_error;
    double elapsed_ns;
    double gflops;
} gemm_result_t;

gemm_result_t gemm_scalar_benchmark(size_t M, size_t N, size_t K);

#endif
