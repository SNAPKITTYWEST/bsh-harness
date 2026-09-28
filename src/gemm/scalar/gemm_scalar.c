#include "gemm_scalar.h"
#include "primitives.h"
#include <time.h>
#include <stdlib.h>

void gemm_scalar_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
) {
    if (!A || !B || !C) return;
    if (A->cols != B->rows) return;
    if (A->rows != C->rows || B->cols != C->cols) return;

    size_t M = A->rows;
    size_t N = B->cols;
    size_t K = A->cols;

    const float* a_ptr = matrix_ptr((matrix_t*)A);
    const float* b_ptr = matrix_ptr((matrix_t*)B);
    float* c_ptr = matrix_ptr(C);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum = scalar_fma_f32(a_ptr[i * A->stride + k], b_ptr[k * B->stride + j], sum);
            }
            c_ptr[i * C->stride + j] = scalar_fma_f32(alpha, sum, scalar_mul_f32(beta, c_ptr[i * C->stride + j]));
        }
    }
}

void gemm_scalar_f64(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    double alpha,
    double beta
) {
    if (!A || !B || !C) return;
    if (A->cols != B->rows) return;
    if (A->rows != C->rows || B->cols != C->cols) return;

    size_t M = A->rows;
    size_t N = B->cols;
    size_t K = A->cols;

    const float* a_ptr = matrix_ptr((matrix_t*)A);
    const float* b_ptr = matrix_ptr((matrix_t*)B);
    float* c_ptr = matrix_ptr(C);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum = scalar_fma_f64((double)a_ptr[i * A->stride + k], (double)b_ptr[k * B->stride + j], sum);
            }
            c_ptr[i * C->stride + j] = (float)scalar_fma_f64(alpha, sum, scalar_mul_f64(beta, (double)c_ptr[i * C->stride + j]));
        }
    }
}

gemm_result_t gemm_scalar_benchmark(size_t M, size_t N, size_t K) {
    gemm_result_t result = {M, N, K, 1.0f, 0.0f, 0.0f, 0.0f, 0.0, 0.0};

    matrix_t* A = matrix_allocate(M, K, 64);
    matrix_t* B = matrix_allocate(K, N, 64);
    matrix_t* C = matrix_allocate(M, N, 64);

    if (!A || !B || !C) {
        matrix_free(A);
        matrix_free(B);
        matrix_free(C);
        return result;
    }

    for (size_t i = 0; i < M * K; i++) A->data[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < K * N; i++) B->data[i] = (float)(rand() % 100) / 100.0f;
    matrix_zero(C);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    gemm_scalar_f32(A, B, C, 1.0f, 0.0f);

    clock_gettime(CLOCK_MONOTONIC, &end);

    long ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    result.elapsed_ns = (double)ns;
    result.gflops = (2.0 * M * N * K) / (double)ns;

    matrix_free(A);
    matrix_free(B);
    matrix_free(C);

    return result;
}
