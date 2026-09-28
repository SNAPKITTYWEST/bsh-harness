#ifndef BSH_GEMM_DISPATCHER_H
#define BSH_GEMM_DISPATCHER_H

#include "../memory/allocator.h"

typedef enum {
    KERNEL_SCALAR,
    KERNEL_BLOCKED,
    KERNEL_AVX2,
    KERNEL_AVX512,
    KERNEL_NEON,
    KERNEL_AUTO
} gemm_kernel_t;

typedef struct {
    gemm_kernel_t kernel;
    const char* name;
} kernel_info_t;

gemm_kernel_t detect_best_kernel(void);
kernel_info_t get_kernel_info(gemm_kernel_t kernel);

void gemm_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta,
    gemm_kernel_t kernel
);

void gemm_auto_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
);

typedef struct {
    size_t M, N, K;
    gemm_kernel_t kernel;
    float max_error;
    float mean_error;
    double elapsed_ns;
    double gflops;
} gemm_perf_t;

gemm_perf_t gemm_benchmark(size_t M, size_t N, size_t K, gemm_kernel_t kernel);
gemm_perf_t gemm_compare_kernels(size_t M, size_t N, size_t K);

#endif
