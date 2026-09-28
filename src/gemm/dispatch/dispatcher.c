#include "dispatcher.h"
#include "../scalar/gemm_scalar.h"
#include "../blocking/blocking.h"
#include "../avx2/gemm_avx2.h"
#include "../avx512/gemm_avx512.h"
#include "../neon/gemm_neon.h"
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

gemm_kernel_t detect_best_kernel(void) {
    #ifdef __AVX512F__
    return KERNEL_AVX512;
    #elif defined(__AVX2__)
    return KERNEL_AVX2;
    #elif defined(__ARM_NEON)
    return KERNEL_NEON;
    #else
    return KERNEL_SCALAR;
    #endif
}

kernel_info_t get_kernel_info(gemm_kernel_t kernel) {
    kernel_info_t info;
    switch (kernel) {
        case KERNEL_SCALAR:
            info.kernel = KERNEL_SCALAR;
            info.name = "Scalar Reference";
            break;
        case KERNEL_BLOCKED:
            info.kernel = KERNEL_BLOCKED;
            info.name = "Cache-Blocked Scalar";
            break;
        case KERNEL_AVX2:
            info.kernel = KERNEL_AVX2;
            info.name = "AVX2 (8-wide)";
            break;
        case KERNEL_AVX512:
            info.kernel = KERNEL_AVX512;
            info.name = "AVX-512 (16-wide)";
            break;
        case KERNEL_NEON:
            info.kernel = KERNEL_NEON;
            info.name = "ARM NEON (4-wide)";
            break;
        default:
            info.kernel = KERNEL_SCALAR;
            info.name = "Unknown";
    }
    return info;
}

void gemm_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta,
    gemm_kernel_t kernel
) {
    if (!A || !B || !C) return;

    switch (kernel) {
        case KERNEL_SCALAR:
            gemm_scalar_f32(A, B, C, alpha, beta);
            break;
        case KERNEL_BLOCKED: {
            block_config_t* cfg = blocking_config_create(64, 128, 64);
            gemm_blocked_f32(A, B, C, alpha, beta, cfg);
            blocking_config_free(cfg);
            break;
        }
        case KERNEL_AVX2:
            #ifdef __AVX2__
            gemm_avx2_f32(A, B, C, alpha, beta);
            #else
            gemm_scalar_f32(A, B, C, alpha, beta);
            #endif
            break;
        case KERNEL_AVX512:
            #ifdef __AVX512F__
            gemm_avx512_f32(A, B, C, alpha, beta);
            #else
            gemm_avx2_f32(A, B, C, alpha, beta);
            #endif
            break;
        case KERNEL_NEON:
            #ifdef __ARM_NEON
            gemm_neon_f32(A, B, C, alpha, beta);
            #else
            gemm_scalar_f32(A, B, C, alpha, beta);
            #endif
            break;
        case KERNEL_AUTO:
            gemm_auto_f32(A, B, C, alpha, beta);
            break;
    }
}

void gemm_auto_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
) {
    gemm_kernel_t best = detect_best_kernel();
    gemm_f32(A, B, C, alpha, beta, best);
}

static float compare_matrices(const matrix_t* expected, const matrix_t* actual) {
    if (!expected || !actual) return 0.0f;
    if (expected->rows != actual->rows || expected->cols != actual->cols) return 1e6f;

    float max_error = 0.0f;
    const float* exp_ptr = matrix_ptr((matrix_t*)expected);
    const float* act_ptr = matrix_ptr((matrix_t*)actual);

    for (size_t i = 0; i < expected->rows * expected->cols; ++i) {
        float diff = exp_ptr[i] - act_ptr[i];
        if (diff < 0) diff = -diff;
        if (diff > max_error) max_error = diff;
    }

    return max_error;
}

gemm_perf_t gemm_benchmark(size_t M, size_t N, size_t K, gemm_kernel_t kernel) {
    gemm_perf_t result = {M, N, K, kernel, 0.0f, 0.0f, 0.0, 0.0};

    matrix_t* A = matrix_allocate(M, K, 64);
    matrix_t* B = matrix_allocate(K, N, 64);
    matrix_t* C = matrix_allocate(M, N, 64);
    matrix_t* C_ref = matrix_allocate(M, N, 64);

    if (!A || !B || !C || !C_ref) goto cleanup;

    for (size_t i = 0; i < M * K; i++) A->data[i] = (float)(rand() % 100) / 100.0f;
    for (size_t i = 0; i < K * N; i++) B->data[i] = (float)(rand() % 100) / 100.0f;

    gemm_scalar_f32(A, B, C_ref, 1.0f, 0.0f);
    matrix_zero(C);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    gemm_f32(A, B, C, 1.0f, 0.0f, kernel);
    clock_gettime(CLOCK_MONOTONIC, &end);

    long ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    result.elapsed_ns = (double)ns;
    result.gflops = (2.0 * M * N * K) / (double)ns;
    result.max_error = compare_matrices(C_ref, C);

cleanup:
    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
    matrix_free(C_ref);

    return result;
}

gemm_perf_t gemm_compare_kernels(size_t M, size_t N, size_t K) {
    gemm_perf_t best = {M, N, K, KERNEL_SCALAR, 1e6f, 0.0, 0.0, 0.0};

    gemm_kernel_t kernels[] = {KERNEL_SCALAR, KERNEL_BLOCKED, KERNEL_AVX2, KERNEL_AVX512, KERNEL_NEON};

    printf("GEMM Kernel Comparison (%zu x %zu x %zu):\n", M, N, K);
    printf("%-25s | Time (ns) | GFLOP/s | Max Error\n", "Kernel");
    printf("------------------------+-----------|-----------|----------\n");

    for (int i = 0; i < 5; ++i) {
        gemm_perf_t perf = gemm_benchmark(M, N, K, kernels[i]);
        kernel_info_t info = get_kernel_info(kernels[i]);

        printf("%-25s | %9.0f | %7.2f | %.2e\n",
               info.name, perf.elapsed_ns, perf.gflops, perf.max_error);

        if (perf.gflops > best.gflops && perf.max_error < 1e-4f) {
            best = perf;
        }
    }

    return best;
}
