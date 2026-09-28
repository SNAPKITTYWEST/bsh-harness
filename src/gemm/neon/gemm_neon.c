#include "gemm_neon.h"
#include "../scalar/primitives.h"
#include <string.h>
#include <stdlib.h>

#ifdef __ARM_NEON
#include <arm_neon.h>

#define MC 64
#define NC 96
#define KC 32

void microkernel_neon_8x8(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
) {
    float32x4_t c[16] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
    };

    for (size_t k = 0; k < K; ++k) {
        float32x4_t b0 = vld1q_f32(&B[k * 8]);
        float32x4_t b1 = vld1q_f32(&B[k * 8 + 4]);

        for (size_t i = 0; i < 8; ++i) {
            float a_val = A[i * K + k];
            float32x4_t a_broadcast = vdupq_n_f32(a_val);
            c[i * 2] = vmlaq_f32(c[i * 2], a_broadcast, b0);
            c[i * 2 + 1] = vmlaq_f32(c[i * 2 + 1], a_broadcast, b1);
        }
    }

    for (size_t i = 0; i < 8; ++i) {
        float32x4_t c0_mem = vld1q_f32(&C[i * ldc]);
        float32x4_t c1_mem = vld1q_f32(&C[i * ldc + 4]);

        vst1q_f32(&C[i * ldc], vaddq_f32(c[i * 2], c0_mem));
        vst1q_f32(&C[i * ldc + 4], vaddq_f32(c[i * 2 + 1], c1_mem));
    }
}

void microkernel_neon_4x4(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
) {
    float32x4_t c[4] = {
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
        vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)
    };

    for (size_t k = 0; k < K; ++k) {
        float32x4_t b_vec = vld1q_f32(&B[k * 4]);

        for (size_t i = 0; i < 4; ++i) {
            float a_val = A[i * K + k];
            c[i] = vmlaq_f32(c[i], vdupq_n_f32(a_val), b_vec);
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        float32x4_t c_mem = vld1q_f32(&C[i * ldc]);
        vst1q_f32(&C[i * ldc], vaddq_f32(c[i], c_mem));
    }
}

#endif

void gemm_neon_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
) {
    #ifndef __ARM_NEON
    return;
    #endif

    if (!A || !B || !C) return;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return;

    size_t M = A->rows;
    size_t N = B->cols;
    size_t K = A->cols;

    float* a_ptr = matrix_ptr((matrix_t*)A);
    float* b_ptr = matrix_ptr((matrix_t*)B);
    float* c_ptr = matrix_ptr(C);

    size_t mc = MC;
    size_t nc = NC;
    size_t kc = KC;

    float* packed_a = (float*)aligned_alloc(32, mc * kc * sizeof(float));
    float* packed_b = (float*)aligned_alloc(32, kc * nc * sizeof(float));

    if (!packed_a || !packed_b) {
        free(packed_a);
        free(packed_b);
        return;
    }

    for (size_t i_block = 0; i_block < M; i_block += mc) {
        size_t i_end = (i_block + mc < M) ? i_block + mc : M;
        size_t i_size = i_end - i_block;

        for (size_t j_block = 0; j_block < N; j_block += nc) {
            size_t j_end = (j_block + nc < N) ? j_block + nc : N;
            size_t j_size = j_end - j_block;

            for (size_t k_block = 0; k_block < K; k_block += kc) {
                size_t k_end = (k_block + kc < K) ? k_block + kc : K;
                size_t k_size = k_end - k_block;

                for (size_t ii = 0; ii < i_size; ++ii) {
                    for (size_t kk = 0; kk < k_size; ++kk) {
                        packed_a[ii * k_size + kk] = a_ptr[(i_block + ii) * A->stride + k_block + kk];
                    }
                }

                for (size_t kk = 0; kk < k_size; ++kk) {
                    for (size_t jj = 0; jj < j_size; ++jj) {
                        packed_b[kk * j_size + jj] = b_ptr[(k_block + kk) * B->stride + j_block + jj];
                    }
                }

                for (size_t ii = 0; ii < i_size; ++ii) {
                    for (size_t jj = 0; jj < j_size; ++jj) {
                        float sum = 0.0f;
                        for (size_t kk = 0; kk < k_size; ++kk) {
                            sum = scalar_fma_f32(packed_a[ii * k_size + kk],
                                                 packed_b[kk * j_size + jj], sum);
                        }
                        c_ptr[(i_block + ii) * C->stride + j_block + jj] += sum;
                    }
                }
            }
        }
    }

    free(packed_a);
    free(packed_b);
}
