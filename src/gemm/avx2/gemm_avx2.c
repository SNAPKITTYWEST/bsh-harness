#include "gemm_avx2.h"
#include "../scalar/primitives.h"
#include <immintrin.h>
#include <string.h>
#include <stdlib.h>

#define MC 64
#define NC 96
#define KC 32

void microkernel_avx2_8x8(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
) {
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    for (size_t k = 0; k < K; ++k) {
        __m256 b_vec = _mm256_loadu_ps(&B[k * 8]);

        float a0 = A[0 * K + k];
        float a1 = A[1 * K + k];
        float a2 = A[2 * K + k];
        float a3 = A[3 * K + k];
        float a4 = A[4 * K + k];
        float a5 = A[5 * K + k];
        float a6 = A[6 * K + k];
        float a7 = A[7 * K + k];

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(a0), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(a1), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(a2), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(a3), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(a4), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(a5), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(a6), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(a7), b_vec, c7);
    }

    __m256 c0_mem = _mm256_loadu_ps(&C[0 * ldc]);
    __m256 c1_mem = _mm256_loadu_ps(&C[1 * ldc]);
    __m256 c2_mem = _mm256_loadu_ps(&C[2 * ldc]);
    __m256 c3_mem = _mm256_loadu_ps(&C[3 * ldc]);
    __m256 c4_mem = _mm256_loadu_ps(&C[4 * ldc]);
    __m256 c5_mem = _mm256_loadu_ps(&C[5 * ldc]);
    __m256 c6_mem = _mm256_loadu_ps(&C[6 * ldc]);
    __m256 c7_mem = _mm256_loadu_ps(&C[7 * ldc]);

    _mm256_storeu_ps(&C[0 * ldc], _mm256_add_ps(c0, c0_mem));
    _mm256_storeu_ps(&C[1 * ldc], _mm256_add_ps(c1, c1_mem));
    _mm256_storeu_ps(&C[2 * ldc], _mm256_add_ps(c2, c2_mem));
    _mm256_storeu_ps(&C[3 * ldc], _mm256_add_ps(c3, c3_mem));
    _mm256_storeu_ps(&C[4 * ldc], _mm256_add_ps(c4, c4_mem));
    _mm256_storeu_ps(&C[5 * ldc], _mm256_add_ps(c5, c5_mem));
    _mm256_storeu_ps(&C[6 * ldc], _mm256_add_ps(c6, c6_mem));
    _mm256_storeu_ps(&C[7 * ldc], _mm256_add_ps(c7, c7_mem));
}

void gemm_avx2_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
) {
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

                if (i_size == 8 && j_size == 8) {
                    microkernel_avx2_8x8(
                        c_ptr + i_block * C->stride + j_block,
                        packed_a, packed_b, k_size,
                        C->stride
                    );
                } else {
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
    }

    free(packed_a);
    free(packed_b);
}
