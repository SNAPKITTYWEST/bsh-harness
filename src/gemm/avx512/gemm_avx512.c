#include "gemm_avx512.h"
#include "../scalar/primitives.h"
#include <immintrin.h>
#include <string.h>
#include <stdlib.h>

#define MC 128
#define NC 192
#define KC 64

#ifdef __AVX512F__

void microkernel_avx512_16x16(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
) {
    __m512 c[16];
    for (int i = 0; i < 16; ++i) {
        c[i] = _mm512_setzero_ps();
    }

    for (size_t k = 0; k < K; ++k) {
        for (int i = 0; i < 16; ++i) {
            __m512 b_vec = _mm512_loadu_ps(&B[k * 16]);
            float a_val = A[i * K + k];
            c[i] = _mm512_fmadd_ps(_mm512_set1_ps(a_val), b_vec, c[i]);
        }
    }

    for (int i = 0; i < 16; ++i) {
        __m512 c_mem = _mm512_loadu_ps(&C[i * ldc]);
        _mm512_storeu_ps(&C[i * ldc], _mm512_add_ps(c[i], c_mem));
    }
}

void microkernel_avx512_4x16(
    float* C,
    const float* A,
    const float* B,
    size_t K,
    size_t ldc
) {
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();

    for (size_t k = 0; k < K; ++k) {
        __m512 b_vec = _mm512_loadu_ps(&B[k * 16]);

        c0 = _mm512_fmadd_ps(_mm512_set1_ps(A[0 * K + k]), b_vec, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(A[1 * K + k]), b_vec, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(A[2 * K + k]), b_vec, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(A[3 * K + k]), b_vec, c3);
    }

    __m512 c0_mem = _mm512_loadu_ps(&C[0 * ldc]);
    __m512 c1_mem = _mm512_loadu_ps(&C[1 * ldc]);
    __m512 c2_mem = _mm512_loadu_ps(&C[2 * ldc]);
    __m512 c3_mem = _mm512_loadu_ps(&C[3 * ldc]);

    _mm512_storeu_ps(&C[0 * ldc], _mm512_add_ps(c0, c0_mem));
    _mm512_storeu_ps(&C[1 * ldc], _mm512_add_ps(c1, c1_mem));
    _mm512_storeu_ps(&C[2 * ldc], _mm512_add_ps(c2, c2_mem));
    _mm512_storeu_ps(&C[3 * ldc], _mm512_add_ps(c3, c3_mem));
}

#endif

void gemm_avx512_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta
) {
    if (!A || !B || !C) return;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return;

    #ifndef __AVX512F__
    return;
    #endif

    size_t M = A->rows;
    size_t N = B->cols;
    size_t K = A->cols;

    float* a_ptr = matrix_ptr((matrix_t*)A);
    float* b_ptr = matrix_ptr((matrix_t*)B);
    float* c_ptr = matrix_ptr(C);

    size_t mc = MC;
    size_t nc = NC;
    size_t kc = KC;

    float* packed_a = (float*)aligned_alloc(64, mc * kc * sizeof(float));
    float* packed_b = (float*)aligned_alloc(64, kc * nc * sizeof(float));

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
