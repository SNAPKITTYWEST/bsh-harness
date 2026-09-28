#include "blocking.h"
#include "../scalar/primitives.h"
#include <stdlib.h>
#include <string.h>

block_config_t* blocking_config_create(size_t mc, size_t nc, size_t kc) {
    block_config_t* cfg = (block_config_t*)malloc(sizeof(block_config_t));
    if (cfg) {
        cfg->mc = mc > 0 ? mc : MC;
        cfg->nc = nc > 0 ? nc : NC;
        cfg->kc = kc > 0 ? kc : KC;
    }
    return cfg;
}

void blocking_config_free(block_config_t* cfg) {
    if (cfg) free(cfg);
}

void pack_A_f32(
    float* packed,
    const matrix_t* A,
    size_t k_start, size_t k_end,
    size_t m_start, size_t m_end
) {
    if (!packed || !A) return;

    size_t kc = k_end - k_start;
    size_t mc = m_end - m_start;

    const float* a_ptr = matrix_ptr((matrix_t*)A);

    for (size_t i = 0; i < mc; ++i) {
        for (size_t k = 0; k < kc; ++k) {
            packed[i * kc + k] = a_ptr[(m_start + i) * A->stride + (k_start + k)];
        }
    }
}

void pack_B_f32(
    float* packed,
    const matrix_t* B,
    size_t k_start, size_t k_end,
    size_t n_start, size_t n_end
) {
    if (!packed || !B) return;

    size_t kc = k_end - k_start;
    size_t nc = n_end - n_start;

    const float* b_ptr = matrix_ptr((matrix_t*)B);

    for (size_t k = 0; k < kc; ++k) {
        for (size_t j = 0; j < nc; ++j) {
            packed[k * nc + j] = b_ptr[(k_start + k) * B->stride + (n_start + j)];
        }
    }
}

void microkernel_f32(
    float* C,
    const float* packed_A,
    const float* packed_B,
    size_t mc, size_t nc, size_t kc,
    size_t ldc
) {
    if (!C || !packed_A || !packed_B) return;

    for (size_t i = 0; i < mc; ++i) {
        for (size_t j = 0; j < nc; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < kc; ++k) {
                sum = scalar_fma_f32(packed_A[i * kc + k], packed_B[k * nc + j], sum);
            }
            C[i * ldc + j] += sum;
        }
    }
}

void gemm_blocked_f32(
    const matrix_t* A,
    const matrix_t* B,
    matrix_t* C,
    float alpha,
    float beta,
    const block_config_t* cfg
) {
    if (!A || !B || !C || !cfg) return;
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) return;

    size_t M = A->rows;
    size_t N = B->cols;
    size_t K = A->cols;

    size_t mc = cfg->mc;
    size_t nc = cfg->nc;
    size_t kc = cfg->kc;

    float* packed_A = (float*)malloc(mc * kc * sizeof(float));
    float* packed_B = (float*)malloc(kc * nc * sizeof(float));

    if (!packed_A || !packed_B) {
        free(packed_A);
        free(packed_B);
        return;
    }

    float* c_ptr = matrix_ptr(C);

    for (size_t i_block = 0; i_block < M; i_block += mc) {
        size_t i_end = (i_block + mc < M) ? i_block + mc : M;
        size_t i_size = i_end - i_block;

        for (size_t j_block = 0; j_block < N; j_block += nc) {
            size_t j_end = (j_block + nc < N) ? j_block + nc : N;
            size_t j_size = j_end - j_block;

            for (size_t k_block = 0; k_block < K; k_block += kc) {
                size_t k_end = (k_block + kc < K) ? k_block + kc : K;
                size_t k_size = k_end - k_block;

                pack_A_f32(packed_A, A, k_block, k_end, i_block, i_end);
                pack_B_f32(packed_B, B, k_block, k_end, j_block, j_end);

                microkernel_f32(
                    c_ptr + i_block * C->stride + j_block,
                    packed_A, packed_B,
                    i_size, j_size, k_size,
                    C->stride
                );
            }
        }
    }

    free(packed_A);
    free(packed_B);

    if (alpha != 1.0f || beta != 0.0f) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                c_ptr[i * C->stride + j] = scalar_fma_f32(alpha, c_ptr[i * C->stride + j],
                                                          scalar_mul_f32(beta, c_ptr[i * C->stride + j]));
            }
        }
    }
}
