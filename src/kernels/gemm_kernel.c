/*
 * Binary Substrate Harness (BSH)
 *
 * Copyright (C) 2026 SNAPKITTYWEST
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 * See LICENSE file for details.
 */

#include <stddef.h>
#include <string.h>

void gemm_micro_kernel(float* C, const float* A, const float* B,
                       size_t M, size_t N, size_t K) {
    if (!C || !A || !B || M == 0 || N == 0 || K == 0) {
        return;
    }

    memset(C, 0, M * N * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
