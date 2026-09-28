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
#include <math.h>

void softmax_fma(float* restrict output, const float* restrict input, size_t count) {
    if (!output || !input || count == 0) {
        return;
    }

    float max_val = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < count; ++i) {
        output[i] *= inv_sum;
    }
}
