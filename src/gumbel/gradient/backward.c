#include "backward.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void softmax_gradient_f32(
    float* grad_logits,
    const float* grad_output,
    const float* softmax_output,
    size_t count
) {
    if (!grad_logits || !grad_output || !softmax_output || count == 0) return;

    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += grad_output[i] * softmax_output[i];
    }

    for (size_t i = 0; i < count; i++) {
        grad_logits[i] = softmax_output[i] * (grad_output[i] - sum);
    }
}

void gumbel_straight_through_gradient_f32(
    float* grad_logits,
    const float* grad_hard,
    const float* grad_soft,
    const float* soft_probs,
    size_t count
) {
    if (!grad_logits || !grad_hard || !grad_soft || !soft_probs || count == 0) return;

    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += grad_soft[i] * soft_probs[i];
    }

    for (size_t i = 0; i < count; i++) {
        grad_logits[i] = grad_hard[i] + soft_probs[i] * (grad_soft[i] - sum);
    }
}

void finite_difference_gradient_f32(
    float* numerical_grad,
    float (*func)(const float* x, size_t i, size_t count),
    const float* x,
    size_t count,
    float eps
) {
    if (!numerical_grad || !func || !x || count == 0) return;

    float* x_plus = (float*)malloc(count * sizeof(float));
    float* x_minus = (float*)malloc(count * sizeof(float));

    if (!x_plus || !x_minus) {
        free(x_plus);
        free(x_minus);
        return;
    }

    memcpy(x_plus, x, count * sizeof(float));
    memcpy(x_minus, x, count * sizeof(float));

    for (size_t i = 0; i < count; i++) {
        x_plus[i] = x[i] + eps;
        x_minus[i] = x[i] - eps;

        float f_plus = func(x_plus, i, count);
        float f_minus = func(x_minus, i, count);

        numerical_grad[i] = (f_plus - f_minus) / (2.0f * eps);

        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    free(x_plus);
    free(x_minus);
}

void gradient_check_f32(
    const float* analytical_grad,
    const float* numerical_grad,
    size_t count,
    float tolerance
) {
    if (!analytical_grad || !numerical_grad || count == 0) return;

    printf("Gradient Check (tolerance: %.2e):\n", tolerance);
    printf("Index | Analytical | Numerical | Rel Error | Match\n");
    printf("------|------------|-----------|-----------|------\n");

    int total = 0;
    int passed = 0;

    for (size_t i = 0; i < count && i < 20; i++) {
        float rel_error = 0.0f;
        float denom = fabsf(analytical_grad[i]) + fabsf(numerical_grad[i]);
        if (denom > 0.0f) {
            rel_error = fabsf(analytical_grad[i] - numerical_grad[i]) / denom;
        }

        int match = rel_error < tolerance ? 1 : 0;
        if (match) passed++;
        total++;

        printf("%5zu | %.2e | %.2e | %.2e | %s\n",
               i, analytical_grad[i], numerical_grad[i], rel_error,
               match ? "PASS" : "FAIL");
    }

    printf("Summary: %d/%d gradients passed\n", passed, total);
}

gradient_check_result_t* gradient_check_create(size_t count) {
    if (count == 0) return NULL;

    gradient_check_result_t* result = (gradient_check_result_t*)malloc(sizeof(gradient_check_result_t));
    if (!result) return NULL;

    result->analytical = (float*)malloc(count * sizeof(float));
    result->numerical = (float*)malloc(count * sizeof(float));
    result->count = count;
    result->rel_error = 0.0f;
    result->abs_error = 0.0f;
    result->passed = 0;

    if (!result->analytical || !result->numerical) {
        gradient_check_free(result);
        return NULL;
    }

    memset(result->analytical, 0, count * sizeof(float));
    memset(result->numerical, 0, count * sizeof(float));

    return result;
}

void gradient_check_free(gradient_check_result_t* result) {
    if (result) {
        free(result->analytical);
        free(result->numerical);
        free(result);
    }
}

void compute_jacobian_row_f32(
    float* jacobian_row,
    float (*func)(const float* x, size_t count),
    const float* x,
    size_t count,
    size_t output_idx,
    float eps
) {
    if (!jacobian_row || !func || !x || count == 0) return;

    float* x_perturbed = (float*)malloc(count * sizeof(float));
    if (!x_perturbed) return;

    memcpy(x_perturbed, x, count * sizeof(float));

    float f0 = func(x, count);

    for (size_t i = 0; i < count; i++) {
        x_perturbed[i] = x[i] + eps;
        float f1 = func(x_perturbed, count);
        jacobian_row[i] = (f1 - f0) / eps;
        x_perturbed[i] = x[i];
    }

    free(x_perturbed);
}
