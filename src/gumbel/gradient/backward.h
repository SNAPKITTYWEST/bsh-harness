#ifndef BSH_GUMBEL_BACKWARD_H
#define BSH_GUMBEL_BACKWARD_H

#include <stddef.h>

void softmax_gradient_f32(
    float* grad_logits,
    const float* grad_output,
    const float* softmax_output,
    size_t count
);

void gumbel_straight_through_gradient_f32(
    float* grad_logits,
    const float* grad_hard,
    const float* grad_soft,
    const float* soft_probs,
    size_t count
);

void finite_difference_gradient_f32(
    float* numerical_grad,
    float (*func)(const float* x, size_t i, size_t count),
    const float* x,
    size_t count,
    float eps
);

void gradient_check_f32(
    const float* analytical_grad,
    const float* numerical_grad,
    size_t count,
    float tolerance
);

typedef struct {
    float* analytical;
    float* numerical;
    size_t count;
    float rel_error;
    float abs_error;
    int passed;
} gradient_check_result_t;

gradient_check_result_t* gradient_check_create(size_t count);
void gradient_check_free(gradient_check_result_t* result);

void compute_jacobian_row_f32(
    float* jacobian_row,
    float (*func)(const float* x, size_t count),
    const float* x,
    size_t count,
    size_t output_idx,
    float eps
);

#endif
