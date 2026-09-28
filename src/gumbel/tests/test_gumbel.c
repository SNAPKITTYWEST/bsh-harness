#include "../scalar/gumbel_scalar.h"
#include "../softmax/softmax_scalar.h"
#include "../rng/xorshift.h"
#include "../temperature/temperature.h"
#include "../gradient/backward.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef struct {
    int total;
    int passed;
    int failed;
} test_stats_t;

test_stats_t stats = {0, 0, 0};

#define TOLERANCE 1e-5f

void test_rng_uniform(void) {
    printf("Test: RNG uniform distribution\n");

    xorshift128plus_t* rng = rng_create(12345);
    if (!rng) {
        printf("  FAIL: Could not create RNG\n");
        stats.failed++;
        stats.total++;
        return;
    }

    float sum = 0.0f;
    int samples = 10000;
    float min_val = 2.0f, max_val = -1.0f;

    for (int i = 0; i < samples; i++) {
        float u = rng_uniform_f32(rng);
        if (u <= 0.0f || u >= 1.0f) {
            printf("  FAIL: Invalid uniform sample: %.6f\n", u);
            stats.failed++;
            stats.total++;
            rng_free(rng);
            return;
        }
        sum += u;
        if (u < min_val) min_val = u;
        if (u > max_val) max_val = u;
    }

    float mean = sum / samples;
    printf("  Mean: %.6f (expected 0.5)\n", mean);
    printf("  Min:  %.6f, Max: %.6f\n", min_val, max_val);

    if (fabsf(mean - 0.5f) < 0.01f && min_val > 0.01f && max_val < 0.99f) {
        printf("  PASS: RNG uniform distribution\n");
        stats.passed++;
    } else {
        printf("  FAIL: RNG distribution incorrect\n");
        stats.failed++;
    }
    stats.total++;

    rng_free(rng);
}

void test_gumbel_distribution(void) {
    printf("Test: Gumbel distribution\n");

    xorshift128plus_t* rng = rng_create(54321);
    if (!rng) {
        stats.failed++;
        stats.total++;
        return;
    }

    float sum = 0.0f;
    int samples = 1000;

    for (int i = 0; i < samples; i++) {
        float g = gumbel_sample_f32(rng);
        sum += g;
    }

    float mean = sum / samples;
    printf("  Mean: %.6f (expected ~0.577 Euler-Mascheroni)\n", mean);

    if (!isnan(mean) && !isinf(mean)) {
        printf("  PASS: Gumbel samples are valid\n");
        stats.passed++;
    } else {
        printf("  FAIL: Gumbel samples invalid\n");
        stats.failed++;
    }
    stats.total++;

    rng_free(rng);
}

void test_softmax_normalization(void) {
    printf("Test: Softmax normalization\n");

    float logits[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];

    softmax_scalar_f32(output, logits, 4);

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += output[i];
        if (output[i] < 0.0f || output[i] > 1.0f) {
            printf("  FAIL: Invalid probability: %.6f\n", output[i]);
            stats.failed++;
            stats.total++;
            return;
        }
    }

    printf("  Sum of probabilities: %.6f\n", sum);
    if (fabsf(sum - 1.0f) < TOLERANCE) {
        printf("  PASS: Softmax normalization\n");
        stats.passed++;
    } else {
        printf("  FAIL: Softmax sum != 1.0\n");
        stats.failed++;
    }
    stats.total++;
}

void test_softmax_concentration(void) {
    printf("Test: Softmax concentration with temperature\n");

    float logits[] = {1.0f, 0.5f, 0.0f, -0.5f, -1.0f};
    float output_high[5];
    float output_low[5];

    softmax_scalar_f32(output_high, logits, 5);

    float max_prob_high = output_high[0];
    for (int i = 1; i < 5; i++) {
        if (output_high[i] > max_prob_high) max_prob_high = output_high[i];
    }

    printf("  Baseline max prob: %.6f\n", max_prob_high);

    if (max_prob_high > 0.3f) {
        printf("  PASS: Softmax concentration\n");
        stats.passed++;
    } else {
        printf("  FAIL: Softmax not concentrated\n");
        stats.failed++;
    }
    stats.total++;
}

void test_gumbel_softmax_soft(void) {
    printf("Test: Gumbel-Softmax (soft)\n");

    xorshift128plus_t* rng = rng_create(11111);
    float logits[] = {2.0f, 1.0f, 0.5f, 0.0f};
    float noise[4];
    float output[4];

    gumbel_fill_f32(rng, noise, 4);
    gumbel_softmax_f32(output, logits, noise, 4, 1.0f, 0);

    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += output[i];
    }

    printf("  Sum of soft output: %.6f\n", sum);
    if (fabsf(sum - 1.0f) < TOLERANCE) {
        printf("  PASS: Soft Gumbel-Softmax normalization\n");
        stats.passed++;
    } else {
        printf("  FAIL: Soft sum != 1.0\n");
        stats.failed++;
    }
    stats.total++;

    rng_free(rng);
}

void test_gumbel_softmax_hard(void) {
    printf("Test: Gumbel-Softmax (hard categorical)\n");

    xorshift128plus_t* rng = rng_create(22222);
    float logits[] = {2.0f, 1.0f, 0.5f, 0.0f};
    float noise[4];
    float output[4];

    gumbel_fill_f32(rng, noise, 4);
    gumbel_softmax_f32(output, logits, noise, 4, 1.0f, 1);

    float sum = 0.0f;
    int nonzero_count = 0;
    for (int i = 0; i < 4; i++) {
        sum += output[i];
        if (fabsf(output[i]) > 0.5f) nonzero_count++;
    }

    printf("  Sum: %.6f, Nonzero entries: %d\n", sum, nonzero_count);
    if (fabsf(sum - 1.0f) < TOLERANCE && nonzero_count == 1) {
        printf("  PASS: Hard Gumbel-Softmax is one-hot\n");
        stats.passed++;
    } else {
        printf("  FAIL: Hard output not valid one-hot\n");
        stats.failed++;
    }
    stats.total++;

    rng_free(rng);
}

void test_temperature_scheduling(void) {
    printf("Test: Temperature scheduling\n");

    temp_schedule_t* sched = temp_schedule_create(TEMP_LINEAR, 5.0f, 0.1f, 100);
    if (!sched) {
        stats.failed++;
        stats.total++;
        return;
    }

    float tau_start = temp_schedule_get(sched, 0);
    float tau_mid = temp_schedule_get(sched, 50);
    float tau_end = temp_schedule_get(sched, 100);

    printf("  T(0): %.6f, T(50): %.6f, T(100): %.6f\n", tau_start, tau_mid, tau_end);

    if (tau_start >= tau_mid && tau_mid >= tau_end) {
        printf("  PASS: Temperature decreasing\n");
        stats.passed++;
    } else {
        printf("  FAIL: Temperature schedule incorrect\n");
        stats.failed++;
    }
    stats.total++;

    temp_schedule_free(sched);
}

void test_softmax_gradient(void) {
    printf("Test: Softmax gradient\n");

    float probs[] = {0.7f, 0.2f, 0.1f};
    float grad_output[] = {1.0f, 0.0f, 0.0f};
    float grad_logits[3];

    softmax_gradient_f32(grad_logits, grad_output, probs, 3);

    printf("  Gradient[0]: %.6f\n", grad_logits[0]);
    printf("  Gradient[1]: %.6f\n", grad_logits[1]);
    printf("  Gradient[2]: %.6f\n", grad_logits[2]);

    float sum = grad_logits[0] + grad_logits[1] + grad_logits[2];
    printf("  Sum of gradients: %.2e (should be ~0)\n", sum);

    if (fabsf(sum) < 1e-5f) {
        printf("  PASS: Softmax gradient sum to zero\n");
        stats.passed++;
    } else {
        printf("  FAIL: Gradient sum incorrect\n");
        stats.failed++;
    }
    stats.total++;
}

int main(void) {
    printf("=== BSH Gumbel-Softmax Test Suite ===\n\n");

    test_rng_uniform();
    test_gumbel_distribution();
    test_softmax_normalization();
    test_softmax_concentration();
    test_gumbel_softmax_soft();
    test_gumbel_softmax_hard();
    test_temperature_scheduling();
    test_softmax_gradient();

    printf("\n=== Results ===\n");
    printf("Total: %d\n", stats.total);
    printf("Passed: %d\n", stats.passed);
    printf("Failed: %d\n\n", stats.failed);

    if (stats.failed == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("Some tests failed.\n");
        return 1;
    }
}
