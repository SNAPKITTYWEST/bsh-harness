#include "../dispatch/dispatcher.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

#define TOLERANCE 1e-5f

typedef struct {
    size_t total;
    size_t passed;
    size_t failed;
} test_result_t;

test_result_t result = {0, 0, 0};

void assert_equal(float expected, float actual, const char* test_name) {
    result.total++;
    float error = expected - actual;
    if (error < 0) error = -error;

    if (error < TOLERANCE) {
        result.passed++;
        printf("  PASS: %s\n", test_name);
    } else {
        result.failed++;
        printf("  FAIL: %s (expected %.6f, got %.6f, error %.2e)\n",
               test_name, expected, actual, error);
    }
}

void test_scalar_1x1(void) {
    printf("Test: 1x1 matrices\n");
    matrix_t* A = matrix_allocate(1, 1, 64);
    matrix_t* B = matrix_allocate(1, 1, 64);
    matrix_t* C = matrix_allocate(1, 1, 64);

    matrix_set(A, 0, 0, 2.0f);
    matrix_set(B, 0, 0, 3.0f);
    matrix_zero(C);

    gemm_f32(A, B, C, 1.0f, 0.0f, KERNEL_SCALAR);
    assert_equal(6.0f, matrix_get(C, 0, 0), "1x1 scalar multiply");

    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

void test_scalar_2x2(void) {
    printf("Test: 2x2 matrices\n");
    matrix_t* A = matrix_allocate(2, 2, 64);
    matrix_t* B = matrix_allocate(2, 2, 64);
    matrix_t* C = matrix_allocate(2, 2, 64);

    matrix_set(A, 0, 0, 1.0f);  matrix_set(A, 0, 1, 2.0f);
    matrix_set(A, 1, 0, 3.0f);  matrix_set(A, 1, 1, 4.0f);

    matrix_set(B, 0, 0, 5.0f);  matrix_set(B, 0, 1, 6.0f);
    matrix_set(B, 1, 0, 7.0f);  matrix_set(B, 1, 1, 8.0f);

    matrix_zero(C);
    gemm_f32(A, B, C, 1.0f, 0.0f, KERNEL_SCALAR);

    assert_equal(19.0f, matrix_get(C, 0, 0), "2x2 C[0,0]");
    assert_equal(22.0f, matrix_get(C, 0, 1), "2x2 C[0,1]");
    assert_equal(43.0f, matrix_get(C, 1, 0), "2x2 C[1,0]");
    assert_equal(50.0f, matrix_get(C, 1, 1), "2x2 C[1,1]");

    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

void test_3x3(void) {
    printf("Test: 3x3 matrices\n");
    matrix_t* A = matrix_allocate(3, 3, 64);
    matrix_t* B = matrix_allocate(3, 3, 64);
    matrix_t* C = matrix_allocate(3, 3, 64);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            matrix_set(A, i, j, (float)(i * 3 + j + 1));
            matrix_set(B, i, j, (float)(i * 3 + j + 1));
        }
    }

    matrix_zero(C);
    gemm_f32(A, B, C, 1.0f, 0.0f, KERNEL_SCALAR);

    assert_equal(30.0f, matrix_get(C, 0, 0), "3x3 C[0,0]");
    assert_equal(36.0f, matrix_get(C, 0, 1), "3x3 C[0,1]");
    assert_equal(42.0f, matrix_get(C, 0, 2), "3x3 C[0,2]");

    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

void test_rectangular_97x103x89(void) {
    printf("Test: Rectangular 97x103x89 (odd dimensions)\n");
    matrix_t* A = matrix_allocate(97, 89, 64);
    matrix_t* B = matrix_allocate(89, 103, 64);
    matrix_t* C_scalar = matrix_allocate(97, 103, 64);
    matrix_t* C_blocked = matrix_allocate(97, 103, 64);
    matrix_t* C_avx2 = matrix_allocate(97, 103, 64);

    for (size_t i = 0; i < 97; i++) {
        for (size_t k = 0; k < 89; k++) {
            matrix_set(A, i, k, (float)(rand() % 100) / 100.0f);
        }
    }

    for (size_t k = 0; k < 89; k++) {
        for (size_t j = 0; j < 103; j++) {
            matrix_set(B, k, j, (float)(rand() % 100) / 100.0f);
        }
    }

    matrix_zero(C_scalar);
    matrix_zero(C_blocked);
    matrix_zero(C_avx2);

    gemm_f32(A, B, C_scalar, 1.0f, 0.0f, KERNEL_SCALAR);
    gemm_f32(A, B, C_blocked, 1.0f, 0.0f, KERNEL_BLOCKED);
    gemm_f32(A, B, C_avx2, 1.0f, 0.0f, KERNEL_AVX2);

    float max_error_blocked = 0.0f;
    float max_error_avx2 = 0.0f;

    for (size_t i = 0; i < 97; i++) {
        for (size_t j = 0; j < 103; j++) {
            float diff = matrix_get(C_scalar, i, j) - matrix_get(C_blocked, i, j);
            if (diff < 0) diff = -diff;
            if (diff > max_error_blocked) max_error_blocked = diff;

            diff = matrix_get(C_scalar, i, j) - matrix_get(C_avx2, i, j);
            if (diff < 0) diff = -diff;
            if (diff > max_error_avx2) max_error_avx2 = diff;
        }
    }

    printf("  Blocked vs Scalar max error: %.2e\n", max_error_blocked);
    printf("  AVX2 vs Scalar max error: %.2e\n", max_error_avx2);

    if (max_error_blocked < 1e-4f) {
        result.passed++;
        printf("  PASS: Blocked GEMM 97x103x89\n");
    } else {
        result.failed++;
        printf("  FAIL: Blocked GEMM error too large\n");
    }
    result.total++;

    if (max_error_avx2 < 1e-4f) {
        result.passed++;
        printf("  PASS: AVX2 GEMM 97x103x89\n");
    } else {
        result.failed++;
        printf("  FAIL: AVX2 GEMM error too large\n");
    }
    result.total++;

    matrix_free(A);
    matrix_free(B);
    matrix_free(C_scalar);
    matrix_free(C_blocked);
    matrix_free(C_avx2);
}

void test_beta_scaling(void) {
    printf("Test: Beta scaling (C = alpha*A*B + beta*C)\n");
    matrix_t* A = matrix_allocate(2, 2, 64);
    matrix_t* B = matrix_allocate(2, 2, 64);
    matrix_t* C = matrix_allocate(2, 2, 64);

    matrix_set(A, 0, 0, 1.0f);  matrix_set(A, 0, 1, 1.0f);
    matrix_set(A, 1, 0, 1.0f);  matrix_set(A, 1, 1, 1.0f);

    matrix_set(B, 0, 0, 2.0f);  matrix_set(B, 0, 1, 2.0f);
    matrix_set(B, 1, 0, 2.0f);  matrix_set(B, 1, 1, 2.0f);

    matrix_set(C, 0, 0, 1.0f);  matrix_set(C, 0, 1, 1.0f);
    matrix_set(C, 1, 0, 1.0f);  matrix_set(C, 1, 1, 1.0f);

    gemm_f32(A, B, C, 2.0f, 0.5f, KERNEL_SCALAR);

    assert_equal(8.5f, matrix_get(C, 0, 0), "Beta scaled C[0,0]");

    matrix_free(A);
    matrix_free(B);
    matrix_free(C);
}

void test_large_32x32(void) {
    printf("Test: 32x32 matrices\n");
    matrix_t* A = matrix_allocate(32, 32, 64);
    matrix_t* B = matrix_allocate(32, 32, 64);
    matrix_t* C_ref = matrix_allocate(32, 32, 64);
    matrix_t* C_opt = matrix_allocate(32, 32, 64);

    for (size_t i = 0; i < 32 * 32; i++) {
        A->data[i] = (float)(rand() % 100) / 100.0f;
        B->data[i] = (float)(rand() % 100) / 100.0f;
    }

    matrix_zero(C_ref);
    matrix_zero(C_opt);

    gemm_f32(A, B, C_ref, 1.0f, 0.0f, KERNEL_SCALAR);
    gemm_f32(A, B, C_opt, 1.0f, 0.0f, KERNEL_BLOCKED);

    float max_error = 0.0f;
    for (size_t i = 0; i < 32 * 32; i++) {
        float diff = C_ref->data[i] - C_opt->data[i];
        if (diff < 0) diff = -diff;
        if (diff > max_error) max_error = diff;
    }

    printf("  32x32 max error: %.2e\n", max_error);
    if (max_error < 1e-4f) {
        result.passed++;
        printf("  PASS: 32x32 matrices\n");
    } else {
        result.failed++;
        printf("  FAIL: 32x32 error too large\n");
    }
    result.total++;

    matrix_free(A);
    matrix_free(B);
    matrix_free(C_ref);
    matrix_free(C_opt);
}

int main(void) {
    printf("=== BSH GEMM Test Suite ===\n\n");

    test_scalar_1x1();
    test_scalar_2x2();
    test_3x3();
    test_beta_scaling();
    test_large_32x32();
    test_rectangular_97x103x89();

    printf("\n=== Results ===\n");
    printf("Total: %zu\n", result.total);
    printf("Passed: %zu\n", result.passed);
    printf("Failed: %zu\n\n", result.failed);

    if (result.failed == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("Some tests failed.\n");
        return 1;
    }
}
