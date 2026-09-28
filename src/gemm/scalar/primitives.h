#ifndef BSH_GEMM_SCALAR_PRIMITIVES_H
#define BSH_GEMM_SCALAR_PRIMITIVES_H

#include <math.h>

typedef float scalar_f32;
typedef double scalar_f64;

static inline scalar_f32 scalar_add_f32(scalar_f32 a, scalar_f32 b) {
    return a + b;
}

static inline scalar_f32 scalar_sub_f32(scalar_f32 a, scalar_f32 b) {
    return a - b;
}

static inline scalar_f32 scalar_mul_f32(scalar_f32 a, scalar_f32 b) {
    return a * b;
}

static inline scalar_f32 scalar_fma_f32(scalar_f32 a, scalar_f32 b, scalar_f32 c) {
    return a * b + c;
}

static inline scalar_f32 scalar_min_f32(scalar_f32 a, scalar_f32 b) {
    return a < b ? a : b;
}

static inline scalar_f32 scalar_max_f32(scalar_f32 a, scalar_f32 b) {
    return a > b ? a : b;
}

static inline scalar_f32 scalar_abs_f32(scalar_f32 a) {
    return a < 0.0f ? -a : a;
}

static inline scalar_f64 scalar_add_f64(scalar_f64 a, scalar_f64 b) {
    return a + b;
}

static inline scalar_f64 scalar_sub_f64(scalar_f64 a, scalar_f64 b) {
    return a - b;
}

static inline scalar_f64 scalar_mul_f64(scalar_f64 a, scalar_f64 b) {
    return a * b;
}

static inline scalar_f64 scalar_fma_f64(scalar_f64 a, scalar_f64 b, scalar_f64 c) {
    return a * b + c;
}

static inline scalar_f64 scalar_min_f64(scalar_f64 a, scalar_f64 b) {
    return a < b ? a : b;
}

static inline scalar_f64 scalar_max_f64(scalar_f64 a, scalar_f64 b) {
    return a > b ? a : b;
}

static inline scalar_f64 scalar_abs_f64(scalar_f64 a) {
    return a < 0.0 ? -a : a;
}

#endif
