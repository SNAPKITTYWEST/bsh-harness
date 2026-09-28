#ifndef BSH_GEMM_SIMD_VECTOR_H
#define BSH_GEMM_SIMD_VECTOR_H

#include <stddef.h>
#include <stdint.h>

#if defined(__AVX512F__)
    #define SIMD_WIDTH 16
    #define SIMD_BYTES 64
    #include <immintrin.h>
    typedef __m512 simd_f32;
    typedef __m512d simd_f64;
    typedef __mmask16 simd_mask;
#elif defined(__AVX2__)
    #define SIMD_WIDTH 8
    #define SIMD_BYTES 32
    #include <immintrin.h>
    typedef __m256 simd_f32;
    typedef __m256d simd_f64;
#else
    #define SIMD_WIDTH 1
    #define SIMD_BYTES 4
    typedef float simd_f32;
    typedef double simd_f64;
#endif

typedef struct {
    simd_f32* data;
    size_t capacity;
} simd_vector_f32;

simd_vector_f32 simd_vector_create_f32(size_t capacity);
void simd_vector_free(simd_vector_f32 vec);

#ifdef __AVX512F__

static inline simd_f32 simd_zero_f32(void) {
    return _mm512_setzero_ps();
}

static inline simd_f32 simd_broadcast_f32(float val) {
    return _mm512_set1_ps(val);
}

static inline simd_f32 simd_load_f32(const float* ptr) {
    return _mm512_loadu_ps(ptr);
}

static inline void simd_store_f32(float* ptr, simd_f32 v) {
    _mm512_storeu_ps(ptr, v);
}

static inline simd_f32 simd_add_f32(simd_f32 a, simd_f32 b) {
    return _mm512_add_ps(a, b);
}

static inline simd_f32 simd_sub_f32(simd_f32 a, simd_f32 b) {
    return _mm512_sub_ps(a, b);
}

static inline simd_f32 simd_mul_f32(simd_f32 a, simd_f32 b) {
    return _mm512_mul_ps(a, b);
}

static inline simd_f32 simd_fma_f32(simd_f32 a, simd_f32 b, simd_f32 c) {
    return _mm512_fmadd_ps(a, b, c);
}

static inline simd_f32 simd_reduce_add_f32(simd_f32 v) {
    return _mm512_reduce_add_ps(v);
}

#elif defined(__AVX2__)

static inline simd_f32 simd_zero_f32(void) {
    return _mm256_setzero_ps();
}

static inline simd_f32 simd_broadcast_f32(float val) {
    return _mm256_set1_ps(val);
}

static inline simd_f32 simd_load_f32(const float* ptr) {
    return _mm256_loadu_ps(ptr);
}

static inline void simd_store_f32(float* ptr, simd_f32 v) {
    _mm256_storeu_ps(ptr, v);
}

static inline simd_f32 simd_add_f32(simd_f32 a, simd_f32 b) {
    return _mm256_add_ps(a, b);
}

static inline simd_f32 simd_sub_f32(simd_f32 a, simd_f32 b) {
    return _mm256_sub_ps(a, b);
}

static inline simd_f32 simd_mul_f32(simd_f32 a, simd_f32 b) {
    return _mm256_mul_ps(a, b);
}

static inline simd_f32 simd_fma_f32(simd_f32 a, simd_f32 b, simd_f32 c) {
    return _mm256_fmadd_ps(a, b, c);
}

#else

static inline simd_f32 simd_zero_f32(void) {
    return 0.0f;
}

static inline simd_f32 simd_broadcast_f32(float val) {
    return val;
}

static inline simd_f32 simd_load_f32(const float* ptr) {
    return *ptr;
}

static inline void simd_store_f32(float* ptr, simd_f32 v) {
    *ptr = v;
}

static inline simd_f32 simd_add_f32(simd_f32 a, simd_f32 b) {
    return a + b;
}

static inline simd_f32 simd_sub_f32(simd_f32 a, simd_f32 b) {
    return a - b;
}

static inline simd_f32 simd_mul_f32(simd_f32 a, simd_f32 b) {
    return a * b;
}

static inline simd_f32 simd_fma_f32(simd_f32 a, simd_f32 b, simd_f32 c) {
    return a * b + c;
}

#endif

#endif
