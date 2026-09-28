#ifndef BSH_GEMM_ALLOCATOR_H
#define BSH_GEMM_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define CACHE_LINE_SIZE 64
#define SIMD_ALIGNMENT 64
#define DEFAULT_ALIGNMENT 32

typedef struct {
    void* ptr;
    size_t size;
    size_t alignment;
    bool is_aligned;
} aligned_buffer_t;

aligned_buffer_t* buffer_allocate(size_t size, size_t alignment);
void buffer_free(aligned_buffer_t* buf);
void* buffer_ptr(aligned_buffer_t* buf);
bool buffer_is_aligned(aligned_buffer_t* buf, size_t required_alignment);

typedef struct {
    float* data;
    size_t rows;
    size_t cols;
    size_t stride;
    size_t alignment;
    bool owns_data;
} matrix_t;

matrix_t* matrix_allocate(size_t rows, size_t cols, size_t alignment);
void matrix_free(matrix_t* mat);
float* matrix_ptr(matrix_t* mat);
float matrix_get(const matrix_t* mat, size_t i, size_t j);
void matrix_set(matrix_t* mat, size_t i, size_t j, float val);
void matrix_zero(matrix_t* mat);
void matrix_fill(matrix_t* mat, float val);
void matrix_copy(const matrix_t* src, matrix_t* dst);

#endif
