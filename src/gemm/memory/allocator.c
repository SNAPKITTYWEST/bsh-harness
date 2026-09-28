#include "allocator.h"
#include <stdlib.h>
#include <string.h>

aligned_buffer_t* buffer_allocate(size_t size, size_t alignment) {
    if (size == 0 || alignment == 0) return NULL;

    aligned_buffer_t* buf = (aligned_buffer_t*)malloc(sizeof(aligned_buffer_t));
    if (!buf) return NULL;

    size_t total = size + alignment - 1;
    void* raw = malloc(total);
    if (!raw) {
        free(buf);
        return NULL;
    }

    uintptr_t addr = (uintptr_t)raw;
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);

    buf->ptr = (void*)aligned_addr;
    buf->size = size;
    buf->alignment = alignment;
    buf->is_aligned = true;

    return buf;
}

void buffer_free(aligned_buffer_t* buf) {
    if (buf) {
        if (buf->ptr) free(buf->ptr);
        free(buf);
    }
}

void* buffer_ptr(aligned_buffer_t* buf) {
    return buf ? buf->ptr : NULL;
}

bool buffer_is_aligned(aligned_buffer_t* buf, size_t required_alignment) {
    if (!buf) return false;
    return (uintptr_t)buf->ptr % required_alignment == 0;
}

matrix_t* matrix_allocate(size_t rows, size_t cols, size_t alignment) {
    if (rows == 0 || cols == 0 || alignment == 0) return NULL;

    if (alignment < 32) alignment = 32;

    size_t total_elements = rows * cols;
    if (total_elements / rows != cols) return NULL;

    size_t bytes = total_elements * sizeof(float);

    matrix_t* mat = (matrix_t*)malloc(sizeof(matrix_t));
    if (!mat) return NULL;

    size_t total_alloc = bytes + alignment - 1;
    void* raw = malloc(total_alloc);
    if (!raw) {
        free(mat);
        return NULL;
    }

    uintptr_t addr = (uintptr_t)raw;
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);

    mat->data = (float*)aligned_addr;
    mat->rows = rows;
    mat->cols = cols;
    mat->stride = cols;
    mat->alignment = alignment;
    mat->owns_data = true;

    memset(mat->data, 0, bytes);
    return mat;
}

void matrix_free(matrix_t* mat) {
    if (mat) {
        if (mat->owns_data && mat->data) {
            free(mat->data);
        }
        free(mat);
    }
}

float* matrix_ptr(matrix_t* mat) {
    return mat ? mat->data : NULL;
}

float matrix_get(const matrix_t* mat, size_t i, size_t j) {
    if (!mat || i >= mat->rows || j >= mat->cols) return 0.0f;
    return mat->data[i * mat->stride + j];
}

void matrix_set(matrix_t* mat, size_t i, size_t j, float val) {
    if (!mat || i >= mat->rows || j >= mat->cols) return;
    mat->data[i * mat->stride + j] = val;
}

void matrix_zero(matrix_t* mat) {
    if (!mat) return;
    memset(mat->data, 0, mat->rows * mat->stride * sizeof(float));
}

void matrix_fill(matrix_t* mat, float val) {
    if (!mat) return;
    for (size_t i = 0; i < mat->rows * mat->stride; i++) {
        mat->data[i] = val;
    }
}

void matrix_copy(const matrix_t* src, matrix_t* dst) {
    if (!src || !dst || src->rows != dst->rows || src->cols != dst->cols) return;
    memcpy(dst->data, src->data, src->rows * src->stride * sizeof(float));
}
