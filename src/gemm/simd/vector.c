#include "vector.h"
#include <stdlib.h>
#include <string.h>

simd_vector_f32 simd_vector_create_f32(size_t capacity) {
    simd_vector_f32 vec;
    vec.capacity = capacity;
    vec.data = (simd_f32*)aligned_alloc(SIMD_BYTES, capacity * sizeof(simd_f32));
    return vec;
}

void simd_vector_free(simd_vector_f32 vec) {
    if (vec.data) free(vec.data);
}
