#ifndef BSH_INTEGRATION_PIPELINE_H
#define BSH_INTEGRATION_PIPELINE_H

#include "../../gemm/memory/allocator.h"
#include "../../gumbel/scalar/gumbel_scalar.h"
#include "../../gumbel/rng/xorshift.h"
#include "../../gumbel/temperature/temperature.h"

typedef struct {
    matrix_t* features;
    matrix_t* weights;
    matrix_t* logits;
    float* gumbel_samples;
    float* categorical_output;
    float* soft_output;

    size_t batch_size;
    size_t feature_dim;
    size_t num_classes;

    float temperature;
    int hard;
} categorical_pipeline_t;

categorical_pipeline_t* pipeline_create(
    size_t batch_size,
    size_t feature_dim,
    size_t num_classes,
    float temperature,
    int hard
);

void pipeline_free(categorical_pipeline_t* pipe);

void pipeline_forward(
    categorical_pipeline_t* pipe,
    xorshift128plus_t* rng,
    int kernel_type
);

void pipeline_set_features(
    categorical_pipeline_t* pipe,
    const float* features
);

void pipeline_set_weights(
    categorical_pipeline_t* pipe,
    const float* weights
);

float* pipeline_get_output(categorical_pipeline_t* pipe);
float* pipeline_get_soft_output(categorical_pipeline_t* pipe);

typedef struct {
    size_t batch_size;
    size_t feature_dim;
    size_t num_classes;
    double gemm_time_ns;
    double gumbel_time_ns;
    double total_time_ns;
    double throughput_samples_per_sec;
} pipeline_perf_t;

pipeline_perf_t pipeline_benchmark(
    size_t batch_size,
    size_t feature_dim,
    size_t num_classes,
    int iterations
);

#endif
