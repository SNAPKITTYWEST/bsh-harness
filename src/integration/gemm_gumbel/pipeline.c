#include "pipeline.h"
#include "../../gemm/dispatch/dispatcher.h"
#include "../../gumbel/softmax/softmax_scalar.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

categorical_pipeline_t* pipeline_create(
    size_t batch_size,
    size_t feature_dim,
    size_t num_classes,
    float temperature,
    int hard
) {
    if (batch_size == 0 || feature_dim == 0 || num_classes == 0) return NULL;

    categorical_pipeline_t* pipe = (categorical_pipeline_t*)malloc(sizeof(categorical_pipeline_t));
    if (!pipe) return NULL;

    pipe->features = matrix_allocate(batch_size, feature_dim, 64);
    pipe->weights = matrix_allocate(feature_dim, num_classes, 64);
    pipe->logits = matrix_allocate(batch_size, num_classes, 64);
    pipe->gumbel_samples = (float*)aligned_alloc(64, batch_size * num_classes * sizeof(float));
    pipe->categorical_output = (float*)aligned_alloc(64, batch_size * num_classes * sizeof(float));
    pipe->soft_output = (float*)aligned_alloc(64, batch_size * num_classes * sizeof(float));

    if (!pipe->features || !pipe->weights || !pipe->logits ||
        !pipe->gumbel_samples || !pipe->categorical_output || !pipe->soft_output) {
        pipeline_free(pipe);
        return NULL;
    }

    pipe->batch_size = batch_size;
    pipe->feature_dim = feature_dim;
    pipe->num_classes = num_classes;
    pipe->temperature = temperature > 0.0f ? temperature : 1.0f;
    pipe->hard = hard;

    return pipe;
}

void pipeline_free(categorical_pipeline_t* pipe) {
    if (pipe) {
        matrix_free(pipe->features);
        matrix_free(pipe->weights);
        matrix_free(pipe->logits);
        free(pipe->gumbel_samples);
        free(pipe->categorical_output);
        free(pipe->soft_output);
        free(pipe);
    }
}

void pipeline_set_features(
    categorical_pipeline_t* pipe,
    const float* features
) {
    if (pipe && features) {
        memcpy(pipe->features->data, features,
               pipe->batch_size * pipe->feature_dim * sizeof(float));
    }
}

void pipeline_set_weights(
    categorical_pipeline_t* pipe,
    const float* weights
) {
    if (pipe && weights) {
        memcpy(pipe->weights->data, weights,
               pipe->feature_dim * pipe->num_classes * sizeof(float));
    }
}

void pipeline_forward(
    categorical_pipeline_t* pipe,
    xorshift128plus_t* rng,
    int kernel_type
) {
    if (!pipe || !rng) return;

    matrix_zero(pipe->logits);
    gemm_f32(pipe->features, pipe->weights, pipe->logits, 1.0f, 0.0f, (gemm_kernel_t)kernel_type);

    gumbel_fill_f32(rng, pipe->gumbel_samples, pipe->batch_size * pipe->num_classes);

    for (size_t b = 0; b < pipe->batch_size; b++) {
        gumbel_softmax_f32(
            pipe->soft_output + b * pipe->num_classes,
            pipe->logits->data + b * pipe->num_classes,
            pipe->gumbel_samples + b * pipe->num_classes,
            pipe->num_classes,
            pipe->temperature,
            0
        );

        if (pipe->hard) {
            gumbel_softmax_f32(
                pipe->categorical_output + b * pipe->num_classes,
                pipe->logits->data + b * pipe->num_classes,
                pipe->gumbel_samples + b * pipe->num_classes,
                pipe->num_classes,
                pipe->temperature,
                1
            );
        } else {
            memcpy(pipe->categorical_output + b * pipe->num_classes,
                   pipe->soft_output + b * pipe->num_classes,
                   pipe->num_classes * sizeof(float));
        }
    }
}

float* pipeline_get_output(categorical_pipeline_t* pipe) {
    return pipe ? pipe->categorical_output : NULL;
}

float* pipeline_get_soft_output(categorical_pipeline_t* pipe) {
    return pipe ? pipe->soft_output : NULL;
}

pipeline_perf_t pipeline_benchmark(
    size_t batch_size,
    size_t feature_dim,
    size_t num_classes,
    int iterations
) {
    pipeline_perf_t perf = {batch_size, feature_dim, num_classes, 0.0, 0.0, 0.0, 0.0};

    categorical_pipeline_t* pipe = pipeline_create(batch_size, feature_dim, num_classes, 1.0f, 0);
    xorshift128plus_t* rng = rng_create(42);

    if (!pipe || !rng) {
        pipeline_free(pipe);
        if (rng) rng_free(rng);
        return perf;
    }

    for (size_t i = 0; i < batch_size * feature_dim; i++) {
        pipe->features->data[i] = (float)(rand() % 100) / 100.0f;
    }
    for (size_t i = 0; i < feature_dim * num_classes; i++) {
        pipe->weights->data[i] = (float)(rand() % 100) / 100.0f;
    }

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < iterations; iter++) {
        pipeline_forward(pipe, rng, KERNEL_AUTO);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    long ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    perf.total_time_ns = (double)ns / iterations;
    perf.throughput_samples_per_sec = (batch_size * 1e9) / perf.total_time_ns;

    pipeline_free(pipe);
    rng_free(rng);

    return perf;
}
