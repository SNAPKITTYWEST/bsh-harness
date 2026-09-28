#include "../gemm_gumbel/pipeline.h"
#include "../../gemm/dispatch/dispatcher.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

void latin_verb_demo(void) {
    printf("=== Latin Verb Feature Processing Pipeline ===\n\n");
    printf("Processing 96-dimensional Latin verb feature vectors\n");
    printf("Mapping to 32 verb conjugation categories\n\n");

    size_t batch_size = 96;
    size_t feature_dim = 96;
    size_t num_classes = 32;

    categorical_pipeline_t* pipe = pipeline_create(batch_size, feature_dim, num_classes, 1.0f, 0);
    xorshift128plus_t* rng = rng_create(2026);

    if (!pipe || !rng) {
        printf("Failed to create pipeline\n");
        return;
    }

    printf("Pipeline Configuration:\n");
    printf("  Batch size: %zu samples\n", batch_size);
    printf("  Feature dimension: %zu\n", feature_dim);
    printf("  Output classes: %zu\n", num_classes);
    printf("  GEMM operation: [%zu × %zu] × [%zu × %zu] -> [%zu × %zu]\n",
           batch_size, feature_dim, feature_dim, num_classes, batch_size, num_classes);
    printf("\n");

    for (size_t i = 0; i < batch_size * feature_dim; i++) {
        pipe->features->data[i] = (float)(rand() % 100) / 100.0f;
    }

    printf("Feature matrix initialized with random [0,1) values\n");

    for (size_t i = 0; i < feature_dim * num_classes; i++) {
        pipe->weights->data[i] = (float)(rand() % 100) / 100.0f;
    }

    printf("Weight matrix initialized with random [0,1) values\n\n");

    printf("=== Kernel Comparison ===\n");
    printf("Dimension | Scalar (ns) | Blocked (ns) | AVX2 (ns) | Speedup vs Scalar\n");
    printf("-----------|-------------|--------------|-----------|-------------------\n");

    gemm_kernel_t kernels[] = {KERNEL_SCALAR, KERNEL_BLOCKED, KERNEL_AVX2};
    const char* kernel_names[] = {"Scalar", "Blocked", "AVX2"};
    double scalar_time = 0.0;

    for (int k = 0; k < 3; k++) {
        struct timespec start, end;

        clock_gettime(CLOCK_MONOTONIC, &start);
        gemm_f32(pipe->features, pipe->weights, pipe->logits, 1.0f, 0.0f, kernels[k]);
        clock_gettime(CLOCK_MONOTONIC, &end);

        long ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);

        if (k == 0) scalar_time = (double)ns;

        printf("%-10s| %11.0f | %12s | %9s | %17.2fx\n",
               kernel_names[k], (double)ns, k > 0 ? "" : "-", k > 0 ? "" : "-",
               k > 0 ? scalar_time / (double)ns : 1.0);
    }
    printf("\n");

    pipeline_forward(pipe, rng, KERNEL_AUTO);

    printf("=== Gumbel-Softmax Output (Sample Row 0) ===\n");
    printf("Soft Output (continuous probabilities):\n  ");
    float sum = 0.0f;
    for (size_t j = 0; j < 32; j++) {
        printf("[%zu]: %.4f  ", j, pipe->soft_output[j]);
        sum += pipe->soft_output[j];
        if ((j + 1) % 4 == 0) printf("\n  ");
    }
    printf("\nSum of probabilities: %.6f\n\n", sum);

    printf("Categorical Output (hard one-hot):\n  ");
    for (size_t j = 0; j < 32; j++) {
        if (pipe->categorical_output[j] > 0.5f) {
            printf("Selected class: %zu\n", j);
            break;
        }
    }
    printf("\n");

    printf("=== Temperature Annealing Test ===\n");
    temp_schedule_t* sched = temp_schedule_create(TEMP_LINEAR, 5.0f, 0.1f, 100);

    printf("Epoch | Temperature | Output Concentration\n");
    printf("------|-------------|---------------------\n");

    for (int epoch = 0; epoch <= 100; epoch += 20) {
        float tau = temp_schedule_get(sched, epoch);
        pipe->temperature = tau;

        matrix_zero(pipe->logits);
        gemm_f32(pipe->features, pipe->weights, pipe->logits, 1.0f, 0.0f, KERNEL_AUTO);

        gumbel_fill_f32(rng, pipe->gumbel_samples, batch_size * num_classes);

        gumbel_softmax_f32(
            pipe->soft_output,
            pipe->logits->data,
            pipe->gumbel_samples,
            num_classes,
            tau,
            0
        );

        float max_prob = 0.0f;
        for (size_t j = 0; j < num_classes; j++) {
            if (pipe->soft_output[j] > max_prob) max_prob = pipe->soft_output[j];
        }

        printf("%5d | %.6f | %.4f\n", epoch, tau, max_prob);
    }
    printf("\n");

    printf("=== Performance Benchmark ===\n");
    pipeline_perf_t perf = pipeline_benchmark(96, 96, 32, 100);
    printf("Average time per forward pass: %.2f µs\n", perf.total_time_ns / 1000.0);
    printf("Throughput: %.0f samples/sec\n", perf.throughput_samples_per_sec);
    printf("\n");

    printf("=== Validation ===\n");

    int valid_count = 0;
    for (size_t b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (size_t j = 0; j < num_classes; j++) {
            sum += pipe->soft_output[b * num_classes + j];
        }
        if (fabsf(sum - 1.0f) < 1e-5f) valid_count++;
    }

    printf("Normalized outputs: %d/%zu (%.1f%%)\n",
           valid_count, batch_size, 100.0 * valid_count / batch_size);

    pipeline_free(pipe);
    temp_schedule_free(sched);
    rng_free(rng);

    printf("\n=== Pipeline Complete ===\n");
}

int main(void) {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   BSH: Hand-Rolled GEMM × Gumbel-Softmax Integration Demo     ║\n");
    printf("║                  Latin Verb Feature Pipeline                   ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    latin_verb_demo();

    return 0;
}
