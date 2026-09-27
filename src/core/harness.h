/*
 * Binary Substrate Harness (BSH) - Core Header
 * 
 * Copyright (C) 2026 SNAPKITTYWEST
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 * See LICENSE file for details.
 *
 * Zero-dependency transformer inference on bare metal
 */

#ifndef _BSH_HARNESS_H
#define _BSH_HARNESS_H

#include <stdint.h>
#include <stddef.h>

/* ============================================================================
   SYSCALL DEFINITIONS (System V AMD64 ABI)
   ============================================================================ */

#define SYS_READ   0x00
#define SYS_WRITE  0x01
#define SYS_MMAP   0x09
#define SYS_EXIT   0x3C

#define PROT_READ  0x1
#define PROT_WRITE 0x2

#define MAP_SHARED     0x01
#define MAP_ANONYMOUS  0x20
#define MAP_PRIVATE    0x02

/* ============================================================================
   NEURAL OPERATION PRIMITIVES
   ============================================================================ */

void softmax_fma(float* restrict output, const float* restrict input, size_t count);
void gemm_micro_kernel(float* C, const float* A, const float* B, 
                       size_t M, size_t N, size_t K);
void attention_sdpa(float* restrict output, const float* restrict Q,
                    const float* restrict K, const float* restrict V,
                    size_t seq_len, size_t d_k);

/* ============================================================================
   MEMORY MANAGEMENT
   ============================================================================ */

#define INGESTION_BUFFER_SIZE   (4 * 1024)
#define EXECUTION_ARENA_SIZE    (960 * 1024)
#define OUTPUT_BUFFER_SIZE      (4 * 1024)
#define TOTAL_ARENA_SIZE        (1024 * 1024)

extern uint8_t ingestion_buffer[INGESTION_BUFFER_SIZE];
extern uint8_t execution_arena[EXECUTION_ARENA_SIZE];
extern uint8_t output_buffer[OUTPUT_BUFFER_SIZE];

/* ============================================================================
   DIGITAL TWIN STATE MODEL
   ============================================================================ */

typedef struct {
    uint64_t sequence_number;
    uint64_t timestamp_utc_sec;
    uint8_t source_identifier[32];
    uint8_t previous_state_hash[32];
    uint8_t current_state_hash[32];
    uint64_t memory_arena_offset;
    uint64_t payload_bytes;
    uint8_t payload_data[4096];
} digital_twin_state_t;

/* ============================================================================
   CRYSTAL TENSOR FORMAT
   ============================================================================ */

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t version;
    uint16_t dtype;
    uint32_t rank;
    uint32_t flags;
    uint64_t shape[4];
    uint64_t payload_bytes;
    uint64_t alignment;
    uint32_t reserved[2];
} xtensor_header_t;

typedef struct {
    const xtensor_header_t* header;
    const void* data;
    size_t total_size;
} crystal_tensor_t;

crystal_tensor_t load_crystal_tensor(const char* filepath);
int verify_crystal_integrity(const crystal_tensor_t* t, const uint8_t expected_hash[32]);

/* ============================================================================
   H100-SPECIFIC PRIMITIVES
   ============================================================================ */

#ifdef TARGET_H100

typedef struct {
    uint64_t global_address;
    uint32_t shape[4];
    uint32_t strides[4];
} tma_descriptor_t;

typedef struct {
    uint32_t m_size;
    uint32_t n_size;
    uint32_t k_size;
    uint8_t dtype_a;
    uint8_t dtype_b;
} wgmma_config_t;

void h100_tma_transfer(uint8_t* smem_dest, const tma_descriptor_t* desc,
                       uint32_t m_idx, uint32_t n_idx);
void h100_wgmma_mma(float* output, const tma_descriptor_t* desc_a,
                    const tma_descriptor_t* desc_b, const wgmma_config_t* cfg);

#endif

/* ============================================================================
   UTILITY FUNCTIONS
   ============================================================================ */

void sha256_compute(const uint8_t* data, size_t len, uint8_t hash[32]);
extern const float exp_poly_coeffs[6];
float fast_reciprocal(float x);

#endif
