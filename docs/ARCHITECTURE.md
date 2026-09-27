# Binary Substrate Harness — Complete Architecture Specification

## Executive Summary

The Binary Substrate Harness (BSH) is a static, zero-dependency, ahead-of-time compiled computational environment for transformer inference. It eliminates dynamic memory allocation, garbage collection, and runtime introspection. The architecture operates exclusively on pre-allocated, contiguous virtual memory pages. All tensor geometries and execution graphs are resolved during the reduction phase, leaving a purely deterministic, flattened instruction pipeline.

---

## 1. Recursive Dependency Graph & Decomposition

The framework abstraction collapses through the following rigid reduction hierarchy:

```
1. NEURAL NETWORK (layers + learned parameters)
   ↓ (unwrap into sequential affine transforms)
2. MODEL (static DAG of tensor operations)
   ↓ (express as topologically sorted operations)
3. GRAPH (memory-mapped buffer order)
   ↓ (flatten each tensor to 1D contiguous space)
4. TENSOR (multi-dimensional array)
   ↓ (unfold as loop nest over strides)
5. OPERATOR (loops with memory access patterns)
   ↓ (block into register-sized tiles)
6. MATRIX/VECTOR (SIMD register tiles)
   ↓ (IEEE 754 bit pattern)
7. SCALAR (32-bit or 16-bit float)
   ↓ (ALU bitwise logic)
8. INTEGER/BIT (machine operations)
   ↓ (hardware instruction)
9. INSTRUCTION (x86_64/ARM64/SASS opcode)
   ↓ (state of CPU registers + condition codes)
10. MACHINE STATE (architectural state)
```

**Key Property**: At each level, all abstractions are fully reduced. The system cannot be simplified further without breaking the I/O state model.

---

## 2. Neural-Operation Lowering Rules

### 2.1 Softmax Recursion

**Mathematical Definition**:
```
softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
```

**Hardware Lowering**:
1. Base conversion: e^x = 2^(x·log₂(e))
2. Range reduction: 2^(I+F) = 2^I × 2^F
3. 2^I implemented as integer addition to IEEE 754 exponent bias
4. 2^F implemented via 3rd-degree minimax polynomial + FMA instructions

**Critical**: No libm calls, no transcendental lookups. Pure minimax polynomial evaluation = deterministic latency.

### 2.2 Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Lowering**:
- Collapse to GEMM (General Matrix Multiplication)
- GEMM → Macro-kernel (L2 blocking) → Micro-kernel (L1 blocking) → Inner loop
- Inner loop: Unrolled independent outer products via SIMD
- No matrix abstraction at runtime; only register streams

---

## 3. Deterministic State Model

**State Transition**:
```
S_{t+1} = Φ(S_t, I_t)
```

Where:
- S: Memory state (buffer contents)
- I: Verified input payload (with provenance header)
- Φ: Static neural transformation

**Invariant**: Identical (S_0, I_{0...n}) → identical output (bit-for-bit) on compatible microarchitectures.

---

## 4. SYSCALL ABI & Assembly Boundary

**Register Layout** (x86_64 System V):
```
%rax: Syscall number (also return value)
%rdi: Arg 1 (e.g., fd)
%rsi: Arg 2 (e.g., buffer pointer)
%rdx: Arg 3 (e.g., count/size)
%r10: Arg 4 (replaces %rcx)
%r8:  Arg 5
%r9:  Arg 6
```

**Permitted Syscalls**:
1. `sys_read (0x00)`: Ingest input payloads
2. `sys_write (0x01)`: Emit inference results
3. `sys_mmap (0x09)`: Allocate memory regions
4. `sys_exit (0x3C)`: Terminate execution

---

## 5. Primitive Operation Inventory

**18 x86_64 Mnemonics**:

Control Flow:
- `jmp`, `je`, `jne`, `syscall`

Memory Movement:
- `mov`, `vmovaps`, `vmovups`, `vbroadcastss`

Scalar/Integer:
- `add`, `sub`, `imul`, `shl`, `shr`, `and`, `or`

Vector/SIMD:
- `vaddps`, `vsubps`, `vmulps`, `vfmadd231ps`, `vmaxps`

---

## 6. Memory Layout

```
0x0000000000400000 │ ELF Header + .text (14.2 KB)
                   │ _start, sys_mmap bootstrap
                   │ gemm_micro_kernel, softmax_fma_loop
                   ├─────────────────────────────────
0x0000000000500000 │ .rodata (Read-Only)
                   │ Minimax Polynomial Coefficients
                   │ Static Model Weights
                   ├─────────────────────────────────
0x0000000000600000 │ .bss (Read-Write, Uninitialized)
                   │ Ingestion Buffer (4 KB)
                   ├─────────────────────────────────
0x0000000000601000 │ Execution Arena (960 KB)
                   │ K/V Cache, Temp buffers
                   ├─────────────────────────────────
0x00000000006F0000 │ Output Buffer (4 KB)
                   └─────────────────────────────────
```

---

## 7. Fixed-Point Analysis

**99% Template Saturation**: System reached irreducible substrate.

Abstractions removed:
- Python GIL
- PyTorch Autograd engine
- CUDA runtime initialization
- Dynamic shape inference
- cuBLAS heuristic dispatcher
- Threading pools
- Garbage collector

Remaining dependencies: Bare-metal kernel ABI only.

---

## 8. Reproducible Build Procedure

```bash
# 1. Offline tensor compiler (PyTorch → C/Assembly DAG)
python3 lower_model.py model.pt --output harness_dag.s

# 2. Assemble
as -mcpu=native harness_dag.s -o harness.o

# 3. Link (no libc, no dynamic linking)
ld -nostdlib -static harness.o -o BSH_Twin_v1.bin

# 4. Verify determinism
sha256sum BSH_Twin_v1.bin
# Compare against declared manifest
```

---

**Last Updated**: 2026-09-27  
**Version**: 1.0  
**Status**: Specification Complete
