# Execution Flowcharts & Diagrams

## Inference Pipeline

INPUT → Read → Parse → Compute → Write → Output

## Memory Layout

0x400000: .text (14.2 KB)
0x500000: .rodata (weights)
0x600000: Input buffer
0x610000: Execution arena
0x6F0000: Output buffer

## Softmax

Input → Find max → Range reduce → Polynomial → Exponent → Sum → Normalize → Output

## State Machine

S_t → Verify Input → Execute Φ → Hash → Emit → S_{t+1}

## Digital Twin

Ingestion → Parse header → Load weights → Compute → Output → Verify

See ARCHITECTURE.md and HARDWARE_SUBSTRATE.md for detailed specifications.
