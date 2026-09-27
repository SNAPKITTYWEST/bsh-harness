.PHONY: all clean x86_64 h100

TARGET_X86_64 = BSH_Twin_x86_64_v1.bin
CC = gcc
CFLAGS = -O2 -march=native -std=c99 -Wall -nostdlib

all: $(TARGET_X86_64)

$(TARGET_X86_64):
	@echo "Building x86_64 substrate..."
	$(CC) $(CFLAGS) -c ../src/harness.c -o harness.o
	$(CC) $(CFLAGS) -c ../src/gemm_kernel.c -o gemm_kernel.o
	$(CC) $(CFLAGS) -c ../src/softmax_poly.c -o softmax_poly.o
	ld -nostdlib -static -Ttext 0x400000 -e _start -o $(TARGET_X86_64) harness.o gemm_kernel.o softmax_poly.o
	@echo "✓ Built $(TARGET_X86_64)"

clean:
	rm -f *.o *.bin

h100:
	@echo "Building H100 substrate..."
	$(CC) $(CFLAGS) -c ../src/h100_harness.c -o h100_harness.o
	$(CC) $(CFLAGS) -c ../src/h100_tma.c -o h100_tma.o
	$(CC) $(CFLAGS) -c ../src/h100_wgmma.c -o h100_wgmma.o
	ld -nostdlib -static -Ttext 0x400000 -e _start -o BSH_Twin_H100_v1.bin h100_harness.o h100_tma.o h100_wgmma.o
	@echo "✓ Built BSH_Twin_H100_v1.bin"
