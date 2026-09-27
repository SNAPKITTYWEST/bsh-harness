# Binary Substrate Harness (BSH) - Complete Build System

## Overview

This directory contains a production-ready, deterministic build system for the BSH C++17 project.

- **CMakeLists.txt** - Modern CMake configuration for C++17 compilation
- **Makefile** - User-friendly make targets that wrap CMake
- **SHA-256 verification** - Binary hash computation and reproducibility checking
- **Cross-platform support** - Works on Linux, macOS, and Windows

## Build Files Location

```
/tmp/bsh-harness/
├── CMakeLists.txt              ← Root CMake configuration
├── build/
│   └── Makefile                ← Build system wrapper
├── src/
│   ├── bsh_tensor.hpp
│   ├── bsh_tensor.cpp
│   ├── bsh_engine.hpp
│   ├── bsh_engine.cpp
│   ├── main.cpp
│   └── test_bsh.cpp
└── scripts/
    ├── compute_hash.sh
    └── verify_build.cmake
```

## Build Configuration

### Compiler Flags

- `-std=c++17` - C++17 standard
- `-O2` - Optimization level 2
- `-Wall -Wextra` - Enable warnings

### Build Targets

- BSH_Engine.bin - Main inference engine executable
- BSH_Test.bin - Unit test suite executable

## Usage

### Basic Build

```bash
cd /tmp/bsh-harness/build
make                    # Build BSH_Engine.bin
make info              # Display build configuration
```

### Run Tests

```bash
cd /tmp/bsh-harness/build
make test              # Build and run unit tests
```

### Clean and Verify

```bash
cd /tmp/bsh-harness/build
make clean             # Remove build artifacts
make verify            # Verify reproducible builds
make hash              # Display binary SHA-256 hash
```

## Compilation Details

### Source Files

- bsh_tensor.cpp - Tensor implementation
- bsh_engine.cpp - Engine implementation
- main.cpp - Main executable entry point
- test_bsh.cpp - Unit test suite

### Executables

- BSH_Engine.bin - Main executable
- BSH_Test.bin - Test executable

## Reproducible Builds

The build system computes SHA-256 hashes and verifies deterministic compilation:

```bash
make hash    # Display SHA-256 of BSH_Engine.bin
make verify  # Verify reproducibility across clean rebuilds
```

Platform-specific hash tools are automatically selected:
- Linux: sha256sum or shasum
- macOS: shasum -a 256
- Windows: certutil

## Test Suite

The project includes 8 unit tests:

1. Tensor creation
2. Tensor fill
3. Tensor element access
4. Tensor transpose
5. Tensor softmax
6. Engine initialization
7. Engine processing
8. Engine version

Run with: `make test`

## Troubleshooting

### CMake not found

Install CMake:
```bash
sudo apt-get install cmake         # Debian/Ubuntu
brew install cmake                 # macOS
```

### C++ compiler not found

Install g++:
```bash
sudo apt-get install g++           # Debian/Ubuntu
xcode-select --install             # macOS
```

## Build Files

All files are located at:
- `/tmp/bsh-harness/CMakeLists.txt` - Root CMake configuration
- `/tmp/bsh-harness/build/Makefile` - Build wrapper
- `/tmp/bsh-harness/src/` - Source files
- `/tmp/bsh-harness/scripts/` - Verification scripts

---

**C++ Standard**: C++17
**Build System**: CMake + Make
**CMake Version**: 3.16+
