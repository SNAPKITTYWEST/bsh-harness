# Binary Substrate Harness (BSH) - Complete Build System
# Deterministic C++17 compilation with SHA-256 verification
# Wraps CMake for portable builds across Unix/Windows

.PHONY: all test clean verify help hash

# CMake build type
CMAKE_BUILD_TYPE := Release
CMAKE_GENERATOR := Unix Makefiles

# Check for Ninja availability (faster)
ifeq ($(shell command -v ninja >/dev/null 2>&1 && echo 1 || echo 0),1)
	CMAKE_GENERATOR := Ninja
endif

# Directories
BUILD_DIR := .
SRC_DIR := ..
BIN_DIR := ./bin

TARGET_MAIN := $(BIN_DIR)/BSH_Engine.bin
TARGET_TEST := $(BIN_DIR)/BSH_Test.bin

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    TARGET_MAIN := $(BIN_DIR)/BSH_Engine.bin
    TARGET_TEST := $(BIN_DIR)/BSH_Test.bin
    SHA256_CMD := sha256sum
endif
ifeq ($(UNAME_S),Darwin)
    TARGET_MAIN := $(BIN_DIR)/BSH_Engine.bin
    TARGET_TEST := $(BIN_DIR)/BSH_Test.bin
    SHA256_CMD := shasum -a 256
endif
ifeq ($(OS),Windows_NT)
    TARGET_MAIN := $(BIN_DIR)/BSH_Engine.exe
    TARGET_TEST := $(BIN_DIR)/BSH_Test.exe
    SHA256_CMD := certutil -hashfile
    TARGET_MAIN_DISPLAY := $(BIN_DIR)\BSH_Engine.exe
    TARGET_TEST_DISPLAY := $(BIN_DIR)\BSH_Test.exe
endif

# Default: build the main executable
all: cmake_build
	@echo ""
	@echo "============================================"
	@echo "✓ Build Complete"
	@echo "============================================"
	@$(MAKE) hash

# Build using CMake
cmake_build:
	@echo "============================================"
	@echo "Binary Substrate Harness - CMake Build"
	@echo "============================================"
	@if [ ! -f CMakeCache.txt ]; then \
		echo "Configuring CMake..."; \
		cmake -G "$(CMAKE_GENERATOR)" -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) ..; \
	fi
	@echo "Building BSH_Engine and BSH_Test..."
	@cmake --build . --config $(CMAKE_BUILD_TYPE)
	@echo "✓ Build complete"

# Run tests
test: cmake_build $(TARGET_TEST)
	@echo ""
	@echo "============================================"
	@echo "Running Unit Tests"
	@echo "============================================"
	@if [ -f "$(TARGET_TEST)" ]; then \
		"$(TARGET_TEST)"; \
	elif [ -f "bin/BSH_Test.exe" ]; then \
		bin/BSH_Test.exe; \
	else \
		echo "Error: Test binary not found"; \
		exit 1; \
	fi
	@echo ""
	@$(MAKE) hash-test

# Compute SHA-256 hash of main binary
hash: $(TARGET_MAIN)
	@echo ""
	@echo "============================================"
	@echo "Binary Hash Verification (Main)"
	@echo "============================================"
	@if [ -f "$(TARGET_MAIN)" ]; then \
		echo "Main executable: $(TARGET_MAIN)"; \
		if command -v sha256sum >/dev/null 2>&1; then \
			echo "SHA-256: $$(sha256sum "$(TARGET_MAIN)" | awk '{print $$1}')"; \
		elif command -v shasum >/dev/null 2>&1; then \
			echo "SHA-256: $$(shasum -a 256 "$(TARGET_MAIN)" | awk '{print $$1}')"; \
		elif command -v certutil >/dev/null 2>&1; then \
			echo "SHA-256: $$(certutil -hashfile "$(TARGET_MAIN)" SHA256 2>/dev/null | grep -i sha256 | awk '{print $$2}')"; \
		fi; \
		echo "File size: $$(stat -c%s "$(TARGET_MAIN)" 2>/dev/null || stat -f%z "$(TARGET_MAIN)" 2>/dev/null) bytes"; \
	elif [ -f "bin/BSH_Engine.exe" ]; then \
		echo "Main executable: bin/BSH_Engine.exe"; \
		if command -v certutil >/dev/null 2>&1; then \
			echo "SHA-256: $$(certutil -hashfile bin/BSH_Engine.exe SHA256 2>/dev/null | grep -i sha256 | awk '{print $$2}')"; \
		fi; \
	fi
	@echo "============================================"

# Hash for test binary
hash-test: $(TARGET_TEST)
	@echo ""
	@echo "Test executable hash:"
	@if [ -f "$(TARGET_TEST)" ]; then \
		echo "Test executable: $(TARGET_TEST)"; \
		if command -v sha256sum >/dev/null 2>&1; then \
			echo "SHA-256: $$(sha256sum "$(TARGET_TEST)" | awk '{print $$1}')"; \
		elif command -v shasum >/dev/null 2>&1; then \
			echo "SHA-256: $$(shasum -a 256 "$(TARGET_TEST)" | awk '{print $$1}')"; \
		elif command -v certutil >/dev/null 2>&1; then \
			echo "SHA-256: $$(certutil -hashfile "$(TARGET_TEST)" SHA256 2>/dev/null | grep -i sha256 | awk '{print $$2}')"; \
		fi; \
	elif [ -f "bin/BSH_Test.exe" ]; then \
		echo "Test executable: bin/BSH_Test.exe"; \
		if command -v certutil >/dev/null 2>&1; then \
			echo "SHA-256: $$(certutil -hashfile bin/BSH_Test.exe SHA256 2>/dev/null | grep -i sha256 | awk '{print $$2}')"; \
		fi; \
	fi

# Verify reproducible builds
verify:
	@echo ""
	@echo "============================================"
	@echo "Reproducible Build Verification"
	@echo "============================================"
	@echo "Building first time..."
	@$(MAKE) clean >/dev/null 2>&1 || true
	@$(MAKE) all >/dev/null 2>&1
	@if command -v sha256sum >/dev/null 2>&1; then \
		FIRST=$$(sha256sum "$(TARGET_MAIN)" 2>/dev/null | awk '{print $$1}'); \
		echo "First build SHA-256: $$FIRST"; \
		$(MAKE) clean >/dev/null 2>&1; \
		$(MAKE) all >/dev/null 2>&1; \
		SECOND=$$(sha256sum "$(TARGET_MAIN)" 2>/dev/null | awk '{print $$1}'); \
		echo "Second build SHA-256: $$SECOND"; \
		if [ "$$FIRST" = "$$SECOND" ]; then \
			echo "✓ Builds are REPRODUCIBLE"; \
		else \
			echo "✗ Builds differ - NOT reproducible"; \
			exit 1; \
		fi; \
	else \
		echo "Note: sha256sum not available, skipping verification"; \
	fi
	@echo "============================================"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf bin/*.o
	@rm -f bin/BSH_Engine.bin bin/BSH_Test.bin
	@rm -f bin/BSH_Engine.exe bin/BSH_Test.exe
	@cmake --build . --target clean 2>/dev/null || true
	@echo "✓ Cleaned"

# Remove CMake generated files
distclean: clean
	@echo "Removing CMake generated files..."
	@rm -f CMakeCache.txt cmake_install.cmake
	@rm -rf CMakeFiles
	@rm -rf bin .cmake
	@echo "✓ Distribution cleaned"

# Display build info
info:
	@echo "============================================"
	@echo "Binary Substrate Harness - Build Info"
	@echo "============================================"
	@echo "C++ Standard: C++17"
	@echo "Build Type: $(CMAKE_BUILD_TYPE)"
	@echo "CMake Generator: $(CMAKE_GENERATOR)"
	@echo "Source Dir: $(SRC_DIR)"
	@echo "Build Dir: $(BUILD_DIR)"
	@echo "============================================"
	@echo "Build Targets:"
	@echo "  make              - Build BSH_Engine (main executable)"
	@echo "  make test         - Build and run unit tests"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make distclean    - Remove all generated files"
	@echo "  make verify       - Verify reproducible builds"
	@echo "  make hash         - Display binary SHA-256 hash"
	@echo "  make info         - Display this help information"
	@echo "============================================"
	@echo "Compilation Flags:"
	@echo "  Standard:     -std=c++17"
	@echo "  Optimization: -O2"
	@echo "  Warnings:     -Wall -Wextra"
	@echo "============================================"

# Help target
help: info

# Print configuration
.SILENT: info help
