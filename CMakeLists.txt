cmake_minimum_required(VERSION 3.16)
project(BSH_Engine VERSION 1.0.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler flags (consistent with Makefile)
set(CMAKE_CXX_FLAGS_INIT "-O2 -Wall -Wextra")

# Ensure reproducible builds
if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:inline)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    # For reproducible builds
    add_compile_options(-fno-canonical-system-headers)
endif()

# Find OpenSSL for SHA-256
find_package(OpenSSL QUIET)

# Source files
set(SOURCES_CORE
    src/bsh_tensor.cpp
    src/bsh_engine.cpp
)

set(SOURCES_MAIN
    ${SOURCES_CORE}
    src/main.cpp
)

set(SOURCES_TEST
    ${SOURCES_CORE}
    src/test_bsh.cpp
)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Main executable: BSH_Engine.bin
add_executable(BSH_Engine ${SOURCES_MAIN})
target_compile_features(BSH_Engine PRIVATE cxx_std_17)

# Test executable: BSH_Test.bin
add_executable(BSH_Test ${SOURCES_TEST})
target_compile_features(BSH_Test PRIVATE cxx_std_17)

# Link with OpenSSL if available (for SHA-256)
if(OPENSSL_FOUND)
    target_link_libraries(BSH_Engine PRIVATE OpenSSL::Crypto)
    target_link_libraries(BSH_Test PRIVATE OpenSSL::Crypto)
endif()

# Custom target for verification: compute SHA-256
add_custom_target(verify
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/scripts/verify_build.cmake
    DEPENDS BSH_Engine BSH_Test
    COMMENT "Verifying reproducible build..."
)

# Custom target to clean build artifacts
add_custom_target(clean-build
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMENT "Cleaning build artifacts..."
)

# Print build configuration
message(STATUS "========================================")
message(STATUS "Binary Substrate Harness - Build Config")
message(STATUS "========================================")
message(STATUS "C++ Standard: C++17")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Flags: ${CMAKE_CXX_FLAGS_INIT}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output Dir: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "OpenSSL: ${OPENSSL_FOUND}")
message(STATUS "========================================")
