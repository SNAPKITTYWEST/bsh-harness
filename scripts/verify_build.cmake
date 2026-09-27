# Verify reproducible build by computing SHA-256 hashes
# Run from CMake: cmake -P verify_build.cmake

set(BIN_DIR "${CMAKE_BINARY_DIR}/bin")
set(MAIN_BIN "${BIN_DIR}/BSH_Engine.bin")
set(TEST_BIN "${BIN_DIR}/BSH_Test.bin")

message(STATUS "============================================")
message(STATUS "Verifying Reproducible Builds")
message(STATUS "============================================")

# Function to compute SHA-256 (requires OpenSSL)
function(compute_sha256 file hash_var)
    if(OPENSSL_FOUND)
        execute_process(
            COMMAND openssl dgst -sha256 "${file}"
            OUTPUT_VARIABLE HASH_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REGEX MATCH "[a-f0-9]{64}" HASH "${HASH_OUTPUT}")
        set(${hash_var} "${HASH}" PARENT_SCOPE)
    else()
        set(${hash_var} "N/A (OpenSSL not found)" PARENT_SCOPE)
    endif()
endfunction()

if(EXISTS "${MAIN_BIN}")
    file(SIZE "${MAIN_BIN}" FILE_SIZE)
    compute_sha256("${MAIN_BIN}" MAIN_HASH)
    message(STATUS "Main binary: ${MAIN_BIN}")
    message(STATUS "  Size: ${FILE_SIZE} bytes")
    message(STATUS "  SHA-256: ${MAIN_HASH}")
else()
    message(WARNING "Main binary not found: ${MAIN_BIN}")
endif()

if(EXISTS "${TEST_BIN}")
    file(SIZE "${TEST_BIN}" FILE_SIZE)
    compute_sha256("${TEST_BIN}" TEST_HASH)
    message(STATUS "Test binary: ${TEST_BIN}")
    message(STATUS "  Size: ${FILE_SIZE} bytes")
    message(STATUS "  SHA-256: ${TEST_HASH}")
else()
    message(WARNING "Test binary not found: ${TEST_BIN}")
endif()

message(STATUS "============================================")
