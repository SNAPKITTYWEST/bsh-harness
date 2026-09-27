#!/bin/bash
# Compute SHA-256 hash of binary and verify reproducibility

set -e

BINARY_PATH="${1:-./bin/BSH_Engine.bin}"

if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Binary not found: $BINARY_PATH"
    exit 1
fi

echo "============================================"
echo "SHA-256 Hash Computation"
echo "============================================"

# Try different hash commands
if command -v sha256sum >/dev/null 2>&1; then
    HASH=$(sha256sum "$BINARY_PATH" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    HASH=$(shasum -a 256 "$BINARY_PATH" | awk '{print $1}')
elif command -v openssl >/dev/null 2>&1; then
    HASH=$(openssl dgst -sha256 "$BINARY_PATH" | awk '{print $NF}')
else
    echo "Error: No SHA-256 utility found"
    exit 1
fi

SIZE=$(stat -c%s "$BINARY_PATH" 2>/dev/null || stat -f%z "$BINARY_PATH")

echo "Binary: $BINARY_PATH"
echo "Size: $SIZE bytes"
echo "SHA-256: $HASH"
echo "============================================"

# Store hash for reproducibility check
mkdir -p .build-cache
echo "$HASH" > .build-cache/last_hash.txt

exit 0
