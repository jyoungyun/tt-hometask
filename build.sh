#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 도움말
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -c, --clean    Clean out directory before building"
    echo "  -h, --help     Show this help message"
    exit 0
}

# 옵션 파싱
CLEAN=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --clean|-c) CLEAN=true ;;
        --help|-h) usage ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo "=== Building tt-hometask ==="

# 초기화
if [ "$CLEAN" = true ]; then
    echo "=== Cleaning out directory ==="
    rm -rf "$PROJECT_ROOT/out"
fi

# cmake configure
mkdir -p "$PROJECT_ROOT/out"
cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/out"

# build
make -C "$PROJECT_ROOT/out" -j$(nproc)

echo "=== Build complete ==="
echo ""
echo "  Run the following command to execute the program:"
echo "  ./out/tt_softmax"
