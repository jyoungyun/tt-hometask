#!/bin/bash
set -e

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null
then
    echo "Error: clang-format is not installed."
    echo ""
    echo "Please install it first:"
    echo "   Ubuntu: sudo apt install clang-format"
    echo "   Mac (brew): brew install clang-format"
    echo ""
    exit 1
fi

FILES=$(find src kernels -name "*.cpp" -o -name "*.hpp" -o -name "*.h")

echo "Checking formatting..."

clang-format -i $FILES

echo "Format check passed!"

