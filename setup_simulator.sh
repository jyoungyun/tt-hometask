#!/bin/bash

set -e

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTSIM_DIR="${SOURCE_ROOT}/ttsim"
TTSIM_VERSION="v1.4.7"
TTSIM_BASE_URL="https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}"

# ─────────────────────────────────────────────
# Download simulator
# ─────────────────────────────────────────────
echo "=== Downloading ttsim ${TTSIM_VERSION} ==="

mkdir -p "${TTSIM_DIR}"

for lib in libttsim_bh.so libttsim_wh.so; do
    if [ -f "${TTSIM_DIR}/${lib}" ]; then
        echo "  [skip] ${lib} already exists"
    else
        echo "  [download] ${lib}"
        wget -q --show-progress -L "${TTSIM_BASE_URL}/${lib}" -O "${TTSIM_DIR}/${lib}"
    fi
done

# ─────────────────────────────────────────────
# Copy SOC descriptor
# ─────────────────────────────────────────────
echo "=== Setting up SOC descriptor ==="

SOC_SRC=$(find /usr -name "wormhole_b0_80_arch.yaml" 2>/dev/null | head -1)
if [ -z "${SOC_SRC}" ]; then
    echo "  [error] wormhole_b0_80_arch.yaml not found"
    exit 1
fi

cp "${SOC_SRC}" "${TTSIM_DIR}/soc_descriptor.yaml"
echo "  [ok] soc_descriptor.yaml copied from ${SOC_SRC}"

echo "=== Setup complete ==="
echo ""
echo "  Run the following command to set up the environment:"
echo "  source ${PROJECT_ROOT}/env.sh"
