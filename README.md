# tt-hometask

Tenstorrent ML Frameworks — Take-Home Interview Task

A minimal end-to-end implementation of tile-based compute kernels using TT-Metal, validated on the simulator environment.

This project demonstrates:
- DRAM ↔ Circular Buffer data movement
- Reader / Compute / Writer kernel pipeline
- Tile-based execution model
- Basic correctness validation against a golden reference

### Setup

Install the simulator environment:

```bash
./setup_simulator.sh
```

This script sets up:
- TT-Metal runtime dependencies
- Simulator environment
- Required libraries and toolchain

### Build

Build the project:

```bash
./build.sh
```

This will:
- Configure CMake
- Compile host code
- Compile TT kernels (JIT at runtime)

### Run

After build, execute:

```bash
./out/tt_softmax
```

Expected output:

```bash
Max ABS error: <value>
```

### Validation

The result is validated against a CPU reference implementation.

Typical output:

```bash
Max ABS error: 0.015625
```

Due to reduced precision (bfloat16), small numerical differences are expected.

### Notes on Performance

- This project runs on the TT simulator, not real hardware

