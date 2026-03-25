# tt-hometask

Tenstorrent ML Frameworks — Take-Home Interview Task

## Overview

Build a standalone C++ project that links against **tt-metal** as a submodule, implements a custom compute kernel, and dispatches it from a host-side main program. The project should be runnable on [**simulator**](https://github.com/tenstorrent/ttsim) (no hardware required).

---

## Requirements

### 1. Project Setup

Create a new repository with the following structure:

```
your-project/
├── CMakeLists.txt
├── README.md
├── tt-metal/                  # git submodule
├── kernels/
│   └── your_kernel.cpp        # device-side kernel
├── src/
│   └── main.cpp               # host-side dispatch
└── tests/                     # optional but appreciated
    └── test_kernel.cpp
```

- Add `tt-metal` as a **git submodule** (pin to a specific commit or tag — document which one).
- Your `CMakeLists.txt` must link against tt-metal's libraries (tt_metal, tt_eager, etc. as needed).
- The project should build cleanly with a documented set of commands.

### 2. Kernel Implementation

Write a **data-parallel compute kernel** that runs on Tenstorrent cores. 
We recommend to implement a row-wise softmax over a 2D tensor. It requires computing max, subtraction, exponentiation, sum, and division — good test of multi-pass data movement.
You can also propose your own of similar complexity.

Your kernel should:

- Operate on at least a non-trivial tensor size (e.g., `[32, 1024]`).
- Distribute work across multiple cores (not single-core).
- Handle the data movement between DRAM/SRAM and compute explicitly (circular buffers, reader/writer kernels as appropriate).

### 3. Host-Side Dispatch (`main.cpp`)

Your main program should:

1. **Initialize** the device (via `CreateDevice` or equivalent API).
2. **Allocate** input tensor(s) on host, fill with known test data.
3. **Create and enqueue** the program — set up circular buffers, configure kernel arguments, and assign kernels to cores.
4. **Transfer** data to the device, launch the program, and read back results.
5. **Validate** results against a CPU reference implementation (print PASS/FAIL with max absolute error).
6. **Tear down** the device cleanly.


## Evaluation Criteria

We're looking at the following, roughly in priority order:

1. **Does it work?** — Builds, runs on simulator, produces correct results.
2. **Understanding of the programming model** — Correct use of circular buffers, reader/writer/compute kernel separation, core assignment, and data flow between host ↔ DRAM ↔ SRAM ↔ compute.
3. **Data movement awareness** — Thoughtful tile/block sizing, awareness of SRAM capacity constraints, and efficient use of double-buffering or pipelining where appropriate.
4. **Code quality** — Clean, readable C++ with reasonable structure. Comments that explain *why*, not *what*.
5. **Validation rigor** — Meaningful correctness checks (not just "it didn't crash"), tolerance handling for floating-point comparison.
6. **Documentation** — Clear README covering build steps, design decisions, and any assumptions or known limitations.

### Bonus (not required)

- Performance comparison: single-core vs. multi-core dispatch, or varying tile sizes.
- A simple benchmark that reports throughput (elements/sec or GB/s).
- Unit tests for the CPU reference implementation.
- Profiling output or analysis of where time is spent.

---

## Deliverables

- A **git repository** (GitHub, GitLab, or tarball) with full commit history.
- A **README.md** that includes:
  - Build and run instructions (including simulator setup).
  - Brief description of your kernel and design choices.
  - Sample output showing correctness validation.
  - Any issues encountered and how you resolved them.

---

## Resources

- [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- tt-metal programming guide and examples in `tt_metal/programming_examples/`
- [Simulator](https://github.com/tenstorrent/ttsim)

---

## Notes

- If you get stuck on environment setup, document where you got stuck and what you tried. We value the kernel design and host-side logic even if the simulator doesn't cooperate.
- You may reference tt-metal's existing examples and tests, but your kernel implementation should be your own work.
- If you choose to implement something other than options A–C, briefly justify why it's a comparable challenge.
- Questions are welcome — reach out to your recruiting contact.
