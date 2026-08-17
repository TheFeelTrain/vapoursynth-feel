# AGENTS.md — vapoursynth-feel

Guidance for AI coding agents working on this repository. This file is about
the vsfeel project as a whole; per-filter implementation notes live next to
the filters themselves.

## What this project is

`vsfeel` is a VapourSynth plugin that provides GPU-accelerated video filters
implemented in **Vulkan** (GLSL compute shaders compiled to SPIR-V with
`glslc`, driven from C++ host code). Each filter is named after, and produces
results equivalent to, a reference implementation in the `reference/` folder.

The primary GPU this project is developed and tuned against is an **AMD Radeon
RX 7900XTX (RDNA3, gfx1100)**. Optimizations are targeted at that GPU; other
configurations are not the priority.

## The `reference/` folder is READ-ONLY

The `reference/` directory contains the source of the reference
implementations (e.g. `vapoursynth-zipcl`, `vapoursynth-zipcu`,
`vs-dfttest2`, `VapourSynth-BM3DCUDA`).

- **Do not modify anything inside `reference/`.** It is for reading only, so
  you can port algorithms and understand expected behaviour.
- The references are the source of truth for **numerical correctness**: a
  vsfeel filter's output should match the reference output closely (exact or
  within a ulp / small tolerance of float rounding differences between
  backends).

## The goal

Make every vsfeel filter **faster than the reference implementations** on the
target GPU. Concretely, the benchmark should show vsfeel beating the fastest
reference (vszipcl / vszipcu, whichever is faster) by a comfortable margin.

Speed matters more than code size or elegance. Do not be afraid to rewrite a
filter wholesale if it makes it meaningfully faster, as long as it stays
correct and keeps passing the tests.

When tuning, use the benchmark (below) to measure before/after, and treat the
GPUs documented here as the target. `MANGOHUD=0` should be set for every
benchmark run — it does not change results, it just suppresses extra messages
in the output.

## Testing

Every filter needs **comprehensive unit tests** in `tests/`, run with pytest.
The committed `tests/noise_24f.mkv` clip (24 frames of random noise) is the
standard test input.

- Tests must verify **correctness against the reference behaviour** and
  **self-consistency** (determinism across runs, multi-stream vs single-stream
  agreement, parallel-load consistency).
- The `tests/` folder has `conftest.py` with shared fixtures/helpers
  (`WIDTH`, `HEIGHT`, `NOISE_MKV`, `frame_to_ndarray`, ...).
- Always run the full test suite for the filter you touch before and after
  changes: `python -m pytest tests/test_<filter>.py -q`
- A rewrite is only acceptable if all tests still pass.

## How benchmarking works

The benchmark is `benchmark/bench.py`. It is data-driven: every filter is one
entry in a `FILTERS` registry describing its CLI args, the input clip
expression, and a builder that maps each supported plugin to the vpy call that
runs it. Plugins are described separately in `PLUGINS`.

- Timing is done with `vspipe`, so results are comparable across plugins and
  with the earlier per-plugin scripts.
- Usage:
  - `python3 benchmark/bench.py` — all filters
  - `python3 benchmark/bench.py --filter <name>` — one filter
  - `python3 benchmark/bench.py --filter <name> vsfeel vszipcl` — a subset of
    plugins, to compare against references
  - `--frames N`, `--num-streams N`, `--clip PATH` to control the run
- The default clip is `/home/encode/test/jpbd.mkv` (1920x1080, YUV420PS).

To benchmark a single filter against the references:

```bash
MANGOHUD=0 python3 benchmark/bench.py --filter dfttest vsfeel vszipcl vszipcu
```

This prints fps for each plugin and ranks them. Compare vsfeel's fps against
the fastest reference.

## Building

The build uses CMake + `glslc` (Vulkan shader compiler).

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The SPIR-V shaders are compiled at build time and embedded into a generated C++
header (`spirv_binaries.h`) via `src/gen_spirv_header.py`. When you add a new
shader variant, update `CMakeLists.txt` (the `VK_*` lists) and
`src/gen_spirv_header.py` accordingly.

### Installing the built plugin

Copy the built shared object into VapourSynth's plugin directory so the
running Python picks it up:

```bash
cp build/libvsfeel.so /usr/lib/python3.14/site-packages/vapoursynth/plugins/vsfeel/
```

Then re-run the tests / benchmark. The copy step is needed every time you
rebuild, or you will benchmark a stale plugin.

## Use web searches

Use web searches often. If you feel like you are getting stuck, do not be
afraid to search for hints. Search even when you think you don't need to — it
is always better to have more information. Look up relevant topics such as
GPU/Vulkan/GLSL/RDNA3 performance, shader optimization techniques, and the
reference projects' own documentation and discussions.

## Typical workflow

1. Read the reference implementation for the filter in `reference/`.
2. Check the current vsfeel implementation and its tests.
3. Build and run the benchmark + tests to get a baseline.
4. Optimize / port, rebuilding and copying the `.so` as described above.
5. Re-benchmark and re-test; keep going until vsfeel is faster than the
   references while still passing all tests.
