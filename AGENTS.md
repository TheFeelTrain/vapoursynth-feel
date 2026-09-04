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

### Reference-comparison coverage and tolerance policy

- **Sweep parameters against the reference** — every scalar parameter, every
  supported input depth, plus special paths (joint processing, guide clips,
  cropped frames). Crash-prone references run in a subprocess.
- **Measure before setting a tolerance**, then document mechanism and
  measured values. Tiers: `1e-6` for ulp-level float32 math; measurement-
  bounded bounds (a few e-3) where discrete decisions flip on rounding order;
  integer output in whole codes (`<= 1 LSB`); self-consistency stays exact.
- Read planes back stride-aware and `.copy()` ctypes arrays.

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
- The default clip is `/home/encode/test/jpbd.mkv` (1920x1080, YUV420P8).
- By default the run **caches real frames in RAM**: the first `--cache-frames`
  (default 1000) frames are decoded while vspipe evaluates the script, and its
  fps figure only covers the output loop, so timing measures filter throughput
  on real content without the BestSource decode bottleneck (~630 fps). Frames
  loop when `--frames` exceeds the cached span, so temporal filters see a seam
  every N frames — fine for throughput, not for output inspection.

To benchmark a single filter against the references:

```bash
MANGOHUD=0 python3 benchmark/bench.py --filter dfttest vsfeel vszipcl vszipcu
```

This prints fps for each plugin and ranks them. Compare vsfeel's fps against
the fastest reference. Judge optimizations on multiple runs over hundreds of 
frames.

## Comparing a vsfeel kernel against the reference kernels

When a vsfeel filter is slower than a reference on the same GPU, the win is
almost always in kernel *codegen* or *launch structure*, not the algorithm.
Both references (vszipcl = OpenCL/ROCm, vszipcu = HIP/ROCm) run on the same
RX 7900XTX, so a fair comparison is possible. Method that worked for DFTTest:

1. **Profile each GPU kernel of the references directly** before theorizing.
   For ROCm references use `rocprofv3 -S --kernel-trace --memory-copy-trace --
   vspipe test.py /dev/null` with a synthetic BlankClip input (decode never
   hides the kernels). This gives per-kernel times in µs; time your own kernels
   the same way (`VSFEEL_DFTTEST_GPU_BENCH=N`, or `RADV_DEBUG=shaderstats`).
   Correlate structural differences (frame caches, launch config, stream/queue
   counts) against the numbers before trusting any theory — e.g. the DFTTest
   frame cache was worth only ~+24%, NOT the whole lead.
2. **Compare compiled instruction streams, not just time.** OpenCL reference
   kernels can be disassembled offline:
   `/opt/rocm/llvm/bin/clang -x cl -target amdgcn-amd-amdhsa -mcpu=gfx1100 -O3
   -cl-std=CL1.2 -cl-denorms-are-zero <prefix+kernel>.cl` then `llvm-objdump -d`.
   For our SPIR-V, `RADV_DEBUG=asm` dumps the ACO ISA; `RADV_DEBUG=shaderstats`
   prints VGPR/LDS/occupancy. Count total instructions and the FP-op
   distribution (v_fma, v_rcp, v_mov, s_mov, v_dual_*). A 3x instruction-count
   gap means ~2.5x time.
3. **Find the bloat source in the higher-level IR first.** DFTTest's culprit
   was `filter_type` as a **runtime push constant**: all 7 filter branches
   stayed alive with full-precision divisions (193 OpFDiv) while OpenCL's
   `#if FILTER_TYPE` compile-time template kept 49 v_rcp. Fix: make it a Vulkan
   **specialization constant** (`layout(constant_id = N)`, `VkSpecializationInfo`
   at pipeline creation, `if (FILTER_TYPE == ...)` chains — `#if` can't see spec
   constants but an if on a spec constant folds). Result: 7644 → 4371
   instructions, 486 → 552 fps.
4. **Check the host dispatch matches the shader's workgroup config.** A stale
   `blocks/4` grid with a `SUB_BLOCKS=8` shader launches 2x idle workgroups.

General lesson: make every branch that is fixed per invocation (filter type,
bit depth, window shape) a specialization constant or `#if` so the shader
compiles to its cheapest form.

## Shared plumbing (src/vsfeel.h)

All filters share an inline (zero-overhead, C++20) plumbing layer in
`src/vsfeel.h`. New filters must build on it — do not re-invent this wheel:

- **`FramePool<T>`** — per-instance pool of per-frame resources: a
  `ticket_semaphore` (caps in-flight frames), `std::vector<T> items`, and a
  `std::mutex lock`. `take()` blocks on the semaphore then pops the LIFO
  under the lock; `give_back(t)` pushes under the lock and releases the
  ticket. Init before first use: set `pool.semaphore.current` to
  `num_streams - 1`, `pool.reserve(num_streams)`, `pool.push(...)` one per
  created resource. Every filter's per-frame `struct` lives in such a pool;
  take at frame start, give back at frame end **and on every error path**
  (the `set_error` lambda pattern). dfttest inlines the take sequence so it
  can timestamp between acquire and lock; `pool.take()` suffices otherwise.
- **`destroy_common(dev, r)`** — frees the shared fields: cmd buffer, command
  pool, fence, staging buffer + memory. The filter unmaps mapped pointers
  *before* and destroys its filter-specific objects (extra command buffers,
  timeline semaphores, query pools, temporaries) on either side of the call.
- **`submit_with_fence(dev, queue, *qlock, cb, fence)`** and
  **`submit_timeline(dev, queue, *qlock, cb, waits, values, stages,
  signal_sem, signal_value, fence)`** — every `vkQueueSubmit` must go through
  these: they take the queue's lock (shared `std::mutex` on every `VK_Queue`
  — all submits to a shared device/queue must serialize on it) and reset the
  fence inside the lock. `submit_timeline` takes equal-sized wait
  vectors (semaphore, value, stage) and optionally signals a timeline value;
  waits are non-destructive so any number of consumers can wait on one signal.
- **`trace_on(env)`** — env-gated debug flags; keep one env name per filter
  (`VSFEEL_DFTTEST_TRACE`, `BM3D_TRACE`, ...).

**ODR rule (learned the hard way):** a filter-local struct used to
instantiate a shared template must have a **unique name per filter**
(`DFTTestResource`, `GaussBlurResource`, `BilateralResource`, `NLStream`,
`Bm3dStream` — never `VK_Resource`). Same mangled name + different
`sizeof` across TUs lets the linker COMDAT-fold one TU's instantiation over
the others, mis-striding the pool's vector and corrupting in-flight
resources.

**What stays per-filter:** frame caches (temporal three: DFTTest slot cache,
NLMeans tile cache, BM3D ring/result stacks), shaders + launch config, sync
choreography (pad→copy→fused ordering, host vs device waits), queue/stream
assignment, cache sizing. `num_queues = min(num_streams, queue_count)`,
`resource.queue = device->queues[i % num_queues]`; in-flight depth is
`num_streams` (DFTTest uses `max(num_streams, 8)`).

When porting a new filter, model the stateless path on gaussblur/bilateral
(fence-only, no cache) and the cached/timeline path on bm3d.

## Porting discipline

Lessons from porting DFTTest and NLMeans that go beyond the method above:

- **MVP first, verbatim.** Port tables, index math, and formulas from the
  reference line-for-line and get the tests passing before optimizing
  anything. Afterwards, every bug you find will be in your own new code, not
  in the ported algorithm.
- **Hold a bit-exact oracle against the closest reference where one exists.**
  Exactness (not just tolerance) is what makes aggressive structural changes
  verifiable in minutes — loose bounds cannot catch a one-column indexing
  slip. Pair small targeted tests with real-content checks; they catch
  disjoint bug classes.
- **Compose variants from shipped filters before writing new code.** A filter
  that is a geometric or parametric transform of an existing one can often be
  a few invokes with zero new state — correct by construction, with a free
  self-consistency oracle, and near-zero maintenance. New kernels are for
  what composition cannot express.
- **Only noise-clip comparisons against the reference prove correctness.**
  Constant/BlankClip input hides bugs (the references themselves deviate at
  borders on such input). In test code, never assume tight pitch when reading
  planes back, and `.copy()` any array extracted through ctypes — both alias
  recycled frame memory and produce phantom nondeterminism.
- **Trust only end-to-end benchmark fps medians over hundreds of frames.**
  Microsecond GPU traces swing ±10–20% run-to-run (clock variance); a change
  that does not move the fps median did not happen. Streams share the compute
  queue, so per-kernel timings taken from multi-stream runs include the other
  stream's interleaved work — attribute kernels only in single-stream traces.
  Identical binaries swing between invocations too, so compare same-session
  pairs or medians over 1000+ frames, never single short bursts.
- **Never chain build → install → test into one command** (a failed compile
  then silently leaves the stale `.so` installed). Verify binary freshness
  (timestamp-compare, `strings`-grep the installed `.so`) before trusting any
  measurement.
- **Prove the host/GPU split before optimizing anything.** Add a small
  env-gated chrono probe around the frame path (CPU staging / GPU
  submit-wait / download-and-blit) and read it on real content first — kernel
  work that looks dominant from reading code is routinely not the bottleneck.
  Keep durable probes like this in-tree; delete one-shot diagnostics.
- **Host orchestration is usually half the performance.** Expect to spend as
  much effort on memory pooling, upload/download paths, cross-stream cache
  sharing, and dispatch/fence structure as on kernels. The big wins come from
  removing work — fusing passes to cut dispatches/barriers/fences, uploading
  once via DMA straight into its final layout, sharing immutable data
  lock-free across streams (only writers exclude readers) — not from making
  the surviving instructions cleverer.
- **Minimize bytes moved, then minimize copies.** Transfer buffers in the
  narrowest exactly-representable type and widen on load; prefer
  non-temporal copies for write-once/read-once staging (which requires
  host-cached memory — uncached kills NT loads); re-measure copy-vs-direct
  after every structural change, because the answer flips as the code
  changes. Dump the target GPU's memory-type table once and rule transfer
  tricks in or out permanently.
- **Sweep in-flight depth; set the default at the knee.** Throughput vs
  `num_streams` is never flat and never monotonic — measure it, set the
  filter default at the knee, and state the per-stream VRAM cost next to it.
  Implementations tied at one depth can differ 2x at another (queue
  starvation vs GPU saturation).
- **Spec constants cannot size arrays in GLSL.** If an array dimension must
  vary, gate it with a compile-time `-D` define instead.
- **Respect the compiler's register tradeoffs.** ACO raises VGPRs deliberately
  for load ILP at an occupancy cost; forcing registers down often regresses.
  Read `RADV_DEBUG=shaderstats` before assuming more waves would help.
- **Reduced-precision storage needs explicit range management** (fp16 hit a
  subnormal cliff; scaling values up on store and down on load fixed it).
  Measure the actual drift against the reference and agree on the accuracy
  policy with the user before relaxing any tolerance.
- **Run Vulkan validation layers when output is inexplicable**
  (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`); they found a zeroed
  buffer-binding table in minutes.
- **Decompose kernel cost with short-lived probes**, not theory: temporary
  `-D` variants or env flags that drop one cost center at a time, measured,
  reverted immediately. Expect plausible theories to be wrong — one seemingly
  expensive memory-access pattern measured neutral because it was L2-resident.
  When the model names a cost, delete it in a scratch build and measure: a
  probe that disagrees with a confident model (barriers modeled 10x over real
  cost here) is always right. Where device profilers are unavailable, bound
  cost centers with workload-shape variants instead — inputs that isolate
  each stage (all-skip, full-work, no auxiliary data) read ceilings directly
  off end-to-end fps.
- **Keep `notes/<filter>.md` updated immediately** after every finding,
  including dead ends, so nothing is re-derived or retried later.

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

## Scratch files

Do all scratch work (temporary scripts, probe outputs, intermediate artifacts)
in the repo's `tmp/` folder instead of the system `/tmp`. Under the sandboxed
shell, `/tmp` is a fresh tmpfs per shell call, so files written there are
cleared between commands — anything a later command needs must live under the
workspace (`tmp/`).

## Commits

Do not make any commits yourself. If you want a commit or a checkpoint, stop
and ask the user to make it for you. Leave your changes staged/unstaged in the
working tree and describe what should be committed.

## Typical workflow

1. Read the reference implementation for the filter in `reference/`.
2. Check the current vsfeel implementation and its tests.
3. Build and run the benchmark + tests to get a baseline.
4. Optimize / port, rebuilding and copying the `.so` as described above.
5. Re-benchmark and re-test; keep going until vsfeel is faster than the
   references while still passing all tests.
