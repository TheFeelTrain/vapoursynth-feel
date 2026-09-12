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

**Per-stream efficiency is the metric, not raw per-frame throughput.** A filter
that only looks fast because it burns 24 streams on a 24 GB card is a worse
result than one that matches it at 8. `num_streams = 8` is the **maximum**
default any filter should ship: it is the established sweet spot across the
existing filters, keeps per-instance VRAM bounded, and leaves headroom in a
user's surrounding graph. If a filter only wins above 8 streams, treat that as
evidence the per-stream path is inefficient — find the inefficiency rather than
raising the count. Sweep below 8 as well (4 and 6 are common knees) and ship
the lowest count that reaches the plateau.

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

Two-tier measurement keeps the iteration loop tight: screen candidates with a
fast custom `.vpy` + `vspipe` (a few hundred frames, BlankClip or a small
cached real clip), and grade only on full `bench.py` same-session pairs over
1000+ frames (a `rep_<filter>.sh` loop of 5000-frame ×3 repeats settles
medians). A one-off fast-vpy number that later pairs contradict was noise,
not a finding.

## Comparing a vsfeel kernel against the reference kernels

When a vsfeel filter is slower than a reference on the same GPU, the win is
almost always in kernel *codegen* or *launch structure*, not the algorithm.
Both references (vszipcl = OpenCL/ROCm, vszipcu = HIP/ROCm) run on the same
RX 7900XTX, so a fair comparison is possible. Method that worked for DFTTest:

1. **Profile each GPU kernel of the references directly** before theorizing.
   For ROCm references use `rocprofv3 -S --kernel-trace --memory-copy-trace --
   vspipe test.py /dev/null` with a synthetic BlankClip input (decode never
   hides the kernels). This gives per-kernel times in µs; time your own kernels
   with warm in-command-buffer timestamp queries, not a cold one-shot bench:
   reset the query pool and stamp (CB head / post-barrier / per-stage ends)
   inside the single submitted CB, arm one shot on a warm frame (~100 —
   frame 0 runs at idle clocks, ~500 MHz vs ~2 GHz steady; check
   `pp_dpm_sclk`), read back with `vkCmdCopyQueryPoolResults` + `WAIT_BIT`.
   Reset and stamps must share one CB or pool reuse across frames/resources
   races; stamps from a CB that was recorded but never submitted read back as
   absolute epoch values, so sanity-check deltas.
   Correlate structural differences (frame caches, launch config, stream/queue
   counts) against the numbers before trusting any theory — e.g. the DFTTest
   frame cache was worth only ~+24%, NOT the whole lead.
2. **Compare compiled instruction streams, not just time.** OpenCL reference
   kernels can be disassembled offline:
   `/opt/rocm/llvm/bin/clang -x cl -target amdgcn-amd-amdhsa -mcpu=gfx1100 -O3
   -cl-std=CL1.2 -cl-denorms-are-zero <prefix+kernel>.cl` then `llvm-objdump -d`.
   For our SPIR-V, `RADV_DEBUG=asm` dumps the ACO ISA; `RADV_DEBUG=shaderstats`
   prints VGPR/LDS/occupancy. Instruction counts are only comparable at
   equal unroll structure — check loop-branch counts first (a fully-unrolled
   kernel vs a rolled loop with a dual-issued body can differ 10x on paper
   and tie on hardware). Always report Subgroups-per-SIMD, LDS, and
   spill/scratch bytes next to the count: a lower count with halved occupancy
   or new spills is a regression. Count the FP-op
   distribution (v_fma, v_rcp, v_mov, s_mov, v_dual_*). A 3x instruction-count
   gap means ~2.5x time. When diffing your own change before/after, watch for
   schedule-damage signatures, not just the total: doubled `buffer_load_*` =
   a branch if-converted into loads issued on both sides; an `s_load_b128`
   spike = push-constant pressure (budget the guaranteed 128 B — growing the
   block pushes reads through SMEM); a `v_dual_*` drop plus `s_waitcnt`
   explosion = broken dual-issue packing across the whole kernel.
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
compiles to its cheapest form. For branches that vary per frame (cache hit
vs fallback path), ship two pipelines — a branchless fast variant plus a
mixed fallback — and let the host pick per dispatch: a uniform `if` in a hot
loop if-converts into loads issued on both sides plus de-dualized ALU and
`s_waitcnt` chains (measured 2x slower), and ACO will not save you.

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
assignment, cache sizing. `num_queues = min(num_streams, queue_count)` is only
the starting point — the cap is swept per filter (see the queue-sharing rule
below), `resource.queue = device->queues[i % num_queues]`; in-flight depth is
`num_streams` (DFTTest uses `max(num_streams, 2)` — queue count and buffer
count are independent; see the knee rule below).

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
- **A recorded dead end decays: re-test it against the CURRENT code, and
  record the mechanism, not the verdict.** "Host-visible VRAM is poison (27.6
  fps)" and "reading staging directly collapses to 178 fps" were both true for
  the technique that was tried (ordinary cached stores into uncached memory)
  and both false for a different technique on the same memory (NT stores: +29%
  on EEDI3). Before inheriting a dead end, ask whether your variant actually
  shares the failed mechanism. The strongest signal that one is stale is that
  **another filter in this repo already ships the thing the notes call
  impossible** — nnedi3 had the exact ReBAR NT-store upload. When notes and
  shipped code disagree, read the code.
- **For a semantics-preserving rewrite, build inputs that straddle the new
  code's decision boundaries.** The oracle alone was not enough: EEDI3's
  replacement mask predicate was off by one (`>= 130` instead of `>= 129`) and
  BOTH binary masks and uniformly-random masks passed — neither can see a
  threshold error. Only masks concentrated at the boundary (values 120–139)
  exposed it. So when you swap a computation for a cheaper equivalent, run the
  old and new implementations side by side on identical inputs across
  distributions that target every threshold, rounding, and tie-break in the
  new code. Comparing both against a reference is weaker: the reference may
  itself be permissive.
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
  pairs or medians over 1000+ frames, never single short bursts. A
  comments-only rebuild that moves a one-shot trace is clock variance, not a
  regression — check `pp_dpm_sclk` and repeat 3x before debugging.
  When one plugin swings ±15–40% between invocations while the other stays
  flat, triage in order: 5x same-command repeats, then alone-vs-pair
  interleaving, then `pp_dpm_sclk` / `gpu_busy_percent` polled in a loop
  *during* the run (a single read after a 2–3 s run only ever sees idle),
  then thermals and host load, then **harness memory** (see the bimodal rule
  below — additive frame caches overflowing RAM produce exactly this
  signature). Same-session pairs stay fair through all of it — grade on those,
  and lengthen the run before trusting any absolute number (short runs are
  clock-ramp-sensitive).
- **Never chain build → install → test into one command** (a failed compile
  then silently leaves the stale `.so` installed). Verify binary freshness
  (timestamp-compare, `strings`-grep the installed `.so`) before trusting any
  measurement.
- **Prove the host/GPU split before optimizing anything.** Add a small
  env-gated chrono probe around the frame path (CPU staging / GPU
  submit-wait / download-and-blit) and read it on real content first — kernel
  work that looks dominant from reading code is routinely not the bottleneck.
  Reset the stage clock after every blocking acquire (pool take, fence wait)
  so waits never leak into the next stage, and cross-check summed stages
  against wall-clock before trusting any split — a stage reporting
  milliseconds for a microsecond memcpy is a timer bug, not a finding.
  Keep durable probes like this in-tree; delete one-shot diagnostics.
- **"I removed the work and nothing changed" is a RESULT, not a failed probe.**
  On EEDI3, deleting the entire DP + backtrack + interpolate changed fps by
  −1%: the GPU kernel was fully hidden behind the host path at that stream
  count. The correct conclusion is *the frame is host-bound and the GPU has
  spare capacity* — stop tuning kernels and go count CPU bytes. Frame cost is
  roughly `max(GPU, host)` per stream, so which side you are on can only be
  read off an ablation, never inferred from kernel timings. Build the ablation
  ladder as remove-all → remove-half → remove-one so the answer is unambiguous.
- **An isolated microbenchmark proposes; only an in-situ same-session A/B
  decides.** A standalone C program said plain `memcpy` beat the NT-store path
  3.7× (0.14 vs 0.51 ms/frame); in the real filter that same change *lost*
  (blit 6.1 → 10.4 ms/frame, 481 → 401 fps), because the isolated test has no
  competing streams, no other cache pressure, and no write-combining
  contention. Prefer a **runtime knob plus a same-session sweep of the real
  binary** — ten lines, and both arms are measured under identical conditions.
  Reach for the scratch C program only to rule a mechanism *out*, never to
  pick a winner.
- **When every host stage costs about the same, you are bound by a shared
  resource — not by a hot loop.** EEDI3's ladder read blit ~20%, upload gather
  ~16%, vcheck ~10%, sclip ~10%, and no stage dominated. Additive costs of
  similar size mean the host memory path; a single dominant item means a loop
  to optimize. Tell them apart with the ladder (one stage, then pairs, then
  all), and expect the fix to be "move fewer bytes", not "make a loop
  tighter". Related: **aggregate CPU copy bandwidth falls as thread count
  rises** (measured ~101 GB/s at 2 threads vs ~54 at 8), so more streams can
  *reduce* total host throughput.
- **Host orchestration is usually half the performance.** Expect to spend as
  much effort on memory pooling, upload/download paths, cross-stream cache
  sharing, and dispatch/fence structure as on kernels. The big wins come from
  removing work — fusing passes to cut dispatches/barriers/fences, uploading
  once via DMA straight into its final layout, pointing the consumer at the
  cache in place (per-slice/per-tile source addresses via push constants,
  e.g. a `slot_base[]` table with a `-1` = fallback sentinel) instead of
  copying cache→working set and then reading it, sharing immutable data
  lock-free across streams (only writers exclude readers) — not from making
  the surviving instructions cleverer. The existing fence/resource-reuse gate
  usually already covers the new (longer) cache lifetime, so no new sync is
  needed.
- **Minimize bytes moved, then minimize copies — and count reads per source
  byte, not just copies.** Transfer buffers in the narrowest
  exactly-representable type and widen on load (native u16 pad instead of f32
  halved EEDI3's upload, H2D and pad-read traffic at once). Then audit the
  frame for any buffer whose *same source bytes* are read by two different
  loops: merging a second pass into the first while the row is still cache-hot
  is nearly free and was worth +3.5% on EEDI3 while deleting 8.3 MB/frame of
  pure DRAM re-reads.
- **Upload path (measured on the 7900XTX, and it has flipped repeatedly —
  re-measure per filter).** The winner is **NT stores directly into a
  host-visible VRAM buffer** (`DEVICE_LOCAL|HOST_VISIBLE|COHERENT`, mapped
  once, `_mm_sfence()` before submit, no H2D copy and no transfer→compute
  barrier): +29% on EEDI3 over cached-system-RAM staging + `vkCmdCopyBuffer`.
  The catch is that the host-visible device-local types here are **uncached**,
  so ordinary cached stores into them are catastrophic (measured 27.6 fps) —
  only the NT store form is fast. Within the older staging approach, NT stores
  also beat plain `memcpy` in situ (400–417 vs 331 fps over 8 combos). The
  isolated microbenchmark said the OPPOSITE (memcpy 0.14 vs NT 0.51 ms/frame) —
  see the probe rule below: **an isolated copy benchmark proposes, only an
  in-situ same-session A/B decides.**
  **UNRESOLVED CONTRADICTION — measure before trusting either side:** a
  previous session recorded the reverse for a plain 64 GB/s vs 31 GB/s
  throughput comparison (memcpy into ReBAR VRAM *beating* NT stores). Both
  numbers are real measurements on this box, so the winner evidently depends
  on the buffer, size, or access pattern. Treat the upload path as
  unfixed: implement it behind an env opt-out, sweep both arms in situ, and
  record which one won and for what shape.
  Downloads stay kernel-direct (the GPU writing results straight into host
  staging beat a device-local buffer + SDMA D2H, 568 vs 553 fps); CPU reads
  from the VRAM BAR remain ~1 GB/s, so never let the CPU read results back
  from a VRAM buffer.
- **Do not invoke a graph node to normalize an input your kernel only reads as
  a predicate.** EEDI3 forced every mask through `SetFrameProps(_Range=1) ->
  resize.Point -> Gray8` so the kernel could test `byte != 0` — a whole extra
  full-frame pass in the graph, every frame. Because the reduction is
  monotonic, that predicate is exactly `v >= 129` on the native u16 input, so
  the node was deleted for an instant +8%. Whenever a filter adds a
  std/resize/format node at create time, ask what the consumer actually needs
  from it; a monotone transform or a threshold can usually be folded into the
  native data. Verify equivalence exhaustively when the domain is small
  (65536 values is a proof, not a hope).
- **Port a proven memory path to the next filter before tuning kernels.**
  Once a transfer structure wins on one filter (device-local buffers,
  host-direct upload, kernel-direct download, native element types), port
  the whole structure wholesale to the next filter with the same IO shape
  before spending anything on kernel cleverness — it routinely contributes
  most of the absolute gain. Ablate each leg with its opt-out env flag so
  the contribution is measured, and keep those flags as durable tuning
  knobs rather than deleting them as one-shot diagnostics.
- **Sweep in-flight depth; set the default at the knee, and never above 8.**
  Throughput vs `num_streams` is never flat and never monotonic — measure it,
  set the filter default at the knee, and state the per-stream VRAM cost next
  to it. **`num_streams = 8` is the hard maximum for a shipped default** (see
  "The goal"); a knee above 8 means the per-stream path needs work, not a
  higher count. Queue count and buffer count (ticket depth) are independent:
  the knee is typically 2 (frame N runs on the GPU while N+1
  uploads/records/submits), and deeper pools only add VRAM.
  **Re-sweep the knee after every structural change** — it is a property of
  where the bottleneck sits, so shifting work between host and GPU moves it.
  EEDI3's knee went 8 → 12 when the upload path was made cheaper, and DFTTest's
  went 8 → 2 when the frame cache landed. Never inherit depth constants, or
  knee conclusions, across redesigns.
  Implementations tied at one depth can differ 2x at another (queue
  starvation vs GPU saturation).
- **Sweep queue sharing independently of stream count.**
  `min(num_streams, queue_count)` is only the starting point. With one
  stream per queue, each queue drains while its worker does post-fence CPU
  work (download memcpy + bookkeeping + next upload) before the next submit,
  leaving idle bubbles; sharing a queue across streams keeps a next CB
  queued. Sweep the cap (`min(num_streams, queue_count, CAP)` for
  CAP = 1..queue_count, via a `VSFEEL_<FILTER>_QUEUES` env override kept as
  a durable tuning knob) at the graded depth. Oversubscribing queues can win
  double digits and collapse run-to-run variance — a variance drop alongside
  the speedup confirms the bubble mechanism. Re-sweep after memory-path
  changes, since removing copies changes bubble sizes.
- **Sweep workgroup shape across workload configs, not just the default.**
  The best tile is a function of the algorithm's workload params (window
  radius, taps, halo overfetch), not a universal constant — a shape that
  ties at one config can win 50% at another and collapse at a third, so
  sweep the matrix (block candidates × representative configs) and re-verify
  under the final queue cap, since overlap changes amplify or shrink shape
  effects. Where the matrix shows a clear workload-dependent winner, auto-
  select the default from the workload params (only when the user leaves the
  args unset — the `mapGetInt` error flag distinguishes explicit from
  default, and explicit args are always respected). Never inherit shapes
  across redesigns: a spill-free shape at one radius can spill at another,
  so check `shaderstats` (VGPR spill/scratch) per matrix cell.
- **Decide allocation-dependent fast paths before the resource loop.**
  If a fast path depends on a memory type existing (host-visible
  device-local for direct upload, and so on), probe once up front and store
  an immutable global bool — never mutate a shared flag per resource inside
  the creation loop. `allocate_memory` relaxes requirements per buffer, so a
  mid-loop flip desyncs already-recorded command buffers (copy vs no-copy)
  from the frame-time upload/download base used by all resources. When the
  fast path is unavailable, every resource must take the fallback
  consistently.
- **Spec constants cannot size arrays in GLSL.** If an array dimension must
  vary, gate it with a compile-time `-D` define instead.
- **Respect the compiler's register tradeoffs.** ACO raises VGPRs deliberately
  for load ILP at an occupancy cost; forcing registers down often regresses.
  Read `RADV_DEBUG=shaderstats` before assuming more waves would help.
  The same applies to `requiredSubgroupSize`: forcing wave32 halves
  Subgroups-per-SIMD and only pays off for kernels with subgroup ops or
  extreme register pressure — otherwise it regresses. Sweep it per filter
  like any other launch param; a win on one filter never implies a win on
  the next.
- **Reduced-precision storage needs explicit range management** (fp16 hit a
  subnormal cliff; scaling values up on store and down on load fixed it).
  Measure the actual drift against the reference and agree on the accuracy
  policy with the user before relaxing any tolerance.
- **One descriptor set per buffer role; aliased sets fail silent and look
  fast.** If two paths need different buffers on the same binding (e.g. pad
  reads upload while fused writes download on binding 1), they need separate
  sets — sharing one silently redirects writes out of bounds (all-zero output
  at *higher* fps). BlankClip never catches this. After any memory-path
  change, noise-diff against the reference *before* trusting fps.
- **Run Vulkan validation layers when output is inexplicable**
  (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`); they found a zeroed
  buffer-binding table in minutes.
- **The CPU-side staging code deserves as much scrutiny as the shaders.**
  Once the host path is the limiter, the wins are in the C++: EEDI3's
  single largest absolute cost was a scalar per-row loop in the *host* code
  (6.1 ms/frame, ~22× slower than the SIMD rewrite), not anything on the GPU.
  Two patterns to hunt:
  - **Serial scans carrying a scalar across iterations** (`last = ...` style
    state that each step depends on). If the loop computes a *window* property
    (is-any-set, coverage, box-sum reachability), it is usually a **dilation**,
    and dilations compose (`dil_a ∘ dil_b == dil_(a+b)`) — so a packed-bit
    formulation collapses O(N) serial steps into O(log N) shift-OR passes
    over machine words. 6.1 → 0.28 ms/frame here. Pack predicate bits with
    `_mm256_cmpeq_epi8` + `movemask`, or for u16 with an xor-bias + signed
    `cmpgt` + `movemask` (remember `cmpgt` is strict: for `v >= T` the
    constant is `T-1`).
  - **Repeated reads of one source buffer by separate loops** (see the
    read-counting rule above).
  Prove a new SIMD kernel against a brute-force oracle over randomized inputs
  *before* wiring it in, including a fallback path for shapes the vector form
  cannot express (narrow rows, oversized parameters).
- **A bimodal measurement is a bug to chase, and the harness's memory budget
  is part of the measurement.** One binary swung 143–260 fps on consecutive
  fp32 runs while the GPU-bound reference stayed flat at ~190. The cause was
  the *harness*: the framework's `max_cache_size` and the harness's own
  decoded-frame cache are additive, and together they exceeded RAM. When you
  see the "one side swings, other stays flat" signature, add **harness memory**
  to the triage list (clocks, thermals, queue count, host load). And when
  fixing it, shrink the *test-specific* cache, not the shared framework one:
  lowering `max_cache_size` instead made the timed region re-run the upstream
  chain and collapsed *every* plugin by 3–4× — a much more confusing failure
  than the original noise.
- **Decompose kernel cost with short-lived probes**, not theory: kill the
  theory with arithmetic before coding anything — bandwidth math (taps ×
  bytes × pixels vs bus), value-range math (min/max exponent vs subnormal),
  tile-size math (halo overfetch, LDS bytes vs budget). Then structure
  probes as a ladder: remove-all first (empty/box kernel) to read the
  ceiling, then remove-half (keep exactly one cost center) to attribute.
  Implement probes as temporary `-D` variants or env flags, back up the
  `.comp` first for a trivial revert, measure, revert immediately. Expect plausible theories to be wrong — one seemingly
  expensive memory-access pattern measured neutral because it was L2-resident.
  When the model names a cost, delete it in a scratch build and measure: a
  probe that disagrees with a confident model (barriers modeled 10x over real
  cost here) is always right. Where device profilers are unavailable, bound
  cost centers with workload-shape variants instead — inputs that isolate
  each stage (all-skip, full-work, no auxiliary data) read ceilings directly
  off end-to-end fps.
- **Keep `notes/<filter>.md` updated immediately** after every finding,
  including dead ends, so nothing is re-derived or retried later. Since notes
  are gitignored, they are local working memory: another checkout will not
  have them, so anything that must survive belongs in this file or in a code
  comment. **Worked examples of most rules above live in `notes/EEDI3.md`
  rounds 10–12** (host-bounded frames, the SIMD dilation rewrite, the ReBAR
  upload, the boundary-input oracle bug, the harness-memory bimodality, and a
  list of measured non-wins) — worth reading before starting a new filter,
  even though none of it is EEDI3-specific.

## Building

The build uses CMake + `glslc` (Vulkan shader compiler).

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The SPIR-V shaders are compiled at build time and embedded into a generated C++
header (`spirv_binaries.h`) via `src/gen_spirv_header.py`. Adding a shader
variant is one step: append its glslc rule to `VK_SPV_OUTPUTS` in
`CMakeLists.txt` — the header script (`--out <header> <spv...>`) derives
symbol names from the filenames (`<stem>.spv` → `<stem>_spv` /
`<stem>_spv_size`), so it never needs editing for new variants.

### How the shader pipeline fits together

Each `src/*.comp` file holds several entry points selected by `-D` defines
(e.g. `-DENTRY_FUSED -DRADIUS=1 -DBITS=16`); a CMake `foreach` loop compiles
one `.spv` per variant into `build/vk_spv/`, and `gen_spirv_header.py` packs
them all into `spirv_binaries.h` as `uint32_t` arrays. The C++ side picks the
arrays it needs (usually per bit depth / radius at filter creation) and
creates one `VkShaderModule` + `VkPipeline` per variant. So a new fast-path
variant means: an `#if` block in the `.comp`, one CMake loop entry (copy a
neighboring `add_custom_command` and change the `-D` flags + output name),
and a module/pipeline pair in the filter's creation function — nothing else.

### Installing the built plugin

Copy the built shared object into VapourSynth's plugin directory so the
running Python picks it up:

```bash
cp build/libvsfeel.so /usr/lib/python3.14/site-packages/vapoursynth/plugins/vsfeel/
```

Then re-run the tests / benchmark. The copy step is needed every time you
rebuild, or you will benchmark a stale plugin.

### Persistent pipeline cache (makes creation and the test suite fast)

Compiling the compute shaders from SPIR-V dominates filter creation on RADV —
about 4.4 s for the first DFTTest variant in a process, versus ~0.11 s once the
driver has it. The plugin therefore keeps a `VkPipelineCache` seeded from and
flushed to a file:

- default location: `$XDG_CACHE_HOME/vsfeel/pipeline_cache_<pipelineCacheUUID>.bin`
  (`~/.cache/vsfeel/...`), with an automatic fallback to `$TMPDIR/vsfeel/` when
  the home cache is not writable (read-only home, sandbox, CI);
- `VSFEEL_PIPELINE_CACHE=<path>` overrides the file, `VSFEEL_PIPELINE_CACHE=0`
  disables the cache entirely (useful when checking that a shader change
  actually recompiles);
- the driver's `pipelineCacheUUID` is both the filename and part of the cache
  contents, so a driver/device change misses instead of feeding the driver
  incompatible data. The file is written via a unique temp file + rename, so
  concurrent test subprocesses cannot corrupt it.

Measured on the full suite: **~7.5 min cold → ~3.5 min for the run that builds
the cache → ~2.5 min warm** (443 tests). The first run after a driver or
`glslc` update pays the compile again by design.

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
4. **Measure the host/GPU split before optimizing** (the chrono probe), then
   run an ablation ladder (remove-all / remove-half) to find which side is
   actually the limiter. Do not assume it is the kernels.
5. Optimize / port, keeping each candidate behind an env opt-out so it can be
   A/B'd in situ, and re-sweep the stream knee afterwards. Rebuild and copy
   the `.so` as described above.
6. Re-benchmark and re-test; keep going until vsfeel is faster than the
   references while still passing all tests.
