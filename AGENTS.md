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
RX 7900XTX (RDNA3, gfx1100)**. Optimizations are targeted at that GPU, but do
consider other configurations if possible.

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

## Comments and docstrings

Keep them short and only where the code is not self-explanatory: aim for 3
lines or less, say *why* rather than *what*, and do not restate the code. Test
docstrings are a few lines at most. Notes files (`notes/<filter>.md`) may be
longer, but do not pad them. Never cite work-order or report IDs (`WO-55`,
`§I.11`, `T9`) in code, comments, docstrings or notes: those files are untracked
working documents, so a checkout reader cannot resolve the codes. When a
number matters, put it next to the config that produced it.

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
  changes: `python -m pytest tests/test_<filter>.py -q`.
- Run the whole suite with **`tools/test.sh`** (extra args are passed through,
  default target `tests`). It uses pytest-xdist (`dev` dependency group) with
  `--dist loadfile` and caps workers at 8. Measured on the RX 7900XTX, 789
  tests: **522 s serial, ~2.5 min via `tools/test.sh`** (133–146 s observed).
  Do not raise the cap and
  do not drop `loadfile`: 16 workers fail with `vkQueueSubmit failed` (GPU
  exhaustion), and a plain `-n 8` is flaky because NLMeans `a=64, d=16`
  hard-recovers the GPU whenever another client shares it — file-level
  distribution stops that config overlapping another heavy file. See
  `notes/NLMEANS.md`. `VSFEEL_TEST_WORKERS=1 tools/test.sh` forces serial.
- **The suite loads the installed plugin, not `build/libvsfeel.so`.** After any
  C++ or shader change, build *and install* with `tools/install.sh` — one
  command — before running pytest, or you are testing the previous binary.
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

The benchmark is `tools/benchmark.py`. It is data-driven: every filter is one
entry in a `FILTERS` registry describing its CLI args, the input clip
expression, and a builder that maps each supported plugin to the vpy call that
runs it. Plugins are described separately in `PLUGINS`.

- Timing is done with `vspipe`, so results are comparable across plugins and
  with the earlier per-plugin scripts.
- Usage:
  - `python3 tools/benchmark.py` — all filters
  - `python3 tools/benchmark.py --filter <name>` — one filter
  - `python3 tools/benchmark.py --filter <name> vsfeel vszipcl` — a subset of
    plugins, to compare against references
  - `--frames N`, `--clip PATH` to control the run
- The default clip is `/home/encode/test/jpbd.mkv` (1920x1080, YUV420P8).
- By default the run **caches real frames in RAM**: the first `--cache-frames`
  (default 1000) frames are decoded while vspipe evaluates the script, and its
  fps figure only covers the output loop, so timing measures filter throughput
  on real content without the BestSource decode bottleneck (~630 fps). Frames
  loop when `--frames` exceeds the cached span, so temporal filters see a seam
  every N frames — fine for throughput, not for output inspection.

To benchmark a single filter against the references:

```bash
MANGOHUD=0 python3 tools/benchmark.py --filter dfttest vsfeel vszipcl
```

This prints fps for each plugin and ranks them. Compare vsfeel's fps against
the fastest reference. Judge optimizations on multiple runs over hundreds of
frames.

Two-tier measurement keeps the iteration loop tight: screen candidates with a
fast custom `.vpy` + `vspipe` (a few hundred frames, BlankClip or a small
cached real clip), and grade only on full `benchmark.py` same-session pairs over
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
5. **Cut a ranked intermediate to the depth its *consumer* reads.** BM3D's
   temporal search kept a per-lane top-8 because the *spatial* search needs all
   eight, but the temporal consumer only ever reads the top `PS_NUM`
   (`merge_group(PS_NUM)` feeds `ginsert8` and the next step's centres).
   Rebuilding it at depth `PS_NUM` cut the insert's shift from seven steps to
   one and its live registers from 24 to 6: kernel 3.72 → 2.85 ms, VGPR 216 →
   192, occupancy 7 → **8 waves/SIMD** — two wins from one cut, because the
   register half crossed an occupancy cliff. The equivalence is a lemma, not a
   hope: in a k-way merge the k-th output is the k-th smallest of the union, so
   per-lane depth k is sufficient for a global top-k. Audit every top-k,
   candidate list and best-match set by asking what the *downstream* stage
   reads, not what the producer computes — the producer is almost always wider.
   Moving such a structure to shared memory instead is a trap; see "Respect the
   compiler's register tradeoffs" under Porting discipline.
6. **Unroll a rolled scan loop by hand, N candidates wide, and hoist the
   loads.** `#pragma unroll` is a no-op in glslc/GLSL (byte-identical SPIR-V),
   and ACO only unrolls where registers allow. When a loop body ends in a long
   serial chain — a sorted insert, a dependency-carrying reduction — the next
   iteration's *independent* loads sit behind it, and every resident wave
   reaches the same `s_waitcnt` at the same time, so occupancy cannot hide it.
   Issuing four candidates' loads before consuming any of them took BM3D's
   estimation kernel 4.94 → 3.96 ms. Sweep the width: 2/4/8 gave
   4.168/3.963/4.093 ms, because register pressure eventually wins.

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

- **The exec-pool frame path** (what every filter builds on now): record one
  command buffer per output frame from `gpuExecAcquire` /
  `gpuExecCommandBuffer`, declare inputs with `gpuExecReadsFrame` and outputs
  with `gpuExecWritesPlane`, then `gpuExecSubmit`; `gpuExecAbandon` on every
  error path before submit. The pool turns producer pairs into device-side
  waits, publishes the outputs' pairs, keeps the frames and any
  `gpu_frame_buffer` scratch alive until the submission completes, and sizes
  its own ring — the host never waits per frame. Model on the ported filters:
  stateless on gaussblur/bilateral/nnedi3, cached/temporal on bm3d/dfttest.
- **Cross-frame private caches** (BM3D's estimate stacks and source ring) stay
  possible without owning submissions: a slot carries a *submitted* ready flag
  instead of a timeline value (the pool allocates values at submit, so a writer
  cannot name one when it reserves), a reader waits host side until its writers'
  estimations are submitted, and a full `vkCmdPipelineBarrier` at the *start* of
  the reader's first command buffer supplies the execution and memory dependency
  — a pipeline barrier's first scope is every earlier command in submission
  order on that queue. Never signal or hand-roll the pool's timeline.
- **`env_flag` / `env_int` / `env_str`** — env-gated debug flags; keep one env
  name per filter (`VSFEEL_DFTTEST_TRACE`, `BM3D_TRACE`, ...). Read per-filter
  *diagnostic* flags through **`vsfeel_debug_flag(name)`** (one-shot: creation
  banners, fallback notices), **`vsfeel_debug_trace(name)`** (per-frame traces)
  or **`vsfeel_debug_probe(name)`** (measurement: host `TIMING`, GPU
  `GPUTRACE`/`TSTAMP`), never `env_flag`: `VSFEEL_DEBUG=1` — the one switch to
  hand a bug reporter: device banner, heap dump, creation banners, full error
  trace — turns the one-shot ones on, and `=2` adds the per-frame firehose and
  the probes, because a hang needs the kernel times as much as the trace.
  A level-2 run is instrumented, so never benchmark it. **A GPU-timing probe
  must also be gated on `GPUDevice::timestamp_valid_bits`** (via
  `vsfeel_probe_timestamps`): a `vkCmdWriteTimestamp2` is invalid usage on a queue
  family that reports 0, and a driver taking one anyway can hang the engine —
  a machine-wide freeze and bugcheck, not a lost device. Gate the *flag*, not
  just the query pool: a filter that writes timestamps off the flag alone would
  otherwise write with a null pool.
- **`vsfeel_trace_error(filter, frame, message, device)`** — every `set_error`
  lambda calls this first, so **one** switch (`VSFEEL_DEBUG=1`, or `VSFEEL_TRACE`
  alone: `=1`, `=2` for no line cap) names the filter, the output frame
  (`create` at filter creation) and the
  order of every error. (No `VK_EXT_device_fault` dump on this path: the core
  creates its device with no extensions.) The order is the point: a lost
  device makes every later
  call fail, so the `(first)` line is the only informative one. Add the call
  whenever a new error path is added — a filter that reports through
  `vsapi->mapSetError` directly bypasses it. Pass `d->gpu.get()`, which is
  null before device creation.
- **`vsfeel_trace_frame_begin()` / `vsfeel_trace_mark(stage)`** — the trail the
  first error dumps: call `frame_begin` when the frame path starts and mark each
  step *before* it runs (`acquire`, `record`, `submit`, ...), so the
  last mark names the step that failed. Marks are inert unless tracing is on;
  thread_local, because the frame path is synchronous per worker thread.

**ODR rule (learned the hard way):** a filter-local struct used to
instantiate a shared template must have a **unique name per filter**
(`Bm3dStream` — never `VK_Resource`). Same mangled name + different
`sizeof` across TUs lets the linker COMDAT-fold one TU's instantiation over
the others, mis-striding the pool's vector and corrupting in-flight
resources. (The template that bit us, `FramePool<T>`, is gone; the rule
stands for any shared template a filter instantiates with its own type.)

**What stays per-filter:** frame caches (temporal three: DFTTest slot cache,
NLMeans tile cache, BM3D estimate/source rings), shaders + launch config, sync
choreography (pad→copy→fused ordering, cross-frame ready flags), cache sizing.
Every filter now records through the core's exec pool: one context per
submission, never its own command pool, timeline or fence.

When porting a new filter, model the stateless path on gaussblur/bilateral
or nnedi3 (no cache) and the cached/temporal path on dfttest or bm3d.

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
  that does not move the fps median did not happen. Concurrent submissions
  share the compute queue, so per-kernel timings taken from a deep pipeline
  include the other frames' interleaved work — attribute kernels only in
  serialized traces.
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
  then silently leaves the stale `.so` installed). Build and install with
  `tools/install.sh` — one command that does both and does not exit until the
  installed copy's sha256 matches the build — then test in a separate command.
- **Prove the host/GPU split before optimizing anything.** Add a small
  env-gated chrono probe around the frame path (acquire / record / submit)
  and read it on real content first — kernel
  work that looks dominant from reading code is routinely not the bottleneck.
  Reset the stage clock after every blocking acquire
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
- **Minimize bytes moved, then minimize copies — and count reads per source
  byte, not just copies.** Transfer buffers in the narrowest
  exactly-representable type and widen on load (native u16 pad instead of f32
  halved EEDI3's pad-build and pad-read traffic at once). Then audit the
  frame for any buffer whose *same source bytes* are read by two different
  loops: merging a second pass into the first while the row is still cache-hot
  is nearly free and was worth +3.5% on EEDI3 while deleting 8.3 MB/frame of
  pure DRAM re-reads.
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
- **Sweep workgroup shape across workload configs, not just the default.**
  The best tile is a function of the algorithm's workload params (window
  radius, taps, halo overfetch), not a universal constant — a shape that
  ties at one config can win 50% at another and collapse at a third, so
  sweep the matrix (block candidates × representative configs) and re-verify
  under the shipped configuration, since overlap changes amplify or shrink
  shape effects. Where the matrix shows a clear workload-dependent winner, auto-
  select the default from the workload params (only when the user leaves the
  args unset — the `mapGetInt` error flag distinguishes explicit from
  default, and explicit args are always respected). Never inherit shapes
  across redesigns: a spill-free shape at one radius can spill at another,
  so check `shaderstats` (VGPR spill/scratch) per matrix cell.
- **Spec constants cannot size arrays in GLSL.** If an array dimension must
  vary, gate it with a compile-time `-D` define instead.
- **Respect the compiler's register tradeoffs.** ACO raises VGPRs deliberately
  for load ILP at an occupancy cost; forcing registers down often regresses.
  Read `RADV_DEBUG=shaderstats` before assuming more waves would help.
  **LDS is not a way out.** Three attempts to move kernel-live data into shared
  memory to free registers (a resolved match group, per-lane sorted lists in a
  bank-conflict-free layout, and loop-invariant centres) each *raised* VGPRs
  216 → 240 and dropped occupancy 7 → 6 waves/SIMD; two of them lost 5–90%.
  The mechanism is data-dependent addressing — extra address registers and
  longer live ranges — so spilling to shared only helps when the index is a
  compile-time constant. Diff VGPR *and* subgroups/SIMD on every kernel edit;
  measured occupancy on this box (wave32, RADV) is VGPR 192 → 8 subgroups/SIMD,
  216 → 7, 240 → 6, so one "small" register change is a whole wave.
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
  **An ablation that changes the data is not an ablation.** BM3D's
  `NOSEARCH=1` kill-switch also made every candidate tie, which suppressed the
  temporal search's insert branch, so its 0.741 ms could not be read as "the
  spatial search costs 2.1 ms". Prefer a **workload sweep** — vary the search
  radius and take the marginal ms per candidate — because that changes how much
  work happens without changing how the branches behave; the sweep is what
  actually localized BM3D's cost.
- **Keep `notes/<filter>.md` updated immediately** after every finding,
  including dead ends, so nothing is re-derived or retried later. The notes
  are **tracked**: they are the durable design record, visible to every checkout
  and the first thing a new contributor reads, so durable findings belong there
  rather than only in a code comment. **Read `notes/README.md` first and follow
  it** — it is the authority on the notes (section shape, style, and the line
  budget, which `notes/EEDI3.md` is already over), so nothing about how to write
  them is repeated here. Keep entries short regardless: conclusion first, a
  number only with its config, one line for a correctness-only change, mechanism
  not story.
  **Worked examples of most rules above live in `notes/EEDI3.md`
  rounds 10–12** (host-bounded frames, the SIMD dilation rewrite, the ReBAR
  upload, the boundary-input oracle bug, the harness-memory bimodality, and a
  list of measured non-wins) — worth reading before starting a new filter,
  even though none of it is EEDI3-specific.

## Building and installing

**Build with `tools/install.sh`. It builds *and* installs in a single command.**
There is no separate copy step and no reason to run `cmake` by hand — a
hand-built `libvsfeel.so` that was never copied into the plugin directory is
invisible to VapourSynth, so the tests and the benchmark quietly keep exercising
the previous binary.

```bash
tools/install.sh                 # configure + build + install, then hash-verify
tools/install.sh -h              # -b build dir, -p plugin dir, -c build type
```

That one command configures `build/` (Release) with CMake + `glslc` (the Vulkan
shader compiler), compiles the plugin, copies `libvsfeel.so` into
VapourSynth's plugin directory, and **fails unless the installed copy's sha256
equals the build's**. It asks `vapoursynth.get_plugin_dir()` where the running
Python loads plugins from, so a venv installs into the venv;
`VSFEEL_BUILD_DIR` and `VSFEEL_PLUGIN_DIR` override the autodetection. A run
whose plugin directory already matches the build is a no-op.

The bare CMake rules are only for a compile-only check that must not touch the
installed plugin:

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Never chain build → install → test into one shell command.** If the compile
fails, the previous `.so` stays installed and the test run silently "passes"
against a binary that does not contain your change. Build *and install* with
`tools/install.sh`, confirm the two hashes it prints match, then run the tests
or the benchmark as a separate command.

Vulkan headers (`Vulkan-Headers`) and the loader shim (`volk`) are pinned and
fetched by `FetchContent` at configure time, so no Vulkan SDK is needed to link
— only `glslc` from one. **Nothing links a Vulkan library**: `volkInitialize()`
dlopens the loader, which is what lets the Linux wheel be repaired to manylinux
(`libvulkan.so.1` is on no manylinux whitelist) and the Windows DLL need no
import library. Every TU includes `<volk.h>`, never `<vulkan/vulkan.h>`, so `vk*`
calls resolve to volk's function pointers; the plugin builds with
`-fvisibility=hidden` and exports only `VapourSynthPluginInit2`.

The SPIR-V shaders are compiled at build time and embedded into a generated C++
header (`spirv_binaries.h`) via `src/gen_spirv_header.py`. Adding a shader
variant is one line in the owning component's `VK_*_VARIANTS` table in
`CMakeLists.txt` — an `"<out>|<source>|<defs>"` entry naming the output, the
`.comp` file and every `-D` (including `--target-env`). `add_spv_variant()`
turns each entry into the `glslc` rule and collects the outputs into
`VK_SPV_OUTPUTS`, which is generated, not edited by hand. The header script
(`--out <header> <spv...>`) derives symbol names from the filenames
(`<stem>.spv` → `<stem>_spv` / `<stem>_spv_size`), so it never needs editing for
new variants.

### How the shader pipeline fits together

Each `src/*.comp` file holds several entry points selected by `-D` defines
(e.g. `-DENTRY_FUSED -DRADIUS=1 -DBITS=16`); a CMake `foreach` loop compiles
one `.spv` per variant-table entry into `build/vk_spv/`, and
`gen_spirv_header.py` packs them all into `spirv_binaries.h` as `uint32_t`
arrays. The C++ side picks the arrays it needs (usually per bit depth / radius
at filter creation) and creates one `VkShaderModule` + `VkPipeline` per
variant. So a new fast-path variant means: an `#if` block in the `.comp`, one
entry in that component's `VK_*_VARIANTS` table, and a module/pipeline pair in
the filter's creation function — nothing else.

Every variant rule also depends on a generated `<out>.spv.flags` stamp holding
that variant's `glslc` arguments. Ninja would rebuild on a flag-only change by
itself, but the default Makefiles generator does not: without the stamp a
changed `-D`, `--target-env` or `PROBE`/`MAXW` cache value leaves a stale
`.spv` in place and silently ships the old kernel (a stale probe kernel already
invalidated a round of measurements once). The stamp is written with
`file(GENERATE)` and content-hashed, so an unrelated `CMakeLists.txt` edit does
not recompile the shaders. Do not remove it.

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

Measured on the full suite when it was 443 tests: **~7.5 min cold → ~3.5 min for
the run that builds the cache → ~2.5 min warm**. The suite has since grown to
789 tests (522 s serial, ~2.5 min via `tools/test.sh` — see Testing). The first
run after a driver or `glslc` update pays the compile again by design.

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
3. Build and install with `tools/install.sh` (one command, hash-verified), then
   run the benchmark + tests to get a baseline.
4. **Measure the host/GPU split before optimizing** (the chrono probe), then
   run an ablation ladder (remove-all / remove-half) to find which side is
   actually the limiter. Do not assume it is the kernels.
5. Optimize / port, keeping each candidate behind an env opt-out so it can be
   A/B'd in situ, and re-measure afterwards. Rebuild and install with
   `tools/install.sh` (it does both), then re-measure.
6. Re-benchmark and re-test; keep going until vsfeel is faster than the
   references while still passing all tests.
