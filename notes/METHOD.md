# Method — notes

Method that is not one filter's: how to find where a reference kernel's lead
comes from, and the discipline that keeps a port honest. These rules are binding
the way a filter note is; start at `## Typical workflow` when porting a filter or
optimizing one.

## Typical workflow

The order to work in; the two sections below are the detail behind it.

1. Read the reference implementation for the filter in `reference/`.
2. Check the current vsfeel implementation and its tests.
3. `uv sync` once per checkout (and after dependency changes), then build and
   install with `tools/install.sh` and take a benchmark + test baseline; build,
   install and test are never one chained command.
4. **Measure the host/GPU split before optimizing** (the chrono probe), then run
   an ablation ladder (remove-all / remove-half) to find which side is actually
   the limiter. Do not assume it is the kernels.
5. Optimize / port, keeping each candidate behind an env opt-out so it can be
   A/B'd in situ; rebuild, install and re-measure after each candidate.
6. Re-benchmark and re-test; keep going until vsfeel is faster than the
   references while still passing all tests.

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
   compiler's register tradeoffs" under Porting discipline below.
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
  borders on such input).
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
- **An active display is a ~12% tax on every absolute number.** KWin compositing
  (plus any browser drawing on the same GPU) holds `gpu_busy_percent` at 12-18
  *at idle*, and a GPU-bound filter pays it straight off the top: the same binary,
  same batch and same command measured **192.5 fps with the monitor off and 171.3
  with it on**. That is the whole of a "the notes say 200, I see 175" gap, so read
  `gpu_busy_percent` before believing any absolute figure, and record the display
  state next to one. Same-session interleaved pairs stay fair through it.
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
- **A comparison subprocess that aborts with `free(): invalid pointer` after
  printing `RESULT` is MangoHud, not the filter.** Its overlay is an implicit
  Vulkan layer, and its own teardown thread aborts under high subprocess
  concurrency (measured 4/128 on the unmodified tree, 0/48 with `MANGOHUD=0`,
  backtrace inside `libMangoHud.so`); `tools/test.sh` now forces `MANGOHUD=0`.
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
  rather than only in a code comment. **Read `notes/AGENTS.md` first and follow
  it** — it is the authority on the notes (section shape, style, and the line
  budgets, which `tools/notes_check.py` enforces on every lint run), so nothing
  about how to write them is repeated here. (It is also injected automatically
  once you touch a file in `notes/`, but only after that tool call, so read it
  before writing a new note file.) Keep entries short regardless: conclusion
  first, a number only with its config, one line for a correctness-only change,
  mechanism not story.
  **Worked examples of most rules above live in `notes/EEDI3.md`
  rounds 10–12** (host-bounded frames, the SIMD dilation rewrite, the ReBAR
  upload, the boundary-input oracle bug, the harness-memory bimodality, and a
  list of measured non-wins) — worth reading before starting a new filter,
  even though none of it is EEDI3-specific.
