# BM3D — notes

**shipped.** Vulkan port of BM3D whose block matcher and collaborative filtering
are mawen's CPU V-BM3D (`reference/VapourSynth-BM3D`), so output differs from any
CUDA-matcher build. The CPU plugin is the only oracle, driven from
`tests/test_bm3d.py` (`bm3d.Basic`, `VBasic`/`VFinal` + `VAggregate`).

- **Agreement: exact except the CPU's tie order.** Reference-only groups
  (`th_mse=0`) agree to 2.4e-7. Over all 32400 reference blocks of frame 28885
  (1080p, r=0, bm_range 9, step 8) both matchers take the same key multiset, in
  the same order, with **zero** ranking differences; 2088 blocks differ only in
  which equally-distant candidate libstdc++'s `partial_sort` kept, the kernel's
  `(error, y, x)` order being stable. A stable-sort reference build lands 4.4x
  closer on that frame (mean 5.15e-5 -> 1.18e-5), which prices the tie order.
- The SSD is the CPU's accumulation order but not its rounding: ACO fuses the
  multiply-adds where the SSE path rounds every product, so ~1/3 of candidates
  differ by 1 ulp.
- The matcher is graded coordinate-for-coordinate against a scalar model of the
  CPU source (`tests/bm3d_oracle.py` + `VSFEEL_BM3D_MATCHTRACE`), 26 configs;
  its SSD reduction now matches the per-row lane order above.
- Every plane of a color clip is denoised: one entry per plane, or one joint
  4:4:4 entry under `chroma=True` whose groups come from luma. A plane below its
  sigma threshold is a bit-exact source copy; all planes below it and the filter
  is the source clip.
- `th_mse` (8-bit MSE, inclusive, 0 = reference only) defaults to
  `sigma[0]*80 + 400` basic and `sigma[0]*10 + 200` final.

Scoreboard — 1080p jpbd, `tools/benchmark.py -f bm3d`, harness defaults
(sigma 0.7, radius 2, bm_range 9, ps_range 4, block_step 8), 1000 frames x3,
default (resident GPU input, no output download):

| arm | fps | |
|---|---|---|
| vsfeel | **1022** | mawen matcher, per-frame accumulators |
| vsfeel, 16 bit in | **1070** | same kernels, half the io bytes |
| vsfeel, f16 in | **1071** | same kernels, half the io bytes |
| bm3dvk | 263 | fixed 8-member groups |
| bm3dvk2 | 883 | |
| vszipcl | 109 | |

- **1.16x the faster reference**, but not on equal work: they always group eight
  while the default threshold accepts ~80% of candidates here and the top-eight
  selection does the filtering.
- bm3dvk2 is the close one now: its per-frame accumulators (the design this
  round copied) are what took it from 146 to ~880; its fixed eight-member groups
  still do less filtering work than the thresholded groups here.
- The 16 bit and f16 rows are separate runs at the same settings, not second
  filters.

u16 and f16 input are the same filter, not two more: the ring, the estimate
stacks and every kernel's arithmetic are float at every depth, a 2-byte clip's
samples are widened once when copied into the ring (an integer sample by the
reference's `Int2Float` gain and chroma neutral, a half sample by an exact
widening with neither), and the aggregation stores back to the native sample
type. A 16 bit chain therefore drops the caller's `depth()` node and uploads
half the bytes, which is where the visible speedup is (see Performance).

## Implementation

- Three kernels: `bm3d.comp` (match, group, transform; one workgroup of WG
  threads = WG/8 independent 8-lane groups, one 8x8 block each), `bm3d_agg.comp`
  (normalize: the box-summed per-block denominator) and `bm3d_copy.comp`, which
  widens a 2-byte clip's planes into the ring as it copies them: an integer
  sample by gain 1/65535 less the 32768 neutral on YUV chroma (the reference's
  `Int2Float`), a half sample by an exact widening with neither. Only the store
  is depth-specific (`-DIO_U16`/`-DIO_F16`/`-DIO_F32`): the u16 arm is the
  reference's `Float2Int` with the clamp that keeps a weight sum a hair over 1.0
  from wrapping, the f16 arm narrows with the SPIR-V default round-to-nearest.
- The f16 arm is bit-for-bit the f32 arm up to that store: on the losslessly
  widened clip the measured worst against the f32 path is 2.85e-4 (one half
  step), 99.99% of pixels rounding identically, the rest differing only where the
  aggregation's ~3e-8 jitter crosses a half tie. That equivalence plus the CPU
  reference run on the widened clip (it refuses a half input) is the oracle.
- A u16 clip follows its frame's **`_Range`** like the CPU reference: full maps
  the whole code range, limited 219/224 with 4096 luma and 32768 chroma offsets;
  an f16 clip is already in the reference's float domain.
- One entry per processed plane, or one over all three planes of a 4:4:4 clip
  under `chroma=True`. An entry owns a source ring `[slot][clip][plane][h][stride]`
  and an accumulator ring `[slot][plane][2][h][stride]` (numerator, denominator);
  its planes share one geometry, so the plane stride is a spec constant.
  NPLANES folds to 1 in the per-plane mode.
- Per-plane `sigma`, `block_step`, `bm_range`, `ps_num`, `ps_range` follow the
  references' inheritance rule; a joint entry takes plane 0's search parameters
  and only the sigmas differ.
- The reference block is group index 0, excluded from the current-frame scan by
  origin. A neighbouring frame is searched over the union of the clipped
  predictive windows around the surviving seeds, each origin once, by
  first-window ownership (two comparisons per earlier window: every window shares
  `PS_RANGE` and the candidate is in-image). Deduplicating before the SSD is what
  the CPU's `GenSearchPos`/`std::unique` does.
- A frame keeps its best eight qualifying candidates in `(error, y, x)` order
  (the CPU's stable sort of a `(y, x)` scan); only its first `ps_num` seed the
  next frame, and both temporal directions restart from the current frame's
  seeds. Candidates are partitioned across lanes for a direct per-candidate SSD
  against the same fixed reference patch -- shifted-column reuse is invalid --
  and each subgroup merges its own top-K.
- The merge makes the selection exactly the global top-K by `(error, y, x)` (a
  dropped entry is below its lane's K-th, hence below the union's K-th), so it
  is the **stable** top-K, which the CPU's `std::partial_sort` is not. It is a
  three-round `subgroupShuffleXor` tournament over the lanes' heads: all eight
  lanes walk it, so every lane ends up with identical copies and no LDS round
  trip, head table or 64-entry scan is needed.
- Per-lane candidate lists are 8 deep in **registers**, one sorted list per lane
  (`pe`/`pxy`), updated by a single compare-exchange per slot. The prediction
  seeds (`s_x`/`s_y`, plus `s0_*` for the forward restart) live in shared memory
  because every lane of a group reads all of them (a register array indexed by a
  runtime count would spill); the earlier windows' seeds are read into registers
  once per window, not per candidate. Eight is the floor: a neighbour's third
  through eighth candidates join the final group.
- The combined group list (`ge`/`gxy`/`gz`) is the one private array left whose
  index is not a compile-time constant (the tail sort's `b`, the insert's `pos`).
  It stays per-lane: a shared copy measured 2.6x slower on Ada (see Historical).
- The group's reference patch (the 64 floats the SSD subtracts from) is either
  sixteen `vec4` locals per lane or a row-major shared copy, and the choice keys
  off the **driver**, not the vendor: registers are only right if the backend
  promotes them, else every one of the SSD's 64 reads per candidate is a memory
  access (RADV/ACO promotes the array: 168 VGPRs, zero scratch). Shared memory is
  the default everywhere except `VK_DRIVER_ID_MESA_RADV`, where it measures 8-10%
  worse, and it is what makes the kernel run at all on a backend that will not
  promote the register arm (Ada; see Historical for the alignment and ISA detail).
- Each walk evaluates one candidate per iteration in place (skip the reference
  origin, SSD, threshold, insert) with scalar coordinates; the row wrap is a
  `while` because the 8-lane step can overshoot a window narrower than 8 (the
  unroll's cost is under Performance).
- The group keeps concatenation order until it overflows past eight, and only
  then keeps the reference plus the best seven of the tail. Each frame's list is
  consumed best-first and an overflowing group's tail is sorted, so the loop
  stops at the first entry that cannot beat the tail's worst; the rest cannot
  either.
- The group-axis transform is a length-N scaled DCT-II (generated tables for
  N = 1..7, the fast butterfly at N = 8) with `A^T A = 2N I`: threshold and
  Wiener shrinkage use `sigma * sqrt(N/8)` and inverse gain `1/(512N)`, and count
  only the 64N real coefficients. Unused members are zeroed, never addressed.
- Filtering follows the CPU: strict `|c| > thr`, no exempt coefficient, weight
  one when every coefficient was thresholded away, Wiener
  `ref^2/(ref^2 + sigma^2)` per coefficient with its `den > 0` fallback, and
  `1/max(FLT_EPSILON, sum coeff^2)`. Sigma factors are the CPU's
  `2.7*normY*64/255` and `normY*64/255` (normY = bt709 luma-row norm
  0.7496149945138504, which `th_mse` carries too); in this kernel's `A = D^-1 F`
  domain the CPU's per-DC-axis threshold collapses to a uniform one (pinned by a
  test).
- A centre frame searches only real frames (`zlo = max(0, r-m)`,
  `zhi = min(2r, nframes-1-m+r)`). Each group member's filtered patch is
  scattered into the frame it was taken from, so an output frame's accumulator
  receives one contribution per centre whose group holds a block in it. The
  accumulator is a numerator plane plus a denominator plane per entry plane;
  the denominator is one weight per block, added at the block's top-left.
- `bm3d_agg.comp` is now a normalize: for each pixel the sum of the block
  weights in the `BLOCK x BLOCK` origins `[p-BLOCK+1, p]` is a separable box
  sum from a shared tile, and the numerator is divided by it. This halves the
  accumulation atomics (one numerator add per pixel, one denominator add per
  block instead of two per pixel) and removes the per-centre 2r+1 slice stack.
- Temporal order is a cursor: process the next centre, scatter it, then
  normalize every frame whose last contributor (`f + r`) is in. The pipeline is
  one command buffer per request, not one submission per centre. A request the
  cursor cannot reach (a seek, or a frame it has passed) is computed in a
  private one-window ring with `focus` set: only members from the requested
  frame are scattered, so the main ring keeps its state and the cursor stays
  monotonic. That is what keeps vspipe's out-of-order prefetch from paying a
  window restart per jump.
- The accumulator ring is `2r+1` slots (capped by the clip length), keyed by
  frame index; a slot is zeroed when it is first assigned and reused only after
  its frame retired. Radius 0 keeps its private per-frame slots.
- Estimation accumulates with hardware buffer float atomics wherever the device
  reports `shaderBufferFloat32AtomicAdd` (`VK_EXT_shader_atomic_float`; the plain
  load/store/exchange bit is not enough), else the reference's `atom_add_f` CAS
  loop (`-DNO_FLOAT_ATOMICS`; `VSFEEL_BM3D_CAS=1` forces it). Both arms declare
  the accumulation buffer `coherent`. No cheaper fallback exists without the bit:
  shared-memory atomics cannot help, a group's patches land anywhere.
- Radius accepts up to 16. The accumulator ring is `2r+1` slots of two planes,
  so 1080p at radius 16 costs ~550 MiB per entry instead of the old
  `(2+2r)(2r+1)` plane pairs that refused radius 8. The source ring is
  `2r+3` slots (one search window plus the frame the next centre adds), and the
  isolated path gets its own copy of it. A device's `maxStorageBufferRange` is
  still checked; every refusal names the size.
- Degenerate paths: `sigma < FLT_EPSILON` copies a plane through, and with no
  plane above it the filter is not built (the clip is handed back). Validation
  runs before that shortcut: `extractor_exp` `[0, 127]` as bm3dvk does, `th_mse`
  finite and non-negative, saturated to `FLT_MAX` when it does not fit a float
  and raised to `FLT_TRUE_MIN` when a positive value underflows (spec constant
  17).
- Everything records through the core's exec pool; the filter owns no command
  pool, timeline or fence. The temporal path records one centre's scatter, its
  normalizes and the source copies into one command buffer and submits once per
  request; the pool overlaps that with the next request. Radius 0 still records
  one submission per frame (`VSFEEL_BM3D_SPLIT=0` collapses its chunks).
- Cross-frame handoff inside a request is a full pipeline barrier: the normalize
  reads the atomics the scatter just wrote, and the next centre's ring copy runs
  after the previous centre's reads. Across submissions the first barrier covers
  everything already queued on the one compute queue. The old per-slot
  submitted-ready handoff remains for radius 0; the temporal path no longer
  waits host side.
- R80 GPU API: `clip:vnode:gpu` in and out with `ffGPUOutput`; an unprocessed
  plane is shared from the centre frame and only planes the aggregation writes are
  published. `num_streams` and `device_id` are registered no-ops. The 8-lane
  transposes and the group-8 reduction are `subgroupShuffleXor` butterflies
  (register-only, no LDS, no barrier) reproducing the reference's tree.
- `extractor_exp`'s `(x + E) - E` pre-rounding must stay under GLSL `precise`:
  RADV folds the pair to `x` when E is a known spec constant.

## Performance

- **Per-frame accumulators: 868 -> 1022 fps (+18%).** The old path stored a
  `2r+1`-slice stack per centre and re-aggregated it with two atomics per pixel;
  the new one scatters into the target frame, adds the weight once per block and
  normalizes with a box sum. Harness A/B (default, 1000 frames x3): slice 867.7,
  accumulators 922.9 before the cursor was finished,
  1022-1028 after the isolated ring removed the restarts (median of five, 2%).
  Kernel table at 1022 fps: blockMatch+filter 0.86 ms, normalize 0.044, convert
  0.018, zero 0.011, and 1000 estimation dispatches for 1000 frames.
- **The candidate bookkeeping, not the SSD, was the cost.** A per-candidate
  ablation ladder on the estimation kernel (fps, 1080p jpbd, harness config)
  prices the pieces as: search ~29% of the frame, estimate/transform 8.7%,
  aggregation ~13%, slot zero-fill ~2%. The load-bearing discovery is that every
  piece of per-candidate code is if-converted: the warp pays the insert's ~80
  instructions whenever *any* lane passes the gate, so per-lane gating buys
  nothing. `gpu_busy_percent` is 100% through a default-config run (~10% during
  the harness's cache preload, what a mistimed poll sees), so kernel work pays.
- The bookkeeping round (1000 frames x3): **845 -> 952 fps**. The ownership
  seeds went to registers (8 LDS loads with a `lgkmcnt(0)` drain each), the
  merge became a shuffle tournament, the candidate lists left LDS for registers,
  and `group_add` stops at the first entry that cannot beat an overflowed
  group's worst.
- The SSD's 1-ulp divergence is deliberate: `precise` costs ~8 ALU ops per row
  to buy back ~10% of the tie floor.
- The estimate phase's 0.40 ms used to be transform plus two `atomicAdd`s; the
  CAS arm's +0.66 ms/frame bounded the atomics as most of it. The denominator
  round removed half of them (one weight add per block, not per pixel), which is
  most of the +18%.
- Timestamp figures before 2026-09-21 read 1000x low; do not quote them.
- **The run-to-run swing was pipeline depth, not the kernels (+5%).** Two
  in-flight frames serialized the source-ring reuse distance (a third blocked in
  `acquire_cache`), putting the host at the GPU's pace: 871-947 fps (8.7%);
  sizing for the pool's four frames gave 933-960 (2.8%). Depth 6-8 add <1%.
- **Do not unroll the walk.** The loop body is 309 instructions per candidate
  (SSD 45%, insert 27%, addressing 9%, waits 6%, scheduler `s_delay_alu` 14%),
  and several candidates per iteration only raise the register count: 1-wide is
  168 VGPRs / 9 waves/SIMD, 2-wide 192 / 8, 4-wide 192 / 8; fine-interleaved A/B
  puts 1-wide 1.6% ahead of 2-wide and 2.7% ahead of 4-wide. The earlier "the
  unroll helps" readings were against baselines that had drifted (Method rules).
- **A u16 ring is a pessimisation; widen at copy time (+3%).** The SSD re-reads
  each source sample once per candidate origin out of an L1-resident window, so
  halving the bytes saves no DRAM traffic while the load path doubles (0.95 ->
  1.27 ms). Widening once at copy time: 970 against 932 fps (`--format u16`),
  and a real 16 bit chain 846 vs 634.
- **f16 is the same io-only win, inside the noise at the benchmark config.**
  f16 972.4-977.9, f32 959.7-972.8, u16 969.7-973.5 (1000 frames x3). The ring
  and the accumulators stay f32, so a half chain's win is dropping the caller's
  `depth()` node and halving its upload/download.
- Other denoisers gain more from u16 because they stream each byte once or twice
  with a few ALU ops per tap (GaussBlur 2284 vs 1093, DFTTest 1132 vs 911).
  BM3D's matcher is the opposite end, so only its io can narrow.
- Marginal cost per candidate per lane: 0.0032 ms (`bm_range`), 0.0030
  (`ps_range`), with a ~1.4-2.0 ms intercept at tiny windows.
- **The estimation launch shape comes from the device, not a fixed warp.** It is
  a specialization constant; `choose_workgroup` starts at two waves (64 threads
  at subgroup 32) since a one-wave CTA can be capped by the CU's workgroup slots,
  then clamps by invocations, subgroup budget and per-group LDS. Seeds and patch
  use a subgroup barrier (no group reads another's). RDNA3 gains little (the
  register file binds first): 64 against 32 threads +0.6% (805.2/800.7, ABBA,
  jpbd, 1000x3), 64 against 128 +0.9%; `VSFEEL_BM3D_WORKGROUP` overrides.
- **LDS is a trap here.** Three attempts to move kernel-live data into shared
  memory each raised VGPRs 216 -> 240 and dropped occupancy 7 -> 6 waves/SIMD,
  losing 5-90%: the cost is dynamic LDS addressing, not traffic, so the wins came
  from reducing the data instead. Two later attempts confirmed it: the reference
  patch in shared memory cost 8-10% (ABBA, 1000 frames x3, n=12: 952.8 against
  879.1 fps), and the group arrays (`g_e`/`g_xy`/`g_z`, 24 registers) 5.8% even
  though they cut VGPRs 192 -> 144 (8 -> 10 waves) and code 118 -> 80 KB: the
  group assembly is a serial chain of dynamic-index accesses, where LDS latency
  costs more than the occupancy buys. On Ada the same move is worse still (2.6x
  for the group list; see Historical).
- Color costs the planes: 4:2:0 is 1.5x the luma-only estimate/source footprint
  and 4:4:4/RGB 3x, all inside the same `maxStorageBufferRange` budget.
- The multi-plane round did not move the luma path (NPLANES == 1: 6131 vs 6133
  instructions, 192/108/4096 B, 8 waves/SIMD; 12 ABBA pairs at 935 vs 924 fps).

## Historical

Chronological; each entry keeps the mechanism, not the story.

- **2026-10-19 — batched and shared-buffer candidate selection; removed.** Two
  selection passes were tried against the per-candidate insert. The register
  batch (eight candidates per lane, 19-comparator odd-even sort, 20-comparator
  bitonic merge) costs 24 VGPRs and one wave on RADV: 987 against 1071 fps (A/B,
  1000 frames x5, jpbd 1080p f32, radius 2, bm_range 9, ps_range 4, block_step
  8, sigma 0.7, default, n=10). On a 3080 it runs at **5
  fps** against the insert's ~240, the private-array staging that costs the
  patch and group-list arms their Ada runs: the batch's 16-element distance/key
  arrays are passed to `sort_batch`/`merge_batch` and the path adds 24 VGPRs. The shared-buffer variant (running top-8 in LDS, 48-candidate
  chunks, eight selection rounds, two-entry cache) holds 168 VGPRs but costs
  803 against 1152 fps on RADV (A/B, 1000 frames x3, same config, `--cpu-cache`,
  n=6; 769 against 1141 with `--download`), because the four 8-lane groups of a
  subgroup must enter every pass together and the per-chunk LDS scans, barriers
  and shuffle rounds dominate. The insert is the selection on both vendors; no
  change shipped.

- **2026-10-19 — per-frame accumulators replace the slice stack.** The estimate
  kernel scatters each group member into the frame it came from instead of into
  the centre's `2r+1`-slice stack: one numerator add per pixel and one weight
  add per block at the block origin. `bm3d_agg.comp` became a normalize whose
  denominator is a separable `BLOCK x BLOCK` box sum of those block weights.
  The temporal order moved from a per-centre cache to a cursor (one command
  buffer per request), with an isolated one-window ring (`focus`) for requests
  the cursor cannot reach, so vspipe's prefetch jumps never move it backward.
  Measured 867.7 -> 1022 fps on the harness graph; the accumulator ring is
  `2r+1` slots instead of `(2+2r)(2r+1)` plane pairs, so 1080p radius 16 now
  fits. The old radius-0 path keeps its private slots and its ready-flag
  handoff; the temporal cache tables are no longer used.

- **2026-10-18 — f16 io ships as the u16 transport with an exact widening.** A
  new `bm3d_copy_f16` variant widens `float16_t` planes into the f32 ring with
  one OpFConvert and no gain or chroma neutral, and `bm3d_agg_f16` narrows the
  aggregated result back with the SPIR-V default round-to-nearest-even. Nothing
  else changes: the estimation, the ring and the estimate stacks are shared with
  the other depths. Graded device-independent against the f32 path on the
  losslessly widened clip (worst 2.85e-4, one half step) and against the CPU
  plugin on that same widened clip (it refuses a half input), with the u16
  suite's radius/sweep/final/color/config coverage mirrored. Perf is io-only and
  inside the noise at the benchmark config, as the u16 round predicted; the
  delivered value is a half chain with no `depth()` node. No change to the u16
  or f32 paths.

- **2026-10-10 — item-based row reuse in the search walk; removed.** One lane
  owned 2 or 4 vertically adjacent candidates in a column, so a source row was
  loaded once for all of them (64 -> 36 -> 22 loads per candidate, bm3dvk2's
  `ADD_ROW`/`DEFINE_DISTANCES` re-expressed). It needed the shared patch, since
  the walk reads the patch by a runtime row, and it composed with the split.
  Nvidia screens against a 285 default: 290 at 2 rows, 277 at 4, and about the
  same or a little slower stacked on the split, so the loads it saves are not
  Ada's limiter. RADV is the other way round and equally small (+1.0% at 4,
  -0.8% at 2; ABBA, 800 frames x3, n=12). Keep the codegen finding: at 4 rows one
  SSD term rounded differently from the per-candidate walk and flipped a near-tie
  (two same-origin candidates in different frames with identical errors swapped
  in the group, 26 output codes apart), because ACO contracts `a + d * d` into an
  FMA per site as register pressure allows. `tests/bm3d_oracle.py` models the
  CPU's two roundings, so `precise` is the fix and `fma()` is not, and a future
  walk over the SSD has to pin its rounding rather than match the shipped arms.
- **2026-10-10 — the matcher/filter split is the default off RADV, -13% on it.**
  `-DSPLIT_MATCH`/`-DSPLIT_FILTER` run the search and the estimate as two
  dispatches with the group handed over through a buffer. The premise was measured
  first: with the patch staged, the matcher alone is 96 VGPRs and 16 waves/SIMD
  against the fused kernel's 120 and 12, and 42 KB of ISA against 109 KB. **The
  two vendors split**: an Nvidia screen reads 285 default against ~310 split
  (+8.8%, which is the 300 fps target met), and RADV ABBA A/B (800 frames x3,
  n=12) is 950.5 fused against 824.9 split, -13.2%. RADV is not occupancy-bound
  there and the extra waves buy nothing: with the estimate made a no-op the split
  is *still* 3.3% slower (1057 against 1023), and the rest of the loss is the
  estimate re-reading the group the fused kernel still had in registers.
  Narrowing the handoff barrier from ALL_COMMANDS to compute->compute changes
  nothing. It now follows the same driver rule as the reference patch
  (`VSFEEL_BM3D_SPLIT_KERNEL=0/1` overrides); the full suite passes on both paths.
- **2026-10-09 — force-unrolling the fixed-trip loops is a dead end; reverted.**
  `[[unroll]]` does work, unlike `#pragma unroll`: it removes the `OpLoopMerge`,
  and it takes the shipped SPIR-V from 132 loops and 25 runtime-indexed
  `float[64]` locals to 36 loops and none, which is the codegen hazard it was
  after. It buys nothing for that: RADV is 898.5 against 896.5 fps (ABBA, 800
  frames x3, n=12, -0.2%, where ACO already unrolled) and a 3080 is 205-240
  against its ~265, inside that card's spread. What it costs is build time: the
  module grows 135 -> 576 KB per variant, the ACO ISA 107.7 -> 136.8 KB and
  `libvsfeel.so` 21.5 -> 24.9 MB, and `glslc -O` over 4.3x the SPIR-V pushed the
  wheel builds past their timeouts (that round also carried build serialization
  and dropped `-O` on the bm3d variants, reverted with it). A private array a
  backend will not promote is not enough reason to force the frontend to unroll.
- **2026-10-09 — the group list in shared memory is a 2.6x regression on Ada;
  reverted.** The list's indices are the only runtime ones left, and the ptxas
  model above stages the per-lane arrays in local memory, so the shared arm
  looked free. It is not: on the 4070, with the patch already shared, moving the
  group list to LDS takes that card's default arm from 232.9 to 90.4 fps. RADV is
  indifferent to the group list itself (861 against 854 at `PATCH_LDS=1`), so the
  cost is that backend's, not the access pattern's. A private array in local
  memory is not by itself a reason to move a datum to shared memory; the model
  proposed and the in-situ A/B disposed. The same report shows the `vec4`
  register arm is a 16% regression on Ada against the `float cur[64]` it replaced
  (193 against 229.7), which is latent while the shared arm is the Nvidia default.
- **2026-10-09 — the shared patch got vector reads and the register arm stopped
  being an array.** The shared copy is row-major now, so a patch row is 16-byte
  aligned and the SSD's eight loads per row become two `ds_load_b64` instead of
  eight `ds_load_b32` (ISA-verified; the source loads are already
  `buffer_load_b128` on ACO either way). The register arm holds the patch in
  sixteen `vec4` locals instead of `float cur[64]`, which is the same 64 values
  with no private array left for a backend to stage: ACO's stats are unchanged
  (168 VGPRs, 0 scratch), the full suite passes against the CPU, and the 7900XTX
  A/B is 952.8 against 879.1 fps (ABBA, 1000 frames x3, n=12), i.e. the shared
  arm's cost is 8-10%, not the 3.5% once recorded. The vectors are the arm to
  re-test on Nvidia: they read the patch from registers, which is what vszipcu's
  350 fps does there.
- **2026-10-08 — the reference patch moved to shared memory where the backend
  will not promote it (60 -> 200 fps).** With the patch in registers the 3080 ran
  the estimation kernel at 60 fps; one shared row per group took it to 200 (GRAYS
  1080p, r=2, bm_range 9, ps_range 4, block_step 8). The mechanism is codegen,
  not memory: the patch is a statically indexed private array, so the arm that
  keeps it in registers is only available on a backend that promotes it, and the
  gate is the driver id rather than the vendor.
- **2026-10-08 — the Nvidia slowdown was struct staging in the search walk.** The
  V-BM3D matcher's two walk loops stage state that Nvidia's backend will not
  promote: the four candidate coordinates (`int c[4]` / `int r[4]`) and, worse,
  `Window`/`Walk` structs crossing `make_window`/`walk_start`/`walk_step`, which
  run once per candidate. `walk_step` now takes scalars and the window fields are
  hoisted into scalars before the loop, so the per-candidate path never touches
  a compound object (`Walk` locals 5 -> 0; the fast arm has none of either). The
  op count had gone *down* (`insert_cand` beats `insertN`), so it was never work:
  0dad9c8 127.3 fps against f36a6ef 1.22 fps, and 449.5 against 240.4 under
  `VSFEEL_BM3D_NOSEARCH=1` (3080, GRAYS 1080p, r=2, bm_range 9, ps_range 4,
  block_step 8): the search alone went 5.6 -> 815 ms/frame. Dead end first tried:
  scalarising only the `c[4]`/`r[4]` arrays left the structs, and the explicit
  4-wide call sites multiplied the inlined `Walk` copies 5 -> 11, taking the 4070
  from 688 to 893 ms. The 7900XTX A/B is 969.2 against 951.1 fps (jpbd 1080p,
  1000 frames, ABBA, n=12), which is only the no-regression check: ACO promotes
  these there, so the fix's own validation has to come from the affected cards.
- **2026-10-08 — the RX 580 TDR watch is closed.** The card has no Vulkan 1.4
  on Windows, so nothing after the R80 port runs on it to verify against; the
  per-position estimation submissions stay as the bound.
- **2026-10-05 — the match trace's record and its recording.** `kTraceWords`
  counted five header words for the marker plus the five counts and allocated
  101, but the kernel writes index 5 and, at radius 16, its last temporal word is
  index 101: the buffer and the host mapping were a word short (the new
  radius-16 case reads 0 where the oracle wants 2 with the old size). Tracing
  also required the *output* frame to be the traced one, so a sequential load
  (frame - radius estimates the centre first, the traced frame then reads it from
  the cache) recorded nothing and the dumper reported "no group matched";
  whichever frame computes the traced centre carries it now. The colour parallel
  tests also trim through `source_clip()`, so they use the fixtures' 24 frames
  instead of the file's 300. No perf change.
- **2026-10-05 — integer chroma: the neutral code and a null binding.** A YUV
  chroma plane is centred on zero in the reference's float domain, so the copy
  now subtracts 32768 and the store adds it back; the u16 arm had been filtering
  a plane offset by 0.5, which showed as up to 30 codes against the f32 arm on
  360p YUV444P16 (now <= 1, the store's jitter; r=2, frames 0/11/23).
  Chroma-only `sigma` also left plane 0 unprocessed, so the estimation kernel's
  unused destination binding was null, invalid without `nullDescriptor`; it now
  binds a processed plane's output. No perf change.
- **2026-10-05 — the ring copy's grid folds into X and Y.** A 4096x4096 plane is
  65536 workgroups, past the X limit Vulkan only guarantees; `bm3d_copy.comp`
  now indexes `ID.y * NumWorkGroups.x + ID.x` and the host folds with
  `vsfeel_fold_grid`, per entry. No perf change on this GPU (X is unconstrained).
- **2026-10-03 — u16 input ships as a f32 ring widened at copy time.** The
  u16-ring build and the mechanism behind its 15% loss are under Performance.
  Three kernels now (`bm3d_copy.comp` is new), `-DIO_U16`/`-DIO_F32` only on the
  aggregation's store, and the reference's 16 bit output is the oracle
  (`VAggregate(sample=INTEGER)`, `Basic` at radius 0). The u16 tests pin that
  output to the f32 one rounded, within the aggregation's one-code jitter, over
  five configurations.
- **2026-10-03 — the search's bookkeeping was the whole cost (+12%).** The
  matcher's per-candidate work is dominated by code that if-converts, so the
  fix was to shrink it, not to gate it:
  - the union's ownership test read `s_x`/`s_y` per candidate, 8 LDS loads each
    followed by `s_waitcnt lgkmcnt(0)`; the seeds now live in registers (seed 0
    is enough for `PS_NUM == 2`, since `ns <= PS_NUM`, so the rest folds) and
    the per-window loop is rolled;
  - `merge_group`'s 8x8 head-table scan (measured at ~2300 instructions per
    call from the ISA, ~4600 per call site) became a three-round
    `subgroupShuffleXor` tournament that leaves identical copies in every lane,
    so the per-lane lists no longer need an LDS round trip, a barrier, or the
    `l_e`/`l_xy` arrays at all;
  - `insert_cand`'s position scan plus two-array shift became one compare-
    exchange per slot;
  - `group_add` bails out of a frame's list once an entry cannot beat the
    overflowing group's worst (the list is ascending, so the rest cannot
    either).
  Each is exact: the merged order is still `(error, y, x)`, the seeds and
  `count` are unchanged, and the full BM3D test suite passes.
- **2026-10-03 — the walk unroll was a mis-read, and 1-wide is the shape that
  ships.** Unrolling looked like a win against baselines that had drifted
  between builds (an ABBA whose arm order tracked a monotonic drift, and later
  an A/B arm that `git show HEAD:src/bm3d.comp` had quietly turned into a copy
  of the optimized kernel once the work was committed). Re-measured with the
  unroll as the *only* difference and the arms fine-interleaved in one binary:
  1-wide 168 VGPRs/9 waves beats 2-wide 192/8 by 1.6% and 4-wide 192/8 by 2.7%.
  The final verification, 1-wide production vs the pre-round kernel, 8 samples
  per arm: 877 vs 803 fps, no overlap.
- **2026-10-03 — not taken.** Reordering the group-key compare to drop the `xy`
  tie-break (invalid for the temporal walk, whose union enumeration is not in
  scan order); dropping the layout-restoring transposes from the collaborative
  filter (it moves the Wiener `sum coeff^2` grouping to different coefficients,
  a rounding change for ~2% of the frame); moving the reference patch or the
  group arrays to LDS (see Performance).

- **2026-10-02 — the CPU oracle was a stale build; the residual is the tie
  order.** PyPI's 10.1 predates `reference/` (upstream force-pushed the
  September commits away, so the wheel lacks the per-coefficient Wiener variance
  and zero-distance matches), which is what every unexplained CPU-comparison
  residual came from. The dev group pins that commit; the bounds are re-measured
  against it (~2-3x tighter, the Wiener pass 2e-3 -> 5e-4 at 4.7e-5..8.9e-5),
  and the same run prices the rest as `partial_sort`'s tie order. No kernel
  change, no perf change.
- **2026-10-02 — two clip-change failures were test bugs.** `chroma_planes_are_denoised`
  built its YUV clip from the untrimmed 300-frame source while the Gray fixture
  is 24, so frame 23 saw a different temporal window (`source_clip()` now). The
  `cas_fallback` bound was an absolute 1e-7 calibrated on near-black pixels; both
  arms are scheduler-order accumulations of 13448 addends, so the floor scales
  with the value (4.2e-7 at |max| ~ 1) and is now a full-scale-relative 1e-5,
  7x under a dropped addend. No perf change.
- **2026-10-02 — filtering becomes mawen's.** The kernel kept vszipcl's
  conventions (non-strict threshold, global-DC exemption, sigma rounded to
  `lambda * 0.75`), which left a 0.0197 speckle floor against the CPU even with
  single-member groups; the ported convention leaves 4e-7. No perf change.
- **2026-10-02 — the matcher becomes mawen's, groups become variable length.**
  The CUDA matcher scored an origin once per window covering it (so it could take
  two group slots), kept only `ps_num` results per frame, had no threshold and
  read clamped endpoint frames. All four are gone; ACO and throughput did not
  move (192 VGPR, 0 spills, 3072 B LDS, 8 waves/SIMD; 790 vs 798 fps).
- **2026-10-02 — every plane is denoised (chroma was a passthrough).** One entry
  per plane, per-plane parameters, the joint 4:4:4 entry, RGB input, the
  `[0, 127]` extractor range and the all-zero shortcut landed together. The cache
  reservation is atomic across entries: reserving one at a time deadlocked ~40%
  of color ref-pass comparisons (a frame holding entry 0 while waiting for
  entry 1, against the mirror image).
- **2026-10-02 — radius cap 4 -> 16 and one aggregation path.** The cap was the
  per-slice table in push constants (three tw-wide int arrays stop fitting the
  guaranteed 128 B at `tw = 11`), not VRAM. A kernel-derived table agreed with
  the host's within 3.0e-8 and at parity in an ABBA A/B, so the host table went.
- **2026-09-26 — the handoff barrier only ordered writes before reads**, leaving
  the write-after-write half of a slot reuse without a memory dependency. Both
  access scopes now cover both directions. No perf change.
- **2026-09-25 — the aggregation trusted slices that were not this frame's.** A
  zero-filled slot divided by zero (black band), a partly accumulated one
  averaged short (dim band), a recycled one contributed another frame's groups.
  Each slice now carries a witness the aggregation checks, falling back to the
  source pixel when none survive. No perf change.
- **Block matching reused shifted SSD columns incorrectly**, changing candidate
  ranks at radius 0-2. All radii now use exact per-candidate SSD: 647 -> 766.5
  fps on the benchmark config.
- **Concurrent first-use tag clearing could erase witnesses** (a whole-buffer
  clear could submit after another estimate wrote tags). Each estimate now clears
  only its exclusively reserved slot in its own ordered command buffer.
- **Radius-0 aggregation fallback used a frame-derived ring slot**; it uses the
  reservation's slot now.
- **A failed frame advertised ring copies that never landed**, so a later frame
  block-matched against uninitialised VRAM (0.038 max diff at frame 1). The error
  path clears the keys whose chunk-0 copy did not submit and releases blocked
  readers instead of hanging them. `VSFEEL_BM3D_FAULT` pins it.
- **The CAS fallback's retry bound counted the wrong contributors.** It is a
  contention bound, `8 * (ceil((2*bm_range + 8)/block_step) + 1)^2` (968 at
  bm_range=16/step 4, 13448 at step 1), not the `8 * ceil(8/block_step)^2` the
  old comment derived; 32 was short at every step and left 16 of 4096 pixels
  1e-5..6.1e-5 out against the hardware-atomic arm. `res_add` derives it now.
- **Creation-failure leak**: `createVideoFilterEx2` returns nullptr without
  running the free callback, so the instance released just before it leaked. Same
  mechanism as `notes/DFTTEST.md`. No perf change.
- **`VSFEEL_BM3D_NOCHUNKBAR` removed**: it dropped the leading barrier of every
  chunk after 0, which is what orders them behind chunk 0's ring copies.
- **Radius-0 trace false-alarmed and read the cache unlocked**; now gated on
  radius and locked like every other table read.
- **Shared plumbing hardening** (no observable behaviour): device registry
  `insert_or_assign`, non-throwing cache load/save, per-save temp names, clamped
  `max_push_descriptors`, dead members deleted.
- **2026-09-23 — exec-pool port: +1%.** Per-stream pools, timelines and raw
  `gpu_submit` replaced by the core's exec pool. The pool allocates timeline
  values at submit, so the cache's pre-reservation pairs became per-slot
  submitted flags and ordering rides a leading `vkCmdPipelineBarrier`. Three
  interleaved A/B runs: +1.0/+0.9/+1.0% on the medians (1080p GRAY32 r=2, 1000
  frames).
- **2026-09-21 — block-match scan rounds, superseded by the mawen matcher.**
  Packing (x, y) into one word 3.96 -> 3.72 ms; cutting the temporal per-window
  list to `ps_num` 3.72 -> 2.85 ms, VGPR 216 -> 192, 7 -> 8 waves/SIMD. The depth
  cut is void now (a frame contributes up to eight members); packing and the
  merge lemma survive. Dead ends, all at VGPR 240 / 6 waves: the resolved group
  in LDS 5.36 ms, per-lane lists in LDS 7.27 ms, invariant centres in LDS 3.88
  vs 3.72 ms. `NOSEARCH=1` cannot price the search (every candidate ties).
- **2026-09-20 — R80 GPU API port: -4..6%, all of it the download.** A CPU
  consumer pays a full-frame DMA into cached staging because a discrete card's
  planes are write-combined; the upload is free (one memcpy into the frame's VRAM
  plane). Dead ends, all slower: own host staging (-12.8%), direct ring writes
  when `createGPUBuffer` returns host-visible (recovered most of it, second IO
  path), cached vs uncached staging and NT stores (neutral).
- **2026-09-20 — runs where buffer float atomics are missing.** RADV gates
  `shaderBufferFloat32AtomicAdd` at GFX11, so Polaris/Vega/RDNA1-2 failed at
  creation. The `-DNO_FLOAT_ATOMICS` build runs zipcl's `atom_add_f` CAS loop,
  bit-identical at `extractor_exp=8`; cost +17% (314.8 vs 260.0 fps, 1080p
  GRAY32 r=2).
- **2026-09-20 — the estimate cache sizes to its working set** (`ns + 2r`
  instead of `tw + ns + 2r`): -36..41% of total VRAM at no in-order cost, which
  is what made radius 4 fit an 8 GiB card. `VSFEEL_BM3D_CACHE=1` restores the
  seek margin.
- **2026-09-20 — the direct-upload path needs a real ReBAR heap, not just the
  memory type**: an RX 580/Windows reports a 256 MiB aperture heap whose type
  looks usable, and staging from it failed with `VK_ERROR_OUT_OF_DEVICE_MEMORY`
  while VRAM sat empty. `rebar_available` now requires the backing heap to be a
  quarter of the largest device-local heap.
- **Stream knee** is 2 (243/313/312/310/312/313 fps at ns = 1/2/3/4/6/8, 800f
  x2), chosen for the 332 MiB it saves. **VRAM**: `src_ring = 4r + ns`,
  `res_cap = tw + ns + 2r`; 1080p r=2 ns=2 is 1107 MiB, 79% of it the shared
  estimate stack.

## Open work

- **The search is the remaining kernel cost**, and the only one: the
  sigma-scaled threshold rejects most candidates on real content, so the filter
  cost behind it is small. The union's ownership test is 4 int ops per candidate
  per earlier window; a merged-row-interval enumeration has not been measured
  against it.
- **The 2-byte copy and store are unvectorized and unmeasured.** `bm3d_copy`
  converts one element per thread and `store_result` writes four scalar halves;
  a 4-element copy (`uint64` in, `vec4` out) and a packed 8-byte store would cut
  their instruction counts 4x. The store's catch is alignment (the pitch is only
  guaranteed to be a multiple of the element size, so a packed store needs an
  8-byte guard or a scalar tail), and the copy's ceiling is low, so measure
  before writing either.

### Do not retry

- **A half-precision or otherwise narrowed *processing* mode.** The filter is not
  bandwidth-bound (1080p r=2: `gpu_busy_percent` 100, `mem_busy_percent` ~35)
  and its compute is depth-independent: u16 is 979.4 against f32's 956.5 fps
  (1000 frames) and 977.5 against 968.5 with the io removed; f16 is the same
  1% band (see Performance). The float estimate stack (83 MB/frame at r=2) is
  the only large narrowable traffic and cannot narrow anyway (no
  `shaderBufferFloat16AtomicAdd`; no u16 atomics in core Vulkan). bm3dvk2's own
  fp16 mode is 16% slower here (133.4 against 159.1 fps, 500 frames,
  `--cpu-cache`) while reading half the source bytes. Narrowing the *transport*
  (u16, f16) is what ships; this prices narrowing the arithmetic or the ring.
- **Emulating libstdc++'s `partial_sort` tie order.** Survivors are heap
  positions, not a positional rule; exactness needs the whole candidate sequence
  replayed through a 7-element heap (~940 keys/block in LDS plus a serial pass,
  ~+25% on a search-bound kernel) while staying tied to one STL's heap.
- **Interleaving the estimate stack's `num`/`den` arrays.** Moot now that the
  numerator and the per-block denominator share one frame accumulator; the old
  measurement (-3..-4% for all three layouts) priced volume, not pattern.
- Any LDS restructure here — see the mechanism under Performance.
- `#pragma unroll`: glslc ignores it in GLSL and ACO already unrolls fixed-trip
  loops; the remaining variable-trip scans are unrolled by hand. `[[unroll]]`
  does force the frontend to unroll, and it is not worth the build cost either
  (see Historical).
- A different SSD accumulation order (a running window sum, say): it decides
  which blocks match.
- An uncapped queue: uncapped is best or tied.

## Debug env vars

All flags are `VSFEEL_BM3D_<FLAG>`, read through `vsfeel.h`'s helpers; `TRACE` and
`DUMP` are cached at creation, not read per frame.

- `VSFEEL_BM3D_TRACE=1` — acquire/submit/wait trace.
- `VSFEEL_BM3D_TIMING=1` — per-frame host-stage split.
- `VSFEEL_BM3D_PROFILE=1` — per-dispatch GPU kernel table at free, bm3dvk2's `BM3DVK_PROFILE` layout.
- `VSFEEL_BM3D_VRAM=1` — creation-time VRAM budget.
- `VSFEEL_BM3D_SPLIT=0` — radius 0: one estimation submission per frame.
- `VSFEEL_BM3D_RINGWAIT=1` — radius 0: wait the ring copier's submission out
  host side instead of riding the handoff barrier.
- `VSFEEL_BM3D_INFLIGHT=<1..8>` — radius 0: private slots for this many frames
  (default 4). The temporal rings are one window deep plus the isolated copy.
- `VSFEEL_BM3D_CAS=1` — force the CAS aggregation build (A/B only).
- `VSFEEL_BM3D_PATCH_LDS=0/1` — force either reference-patch arm; unset follows
  the driver rule (shared memory everywhere except RADV).
- `VSFEEL_BM3D_SPLIT_KERNEL=0/1` — force either estimation path; unset follows
  the driver rule (split everywhere except RADV).
- `VSFEEL_BM3D_NOSEARCH=1` / `VSFEEL_BM3D_NOESTIMATE=1` — ablation knobs.
  `NOSEARCH` is the matcher's reference-only path, the same thing `th_mse=0`
  selects, so it cannot price the search.
- `VSFEEL_BM3D_MATCHTRACE=<frame>,<x>,<y>` — record and print the group the
  matcher selects for one reference block (members with errors and window
  positions, list sizes and per-frame seed counts). `<x>,<y>` must be a
  reference-block origin, else a creation error. Waits the traced frame out.
- `VSFEEL_BM3D_DUMP=1` / `VSFEEL_BM3D_GPUTRACE=1` — slot dump / GPU timestamps
  (the probe waits each instrumented frame out; never a benchmark).
- `VSFEEL_BM3D_FAULT=<n>` — fail frame `n`'s estimation on a fresh instance,
  after its reservations but before any command buffer; pins the error path's
  ring-key clearing and reader release.
