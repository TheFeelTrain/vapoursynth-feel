# WNNM — notes

Status: **shipped**. Port of `reference/vapoursynth-zip` (`src/filters/wnnm.zig`,
`src/vapoursynth/wnnm.zig`); vszipcl/vszipcu never had a GPU WNNM. Everything is
**f32**: the estimate is a three-kernel split (Gram, eigen step, scatter), the
spatial match a distance kernel plus a one-thread-per-group register top-8, and
temporal mode runs the fused matcher one centre at a time (32 threads) into a
per-centre match cache, then the same three kernels. `group_size > 8` still runs
the fused kernel.

| config (jpbd 1080p, `sigma=3`, 5 s medians) | vsfeel | vszip |
|---|---|---|
| chroma U/V, `radius=1` (the benchmark default) | **839 fps** | 207 fps |
| chroma U/V, radius=0 | **1615 fps** | 408 fps |
| luma, radius=0 | **1291 fps** | 204 fps |

The radius=1 row is 4.1x vszip and the temporal matcher is ~57% of it; the same
content at radius=0 is 3x faster (see Open work for what is left).

Accuracy, against vszip and against an independent f64 exact-SVD oracle
(320x240 / 640x360, `sigma=3`; max abs diff, "bad" = pixels over 1e-5):

| | vs vszip | vs f64 exact oracle |
|---|---|---|
| vsfeel (default) | 2.9e-3 (640x360) | 3.1e-5, 47 bad |
| nssfactory CPU (`nss.WNNM`, built from `reference/vapoursynth-nssfactory`) | 4.7e-3, 889 bad | same |

The port is f32 end to end; there is no f64 path any more.

## Implementation

- `src/wnnm.cpp` (host) + `src/wnnm.comp`: one shader module with
  `-DENTRY_GRAD/…` variants selected by `-D`. f32 input only, like the
  reference. One workgroup per reference block on a `step` lattice; the block
  position is `min(k*step, W-bs)` on both axes, which is what the reference's
  `while (_x < temp_c + step)` lattice does.
- **Spatial match phase** (radius 0), two kernels:
  `ENTRY_DIST` writes one SSD per (group, candidate) with 128 threads per group
  and two candidates per thread, `ENTRY_SELECT` reads them back with one thread
  per group into a register top-8 (a warp sorts 32 groups at once) and writes
  `{count, xy, t}` groups. The strict-`<` shift-insert reproduces
  `selectTopIdx`'s stable `(error, scan index)` order, and the centre block is
  forced into slot 0 when the selection dropped it.
- **Temporal match** (`radius > 0`): the predictive search is inherently
  sequential (each layer seeds from the previous layer's selection) and stays in
  the fused kernel, which stops after the selection and publishes one match
  record per *centre*. Its candidate selection is parallel (`fr_tile`): every
  candidate counts how many rank below it, error first and then arrival order,
  which is the reference's stable order, and the tile's list is merged into the
  running one. The kernel runs at **32 threads per group** (`-DWG=32`): the
  candidate tile is 32 wide, so a 128-thread workgroup left three quarters of it
  idle while still paying four-warp barriers (1.6x at 32).
- **Per-centre match cache** (`radius > 0`, `group_size <= 8`): a centre's
  selection is identical for every output that consumes it, and there are
  `2r+1` of those, so `CACHE`/`RING` spec constants key the match records by
  centre frame in a persistent ring. `RING` spans the frames the core keeps in
  flight (one request per thread), `numThreads + 2folds` rounded up to a power
  of two; a smaller ring evicts a tag before the neighbouring frame that would
  hit it runs, and every frame then re-searches all folds. The host keeps a tag
  per slot (`centre frame or INT_MIN`) and dispatches the matcher only for the
  folds whose centre is not in the ring, one dispatch per centre under a
  push-constant fold index; the estimate still runs over every slot. Records
  store the position *within the centre's own window*, and the consumer rebases
  it onto its own address table (`layer_base_of`), which is what makes a record
  reusable by a frame whose table starts elsewhere. Tags are published after
  the submit and every cached frame starts with a whole-queue barrier, the same
  cross-submission handoff the BM3D ring uses; the record slot is the only state
  shared across frames, and holding one lock across record-to-submit keeps
  record order equal to submission order. A 160 MiB budget halves the ring when
  its records do not fit (a partial cache still hits), and `VSFEEL_WNNM_NOCACHE`
  drops the ring for per-frame slots.
- **Estimate phase** (any radius, `group_size <= 8`), three kernels over
  `NG * (2*radius+1)` slots, so each phase runs at the width its work needs:
  - `ENTRY_GRAM`: 8 lanes per group, lane l takes column l of every patch row.
    Each lane accumulates the whole 8x8 outer-product sum over its pixels and a
    three-step `subgroupShuffleXor` tree adds the eight partial matrices, so all
    eight lanes end with the same matrix and store it row-major.
  - `ENTRY_SOLVE`: one thread per group. The Gram, the eigenvectors and every
    rotation index are **registers**: the sweep is written out over its 28 pairs
    (`JSWEEP`) because one dynamic index moves both 64-float arrays to local
    memory. 20 sweeps, branch-free rotations, and the four corner elements set
    from the closed form (`app - h`, `ajj + h`, zeros) rather than rotated. No
    sorting: the shrink only needs to know which value holds rank 0 (`top0…`),
    since `M = sum_k gain_k v_k v_k^T` is order-free, and a value's gain is
    `1` for rank 0 or `(s + sqrt(s^2 - c)) / 2s` while `s^2 - c > 0`.
  - One accessor, `match_rec(slot)`, is the record's address for *every* read
    (count, positions, layers); reading the count from the linear slot instead
    is invisible while the ring slot equals the fold and wrong everywhere else.
  - `ENTRY_APPLY`: 8 lanes per group, lane l taking column l of every patch row
    so both the patch loads and the atomics are consecutive words; all eight
    output columns come from the 64-entry `M` in registers. A temporal fold
    scatters only the patches whose source layer is its own destination slab
    (`fold_dest`, recomputed from the fold index and the push constants), which
    is how the reference's VAggregate hands each output one slab.
  - Scratch: `kSplitStride` 136 floats per group (`[0,64)` Gram then M,
    `[64,128)` the residual mean, `[128]` the aggregation weight), in the same
    per-frame buffer the fused path uses for its patch matrix.
- **Fused path** (`group_size > 8`), one workgroup per group:
  streaming top-k match, Gram, the reference's round-robin parallel Jacobi in
  f64 (from `reference/vapoursynth-nssfactory/src/cuda/common/block_jacobi.cuh`,
  mechanism only), shrink, `V diag(r) V^T`, scatter. Odd `q` takes the
  reference's serial rotation order: the padded round-robin does not converge.
- `residual` subtracts each pixel's mean over the group (sequential f32 sum,
  divide, subtract) and adds it back in the scatter.
- Temporal (`radius > 0`) reproduces the VAggregate fold's `(frame, slab)` pairs
  including the clipped boundaries, which read some slabs twice.
- `q = min(bs^2, group_size)` picks a `-DQMAX` 8/16/32 variant; q > 32 is
  rejected (the reference allows 64). Scratch is budgeted against the core's VRAM
  allowance at creation.

### Numerics

The f32 estimate reproduces vszip's f64 *decisions* except where an eigenvalue
sits within ~1e-5 of the shrink threshold, which happens on ~0.2% of groups and
moves the affected pixels by up to 1.4e-2 (typically ~3e-3). That is a deliberate
trade for 13.5x on the whole frame, and the bound is documented in the tests
(`REF_TOL`). The default-parameter error against an f64 exact-SVD oracle is
3.1e-5 with 47 bad pixels at 640x360, i.e. 150x closer than the nssfactory CPU
reference (4.7e-3, 889 bad); `REF_TOL` keeps both references' own spread visible.

### Tests

`tests/test_wnnm.py`: parity against `vszip.WNNM` (spatial sweep, `rclip`,
per-plane sigma, `residual`, `adaptive_aggregation`, starved and tiny frames, odd
geometry, temporal with the boundary fold), the f64 path at `EXACT_TOL` 5e-6
(`VSFEEL_WNNM_NOSPLIT`), self-consistency (determinism, parallel load, request
order, frame props, GPU chaining), creation variants and every error path.

## Performance

- The temporal frame at the benchmark default is ~1.19 ms for two chroma
  planes. Per plane the GPU probe (`VSFEEL_WNNM_GPUTRACE`) sums **0.60 ms**
  (match 0.36, gram 0.16, solve 0.01, apply 0.07, finish 0.00), which is the
  whole plane: the GPU is saturated and the wall time is its own work. An extra
  fold costs ~0.15 ms per plane, and the frame divides roughly 57% matcher,
  25% Gram, 12% apply by ablation (an isolated kernel's ablation understates its
  share because the phases overlap).
- The per-centre cache took the temporal default from 270 to 430 fps by
  searching one centre per frame instead of three; making the tags
  order-independent (no cursor invalidation) recovered the out-of-order requests
  to 473, fixing the record-count read took it to 505, and spanning the core's
  request depth took it to **839**. Request order is not sequential: 38% of
  vspipe's frames arrived out of order when the cache trusted a cursor.
- The spatial frame is 0.62 ms: **match ~0.25 ms, estimate 0.37 ms** (the
  staging had taken the match phase from 0.50 ms).
- The benchmark chain (device-resident frames, no decode) and the live-decode
  screen now differ only by the decoder: 839 fps for the filter's own work, and
  a live 1080p source is decode-bound at about 560-630 fps. The screen that read
  ~630 while the benchmark read 505 was not a harness residual or the decoder:
  the small ring made the filter re-search every fold once enough frames were
  queued (see Historical).
- Staging the search window in LDS is worth 1.17x on the whole frame (ABBA,
  n=12: 1140 -> 1329 fps): the SSD's two global loads per term become two LDS
  loads out of a 2 KB window that the whole workgroup shares.
- The three-kernel split took the estimate from 0.74 ms (one 32-thread
  workgroup per group, fused) to 0.41 ms, and 815 -> 1100 fps, because the eigen
  step gets a whole thread per group with the matrices in registers instead of
  one warp cooperating over LDS.
- f32 is **13.5x** the f64 estimate (ABBA, n=8: 41.8 vs 553.5 fps when the frame
  was 24 ms), which is why there is no f64 path at all now; sizing the fused
  group kernel to 32 threads was another 553 -> 815.
- Porting the split estimate to temporal took the chroma `radius=1` default from
  24.5 to 99.5 fps: the estimate was the fused kernel's f64 half there too.
- The temporal matcher's cost was almost entirely its **serial top-k insert**:
  ablating it gave 2353 fps, and replacing it with the parallel rank selection
  gave 101 -> 178. Shrinking the selection tile from 256 to 32 candidates (the
  rank is quadratic in the tile) took it to 190, and cutting the list arrays from
  256 to 64 entries (they were the whole 64 KB LDS budget, i.e. occupancy) to
  **270 fps**. `ps_num` now stops at 64, where those arrays do.
- The scatter is *free*: replacing both atomics with a plain store gave 769 vs
  814 fps (noise). Do not spend effort on fixed-point accumulation or den
  box-summing.
- Match cost is bounds-checked: `bm_range` 3 vs 7 was flat when the Jacobi
  dominated, so the window size is not what makes the matcher slow; it is one
  global load pair per SSD term.
- `group_size=1` and `bm_range=0` collapse to a 1x1/1x1 estimate and are 5x
  faster, which is what pointed at the estimate path in the first place.

## Historical

- **2026-10-24 - the cache ring spans the request depth (505 -> 839 fps).** The
  ring held two folds' worth of slots (8), but the core keeps one request per
  thread in flight (32 here), so a centre's tag was evicted before the
  neighbouring frame that would hit it ran: 0.95 of 3 centres hit and every
  frame dispatched the matcher for all folds, roughly doubling its GPU time
  (2.45 vs 1.18 ms). Sizing it from `numThreads + 2folds` (64 slots, 68 MiB at
  1080p chroma) gives 1.99 of 3 and one matcher dispatch per frame; a 160 MiB
  budget halves the ring when the records do not fit, which keeps a partial
  cache instead of dropping to per-frame slots. The live-decode screen (631 fps)
  had hidden this because the decoder limits how many frames are queued, so the
  filter looked like it had submission-side headroom; `vspipe -r 8` reproduces
  the fast arm exactly and `-r 32` the slow one.
- **2026-10-24 - the temporal matcher looks a centre up once (270 -> 505 fps).**
  Three changes, each measured A/B on the default arm: 32 threads and a
  32-candidate tile for the temporal matcher (the fused matcher ran 128 threads
  for a 32-wide tile, so 96 lanes hit barriers and nothing else: 1.6x), the
  per-centre match cache (one search per centre, tags and a ring instead of a
  request-order cursor), and a one-line fix to the record count. Traps this
  round: (1) the count read went through the *linear* slot while every other
  field went through `match_rec`, which is invisible while the ring slot equals
  the fold and drops a fold's whole contribution wherever a clip boundary clamps
  the centre - the request-order test caught it on frame 0 only, at 5.2e-3,
  while every other frame agreed to 3e-7; (2) wrapping the temporal path in a
  cursor that invalidated the cache on any out-of-order frame made it useless,
  because vspipe's requests are not in order; (3) a vspipe screen whose input is
  CPU-resident measures the upload, not the filter, and made radius 0 and 1 look
  identical - probe with `std.GPUUpload` or not at all.
- **2026-10-24 - f32 only, and the benchmark measures chroma + `radius=1`.** The
  benchmark now mirrors the filter's real use (`sigma=[0, s, s]` on the f32 clip,
  temporal search): 24.5 fps with the fused f64 estimate, 83 with the same kernel
  in f32, 99.5 with the split estimate on all folds. Deleting f64 removed the
  `EIG_DOUBLE` variants, `VSFEEL_WNNM_NOF64` and the `shaderFloat64` device flag.
  Two traps: the temporal match record must be keyed by *fold index* (the
  clamped centre can repeat), and the fold's destination slab has to filter the
  scatter, or the three folds average instead of accumulating.
- **2026-10-24 - LDS staging for the matcher (+17%).** The distance kernel read
  both patch pixels through L1 for every term. Staging the clamped search window
  (1024 floats = 4 KB, shared by the workgroup's 128 threads) took 1140 -> 1329
  fps. Windows above 32x32 (bm_range > 12 at bs 8) keep the global path: the
  test folds at pipeline creation because both sizes are spec constants.
- **2026-10-24 - the split estimate and the f32 decision.** Three kernels
  (Gram / solve / apply) plus a one-thread-per-group register Jacobi took the
  estimate from 0.74 to 0.41 ms and 815 -> 1100 fps; the f32 solve was chosen
  after measuring that f64 was 13.5x. Two bugs worth remembering: the lane
  partials were per-lane *rows* of the Gram, which no shuffle can reduce (each
  lane must own the whole outer-product sum), and a rotation must set the four
  corner elements from the closed form after the column and row passes, because
  both passes touch them. 16 sweeps is not converged in f32 (2.4e-3 output
  error); 20 is, and 24/30 give the identical answer.
- **2026-10-24 - the references are not identical, and vszip is the accurate
  one.** Building the nssfactory CPU plugin (`.scratch/nss`,
  `cmake -DNSS_BUILD_CPU=ON`) and diffing it against vszip showed 1.1e-6
  agreement on small frames but 4.7e-3 / 889 pixels at 640x360, up to 2.0e-2
  with `sigma=10`. An independent f64 exact-SVD implementation of the same
  algorithm (`.scratch/nss/oracle2.py`, added as mode E of
  `.scratch/wnnm/proto.py`) matches **vszip to 1.8e-7 and the f64 port to
  1.8e-7, but nss by 4.7e-3**: where they disagree, nss's f32 SVD is the one
  that is wrong. So matching vszip is matching the algorithm, and the f32
  estimate's 3.1e-5 is still two orders closer than the "trusted" oracle.
- **2026-10-23 - round-robin Jacobi (3.77x).** The nssfactory schedule replaced a
  serial chain of ~28 dependent f64 rotations per sweep: 11.04 -> 41.57 fps
  (ABBA, n=6). Two traps: the schedule needs an even index count, so odd `q`
  pads the stride and keeps the serial order, and the parallel form accumulates
  `v <- v J` where the reference accumulates `J^T v`, so the eigenvectors arrive
  as columns and need one transpose before the sort.
- **2026-10-23 - f32 was rejected on the old kernel, and that verdict is void for
  the new one.** With the fused f64-era kernel, f32 Gram/Jacobi gave 1.1e-5 on
  64x48 but 2.2e-3 on 640x360 against vszip, and feeding f32 vectors to f64
  Rayleigh quotients did not rescue it (2.17e-3) because a near-degenerate pair
  straddling the threshold rotates the f32 basis itself. The *mechanism* stands,
  but the current f32 split reaches 3.1e-5 against the exact oracle, so the new
  arithmetic is not the old arithmetic; the difference is the branch-free
  20-sweep register Jacobi with the corner corrections.
- **2026-10-23 - first port.** A numpy prototype picked the arithmetic and
  fixed two reading errors of the temporal fold (the slab index is `n-m+r`, and
  the clipped boundaries read some `(frame, slab)` pairs twice); three kernel
  bugs then showed up as 1e-3..1e-1 diffs: the temporal seed list was not
  restored per direction, the stable merge filled from the wrong end, and the
  plane base offsets were byte counts where the kernels index elements.
- **Void, do not quote.** All per-phase numbers taken on 12-frame vspipe
  "screens" before this round are garbage: they said the match phase was 15 ms of
  24 ms when the graded A/B said 0.5 ms. The screens swing 10-25% and one of
  them misattributed the whole frame.

## Open work

- **The GPU is no longer idle: the dispatches sum to the graded frame.** The
  1.98 ms frame that used to look submission-bound was the ring thrash; with the
  depth-sized ring the probe's 1.18 ms of dispatches match the 1.19 ms wall, so
  the remaining headroom is kernel work, not submission. The probe's
  `BOTTOM_OF_PIPE` pairs are reliable for the total and not for the short spans,
  so measure whole-frame totals.
- **The temporal matcher is 57% of the frame at 7-10 warps per CU.** Its 32
  threads are right for the tile and the barriers, but the SSD chains are
  latency-bound and the CU is nearly empty. A 128-thread workgroup with a
  128-candidate tile is *worse* (the rank is quadratic in the tile). The
  candidates are four groups per workgroup with subgroup barriers (four warps,
  each running its own group) or one warp per reference block with the top-K by
  shuffle argmin (nssfactory `match.cu`) - both need the candidate loop to stop
  using workgroup barriers.
- The estimate still runs over all `2r+1` slots even when the cache supplied
  their matches; caching `M` and the weight per centre as well would drop the
  Gram+solve to once per centre (~0.2 ms of the frame) for twice the ring
  memory.
- What is left in the matcher itself is the SSD traffic: each candidate reads
  its 64 pixels from global (the reference block is already staged). The
  spatial pass could stage its window like `ENTRY_DIST` does, but `win` costs
  4 KB of the LDS that the occupancy depends on.
- **The spatial matcher is still ~40% of its frame (~0.33 ms).** Its SSDs read both
  patches from LDS but the *reference* block is re-read for every candidate, and
  the 225 distances per group go out to a 52 MB array and come back. The shape
  that removes both (nssfactory `match.cu`, `spatial_match_warp_kernel`) is one
  warp per reference, the reference block in registers, the top-K by warp-wide
  argmin and no distance array at all.
- `group_size > 8` and `radius > 0` still take the fused f64 kernel, i.e. tens of
  fps, not 1100. Porting the split estimate to `q = 16/32` needs a different
  eigen step: 16x16 f32 no longer fits in registers (nssfactory switches to
  16 lanes per group over equivalent LDS).
- q > 32 is rejected (LDS eigen workspace); the reference allows 64.

## Debug env vars

- `VSFEEL_WNNM=1` - creation banner (per-plane geometry, scratch bases, qmax).
- `VSFEEL_WNNM_TIMING=1` - per-frame host split (acquire/record/submit).
- `VSFEEL_WNNM_DUMP=<frame>` - dump each centre's selected group (errors,
  positions, temporal slot, kept count, weight) to stderr.
- `VSFEEL_WNNM_NOCACHE=1` - no per-centre match cache: one record per fold per
  frame (the fallback path, and the A/B arm for the cache).
- `VSFEEL_WNNM_GPUTRACE=<frame>` - one-shot GPU timestamps around the match,
  Gram, solve, apply and finish dispatches of that frame, printed to stderr.
  Feed the filter device-resident frames or the number is the upload.
- `VSFEEL_WNNM_NOSPLIT=1` - take the fused kernel for the split path's configs
  too (the `group_size > 8` code path).
- `VSFEEL_WNNM_SKIPMATCH=1` / `VSFEEL_WNNM_SKIPGROUP=1` - ablations: skip the
  match phase or the estimate phase (output is garbage, timing only).
