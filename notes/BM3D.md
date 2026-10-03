# BM3Dv2 — notes

**shipped.** Vulkan port of BM3D whose block matcher and collaborative filtering
are mawen's CPU V-BM3D (`reference/VapourSynth-BM3D`), so output differs from any
CUDA-matcher build. The CPU plugin is the only oracle, driven from
`tests/test_bm3dv2.py` (`bm3d.Basic`, `VBasic`/`VFinal` + `VAggregate`).

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
- Every plane of a colour clip is denoised: one entry per plane, or one joint
  4:4:4 entry under `chroma=True` whose groups come from luma. A plane below its
  sigma threshold is a bit-exact source copy; all planes below it and the filter
  is the source clip.
- `th_mse` (8-bit MSE, inclusive, 0 = reference only) defaults to
  `sigma[0]*80 + 400` basic and `sigma[0]*10 + 200` final.

Scoreboard — 1080p jpbd, `tools/benchmark.py -f bm3dv2`, harness defaults
(sigma 0.7, radius 2, bm_range 9, ps_range 4, block_step 8), 1000 frames x3:

| | fps | |
|---|---|---|
| vsfeel | **942** | mawen matcher and filtering |
| bm3dvk | 296 | fixed 8-member groups |
| vszipcl | 133 | |

3.2x the faster reference, but not on equal work: they always group eight while
the default threshold accepts ~80% of candidates here and the top-eight
selection does the filtering. vsfeel was 845 fps before the bookkeeping round
below; the reference columns are the same session as the 942 and swing with the
box like everything else here (bm3dvk measured 237 in an earlier one).

## Implementation

- Two kernels: `bm3d.comp` (match, group, collaborative transform; one warp of
  32 lanes = four 8-lane groups, one 8x8 block each) and `bm3d_agg.comp`
  (aggregation over the TW = 2r+1 stack slices).
- One entry per processed plane, or one over all three planes of a 4:4:4 clip
  under `chroma=True`. An entry owns a source ring and an estimate stack laid out
  `[slot][clip][plane][h][stride]` and `[slot][plane][tw][2][h][stride]`; an
  entry's planes share one geometry, so the plane stride is a spec constant.
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
  (`pe`/`pxy`), updated by a single compare-exchange per slot. Only the
  prediction seeds (`s_x`/`s_y`, plus `s0_*` for the forward restart) live in
  shared memory: every lane of a group reads all of them, and a register array
  indexed by a runtime count would spill to scratch. The seeds of the earlier
  predictive windows are read into registers once per window, not per candidate.
- Each walk evaluates four candidates per iteration (stride 32), so the
  scheduler has four independent load chains in flight; the walk state is
  stepped between the candidates so a row wrap lands correctly.
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
  `2.7*normY*64/255` and `normY*64/255` (normY = the bt709 luma-row norm
  0.7496149945138504, which `th_mse` carries too). In this kernel's
  `A = D^-1 F` domain the CPU's per-DC-axis threshold collapses to a uniform one
  and the masked reconstruction is identical (pinned by a test).
- A centre frame searches only real frames (`zlo = max(0, r-m)`,
  `zhi = min(2r, nframes-1-m+r)`); the aggregation reads the real centres
  `[n-r, n+r]` once each at slice `z = n-m+r` and leaves a slice outside that
  range unwitnessed rather than a clamped endpoint copy.
- The witness (`tags`) has one region per entry, since entries key slots by frame
  index and can hold different frames in the same slot.
- Estimation accumulates with hardware buffer float atomics where available
  (`VK_EXT_shader_atomic_float` plus `shaderBufferFloat32AtomicAdd`; the plain
  load/store/exchange bit is not enough), else the same kernel's
  `-DNO_FLOAT_ATOMICS` build runs the reference's `atom_add_f` CAS loop
  (`VSFEEL_BM3D_CAS=1` forces it).
- Radius accepts up to 16, but the estimate cache binds first: it grows as
  `(2+2r)(2r+1)` plane pairs, so 1080p runs radius 7 and refuses 8 while 640x360
  reaches 16. Every refusal is a creation-time error naming the size.
- Degenerate paths: `sigma < FLT_EPSILON` copies a plane through, and with no
  plane above it the filter is not built (the clip is handed back). Validation
  runs before that shortcut: `extractor_exp` `[0, 127]` as bm3dvk does, `th_mse`
  finite and non-negative, saturated to `FLT_MAX` when it does not fit a float
  and raised to `FLT_TRUE_MIN` when a positive value underflows. `th_mse` rides
  spec constant 17.
- Everything records through the core's exec pool; the filter owns no command
  pool, timeline or fence. One submission per recomputed window position (up to
  2r+1 per frame; `VSFEEL_BM3D_SPLIT=0` reverts) keeps a single submission off
  Windows' TDR watchdog, then one aggregation submission whose planes the pool
  publishes.
- Cross-frame handoff is a per-slot *submitted* ready flag: a reader waits host
  side for its writers' estimations, then a full pipeline barrier at the start of
  its first command buffer supplies the execution and memory dependency (both
  access scopes cover both directions, because the slots are written again). No
  recording context is held while waiting, so the pool cannot deadlock.
- R80 GPU API: `clip:vnode:gpu` in and out with `ffGPUOutput`; an unprocessed
  plane is shared from the centre frame and only planes the aggregation writes are
  published. `num_streams` and `device_id` are registered no-ops. The 8-lane
  transposes and the group-8 reduction are `subgroupShuffleXor` butterflies
  (register-only, no LDS, no barrier) reproducing the reference's tree.
- `extractor_exp`'s `(x + E) - E` pre-rounding must stay under GLSL `precise`:
  RADV folds the pair to `x` when E is a known spec constant.

## Performance

- **The candidate bookkeeping, not the SSD, was the cost.** A per-candidate
  ablation ladder on the estimation kernel (fps, 1080p jpd, harness config)
  prices the pieces as: search 42% of the frame, estimate/transform 12.5%,
  aggregation 13%, slot zero-fill ~2%. Removing the SSD itself is worth 32%.
  The load-bearing discovery is that every piece of per-candidate code is
  if-converted: the warp pays the insert's ~90 instructions whenever *any* lane
  passes the gate, so per-lane gating buys nothing on its own.
- The bookkeeping round (same config, 1000 frames x3): **845 -> 941 fps**. The
  four changes, each verified by an interleaved same-binary A/B over 1200-2000
  frames: the ownership test's LDS seeds became registers (its 8 LDS loads each
  carried a full `lgkmcnt(0)` drain, per candidate); the merge became a
  shuffle tournament; the per-lane lists moved out of LDS entirely; the walk
  evaluates four candidates per iteration.
- Marginal cost per candidate per lane: 0.0032 ms (`bm_range`), 0.0030
  (`ps_range`), with a ~1.4-2.0 ms intercept at tiny windows.
- **LDS is a trap here.** Three attempts to move kernel-live data into shared
  memory each raised VGPRs 216 -> 240 and dropped occupancy 7 -> 6 waves/SIMD,
  losing 5-90%: the cost is dynamic LDS addressing, not traffic. The wins came
  from reducing the data instead. A fourth attempt (the reference patch, 64
  registers per lane, in `l_cur`) cost 3.5% for the same reason: 64 extra LDS
  reads per candidate beat the 64 registers it freed.
- Colour costs the planes: 4:2:0 is 1.5x the luma-only estimate/source footprint
  and 4:4:4/RGB 3x, all inside the same `maxStorageBufferRange` budget.
- The multi-plane round did not move the luma path (NPLANES == 1: 6131 vs 6133
  instructions, 192/108/4096 B, 8 waves/SIMD; 12 ABBA pairs at 935 vs 924 fps).

## Historical

Chronological; each entry keeps the mechanism, not the story.

- **2026-10-03 — the search's bookkeeping was the whole cost (+11%).** The
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
  - the walks evaluate four candidates per iteration (stride 32), which gives
    the scheduler independent load chains; 2-wide was worth ~1% over 1-wide,
    4-wide another ~1%;
  - `group_add` bails out of a frame's list once an entry cannot beat the
    overflowing group's worst (the list is ascending, so the rest cannot
    either). Each is exact: the merged order is still `(error, y, x)`, the seeds
    and `count` are unchanged, and the full BM3D test suite passes.
  - Not taken: reordering the group-key compare to drop the `xy` tie-break
    (invalid for the temporal walk, whose union enumeration is not in scan
    order), and moving the reference patch to LDS (see Performance).

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
  of colour ref-pass comparisons (a frame holding entry 0 while waiting for
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

- **Emulating libstdc++'s `partial_sort` is not worth its price.** The survivors
  among tied candidates are heap positions, not a positional rule (neither
  first- nor last-in-scan matches, and dropping the large-key candidates the
  heap evicts early changes 2113 of 32400 outcomes), so exactness needs the
  whole candidate sequence replayed through a 7-element heap: ~940 keys per
  block in LDS plus a serial pass, ~+25% on a kernel whose search is already the
  whole filter, and correctness tied to one STL's heap.
- **The 1-ulp SSD divergence is a deliberate trade**: `precise` on the
  accumulators costs ~8 ALU ops per row (the fused multiply-add is the search's
  throughput) to buy back ~10% of the tie floor.
- **The search is the remaining kernel cost**, and the only one: the
  sigma-scaled threshold rejects most candidates on real content, so the filter
  cost behind it is small. The union's ownership test is 4 int ops per candidate
  per earlier window; a merged-row-interval enumeration has not been measured
  against it.
- **The per-lane depth cannot drop again**: a neighbouring frame's third through
  eighth candidates are members of the final group.
- **RX 580/Windows: the long-submission TDR is bounded, awaiting a field run.**
  `bm_range=4` and `NOSEARCH=1` both make it disappear, so it tracks the search's
  duration in one submission; the estimation now submits per position.
- **No cheaper atomics exist on the devices that need the fallback.** GFX8-10
  have no `buffer_atomic_add_f32`, so CAS is the floor;
  `shaderSharedFloat32AtomicAdd` cannot help because a group's eight matched
  patches land anywhere in the plane.
- **The estimate phase's 0.40 ms is not decomposed** into transform vs atomics.
  The CAS build bounds it: its +0.66 ms/frame for a read-modify-write loop means
  the two `atomicAdd`s are most of it.

### Do not retry

- Any LDS restructure here — see the mechanism under Performance.
- `#pragma unroll`: glslc ignores it in GLSL and ACO already unrolls fixed-trip
  loops; the remaining variable-trip scans are unrolled by hand.
- A different SSD accumulation order (a running window sum, say): it decides
  which blocks match.
- An uncapped queue: uncapped is best or tied.

### Method rules

- Grade kernel changes on `VSFEEL_BM3D_GPUTRACE=1` at `-r 1` over a few hundred
  frames (the printed value settles to ±0.2%), then confirm with an interleaved
  `tools/benchmark.py` pair over 1000+ frames.
- Interleave ABBA within a round before reading any difference under ~3%: this
  harness swings ±5-10% per invocation even for one binary.
- Radius limits are geometry-dependent: check 1080p before quoting a cap (7
  there, not the 10 the int32 guard alone implies).
- Every GPU timestamp figure printed before 2026-09-21 is 1000x off (the
  accumulator truncated and printed ms as ns). Do not quote them.
- Never chain build -> install -> test; use `tools/install.sh` (hash-verified).

## Debug env vars

All flags are `VSFEEL_BM3D_<FLAG>`, read through `vsfeel.h`'s helpers; `TRACE` and
`DUMP` are cached at creation, not read per frame.

- `VSFEEL_BM3D_TRACE=1` — acquire/submit/wait trace.
- `VSFEEL_BM3D_TIMING=1` — per-frame host-stage split.
- `VSFEEL_BM3D_VRAM=1` — creation-time VRAM budget.
- `VSFEEL_BM3D_SPLIT=0` — one estimation submission per frame.
- `VSFEEL_BM3D_RINGWAIT=1` — wait the ring copier's submission out host side
  instead of riding the handoff barrier (the `tags` witness cannot see a late
  ring copy).
- `VSFEEL_BM3D_CACHE=1` — add the seek margin back to the estimate cache.
- `VSFEEL_BM3D_CAS=1` — force the CAS aggregation build on a device that has
  buffer float atomics (A/B only).
- `VSFEEL_BM3D_NOSEARCH=1` / `VSFEEL_BM3D_NOESTIMATE=1` — ablation knobs.
  `NOSEARCH` is the matcher's reference-only path, the same thing `th_mse=0`
  selects, so it cannot price the search.
- `VSFEEL_BM3D_MATCHTRACE=<frame>,<x>,<y>` — record and print the group the
  matcher selects for one reference block of one centre frame (members with
  errors and window positions, the current-frame list size, each temporal frame's
  retained/seed counts). `<x>,<y>` must be a reference-block origin; anything
  else is a creation error rather than a silent snap. Debug only: the traced
  frame's aggregation is waited out.
- `VSFEEL_BM3D_DUMP=1` / `VSFEEL_BM3D_GPUTRACE=1` — slot dump / GPU timestamps.
  The probe waits each instrumented frame out (one shared query pool) and never
  runs in a benchmark.
- `VSFEEL_BM3D_FAULT=<n>` — fail frame `n`'s estimation on a fresh instance,
  after its reservations but before any command buffer; the error-path test uses
  it to pin the ring-key clearing and the reader release.
- `VSFEEL_BM3D_NOCHUNKBAR`, `VSFEEL_BM3D_HD`, `VSFEEL_BM3D_QUEUES` — **gone**.
