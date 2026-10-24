# WNNM — notes

Status: **iterating**. The first GPU port ships and matches the reference; it is
now 5.2x slower than vszip's CPU filter, and the block-match bookkeeping is the
remaining cost. Port of `reference/vapoursynth-zip` (`src/filters/wnnm.zig`,
`src/vapoursynth/wnnm.zig`), which is CPU-only (vszipcl/vszipcu have no WNNM);
the parallel Jacobi layout is borrowed from `reference/vapoursynth-nssfactory`
(`src/cuda/common/block_jacobi.cuh`), which is CUDA-only.

One workgroup per reference block on a `step` lattice: block match (`bm_range`
window, optional temporal search), WNNM shrinkage of the group's patch matrix,
weighted scatter. Gram matrix, Jacobi and shrink factors are f64 like the
reference; the reconstruction is f32.

| config | vsfeel | vszip |
|---|---|---|
| jpbd 1080p GRAYS f32, `sigma=3`, radius=0, 5 s medians | 40.83 fps | 210.91 fps |

## Implementation

- `src/wnnm.cpp` (host) + `src/wnnm.comp` (one module, `-DENTRY_GROUP` /
  `-DENTRY_FINISH`). f32 input only, like the reference.
- One dispatch covers the whole plane: `gx*gy` workgroups, one per lattice point
  `(min(k*step, W-bs), ...)`. Per-frame `gpu_frame_buffer` scratch: the weighted
  accumulator (`num`/`den`), the group patch matrix, and (temporal) the
  per-group search-location list; the source planes travel as a host-written
  address table because a temporal window reaches `4*radius+1` buffers.
- Matching is the reference's streaming top-k: candidates are scanned in
  `(y, x)` order in tiles, one thread per candidate computing the sequential f32
  SSD, thread 0 inserting into a sorted list with the reference's strict-`<`
  stable rule. That reproduces `selectTopIdx`, including the all-equal-window
  case where the first k in scan order win.
- Group size `n` may be below `group_size` (small frames, `bm_range=0`); the same
  path handles it, and the centre block is forced into slot 0 when the selection
  dropped it, exactly like the reference.
- Eigen path: Gram in f64 (fma, element order as `gramAtA`/`gramAAt`), then a
  **round-robin parallel Jacobi** in f64: a round rotates a schedule of disjoint
  index pairs, so its rotations commute and one warp applies all of them at
  once, with the rotations themselves computed by the first half-warp (the
  whole workgroup runs the barriers). `v <- v J` accumulates the eigenvectors as
  its *columns*, so the q x q block is transposed once before the sort, which
  still permutes rows; the eigenvalues are `max(diagonal, 0)` sorted descending.
  Reconstruction folds `V diag(r) V^T` into one f32 matrix and scatters
  `sum_kk A[p][kk] * M[kk][j]`.
- Odd `q` still takes the reference's serial rotation order (one thread): the
  round-robin does not converge when its schedule is padded to an even index
  count. Even `q` is every default and every test case that is not `group_size`
  3/5/7/9/..., where the serial path costs the old ~63 ms. The two paths share
  the padded stride `q + (q & 1)`.
- `residual` subtracts the group's per-pixel mean (sequential f32 sum, divide,
  subtract), filters, and adds it back in the scatter.
- Temporal (`radius>0`): the reference writes a tall raw frame per centre frame
  and a second filter folds slabs; at 1080p that is hundreds of MB per frame.
  This port recomputes a centre frame's group for each output frame that consumes
  it, so `2*radius+1` times the spatial work and no cross-frame cache. The fold's
  per-`i` `(frame, slab)` pairs are reproduced including the clipped boundaries,
  where the reference reads one slab twice.
- `q = min(bs^2, group_size)` picks a `-DQMAX` 8/16/32 shader variant; q > 32 is
  rejected (the reference allows 64). f64 is used when the device reports
  shaderFloat64, else the f32 variant (`VSFEEL_WNNM_NOF64=1` forces it).
- Scratch is budgeted at creation against the core's VRAM allowance.

### Tests

`tests/test_wnnm.py`: reference parity against `vszip.WNNM` (spatial sweep,
`rclip`, per-plane sigma, `residual`, `adaptive_aggregation`, starved and tiny
frames, odd geometry, temporal with the boundary fold, all-planes temporal),
self-consistency (determinism, parallel load, request order, frame props, GPU
chaining), creation variants (QMAX, eigen precision) and every error path.
Measured max abs diff vs the reference: 1.8e-7 (defaults) to 7.7e-7
(`bs6/step3/gs16/residual`); the bound is 5e-6. The f32 eigen path differs by
1.2e-5 on a 64x48 crop but 2.2e-3 on 640x360 (a shrink decision flips), so its
test allows 5e-3.

## Performance

- **The round-robin Jacobi is worth 3.77x** (ABBA, same session, n=6, benchmark
  defaults): 11.04 -> 41.57 fps. It replaced a serial chain of ~28 dependent
  f64 rotations per sweep (two divisions and two square roots each) that one
  lane executed while the other 31 sat at the barriers.
- Cost split before the round-robin (1080p GRAYS f32 `sigma=3`, scratch
  ablations, 12-frame screens): Jacobi 63 ms of 86 ms, match bookkeeping
  13.3 ms, SSD 4 ms, patch load + Gram + M 4.3 ms, scatter 1.4 ms. After it,
  the frame is ~24 ms and the search bookkeeping (~13 ms) is ~half: the matcher
  is the next target, not the estimate.
- `bm_range` 3 vs 7 was flat even when the Jacobi dominated; `group_size=1` and
  `bm_range=0` collapse to the trivial 1x1 estimate and were 5x faster, which is
  what pointed at the estimate path in the first place.
- The Jacobi's sweep count barely matters once parallel: 1/3/5/10/30 sweep caps
  gave 33.6/14.8/11.9/11.6/10.8 fps on the old serial chain, i.e. the dependent
  chain, not the number of sweeps, was the cost.
- The remaining estimate cost is now small enough that the serial top-k insert
  is the frame; a parallel selection (the reference's block top-k over a shared
  chunk buffer, or an 8-lane private-list tournament like BM3D's) is the next
  round.

## Historical

- **2026-10-23 — round-robin Jacobi (+3.77x).** The nssfactory CUDA WNNM takes
  its SVD through the Gram matrix with FP32 and a warp-cooperative rotation
  schedule; this round kept the f64 arithmetic (an f32 eigen path is measurably
  too far from the reference, below) and replaced only the rotation loop with
  that schedule. Two porting traps: the schedule needs an even index count, so
  odd `q` pads the stride and still takes the serial order; and the parallel
  version accumulates `v <- v J` where the reference accumulates `J^T v`, so
  the eigenvectors arrive as columns and need one transpose before the sort. A
  missing barrier between "zero v" and "set its diagonal" corrupted the identity
  as well. Reading the reference's precision note was worth it for the
  mechanism, not the precedent: their f32 claim does not hold against *vszip*
  (see the f32 experiment below).
- **2026-10-23 — f32 eigen path is not accurate enough against vszip.** f32
  Gram/Jacobi gives 1.1e-5 on a 64x48 crop but 2.2e-3 on 640x360, so the bound
  for `VSFEEL_WNNM_NOF64` is 5e-3. Feeding the f32 *vectors* to f64 Rayleigh
  quotients for the eigenvalues does not rescue it (2.17e-3): `kept` matched the
  f64 path on all 300 groups of a 160x120 crop, but 30 of them had an M
  difference > 1e-5 (worst 7.4e-5), because a near-degenerate pair straddling
  the shrink threshold rotates its f32 basis enough to move the reconstruction.
  The decision is not the only f32-sensitive quantity, so f64 stays.
- **2026-10-23 — first port.** A numpy prototype of the planned GPU arithmetic
  picked the numbers: f64 Gram+Jacobi with an f32 `V diag(r) V^T` reconstruction
  is 1.8e-7 from vszip on 64x48, while an f32 Jacobi reaches 1.1e-5 there
  (shrink decisions flip), so the eigen path stayed f64. The prototype also
  fixed two reading errors: the VAggregate slab for output `n` in centre frame
  `m` is `n-m+r` (not `m-n+r`), and the fold's clipped boundaries read some
  `(frame, slab)` pairs twice, which a per-`(centre, output)` recompute must
  reproduce. Three kernel bugs then showed up as 1e-3..1e-1 diffs: the temporal
  seed list was not restored at each direction (`b.center` in the reference),
  the stable merge filled from the wrong end, and the plane base offsets were
  byte counts where the kernels index elements.

## Open work

- **Parallelise the top-k.** The serial insert (thread 0 over up to 225
  candidates) is now ~half the frame. Two proven shapes exist in the references:
  a block-wide sort of a shared candidate chunk (nssfactory's `topk.cuh`, a
  radix sort on CUDA, a bitonic here) and BM3D's 8-lane private top-8 with a
  shuffle tournament. The tie order must stay `(error, scan index)`.
- **Make odd `q` take the round-robin path.** Its padded schedule does not
  converge there; the serial fallback keeps it correct at the old cost.
- A separate solve kernel (one warp per group, `M`/Gram/match handoff through
  scratch) would fit ~2-3x more Jacobi warps per CU than the fused kernel's
  LDS-bound occupancy. The nssfactory layout is match+Gram / solve / apply.
- Temporal recompute is `2r+1` times the work; a raw-frame scratch ring (the
  reference's tall frames) or a merged multi-output dispatch is the fix.
- The patch matrix lives in global scratch; a small-config LDS variant would cut
  the Gram/reconstruction traffic.
- q > 32 is rejected (LDS eigen workspace); the reference allows 64.
- The serial top-k insert is one thread; a parallel selection would need the
  same stable tie order.

## Debug env vars

- `VSFEEL_WNNM=1` — creation banner (per-plane geometry, bases, qmax, eigen
  precision) and the CAS-accumulator notice.
- `VSFEEL_WNNM_TIMING=1` — per-frame host split (acquire/record/submit).
- `VSFEEL_WNNM_DUMP=<frame>` — dump each centre's selected group (errors,
  positions, temporal slot, kept count, weight) to stderr.
- `VSFEEL_WNNM_NOF64=1` — force the f32 eigen path.
