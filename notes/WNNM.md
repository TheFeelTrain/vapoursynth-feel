# WNNM — notes

Status: **iterating**. The first GPU port ships and matches the reference, but it is
~17x slower than vszip's CPU filter: the per-group f64 Jacobi runs on one lane.
Port of `reference/vapoursynth-zip` (`src/filters/wnnm.zig`,
`src/vapoursynth/wnnm.zig`), which is CPU-only (vszipcl/vszipcu have no WNNM).

One workgroup per reference block on a `step` lattice: block match (`bm_range`
window, optional temporal search), WNNM shrinkage of the group's patch matrix,
weighted scatter. Gram matrix, Jacobi and shrink factors are f64 like the
reference; the reconstruction is f32.

| config | vsfeel | vszip |
|---|---|---|
| jpbd 1080p GRAYS f32, `sigma=3`, radius=0, 5 s medians | 12.46 fps | 208.31 fps |

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
- Eigen path: Gram in f64 (fma, element order as `gramAtA`/`gramAAt`), Jacobi in
  f64 with the reference's rotation order and tolerance, descending eigenvalues
  (`max(.,0)`), then the shrink factors. Reconstruction folds `V diag(r) V^T`
  into one f32 matrix and scatters `sum_kk A[p][kk] * M[kk][j]`.
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

The per-group estimate path, not the search, is the cost. Screens (jpbd 1080p
GRAYS f32 `sigma=3`, 12-frame runs, single measurement):

| change | fps |
|---|---|
| defaults (bm_range=7, gs=8) | 11.5 |
| `bm_range=3` | 11.5 |
| `bm_range=0` (one candidate) | 65 |
| `group_size=1` (q=1, no Jacobi) | 57 |
| `block_size=4` (4x groups) | 4.75 |

- `bm_range` 3 vs 7 is flat: the 225-candidate search is not the limiter.
- `group_size=1` and `bm_range=0` collapse to the trivial 1x1 estimate and are
  5x faster, so the Gram + Jacobi + reconstruction + scatter of a full group is
  ~5/6 of the frame.
- Capping the Jacobi's sweep loop is the price tag: 1 sweep 33.6 fps, 3 sweeps
  14.8, 5 sweeps 11.9, 10 sweeps 11.6, 30 sweeps 10.8. The first sweeps cost
  most because each rotation is a serial chain of f64 divisions/sqrts, and RADV
  emulates those at long latency; only one lane of the workgroup runs it.
- The reference batches four groups per f64 SIMD lane set, which is the same
  latency amortisation this kernel lacks.

## Historical

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

- **Batch the Jacobi across groups** (the reference's `patchEstimateBatch`): 4-8
  groups per workgroup so one SIMD instruction advances several independent
  chains and the f64 div/sqrt latency is amortised. This is the main lever; the
  sweep-cap ladder says ~2/3 of the frame is in that serial chain.
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
