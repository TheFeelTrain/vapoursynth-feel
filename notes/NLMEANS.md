# NLMeans — notes

Status: **shipped.** Verified against `src/nlmeans.{cpp,comp}` and
`CMakeLists.txt`.

- **Slot-direct cache** — shared pool of padded per-(clip,frame,channel) tiles;
  kernels resolve `layer -> slot` via `slot_table` (binding 8), so no
  staging→u1 copies and no D2D refreshes.
- **Reservation tokens** — `res_token`/`writer_token` identify holders, not
  frame indices (BM3D pattern).
- **fp16 weight ring ×4096** (`float16_t u4a[]`, binding 4) — the
  subnormal-cliff fix; drift ≤5 LSB / ~1e-4 float.
- `first`-flag init ships (`pc.pc2`); finish fused into the last acc round
  (`pc.pc3`).
- `NLM_REF` is a spec constant (`nlmeans.comp:43`), not a compile-time `-D`.
- Single queue (`queues[0]`), tile reuse ordered host-side by submit count.
- Accuracy: ≤1 LSB (16-bit) / ~3e-8 (float) vs vszipcl.
- Perf, standard clip (3000f, ns=2, d=2, `channels=UV`, medians, same-session
  pairs): u16 **1094** vs 767 (1.43x), f32 **1015** vs 672 (1.51x).
- **ReBAR upload staging** (default on, `VSFEEL_NLMEANS_HD=0` opts out): the
  per-stream staging buffer is host-visible VRAM, so the compose writes VRAM and
  the staging→slot copy is a device-local read; output bit-identical against
  both arms and vszipcl, ~2 MiB/instance.

## Implementation

- Sweep tables ported verbatim from vszipcl's `create()`: half-space of
  displacements (`kk*spt_area + j*spt_side + i < 0`), mirror entries double the
  weight rows when `kk!=0`, stride-8 rows, `qb=8` batches, variants by
  `m=min(d,n)`.
- Padded margins zeroed once at staging init; flat-index reads wrap OOB columns
  into neighbouring rows' zero margins, so candidate coords need no bounds
  checks.
- u2 pixel-interleaved (C+1) floats; u5 starts at the `first`-flag constant
  `1.1920929e-7f` (FLT_EPS) so the finish denominator stays > 0.
- Barriers between EVERY weight→acc and acc→next-acc pair (u4a RAW through the
  slot ring, u2/u5 RMW), plus transfer→compute after copies/fills — the
  serialization zipcl gets free from an in-order OpenCL queue.
- Host wait for a writing slot is on the writer's **submit count**, not a
  device timeline (one FIFO queue). Deadlock lesson: device-side timeline waits
  here hang the GPU (an early CB waits on a signal only a later CB produces).
- `copy_stream_read` needs 32-byte-aligned sources — unusable for odd-width row
  downloads (segfaulted at width 613); plain memcpy.
- int32 addressing bound: reject when
  `max(lay·layers·C, npix·slots, npix·C) >= 2^31`.
- Workgroup tile `BLK_X=32`, `BLK_Y=8`, `VRT_RESULT=3`; weight shaderstats
  VGPRs=24 SGPRs=108 LDS=6144 (6144 vs zipcl's 5120), no spills.
- ACO raises VGPRs 11→24 for load ILP on purpose; forcing them down regressed
  (guarded-load variant → 713–733 fps, reverted). 3rd WG/SIMD needs VGPR≤21.

## Historical

- **2026-09-20 — the staging ReBAR path is taken only if the heap can hold the
  allocation** (no change on this box). Same mechanism as `notes/BM3D.md`.

### 2026-09-20 — dead pad/finish kernels deleted; pipeline creation was not the wall

`ENTRY_PAD` and `ENTRY_FINISH`, their modules/pipelines and the
`nlmeans_{16,32}_{pad,finish}.spv` variants were built and destroyed at every
creation but never dispatched: finish has lived in the last acc round (`pc3`)
since the slot-direct rewrite, and tiles ship by `vkCmdCopyBuffer`. Deleting
them also drops the dead `{dst,src}` pair table the pad kernel read, the
`2*window_tiles` int tail of the tables buffer, the binding-9 descriptor and its
`create_staging` rewrite, the staging `STORAGE_BUFFER` usage flag, and the
unused `PH`/`NLM_H` spec constants (`NLM_H` was `h`, dead in every live kernel).
Cold creation (`VSFEEL_PIPELINE_CACHE=0`, 640x360 GRAY16) 20.3 → 18.2 ms median
of 2; warm per-instance creation unchanged (32.8 vs 34.1 ms, run spread), so
the two pipeline compiles were **not** the dominant per-instance cost. No fps
change; 189/189 filter tests, 800 full suite.

### 2026-09-19 — ReBAR upload staging (+7.7%)

The per-stream upload buffer was GTT, so every composed tile crossed PCIe twice
(host→GTT, then the CB's GTT→VRAM copy). Allocating it
`DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (a device property probed once at
creation into `staging_direct`; `VSFEEL_NLMEANS_HD=0` opts out) makes the compose
land in VRAM and the existing copy a VRAM read. Command buffer and copy
structure are unchanged, and the buffer holds only the tiles a frame ships, so
the VRAM cost is one tile set. 1000f u16 ns=2, 3 interleaved pairs: 1076.7 vs
999.6 fps, every pair favouring the ReBAR arm; `|HD-GTT|` and `|either-ref|` are
exactly 0. Ceiling is small by design (only new tiles are uploaded).

### Cache + upload (2026-08-22 → 2026-09)

- **Slot-direct cache** replaced a per-stream contiguous u1 window: steady
  state pays ~1 new frame H2D instead of ~20 MB DMA, and VRAM drops.
- **Pad kernel eliminated**: staging tiles use the padded slot layout, compose
  writes interior rows at `(PAD+y)*pstride+PAD`, margins ride along as
  one-time-init zeros.
- **Cache-sharing fix (606 → 714 fps)**: pool sized for full concurrent
  windows, held tiles share-read safe, never block on held tiles. Superseded:
  the duplicate-upload branch this introduced was later found to be the root
  cause of a 548 fps regression.
- **Duplicate-upload fix (548 → 664 fps, +21%)**: a writing slot is now REUSED
  plus a host-side submit-order dependency; error path bumps the writer counter
  too. First version used device-side timeline waits → hard GPU deadlock
  (`test_parallel_load_matches_serial` hung, 24 workers stuck); host-side
  ordering is acyclic because acquires are totally ordered by `cache_lock`.
- **Benchmark restructure (664 → 939 fps, +22–23%)**: the real-clip gap was the
  chain's `depth(clip,16)` conversion running inside the timed region and
  stealing bandwidth from compose/download (real compose 0.83 ms at 32 VS
  threads vs 0.42 at 4). `make_vpy` now pre-converts the cache to the filter's
  input format; 68/68 pass, filter untouched.

### Kernel levers (measured)

- **Exact per-instance LDS sizing** via spec-constant array dims
  (`shared float dist[VRT*BY + 2*NLM_S][DIST_W]`): weight batch 0.237 → 0.207 ms
  (+13%).
- **Native u16 io** (`GL_EXT_shader_16bit_storage`): matches the reference's
  `global_load_u16`; finish stores u16 directly (drops the u1z fill and the GTT
  `atomicOr`; finish 0.131 → 0.086 ms); overall +14% (250 → 284).
  glslc `-O` dies on direct float→uint16_t; use
  `uint16_t(uint(x) & 0xFFFFu)`.
- **Interior/border tile split + batched loads** (687 → 728): uniform
  whole-tile-in-bounds predicate kills per-cell checks for ~95% of WGs; each
  thread batches both cells' guide loads before dependent math.
- **Dead per-cell bounds checks in the interior path** (945/1011 vs vszipcl 768
  = +32%): keep both cells' loads batched, store directly into LDS. Splitting
  into sequential per-cell blocks regressed 945→855 by destroying the 8-load
  ILP. Weight code 6252 → 6060 instr, VGPR still 24.
- **Ring budget**: pack chosen so the u4a ring ≤64 MiB; pre-fp16 optimum was
  pack=2, post-fp16 pack=4.
- Warm weight batch = 49 µs @540-row: dist phase 32 µs (65%), hsum/vsum/exp/
  stores/launch 17 µs. Box sums are free in steady state (NOSUM == NODIST).
- GPU budget, full 1080p (measured ×2 of a half-height trace): pad 0.17 |
  weight 0.86 | acc 0.32 | finish 0.09 ms ≈ wall 1.46 (queue saturated).
- Weight kernel = 72% of GPU time. The residual gap vs zipcl is a uniform
  ~1.5–1.9× per-launch, not codegen (our ACO ISA is smaller: weight 188 vs 263
  instr, acc 119 vs 146, finish 65 vs 86).

### Accuracy work

- **fp16 ring cliff**: first attempt drifted 2572 LSB / 0.036 float because
  weights < 6e-5 lose mantissa bits exponentially. Storing `w*4096` and
  multiplying by 1/4096 on load brought drift to ≤5 LSB / ~1e-4 float with the
  original test tolerances unchanged. Weights are format-independent f32
  intermediates, so both i16 and f32 get the same benefit.
- **16-bit UV output "garbage"** was a stale `nlmeans_16_acc.spv` in the build
  dir (14616 vs 15664 B), not a source bug: a clean rebuild matches vszipcl
  within 1–4 LSB on every 16-bit UV case. Keep the whole `vk_spv` tree
  consistent with `src/*.comp`.
- **The fp16 D3 cliff cannot be scaled away**: weights ≤1 and fp16 max 65504,
  so 2^16 already overflows; the largest safe scale (2^15) moves the flush only
  from arg 25.65 to 27.73. `finish_sample()` now returns the centre sample on a
  zero total weight instead of 0/0 = NaN.
- Noise clip, d=0 a=2 s=4, max over frames 0/11/23: `wref=1, h=1.2` 2 codes
  (u16) / 1.9 (f32); `wref=0, h=1.2` 4573 codes both depths, no NaN (was NaN at
  f32); `wref=0, h=3.0` ≤1 code; `h=0.6` 8192 codes, and vszipcl itself emits
  non-finite pixels (gray32 and UV32).
- **16-bit UV reference sweeps**: tol 1e-4 measured 8.31e-5 (32-bit),
  tol 8.0 LSB measured max 4.0 (16-bit).

### Correctness fixes (pool / rclip / holders)

- **rclip layer-table OOB when n < d**: clip-1's table write used the
  full-window stride `key_off = C*layers`, but `count = 2*min(d,n)+1 < layers`,
  so the read ran `C*(layers-count)` elements past the vector and the
  `*slot_elems` multiplier turned heap garbage into an out-of-range slot base.
  Fix: `key_off = C*count`. Every earlier rclip fixture used d=0 (count ==
  layers), which is why 106 tests missed it.
  Measured (4-frame GrayS, BoxBlur guide, vs vszipcl): before d=1 frame 0
  2.38e-2 / d=2 frame 1 6.68e-3, n≥d ~1.0e-6; after all frames ≤1.6e-6.
  Test: `test_rclip_temporal_matches_reference_early_frames` (d=1,2; frames
  0..3; tol 1e-4).
- **Slot pool smaller than one window hung on cache_cv**: a frame at n≥d must
  hold `clips*C*(2d+1)` slots; when the 512 MiB budget bound below one window
  the all-or-nothing acquire could never set `ok`, the frame held nothing and
  submitted nothing, so nothing could notify it. Fix: reject at creation when
  `n_slots < window_tiles`, before any large allocation. The rejection set is
  exactly the previous hang set.
  Examples: 1080p f32 YUV444 d=16 (pool 70, need 99) and 4K f32 Gray d=9
  (18, 19) HANG → precise create error; 1080p f32 Gray d=16 (37, 33) runs both.
  Enumeration over 720p–8K × 16/32-bit × C{1,2,3} × d{1..16} × ns{1,2,8} ×
  rclip{0,1} (2880 configs): 403 hang at the shipped ns=1, 1209 across the ns
  set; the fix rejects exactly that set.
  Tests: `test_reject_window_larger_than_slot_pool` +
  `test_accept_window_that_fits_the_slot_pool`.
- **holder identity was a frame index**: two concurrent evaluations of frame n
  each pushed n, so `std::remove(..., n)` at the first release dropped BOTH and
  marked the slot evictable while the other reader still used it. Now a
  per-reservation `uint64_t res_token` under `cache_lock`, `holders` is
  `vector<uint64_t>`, and the slot stores `writer_token`.
  4 concurrent requests for the SAME frame cost 7.6 ms vs 8.5 ms for 4
  DIFFERENT frames at ns=4, so duplicate reservations do coexist — but no
  divergence could be produced on this box (three repro harnesses + 20 reps
  under `taskset -c 0,1` all bit-identical to the ns=1 oracle). No perf change:
  one u64 increment per frame under a held lock.

### Validation hardening

- `wref` rejects NaN and infinity (`if (wref < 0)` let NaN into the weight
  denominator).
- Frame-request dependencies use `rpGeneral` when `d > 0` (the filter requests
  `n ± d`, which `rpStrictSpatial` forbids; `d = 0` stays strict-spatial).
- Flush/invalidate ranges go through the shared `mapped_range` helper.
- `NLStream` is created via `FramePool::emplace()`, so a creation error is torn
  down by `~NLMeansData` instead of leaking buffers, memory, mapped windows,
  command pool and fence.
- **Upload staging sized for the steady state**: starts at `2*clips*C` tiles
  (was `clips*C*(2d+1)`, memset in full at creation — 263 MiB at 1080p f32
  d=16, 2.1 GiB at ns=8, 4.1 GiB at 8K d=1 of GTT/BAR); `create_staging()`
  grows only when a frame needs more (1–2 new tiles/frame at d=16 in a linear
  decode, more on seeks).
  GTT delta at creation: 1080p f32 d=16 ns=1 271→24 MiB (41→17 ms); ns=8
  2168→191 MiB (213→29 ms); rclip 534→40; UV d=16 137→12; 4K f32 d=6 445→95;
  8K f32 d=1 505→380. Device VRAM unchanged; a random-order 1080p f32 d=16 ns=8
  run that grew the cap 2→33 is bit-exact vs the linear run.
- Wire the dead `VSFEEL_NLMEANS_QUEUES` knob and you lose the FIFO guarantee
  the tile reuse and transfer→shader visibility depend on; it needs semaphores
  first, so the dead call was removed instead.

## Open work

- **Subgroup-shuffle box sums** to cut LDS phases (complex).
- **fp16 dist/hsum LDS arrays** with range scaling — numerics risk.
- **acc src-tile LDS cache** gated on `LAYERS*CH` (~+30 fps est).
- Residual kernel gap: candidates left are cooperative-matrix (WMMA) box-sum
  and launch structure.

## Do not retry

- **Ranked-intermediate depth cut** (BM3D's per-window-list win): no
  application. NLMeans has no top-k or candidate list — it is a dense box-sum
  plus weighted accumulation where every candidate and every `dist`/`hsum` cell
  is read. The register half has nothing to bite on either: the weight kernel
  measures VGPR 24 at 2 workgroups/SIMD, and the guarded-load test above already
  showed that forcing VGPRs to ≤21 for a 3rd workgroup regressed.
- **Compile-time `-DNLM_S=4`**: within noise of the spec-constant version. Do
  not build the 9-value s-variant matrix.
- **wave32 required-subgroup-size pNext**: no effect (unlike DFTTest's 552→673).
- **Manual straight-line unroll of the distance loops**: no effect — the dist
  loads were not serialization-bound.
- **Even-stride LDS matching zipcl's 5120 B** (vs our odd-stride 6144 B): no
  effect; occupancy is not the limiter either way.
- **Grouped all-weights-first**: BROKE correctness — the u4a ring is reused by
  every batch, so batch k+1 overwrites slots batch k still reads. The
  interleaved W→A barrier sequence is load-bearing.
- **Run-merging** (one WG sweeping consecutive-i displacements with LDS-resident
  union tiles): NEUTRAL, 52 µs/batch unchanged. The 65% "dist phase" saving
  measured cold was mostly loads that are already L2 hits warm, so the warm
  weight kernel is ALU/LDS/barrier-bound, not load-bound. Not in the tree (no
  `RUNMAX`/`run_len`/`weight_r`).
- **Round packing >1** (bigger W/A rings): pack=2 tied baseline, pack=4/8 and
  pack=all lost; rings >33 MB stream u4a through DRAM instead of staying
  cache-resident between W and A.
- **Guarded loads** (−6%), VRT 4/6, BY changes, ns=3/4 (cache thrash).
- **Bulk compose memcpy**: impossible into the padded layout; the row loop is
  overhead-bound at ~8 GB/s.

## Debug env vars

Flags are `VSFEEL_NLMEANS_<FLAG>`; the pre-standardisation `NLMEANS_<FLAG>`
spelling still works for one release (read as a fallback, so the new name wins).

- `VSFEEL_NLMEANS_TRACE=1` — `[perf]` host phase averages every 100 frames
  (up/sub/wait/dl); `=2` adds per-phase acq/comp/tab splits and `[grow]`.
- `VSFEEL_NLMEANS_GPUTRACE=1` — per-frame GPU timestamp splits per W/A dispatch.
- `VSFEEL_NLMEANS_PACK=N` — entries per W/A round (clamped 1..16384; default from
  the 64 MiB ring budget).
- `VSFEEL_NLMEANS_HD=0` — force the GTT staging + PCIe upload (no ReBAR
  staging); default is the host-visible VRAM staging when the device has it.
- `VSFEEL_NLMEANS_FORCE_PAD=1` — disables slot reuse entirely. **Currently crashes**
  (heap corruption: the acquire loop leaves `chosen[ti] = -1` and phase 2
  indexes `cache[-1]`, pre-existing). No test uses it.
- `VSFEEL_NLMEANS_PROBE=1` — page the slots map. `VSFEEL_NLMEANS_DBG=1` — dump the
  slot map and use the debug pool.
- `NLMEANS_TS_MAX` (130) and `NLMEANS_TS_RESERVED` (4) are C++ constants in
  `nlmeans.cpp`, and `NLMEANS_PROBE_ACC_NOSRC` is a compile-time `-D` in
  `nlmeans.comp` — none of the three is an env var.
- `VSFEEL_NLMEANS_PACK`/`VSFEEL_NLMEANS_HD` are the runtime tuning knobs; there
  is no queue knob.
- `RADV_DEBUG=asm` / `RADV_DEBUG=shaderstats` work on this Mesa build.

## Default num_streams 1 -> 2

ns=1 is the "cannot overlap" case (the same one DFTTest refuses to ship, having
an in-flight floor of 2). Same-session sweep on the current binary, 500 cached
real-clip frames x3 (medians, bench channels='UV'): ns=1 548.59, ns=2 842.58,
ns=4 838.14 — a 1.54x win at ns=2, and ns=4 is *not* better. The benchmark
already used 2; the plugin default now matches it (and the README table).

Per-stream cost is ~100 MiB at 1080p YUV420P16 d=2/a=2/s=4 (sysfs deltas):
default 200.6 MiB == explicit ns=2 206.7, ns=1 107.6, ns=4 404.8. The 512 MiB
slot-pool budget is unchanged and the created configs still satisfy the
one-full-window requirement; streams stay pinned to the single shared queue.

## Positive maxima (a=64, s=8, d=16) vs the reference

These exercise the `m=0..2` sweep-table variants and the run-group boundaries
the a<=4 / d<=2 sweep never reached. Measured vs vszipcl on the noise clip,
frames 0/11/23, `num_streams=1`:

| config | fp32 max diff | 16-bit codes |
|---|---|---|
| `a=64` (h=1.2) | 2.02e-3 | 133 |
| `s=8,a=64` (h=1.2) | 1.42e-2 | — |
| `a=64` (h=3.0) | 1.61e-6 | 1 |
| `s=8,a=64` (h=3.0) | 1.46e-6 | 1 |
| `d=16` | 4.22e-6 | 1 |
| `d=16,a=64` | 6.97e-5 | 5 |

Only wmode 0 at low h diverges, growing with tap count; h>=3 or wmode 1/2/3
stays ulp-level and self-consistency is exactly 0.0 — the D3 fp16 weight-ring
envelope, not indexing.

**`a=64, d=16` hard-recovers the GPU when the device is shared.** Serially it
is stable (5/5 bit-identical; `s=8` also passes once, so the old "do not add
it" line no longer reproduces at `ns=1`). One instance's weight sweep is
~1.16 s/frame at 640x360 (GPUTRACE slot 2; bandwidth-bound guide re-reads,
unchanged by `VSFEEL_NLMEANS_PACK` 1/9/18 — verified 69/118/180 MB VRAM). Two
concurrent *processes* at this config — or one vsfeel plus one heavy vszipcl
process — lose the context on the 3rd frame; one process with `num_streams=2`
and 6 frames (a 5.96 s frame), 256x256/320x180 clips, and the reference 4-way
concurrent are all fine. `RADV_DEBUG=hang` blames `nlmeans_32_weight.spv`, wave
parked on a memory op (`TCP_BUSY`/`GL1CC_BUSY` set, `vm_fault.log` empty); no
compute TDR applies (the driver enforces none by default). Every loop is
constant- or push-constant-bounded and both `barrier()`s are reached
unconditionally, so the co-scheduled weight sweep itself is the trigger, not an
OOB or a timeout. Do not share the GPU with this configuration; `-n 8 --dist
loadfile` keeps it serial per file.
