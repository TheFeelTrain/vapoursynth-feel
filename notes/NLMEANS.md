# NLMeans — notes

Status: **shipped on the R80 GPU API.** Verified against
`src/nlmeans.{cpp,comp}`, `CMakeLists.txt` and `tools/benchmark.py`.

- `clip:vnode:gpu` in, `ffGPUOutput` out: the core owns every transfer and the
  filter records one submission per output frame into one exec pool. No slot
  pool, frame cache, staging, host upload/download, per-stream buffers,
  fences, command pools, descriptor pool or custom queue.
- Per output frame: request the `2d+1` (clamped) source frames, acquire a
  context, build a tiny address table of every `(clip, channel, layer)` plane
  address, **compose** those planes once into one zero-padded device-local
  window, barrier, then the interleaved weight/accumulation sweep rounds,
  submit. Scratch (window, f32 weight ring, u2, u5) is per-frame from
  `createGPUBuffer`, retired by the submission.
- The sweep kernels keep the pre-R80 padded-tile indexing (`(y+PAD)*PSTRIDE`,
  `layer*TILE_ELEMS`), i.e. 32-bit offsets into one storage buffer. Reading the
  core's per-layer planes directly through buffer references costs 64-bit
  addressing per load and ~10% on the chroma benchmark (see Performance).
- Sweep tables (`wq`/`aq`, stride-8 rows, rounds of `batch` entries, variants by
  `m=min(d,n)`), the f32 weight ring, `first`-flag init and the finish in the
  last acc round are unchanged from the pre-R80 implementation.
- Three io variants — u16, f16, f32 — selected by the input format (half is
  `stFloat` + 16 bits). A half element is a storage width, not a precision: the
  temporal window holds the clip's own 2-byte samples, every weight,
  accumulator and LDS tile stays f32 exactly as the reference's own u4a/u2/u5,
  and the finish store narrows once with `OpFConvert`'s round-to-nearest-even
  (vszipcl's `vstore_half_rte`).
- u16 transport divides by 65535 on load and uses the reference's saturating
  round-to-nearest-even store, so its code values match vszipcl's conversion;
  the earlier reciprocal-multiply/half-up pair is gone.
- Accuracy: ≤1 LSB u16 (≤2 on the joint YUV444 path), ≤4.4e-6 f32 and ≤1 half
  step f16 against vszipcl across the whole parameter surface (spatial, temporal,
  joint, rclip, cropped, `a=64`, `d=16`).
- `num_streams` and `device_id` are registered no-ops (never read): depth is
  the core's exec ring, device choice is `core.set_vulkan_device`.

Scoreboard — jpbd 1080p YUV420P16, `d=2 a=2 s=4 h=0.2 wmode=0 wref=1`,
medians of 3 (5 s window) through `tools/benchmark.py`:

| `channels` | vsfeel | vszipcl | knlmvk | vs vszipcl | vs knlmvk |
|---|---|---|---|---|---|
| `UV` (chroma) | 1270.7 | 456.8 | 599.6 | 2.78x | 2.12x |
| `Y` (luma) | 329.7 | 175.1 | 239.5 | 1.88x | 1.38x |

Transfers, `channels=UV`:

| mode | vsfeel | vszipcl | knlmvk | vs knlmvk |
|---|---|---|---|---|
| `--cpu-cache --download` (CPU cache) | 1260.0 | 561.8 | 647.2 | 1.95x |
| `--download` (resident GPU input) | 1272.0 | 460.1 | 649.7 | 1.96x |

- The two transfer modes agree within 1%: the frame is the interleaved sweep rounds, and
  the transfers ride along on the transfer queue.
- Luma cannot reach the chroma ratio: one weight plane serves U and V jointly,
  so a chroma pixel buys two output channels' worth of weight, while a luma
  pixel pays a whole weight plane for one channel (see Performance).

io depth (default: resident GPU input, no download, same config, medians of 3):

| io | vsfeel | vs u16 | vszipcl |
|---|---:|---:|---:|
| u16 | 1273.5 | — | 461.0 |
| f16 | 1311.8 | +3.0% | 452.6 |
| f32 | 1141.6 | −10.4% | 488.3 |

- The half window is 2 bytes like u16's, and the sweep (not the transfers) is
  the frame, so f16 lands +3% over u16 and +15% over f32. The io width is
  the whole difference: the kernels are the same shape at every depth.

The f32 weight ring (2026-10-02) replaces the fp16 one for accuracy: 2^-11
relative weight quantisation drifts the average by up to ~2e-3 f32 / ~180 codes
u16 at `a=64`, while f32 holds the whole comparison surface at ≤1 LSB /
≤4.4e-6. The −7.2% it cost when widened was measured under the superseded fixed
128 MiB ring budget; the round sizing below absorbs its 2x bytes.

## Implementation

### Frame path (`nlmeans_gpu_frame`, `src/nlmeans.cpp`)

- `arInitial` requests layers `clamp(n-d+l)` for `l` in `[0, 2d]`;
  `arAllFramesReady` re-requests them (the core dedups the clamped repeats),
  takes the output frame (`newGPUVideoFrame` when every plane is processed,
  `newVideoFrame2` sharing the unprocessed planes from the centre otherwise)
  and acquires one exec context.
- Per frame, in one command buffer: address table (binding 0) → compose
  (one dispatch, `z = clips*C*(2d+1)` tiles) → barrier → the weight/acc rounds
  → `gpuExecReadsFrame` per distinct source, `gpuExecWritesPlane` per processed
  plane → `gpuExecSubmit`.
- The address table is a small host-visible buffer of `2*C*(2d+1)`
  `VkDeviceAddress` values, filled with `getGPUPlane` +
  `vkGetBufferDeviceAddress`. It may land in the write-combining VRAM BAR, so
  the fill is followed by `store_fence()` before submit; the sweep tables get
  the same fence at creation. The layout is `[source half][guide half]`; with no
  `rclip` the guide half repeats the source addresses and `GUIDE_OFF` is 0.
- Window geometry: `PAD = a`, `PSTRIDE = (width + 2a + 7) & ~7`,
  `PH = height + 2a`, one tile of `PSTRIDE*PH` elements per `(clip, channel,
  layer)`. The `PAD` margin is the reference's zero border, so the sweep needs
  no bounds checks. The compose writes the whole tile (margins zeroed) and the
  interior is a straight copy of the core's plane at its own row stride.
- Per-frame size is not capped in bytes; a window whose element counts pass
  2^31, or whose buffers exceed `maxStorageBufferRange`, is a clean creation error
  (the frame-time allocation keeps the check as a backstop) instead of an
  out-of-bounds access.
- `num_streams`/`device_id` stay registered so existing scripts load; both are
  never read (see the banner).

### Kernels (`src/nlmeans.comp`)

- `ENTRY_COMPOSE`: one thread per padded tile element; `PAD`/`PSTRIDE` margins
  write 0, the interior loads the core's plane through the address table
  (buffer reference) and stores raw.
- `ENTRY_WEIGHT`: unchanged box-sum/weight transform; `BX=32`, `BY=8`, the
  interior tile+halo fast path, and the exact per-instance LDS sizing
  (`dist[VRT*BY+2S][BX+2S]`, `hsum[...][BX]`) are the pre-R80 ones. The border
  path walks tile rows instead of a fixed chunk unroll, so the tile height is a
  free parameter: `VRT_RESULT` is 6 (48 rows) or 3 (24 rows), chosen at creation
  from the device's `maxComputeSharedMemorySize` (the tall tile needs 17-20 KiB
  at `s>=5`, over the 16 KiB Vulkan 1.4 guarantees) and shipped as
  `-DNLM_VRT` variants, 24-row weight only. `VSFEEL_NLMEANS_TILE=48|24` forces
  one for an A/B.
- The u16 load is one `precise` multiply by 1/65535. Without `precise` the
  batched interior path let the compiler fold the scale into the sample
  difference (`(a-b)*k` for `a*k - b*k`), so a pixel's weight depended on which
  tile computed it; every path is bit-identical now. Costs <=0.5% on the u16
  benchmark (three A/B pairs against the unfolded build: +0.05%, +0.47%,
  -0.45%), which is the run-to-run spread.
- `ENTRY_ACC`: unchanged per-pixel `+q/-q` accumulation with `first`-flag init
  and the finish folded into the last round (`pc3`).
- `VI_DIM`, `STRIDE`, `PSTRIDE`, `PH`, `PAD`, `NLM_S`, `NLM_D`, `NLM_REF`,
  `NLM_CHANNELS`, `WMODE`, `WREF`, `H2_INV_NORM` and `GUIDE_OFF` are
  specialization constants; the io type is a compile-time `-DIO_U16`/`-DIO_F16`/
  `-DIO_F32` (three SPIR-V variants per entry point, selected from the input
  format: half is `stFloat` + 16 bits). A half element widens exactly on load
  and narrows once on the finish store; the window is the only buffer whose
  element type follows the io depth.
- `--target-env=vulkan1.4` (SPIR-V 1.6): the compose pass names the core's
  planes through buffer references (`PhysicalStorageBuffer`). The half variant
  makes those references `float16_t`, which SPIR-V 1.6 accepts under
  `StorageBuffer16BitAccess` + `PhysicalStorageBufferAddresses` (validated; it
  needs no `PhysicalStorageBuffer16BitAccess`); the half LDS tiles a full-fp16
  build would need are the only thing that pulls in the explicit-arithmetic-types
  extension, one more reason that path is not free (see Open work).

### Tests

`tests/test_nlmeans.py` runs the u16/f32 parameter surface at every depth and
mirrors it for f16 (sweeps vs vszipcl and knlmvk, positive maxima, UV/joint/
rclip/stride paths, determinism, parallel load, frame-request order, wref=0
finiteness and the bit-exact centre fallback), plus the device-independent
oracle that the half output is the f32 pipeline narrowed once. The half path is
also in `tests/test_geometry.py`'s cropped-width matrix and in
`tests/test_validation.py`'s validation-layer smoke (`NLMeans_f16`).

## Performance

- **Round sizing is a cache-residency decision.** One weight+acc round's live
  set is the f32 ring, the accumulators and the window; it is sized to 96 MiB,
  the card's Infinity Cache, so the accumulation reads a round's weights from L2
  instead of DRAM. At 1080p that is batch 3 (21 rounds) for luma and batch 19
  (4 rounds) for chroma; the old fixed 128 MiB ring gave 8 and 2 rounds and made
  every round's 133 MiB of weights re-read from DRAM. Graded medians:
  `channels=Y` 249 → 330 fps, `UV` 1042 → 1271 fps. The trade is the extra
  rounds' accumulator traffic, which grows as 1/batch; batch 2/5/6/8 measured
  272/272/263/247 fps against 293 at 3-4 (400-frame screen, `VRT_RESULT=3`).
- **The weight tile is 32x48 (`VRT_RESULT=6`).** A taller tile cuts the separable
  box sum's halo overfetch (40x32 dist cells per 32x24 outputs → 40x56 per
  32x48) and the hsum cells per output (1.33 → 1.17). 400-frame probe ladder:
  294 (3) → 314 (5) → 322 (6) → 296 (7) → 299 (8) fps; 7+ loses occupancy.
  LDS is 16 KiB at `s=4`, still 16 subgroups/SIMD. The box sum itself runs at
  ~80% of the LDS issue rate at `s=4`, so it is the pipeline floor, not codegen.
- **The f32 weight ring is an accuracy purchase, not a free one.** The fp16
  ring (×4096 to dodge the subnormal cliff) quantises every weight to 2^-11
  relative, which on real content drifts the weighted average by up to 2.1e-3
  f32 / ~180 codes u16 at `a=64` and ~6 codes on the cropped geometry - the
  recorded 1 LSB band held only on the old near-black clip, where every weight
  sat on exactly 4096. f32 collapses the whole comparison surface to ≤1 LSB /
  ≤4.4e-6; its 2x bytes are paid inside the cache-sized round above, so the
  −7.2% that widening first cost does not apply to the shipped sizing.

- **Full-fp16 internals are a measured dead end: +10.4%.** With the half io
  window already shipped, a probe that also narrowed the weight ring, the
  accumulators and the LDS tiles to half (no range management, output wrong by
  design) measured 1053.7 → 1162.8 fps; narrowing only the ring measured
  1056.2 → 1168.2 fps (f16 io, default: resident GPU input, no download, 3000
  frames, medians of 3). So the ring is the whole of it and the half
  LDS/accumulators are slightly negative, the same mechanism as
  `notes/BILATERAL.md` and `notes/GAUSSBLUR.md`: a scalar half access still
  takes a load slot and every tap adds a widening convert, and ACO emits scalar
  `v_fma_f16`, never a packed `v_pk_*`. The fp16 ring alone already fails the
  accuracy bar (2^-11 relative, ~180 codes u16 at `a=64`), so f16 ships mixed
  precision.
- **Where the half io win comes from.** The window halves to the clip's own
  2-byte samples, so the f16 sweep reads half the bytes per guide/source load
  that the f32 io arm does; weights, accumulators and LDS are identical at every
  depth. Compute-only f16 is +3.5% over u16 (same 2-byte io, one 4-byte ring
  load per entry unchanged) and +12.2% over f32 (the window loads halve).
- **Half costs one store rounding, not a precision loss.** Over the 47-case f16
  comparison surface (640x360 fixture, frames 0/11/23) the residual vs vszipcl is
  one half step, 4.88e-4 at |v| in [0.5, 1), and against the same-input f32 run
  narrowed once it is half a step, 2.44e-4. The RGB joint path measures one step
  in the [1, 2) binade, 9.77e-4, because the RGB fixture's Bicubic matrix
  conversion overshoots the unit peak (max 1.11) and doubles the ulp there; the
  test bound is 2e-3 over the worst of the 47.
- **`wref=0` has no usable oracle on real content.** The reference degenerates
  there (0 total weight: 53983 non-finite samples on frame 0 at `h=1.2`, finite
  ones dividing by a denormal sum up to 0.8 off; knlmvk's same fallback is 0.14
  off). Pinned instead: finiteness, and the exact source bit-for-bit fallback
  at `h <= 0.1` / `wmode` 1-3.
- **knlmvk is a spatial-only second oracle**: bit-for-bit f32, ≤1 LSB u16 at
  `d=0`; its temporal pairing differs (3.7e-2 at `d=2`), as it does from vszipcl.
- **`d=16` with `a=64` stays out of the comparison sweep**: it hard-recovers a
  shared GPU (see Do not retry); the neighbouring `a=64` entries are all within
  4.4e-6 on the f32 ring.
- **The compose pass exists for codegen, not correctness.** Reaching each
  `(channel, layer)` plane directly with buffer references cost 64-bit address
  arithmetic on every guide load: the UV weight kernel compiled to 1515 ACO
  instructions / VGPR 48 against the pre-R80 1055 / VGPR 24 (+44%, both at the
  16-subgroups/SIMD cap) and ran ~10% slower at 1080p `channels='UV'`. Copying
  the window once per frame into one device-local buffer restores the
  `buffer_load`/32-bit-offset form (1055 instructions) and the gap goes away;
  the copy is ~2% of the frame at `d=2` (10.0 MiB window at 1080p UV). Compose
  work scales with `(2d+1)` and the sweep with `(2d+1)(2a+1)^2`, so their ratio
  is independent of `d`, not shrinking with it. GRAY16 hid the same instruction
  increase behind its transfer cost, which is why the first A/B looked
  plane-count dependent.
- **Transfers** (mechanism: `notes/BILATERAL.md`): with the SDMA opt-in the
  CPU-fed chain (`--cpu-cache --download`) is at parity and `--download` is
  +1.6% (core planes in place).
- **The window is rebuilt every frame.** A persistent GPU slot cache copying
  only the layers that newly entered the window would cut the compose cost, but
  at the shipped `d=2` it is ~2% and not worth the holder/lifetime machinery the
  port exists to delete. Revisit only if a large-`d` config measures the copy
  above a few percent.

## Historical

- **2026-11-07 - the weight tile follows the device's workgroup memory.** The
  single 48-row shape needs 17,168/18,240/19,344/20,480 B at `s=5..8`, over the
  16 KiB a Vulkan 1.4 device may report, so `s>=5` failed creation there. The
  tile is now `VRT_RESULT` 6 or 3 by `maxComputeSharedMemorySize`, with a
  24-row weight module per io type; 16 KiB devices run `s=8` at 12,800 B and
  the pixels are identical, which the u16 `precise` fix above made true (it
  was not before: a pixel's weight depended on its tile). The occupancy idea is measured and rejected: on the target
  the 24-row tile is 1.9% slower at `s=6` (1020 vs 1001 fps) and 14.8% slower
  at `s=8` (946 vs 806 fps), `d=2 a=2 h=0.2` 1080p UV16 medians of 12, because
  the extra halo work outweighs the 3 -> 5 resident workgroups, so the tall
  tile stays the default wherever it fits.

The pre-R80 design and every round that shaped it, kept for the mechanisms:

- **2026-11-01 — round sizing was past the Infinity Cache.** The f32 ring's fixed
  128 MiB budget made a round's live set 130 MiB+, so the accumulation re-read
  every weight plane from DRAM; sizing the round to the 96 MiB cache and taking
  the weight tile to 32x48 gave luma 249 → 330 fps and chroma 1042 → 1271 fps
  (jpbd 1080p u16, graded medians). A unorm16 ring measured +2% on luma and lost
  on chroma, so the ring stays f32.
- **2026-11-01 — the `channels` comparison is portable.** `strcasecmp` is
  POSIX-only, so an MSVC build of the filter failed; the call sites now use a
  local `iequals` that selects `_stricmp` on Windows. No behavior or perf
  change.
- **2026-10-18 — f16 io shipped; full-fp16 internals rejected.** The half window
  measured +3.5% over u16 and +12.2% over f32 (default: resident GPU input, no
  download, 3000 frames, medians of 3). The probe that also narrowed the ring,
  accumulators and LDS tiles measured +10.4%, below the 30% bar that would have
  paid for its accuracy loss, so weights/accumulators/LDS stay f32 and half is a
  storage width. The plain fp16 ring's own accuracy failure is the banner's
  earlier round.
- **The R80 port's scoreboard is void.** It read 1082.9 → 1080.3 (CPU cache) and
  1065.9 → 1083.4 (`--download`) fps against the pre-fix benchmark harness
  (`notes/METHOD.md`); the same-session deltas it carried (−0.2%, +1.6%) stand.
  The banner's table replaces the absolute numbers.
- **R80 port** (`src/nlmeans.cpp` 1818 → 1295 lines). Deleted: the slot pool
  (`CacheSlot`, `res_token` holders, `writer_token`/`writer_count`,
  `cache_cv`, the all-or-nothing acquire), host compose into per-stream staging,
  the ReBAR staging option (`VSFEEL_NLMEANS_HD`), per-stream
  `u1z/u2/u4a/u5`/command pools/fences/descriptor sets, the host download
  memcpy, the `submit_count`/`submit_cv` cross-submission handshake, the
  per-stream queue assignment and the 512 MiB slot budget.
- **Speculative out-of-bounds loads (correctness).** Anything selected by `?:`
  — the zero-border candidate, the out-of-frame cell, the mirrored weight — was
  if-converted by ACO into an unconditional load. `PhysicalStorageBuffer`
  accesses get no robust-buffer clamp (and the ring read did not behave as if
  SSBO reads did either), so a far-negative index read unrelated VRAM:
  nondeterministic garbage and periodic device recovery, worst at
  `a=64, d=16, 640x360` where the search reaches 64 px past the frame. A
  determinism probe (same config, same process) separated it from the fp16
  weight envelope. Fixed by clamping the index; the compose pass then removed
  the whole class by giving the sweep a zero margin again.
- **Address-table write fence.** A host-written table in the VRAM BAR is
  write-combining; `store_fence()` before submit.
- **Slot-direct cache + reservation tokens** — a shared pool of padded
  per-(clip,frame,channel) tiles with holder tokens, filled by the host. Worth
  ~1.4x over vszipcl pre-R80; superseded by the core's frame cache.
- **ReBAR upload staging (+7.7%)** — the pre-R80 per-stream staging was
  host-visible VRAM so the host compose wrote VRAM directly.
- **Kernel levers (measured, still shipped).** Exact per-instance LDS sizing
  via spec-constant array dims (+13% weight batch); native u16 io (+14%);
  interior/border tile split with batched loads and no per-cell checks in the
  interior (+32%).
- **fp16 weight ring ×4096**: unscaled weights <6e-5 lose mantissa bits
  exponentially, so `w*4096` on store and `1/4096` on load replaced them; drift
  measured ≤5 LSB / ~1e-4 on the old clip, up to ~180 LSB on real content at
  `a=64` (superseded by the f32 ring above). `finish_sample` returns the centre
  sample on a zero total weight instead of 0/0; that fallback stays.
- **Boundary/rclip fixes**: the guide half's slot table used the full-window
  stride when `n < d` and read past the vector (fixed key offset), and holder
  identity is a per-reservation token, not a frame index.
- **Validation hardening (no perf change).** `h` is rejected unless finite and
  > 0: `+Inf` drove `h2_inv_norm` to 0 and turned the filter into a plain box
  mean where the reference errors. `VSFEEL_NLMEANS_PACK` and the ring-budget
  default are clamped so the weight dispatch's Z — one table row per sweep
  entry, two when the temporal offset is non-zero, so up to `2*qb*pack` — fits
  `maxComputeWorkGroupCount[2]`; a tiny frame could pass the guaranteed 65535 on
  the default alone. The unreachable `any_process` early return is gone (every
  plane0/channels combination processes at least one plane).
- **GPU-trace stage labels.** The probe wrote three stamps per round but the
  summary consumed two, so "weight"/"acc" rotated over weight, acc and the
  inter-round gap and the batch count printed 1.5× the rounds. It now consumes
  three (weight start/end, accum end) and reports the gap separately.

## Open work

- **A packed two-pixels-per-lane half2 sweep is the only untried f16 form.**
  Every half variant tried so far narrows storage only and ACO emits scalar
  `v_fma_f16` (see Performance), so the FMA rate never moves. Packing two output
  pixels per lane in `half2` could reach `v_pk_*`, but the accumulator's range
  (hundreds of thousands at `a=64, d=16`) and the weight ring both need scaling,
  and the u2/u5 interleave has to be repacked. Estimated as a rewrite, not a
  tweak.
- **Subgroup-shuffle box sums** to cut LDS phases (complex). On AMD a shuffle
  lowers to `ds_bpermute`, i.e. the same LDS pipe the box sum already saturates,
  so this trades one LDS instruction for another rather than relieving it.
- **Incremental window cache**: copy only newly-entered layers instead of
  rebuilding the whole window, if a large-`d` workload ever makes the copy
  matter.
- Residual kernel gap vs vszipcl: cooperative-matrix (WMMA) box sums and launch
  structure.

### Do not retry

- **Full-half internals** (ring + LDS + accumulators): +10.4% vs the shipped
  mixed path, and the ring alone is +10.6%, so the LDS/accumulator half reach is
  slightly negative (mechanism in Performance). Do not retry without a shape ACO
  packs into `v_pk_*`.
- **Buffer-reference reads in the sweep kernels** — measured −10% on 1080p
  `channels='UV'` (mechanism above). The compose is cheaper.
- **Compile-time `-DNLM_S=4`**: within noise of the spec-constant version.
- **wave32 required-subgroup-size pNext**: no effect here.
- **Manual straight-line unroll of the distance loops**: no effect.
- **Ranked-intermediate depth cut** (BM3D's per-window-list win): no
  application — NLMeans has no top-k or candidate list.
- **Grouped all-weights-first**: BROKE correctness — the u4a ring is reused by
  every batch; the interleaved W→A barrier order is load-bearing.
- **Run-merging** (one WG sweeping consecutive-i displacements with
  LDS-resident union tiles): neutral; the warm weight kernel is ALU/LDS/barrier
  bound, not load bound.
- **A narrower weight ring does not pay: unorm16 measured +2%.** A weight is
  already in [0,1], so 16-bit fixed point has a clean ≤0.5 LSB bound and the
  round it funds doubles the batch (luma batch 3 → 6, chroma 19 → 39). It
  measured 330 vs 322 fps on luma and 1160 vs 1221 on chroma (400-frame screen):
  the accumulation is bound per entry-pixel, not by ring bytes, so halving the
  ring's width is not worth its accuracy cost. f32 stays.
- **A bigger round than the cache allows loses.** Past the 96 MiB live set the
  accumulation re-reads the round's ring from DRAM: luma batch 5/6/8 measured
  272/263/247 fps against 293 at 3-4 (400-frame screen). The old note's
  "pack=2 tied, 4/8 lost" was the fp16 ring at the fixed 128 MiB budget. The
  10-binding push-descriptor layout is bounded by `maxPushDescriptors` (checked
  at layout creation; the per-stage pool limits do not apply to push
  descriptors), which RADV reports at the minimum of 32. The ring target is
  capped by the core's VRAM limit, so a device with a smaller allowance plans a
  smaller round.
- **`a=64, d=16` hard-recovers the GPU when the device is shared** — a
  co-scheduled weight sweep is the trigger, not an OOB or a timeout. Do not run
  it beside another GPU client; `-n 8 --dist loadfile` keeps it serial per file.

## Debug env vars

Flags are `VSFEEL_NLMEANS_<FLAG>`.

- `VSFEEL_NLMEANS_TIMING=1` — per-frame host stage split (acquire/record/submit).
- `VSFEEL_NLMEANS_GPUTRACE=<frame>` — one-shot per-batch GPU timestamps
  (default 100; the value is the first traced frame).
- `VSFEEL_NLMEANS_PACK=N` — entries per weight+acc round (default: the largest
  that fits the 96 MiB live-set target; clamped 1..16384, at most the sweep's
  entry count and the device's Z group-count limit).
- `VSFEEL_NLMEANS_VRAM=1` — creation banner with ring/window/per-frame bytes.
- `VSFEEL_NLMEANS_TILE=48|24` — force the weight tile's height instead of
  letting the workgroup-memory budget choose it.
- `VSFEEL_DEBUG`/`VSFEEL_TRACE` and `RADV_DEBUG=asm|shaderstats` as elsewhere.
