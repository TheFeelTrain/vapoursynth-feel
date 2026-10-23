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
- Sweep tables (`wq`/`aq`, stride-8 rows, `qb` batches, variants by
  `m=min(d,n)`), the f32 weight ring, `first`-flag init and the finish in the
  last acc round are unchanged from the pre-R80 implementation.
- Three io variants — u16, f16, f32 — selected by the input format (half is
  `stFloat` + 16 bits). A half element is a storage width, not a precision: the
  temporal window holds the clip's own 2-byte samples, every weight,
  accumulator and LDS tile stays f32 exactly as the reference's own u4a/u2/u5,
  and the finish store narrows once with `OpFConvert`'s round-to-nearest-even
  (vszipcl's `vstore_half_rte`).
- Accuracy: ≤1 LSB u16, ≤4.4e-6 f32 and ≤1 half step f16 against vszipcl across
  the whole parameter surface (spatial, temporal, joint, rclip, cropped, `a=64`,
  `d=16`).
- `num_streams` and `device_id` are registered no-ops (never read): depth is
  the core's exec ring, device choice is `core.set_vulkan_device`.

Scoreboard — jpbd 1080p YUV420P16, `d=2 a=2 s=4 h=0.2 wmode=0 wref=1
channels=UV`, 3000 frames, medians of 3 through `tools/benchmark.py`:

| mode | vsfeel | vszipcl | knlmvk | ratio |
|---|---|---|---|---|
| `--cpu-cache --download` (CPU cache) | 1020.9 | 550.9 | 622.0 | 1.64x |
| `--download` (resident GPU input) | 1026.7 | 460.0 | 625.5 | 1.64x |

- The two modes agree within 1%: the frame is the interleaved sweep rounds, and
  the transfers ride along on the transfer queue.
- GRAY16 (all planes processed) was at parity in a manual same-session pair
  (281 vs 279 fps, ~0.7%); its absolute fps are void with the rest of the
  pre-fix numbers, but the parity it showed is a relative result.

io depth (default: resident GPU input, no download, same config, medians of 3):

| io | vsfeel | vs u16 | vszipcl |
|---|---:|---:|---:|
| u16 | 1033.1 | — | 443.2 |
| f16 | 1069.4 | +3.5% | 457.1 |
| f32 | 953.4 | −7.7% | 465.8 |

- The half window is 2 bytes like u16's, and the sweep (not the transfers) is
  the frame, so f16 lands +3.5% over u16 and +12.2% over f32. The io width is
  the whole difference: the kernels are the same shape at every depth.

The f32 weight ring (2026-10-02) replaces the fp16 one and costs **−7.2%** on
the same config: same-session pair, 1080.2 → 1002.4 fps (vszipcl control flat at
548.9/551.4, n=3 each). The rings are the same size (576 slots, pack 36); only
the element width doubled (128 MiB budget).

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
  the fill is followed by `_mm_sfence()` before submit; the sweep tables get
  the same fence at creation. The layout is `[source half][guide half]`; with no
  `rclip` the guide half repeats the source addresses and `GUIDE_OFF` is 0.
- Window geometry: `PAD = a`, `PSTRIDE = (width + 2a + 7) & ~7`,
  `PH = height + 2a`, one tile of `PSTRIDE*PH` elements per `(clip, channel,
  layer)`. The `PAD` margin is the reference's zero border, so the sweep needs
  no bounds checks. The compose writes the whole tile (margins zeroed) and the
  interior is a straight copy of the core's plane at its own row stride.
- The per-frame window has no explicit cap (the old 512 MiB slot budget is
  gone); an over-large window surfaces as a core allocation failure, not a
  creation-time rejection.
- `num_streams`/`device_id` stay registered so existing scripts load; both are
  never read (see the banner).

### Kernels (`src/nlmeans.comp`)

- `ENTRY_COMPOSE`: one thread per padded tile element; `PAD`/`PSTRIDE` margins
  write 0, the interior loads the core's plane through the address table
  (buffer reference) and stores raw.
- `ENTRY_WEIGHT`: unchanged box-sum/weight transform; `BX=32`, `BY=8`,
  `VRT_RESULT=3`, the interior tile+halo fast path, and the exact per-instance
  LDS sizing (`dist[VRT*BY+2S][BX+2S]`, `hsum[...][BX]`) are the pre-R80 ones.
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

- **The f32 weight ring is an accuracy purchase, not a free one.** The fp16
  ring (×4096 to dodge the subnormal cliff) quantises every weight to 2^-11
  relative, which on real content drifts the weighted average by up to 2.1e-3
  f32 / ~180 codes u16 at `a=64` and ~6 codes on the cropped geometry - the
  recorded 1 LSB band held only on the old near-black clip, where every weight
  sat on exactly 4096. f32 collapses the whole comparison surface to ≤1 LSB /
  ≤4.4e-6 for −7.2%: the ring carries 2x the bytes at the same pack, which the
  measured optimum wants at 128 MiB rather than 64 (halving pack costs more in
  rounds than the wider ring costs in DRAM traffic).

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
  the copy is ~2% of the frame at `d=2` (10.5 MiB window at 1080p UV) and
  shrinks relative to the sweep as `d` grows. GRAY16 hid the same instruction
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

The pre-R80 design and every round that shaped it, kept for the mechanisms:

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
- **R80 port** (`src/nlmeans.cpp` 1818 → ~1090 lines). Deleted: the slot pool
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
  write-combining; `_mm_sfence()` before submit.
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
- **The ring's DRAM traffic is the one f16 lever that pays** (+10.6% alone), but
  only the accuracy budget can unlock it: 2^-11 relative weight quantisation is
  ~180 codes u16 at `a=64`. A scaled half ring that keeps more range (a shared
  exponent per tile, or block floating point) is the untried route; the plain
  fp16 ring is already rejected (see the banner's f32-ring round).
- **Subgroup-shuffle box sums** to cut LDS phases (complex).
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
- **Round packing >1** (bigger W/A rings): pack=2 tied, 4/8 lost; rings >33 MB
  stream u4a through DRAM. Its 10-binding push-descriptor layout is bounded by
  `maxPushDescriptors` (checked at layout creation; the per-stage pool limits do
  not apply to push descriptors), which RADV reports at the minimum of 32. The 64 MiB target is capped by the core's VRAM limit,
  so a device with a smaller allowance packs fewer entries per round instead of
  planning the optimum anyway.
- **`a=64, d=16` hard-recovers the GPU when the device is shared** — a
  co-scheduled weight sweep is the trigger, not an OOB or a timeout. Do not run
  it beside another GPU client; `-n 8 --dist loadfile` keeps it serial per file.

## Debug env vars

Flags are `VSFEEL_NLMEANS_<FLAG>`.

- `VSFEEL_NLMEANS_TIMING=1` — per-frame host stage split (acquire/record/submit).
- `VSFEEL_NLMEANS_GPUTRACE=1[,frame]` — one-shot per-batch GPU timestamps.
- `VSFEEL_NLMEANS_PACK=N` — entries per W/A round (clamped 1..16384, and so the
  weight dispatch's Z fits the device's group-count limit; default from the
  64 MiB ring budget, itself capped by the core's VRAM limit).
- `VSFEEL_NLMEANS_VRAM=1` — creation banner with ring/window/per-frame bytes.
- `VSFEEL_DEBUG`/`VSFEEL_TRACE` and `RADV_DEBUG=asm|shaderstats` as elsewhere.
