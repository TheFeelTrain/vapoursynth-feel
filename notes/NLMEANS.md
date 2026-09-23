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
  submit. Scratch (window, fp16 weight ring, u2, u5) is per-frame from
  `createGPUBuffer`, retired by the submission.
- The sweep kernels keep the pre-R80 padded-tile indexing (`(y+PAD)*PSTRIDE`,
  `layer*TILE_ELEMS`), i.e. 32-bit offsets into one storage buffer. Reading the
  core's per-layer planes directly through buffer references costs 64-bit
  addressing per load and ~10% on the chroma benchmark (see Performance).
- Sweep tables (`wq`/`aq`, stride-8 rows, `qb` batches, variants by
  `m=min(d,n)`), the fp16 weight ring, `first`-flag init and the finish in the
  last acc round are unchanged from the pre-R80 implementation.
- Accuracy: identical to the pre-R80 implementation on every tested config
  (≤1 LSB 16-bit, ≤6.6e-5 fp32 vs vszipcl).
- `num_streams` and `device_id` are accepted and ignored: depth is the core's
  exec ring, device choice is `core.set_vulkan_device`.

Scoreboard — jpbd 1080p YUV420P16, `d=2 a=2 s=4 h=0.2 wmode=0 wref=1
channels=UV`, 3000 frames, `tools/benchmark.py --filter nlmeans vsfeel`,
interleaved pre-port/R80 pairs, `--repeat 3` (vszipcl control flat at
~755 fps throughout):

| input to vsfeel | pre-R80 | R80 port | delta |
|---|---|---|---|
| CPU cache (benchmark default) | 1082.9 | 1080.3 | **−0.2%** |
| `--gpu-cache` (GPU clip consumed) | 1065.9 | 1083.4 | **+1.6%** |

GRAY16 (all planes processed) is at parity in a manual same-session pair
(281 vs 279 fps, ~0.7%).

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
- `num_streams`/`device_id` are accepted for compatibility and no longer select
  anything; `num_streams` is a registered no-op and `device_id < 0` is still
  an error.

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
  specialization constants; only the io type is a compile-time `-DBITS`.
- `--target-env=vulkan1.4` (SPIR-V 1.6): the compose pass names the core's
  planes through buffer references (`PhysicalStorageBuffer`).

## Performance

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
- **Transfers.** `RADV_EXPERIMENTAL=transfer_queue` is required and
  `tools/benchmark.py` forces it: without the SDMA family the core's
  `GPUDownload` runs on the graphics engine. With it, the CPU-in default is at
  parity and the GPU-in default is slightly ahead.
- **GPU in / CPU out** (`--gpu-cache`): +1.6% for the port, which reads the
  core's planes in place instead of downloading the clip for its CPU filter.
- **The window is rebuilt every frame.** A persistent GPU slot cache copying
  only the layers that newly entered the window would cut the compose cost, but
  at the shipped `d=2` it is ~2% and not worth the holder/lifetime machinery the
  port exists to delete. Revisit only if a large-`d` config measures the copy
  above a few percent.

## Historical

The pre-R80 design and every round that shaped it, kept for the mechanisms:

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
  interior (+32%); fp16 weight ring ×4096 (drift ≤5 LSB / ~1e-4).
- **fp16 ring cliff**: unscaled weights <6e-5 lose mantissa bits exponentially;
  `w*4096` on store, `1/4096` on load fixed it. `finish_sample` returns the
  centre sample on a zero total weight instead of 0/0.
- **Boundary/rclip fixes**: the guide half's slot table used the full-window
  stride when `n < d` and read past the vector (fixed key offset), and holder
  identity is a per-reservation token, not a frame index.

## Open work

- **Subgroup-shuffle box sums** to cut LDS phases (complex).
- **fp16 `dist`/`hsum` LDS arrays** with range scaling — numerics risk.
- **Incremental window cache**: copy only newly-entered layers instead of
  rebuilding the whole window, if a large-`d` workload ever makes the copy
  matter.
- Extreme configs: the per-frame window has no explicit cap (the old 512 MiB
  slot budget is gone); an over-large window now surfaces as an allocation
  failure from the core rather than a creation-time rejection.
- Residual kernel gap vs vszipcl: cooperative-matrix (WMMA) box sums and launch
  structure.

## Do not retry

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
  stream u4a through DRAM.
- **`a=64, d=16` hard-recovers the GPU when the device is shared** — a
  co-scheduled weight sweep is the trigger, not an OOB or a timeout. Do not run
  it beside another GPU client; `-n 8 --dist loadfile` keeps it serial per file.

## Debug env vars

Flags are `VSFEEL_NLMEANS_<FLAG>`.

- `VSFEEL_NLMEANS_TIMING=1` — per-frame host stage split (acquire/record/submit).
- `VSFEEL_NLMEANS_GPUTRACE=1[,frame]` — one-shot per-batch GPU timestamps.
- `VSFEEL_NLMEANS_PACK=N` — entries per W/A round (clamped 1..16384; default
  from the 64 MiB ring budget).
- `VSFEEL_NLMEANS_VRAM=1` — creation banner with ring/window/per-frame bytes.
- `VSFEEL_DEBUG`/`VSFEEL_TRACE` and `RADV_DEBUG=asm|shaderstats` as elsewhere.
