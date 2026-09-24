# DFTTest — notes

Status: **shipped on the R80 GPU API.** Verified against `src/dfttest.{cpp,comp}`.
Target GPU: RX 7900 XTX (RDNA3, gfx1100), Mesa 26.2 RADV.

- `clip:vnode:gpu` in, `ffGPUOutput` out: the core owns every transfer and the
  filter records one submission per output frame into one exec pool. No staging
  buffers, ReBAR upload, slot frame cache, per-slot timelines, fences or
  filter-owned descriptor sets.
- Per output frame and processed plane: `pad` (one dispatch per temporal slice,
  reading that slice's own source plane) → barrier → `fused` → barrier →
  `col2im` straight into the output plane. Scratch is two `createGPUBuffer`
  allocations per frame, handed to the context.
- **Runs need `RADV_EXPERIMENTAL=transfer_queue`**, which `tools/benchmark.py`
  now forces: RADV gates its SDMA transfer family behind that opt-in, and
  without it every GPU filter's `GPUDownload` runs on the graphics engine and
  costs 10–31%. Mechanism in `notes/BILATERAL.md`.

Scoreboard — jpbd 1080p GRAY16, 3000 frames, `tools/benchmark.py --filter
dfttest vsfeel`, interleaved pre-port/R80 pairs, `--repeat 2`:

| input to vsfeel | pre-R80 | R80 port | delta |
|---|---|---|---|
| CPU cache (benchmark default) | 1452.6 / 1376.0 | 1361.3 / 1360.3 | **−6% … −1%** |
| `--gpu-cache` (GPU clip consumed) | 1182.3 | 1350.3 | **+14%** |
| `--synthetic` BlankClip | 1376.2 | 1364.7 | −1% |

- vs `vszipcl` in a separate same-session pair (3000 frames, `--repeat 2`):
  1254 vs 801 fps, i.e. the port is **1.57x** the reference.
- The transfer was the whole pre-port gap: the default run measured −18.9%
  before the opt-in, and chained instances (middle frames stay in VRAM, so only
  the ends transfer) measured 640 µs pre-port vs 666 µs per instance. The ~26 µs
  is the two extra pad dispatches the port does by re-padding all `tw` slices
  instead of reusing a slot.

## Implementation

### Frame path (`dft_gpu_frame`, `src/dfttest.cpp`)

- `arInitial` requests the `2*radius+1` source frames; `arAllFramesReady` takes
  them, makes the output frame (`newGPUVideoFrame` when every plane is
  processed, `newVideoFrame2` sharing the unprocessed planes from the centre
  frame otherwise) and acquires one exec context.
- One recording per output frame, per processed plane: `pad` (one dispatch per
  temporal slice, reading that slice's own source frame plane) → barrier →
  `fused` → barrier → `col2im` (straight into the output frame plane). Then
  `gpuExecReadsFrame` per distinct source, `gpuExecWritesPlane` per processed
  plane, `gpuExecSubmit`.
- Per-frame scratch is two `createGPUBuffer` allocations per plane — the padded
  window (`tw*pw*ph*bytes`) and the float block buffer (`num_blocks*256*4`) —
  handed to the context with `gpuExecUsesBuffer`. Allocation is measurably free:
  at 1080p GRAY16 that is 12.97 + 136.0 MB per in-flight frame (142.1 MiB,
  `VSFEEL_DFFTEST_VRAM=1` banner), and aliasing the block buffer onto the padded
  one changes the frame time by <1%. Depth is the core's exec ring (2..8
  contexts), not `num_streams`.
- Push descriptors with five whole-buffer bindings (`wt`, `padded`, `spatial`,
  `src` plane, `dst` plane). Binding 3 is a different source frame per pad
  dispatch, which is why nothing is allocated from a descriptor pool.
- `num_streams` and `device_id` are registered no-ops (never read): depth is
  the core's call, device choice is `core.set_vulkan_device`.

### Kernels (`src/dfttest.comp`)

- `ENTRY_PAD` reads the core's source frame plane with its own row stride and
  writes slice `pad_t0` of the padded buffer; `ENTRY_FUSED` is im2col + window +
  3D DFT + frequency filter + inverse, writing the centre temporal slice of each
  block into the float block buffer; `ENTRY_COL2IM` overlap-adds into the output
  plane with its stride.
- `RADIUS` (0..3) is a compile-time define so the temporal loops unroll and
  `td[]` stays in registers; `FILTER_TYPE` and `ZMEAN` are specialization
  constants (the dead filter branches and their divisions vanish); wave32 is
  requested. `SUB_BLOCKS=8` (128-thread workgroups) and the 288-float
  bank-conflict-free transpose window (10 240 B LDS) are unchanged.
- Each 16-lane tile exchanges `trans` across `subgroupBarrier()`, whose scope is
  **one subgroup**, so the size in use has to be a multiple of 16 — wave32 is
  requested when the device offers it, and when it does not, the driver's default
  is accepted only if it is a multiple of 16. Otherwise creation fails with that
  reason instead of running tiles that span two subgroups.
- FFTW codelets, window/sigma table math, filter formulas and the reflect-pad
  rule are unchanged from the ported reference; only the buffers the data comes
  from and goes to changed.
- 12 SPIR-V blobs (`pad`/`col2im` × 2 depths, `fused_r0..3` × 2 depths), down
  from 22: the `pad_slot`/`pad_direct` and `fused`/`fused_direct` variants
  collapsed into one each.

### What the port deleted

`SlotState`/`slot_lock`/`frame_gen`, the per-slot timeline semaphores, the
ReBAR host-direct upload buffer, staging + download buffers, the `FramePool` of
per-stream resources, per-resource fences, the pad command-buffer ring, the
`pad_slot`/`pad_direct` split, the `FUSED_DIRECT` pair, and the dump-pad /
force-pad / fail-pad probes: 2854 lines of C++ became 1565.

## Performance

- **The pre-port gap was the core's `GPUDownload`, not the filter.** A frame was
  ~835 µs = ~660 µs of GPU work plus ~175 µs of exposed transfer, and the
  transfer was a copy running on the graphics engine. With the SDMA family
  opted in it is free and the default run lands within a few percent of pre-R80.
- **CPU in / CPU out** (benchmark default): core-inserted `GPUUpload` +
  `GPUDownload`. The upload is free — a host memcpy into ReBAR-mapped planes
  that runs ahead — and on SDMA so is the download. The R80 column is therefore
  the same in both input rows (1361 vs 1350 fps), while the pre-R80 column
  drops from 1376 to 1182 because that build took the GPU clip through
  `GPUDownload` before its CPU filter could run.
- **GPU in / CPU out** (`--gpu-cache`): +14% for the port, which reads the
  core's planes in place instead of downloading the clip and re-uploading its
  own padded slices.
- **GPU in / GPU out** (chained instances): parity, +4%. The residual is the two
  extra pad dispatches per frame — the pre-R80 build padded each source frame
  once into a slot and reused it across the temporal window.
- The 1080p GRAY16 round trip costs ~276 µs with the copy on the graphics
  engine (`GPUUpload→GPUDownload` 404 µs against a 126 µs BlankClip baseline,
  ~14 GB/s each way) and is free with SDMA.

## Historical

The pre-R80 design and every round that shaped it, kept for the mechanisms:

- **Slot-direct frame cache** — each source frame reflect-padded once into a
  shared device-local slot, fused reading the slots in place via `slot_base[7]`,
  reclaim gated on a per-resource `frame_gen` and published on per-slot
  *timeline* semaphores. Worth +22% then; the port replaced it with the core's
  own GPU frame cache plus re-padding, which costs ~26 µs/instance.
- **Semaphore protocol** — the cache first used one binary semaphore per pad
  generation, which deadlocked at 3+ chained instances (an upstream frame
  processed twice left the second consumer waiting on a consumed signal); a
  timeline semaphore with one signal value per generation, non-destructive
  waits, fixed it. All of it is gone with the cache.
- **ReBAR host-direct upload** — plain `memcpy` into host-visible device-local
  memory, pad reading VRAM instead of GTT (+15%). The measured contradiction
  (`memcpy` 64 GB/s vs NT stores 31 GB/s here, the reverse in an earlier
  session) was never resolved; the port removed the question.
- **Fused `IM2COL` branch** — the `sb >= 0` slot/padded branch if-converted into
  doubled u16 loads and +143 `s_waitcnt`; the fix was a branchless
  `FUSED_DIRECT` variant plus a mixed one, host-picked per frame. The port has
  one addressing mode, so there is nothing to pick.
- **Second-queue transfer split** (pre-R80, neutral) and **a second
  compute-family queue for the R80 core's transfers** — both the minimal form
  (downloads on compute-family index 1, uploads left on compute) and the full
  one (uploads index 1, downloads index 2), built from `reference/vapoursynth`:
  measured **neutral**, 1192/1194 fps with and without on DFTTest and 1699 vs
  1707 on Bilateral. Both were neutral for the same reason: a second queue of
  the *compute* family is the same engine as the compute queue, so the copy
  still competed with the dispatches. Only the dedicated transfer family, which
  is what `RADV_EXPERIMENTAL=transfer_queue` exposes, moves it off. The patch
  trees were deleted and the stock core is what runs here.
- **ODR COMDAT hazard** — filters that instantiate the shared `FramePool<T>`
  with a filter-local struct of the same name but different size silently
  corrupt the pool. No `FramePool` here any more; the rule lives in `AGENTS.md`.
- **The subgroup size was assumed, not checked** — the fused kernel used to take
  whatever default the device reported. Measured on lavapipe (the one device here
  whose subgroups are 8 lanes wide, `VSFEEL_DFFTEST_SGSIZE=8` to reproduce the old
  selection) it still matched gfx1100 to 3.7e-9 (one ulp), because a software
  backend implements the subgroup-scoped barrier as a workgroup one; on hardware
  with real 8-lane subgroups the same selection is a race, so the requirement is
  now enforced at creation. `src/vsfeel.h` also gained the two checks the request
  was missing: size control actually enabled, and subgroups per workgroup within
  `maxComputeWorkgroupSubgroups`.
- Dead ends that stay dead: `SUB_BLOCKS=16` (worse), the LDS-slice fused
  restructure (2x slower, LDS-bound), a host-side cache with mutex/cv
  ordered-submission waits (starves the worker pool).

## Open work

- Do not re-derive: every in-filter alternative to the download was measured
  and rejected — a second *compute-family* queue for the copy (neutral: same
  engine), the core's host-visible direct-read path with streaming loads (fast
  at 1080p, collapses past ~8 MB), host-cached plane memory (much worse), and
  gating the streaming path by size (helps DFTTest, does not generalize).
  The fix was the driver opt-in, not filter code.
- **Fused codegen residue**: remaining IM2COL/window ALU and pointer-walk
  strength reduction. col2im is already ~2x the references; leave it.

## Debug env vars

- `VSFEEL_DFFTEST_GPUTRACE=<frame>` — one-shot warm GPU timings for that frame
  (`pad`/`fused`/`col2im`/total); defaults to frame 100 and requires the queue
  family to report timestamp bits.
- `VSFEEL_DFFTEST_TIMING=1` — per-frame host stage averages
  (`acquire`/`record`/`submit`).
- `VSFEEL_DFFTEST_TRACE=1` — one line per submitted frame.
- `VSFEEL_DFFTEST_VRAM=1` — per-in-flight-frame scratch banner.
- `VSFEEL_DFFTEST_SGSIZE=N`, `VSFEEL_DFFTEST_SGSIZE_INVALID` — force (or
  deliberately break) the requested subgroup size; both bypass the multiple-of-16
  requirement above on purpose, and an unsupported size is still rejected when the
  pipeline is created.
- `RADV_DEBUG=asm`, `RADV_DEBUG=shaderstats` — ACO ISA and VGPR/LDS/occupancy.
