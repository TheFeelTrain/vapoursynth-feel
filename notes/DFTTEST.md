# DFTTest — notes

Status: **shipped on the R80 GPU API.** `clip:vnode:gpu` in and `ffGPUOutput`
out, so the core owns every transfer (`std.GPUUpload`/`GPUDownload`) and the
filter records one submission per output frame into one exec pool. Every piece
of host-side IO is gone: no staging buffers, no ReBAR upload, no slot frame
cache, no per-slot timelines, no fences, no filter-owned descriptor sets.
Verified against `src/dfttest.{cpp,comp}`. Target GPU: RX 7900 XTX (RDNA3,
gfx1100), Mesa 26.2 RADV.

Scoreboard — 3000 frames, `tools/benchmark.py --filter dfttest vsfeel`,
3 interleaved pre-port/R80 reps of 2 runs each, medians:

| clip / mode | pre-R80 | R80 port | delta |
|---|---|---|---|
| jpbd 1080p GRAY16 (default, CPU cache) | 1477.8 | 1198.2 | **−18.9%** |
| BlankClip 1080p GRAY16 (`--synthetic`) | 1543.1 | 1209.0 | **−21.6%** |
| jpbd + `--gpu-cache` (GPU clip consumed) | 1151.2 | 1205.7 | **+4.7%** |

- vszipcl on the same runs: 898 fps, i.e. the port is 1.33x the reference.
- **The CPU-sink delta is entirely the core's `GPUDownload`, not the filter.**
  Chained instances (middle frames stay in VRAM, so only the ends transfer)
  measure **640 µs pre-port vs 666 µs per instance (+4%)**. A port frame is
  ~835 µs = ~660 µs of GPU work plus ~175 µs of exposed transfer.
- A 1080p GRAY16 GPU round trip costs ~276 µs here (`GPUUpload→GPUDownload`
  404 µs against a 126 µs BlankClip baseline; ~14 GB/s each way). The pre-R80
  filter paid none of it: col2im wrote into host-visible staging and the CPU
  copied it out. `notes/BILATERAL.md` documents the same mechanism at −16.4%.

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
- `num_streams` and `device_id` are accepted and ignored: depth is the core's
  call, device choice is `core.set_vulkan_device`.

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

The three lines of the scoreboard differ only in what crosses PCIe:

- **CPU in / CPU out** (benchmark default): core-inserted `GPUUpload` +
  `GPUDownload`. The upload is free (it is a host memcpy into ReBAR-mapped
  planes and it runs ahead); the download is the ~175 µs that is not hidden.
  `--gpu-cache` (same frames pre-uploaded) measures the same 1205 fps as the
  default, which is what makes the upload's cost visible as zero.
- **GPU in / CPU out**: the port is 4.7% *faster* than pre-R80, because the
  pre-R80 filter still had to download the GPU clip and then upload its own
  padded slices, while the port reads the core's planes in place.
- **GPU in / GPU out** (chained instances): parity, +4%. The residual is the
  two extra pad dispatches per frame — the pre-R80 build padded each source
  frame once into a slot and reused it across the temporal window, the port
  re-pads all `tw` slices every frame. Measured at ~26 µs/instance.

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
  1707 on Bilateral. The download is bandwidth-bound and its host wait sits on
  the consumer's critical path, not queue-order-bound; the patch trees were
  deleted and the stock core is what runs here.
- **ODR COMDAT hazard** — filters that instantiate the shared `FramePool<T>`
  with a filter-local struct of the same name but different size silently
  corrupt the pool. No `FramePool` here any more; the rule lives in `AGENTS.md`.
- Dead ends that stay dead: `SUB_BLOCKS=16` (worse), the LDS-slice fused
  restructure (2x slower, LDS-bound), a host-side cache with mutex/cv
  ordered-submission waits (starves the worker pool).

## Open work

- **The CPU-sink gap cannot be closed in the filter.** Reducing the GPU work
  only helps below the transfer floor, and the kernel work is already at
  parity. The levers left are outside `src/dfttest.*`:
  - the core's `GPUDownload` path (staged copy + host wait; the direct path
    needs `HOST_CACHED` plane memory, which a discrete card does not have);
  - a `vnode:all` filter with its own host-visible output path — i.e. the
    pre-R80 IO the port deliberately removed — worth a few percent at most
    (kernel-direct download measured 568 vs 553 fps against SDMA D2H).
- **Fused codegen residue**: remaining IM2COL/window ALU and pointer-walk
  strength reduction. col2im is already ~2x the references; leave it.
- Do not re-derive: the 1080p GRAY16 frame cost decomposes as ~660 µs of GPU
  work + ~175 µs of exposed transfer, and neither the pad (hidden behind the
  transfer floor) nor the intra-frame barriers (removing both changed the
  chained marginal by <1%) is a lever.

## Debug env vars

- `VSFEEL_DFFTEST_GPUTRACE=<frame>` — one-shot warm GPU timings for that frame
  (`pad`/`fused`/`col2im`/total); defaults to frame 100 and requires the queue
  family to report timestamp bits.
- `VSFEEL_DFFTEST_TIMING=1` — per-frame host stage averages
  (`acquire`/`record`/`submit`).
- `VSFEEL_DFFTEST_TRACE=1` — one line per submitted frame.
- `VSFEEL_DFFTEST_VRAM=1` — per-in-flight-frame scratch banner.
- `VSFEEL_DFFTEST_SGSIZE=N`, `VSFEEL_DFFTEST_SGSIZE_INVALID` — force (or
  deliberately break) the requested subgroup size.
- `RADV_DEBUG=asm`, `RADV_DEBUG=shaderstats` — ACO ISA and VGPR/LDS/occupancy.
