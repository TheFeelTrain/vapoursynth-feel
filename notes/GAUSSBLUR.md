# GaussBlur — notes

Status: **shipped on the R80 GPU API.** Verified against `src/gaussblur.{cpp,comp}`
and `tools/benchmark.py`. Target GPU: RX 7900 XTX (RDNA3, gfx1100), RADV, ReBAR.

Design (current):
- `clip:vnode:gpu` in, `ffGPUOutput` out: the core owns every transfer
  (`std.GPUUpload`/`GPUDownload`); the filter owns its pipelines, one constant
  weights buffer and one exec pool — no staging, per-stream resources, fences,
  queues or queue cap.
- Two code paths, three kernels: fused small (radius ≤ 32) and two-pass
  vertical/horizontal (larger). Each pass carries its own workgroup shape and
  outputs per thread, because the two want opposite ones: the vertical pass is
  32x4 with 16 outputs per thread (a warp then reads one row of 32 consecutive
  elements, one full cache line, and its strided row loads amortize over 16
  outputs), while the fused path and the horizontal pass stay 16x8 with 8
  outputs per thread (a warp spanning two rows, where a wider per-lane stride
  splits each load across more lines).
- Per-plane float scratch is one transient `createGPUBuffer` per frame (record
  stage: 10 µs/frame with it).
- `WIDTH/HEIGHT/STRIDE/KLEN/RAD` are specialization constants read off a probe
  frame at creation; five push-descriptor bindings (`wt`, `src`, `dst`, `tmp`,
  `dst` dword view), only `tmp_elem`/`wt_base` push constants carry values.
- Unprocessed planes ride along via `newVideoFrame2`; `num_streams` and
  `device_id` are registered no-ops (never read).
- Runs need `RADV_EXPERIMENTAL=transfer_queue`, which `tools/benchmark.py`
  forces (mechanism: `notes/BILATERAL.md`).

Scoreboard — jpbd 1080p GRAY16, sigma=16 (95 taps, two-pass), 5000 frames,
3 repeats, same-session medians through `tools/benchmark.py`:

| mode | vsfeel | vszipcl | ratio |
|---|---|---|---|
| CPU cache (benchmark default) | 2949.9 | 1608.0 | 1.83x |
| `--gpu-cache` (resident GPU input) | 3986.7 | 1614.7 | 2.47x |
| `--gpu-cache --no-download` (compute only) | 9877.9 | 1589.8 | 6.21x |

- The first two rows carry transfers: row 1 uploads the CPU cache per frame and
  downloads the output, row 2 removes the input upload and still downloads; the
  filter runs at 101 µs/frame in all three. Repeat sessions land within ~2–4%.
- vszipcu is not installed in this checkout, so vszipcl is the reference here.
- Bench: `MANGOHUD=0 uv run tools/benchmark.py --filter gaussblur
  [--gaussblur-args "sigma=X"] [--gpu-cache] [--no-download] [--scale N] vsfeel
  vszipcl`.

## Implementation

`src/gaussblur.cpp` + `src/gaussblur.comp`.

- Frame path (`gauss_gpu_frame`): take the source, make the output frame
  (`newGPUVideoFrame` when every plane is processed, `newVideoFrame2` sharing
  unprocessed ones), acquire one exec context, allocate the two-pass scratch
  when needed, record one dispatch per processed plane (large path: vertical →
  `gpu_barrier` → horizontal), declare `gpuExecReadsFrame`/`WritesPlane`,
  submit. `getGPUPlane` binds the core's own plane buffers — the filter moves
  no bytes itself.
- Creation: per-plane config keys deduped for identical planes (pipelines,
  grids, `wt_base` shared) — but the **scratch region is deliberately not
  shared**: identical planes' dispatches are unordered inside one command
  buffer, so a shared region would race (WAR between the reused plane's
  horizontal pass and this plane's vertical pass). Weights: all kernels
  concatenated in one host-visible/coherent, device-local-preferred buffer,
  one memcpy + `_mm_sfence()` (may land in the write-combined VRAM BAR).
- Each kernel's `GpuWorkgroup` is set at its `gpu_create_pipeline` call, so the
  passes can differ; `uv run python tools/shader_limits.py` prints the table
  (16x8 fused, 32x4 vertical, 16x8 horizontal, 128 invocations each).
- Scratch is ~7.9 MiB transient per in-flight frame at 1080p GRAY16 sigma=16;
  `gpu_frame_buffer` retires it with the submission.
- The fused small path covers `radius <= 32` and declares
  `vblur[VRT*BLK_Y][BLK_X + 2*RAD]` — 7 680 B at the widest radius, under the
  16 KiB of workgroup memory Vulkan guarantees, so the device's own limit is the
  only check that can reject it (`gaussblur.cpp` static_asserts that bound).
  Wider kernels take the two-pass vertical + horizontal path, which declares no
  shared memory.
- int32 guards on the plane's last element and the scratch end (same 32-bit
  addressing bound as Bilateral).
- 6 SPIR-V variants (gauss/vert/horiz × 2 depths), selected by bit depth; the
  entry point selects the path.
- What the port deleted: `GaussBlurResource`/`FramePool`, per-stream staging +
  VRAM src/dst buffers, the pre-recorded command buffer and its H2D/D2H
  copies, descriptor pool/sets, per-stream command pools/fences/queues,
  `retire_instance`, the host `copy_plane_out`/`copy_plane_read` loop and its
  flush/invalidate ranges, and the HD/KD/QUEUES/GPU_BENCH knobs: 1443 → 834
  lines.

### Reference structure (same GPU, READ-ONLY)

- **vszipcu:** device-local `d_src`/`d_dst` + float `d_tmp`; per plane async
  `memcpyHtoD` (copy stream) → kernels → async `memcpyDtoH`, then sync both.
- **vszipcl:** same shape with OpenCL pinned staging + `clFinish` per frame.
- `rocprofv3` profile of vszipcu (BlankClip gray16 1080p, sigma=16, ns=1, 200
  frames): vertical 118 µs, horizontal 177 µs, H2D 153 µs, D2H 152 µs.

## Performance

- **Both passes are load-bound, not FMA-bound.** 95 taps over two passes is
  ~394 M FMAs per 1080p frame (~14 µs at peak), while the passes spend their
  time in loads: the vertical pass' taps are rows a stride apart, so a warp
  load touches two cache lines half-used, and the horizontal pass re-reads an
  8.3 MB float intermediate. That is what the shape and width changes below
  attack, and why a 2x cut in FMA would buy nothing.
- **Per-pass block shape and outputs per thread**, worth +9.4% to +9.7% across
  three same-session A/Bs (shipped build 8949.8 → 9808.0 fps; 1080p sigma=16,
  5000 frames, `--gpu-cache --no-download`, 6 interleaved 2-repeat runs per arm,
  spreads ≤ 2.7%). What lost, and why:
  - vertical pass at 16x8: a warp spans two rows, splitting every load across
    two half-used cache lines (~115 → 90 µs in the probe's stage split).
  - horizontal pass at 32x4 (83 → 106 µs) and at 16 outputs per thread
    (99 → 197 µs): it reads along the row it writes, so a wider per-lane stride
    spreads the warp's loads over more lines.
  - 32 outputs per thread for the vertical pass (~230 vs ~202 µs total) and 2
    or 4 for the horizontal (no better than 8): register pressure and the extra
    loads per output cancel the amortization.
- **The probe's absolute stage times are inflated.** `VSFEEL_GAUSS_GPUTRACE`
  stamps inside the recorded command buffer and drains the traced frame, but a
  deep pipeline still interleaves the other in-flight frames into a traced
  frame's span, so its numbers sit above the throughput the same binary reaches
  (~166 µs spanned vs 101 µs/frame). Use it for attribution within one run and
  the benchmark for a verdict.

## Historical

- **The 16-bit variants were invalid Vulkan** (build/correctness only, no perf
  change). SPIR-V 1.0 emits `buffer` blocks as Uniform + BufferBlock, so a
  16-bit element needs `uniformAndStorageBuffer16BitAccess` — a feature the
  core's device baseline does not enable — and the validation layer flagged it.
  The 16-bit variants (all three entries) now build with
  `--target-env=vulkan1.1`, which emits the StorageBuffer class covered by the
  baseline's `storageBuffer16BitAccess`; bit-exact vs vszipcl, 32-bit SPIR-V
  unchanged.
- **The fused path's 48 KiB tile cap was dead.** The widest small-path tile is
  7 680 B, so `min(48 KiB, device LDS)` was always decided by the device; the cap
  is gone and the device limit is now the only check. No perf change.
- **An over-wide plane is refused, not truncated.** Both passes tile by
  (ID.x, ID.y), so the old clamped grids left everything past
  `maxComputeWorkGroupCount[0]` (1 048 560 px at BLK_X=16 on a 65 535-group
  device) unwritten. Creation now errors naming the failing pass and the limits.
- **The fused path at radius 47 is 7x the two-pass time** (probe, serialized:
  1 430 vs ~200 µs at 1080p sigma=16), so `LARGE_THRESHOLD` stays at 32. Its
  shared tile carries a ±RAD column halo, so the vertical phase covers
  `(BLK_X + 2*RAD)/BLK_X` = 6.9x the columns at that radius.

### 2026-10-12 — per-pass block shapes and outputs per thread

The vertical pass moved to 32x4 with 16 outputs per thread and the horizontal
stayed 16x8 with 8, each pass now carrying its own block and width. Same-session
A/B against the previous shape (banner config, `--gpu-cache --no-download`):
8883.8 → 9716.7, 8951.1 → 9815.7 and 8949.8 → 9808.0 fps across three runs of
the pair, 6 interleaved 2-repeat runs each, spreads ≤ 2.7%. Correctness
unchanged: 72 tests, bit-exact against vszipcl.

### 2026-09-22 — R80 GPU API port

Everything the port deleted is listed under Implementation; correctness held
bit-exact against vszipcl on the whole existing sweep (58 tests) with no kernel
change. Its perf figures were taken against the pre-fix benchmark harness and
are void; the banner table replaces them.

Pre-R80 rounds, superseded by the port but kept for their mechanisms:

- **First structure (deleted before the port):** GTT staging only, kernels
  reading/writing sysmem over PCIe, 16-bit writes via dword `atomicOr` RMW with
  a per-frame `vkCmdFillBuffer` pre-clear, fully synchronous per frame, all 4
  compute queues at ns=4.
- **VRAM rework:** device-local `src`/`dst`/`tmp` + one H2D and one D2H
  `vkCmdCopyBuffer`. The port then deleted these copies too.
- **CPU reads from the VRAM BAR are ~1.3 GB/s NT / 0.1 GB/s memcpy** (64 MiB
  probe, memoryType 3) → mapped-VRAM staging is dead for the read path. Durable
  memory fact, independent of this filter.
- **Plain memcpy into the mapped VRAM window beats NT stores on upload**:
  22.5 vs 11 GB/s in the probe — the *opposite* of the EEDI3/NNEDI3 upload
  verdict. The path is buffer- and shape-dependent; re-measure rather than
  inherit.
- **`rocprofv3` does not see RADV/Vulkan work** — kernel times must come from
  in-command-buffer timestamps or fps.
- **Default `num_streams` 1 → 4**: the shipped default was the "cannot
  overlap" case; the knee was at 4. Superseded: in-flight depth is the core's
  exec ring now, the argument is validated and ignored.
- Validation hardening (correctness-only, no perf change): upload
  flush/download invalidate ranges through the shared `mapped_range` helper,
  unconditional `_mm_sfence()` before submit for the NT-staged upload, creation
  errors torn down by `~GaussData` via `FramePool::emplace()`, frame-path
  `set_error` frees `dst`. All but the sfence were deleted with the port.
- **Non-finite samples no longer spread past the kernel window.** The two-pass
  rolling window pads out-of-kernel taps with `wreg[j] == 0.0f`; that is an
  exact no-op only for finite operands, because `fma(NaN, 0, s)` is NaN. The
  poison is thread-aligned (up to `R-1` extra rows/cols) but ACO sometimes folds
  `fma(v, 0, s)` to `s`, so it showed up as a rectangle instead of a square and
  only in some pipelines. Each accumulation now skips zero-weight taps
  (`accum()`), matching vszipcl's per-index skip bit-for-bit; the unreachable
  `k + 1 < taps` else arms went with it (KLEN is always odd, so `taps` is even).
  Grid counts (`width + BLK_X - 1` and friends) are now computed in `int64_t`,
  so a near-`INT32_MAX` plane cannot wrap one before the quotient widens. No
  perf change: VGPR 48 / 32 subgroups/SIMD / 0 spills both ways.

## Open work

- **Sweep the shipped shapes across sigma ∈ {4, 40, 80}** — the block and width
  choice was swept at radius 47 only. The fused path (radius ≤ 32) carries
  `VRT`/`BLK` that have never been swept at all, and the two-pass shapes might
  have a different winner at radius 120/240.
- The two-pass path still re-reads its 8.3 MB float intermediate ~13x. Staging
  the horizontal pass' taps in shared memory is the untried structural cut
  (radius-sized tiles, so it needs a bounded variant plus a fallback); the
  first step is to price its LDS traffic against the current L1 hits.
- If a kernel-vs-copy split is ever needed again, `VSFEEL_GAUSS_GPUTRACE` is in
  the dfttest/nlmeans form; its caveat is under Performance.

### Do not retry

- `VSFEEL_GAUSS_GPU_BENCH` (it idled the device from a frame callback without
  the other streams' queue locks) was deleted — do not resurrect it.
- Do not re-introduce a host transfer path to win the CPU-sink row — that
  is exactly what the port deletes, and it is what loses the resident chain
  (pre-R80: 1408 resident fps vs 2239).
- Do not raise `LARGE_THRESHOLD` to cover radius 33–64: the fused path is 7x the
  two-pass time there (mechanism in Historical).

## Debug env vars

- `VSFEEL_GAUSS_TIMING=1` — per-frame host stage `[gauss-timing]` averages
  (acquire/record/submit/total) at instance destruction. The stages are summed
  over concurrently recorded frames, so with the GPU saturated the total can
  exceed the frame time and `acquire` dominates; read it as the host/GPU split,
  not as an absolute per-frame cost. Record+submit is ~23 µs per call.
- `VSFEEL_GAUSS_GPUTRACE=<first frame>` and `VSFEEL_GAUSS_GPUTRACE_RANGE=<n>`
  (default 100/40) — in-command-buffer GPU timestamps: per-plane per-dispatch
  p25/median/p75 over the traced window, printed at destruction. The window's
  first 8 frames are skipped (the first stamped frame reads back an impossible
  value) and the median is what to quote. Gated on the queue family's timestamp
  bits.
- `VSFEEL_DEBUG=1|2`, `VSFEEL_TRACE=1|2` — the shared error trace and trail
  (`acquire`/`record`/`submit` marks are recorded in the frame path).
- Removed with the legacy path: `VSFEEL_GAUSS_HD`, `VSFEEL_GAUSS_KD`,
  `VSFEEL_GAUSS_QUEUES`, `VSFEEL_GAUSS_GPU_BENCH`.
