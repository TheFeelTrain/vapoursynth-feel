# GaussBlur — notes

Status: **shipped on the R80 GPU API.** Verified against `src/gaussblur.{cpp,comp}`
and `tools/benchmark.py`. Target GPU: RX 7900 XTX (RDNA3, gfx1100), RADV, ReBAR.

Design (current):
- `clip:vnode:gpu` in, `ffGPUOutput` out: the core owns every transfer
  (`std.GPUUpload`/`GPUDownload`); the filter owns its pipelines, one constant
  weights buffer and one exec pool — no staging, per-stream resources, fences,
  queues or queue cap.
- Two code paths, kernels unchanged from pre-R80: fused small (radius ≤ 32) and
  two-pass vertical/horizontal (larger); per-plane float scratch is one
  transient `createGPUBuffer` per frame (record stage: 10 µs/frame with it).
- `WIDTH/HEIGHT/STRIDE/KLEN/RAD` are specialization constants read off a probe
  frame at creation; five push-descriptor bindings (`wt`, `src`, `dst`, `tmp`,
  `dst` dword view), only `tmp_elem`/`wt_base` push constants carry values.
- Unprocessed planes ride along via `newVideoFrame2`; `num_streams` and
  `device_id` are registered no-ops (never read).
- Runs need `RADV_EXPERIMENTAL=transfer_queue`, which `tools/benchmark.py`
  forces (mechanism: `notes/BILATERAL.md`).
- The filter sits at the API's transfer ceiling: a bare core
  `GPUUpload→GPUDownload` graph (same cached real-clip vpy, 3000 frames) runs
  2373 fps, GaussBlur 2352 fps — the CPU-sink delta below is the deleted
  HD/KD transfer path, not the filter.

Scoreboard — jpbd 1080p GRAY16, sigma=16 (radius 48 → two-pass), 3000 frames,
interleaved pre-R80/R80 pairs through `tools/benchmark.py --repeat 1`, 3 rounds,
medians:

| input to vsfeel | pre-R80 | R80 port | delta |
|---|---|---|---|
| CPU cache (benchmark default) | 2499.6 | 2346.4 | **−6.1%** |
| `--gpu-cache` (resident GPU clip) | 1408.3 | 2239.6 | **+58.9%** |

- Mechanism of the CPU-sink delta (same as `notes/BM3D.md`'s port): the old
  kernel's stores *were* the download (KD) and the upload wrote VRAM directly
  (HD), so the old build moved one full-frame DMA less than the core's round
  trip — it even beat the bare 2373 fps ceiling for that reason. Under the API
  the output plane is core-owned VRAM and `GPUDownload` is a separate pass.
- References, same-session triples (3000 frames, `--repeat 2`): CPU cache —
  vsfeel **2311.1**, vszipcl 1607.4, vszipcu 3391.0; `--gpu-cache` — vsfeel
  **2276.7 (#1)**, vszipcl 1090.3, vszipcu 1766.2. The port wins exactly where
  a resident chain is involved (+29% over the fastest reference); the CPU-sink
  row keeps the pre-port shape (beats vszipcl 1.44x, trails vszipcu).
- Host split (`VSFEEL_GAUSS_TIMING=1`, same 3000-frame run): acquire 21.3 /
  record 10.0 / submit 84.5 / total 115.8 µs per frame — nowhere near the
  425 µs frame.
- Bench: `MANGOHUD=0 python3 tools/benchmark.py --filter gaussblur
  [--gauss-sigma X] [--gpu-cache] [--frames N] vsfeel vszipcl vszipcu`.

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
- Scratch is ~7.9 MiB transient per in-flight frame at 1080p GRAY16 sigma=16;
  `gpu_frame_buffer` retires it with the submission.
- int32 guards on the plane's last element and the scratch end (same 32-bit
  addressing bound as Bilateral).
- 6 SPIR-V variants (gauss/vert/horiz × 2 depths) are unchanged; only the
  header comment moved (it no longer claims the host moves frames by DMA).
- What the port deleted: `GaussBlurResource`/`FramePool`, per-stream staging +
  VRAM src/dst buffers, the pre-recorded command buffer and its H2D/D2H
  copies, descriptor pool/sets, per-stream command pools/fences/queues,
  `retire_instance`, the host `copy_plane_out`/`copy_plane_read` loop and its
  flush/invalidate ranges, and the HD/KD/QUEUES/GPU_BENCH knobs: 1443 → 834
  lines.

### Reference structure (same GPU, READ-ONLY)

- **vszipcu (fastest on a CPU sink):** device-local `d_src`/`d_dst` + float
  `d_tmp`; per plane async `memcpyHtoD` (copy stream) → kernels → async
  `memcpyDtoH`, then sync both.
- **vszipcl:** same shape with OpenCL pinned staging + `clFinish` per frame.
- `rocprofv3` profile of vszipcu (BlankClip gray16 1080p, sigma=16, ns=1, 200
  frames): vertical 118 µs, horizontal 177 µs, H2D 153 µs, D2H 152 µs.

## Historical

### 2026-09-22 — R80 GPU API port

Banner table above; interleaved pairs, medians of 3 rounds. Mechanism of the
−6.1% CPU-sink arm and the +58.9% resident arm is in the banner. Everything
the port deleted is listed under Implementation; correctness held bit-exact
against vszipcl on the whole existing sweep (58 tests) with no kernel change.

Pre-R80 rounds, superseded by the port but kept for their mechanisms:

- **First structure (deleted before the port):** GTT staging only, kernels
  reading/writing sysmem over PCIe, 16-bit writes via dword `atomicOr` RMW with
  a per-frame `vkCmdFillBuffer` pre-clear, fully synchronous per frame, all 4
  compute queues at ns=4.
- **VRAM rework:** device-local `src`/`dst`/`tmp` + one H2D and one D2H
  `vkCmdCopyBuffer` took vsfeel 1191–1342 → **1808** fps (vszipcl 1412,
  vszipcu 3489; sigma=16 config of the time). The port then deleted these
  copies too.
- **CPU reads from the VRAM BAR are ~1.3 GB/s NT / 0.1 GB/s memcpy** (64 MiB
  probe, memoryType 3) → mapped-VRAM staging is dead for the read path. Durable
  memory fact, independent of this filter.
- **Plain memcpy into the mapped VRAM window beats NT stores on upload**: 22.5
  vs 11 GB/s in the probe — the *opposite* of the EEDI3/NNEDI3 upload verdict.
  The path is buffer- and shape-dependent; re-measure rather than inherit.
- **`rocprofv3` does not see RADV/Vulkan work** — kernel times must come from
  in-command-buffer timestamps or fps.
- **No HD/KD ablation was ever recorded for GaussBlur** — the +34%/+3% once
  quoted here were Bilateral's. Do not quote a GaussBlur split; the knobs are
  gone with the port.
- **Default `num_streams` 1 → 4**: the shipped default was the "cannot
  overlap" case; same-session sweep (500 cached frames ×3 medians, sigma=16):
  ns=1 1236, ns=4 2495, ns=8 2255 — knee at 4, which the benchmark already
  used. Superseded: in-flight depth is the core's exec ring now, the argument
  is validated and ignored.
- Validation hardening (correctness-only, no perf change): upload
  flush/download invalidate ranges through the shared `mapped_range` helper,
  unconditional `_mm_sfence()` before submit for the NT-staged upload, creation
  errors torn down by `~GaussData` via `FramePool::emplace()`, frame-path
  `set_error` frees `dst`. All but the sfence were deleted with the port.

## Open work

- **Workgroup-shape sweep across sigma ∈ {4, 16, 40, 80}** — never done, and
  the port did not touch the kernels or their `16x8` launch, so it still
  applies (sweep block candidates × configs per `AGENTS.md`).
- If a kernel-vs-copy split is ever needed again, build the warm
  in-command-buffer timestamp probe in the dfttest/nlmeans form;
  `VSFEEL_GAUSS_GPU_BENCH` (it idled the device from a frame callback without
  the other streams' queue locks) was deleted — do not resurrect it.
- The remaining CPU-sink gap to vszipcu is transfer arrangement plus its HIP
  kernels: kernel-direct download cannot exist under the API (the output plane
  is core-owned VRAM), so judge the filter on `--gpu-cache` too, where it is
  #1. Do not re-introduce a host transfer path to win the CPU-sink row — that
  is exactly what the port deletes, and it is what loses the resident chain
  (pre-R80: 1408 resident fps vs 2239).

## Debug env vars

- `VSFEEL_GAUSS_TIMING=1` — per-frame host stage `[gauss-timing]` averages
  (acquire/record/submit/total) at instance destruction.
- `VSFEEL_DEBUG=1|2`, `VSFEEL_TRACE=1|2` — the shared error trace and trail
  (`acquire`/`record`/`submit` marks are recorded in the frame path).
- Removed with the legacy path: `VSFEEL_GAUSS_HD`, `VSFEEL_GAUSS_KD`,
  `VSFEEL_GAUSS_QUEUES`, `VSFEEL_GAUSS_GPU_BENCH`.
