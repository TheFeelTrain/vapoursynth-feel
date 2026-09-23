# Bilateral — notes

Status: **shipped on the R80 GPU API.** `clip`/`ref` are `vnode:gpu` and the
output carries `ffGPUOutput`, so the core's `GPUUpload`/`GPUDownload` cross the
bus and the filter owns only its pipelines and one exec pool. Every plane is the
core's own GPU frame plane: the kernels read the source and guide in place and
write the output frame in place. The old per-stream staging buffers, VRAM
src/dst buffers, queue cap and fence are gone.

Design (current):
- Two kernels, chosen per plane at creation: `bilateral_shared.comp` (LDS-tiled
  gather, ~3.2–3.8x faster where the tile fits) and `bilateral_plain.comp`.
- Three whole-buffer bindings per dispatch (src, dst, guide); without a guide the
  source stands in on binding 2 and `HAS_REF=0` means the shader never reads it.
- Native `uint16_t`/`float` SSBOs at `#version 450`, no push constants; every
  geometry/weight parameter is a specialization constant
  (`WIDTH/HEIGHT/STRIDE/RADIUS/SIGMA_*/HAS_REF/TILE_*/BLOCK_*`).
- `STRIDE` is the core's plane pitch in elements, read off a scratch frame at
  creation (a GPU frame uses the identical stride math).
- Auto block shape (`32x8` for max radius ≤ 12, `32x16` above); explicit
  `block_x`/`block_y` are respected and clamped to the device limits.
- Unprocessed planes (`sigma < FLT_EPSILON`) ride along from the source through
  `newVideoFrame2`, keeping their own producer pairs; nothing is submitted when
  every plane is unprocessed.
- `num_streams` and `device_id` are accepted and **ignored**: depth is the core's
  call and device choice is `core.set_vulkan_device`.
- **`RADV_EXPERIMENTAL=transfer_queue` is required for the numbers below**, and
  `tools/benchmark.py` now forces it. Without it RADV exposes no transfer-only
  queue family, so the core's `GPUDownload` is a copy on the graphics engine
  that competes with the kernel; with it the copy moves to SDMA and costs
  nothing. Every −16% figure in this file's history predates that and is void.

Performance — 1080p GRAY16 jpbd, 5000 frames, sigma 3.0/0.02 (R=9), interleaved
pre-R80/R80 pairs through `tools/benchmark.py`, `--repeat 2`:

| input to vsfeel | pre-R80 | R80 port | delta |
|---|---|---|---|
| CPU cache (benchmark default) | 1982.1 | 1980.5 | **−0.1%** |
| `--gpu-cache` (pre-uploaded GPU clip) | 1975.3 | 1974.0 | **−0.1%** |

- The R80 figure is the same in both rows: the timed unit is a CPU sink, so the
  upload is free, and with the transfer queue the download is free too.
- vs `vszipcl` in a separate same-session pair: 2009 vs 1329 fps (**+51%**).

## Implementation

`src/bilateral.cpp` + `src/bilateral_shared.comp` / `src/bilateral_plain.comp`.

- Per plane at creation: `BilateralPlaneConfig{width, height, stride, pipeline,
  grid_x, grid_y}`; identical plane configs share one `VkPipeline`.
- Frame path: `getGPUPlane` on src/guide/dst, bind pipeline, `gpu_push_buffers`
  (3 storage buffers via push descriptors, `VK_WHOLE_SIZE`), `vkCmdDispatch`,
  then `gpuExecReadsFrame`/`gpuExecWritesPlane`/`gpuExecSubmit`. No barrier is
  needed: the plane dispatches touch disjoint buffers.
- `use_shared_memory=0`, or a tile that exceeds
  `maxComputeSharedMemorySize`, selects the plain kernel. The gate counts
  `(1 + has_ref) * (2R + block_x) * (2R + block_y) * 4` bytes, which is exactly
  what `buf[SHARED_FLOATS]` reserves (the output goes straight to `dst[]`).
- The 32-bit element-offset guard rejects a plane whose last element
  `(h-1)*stride + w - 1` would not fit an int.
- `round(sigma_spatial*3)` is clamped to 1e6 before the float→int cast (the
  reference clamps identically); `sigma_spatial`/`sigma_color` reject NaN/inf.
- 16-bit rounding is `uint(v*PEAK + 0.5) & 0xFFFF`, matching the reference's
  round-half-away inside the 1-LSB test tolerance.

## Performance

- **The exposed download was a copy on the graphics engine; the fix is a driver
  opt-in, not a core change.** RADV gates its dedicated transfer-only SDMA
  family behind one (`radv_transfer_queue_enabled` in
  `radv_physical_device.c`); the Mesa 26.0 notes spell the switch
  `RADV_PERFTEST=transfer_queue`, but this machine's Mesa 26.3-devel honours
  only `RADV_EXPERIMENTAL=transfer_queue` and silently ignores the old name.
  With it the device reports family 3 = `TRANSFER|SPARSE_BINDING` ×2, the
  core's existing code points `transferPtr`/`downloadPtr` at it
  (`vsvulkan.cpp:882-895`), and the copy leaves the graphics engine: Bilateral
  **1590 → 2079 fps (+31%)** on interleaved pairs, DFTTest +22%, EEDI3 +16%,
  EEDI3AA +13%, BM3Dv2 +10% — each at 97–100% of its "download node deleted"
  ceiling, output byte-identical.
- **`GPUDownload` is the only asymmetric leg.** On ReBAR the upload is a plain
  memcpy into the mapped plane with no submission at all
  (`vsvulkanframe.cpp:259-276`), so it is free. The download cannot take its
  direct path — that requires `HOST_CACHED` plane memory
  (`vsvulkanframe.cpp:363-367`), and a discrete card's planes are
  `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` but not cached (reading the BAR runs
  at 0.02 GB/s) — so it stages `stride*height` bytes VRAM → a cached host
  readback slot by a GPU copy, then a host memcpy (4 slots, `vscore.cpp:1491`;
  slot memory at `vsvulkanframe.cpp:185-193`). At 1080p u16 that copy is 4.1 MB
  ≈ 0.10–0.16 ms: the measured pre-opt-in gap was 0.606 vs 0.507 ms.
- **The kernel is not the regression — it is ISA-identical.** `RADV_DEBUG=shaderstats`
  (R=9, BITS=16, pipeline cache off): new 3101 instructions / 14700 B code / 192
  VGPR / 0 spill; pre-R80 3102 / 14708 / 192 / 0. `glslc -O` output is 315 vs 321
  SPIR-V lines. The shader rewrite (three whole-buffer bindings, no push
  constants) did not move the machine code.
- **The frame is kernel-dominated.** A timestamp probe around the dispatches
  (BlankClip GRAY16 1080p, `-r 1`, serialized for a clean read): R=3 0.080 ms,
  R=9 0.507, R=12 0.678, R=24 2.442. At R=9 the 0.507 ms was exactly the pre-R80
  frame time (1972 fps) and 84% of the pre-opt-in post-port frame (0.606 ms), so
  the pre-port filter was already at the kernel floor with its host copies fully
  overlapped. This is also why the pre-opt-in delta tracked the kernel: R=3
  −9.3%, R=9 −16.4%, R=24 −8.8% (a 2.44 ms kernel hides most of the transfer).
- Transfer decomposition on a BlankClip GRAY16 1080p graph: an upload+download
  round trip with no filter is 2414 fps (0.414 ms/frame) at the benchmark's
  default request depth; adding Bilateral is 1668 fps (0.599 ms). Only ~0.10 of
  that 0.414 ms was exposed — the rest already overlapped the kernel.
- The filter is not the limiter: `vspipe --filter-time` on the same graph
  (3000 frames, `-r 8`) puts Bilateral at **3.97 % / 0.07 s** of summed thread
  time (23 µs/frame) against `GPUDownload` 735.62 % / 13.24 s and `GPUUpload`
  32.72 % / 0.59 s.
- Host split (`VSFEEL_BILAT_TIMING=1`, 3000 frames): acquire 19.1 / record 5.8 /
  submit 59.0 µs per frame — the host is nowhere near the wall, which is why the
  pre-port gap had to be the core's download.
- The port wins exactly where a resident chain is involved: fed a GPU clip the
  old filter had to `GPUDownload` it first, the new one reads VRAM in place.

Benchmark call: `MANGOHUD=0 python3 tools/benchmark.py --filter bilateral
[vsfeel vszipcl] [--gpu-cache] [--bits 32] [--bilateral-sigma-spatial X
--bilateral-sigma-color Y]`.

## Historical

### 2026-09-21 — R80 GPU API port

The whole pre-R80 transfer path was deleted: per-stream mapped staging, the
ReBAR host-direct upload (`VSFEEL_BILAT_HD`), the GTT kernel-direct download
(`VSFEEL_BILAT_KD`), the queue cap (`VSFEEL_BILAT_QUEUES`), and the A/B copy
probes (`NOCPU`/`NODL`/`NODISPATCH`). Mechanism of the cost, measured on the
BlankClip decomposition above: the old kernel's stores *were* the download, so
the copy overlapped compute for free; under the API the output plane is VRAM and
`GPUDownload` is a separate pass ordered after the kernel. That is the same
mechanism `notes/BM3D.md` records for its −4% port, with the difference that
Bilateral is light enough for the download to dominate. The pre-R80 numbers in
this file's older rounds are therefore void — do not quote them as the current
filter.

### 2026-09-21 — the gap was a driver opt-in, not a second queue

Two core-side attempts were built and measured before the real cause was found,
both neutral and both reverted: `downloadPool` on a second *compute-family*
queue (family 1 index 1) and the full form (uploads index 1, downloads index 2).
A second queue of the same family is the same engine as the compute queue, so
the copy still competed with the dispatches — the core-only `GPUUpload →
GPUDownload` + k BoxBlur(r=96) probe that read as +45% did not transfer to a
plugin's submission. RADV exposes a genuinely separate engine only through its
dedicated transfer-only SDMA family, and gates that behind an opt-in. With
`RADV_EXPERIMENTAL=transfer_queue` (the Mesa 26.0 notes call it
`RADV_PERFTEST=transfer_queue`; current Mesa ignores that name) the copy moves to
SDMA and the port reaches parity without any core change.

### Pre-R80 host path (deleted, mechanism kept)

- **Host-direct upload / kernel-direct download**: with both on, the per-frame
  GPU work was kernel only. HD alone was worth +34%, KD +3%. Both are now the
  core's job.
- **Queue cap 2**: one stream per queue left idle bubbles while a worker did
  post-fence host work; sharing a queue kept a next command buffer queued
  (ns=4: 2 queues 1986 vs 4 queues 1709 fps). Meaningless now: the core exposes
  one compute queue and sizes the exec pool itself.
- **Regime variance ±15–40% between invocations at ns=2** while vszipcl stayed
  ±1%: the queue cap fixed it. Same-session interleaved pairs remain the only
  fair comparison — the A/B table above uses 4 alternated reps for that reason.

### Kernel findings that still ship

- **Shared vs plain is worth ~3.2–3.8x, and the gate must be the device limit.**
  The old hardcoded 48 KiB cap used to drop the tiled kernel for no-guide
  R∈[28,43] / guide R∈[21,27] even though this device reports 65 536 B. Raising
  the gate to `maxComputeSharedMemorySize` measured 3.29x (no-guide R=45),
  3.12x (guide R=28) and 3.31x (guide R=32) against vszipcl at 1.50–3.10x, and
  improved reference agreement in the changed band (the plain kernel's border
  semantics are the divergent side). Every shared-kernel cell in the radius ×
  guide matrix wins (1.00–2.65x); every plain-kernel cell loses.
- **The R=24 loss was LDS occupancy, not ACO VOPD packing.** Reserving one whole
  tile instead of `(1 + has_ref)` doubled the LDS at every radius and pushed
  R∈[21,27] guide and R∈[28,43] no-guide over the cap; with the sizing fixed,
  R=24 no-guide went 345 → 407.8 fps (1.12x vs reference) because 40 960 →
  20 480 B raised the fit from 1 to 3 workgroups/CU. The "ACO VOPD=0" theory is
  retired.
- **Default block shape is a real knob at wide radii**: R=3 32x8 (2449), R=9 32x8
  (1742) > 16x8 (1707) > 16x16 (1705), R=12 32x16 (1360) ≈ 32x8, R=24 16x16
  (345) ≫ 16x8 (228). The auto rule (`32x8` R≤12, `32x16` above) is unchanged by
  the port.
- **Dead ends, with mechanism.**
  - ISA instruction counts are not comparable: our fully-unrolled 361-tap shared
    loop is ~3100 instructions with VOPD=0, the reference's rolled LLVM body
    ~183 with heavy `v_dual_*`. End-to-end fps is the only valid comparison.
  - `wave32` shrank the instruction count (3126 → 2862) but lost 1.6–4%: it
    halves subgroups/SIMD and there are no subgroup ops to win back.
  - `#pragma unroll` is a no-op in glslc (byte-identical SPIR-V) and ACO would
    not unroll this body. Manual unscheduling is the lever, not the pragma.
  - Denormal stalls: the minimum weight is ~3e-14, nowhere near subnormal; the
    wide-radius gap was ALU throughput.
  - Reading the source per-pixel instead of through LDS (`use_shared_memory=0`)
    at R=24: 100 fps vs 345 — halo reuse is essential.
  - Vectorising the shared gather to `vec4` changed nothing at R=24 (loads are
    not the limiter).
- **Flagged, not changed**: `tests/test_bilateral.py` still bounds the wide-sigma
  cases by `BORDER_TOL` / `BORDER_TOL_CODES` with a mechanism comment that
  predates the LDS fix; measured now the shared kernel agrees with the reference
  to 1 code / ≤5.6e-9 at every sigma, so the bound is ~655x (16-bit) looser than
  measured. Left for the test-integrity work orders.

## Open work

- **Nothing is open on the transfer path.** The pre-port gap was the missing
  `RADV_EXPERIMENTAL=transfer_queue` opt-in, not a core or filter defect, and it
  is closed. Measure a GPU-resident chain (`--gpu-cache`) before concluding
  anything about the filter's own throughput.
- The only remaining lever is the kernel itself: at R=9 it is 84% of the frame
  and is ALU/`exp2`-bound, and the block-shape, wave32, unroll and vec4 sweeps
  below already came back empty — a kernel project, not a scheduling one.
- **EEDI3-style batching is a measured dead end here; do not implement it.**
  EEDI3 batched because one row dispatch (1080 small workgroups of a
  latency-bound scan) left the GPU under-occupied. Bilateral's R=9 dispatch is
  60×135 = 8100 workgroups of 256 threads and the GPU is already **93–97% busy
  with a single node** (BlankClip GRAY16 1080p, polled `gpu_busy_percent`), so
  there is no idle to fill: chaining k nodes scales the frame linearly
  (0.606 / 1.031 / 1.906 ms for k=1/2/4) and the filter's whole host side is
  23 µs/frame.
- Do not re-introduce a host transfer path to "win back" the CPU sink: the old
  one is exactly what the port deleted, and it lost the resident-chain case.
- Do not retry the LDS/register tradeoffs or the second-compute-queue attempt
  listed under Historical.

## Debug env vars

- `VSFEEL_BILAT_TIMING=1` — per-frame host stage `[bilat-timing]` averages
  (acquire/record/submit/total) at instance destruction.
- `VSFEEL_DEBUG=1|2`, `VSFEEL_TRACE=1|2` — the shared error trace and trail
  (`acquire`/`record`/`submit` marks are recorded in the frame path).
- Removed with the legacy path: `VSFEEL_BILAT_HD`, `VSFEEL_BILAT_KD`,
  `VSFEEL_BILAT_QUEUES`, `VSFEEL_BILAT_NOCPU`, `VSFEEL_BILAT_NODL`,
  `VSFEEL_BILAT_NODISPATCH`, `VSFEEL_BILAT_TRACE`.
