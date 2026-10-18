# Bilateral — notes

Status: **shipped on the R80 GPU API.** `clip`/`ref` are `vnode:gpu` and the
output carries `ffGPUOutput`, so the core's `GPUUpload`/`GPUDownload` cross the
bus and the filter owns only its pipelines and one exec pool. Every plane is the
core's own GPU frame plane: the kernels read the source and guide in place and
write the output frame in place. The old per-stream staging buffers, VRAM
src/dst buffers, queue cap and fence are gone.

Design (current):
- Three kernel tiers per plane, chosen at creation: the two-column
  `bilateral_shared.comp` (`-DCOLS2`, the default), its one-column form
  (`bilateral_shared_*_x1`) and `bilateral_plain.comp`. The two-column kernel
  runs two adjacent output columns per thread off one tile read per tap: their
  windows overlap in 2R of 2R+1 columns, so the LDS reads halve. It is worth
  ~16% over the one-column shared kernel. A plane drops a tier when its tile
  does not fit the device's LDS (28672 B vs 20480 B at R=24/32x16).
- Three io variants — u16, f16, f32 — selected by the input format (half is
  `stFloat` + 16 bits). A half element is a storage width, not a precision: the
  f16 path runs the f32 tile and arithmetic, widens exactly on load and narrows
  once on store (OpFConvert, round to nearest even), so its output is the f32
  output rounded once and sits within one half step of vszipcl's f16 output.
- Three whole-buffer bindings per dispatch (src, dst, guide); without a guide the
  source stands in on binding 2 and `HAS_REF=0` means the shader never reads it.
- Native `uint16_t`/`float16_t`/`float` SSBOs at `#version 450`, no push
  constants; every geometry/weight parameter is a specialization constant
  (`WIDTH/HEIGHT/STRIDE/RADIUS/SIGMA_*/HAS_REF/TILE_*/BLOCK_*`).
- `STRIDE` is the core's plane pitch in elements, read off a scratch frame at
  creation (a GPU frame uses the identical stride math).
- Auto block shape (`32x8` for max radius ≤ 12, `32x16` above); explicit
  `block_x`/`block_y` are respected and clamped to the device limits. The block
  is the thread count, so with the two-column kernel a workgroup covers
  `2*block_x` columns and the tile is `2R + 2*block_x` wide.
- Unprocessed planes (`sigma < FLT_EPSILON`) ride along from the source through
  `newVideoFrame2`, keeping their own producer pairs; nothing is submitted when
  every plane is unprocessed.
- `num_streams` and `device_id` are registered no-ops (never read): depth is
  the core's call and device choice is `core.set_vulkan_device`.
- **`RADV_EXPERIMENTAL=transfer_queue` is required for the numbers below**
  (mechanism in Performance), and `tools/benchmark.py` now forces it. Every
  −16% figure in this file's history predates that and is void.

Scoreboard — 1080p jpbd, 5000 frames, sigma 3.0/0.02 (R=9), medians of 3
through `tools/benchmark.py`. u16 input unless noted:

| mode | vsfeel u16 | vszipcl | ratio | vsfeel f16 | vsfeel f32 |
|---|---|---|---|---|---|
| CPU cache (benchmark default) | 2658.3 | 572.4 | 4.64x | 2660.5 | 1480.9 |
| `--gpu-cache` | 2684.8 | 710.7 | 3.78x | 2677.2 | 1938.9 |
| `--gpu-cache --no-download` | 2689.6 | 709.2 | 3.79x | 2692.7 | 2662.0 |

- The two-column kernel lifted every compute-bound row by ~15–16% (u16 2343 →
  2690, f16 2346 → 2693, f32 2326 → 2662, compute-only). The reference rows are
  unchanged, so the ratios moved with it (u16 3.22x → 3.79x).
- Compute-only the three depths tie: the kernel is `exp2`-bound and the tile and
  arithmetic stay f32, so the io width buys nothing there. It shows only where
  the output is downloaded (f32 CPU cache 1481 against f16 2661 — the download
  is twice the bytes), and the transfer-carrying modes are within ~1% of
  compute-only for u16/f16, i.e. the transfers overlap the kernel.

## Implementation

`src/bilateral.cpp` + `src/bilateral_shared.comp` / `src/bilateral_plain.comp`.

- Per plane at creation: `BilateralPlaneConfig{width, height, stride, pipeline,
  grid_x, grid_y}`; identical plane configs share one `VkPipeline`.
- The input format's `bits`/`half` select the SPIR-V pair: u16 →
  `bilateral_{shared,plain}_u16`, f16 → `..._f16`, f32 → `..._f32`. The LDS
  tile and the tap arithmetic are f32 at every depth; only the element type and
  the load/store conversions change.
- Frame path: `getGPUPlane` on src/guide/dst, bind pipeline, `gpu_push_buffers`
  (3 storage buffers via push descriptors, `VK_WHOLE_SIZE`), `vkCmdDispatch`,
  then `gpuExecReadsFrame`/`gpuExecWritesPlane`/`gpuExecSubmit`. No barrier is
  needed: the plane dispatches touch disjoint buffers.
- The kernel tier is per plane: the two-column shared kernel while
  `(1 + has_ref) * (2R + 2*block_x) * (2R + block_y) * 4` fits
  `maxComputeSharedMemorySize`, else the one-column shared kernel while
  `(1 + has_ref) * (2R + block_x) * (2R + block_y) * 4` does, else the plain
  gather. `use_shared_memory=0` forces the last. `SHARED_FLOATS` is exactly what
  `buf[SHARED_FLOATS]` reserves (the output goes straight to `dst[]`), and the
  grid follows the chosen column count.
- The two-column kernel indexes the tile from the workgroup's first plane column
  (`gx0 = group.x * 2*gl_WorkGroupSize.x`), so the load's `clamp` and the
  accumulation share one base. Per row it walks the `2R+2` columns spanning both
  windows once: the `k <= 2R` / `k >= 1` tests fold, so each tile value feeds
  both outputs and carries no per-tap predicate. The last column of an odd-width
  plane is skipped at the store only.
- The 32-bit element-offset guard rejects a plane whose last element
  `(h-1)*stride + w - 1` would not fit an int.
- `round(sigma_spatial*3)` is clamped to 1e6 before the float→int cast (the
  reference clamps identically); `sigma_spatial`/`sigma_color` reject NaN/inf.
- u16 rounding is `uint(v*PEAK + 0.5) & 0xFFFF`, matching the reference's
  round-half-away inside the 1-LSB test tolerance. f16 rounding is
  `float16_t(v)` — an OpFConvert, round to nearest even with no saturation, the
  reference's `vstore_half_rte` — and half->float widening is exact, so the
  f16 path adds no arithmetic precision loss of its own.

## Performance

- **Two output columns per thread: +16%, and the win was a predicate, not the
  LDS.** Sharing each tile read between the windows of columns x and x+1 is
  worth ~2.75% on its own. Putting the "is there a second column" test *inside*
  the innermost loop as a select cost the other 13 points: ACO emitted two
  `v_cndmask_b32` per tap (722 per thread) and, more importantly, could no
  longer keep the body in registers (430 instructions, VGPR 48). Moving the test
  to the store alone dropped the body to 381 instructions and measured
  2319 → 2694 fps (u16/f32 compute-only, 5000 frames, 3-repeat medians; +16.0%
  on an ABBA A/B, n=24 per arm). Same mechanism as
  `notes/GAUSSBLUR.md`'s skipped padding taps: one select per tap in an unrolled
  body is never free.
- **The two-column kernel rolls the row loop, and that is what freed the
  registers.** Full unrolling of 361 taps for two outputs does not fit: ACO
  keeps the inner 2R+2-tap loop unrolled and rolls the 19 rows instead, so the
  kernel runs at VGPR 48 / 28 subgroups per SIMD against the one-column
  kernel's 192 / 8, with no spills. The cost is that `cy*cy` is no longer a
  constant, so the spatial term is a per-row value rather than a literal; the
  LDS and occupancy gains still dominate.
- **The gain tracks the radius.** Same-session A/B against the previous build
  (u16, compute-only): R=3 +9.7%, R=9 +17.1%, R=12 +3.8%, R=24 +1.1%, R=36
  +8.6%. The tier choice stays right where the gain shrinks — at R=24 the
  two-column kernel is still +1.2% over the one-column one — but the auto block
  rule (`32x8` R≤12, `32x16` above) was tuned on the one-column kernel, so the
  R=12 dip is most likely a shape artifact rather than the kernel.
- **The one-column kernel is `exp2`-bound, and `exp2` is 44% of its frame.** An
  ablation ladder on the old kernel (f32, R=9, compute-only): deleting `exp2`
  alone 2324 → 4174 fps, deleting the whole exponent term (three ALU ops: the
  difference, its `sc` multiply and the fused `space*ss + range*sc`) 2324 →
  7855. So `exp2` ≈ 190 µs, the exponent math ≈ 113 µs, LDS + accumulate ≈
  127 µs of the 432 µs frame. `exp2` is one SFU op per tap (~6 cycles per
  wave32 instruction, near the SFU's throughput) and 361 calls/pixel is the
  algorithm's floor at R=9, so the remaining headroom is in hiding the rest
  behind it, not in the weight math.
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
  EEDI3AA +13%, BM3D +10% — each at 97–100% of its "download node deleted"
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
  (R=9, u16, pipeline cache off): new 3101 instructions / 14700 B code / 192
  VGPR / 0 spill; pre-R80 3102 / 14708 / 192 / 0. `glslc -O` output is 315 vs 321
  SPIR-V lines. The shader rewrite (three whole-buffer bindings, no push
  constants) did not move the machine code.
- **The filter, not the transfer, is the frame.** A bare
  `GPUUpload→GPUDownload` round trip of one 1080p plane measures 264 µs
  (3786 fps) against Bilateral's 372 µs (2690 fps) and the transfer-carrying
  modes land within ~1% of compute-only for u16/f16: the transfers overlap the
  kernel and the output download is not on the critical path for a 2-byte
  sample. At f32 the download is twice the bytes and stops overlapping
  (CPU cache 1481 against 2662 compute-only). The earlier reading (a 414 µs
  round trip with "only ~0.10 ms exposed") was measuring the harness's own
  per-frame cost (`notes/METHOD.md`), not the core's.
- Host split (`VSFEEL_BILAT_TIMING=1`, 3000 frames): acquire 19.1 / record 5.8 /
  submit 59.0 µs per frame — the host is nowhere near the wall.
- The port wins exactly where a resident chain is involved: fed a GPU clip the
  old filter had to `GPUDownload` it first, the new one reads VRAM in place.

- **Half storage costs nothing on compute and one step on accuracy.** The f16
  kernel is the f32 one plus an `OpFConvert` on every tile load and on the store
  (the 361-tap sweep is the same shape at both depths), so compute-only is a
  tie: 2693 fps f16 vs 2662 f32 (1080p, sigma 3.0/0.02, 5000 frames, 3-repeat
  medians). Every f16 reference config measures exactly 4.88e-4 (one half step
  at |v| ∈ [0.5, 1)) on a few tens of samples per frame, and 125 of 2.07M
  samples differ from the *same-input* f32 run: the extra `OpFConvert` shifts
  ACO's contraction by one instruction, so the two variants' f32 sums differ in
  the last bits and the store rounds that to a step. That is the storage width's
  price, not the arithmetic's.
- **Both full-half prototypes lost, for the GAUSSBLUR mechanism.** Measured
  against the then-shipped one-column kernel: a half LDS tile measured 2250 fps
  (−3.6%): a scalar half load still takes a load slot and every tap adds a
  widening convert, so halving the bytes saves nothing. Half tile *and* half tap
  arithmetic (`float16_t` LDS, weights and accumulator) measured 2270 fps
  (−2.6%) while drifting up to 9.3e-3 (~19 half steps) off the f32 result. ACO
  emits scalar `v_fma_f16`/`v_exp_f16` for this shape, never a packed `v_pk_*`,
  so the FMA rate does not move and the converts cost more than the width saves
  (`notes/GAUSSBLUR.md` records the same result for its own shape).

Benchmark call: `MANGOHUD=0 uv run tools/benchmark.py --filter bilateral
[vsfeel vszipcl] [--gpu-cache] [--format u16|f16|f32] [--bilateral-args
"sigma_spatial=X, sigma_color=Y"]`.

## Historical

- **2026-10-12 — the R80 port's scoreboard is void.** It read 1982.1 → 1980.5
  (CPU cache) and 1975.3 → 1974.0 (`--gpu-cache`) fps, measured against the
  pre-fix benchmark harness (`notes/METHOD.md`); the same-session result it
  carried (the port changed nothing, **−0.1%**) stands. The banner's table
  replaces the absolute numbers.
- **2026-09-30 — u16 SSBOs need the StorageBuffer class.**
  `bilateral_shared_16`/`bilateral_plain_16` carried the same invalid SPIR-V 1.0
  Uniform+BufferBlock 16-bit buffer access as GaussBlur (mechanism and feature
  names in `notes/GAUSSBLUR.md`); both now build with `--target-env=vulkan1.1`.
  Bit-exact vs vszipcl, f32 SPIR-V unchanged.

### 2026-09-30 — creation-failure leak

`createVideoFilterEx2` returns `nullptr` without running the free callback when
the node constructor throws, so the instance released just before the call would
leak its buffers and both node references. The `nullptr` arm now releases through
`BilateralFree`; same mechanism as `notes/DFTTEST.md`. No perf change.

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
- **An over-wide plane is refused, not truncated.** The kernel addresses a tile
  as (ID.x, ID.y), so a width past `maxComputeWorkGroupCount[0]` has nowhere to
  fold: the grid used to be clamped and the tail silently unwritten (2 097 120 px
  at 32-wide blocks on a 65 535-group device). Creation now errors naming the
  grid and the device's limits.
- **`radius` is bounded at 32767** (correctness-only, no perf change). The tile
  math (`2*radius + block`) and the squared-distance term are signed 32-bit: the
  explicit branch reached `INT_MAX`, wrapping the tile so a shared kernel could
  be built whose LDS indices ran off the end (or a plain kernel that loops over
  the plane), and the sigma-derived branch reached 1e6. Both are now rejected at
  creation.
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
- **Test bounds tightened to the measured agreement, gate covered** (test-only,
  no perf change). The wide-sigma cases were bounded by a `BORDER_TOL`
  (655 codes of u16) whose comment described a staging-tile/border mechanism
  that is not on the shipped path; the shared kernel agrees with the reference
  to 1 code / ≤5.6e-9 at every sigma, so the bound is now REF_TOL and 2 codes.
  Added coverage for the automatic LDS gate (a forced 12288-byte device must
  pick the plain kernel — radius 24's 20 480 B one-column tile no longer fits —
  and match the explicit plain run bit-for-bit) and for the per-plane share path
  (`sigma_spatial=[2.0, 0, 0]` must copy chroma through bit-identically). With
  three kernels, `test_device_limits.py` pins all three tiers from the creation
  banner's `lds=` (28672 / 20480 / 0) and the mixed per-plane case.

## Open work

- The kernel is at ~372 µs/frame against a ~211 µs `exp2` floor, so the lever
  left is hiding the ALU/LDS behind the SFU, not shortening the weight math. The
  two-column kernel's shape was swept at R=9 only (32x8 still wins there); the
  wide-radius auto rule (`32x16` above radius 12) was tuned for the one-column
  kernel and has not been re-swept for the two-column one.
- **The exponent math is provably minimal at three ALU ops.** `sc*d² + lit` needs
  the difference, a square (or an `sc`-scaled product) and the fused `+ lit`;
  `v_fmaak` only folds a *linear* literal term, so there is no two-op form that
  keeps the subtraction's cancellation.
- A packed two-pixels-per-lane f16 kernel is the only untried f16 form. It would
  halve the per-tap ALU, but `exp2` is emitted per tap (`v_exp_f32`, a
  quarter-rate SFU op) and cannot be packed, so it cannot halve the frame and is
  unlikely to clear the 30% bar that would justify the accuracy loss.

### Do not retry

- **Pre-scaling the tile to fold the exponent to two ops.** Scaling the tile by
  `k = sqrt(-SIGMA_COLOR_SCALED)` at load time makes the exponent
  `fma(-d, d, lit)` (two ops instead of three) and measured **+6.8%** — but it
  scales *before* the subtraction, so `fl(v*k) - fl(c*k)` loses the
  cancellation that makes `v - c` accurate when `v ≈ c`, i.e. exactly the taps
  whose range weight is near 1. Measured 1.7e-6 (default sigma) and 3.8e-6
  (sigma_color 0.15) against the reference where f32 asserts < 1e-6. The same
  mechanism rules out any pre-scaled tile, power-of-two scale included (the
  residual `sc/4^m` still needs its own multiply, which puts the op count back).
- **A predicate inside the tap loop.** The two-column kernel's `second` test
  cost 13 of its 16 points as two `v_cndmask_b32` per tap; moving it to the
  store is what the win actually was. Any per-tap select in an unrolled body
  needs its price measured before it ships.
- **Half LDS, and half LDS with half tap arithmetic** (measured, mechanism under
  Performance): −3.6% and −2.6% against the f32 tile, the second for up to 19
  half steps of error. Do not retry without a code shape ACO actually packs into
  `v_pk_*`.
- **EEDI3-style batching.** EEDI3 batched because one small latency-bound
  dispatch left the GPU under-occupied; Bilateral's R=9 dispatch is 60×135 =
  8100 workgroups of 256 threads at 93–97% busy with a single node, so there
  is no idle to fill (chained nodes scale linearly).
- Do not re-introduce a host transfer path to "win back" the CPU sink: the old
  one is exactly what the port deleted, and it lost the resident-chain case.
- The LDS/register tradeoffs and the second-compute-queue attempt under
  Historical.

## Debug env vars

- `VSFEEL_BILAT_TIMING=1` — per-frame host stage `[bilat-timing]` averages
  (acquire/record/submit/total) at instance destruction.
- `VSFEEL_DEBUG=1|2`, `VSFEEL_TRACE=1|2` — the shared error trace and trail
  (`acquire`/`record`/`submit` marks are recorded in the frame path).
- Removed with the legacy path: `VSFEEL_BILAT_HD`, `VSFEEL_BILAT_KD`,
  `VSFEEL_BILAT_QUEUES`, `VSFEEL_BILAT_NOCPU`, `VSFEEL_BILAT_NODL`,
  `VSFEEL_BILAT_NODISPATCH`, `VSFEEL_BILAT_TRACE`.
