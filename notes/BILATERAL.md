# Bilateral — performance notes

Status: **done for now** — vsfeel beats vszipcl everywhere except the
extreme wide-sigma edge (R=24). Default config (GRAY16, sigma 3.0/0.02):
**ns=4: ~1975 vs ~1520 (+30%)**, ns=1 +26%, ns=2 +121%, ns=8 +18%,
32-bit +67%, YUV420P16 +76%. Target was +10% at ns=4.

Benchmark call: `MANGOHUD=0 python3 benchmark/bench.py --filter bilateral
[--num-streams N] [--bits 32] [--bilateral-sigma-spatial X
--bilateral-sigma-color Y] vsfeel vszipcl`

- Input: `/home/thefeeltrain/Encode/test/jpbd.mkv` 1920x1080, GRAY16
  (`depth(get_y(clip), 16)`), 5000 frames cached in RAM.
- sigma 3.0 → radius 9 → shared-memory tiled path on every frame.
- Target GPU: RX 7900 XTX (gfx1100), RADV. ReBAR enabled (memoryType 3 =
  DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT, host-mapped VRAM).

## Final scoreboard (ns=4 real-clip medians, vszipcl in parens)

| config | vsfeel | vszipcl | delta |
|--------|--------|---------|-------|
| default R=9 16-bit | 1975 | 1520 | **+30%** |
| R=3 (sigma 1.0) | 2528 | 1665 | +52% |
| R=12 (sigma 4.0) | 1302 | 1279 | +2% |
| R=18 (sigma 6.0) | 652 | 595 | +9% |
| R=24 (sigma 8.0) | 345 | 367 | −6% (only loss) |
| 32-bit default | 1288 | 771 | +67% |
| YUV420P16 3-plane | 1610 | 915 | +76% |
| ns=1 / ns=2 / ns=8 (default) | 908/1828/1970 | 718/826/1663 | +26/+121/+18% |

ns=6: 1985 vs 1645 (+21%). Knee is ~6; default num_streams stays 4
(benchmark-graded depth; per-stream VRAM ~8 MB: 4 MB src + 4 MB staging).

> **R≥18 rows above predate the WO-03 LDS fix (see below)** — the reserved
> LDS changed at every radius and the kernel selection changed for
> R∈[28,43] (no-ref) / R∈[21,27] (guide), so those rows must be re-measured
> by WO-24 before being quoted.

## Implementation (current)

Mirrors the GaussBlur VRAM structure (see `notes/GAUSSBLUR.md`):

- Per-stream device-local `src_buf` (VRAM, holds src+ref planes as **native
  `uint16_t`/`float` elements**) + host-visible cached `staging` (upload +
  download regions, byte-identical layout to the VRAM buffers).
- **Host-direct upload** (default; `VSFEEL_BILAT_HD=0` opts out): the src
  VRAM is ReBAR host-mapped (memoryType 3) and the CPU memcpy lands bytes
  directly in VRAM with plain `memcpy` (WC window: streaming stores measured
  slower, same as GaussBlur). No GPU-side H2D copy. Falls back to plain
  VRAM + in-CB `vkCmdCopyBuffer` when no host-visible device-local type
  exists (the `allocate_memory` relaxation is re-checked; the flag is global
  so mixed allocation across resources in one instance is not supported —
  fine on ReBAR GPUs where it never triggers).
- **Kernel-direct download** (default; `VSFEEL_BILAT_KD=0` opts out): the
  kernels' plain coalesced stores write the GTT staging download region
  over PCIe directly (posted writes are cheap); no `dst_buf`, no D2H copy.
  With both fast paths on, the per-frame GPU work is **kernel only**.
- Shaders use native `uint16_t`/`float` SSBOs
  (`GL_EXT_shader_16bit_storage`, `storageBuffer16BitAccess` already enabled
  device-wide): no shift/mask unpacking, no dword packing, no `atomicOr`,
  no `vkCmdFillBuffer` pre-clear, no `out_tile` pack dance. Push constants
  are element offsets. 16-bit rounding is `uint(v*PEAK+0.5)&mask`
  (matches the reference's round-half-away within the 1-LSB test tolerance;
  convex-combination outputs stay non-negative).
- **Adaptive default workgroup shape** (only when the user leaves
  `block_x`/`block_y` unset): `32x8` for max_radius ≤ 12, `16x16` above.
  Explicit args are always respected.
- **Queue cap 2** (`num_queues = min(ns, queue_count, 2)`; override with
  `VSFEEL_BILAT_QUEUES=N`): sharing a queue across streams keeps a next CB
  queued while a worker does post-fence CPU work; 1:1 stream:queue leaves
  idle bubbles (ns=4: 2 queues = 1986 fps vs 4 queues = 1709 fps).
- Fixed env-gated `[perf]` probe timer bug: `t_up` used to include the
  `pool.take()` wait (no clock reset after acquire). True costs at ns=4:
  acq 13.3 / up 0.29 / sub 0.02 / wait 1.46 / down 0.21 ms.

## How the wins break down (default R=9, ns=4; ablation medians)

1. VRAM rework + u16 native + HD + KD: 1316 → 1735 (**+32%**).
   HD alone is worth +34% (1730 vs 1289 with `VSFEEL_BILAT_HD=0`);
   KD is worth +3% (1730 vs 1678 with `VSFEEL_BILAT_KD=0`).
2. Default block 32x32 → auto (32x8 at R=9): single-digit % on top
   (32x32 also spilled 93 VGPRs + 23 KB scratch at 3541 instr; all smaller
   blocks sit at ~3160 instr, 0 spill).
3. Queue cap 4 → 2: 1709 → 1986 (**+16%**), and killed most run-to-run
   variance (±0.7% across 5 pairs vs ±4% before).

## Reference structure (same GPU, READ-ONLY)

vszipcl (`reference/vapoursynth-zipcl/src/bilateral.zig`): per-stream pinned
staging + device-local `d_src`/`d_dst` (float32) + `convert_in`/`convert_out`
boundary kernels + baked (`-DBAKE_W/H/S/R`) `bilateral_sm` (16x8) or
`bilateral_gl`, 4 streams, `clFinish` per frame. rocprofv3 ns=1 BlankClip
GRAY16: sm **484 µs** + convert 42 µs + H2D/D2H **150 µs each**.
Its scaling (ns=1 727 → ns=4 1500 = 2.06x) beats a naive port because H2D/D2H
run on SDMA engines full-duplex with kernels; our HD+KD removes the copies
instead of overlapping them.

## Analysis trail (dead ends kept so they are not retried)

- **ISA comparison (the trap):** our R=9 shared shader is 3126 instr, 192
  VGPR, 0 spill, VOPD=0; the reference `bilateral_sm` section is ~183 instr
  with heavy `v_dual_*` packing. The counts are not comparable (different
  unroll points: we fully unroll 361 taps, LLVM keeps a rolled loop with
  dual-issued body). End-to-end fps is the only valid comparison.
- **Default block was 32x32** (tile 50x50 = 2500 floats, LDS 26 KB, VGPR
  spill): the reference uses 16x8 (tile 34x26 = 884). Block sweep
  (BlankClip ns=4): 32x4=1445, 16x8=1421, 32x8=1404, 16x16=1403, 32x32=1043.
- **Wide-sigma (R=24) gap** (ours 228 → 345 after auto-16x16, ref 367):
  box-blur probe (weight=1) 197→628 fps, spatial-only probe (keep exp, drop
  range ALU) 197→412 fps → the loop is ALU-throughput-bound, not
  exp-latency-bound; the reference's LLVM dual-issue packing (VOPD) vs ACO's
  VOPD=0 is the structural gap. Denormal-stall theory killed by arithmetic
  (min weight exp2(−45) ≈ 3e-14, nowhere near subnormal). Manual `#pragma
  unroll` on the inner loop: no effect (565 instr before/after — ACO
  ignores it at that size). `use_shared_memory=0` at R=24: 100 fps (LDS
  halo-reuse is essential; plain re-reads 2401 taps/px from VRAM).
- **wave32 tried, reverted:** 3126→2862 instr but fps −1.6% default / −4%
  wide; Subgroups/SIMD halved 16→8 (fewer waves in flight, no subgroup ops
  to benefit — dfttest's win does not transfer).
- **Regime variance (important methodology note):** identical binaries swing
  ±15–40% between invocations at ns=2 (1090 vs 1900), while vszipcl stays
  ±1%. Same-session pairs stay fair (both plugins share the power state);
  the queue cap reduced our swing to ±0.7% at ns=4. Judge only same-session
  pairs / medians over 1000+ frames — never single short bursts.
- **Queue bubble finding:** ns=6 outran ns=4 by 23% (1985 vs 1604) while the
  reference gained only 10% → the GPU was starved at ns=4, not saturated.
  Capping queues at 2 fixed it. Per-frame GPU work is kernel-only (HD+KD),
  so overlap quality is everything.
- **Adaptive-block matrix** (ns=4 real-clip): R=3: 32x8 best (2449);
  R=9: 32x8 (1742) > 16x8 (1707) > 16x16 (1705); R=12: 32x16 (1360) ≈ 32x8
  (1350) > 16x16 (1308); R=24: 16x16 (345) > 32x16 (318) ≫ 16x8 (228).
  Threshold R>12 → 16x16 is optimal as implemented; re-verified under the
  queue cap (the block_y effect shrank once bubbles were gone).

## WO-03 — LDS tile over-reserved by one whole tile (FIXED)

`bilateral.cpp` sized the shared kernel's spec-constant array as
`(2 + has_ref) * tile_x * tile_y * 4` ("source tile(s) plus the output tile"),
but `bilateral_shared.comp` keeps only `(1 + has_ref)` tiles — the output
goes straight to `dst[]` (`ref_offset = HAS_REF*TILE_Y*TILE_X`, no output
tile). The reference sizes identically (`bilateral.zig:180`). Since
`SHARED_FLOATS` sizes `shared float buf[SHARED_FLOATS]`, the host's number
*was* the reserved LDS. Fixed to `(1 + has_ref)` in both places.

Creation-time probe (`VSFEEL_BILAT_LDS_TRACE`, temporary; removed after use),
640x360 GRAY16, auto block shape (`32x8` R≤12, `32x16` above):

| R | has_ref | tile | new shared | old shared | new use_shared | old use_shared |
|---|---|---|---|---|---|---|
| 9 | 0 | 50x26 | 5 200 | 10 400 | 1 | 1 |
| 9 | 1 | 50x26 | 10 400 | 15 600 | 1 | 1 |
| 12 | 1 | 56x32 | 14 336 | 21 504 | 1 | 1 |
| 18 | 1 | 68x52 | 28 288 | 42 432 | 1 | 1 |
| 21 | 1 | 74x58 | 34 336 | **51 504** | 1 | **0** |
| 24 | 0 | 80x64 | 20 480 | 40 960 | 1 | 1 |
| 24 | 1 | 80x64 | 40 960 | **61 440** | 1 | **0** |
| 27 | 1 | 86x70 | 48 160 | **72 240** | 1 | **0** |
| 28 | 0 | 88x72 | 25 344 | **50 688** | 1 | **0** |
| 28 | 1 | 88x72 | **50 688** | 76 032 | **0** | 0 |
| 32 | 0 | 96x80 | 30 720 | **61 440** | 1 | **0** |
| 32 | 1 | 96x80 | **61 440** | 92 160 | **0** | 0 |
| 43 | 0 | 118x102 | 48 144 | **96 288** | 1 | **0** |
| 45 | 0 | 122x106 | **51 728** | 103 456 | **0** | 0 |
| 45 | 1 | 122x106 | 103 456 | 155 184 | 0 | 0 |

Matches REPORT P0-3 exactly: the old gate dropped the tiled kernel for
no-ref **R ∈ [28,43]** and guide **R ∈ [21,27]**; the fix keeps it there and
halves/third-s the reserved LDS everywhere else (default R=9 32x8: 10 400 →
5 200 B, i.e. 6 → 12 workgroups/CU of 64 KiB).

**Measured effect (screen, not graded).** 1920x1080 GRAY16 BlankClip, R=32
no-ref, ns=4, `vspipe -e 499`, 2 alternating reps: plain kernel (what the
old gate chose at R=32) **54.1 / 54.0 fps** vs shared kernel (fixed) **205.9
/ 205.1 fps** — ~3.8x. This is the R∈[28,43] fallback cliff, now reopened.

**Default config is unchanged.** Same-session pre/post A/B (both binaries built
from this tree and hash-verified at each arm; 1920x1080 GRAY16/GRAYS BlankClip,
R=9, ns=4, 1500 frames, 3 order-alternating reps, medians): u16 **1992.0 →
1998.4 fps** (+0.3%), fp32 **1268.6 → 1259.8 fps** (−0.7%) — both inside the
run-to-run swing, so the README default-config row needs no numeric change.
The R=9 shared kernel is evidently not LDS-occupancy-limited (VGPR-bound at
~192 VGPR); only the radius bands near the 48 KiB gate were.
The graded pre/post at R=24/R=32 was attempted and blocked: swapping the two
binaries in the system plugin directory was denied by the sandbox, and the
"Final scoreboard" R≥18 rows are from other sessions, so no same-session
graded number exists for them yet (WO-24 should take it).

**Correctness in the newly-shared band** (noise_24f, 640x360 GRAY16/f32,
frames 3+17, vszipcl reference):

| config | shared-vs-ref | plain-vs-ref |
|---|---|---|
| R=32 no-ref 16-bit | 1 code | 451 codes |
| R=32 no-ref 32-bit | 5.6e-9 | 6.9e-3 |
| R=43 no-ref 16-bit | 1 code | 451 codes |
| R=24 +ref 16-bit | 1 code | 477 codes |
| R=24 +ref 32-bit | 5.6e-9 | 7.3e-3 |

So the fix *improves* reference agreement in the changed band (the plain
kernel's border semantics are the divergent side, ~7e-3), it does not trade
correctness for speed.

**R=24 (the documented R=24 loss) is unchanged in kind:** old reserved
40 960 B ≤ 48 KiB → shared both before and after; only the reserved LDS
drops (40 960 → 20 480 B → 1 → 3 workgroups/CU). The notes' "ACO VOPD=0"
explanation for the 345-vs-367 loss may now be the only remaining cause, but
the occupancy change means **R≥18 must be re-measured (WO-24)**; do not
inherit the table in "Final scoreboard" for R≥18.

**Flagged, not changed:** `tests/test_bilateral.py:142-144` (16-bit) and
`tests/test_bilateral.py:204-205` (32-bit) justify `BORDER_TOL_CODES`/
`BORDER_TOL` by "staging tile exceeds the 48 KiB budget". Measured now, the
shared kernel agrees with the reference to 1 code / ≤5.6e-9 at *every* sigma
including the wide-sigma R=24 case, so the stated mechanism is not active
and the tolerance is ~655x (16-bit) / ~10^5x (32-bit) looser than measured.
The R=24 kernel choice is not affected by WO-03, so the comment was already
stale before this fix. Flagged for the test-integrity work orders (WO-48/
WO-55); left untouched per this work order.

## Debug env vars

- `VSFEEL_BILAT_HD=0` — staging + H2D copy instead of host-direct VRAM upload.
- `VSFEEL_BILAT_KD=0` — VRAM dst + D2H copy instead of kernel-direct download.
- `VSFEEL_BILAT_QUEUES=N` — override the queue cap (default 2).
- `VSFEEL_BILAT_TRACE=1` — host phase `[perf]` averages every 200 frames.
- `BILATERAL_NOCPU` / `BILATERAL_NODL` / `BILATERAL_NODISPATCH` — pre-existing
  diagnostics (empty upload / skip download / extra null dispatch).

## Validation hardening (cross-cutting pass)

`num_streams` is now `1..32` (was `> 0` only, so `1000` reached a 1000-deep pool
and a 2000-set descriptor pool before failing mid-loop), `sigma_spatial` /
`sigma_color` reject NaN and infinity, and the default-radius derivation clamps
`round(sigma*3)` to `1e6` *before* the float-to-int cast (huge finite sigma was
UB; the reference clamps identically). Flush/invalidate ranges go through the
shared `mapped_range` helper (see `notes/EEDI3AA.md` round 8). No performance
change; all `test_bilateral.py` tests pass.

## NT-store ordering

The guide plane is always NT-stored (`copy_plane_out(..., nt=true)`, hardcoded)
and the source planes are too on the staging path, but nothing ordered those
weak stores before the submit that tells the kernels to read the upload window,
so a kernel could see stale or partially written input. An unconditional
`_mm_sfence()` now sits immediately before `submit_with_fence`, matching
EEDI3/NNEDI3. The requirement is documented on `copy_stream_out` /
`copy_plane_out` in `vsfeel.h`. Correctness-only change, no fps effect
expected; `test_bilateral.py` passes.

## Creation and frame error paths

The per-stream resource is created straight into the pool
(`FramePool::emplace()`), so any error return inside the creation loop leaves it
to `~BilateralData` instead of abandoning its buffers, device memory, command
pool, fence and mapped windows. The frame-path `set_error` also frees `dst`,
which used to leak one full output frame per failed frame. Correctness-only; the
full suite passes.
