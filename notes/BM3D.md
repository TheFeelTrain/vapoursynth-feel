# BM3Dv2 notes

Goal: `vsfeel.BM3Dv2` faster than the reference plugins on the target GPU
(RX 7900 XTX, RDNA3, gfx1100, Mesa RADV), numerically faithful to vszipcl.
This filter had no notes file until the 2026-09-16 correctness pass below, so
nothing about it had been recorded; only the correctness work is written up
here, no tuning has been done yet.

## TL;DR (2026-09-16, correctness pass)

Three creation/kernel-safety fixes. 43/43 tests pass. **No throughput claim**:
nothing added is on the steady-state frame path, and the `NOSEARCH=0`
specialization is unchanged (see the digest check at the bottom).

1. Reject a configuration whose estimate-stack size exceeds the kernel's 32-bit
   `res` addressing.
2. Require a known, positive clip frame count.
3. `BM3D_NOSEARCH` no longer consumes uninitialized shared memory.

## 1. `res` addressing overflows signed 32-bit at 4K / radius 4

**Mechanism.** The host pushes the per-slot base of the estimate stack as an
`int32_t` (`record_bm3d_kernels`, `record_bm3d_agg`) and every `res[]` offset in
`bm3d.comp` is computed in 32-bit `int`
(`src_search`/`src_input` aside, the aggregation uses
`rplane + offset + j * STRIDE`). The buffer itself is
`res_cap * tw * 2 * pe` floats; once that reaches 2^31 the pushed base and the
shader-side sums wrap, and a wrapped base is negative, so the fill and the
atomic accumulation write outside the slot.

**Arithmetic.** `tw = 2r+1`, `res_cap = tw + num_streams + 2r` for `r > 0`.

| config | src_ring | res_cap | res floats | x 2^31 | buffer |
|---|---|---|---|---|---|
| 1080p r=4 ns=4 | 20 | 21 | 783 820 800 | 0.36 | 2.92 GiB |
| 1080p r=4 ns=8 | 24 | 25 | 933 120 000 | 0.43 | 3.48 GiB |
| 4K r=4 ns=4 | 20 | 21 | 3 135 283 200 | **1.46** | 11.68 GiB |

The 4K cell allocates successfully on a 24 GiB card, which is why this is a
silent-corruption bug rather than an allocation failure.

**Fix.** Reject at creation after the plane geometry is known, before any GPU
buffer is created:
`res_cap * tw * 2 * pe > INT32_MAX` →
`"frame is too large: the estimate cache needs N floats per plane (radius R,
num_streams S), which overflows the 32-bit kernel addressing; reduce
num_streams or radius"`.

**Measured (scratch scripts, `tmp/` is gitignored).** `tmp/bm3d_res_overflow.py`
prints the table above and tries the three configs:

- before: `4K radius=4 ns=4: created OK ... -> 2 frames`, `RESULT: overflow
  config was accepted -> the guard is missing`
- after: `4K radius=4 ns=4: Error: BM3D: frame is too large: ... overflows the
  32-bit kernel addressing ...`, controls (1080p r=4 ns=4, 1080p r=2 ns=4,
  1080p r=4 ns=8) still create, `RESULT: overflow config was rejected -> guard
  active`

A permanent creation-time test covers the rejection
(`test_bm3dv2_rejects_int32_res_overflow`); it is cheap because the guard runs
before the buffers are allocated.

## 2. Unknown / empty clip length

**Mechanism.** `nframes` is used as `std::clamp(v, 0, nframes - 1)` in
`acquire_cache`, `release_cache` and the frame path, and the ring tables are
indexed by `f % src_ring`. With `numFrames <= 0`, `clamp` returns `-2` (or
`-1`), and `slot = f % src_ring` is negative, so `src_frame[slot]` /
`src_holders[slot]` index before the base — host heap UB, and a negative
`src_slot` can reach the shader.

**Finding: this state is unreachable on current VapourSynth.** The core rejects
any filter whose `VSVideoInfo` has `numFrames < 1` ("The VSVideoInfo structure
passed by <filter> is invalid", `src/core/vscore.cpp:1704` on master). Measured
with a scratch source plugin (`tmp/fakesrc.cpp`, `tmp/libfakesrc.so`, loaded
with `core.std.LoadPlugin`, no core changes): `numFrames = 24` and
`2^31-1` are accepted, `numFrames = 0`, `-1`, `-2` are all rejected before the
node exists. Built-in filters agree (`std.BlankClip(length=0)` and
`std.Trim(length=0)` are rejected; `std.Loop(times=0)` reports `2^31-1`, not
`-1`). So `-1` cannot be produced by any installed source and cannot be
propagated through any filter.

**Fix.** Still added the `numFrames <= 0` rejection at creation: it is one
branch, it documents the invariant the frame path depends on, and it makes the
filter safe if a future VapourSynth reintroduces unknown-length clips. It is
defence in depth, not a fix that can be exercised on this build — do not spend
time trying to build a repro for it again.

## 3. `BM3D_NOSEARCH` read uninitialized shared memory

**Mechanism.** With `NOSEARCH != 0` the kernel sets its synthetic group
(`gx = x`, `gy = y`, `gz = KRADIUS`) and skips the entire search, but the
aggregation phase unconditionally indexes the shared match tables
(`l_x[group][i]`, `l_y[group][i]`, `l_s[group][i]`, `bm3d.comp`) to build the
`src` patch offsets and the `res` accumulation offsets. The flush of those
tables from the merged group was inside the `else` arm, so the no-search arm
read whatever the previous workgroup left in LDS and used it as block
coordinates (and as a temporal window index for `offset = tmp_z * 2 *
TEMPORAL_STRIDE + ...`). `robustBufferAccess` is off by design, so that is a
wild read and a wild write.

**Fix.** Move the flush block out of the `else` (it still runs the same
`match`/insert logic, which is a no-op for a synthetic group that already
contains the centre 8 times) so both arms write the tables. One `barrier()`
before and after, in uniform control flow.

**Oracle.** A *constant* clip makes every 8x8 patch identical, so a correct
no-search run (eight copies of the centre block) must produce the same constant
the searched run does. `tmp/bm3d_nosearch.py`, 64x64 GRAYS constant 0.5,
3 frames, `radius=2`, `num_streams=1`:

| | search arm | no-search before | no-search after |
|---|---|---|---|
| min / max | 0.49999994 / 0.49999997 | NaN | 0.49999997 / 0.49999997 |
| non-finite | 0 | **12224 / 12288 (99.5%)** | 0 |
| digest (sha256, first 16) | c3921581a525861d | 080eb4202c408a6b | 4b7812ac2dcd102b |

`max |search - nosearch| = 2.98e-8` (one ulp) after the fix, both arms within
`6e-8` of the constant. The one-ulp gap is expected, not a defect: the two arms
give a pixel a different *number* of identical contributions (the no-search
group stacks all eight on the centre block, the searched group spreads them),
and the weighted sums round differently before the division. The no-search
output is reproducible across processes (same digest on repeated runs); the
pre-fix garbage happened to be reproducible too, so the non-finite count and
the constant oracle are the discriminators, not run-to-run determinism.

This is now covered by
`test_bm3dv2_nosearch_matches_search_on_constant_clip`.

## Measured facts worth keeping

- **`NOSEARCH=0` is bit-identical before and after this pass.** The constant
  clip's searched-arm digest is `c3921581a525861d` both pre- and post-fix, so
  the code motion only affected the `NOSEARCH=1` specialization.
- `BM3D_NOSEARCH` is an ablation knob, not a production path: it makes the
  "group" eight copies of the centre block, so the collaborative transform sees
  eight identical patches and the hard threshold keeps only the DC term. It is
  what remains of a vsfeel-invented flag, not something the references have.
- `std.Loop(times=0)` reports `numFrames = INT32_MAX`, not `-1`; that is
  accepted by the `numFrames <= 0` guard (and by BM3D's clamps).
- The `res` guard also bounds the `src` addressing: `src_search(z)` reaches at
  most `(src_ring - 1) * pe * (FINAL ? 2 : 1)`, and `res_floats >=` that for
  every radius / stream count (for `r = 0`, `res = 2*ns*pe` vs
  `src = 2*(ns-1)*pe`; for `r > 0` the `tw * 2` factor dominates), so a
  configuration whose source offsets would wrap is rejected by the same check.
- **Not fixed here** (creation-time parameter validation, a separate change):
  `bm_range` and `ps_range` are only checked for positivity, so
  `bm_range = INT32_MAX` divides by zero in the shader, and `extractor_exp` has
  no upper bound (`std::ldexp(1.0f, 200)` is NaN).
