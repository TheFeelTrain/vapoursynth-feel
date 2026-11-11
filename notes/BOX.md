# Box — notes

**Shipped.** `core.vsfeel.Box(clip, width, height)` is a separable box downscale
that runs on the device: each output pixel averages a group of source pixels with
equal weight. It handles every plane of Gray/YUV/RGB, u16, f16 or f32, any
downscale ratio, and refuses upscales.

The reference is `vskernels.Box` (resize2's `Box` kernel), which `vsfeel.Box` is
written to match. Measured on the 640x360 fixture, every depth, every plane:

- bit-exact on u16 at every ratio whose tap counts are all powers of two and at
  the fractional ratios that pair only one or two source pixels (2, 4, 8, 1.5,
  1.25, 1.75, 1.1, 4/3 all measure 0);
- ≤ 2 codes of 65535 on u16 where resize2's own float32 `1/count` is inexact,
  i.e. the tap counts 3, 5, 6, 7 (ratios 3, 2.5, 3.5, 5, 6);
- ≤ 2.4e-7 of the peak on f32 everywhere, and exactly 0 at most ratios;
- bit-exact on f16 at the power-of-two and one/two-pixel fractional ratios other
  than 8 (2, 4, 1.5, 1.25, 1.75, 1.1, 4/3), and one half step (4.88e-4 at the
  unit peak) otherwise. Two mechanisms: our f32 divide is allowed 2.5 ULP on the
  device, so a sum landing exactly on a half tie can narrow to the wrong
  neighbour (3, 2.5, 3.5, 5); and resize2's own AVX2 f16 vertical pass splits its
  window into two 8-tap blocks and round-trips the accumulator through half
  between them (6, 8).

| 1080p GRAY16 source → half size, 1000 f, medians of 3 | 1080p→540p | 4K→1080p |
|---|---:|---:|
| resize2 (CPU), CPU frames both sides | 5522 | 1471 |
| **vsfeel.Box, device frames both sides** | **8143** | **2099** |
| vsfeel.Box, CPU frames both sides | 3408 | 874 |
| control: the same graph with no filter | 10811 | 2434 |

This table predates the wall-clock harness (`--frames` was removed; every arm is
now a `--seconds` window), so use the ratios, not the absolute fps.

Box is the fastest option only in the middle row. Standalone it loses to the CPU
resizer, because an upload and a download cost more than the resize does; it is
worth reaching for when its input and output are already on the device, which is
what `vsfeel.vsaa.feel_aa` does (see `notes/EEDI3AA.md`).

## Implementation

`src/box.cpp` and `src/box.comp`. Two dispatches per plane with a barrier between
them: vertical into a transient per-frame scratch (`gpu_frame_buffer`, in the
frame's own sample type), then horizontal out of it into the output frame's
plane. The scratch is a buffer, not a frame, pitched at the source plane's
stride, so the horizontal pass reads contiguous taps.

The *rounded* intermediate is not an optimisation choice. resize2 quantises the
vertical result to the sample type before the horizontal pass; a single-pass
float accumulate differs from it by 1 code on ~0.2% of pixels. `store_mid`
therefore rounds half away from zero for u16, which is what resize2's integer
output does rather than the `roundEven` the other filters use, and narrows a
half with `OpFConvert`'s round-to-nearest-even for f16. The scratch is the
frame's own sample type at every depth: the f16 path loads half, keeps every
accumulator f32, and narrows once per pass, so it is the f32 arithmetic with the
reference's own storage rounding.

**The tap window is a host-computed table, not a formula.** resize2's box is not
an area average: it averages the source pixels whose *centres* fall inside the
output pixel's box, so at a fractional ratio the tap count alternates and some
output pixels decimate outright (120→80 is 1,2,1,2…). `box_taps()` reproduces
zimg's `compute_filter` coordinate loop expression for expression (the same
`pos = (i + 0.5) / scale`, `round_halfup`, half-integer sample positions and
`(xpos - pos) * step` box test) and stores the resulting contiguous run as
`(first tap, count)`; the shader runs an exact integer loop over it. A
closed-form "cell boundaries" formula disagrees with zimg whenever a boundary
lands exactly on a sample centre (measured: 613→306 and 613→612), which is why
the loop is mirrored rather than simplified. Two tables cover the whole frame:
chroma is the same lattice shifted, so it reads a prefix of the luma table.

An output width/height that a subsampled format cannot store (odd width on
4:2:0/4:2:2, odd height on 4:2:0) is rejected at creation. The output-stride
probe frame is allocated at the requested size, and the core treats an illegal
subsampled size as a fatal error, so this cannot be left to the probe.

Smaller things worth knowing: `skip_vert` drops the vertical dispatch when the
output height is unscaled; a height-only downscale still runs both passes
because the horizontal pass is never skipped. Both strides are read off scratch
frames at creation, because a GPU frame's stride is the CPU frame's; the two
tables are concatenated into one host visible buffer like GaussBlur's weights;
the io variant (u16/f16/f32) is one compile-time `-D` selected from the input
format (half is `stFloat` + 16 bits), so Box ships six SPIR-V variants, two
passes times three depths.

### Tests
`tests/test_box.py` — reference comparison per ratio × depth and per color family
at every depth, boundary ratios on cropped/odd sources (613→306, 613→612), the
subsampled-dimension validation (a subprocess, because the pre-fix path aborted
the interpreter), width-only/height-only, tiny legal targets, parallel load,
residency, identity, determinism, frame properties, validation.
`tests/test_vsaa.py` covers the chain that uses it.

## Performance

The kernel is two dispatches, one load per tap, no shared memory: the vertical
pass runs `SRC_W × DST_H` outputs and the horizontal pass `DST_W × DST_H`.
Subtracting the no-filter control above, Box's own work is **~30 µs/frame at
1080p→540p and ~65 µs at 4K→1080p**; total-to-total (device frames both sides,
which is what the table's fps implies) it is ~123/476 µs against ~181/680 µs for
the CPU resizer. The marginal figure is not comparable to the resizer's total
frame time.

Everything else in the standalone row is the round trip: at 1080p a u16 upload or
download is ~100 µs, so CPU frames on both sides cost ~200 µs more than the
resize itself. With device frames on both sides there are no transfers at all,
and that is the entire win — 1.5x the CPU resizer at 1080p and 1.4x at 4K, plus
the transfer saving it contributes to the `feel_aa` chain. That chain's headline
is driver-dependent (see `notes/EEDI3AA.md`): the same saving that is +11.9%
under the default RADV queue behaviour is +1.5% with
`RADV_EXPERIMENTAL=transfer_queue`, because a real transfer queue hides the
copies this filter removes.

## Historical

- **2026-11-10 — the workgroup shape was swept: 64x4 ships, 32x8 is noise.** Both
  passes were 16x8, so a warp spanned two output rows and only half a 64-byte
  sector per u16 row. A same-session A/B with a shape knob (ABBA, n=12, 3 s,
  medians) puts **64x4** at +2.1% (1080p->540p u16), +1.3% (4K->1080p u16) and
  +1.9% (4K->1080p f32), and flat at 1080p->540p f32 (-0.1%); **32x8** is noise
  across the same four cases (+1.1%, -0.4%, +0.6%, +0.0%). Device's shape is now
  chosen at creation: 64x4 when `maxComputeWorkGroupInvocations >= 256` and a
  64-wide workgroup, else 16x8 (the 128-invocation floor shape). The pixels do
  not move: the local size is the only difference, and a sha1 comparison across
  the shapes is in `tests/test_device_limits.py`.

- **2026-10-12 — the Merge composition.** The first version reproduced resize2's
  rounded 2x2 mean from shipped filters only: `std.Merge(0.5)` of a plane and a
  one-pixel-shifted `Point` copy, bit-exact on u16 and within 1e-6 on f32, with
  no new filter to maintain. It lost on cost — 5–6 full-size passes per
  downscale, twice per `feel_aa` frame, leaving that chain at 135.0 fps against
  143.0 for the native kernel (default driver; both arms measured under it, so
  the +6% is the same-session result, the absolutes are not — see Performance).
- **Fractional ratios were going to be refused.** The first design supported only
  integer tap counts, on the theory that a shader-side `ceil(i*s - 0.5)` at a
  fractional ratio is a float32 tie waiting to move a tap. Computing the window
  in `double` on the host removes that risk, so the general case came back for
  free.
- **A `std.Convolution` composition was rejected**: the core refuses even kernels
  ("matrix must contain exactly 9, 25, 49…"), and this build's `std.Expr` has no
  relative pixel access.
- **2026-10-18 — f16 shipped without full f32.** The scratch stays the frame's
  sample type, so a half clip rounds the intermediate to half like resize2; the
  residuals are under the banner's f16 bullet. A host-computed reciprocal was
  tried to remove the fdiv ties: ACO contracts the multiply into the half store's
  `v_fma_mixlo_f16` and re-rounds the tie once, with or without a `precise`
  qualifier (`NoContraction` survives into the SPIR-V and is ignored), so it
  changed nothing and was reverted.

## Open work

- `width == src.width` and `height == src.height` still runs the horizontal
  dispatch; `skip_vert` drops only the vertical pass (`src/box.cpp:579`). A
  no-op path could copy instead; it is not on any current route.
- The workgroup shape is now measured (64x4 where the device allows 256
  invocations, else 16x8): see Historical. The remaining shape question is
  whether a ratio-dependent width beats both, which nothing points at.
- Do not re-introduce the Merge composition "because it needs no new filter": it
  is bit-exact and ~8 fps slower on the chain that uses this one.
- Do not chase the f16 half-step residue at ratios 3/2.5/3.5/5 with a host
  reciprocal: mechanism under Historical. Ratios 6 and 8 are resize2's own f16
  tap blocking and are not ours to match.

## Debug env vars

- `VSFEEL_BOX_BLK_X=16|64` — forces the workgroup width (`64` is 64x4, `16` is
  16x8); creation refuses a shape the device cannot launch. `VSFEEL_DEBUG=1`
  prints the chosen one as `[box] shape=`.
