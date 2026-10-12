# Box — notes

**Shipped.** `core.vsfeel.Box(clip, width, height)` is a separable box downscale
that runs on the device: each output pixel averages a group of source pixels with
equal weight. It handles every plane of Gray/YUV/RGB, 16-bit integer or 32-bit
float, any downscale ratio, and refuses upscales.

The reference is `vskernels.Box` (resize2's `Box` kernel), which `vsfeel.Box` is
written to match. Measured on the 640x360 fixture, both depths, every plane:

- bit-exact on u16 at every ratio whose tap counts are all powers of two and at
  the fractional ratios that pair only one or two source pixels (2, 4, 8, 1.5,
  1.25, 1.75, 1.1, 4/3 all measure 0);
- ≤ 2 codes of 65535 on u16 where resize2's own float32 `1/count` is inexact,
  i.e. the tap counts 3, 5, 6, 7 (ratios 3, 2.5, 3.5, 5, 6);
- ≤ 2.4e-7 of the peak on f32 everywhere, and exactly 0 at most ratios.

| 1080p GRAY16 source → half size, 1000 f, medians of 3 | 1080p→540p | 4K→1080p |
|---|---:|---:|
| resize2 (CPU), CPU frames both sides | 5522 | 1471 |
| **vsfeel.Box, device frames both sides** | **8143** | **2099** |
| vsfeel.Box, CPU frames both sides | 3408 | 874 |
| control: the same graph with no filter | 10811 | 2434 |

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
therefore rounds half away from zero, which is what resize2's integer output
does rather than the `roundEven` the other filters use.

**The tap window is a host-computed table, not a formula.** resize2's box is not
an area average: it averages the source pixels whose *centres* fall inside the
output pixel's box, so at a fractional ratio the tap count alternates and some
output pixels decimate outright (120→80 is 1,2,1,2…). `box_taps()` computes
`(first tap, count)` per output position in `double` and the shader runs an exact
integer loop over it, so no float32 tie can move a tap. Two tables cover the whole
frame: chroma is the same lattice shifted, so it reads a prefix of the luma table.

Smaller things worth knowing: `skip_vert` drops the first dispatch when an axis is
unscaled; both strides are read off scratch frames at creation, because a GPU
frame's stride is the CPU frame's; the two tables are concatenated into one host
visible buffer like GaussBlur's weights.

### Tests
`tests/test_box.py` — reference comparison per ratio × depth and per color family
at both depths, chroma geometry, residency, identity, determinism, frame
properties, validation. `tests/test_vsaa.py` covers the chain that uses it.

## Performance

The kernel is two dispatches over `SRC_H` and `SRC_W` outputs, one load per tap,
no shared memory. Subtracting the no-filter control above, Box's own work is
**~30 µs/frame at 1080p→540p and ~65 µs at 4K→1080p**, against ~175 µs and
~675 µs for the CPU resizer.

Everything else in the standalone row is the round trip: at 1080p a u16 upload or
download is ~100 µs, so CPU frames on both sides cost ~200 µs more than the
resize itself. With device frames on both sides there are no transfers at all,
and that is the entire win — 1.5x the CPU resizer at 1080p and 1.4x at 4K, and
+13% on the `feel_aa` chain that previously paid for a 4K round trip.

## Historical

- **2026-10-12 — the Merge composition.** The first version reproduced resize2's
  rounded 2x2 mean from shipped filters only: `std.Merge(0.5)` of a plane and a
  one-pixel-shifted `Point` copy, bit-exact on u16 and within 1e-6 on f32, with
  no new filter to maintain. It lost on cost — 5–6 full-size passes per
  downscale, twice per `feel_aa` frame, leaving that chain at 135.0 fps against
  143.0 for the native kernel.
- **Fractional ratios were going to be refused.** The first design supported only
  integer tap counts, on the theory that a shader-side `ceil(i*s - 0.5)` at a
  fractional ratio is a float32 tie waiting to move a tap. Computing the window
  in `double` on the host removes that risk, so the general case came back for
  free.
- **A `std.Convolution` composition was rejected**: the core refuses even kernels
  ("matrix must contain exactly 9, 25, 49…"), and this build's `std.Expr` has no
  relative pixel access.

## Open work

- `width == src.width` and `height == src.height` still runs both dispatches.
  A plane-unchanged path could copy instead; it is not on any current route.
- The workgroup shape (16x8) is inherited, not swept. With one load per tap and
  no LDS the shape looks unlikely to be the lever, but that is unmeasured.
- Do not re-introduce the Merge composition "because it needs no new filter": it
  is bit-exact and ~8 fps slower on the chain that uses this one.

## Debug env vars

None. The filter has no probe, opt-out or tunable.
