# EEDI3 — notes

*Accuracy landscape, vcheck contract, and the round-by-round performance record.*

## Status

Benchmark defaults (2000 f, ns=8, real based_aa mask, 2x2160p GRAY16, field=3,
mdis=20, vcheck=2): EEDI3 **~628** (u16, parallel vcheck default), EEDI3H
**~378**, EEDI3AA **~160** fps — 2.93x vszipcl u16 / 1.71x fp32.

| depth | vsfeel | vszipcl | speedup |
|---|---|---|---|
| u16 | 628 (622/628) | 214 (213/216) | 2.94x |
| fp32 | 318 (316/319) | 186 (184/191) | 1.71x |

- **The graded `mclip` was 100% zero until round 14**, making DP/backtrack,
  vcheck and vcopy dead code. Pre-round-14 verdicts describe that path and are
  marked *(stale)* below.
- Round 14's cost model was superseded by round 20's: row kernel **~17%** of the
  vertical frame (~25% of EEDI3AA), vcheck ~14%. Round 27 parallelised the
  vcheck (level 6 default), so that share is now spread over 1080 workgroups.
- `src/eedi3.comp`: MDIS 20 → TPITCH 41, CENTER 20, BT_TILE 32, SGSIZE 32, K 2,
  RING_CAP 7. `VCHECK_LDS` defaults to 0 (global-read form, round 21); the
  parallel vcheck (`VCHECK_PARA=6`) is the default since round 27.

## References (five backends, same GPU)

| ref | type | funcs | mclip | io convention |
|---|---|---|---|---|
| `eedi3m` (ORIGINAL) | C++ AVX2 | EEDI3 | per-plane same-fmt→8bit Point | native int scale (params ×1<<(bits-8)); f32 beta/gamma/vthresh ÷255; alpha unscaled; cost3 → alpha÷3; **f64 DP** |
| `eedi3vk2` | Vulkan GLSL | EEDI3 | same as clip (planes[]) | native int scale; f32 `precise` DP |
| `vszip` | CPU Zig SIMD | EEDI3, EEDI3H | single Gray→Gray8, all planes | native int scale; alpha÷3 beta÷255 gamma÷255; NO per-column clamp |
| `vszipcl` | OpenCL | EEDI3, EEDI3H | **none** | **u16 normalized [0,1]** (÷65535); f32 identity; mirror-pad full mdis |
| `vszipcu` | CUDA/HIP | EEDI3, EEDI3H | none | u16 normalized [0,1]; same structure as vszipcl |

## Accuracy landscape and target

jpbd frame 100, 1920x1080, field=1, interp rows only (kept rows identical
everywhere):

| comparison | f32 max abs | u8 max LSB |
|---|---|---|
| vszipcl vs vszip | 1.2e-7 (ulp) | — |
| eedi3m vs vszipcl | 0.067 on 1e-4 of px | 17 on 3.1% |
| eedi3vk2 vs vszipcl | 0.012 | — |
| eedi3m vs eedi3vk2 | 0.067 on 1e-4 px | 17 on 0.003% |
| eedi3m vs vszip | — | 17 on 3.3% |

**Two semantic families.** **A = eedi3m + eedi3vk2:** clamp
`umax=min(x,W-1-x,mdis)`; relax `v ∈ [max(-umax2,u-1), min(umax2,u+1)]`,
`umax2=min(x-1,W-x,mdis)`; MARGIN_H=12; `pbackt` **absolute**; guard
`x>=|3dir| && x<=W-1-|3dir|` else avg2; tie-break ascending v, strict `<`.
**B = vszip + vszipcl:** full direction set per column via mirror pad
~3·mdis+nrad+8; `pbackt` **deltas**; center-pref tie-break. vszipcl's [0,1] u16
is a **third numeric domain** (thousands of LSB vs A).

**Target.** Ground truth = **eedi3m** (user); vk2 shows A + f32 DP ≈ eedi3m.
vsfeel = **family-A semantics, f32 DP**: u16 native int scale, int32 SAD costs,
f32 combine. Bar from round 2: **bit-exact vs eedi3vk2**; eedi3m = loose oracle.

## vCheck exact semantics (family A)

Runs once per plane **after** all interp rows exist (masked rows = cubic). Pad
rows `y=MARGIN_V+field .. height-MARGIN_V` step 2 → dst row `dy=y-MARGIN_V`
steps 2; the pad-row write gate ≡ `dy ∈ [2, dstH-3]`.

Taps at column x for interp row dy: `dst2p` = dst dy-2 (**previous interp row,
already vchecked**); `dst1p/dst1n` = dst dy∓1 (**kept source rows, never
vchecked**); `dst3p/dst3n` = **source** kept rows dy∓3 (from pad); `dst2n` = dst
dy+2 (next interp row, **still plain**); `dmap[x±width]` = neighbour interp rows'
dmap.

`cint = sclip ? scpp[x] : cubic(dst1p,dst1n,dst3p,dst3n)`; u16 cubic
`(9*(dst1p+dst1n)-(dst3p+dst3n)+8)/16` clamped [0,peak], floats unclamped.
`tline[x]=cint` directly if `dirc==0`, `max(dirc*dirt,dirc*dirb)<0`,
`dirt==dirb==0`, or `x±|dirc|` leaves `[0,width)`. Else, with
`it=(dst2p[x+dirc]+dstp[x-dirc]+1)/2`, `ib=(dstp[x+dirc]+dst2n[x-dirc]+1)/2`,
`vt=|dst2p[x+dirc]-dst1p[x+dirc]|+|dstp[x+dirc]-dst1p[x+dirc]|`,
`vb=|dst2n[x-dirc]-dst1n[x-dirc]|+|dstp[x-dirc]-dst1n[x-dirc]|`,
`vc=|dstp[x]-dst1p[x]|+|dstp[x]-dst1n[x]|`, `d0=|it-dst1p[x]|`,
`d1=|ib-dst1n[x]|`, `d2=|vt-vc|`, `d3=|vb-vc|`; mdiff0/1 combine per mode (mode 2
int `(d+d+1)/2`, else float /2); `a=min(max(mdiff0*rcpVthresh0,
mdiff1*rcpVthresh1, max((vthresh2-|dirc|)*rcpVthresh2, 0)), 1)`;
`tline=(1-a)*dstp[x]+a*cint`; u16 store **truncates toward zero**, float
unclamped; the row is then memcpy'd to dst.

**Structural consequence (all the implementation may rely on):** the row
dependency is **only** dy-2 (finalized) and dy+2 (still plain) → two separate
passes, or one pass holding the previous row's final tline in shared.

## Constraints vsfeel implements

- `eedi3m` validation: field 0..3; !dh ⇒ processed planes even; dh ⇒ field≤1;
  alpha+beta≤1; nrad 0..3; mdis 1..40; vcheck 0..3; vthresh*>0 if vcheck>0;
  mclip/sclip same videoinfo+numFrames (sclip only when vcheck>0). `cost3` gates:
  s1 if `(u>=0&&x>=u2)||(u<=0&&x<W+u2)`; s2 if `(u<=0&&x>=-u2)||(u>=0&&x<W-u2)`;
  `s1=s2=s0` fallbacks.
- `eedi3m` DP/backtrack: `z=(double)ppT[mdis+v] + (double)(gamma*|u-v|)` capped
  `FLT_MAX*0.9` on the f32 store; `fpath[x]=pbackt[tpitch*x+mdis+fpath[x+1]]`
  (column x's pbackt written during column x+1's DP); `copyMask` = per interp line
  last-scan dilation, `minmdis=min(W,mdis)`. `eedi3vk2`: 2-subgroup WG per row, no
  per-column WG barrier, rolling vs full recompute, bt tile 32 (16 when
  hp&&!mclip), quantized tline ping-pong in shared, half-pel in ±2mdis half-pel
  units. vsfeel: MARGIN_H=12, MARGIN_V=4, BT_TILE=32, `RING_CAP=7` (=2*NRAD+1 for
  NRAD≤3) — `src/eedi3.cpp:71`, `src/eedi3.comp:92`.

## Historical — pre-round-14 (the benchmark's mask was broken)

**Every perf conclusion measured before round 14 is void**, not just suspect: the
benchmark passed an already-scaled threshold to `Morpho.binarize_mask`, so `mclip`
was **100% zero** and every row took the fully-masked early-out. "The row kernel
is at its floor", "the host is the wall", the stream/queue knees and all throughput
totals from these rounds describe a path the shipped filter never takes. The
*accuracy* work, *correctness* fixes and *mechanisms* below are unaffected (they
are mask-independent or bit-exactness proofs) and are what survives.

### What survived

- **u16 bit-exact vs eedi3vk2** from round 2 onward, on every shared config
  (field 0/1/2/3, dh, mdis 3..40, nrad 0..3, vcheck 0..3, custom
  alpha/beta/gamma/vthresh, sclip, mclip Gray8/16/32). f32 ~1 ulp (7.45e-9);
  gamma=5 + vcheck ≤5.45e-3 (tol 5e-3). Residual flips vs eedi3m (13..1500 px of
  115200, max ~3276 LSB) match eedi3m-vs-vk2's own count → eedi3m's f64 ordering.
- **Two families, two numeric domains** (front matter) — established here.
- **Correctness fixes** (round 2): the dh/field>1 segfault (`newVideoFrame2` got a
  null plane array while vscore derefs it; dst sized at source height); and the
  nrad≥2 u16 divergence — the int path needs `ip=floor((t1+t2+1)*0.5)` and integer
  `v`, not the f32 midpoint, or 0.5 on odd sums flips near-tie argmins. The `s2`
  gate keeps vk2's corrected `x<width-u2`, not eedi3m's `x<width+u2`.
- **Round 3 — dropped `ucubic`/`cost3`.** No other backend exposes them; the GPU
  family hardcodes `cost3=true` (alpha/=3, cost=s0+s1+s2) and `ucubic=true`.
- **Round 4 — sclip output semantics.** Under field>1 sclip describes the OUTPUT
  (2N frames; 2x height under dh), which is what `based_aa` supplies via
  `Interleave([s,s])`; validating against the pre-doubling vi rejected the clip
  every reference requires. Also: based_aa's real call is `field = tff+2` = 3 even
  on progressive content.
- **Round 5 — memory fix, still the shipped shape.** Reading pad/bmask/sclip from
  cached system-RAM staging per cost/DP access was PCIe-bound. Allocating staging
  as `DEVICE_LOCAL|HOST_VISIBLE|COHERENT` **collapsed to 27.6 fps** — RADV's
  coherent device memory has no write-combining, so the ~16.6 MB/frame CPU pad
  write went uncached (vk2 gets WC via VMA's `HOST_ACCESS_SEQUENTIAL_WRITE`, not
  expressible here). Shipped: cached staging + a device-local `pad_dev` mirror,
  one H2D at the CB head. **The uncached types are why "host-visible VRAM is
  poison" was recorded — it is wrong for NT stores.**
- **Rolling-window costs** (round 5): each lane's s0/s1/s2 are (2·nrad+1)-term
  windows, so a column advance costs one new leading term (~9 loads) instead of
  ~90. The single largest kernel win (re-measured honestly at **+76%** in round
  15, where it had been gated off). Bit-exact for integer pad (sums < 2^24).
- **vcheck must not read the row it is writing**: vszipcl's 1-barrier global-read
  model races in place (a lane writing `dst[r][x]` can be read by another
  computing `dst[r][x±dirc]`) — up to 28 LSB at mdis=20, nondeterministic.
  vszipcl only gets away with it by writing a separate out buffer.
- **Subgroup-register DP rewrite is a wash** (round 6, A/B 219.4 vs 220.3 fps —
  the *comparison* is valid even though the absolute numbers are void): one
  32-lane subgroup/row, DP in registers (`K=ceil(TPITCH/32)`, `rings[RING_CAP]`),
  ±1 window via `subgroupShuffleUp/Down`, no per-column barriers, `pbt` as relative
  deltas, `bmask` packed in shared. Correctness preserved, speed identical —
  which is the point: the masked-column *iteration*, not the DP, is the row cost.
- **Round 6/15 rejected variants:** 64-lane / 2-subgroup / 2-rows-per-WG layout,
  `[[unroll]]` on the K loops, `restrict` on `pbt`, `subgroupcoherent` on `pbt`,
  packed-bit `bmaskSh`.
- **Round 7 mechanisms, all bit-exact and shipped:** native u16 pad (halves CPU
  writes, upload 16.6→8.3 MB, H2D and pad-read bytes; vk2 uploads float pad);
  `copy_stream_out`/`_read` NT copies (need `row_bytes%32==0` — `movntdqa` faults
  unaligned); bit-packed bmask rows (8x smaller upload, shader burst-copies 120
  words to shared — **bug: `bmask_base` left a byte offset, 4x too far, caught by
  5 mclip tests**).
- **Span-skip** (round 7, shader-only, bit-exact): lane 0 scans packed words for
  the first set bit into shared `rowXmin`; a fully-masked row takes a parallel
  cubic loop; else the DP starts at `max(1,xmin)` with an analytic predecessor
  seed (`pp=0` exact) and the backtrack stops left of xmin. The masks it relies on
  are real (53% of mask rows and 64% of interp rows are fully masked in the AA
  workload; non-empty rows average a 22% span) — the *measurement* of it was
  degenerate, not the mechanism.
- **Round 8 GPU-kernel outcomes:** the GPU pad kernel is **KEPT** (CPU gathers
  tight kept rows only; the kernel expands eedi3m's mirrors in VRAM); the GPU
  bmask kernel is **REVERTED** (+7.3 MB H2D + 2 launches to free CPU work that was
  already hidden); `ENTRY_VCOPY` + empty-row flag is **KEPT** (drops ~690 of 1080
  barriers — and proved barriers cost ~40 ns, not ~500 ns, so vcheck is
  *traffic*-bound, not barrier-bound).
- **Round 10, mask-independent lever — `build_bmask_row` was 6 ms/frame.** The
  scalar last-propagation dilation is a serial chain, and dilations compose, so it
  collapses to O(log mdis) whole-array 64-bit shift-OR passes
  (`_mm256_cmpeq_epi8`+`movemask`; `B=b<<mdis`; `acc |= acc >> s`): **6.11 → 0.28
  ms/frame (22x)**. Gotchas: the accumulator must span `width+mdis` bits or the
  right edge silently loses coverage; clear bits past `width`; the shift form
  equals the scalar only when `width >= 2*mdis`.
- **`VSFEEL_EEDI3_COPY`** (bit0 upload gathers, bit1 blit, bit2 blit load flavor):
  **m3 (NT load+store on both) wins in situ** at ns=8 (400-417 fps) and ns=4
  (330-332). The isolated microbenchmark says the opposite; it does not transfer.
- **ReBAR direct upload is the shipped path.** Rounds 8/10 recorded "host-visible
  VRAM is poison" from a 178 fps direct-bind collapse — wrong, because that used
  ordinary cached stores into uncached memory. NT stores bypass the cache: a
  `DEVICE_LOCAL|HOST_VISIBLE|COHERENT` per-resource `up_dev`, CPU-written by the NT
  path and read directly, deletes the H2D and its barrier (`_mm_sfence()` before
  submit; `VSFEEL_EEDI3_NOREBAR=1` restores the old path). Honest-path value is
  round 14's **+14%**, not round 10's.
- **Two silent-corruption bugs found while porting it** (keep): the descriptor pool
  had one set per resource, but the ReBAR path needs two (pad kernel b0 = raw
  upload, row kernel b0 = built pad); and the pad set's b8 (built-pad *output*)
  must stay `pad_dev`, or the pad kernel overwrites its own input (isolated pixels
  off by 65535/0).
- **Round 11 — the in-graph mask conversion was a whole extra pass.** Native
  16-bit masks read directly: zimg's full-range 16→8 is monotonic and
  `round(v*255/65535)`, so `converted != 0` is exactly `v >= 129` (verified over
  all 65536 values, 0 mismatches), packed via `bmask_bits16` (xor 0x8000 → signed
  `cmpgt` → `movemask`). **`cmpgt` is strict: the constant is `128 - 32768`, not
  `-32639`** — the off-by-one silently tests `v >= 130`, and only a
  near-threshold input exposes it. Float and 10/12/14-bit masks keep the reference
  conversion.
- **Round 12 — the harness memory bug.** A bimodal fp32 result ("one plugin
  swings, the other flat") was `make_aa_vpy`'s 48 GB `max_cache_size` plus the
  benchmark's own decoded-frame list being **additive**, past 62 GB of RAM. Lower
  the fp32 Python cap, **never** `max_cache_size` (that re-runs the decode chain
  and collapses every plugin 3-4x).
- **Round 13 — direct-to-frame via `VK_EXT_external_memory_host`: built,
  bit-exact, LOSS, default OFF** (`VSFEEL_EEDI3_DSTHOST`). Only the import must be
  page-aligned (4096), not the binding (64). Reusable findings: **never use a
  multi-region `vkCmdCopyBuffer` for strided row copies** (1080 regions of 7680 B:
  1 region 710 fps vs 1080 regions 326), and per-frame host-pointer imports are a
  **GPU** cost on RADV (VM map/unmap + TLB sync per 16.6 MB BO), so zero-copy
  needs imports stable across frames.
- **Round 9 — EEDI3H was first `Transpose → EEDI3 → Transpose`** (no new kernels),
  with a 25-test transpose oracle (bit-exact u16 and f32). Superseded by the native
  implementation in round 18; the oracle and the vsaa `supports_mclip`/`supports_h`
  integration survive. Gotcha: after `mapConsumeNode` into an args map, forget the
  pointer — `freeMap` releases it.
- `tests/test_dfttest.py` alone takes ~127 s (61 tests), over the harness's 120 s
  tool cap — never hung, just slow. Run the full suite in the background.

### Void perf record (do not quote)

Rounds 2–13's totals, standings, and stream/queue knees were all measured on the
zero-mask path: 74.6 → 107 → 155-167 → 363 → 398 → 604 @8s. So were round 7's
"streams plateau at 8", round 10's "host is the wall / the DP is −1% / the row
kernel is at its floor", round 10's +15%/+6% ladder, round 11's ladder, and the
ReBAR +29%. Round 14 re-measured the cost model on a real mask and round 15
overturned the row-kernel verdict; round 20's model (row kernel **~17%** of the
vertical frame, ~25% of EEDI3AA, vcheck ~14%) is the one to use. Stream/queue knees
were re-swept honestly in rounds 14/21 — see those rounds, not these.

## Round 14 — the benchmark's mask was a no-op; first honest cost model

- Bug: `benchmark.py::make_aa_vpy` passed an already-16-bit-scaled threshold to
  `Morpho.binarize_mask`, which re-scales from the 32-bit range → 65535 → **mask
  100% zero** (real `vsaa`: `scale_mask(60, 8, 32)` = 15420). So `xmin >= WIDTH`
  every row and the pipeline was dead code — the origin of the pre-14 "deleting the
  DP changed fps by −1%". Measured (700 f, ns=8): all-zero 593 vs vszipcl 203, real
  mask 274 / 202.
- Honest ladder (900-frame runs, ns=8): **vcheck+vcopy 18-27%**, pad ~8%, raw
  gather ~8%, sclip gather ~7%, blit 4-5%, ReBAR upload **+14%** (kept).
- LDS ping-pong vcheck RE-TESTED (round 6's "flat-to-worse" was degenerate):
  interleaved 900-frame pairs global 294/304/289/293 (mean 295) vs LDS
  302/304/304/305 (mean **304**, +3%, tighter) → default then (reversed in 21).
  Bit-exact vs eedi3vk2 (50/50) and vs the global path over 20 real 4K frames.
- Pad parity skip **+0.8%** (4/4 pairs): every read pad row has parity
  `(field+1)&1`; `VSFEEL_EEDI3_PADPAR=0` restores the full build. Queue cap 3 →
  256-269 fps, 4/6/8 → 294-322 (tie); default stays 8.
- Structured-mclip nondeterminism found here (fixed 16.2): `field=3, vcheck=0`,
  `num_streams=1` and `NOREBAR=1` all nondeterministic; all-zero, all-white and
  no-mclip deterministic. `subgroupMemoryBarrierBuffer()` before the `pbt`→
  `tileSh` barrier changed nothing (883 K vs 1.06 M differing B).

## Round 15 — row kernel is the dominant cost (overturns round 10)

- Baseline (2000 f, ns=8, real mask): vc2 264.1, vc0 308.5, vszipcl 181.5,
  eedi3vk2 125.2. `PROBE=2` 564.3 / 693.6 — but it leaves `dmap` stale and so
  overstates the row kernel (20.3).
- 15.C **walk reads `pbt` directly instead of staging the full TPITCH through LDS:
  +9-13%** (vc0 326→368, vc2 295→329). The serial lane-0 walk consumes one byte
  per column, so staging pulled ~41x the needed data through the store→load
  dependency. KEPT.
- 15.D **rolling window under mclip: +76%, the round's big win** (vc2
  290.3/290.5 → 513.5/512.7, interleaved 2000 f). Round 6 gated it because its
  "no gain" was measured with zero unmasked columns; ungated, unmasked columns cost
  one 9-load seed + 9-load pushes instead of ~51 loads per direction. Bit-exact for
  integer pad (sums < 2^24). KEPT.
- 15.B two-dispatch DP/backtrack split **NEUTRAL** (vc0 322 vs 326, vc2 295 vs
  295): the phase boundary costs exactly what removing the same-WG store→load
  hazard saved. Do not retry without a different overlap story. The "row kernel
  hidden" and "same-WG coupling costs 2.4x" readings from 15/15.A are **void** —
  probe-harness bug, see round 20.
- 15.G the u16 (+77%) vs fp32 (+21%) asymmetry is Amdahl: fp32's no-row-kernel
  floor is 339 fps vs the real 340, i.e. **the fp32 row kernel is entirely
  hidden**. Its floor is 2.1x heavier because every byte doubles *and* the GRAY32
  mask triggered the reference `SetFrameProps(_Range=1) → resize.Point → Gray8`
  node (fixed 16.1).

## Round 16 — native float mask; the structured-mclip bug was a stale `dst`

- 16.1 **Float (GrayS) masks folded natively** (`mclip_native32`); based_aa passes
  the mask in the clip's format. The reference conversion's nonzero boundary is
  **`v > fl(0.5/255)`** — strict, since `>=` is off by one ULP (`0x3b008081` →
  byte 0, `0x3b008082` → byte 1); verified over 63 cases with modal outputs over 5
  runs. One deliberate divergence: above ~8.42e6 the reference conversion overflows
  int32 and emits byte 0, while the native path treats those as nonzero, matching
  eedi3vk2. **+2% fp32** (302.8→308.3, 301.4→307.4).
- 16.2 **Structured-mclip nondeterminism = a `break` that skipped the write.** The
  backtrack tile loop `break`ed for tiles entirely left of `xmin`, correctly
  skipping the *walk* but also the **cubic `dst` write** — every column below
  `xmin` kept uninitialized/recycled memory. Hence the structured-mask trigger
  (needs a whole 32-column tile skipped), all-zero (`xmin = WIDTH`), all-white
  (`xmin = 0`) and the noise-clip mask (`_right_half_mask` is white on the LEFT)
  being deterministic, and `num_streams=1`/`NOREBAR` reproducing it. Fix: write the
  vertical cubic for such tiles with the whole workgroup and `continue`.
  Perf-neutral (interleaved 510.6/511.5 fixed vs 515.2/518.5 unfixed).
- Agreement with eedi3vk2 over 4 real 4K frames: 22758 (r14) → 5902 (r15) → **3**
  differing px, max 1 LSB; real-chain determinism 3-5 distinct outputs in 8 runs →
  **8/8 identical**. `test_eedi3_mclip_long_mask_off_prefix` (black LEFT half) is
  mutation-verified. **Lesson: a skip justified by what a region *reads* must be
  checked for what it *writes*.**
- 16.3 medians of 3 x 2000 f: u16 **518** (512/518/520) vs vszipcl 200 = 2.59x;
  fp32 **306** vs 202 = 1.52x. 81 tests pass.

## Round 17 — EEDI3H's four `std.Transpose` passes (analysis only)

Real based_aa chain, 400 f, ns=8: EEDI3 488.6, EEDI3H (four transposes) **226.6**,
transposes only 424.0, chain floor 1881 fps.

- `std.Transpose` ≈ **0.91 ms/pass** at 2x2160p u16 = 16.6 MB read + 16.6 MB
  write, **~36 GB/s** (CPU memory path); the four account for essentially all of
  EEDI3H's 2.16x penalty. Ceiling if free: **~420-470 fps**.
- Two traps: unused `std.Transpose` nodes are **pruned** (a 4-node chain measured
  identical to a 2-node one), and dropping `sclip` as a proxy for fewer transposes
  is invalid — it flips `HAS_SCLIP == 0`, making the row kernel compute `cint`
  everywhere (258 vs 489 fps).
- Options ranked: (A) fold the transpose into passes that already move the data
  (pad via a shared-memory tile, output into the download/blit, mclip transposed as
  **packed bits**, sclip read with swapped addressing); (B) the GPU (~0.11 ms/pass
  but still 4 passes); (C) only the two clip transposes (~+30%).
- `transpose_plane()` (`src/eedi3.cpp:112`) was dead AND wrong (overlapping
  destination ranges); now a verified 16x16 SSE2 byte transpose, live in the EEDI3H
  mask path.

## Round 18 — EEDI3H implemented natively (transposed staging + 2 GPU passes)

Option A: the same EEDI3 pipeline on the transposed plane, so the
row/vcheck/vcopy/pad kernels are untouched and the 25-test bit-exact transpose
oracle still passes.

- `Eedi3HCreate` is a thin wrapper over the shared create path with
  `d->horiz = true`, no `std.Transpose` nodes. Kernel dims are the source dims
  swapped; `dh` doubles the transposed height (= output *width*), so `out_w`/`out_h`
  are tracked separately. Host gather reads source ROWS and writes the needed
  COLUMN parity compactly (`K[y][k] = src[y][first + step*k]`, step 2, or 1 under
  `dh`) via SIMD deinterleave with NT stores.
- **`ENTRY_XPOSE`** (16x16 tiled transpose with an LDS stage, twice per plane:
  clip → R', sclip → B') and **`ENTRY_COMPOSE`**
  (`O[y][2k+p] = (p == field) ? vout[k][y] : R'[k][y]`, writing the whole plane
  into staging so the CPU blit is a plain row copy) — end of `src/eedi3.comp`,
  CMake list `VK_EEDI3_UP_ENTRIES`. The pad builder reads binding 9 (transposed
  raw); vertically binding 9 aliases binding 0, so no second pad variant.
- Measured (1000 f, ns=8, interleaved): EEDI3 501.9, old EEDI3H 233.2, **new
  EEDI3H 360.9 = 1.55x**; penalty 2.29 → 0.76 ms. Holds across configs (mdis 3
  1.68x, mdis 20 1.56x, mdis 40 1.24x, vcheck 0 1.45x). Stream knee still 8; queue
  cap still does not help. 449 tests pass.
- Findings that cost time: ordinary stores into uncached ReBAR VRAM are
  catastrophic and the packed dilation bits were landing there — moving them to the
  cached staging mirror took vertical 474.9 → 492.2 fps and cut the horizontal
  dilate from ~2.9 ms to ~0.34 ms; **vout must be device-local** (staging 328 →
  360 fps); the compose output needed its own binding (b10) or moving vout dragged
  the output to VRAM; the mask gather must key off the **mask's** depth, not the
  clip's; the direct-to-frame import re-measured far worse than round 13 (−59%:
  vertical 505.3 → 205.3 fps).
- Left undone: the compose's 16.6 MB GPU→staging write is the largest remaining
  item and "write interp-only + CPU merge" is a wash; the mask path's two passes
  could fuse to save one 4.15 MB write+read.

## Round 19 — EEDI3H host-path tuning (+15.2%); the pivot to a fused AA call

Order-reversed 6 reps x 1000 f: horiz 384.9 base, both opts off 334.2; vertical
flat as control.

- 19.1 **Fused mask path (`gather_mask_bitmat`, +6.6%)**: threshold 16 mask
  elements into 16 predicate bytes, transpose the 16x16 tile **in registers**,
  movemask straight into the transposed bit row — ~17 MB vs the old ~29 MB (byte
  matrix → `transpose_plane` → pack). `VSFEEL_EEDI3_MASKFUSE=0` A/Bs it.
- 19.1 traps: **loop order matters more than byte count** (k-outermost keeps 64 row
  streams open and re-reads the whole mask frame per block, 2x slower than
  y-outermost / 4x16-row groups), and it is the READ not the ALU (16.6 MB ≈ 1.6 ms
  at ns=1, byte transpose ~0.3 ms — inherent).
- 19.2 **Fused kept+interp gather (`gather_columns_pair`, +5.3%)**: when `sclip` is
  the clip (based_aa's `Interleave([clip, clip])` hands out the identical plane
  pointer) the two parities are halves of one 64-byte chunk, so one pass produces
  both and a full-frame read disappears. `VSFEEL_EEDI3_PAIR=0` restores the
  two-pass form; falls back on `dh`, unequal NT flags, or unaligned rows.
- 19.3 **Aligned parity selects** (`deint_row_*` / `nz_bytes_*`): loading from
  element 0 and selecting parity in the deinterleave keeps 32-byte loads aligned
  (the old `first` offset straddled two lines when `first == 1`) and makes the last
  16-column block exactly in bounds. Vertical unchanged, horizontal equal-to-better.
- Dead ends: **CPU merge of the kept columns** (`compose_tight`, −4.8% over 8
  order-reversed reps — the extra 16.6 MB source re-read costs more than the 8.3 MB
  PCIe + 8.3 MB VRAM saved); k-outermost mask tiling (2x slower); **ablations that
  change the data are not ablations** (`NOMASKX` showed +33% by keeping stale bits,
  which changes which columns are masked).
- 19.5 **Why EEDI3H cannot be closed further**: mask read 16.6 MB vs 8.3 (a
  transposed plane needs all `height` rows) and compose+blit 49.8 MB vs 33.2 (the
  output must be assembled from transposed interp values *and* kept columns; every
  rearrangement — CPU merge, spread stores, SDMA D2H, `vout` in dev_buf, host
  import — measures worse or neutral in situ). Both are properties of "transpose
  the data, reuse the vertical kernels"; the way out is to fuse the whole based_aa
  chain — `notes/EEDI3AA.md` (chain 98.9 fps vs 509.6/374.0 for its two constituent
  calls, projected 1.6–1.8x). This file keeps only the EEDI3-side facts.
- Method: with a fixed variant order an **inert knob moved 5%** — session drift
  exceeds the effects. `tmp/abl.py` reverses order on odd reps; grade only those.

## Round 20 — pbt-packing / walk-chain levers are DEAD; probe harness was broken

- 20.0 **Harness bug**: `ENTRY_ROW`'s guard was `#if PROBE >= 3` instead of `== 3`,
  so levels 4-7 returned immediately — a **1656-byte empty kernel** vs 43720 B for
  PROBE=0. All round-15 `PROBE=4/5/6` numbers are void. Fixed; the shipping PROBE=0
  binary is byte-identical (md5 `3f53fffe2b81…`). New levels: 7 = real walk + an
  extra discarded walk over the previous row's pbt (f-dependent), 8 = same with an
  f-independent address, 9 = DP with every column stored to the same 41 bytes,
  10 = 8 with an explicit 4x unroll, 12 = faithful no-row-kernel (dmap filled with
  the real pattern).
- 20.1 Current vertical cost model (3 interleaved reps, within-arm spread <0.5%):

  | build | fps | frame | attributable |
  |---|---|---|---|
  | real (PROBE=0) | 495.3 | 2.019 ms | |
  | PROBE=2 no row tail | 594.4 | 1.682 ms | whole row kernel 0.337 ms = 16.7% |
  | PROBE=5 DP+stores only | 562.4 | 1.778 ms | walk+interp 0.241 ms = 11.9% |
  | PROBE=9 DP, ~no store traffic | 514.0 | 1.946 ms | *slower than real* |

  Walk isolation: real 492.5; **PROBE=6 (previous row's pbt) 532.6 vs 530.6 → the
  same-WG store→load dependency is worth 0**; PROBE=7 453.9, PROBE=8 451.4,
  PROBE=10 457.0 (identical within 0.5%).
- **Breaking the walk's serial load chain is worth exactly zero** — issue/sector-
  throughput bound, not latency bound; 3840 dependent L1 loads per row are already
  overlapped. **2-bit pbt packing: ceiling ~5%, and the store is not bandwidth
  bound.** DP+store is 4.8% of the vertical frame (0.091 ms for 170 MB = 1.9 TB/s,
  L2/port). PROBE=9 is direct proof bytes are not the cost: 1/40th the traffic is
  *slower* (0.264 vs 0.096 ms) because repeated writes to one dirty line serialise.
  Packing adds 4 `subgroupBallot`s + a vector store to an issue-bound kernel.
- 20.3 **EEDI3AA row kernel ~25%, not 35%**: `PROBE=2` returns with `dmap` stale,
  so the vcheck takes its `dirc == 0` fast branch and the row kernel is overstated
  (real 98.19 vs PROBE=2 150.39 = 1.53x; faithful **PROBE=12** 89.76 vs 72.00 =
  1.25x). Use PROBE=12 for any future row-kernel ablation.
- 20.4 **The environment is bimodal**: the same binary and vpy measured EEDI3AA
  97.7 / 90.9 / 79.8 fps and vertical 530/495/451/410 while sclk 2304-2338, mclk
  1249, fclk 2000 MHz stayed pinned, `gpu_busy_percent` 100, junction 86-90 C,
  vspipe CPU <0.3%, MemAvailable flat. Not clocks, thermals, host CPU or harness
  memory. **Effects under ~5% are not measurable here.**
- 20.5 What is left: **lane utilisation** — the walk is a serial chain run by 1 of
  32 lanes while issue-bound. Splitting it into its own dispatch with **one lane
  per row** (1080 independent chains, directions written to the existing `dmap`,
  tile-interpolate left coalesced) is projected **~5-9%** vertical. Not attempted;
  prototype behind the PROBE harness first.

## Round 21 — vcheck verdict REVERSED; EEDI3AA compose fused into VRAM

- 21.1 **The vcheck default flips to the global-read form.** Round 14's LDS
  ping-pong removes one global read per `dirc != 0` pixel, but the LDS walk must
  visit **every** row (to advance the ping-pong) and ends with an extra full-width
  flush per row, while the global form skips fully-masked rows via `ENTRY_VCOPY` —
  and on the AA workload ~2/3 of rows are fully masked. Same-session order-reversed
  medians (1500 f, ns=8, 4 reps): EEDI3 vertical 498.3 → **513.6** (+3.1%), EEDI3H
  horizontal 340.9 → **378.4** (+11.0%), H/V ratio 0.684 → **0.737**. `VCHECK_LDS`
  now defaults to 0 (`src/eedi3.comp:126`); the LDS pipeline is still built and
  selectable with `VSFEEL_EEDI3_VCLDS=1`. Zipcl's H/V ratio is ~0.91; the reversal
  is row-mask-density dependent.
- 21.2 **EEDI3AA horizontal compose merges in VRAM** (+3.0%, bit-exact). It used
  to assemble two full horizontal planes into staging (16.6 MB PCIe write each) and
  have the CPU 50/50-merge them; `NOCOMPOSE` priced the two dispatches at **+14%**
  of the frame (104.6 → 119.4 fps). `ENTRY_COMPOSE` gained a `comp_fuse` push
  constant: sub-pass 0 writes its assembled plane into a device-local `o0` region,
  sub-pass 1 reads it back, assembles O_1, averages exactly like `std::Merge`
  (`(a+b+1)>>1` u16, `0.5a+0.5b` f32) and writes the merged plane to staging; the
  CPU then does a plain row blit. `VSFEEL_EEDI3_AATIGHT=0` restores the old form;
  `vcheck == 0` falls back automatically. Bit-exact on 30 real 4K frames and 133 AA
  tests; 89.7 → 108.8 fps (+21.3%) with both changes.
- 21.3 **What the remaining gap is made of.** `VSFEEL_EEDI3_GBENCH` at ns=1 (horiz,
  mdis=20, vcheck=2): xpose 0.05, pad 0.04, row ~2.7, vcheck ~2.7, **compose
  0.62** ms; vertical pad 0.05, row ~2.9, vcheck ~2.5. At ns=8 the EEDI3H gap
  (0.69 ms, 513.6 vs 378.4) is compose 0.31 + blit 0.28 + ~0.1 host. The compose is
  a 16.6 MB transpose writing **directly to host memory** at 27 GB/s = the PCIe 4.0
  x16 line rate; a VRAM write + SDMA D2H moves the same bytes at the same rate.
  Every "interp half only" variant trades an 8.3 MB PCIe write for an 8.3 MB host
  read (`compose_tight`, −4.8%) or host write — a wash. **Zipcl is not a
  counterexample**: its absolute horizontal overhead is 0.48 ms vs 0.67, but its
  vertical baseline is 2.5x slower so the ratio looks worse.
- 21.4 **Tooling: `VSFEEL_EEDI3_GBENCH`** — one timestamp query pool per resource,
  a `BOTTOM_OF_PIPE` mark at every `record_pass` stage boundary, read back after
  the fence, mean stage deltas printed every 50 frames tagged `single` / `aa-v` /
  `aa-h`; zero-cost when unset. Navi31 `timestampPeriod` is 10 ns/tick and the
  delta must be **multiplied** by it (`ticks * period / 1e6` for ms) — the first
  version divided and printed nonsense. Multi-stream marks include other streams'
  work, so attribute kernels only in the ns=1 trace.
- 21.5 Re-confirmed dead ends: `VSFEEL_EEDI3_COPY=7` (cached blit load instead of
  NT) horizontal 387.4 → 377.1, vertical flat; per-frame host-pointer import for
  the output still catastrophic (vertical 504.2 → 215.7 fps); `QUEUES=8` still wins
  (526.8/394.5 vs 519.9/387.2 at cap 4); `NOPAD`/`NOXPOSE` remain unusable as cost
  ablations because they change the row kernel's input and branch mix (gbench puts
  pad+xpose under 0.1 ms).

## Round 22 — cross-cutting validation hardening

- Flush/invalidate ranges go through the shared `mapped_range` helper (EEDI3's
  duplicated per-plane `[0, upload_total)` entries collapsed to one);
  `VK_EXT_subgroup_size_control` enabled only when advertised (or Vulkan 1.3+);
  extension list enumerated once; `apiVersion` checked with an `"EEDI3 requires
  Vulkan 1.3"` message; `vkCreateComputePipelines` takes `pipeline_cache_lock`;
  `allocate_memory` refuses rather than relaxing a `DEVICE_LOCAL|HOST_VISIBLE`
  request. No behaviour change here, no measurement.
- The per-stream `Eedi3Resource` is created into the pool via
  `FramePool::emplace()`, so a creation error is torn down by `~Eedi3Data` instead
  of leaking it.

## Round 23 — NT-store alignment predicate

- `deint_row_u16/f32` gated their streaming store on the **element index**, not the
  pointer, so on any destination row whose cell length was not a multiple of 32
  bytes (EEDI3H's transposed kept-row cells: 1260 B u16 / 2520 B f32 at a 630-px
  source) *every* body store silently became a cached store into the uncached VRAM
  window. Predicate dropped; `nt` alone selects `_mm256_stream_*`.
- Measured on the AA geometry (630-px luma, EEDI3H field=3, mclip, 800 f, ns=8,
  interleaved old/new): old 799.9/784.1/798.9, new 803.1/824.9/803.4 fps —
  **perf-neutral in practice** (same-order pairs +0.4%/+0.6%; the middle pair's +5%
  is order-confounded). Why: full-width 32-byte AVX stores coalesce in the WC
  buffer, unlike the narrow scalar stores behind the old "27.6 fps" reading;
  `NORAW=1` is worth only +2.6% at this width. Output bit-identical at 630/638/640
  px.

## Round 24 — four latent defects

1. The pipeline dedup array `VkPipeline destroyed[8*3]` had zero headroom
   (EEDI3AA's worst case is exactly 24) → a `std::vector` with `std::find` dedup.
2. `import_plane_host_memory` bound at `addr & (align-1)` without reading the
   imported buffer's own `VkMemoryRequirements`; it now queries and rejects (falling
   back to the CPU blit) on an offset below `mem_req.alignment`, a region below
   `mem_req.size`, or a rejecting `memoryTypeBits`.
3. `field > 1` doubled `numFrames` unconditionally, turning a `-1` unknown-length
   source into `-2`; guarded with `if (numFrames > 0)` in EEDI3's sclip validation
   and output vi and NNEDI3's output vi (verified by a standalone compile of the
   guard, since no installed source reports `-1`).
4. `RAWSTAGE=1` alone was garbage (it needed `NOREBAR=1`; it now selects the DMA
   path itself), and `NOBLIT` was a silent no-op on EEDI3AA → now a create-time
   error. 214 EEDI3/EEDI3H/EEDI3AA + 47 NNEDI3 tests pass. No perf measured.

## Round 25 — probe cleanup, MAXW single-sourcing, dead code

- The GBENCH accumulator indexed slots by `tag_id & 1`, so EEDI3, EEDI3H and
  EEDI3AA's vertical pass all accumulated into slot 0 and "stage N" meant different
  work depending on the graph; and the claim that `EEDI3_TS_CAP = 24` was exactly
  consumed was wrong (measured marks 6/6/11/12 of 24 — none dropped). Fixed with
  four slots and an explicit slot argument; EEDI3H reports `single-h`.
- `MAXW` had two unlinked copies — the host's `MAXW_LDS` claimed CMake passed
  `-DMAXW` but the rule did not, so the LDS-fit check and
  `shared float tlineSh[2][MAXW]` could drift; `EEDI3_MAXW` in CMakeLists is now
  the single source for both, default 4096. Per-frame flush vectors became a
  reusable `Eedi3Resource::mapped_ranges`; dead code and warnings removed. 603
  tests pass; bit-identical to `8e65f1f` across 14 configs.

## Round 26 — probe corruption fixed; warning baseline

- The PROBE 7/8/10 "fake store" did corrupt the ablation: `f2` was kept alive with
  `if (f2 == pc.pbt_base) dmap[rempty_base + r] = 0`, but `pbt_base` is a byte
  offset while `f2` is a sum of byte-valued steps, so when it fired it cleared the
  per-row empty flag the vcheck reads — changing the vcheck's branch mix and the
  output. Replaced by a read-only sink into `rowXmin` (dead after the `xmin`
  snapshot): PROBE=7 row-kernel SPIR-V 43744 → **44432 B** (BITS=16), so the walk
  is still compiled in, and the output is byte-identical to PROBE=0 and stable
  (sha256 `283717477cead12f`). PROBE 11 and 13+ documented as unused gaps.
- Warning baseline: `-Wall -Wextra -Wshadow` gave 191 hits, all but 2 being
  `-Wmissing-field-initializers` on the Vulkan/VS designated-initializer convention;
  that one is silenced for the plugin and the baseline enforced in CMakeLists. Real
  defects fixed: an unconditional DFTTest `create_pipeline` print (its `getenv`
  gate was unused); NLMeans gputrace accumulators computed and discarded; a dead
  `shared` flag; three shadowed locals in nnedi3; a signed/unsigned block-size
  comparison in bilateral. 603 tests pass.

## Round 27 — the vcheck is parallel by default; a Jacobi ladder controls drift

- **The serial row walk is gone from the shipped path.** Row r takes `d2p` from
  the *un-vchecked* dst row r-1, so no vcheck write feeds another row: the pass
  runs one workgroup per interp row (`grid (1, rows, 1)`) and needs no vcopy
  (the parallel walk writes every row itself). Same-session A/B, 5 order-reversed
  2000-frame pairs on the honest path (ns=8, real 2x2160p based_aa chain):
  EEDI3 **497.96 -> 628.68 fps, +26%** (per-pair ratio 1.253-1.285x; the serial
  arm alone swings ~10% between sessions, the parallel arm does not). EEDI3AA,
  1000-frame reps: **108.3 -> 160.2 fps, +48%** at the shipped level.
- **The drift is a blend decision flip, and extra Jacobi steps decay it
  geometrically.** At `a == 1` the output is `cint_s` whatever `d2p` is, so
  feeding the predecessor row's own (one-step-less) parallel value reproduces
  the serial value at those pixels. Level N = N-1 extra steps. Drift vs the
  serial walk, real jpbd frames 300-305, benchmark config:

  | level | EEDI3 fps | u16 max (of 65535) | fp32 max |
  |---|---|---|---|
  | serial | 498.0 | — | — |
  | 1 | 631.3 | 4157 on 0.072% px | 0.088 on 0.108% px |
  | 3 | 635.7 | 580 on 0.018% | 0.021 on 0.037% |
  | 5 | — | 36 on 0.003% | 0.0017 on 0.012% |
  | 6 | 628.7 | 5 on 0.0007% | 2.8e-4 on 0.006% |

  (The fps column mixes two adjacent sessions; the vertical path pays nothing
  for the steps. EEDI3AA runs the vcheck twice, so the steps cost it ~5 points:
  167.6 fps (+55%) at level 1 against 160.2 (+48%).)
- **Default is level 6**, the lowest that is bit-exact on the whole tested
  surface: all 236 EEDI3/EEDI3H/EEDI3AA tests pass unmodified, u16 vs eedi3vk2
  stays 0 and fp32 stays at the serial ulp on the noise clip. Level 3 is the
  first that is not (1 code on three reference cases). `VSFEEL_EEDI3_VPARA=0`
  restores the serial walk as the A/B control; 1..6 pin a level. Test
  `test_eedi3_parallel_vcheck_matches_serial` pins the two arms together.
- The ladder costs no extra dispatches: one shared `vc_blend` and a
  macro-generated driver per d2p source. The LDS form (`VCHECK_LDS`) is
  untouched and still selected by `VSFEEL_EEDI3_VCLDS=1`; the parallel level
  wins when both are set.

## Round 28 — EEDI3 host-pass: three levers measured, none pays

The host-stage probe (`VSFEEL_EEDI3_HBENCH` / `VSFEEL_EEDI3AA_HBENCH`, sampled
at sn=80) was re-run before touching anything. Its stages sum to the reported
total almost exactly, i.e. **EEDI3AA is fully serialized**: per stream at
1080p/ns=1, vGather 3.4 + vWait 8.7 + hGather 2.6 + hWait 6.5 + merge 0.6 =
21.9 ms. Command-buffer recording is 0.009-0.015 ms of that.

1. **Pre-recording the CBs — bounded out, not implemented.** `record` is
   0.01-0.02 ms/frame in every measured arm (EEDI3 vertical 0.014, AA 0.009 and
   0.013), so the entire recording path is <=0.07% of the frame. The earlier
   A3 note (round 19) already said this; the number reproduces.
2. **Fusing the AA horizontal stage's two parities into one pass — bit-exact,
   measured NEUTRAL, reverted.** `gather_columns_pair` (the EEDI3H pair form)
   reads the merged frame once and writes both parities, deleting one of the
   two full-frame column gathers *and* the duplicated mask-bit build.
   A/B at ns=1, same frame: hGather 2.630 -> 2.650 ms, total 21.933 -> 21.881 ms
   (0.2%, below the probe's own resolution). Mechanism: the pair halves 8.3 MB
   (u16) of reads per frame into a staging buffer whose traffic is evidently
   not the binding cost — a 6-pair order-reversed A/B at ns=8 could not resolve
   it either (153.6 vs 156.1 fps median, 7% spread). Reverted: neutral
   complexity is not worth an index-parity parameter.
3. **Not building the vertical `xpose`/`compose` pipelines for AA — premise is
   false.** `EEDI3AA` *does* dispatch `xpose`: its horizontal sub-pass calls
   `record_pass(..., horiz=true, planes=d->aplanes)`, and `d->aa` is set while
   `d->horiz` is not, so gating on `d->horiz` hands `record_pass` a null
   `xpose_pipeline` and RADV faults in `vkCmdBindPipeline`. `create_pipeline`
   with a null `VkShaderModule` still returns a `VkPipeline`, so the failure
   surfaces at dispatch, not creation. AA needs both kernels; nothing to prune.
4. **Kept: the per-frame mask scratch is now thread-local.** The four
   `std::vector<uint64_t>` scratch buffers the gather paths allocated per plane
   per frame (up to 12 allocations/AA frame) become one `bmask_scratch()` buffer
   per worker thread. Every builder fully overwrites the span it uses, so reuse
   is safe; the win is bounded by the ~1 us/alloc it removes, i.e. well under
   the probe's resolution — it is correctness-neutral cleanup, not a speedup.
   `env_flag("...HBENCH")` is also cached in a `static` instead of re-read by
   `getenv` every frame.

Same-session A/B against an unmodified HEAD build of the same shaders
(`--filter eedi3aa`, 1000 frames, ns=8, 6 order-reversed pairs): HEAD median
153.6 fps, this build 156.1 fps (ratio 1.017x) — inside the run-to-run spread,
so treat the change as neutral and read neither number as a regression.

The limiter the probe exposes is the **GPU fence wait** (15.2 ms of the 21.9 ms
per-frame path at ns=1), not the host CPU stages. The one structural lever it
implies that was *not* tried: both AA horizontal sub-passes read only the merged
frame, so they could share a single command buffer and a single fence wait
(saving the ~6.5 ms second wait, ~30% of the ns=1 path) — that needs the
two-parity pre-gather from item 2 to be worth it, which is why item 2 was built
and measured first.

## Open work

- **A2. Row-kernel work — round 20 closed the two proposed levers as dead; the
  kernel is ~17% of the vertical frame (~25% of EEDI3AA).** Done and kept:
  rolling-with-mclip (+76%, 15.D) and the walk reading pbt directly (+9-13%, 15.C).
  Remaining, in order:
  - **Split the walk into its own dispatch with one lane per ROW** (20.5): 1080
    independent chains instead of 1080 sequential ones, directions to the existing
    `dmap`, tile-interpolate left coalesced. PROJECTED ~5-9%, not attempted — a
    real `ENTRY_ROW` + host dispatch restructure.
  - **SGSIZE 64 / K=1** (TPITCH 41 <= 64 at mdis 20): one wavefront per row, one
    direction per lane; the r0/r1/r2 rings collapse to ~21 VGPRs instead of ~42,
    likely better occupancy at this mdis. Not tried (round 6's 2-subgroup x 2-row
    64-lane regression was a different shape).
  - BT_TILE as a spec constant swept {16,32,64}; hoisting duplicated `cubic_float`
    evaluations; fixed-per-invocation floats as spec constants; parallelising the
    serial span scan. All bounded by the ~17% share.
  - The row kernel is now fast enough, and the vcheck is parallel since round
    27, that **the host stages matter again** — re-run the ladder before
    assuming anything.
- **A3. Fuse the vcheck across planes** (one launch, `gl_WorkGroupID.y = plane`, as
  vszipcl does) — YUV-only, cannot move the Gray flagship, real for colour AA. Not
  attempted. Nothing else in the host path pays: `record_command_buffer` is
  0.004-0.012 ms/frame (~0.2%) and `vkQueueSubmit` 0.06-0.07 ms, so prerecording
  per parity, `vkResetCommandBuffer` and single-bind variants cannot pay.
- **A2b/A2c. EEDI3H — done (rounds 18-19); the remaining gap is structural.**
  Native transposed-plane pipeline (1.55x over the composition) + host-path tuning
  (+15.2%); what is left is the 2x mask read amplification and the compose/blit
  round trip, both inherent. Next real lever is fusing the whole based_aa chain —
  `notes/EEDI3AA.md`.
- **A4. Accuracy-for-speed items are void on mechanism**: fp16 pad/cost storage has
  no traffic to remove (pad is native u16, DP costs live in registers); adaptive
  `nrad`/`mdis` trades accuracy for ~0 (`mdis` is VRAM-only for pbt plus a
  row-kernel parameter, `nrad` is purely row-kernel); dropping the sclip upload
  when the caller passes the same clip is undetectable through a node-pointer test
  and saves only the row-copy half of the gather. Any accuracy relaxation still
  requires measuring the drift on the noise clip and updating `tests/test_eedi3.py`
  in the same change.

## Do-not-retry (mechanism, not verdict)

*(stale)* marks a verdict measured on the degenerate pre-round-14 config — kept as
the reason a variant failed, not as a current number.

- **All round-15 `PROBE=4/5/6` numbers are void**: those levels compiled to an
  empty row kernel (1656 B vs 43720 B), which is why "store nothing", "walk another
  row" and "skip the backtrack" all land within 1.4% of each other.
- **Breaking the walk's serial load chain** by lookahead, ±1 candidate windows, or
  a layout making the address f-independent: 453.9 / 451.4 / 457.0 fps — identical
  — and PROBE=6 is a dead tie with real (532.6 vs 530.6). Issue/sector-throughput
  bound, not latency bound.
- **2-bit `pbt` packing**: DP+store is 4.8% of the vertical frame and the 170 MB
  store runs at ~1.9 TB/s (L2/port, not DRAM). PROBE=9 proves bytes are not the
  cost — 1/40th the traffic on one dirty line is *slower* (514.0 vs 562.4 fps).
  Packing also adds 4 `subgroupBallot`s + a vector store to an issue-bound kernel.
- **4x-unrolling the walk with merged `tileF` stores**: bit-exact and 56/56 tests,
  but a wash over 8 order-reversed 2000-frame pairs (median ratio 0.99).
- **Ablating the row kernel with `PROBE=2` or `PROBE=5`**: they return with `dmap`
  stale, so the vcheck takes its `dirc == 0` fast branch and the row kernel is
  overstated by ~9 points on EEDI3AA (35% vs 25%). Use `PROBE=12`.
- **Splitting DP and backtrack into two dispatches** (15.B): neutral (vc0 322 vs
  326, vc2 295 vs 295) — the boundary costs exactly what removing the same-WG
  store→load dependency saved; measured on the pre-15.C/15.D kernel. 20.5's
  proposal splits to *parallelise* the walk, a different thing.
- **Per-frame `VK_EXT_external_memory_host` imports** (`VSFEEL_EEDI3_DSTHOST=1`,
  default off): the import is GPU work on RADV/amdgpu (VM map/unmap + TLB sync per
  16.6 MB BO), and 8 streams saturate the GPU on mapping (`fence_wait` grew to
  12-31 ms at cache=500 while `gpu_busy_percent` read 95-100%). Arms: staged 608 |
  `vout` in dev_buf no import 588 | import the GPU never touches 487 | import+blit
  390; leaking imports is worse (269). Re-test only if RADV's import path gets
  cheaper or VS exposes stable-address frames.
- **Multi-region `vkCmdCopyBuffer` for strided row copies** (~13 us *per region*
  in situ; 1080 regions of 7680 B ≈ 14 ms of frame latency vs ~1 ms for the
  compute blit kernel that replaced it), and **`vout` in dev_buf + SDMA D2H**
  (`VSFEEL_EEDI3_VOUTDEV=1`, re-measured on the honest path, still a loss: 289 vs
  313-323; 327 vs 375 with vcheck off).
- **Lazy `cint`/`sclip` load in vcheck**: void by inspection — `cint_s` is both the
  `dirc == 0` result and the `dirc != 0` blend operand, so every `do_row` pixel
  reads it.
- **`LDS ping-pong tlineSh`** (round 6: "flat-to-worse, the 30 KB of LDS was the
  cost"): **(stale)** — promoted to default in round 14 (+3%), then **REVERSED
  again in round 21** (+11% EEDI3H to go back to the global form), because the LDS
  walk must visit every row while the global form skips fully-masked rows. Both
  sides are real and the reversal is workload-dependent (row-mask density), so
  re-check it if the mask mix changes.
- **Parallel vcheck — SHIPPED (round 27).** Feeding `d2p` from the un-vchecked
  `dst[r-1]` is a real accuracy change (u16 drift up to 4157 codes on 0.07% of
  real-content pixels), but each extra Jacobi step multiplies the exact pixels,
  so level 6 (five steps, free on the vertical path, ~5% on EEDI3AA) is
  bit-exact on the whole tested surface. Do not re-open level 1: it is the
  fastest arm but gives up the u16 bit-exactness invariant.
- **(stale) batch**: a single contiguous blit instead of per-row strided copies
  (525 vs 555); plain `memcpy` instead of the NT load/store pair anywhere in the
  frame path (blit 6.1 → 10.4 ms/frame; vc0 481 → 401); skipping the pbt global
  stores (`PROBE=1`); a 2-subgroup x 2-row 64-lane workgroup for `ENTRY_ROW`
  (regressed in every config; NOT the untried one-subgroup 64-lane / K=1 shape in
  A2); `subgroupcoherent` on pbt (259 → 229); `restrict` on pbt (214 → 201);
  `[[unroll]]` on the K-direction loops (227 → 196); dropping `HOST_CACHED` from
  staging (collapsed to 43 fps — NT loads need WB/WC memory).
- **Rolling window connection costs with mclip**: ~~no gain~~ — **REVERSED in round
  15, +76%, shipped default.** The old verdict was measured with an all-zero mask,
  i.e. zero unmasked columns to roll over.
- **Queue cap** (`VSFEEL_EEDI3_QUEUES`): 3 loses ~20%, 4/6/8 tie; the default
  `min(num_streams, queue_count)` = 8 stays and the knob stays for tuning.
- **Serial per-frame `get_frame` millisecond probes**: misleading at >1 stream. Use
  `VSFEEL_EEDI3_HBENCH` with `VSFEEL_EEDI3_HFRAME`.
- **`core.max_cache_size` in the benchmark harness**: lowering it collapses *every*
  plugin (vsfeel 61 vs 220, vszipcl 39 vs 200) because the timed region starts
  re-running the decode/mask chain. Shrink the harness's own frame cache instead.

## Method rules

- **Grade sweeps at >=900 frames**: a 700-frame queue sweep reported +14% for
  `queues=4`; at 900 it was a tie (clock ramp).
- **Effects under ~5% are not measurable on this box** (round 20's bimodal state —
  not clocks, thermals, host CPU or harness memory). Grade the median per-pair
  ratio over >=5 order-reversed 2000-frame pairs (`tmp/probeab.py`,
  `tmp/probeabaa.py`); treat a single-pair <5% claim as unproven. The vertical arm
  is *not* less noisy than EEDI3AA — it looked stable for an hour, then swung 20%.
- **Never compare a number recorded in a different command.** The same committed
  binary measured 264 fps early in round 15 and 290 later; interleave A/B in one
  session (`tmp/ab.sh`), or use modal outputs over many runs when comparing
  *outputs* (a single run can land on any race outcome).
- **An isolated microbenchmark proposes; only an in-situ same-session A/B decides.**
- **Never use timestamp queries for kernel attribution here** — the env-gated
  ablation ladder answered host-vs-GPU faster and more robustly; an early query-pool
  attempt hung the queue. (`GBENCH` is the one exception, and only for stage
  boundaries at ns=1.)
- **Prove the stage split with the ablation ladder, not theory**: the
  `VSFEEL_EEDI3_*` opt-outs (`NOBLIT`, `NOSCLIP`, `NORAW`, `NOH2D`, `NOVC`,
  `NOPAD`, plus `NOREBAR`) at ns=8, and `HBENCH` + `HFRAME` for the host split.
  Additive costs of similar size mean a shared memory path; one dominant item means
  a loop.
- **An ablation that removes a producer must reproduce what the consumer reads**
  (`PROBE=2/5` leave `dmap` stale → cheaper vcheck → overstated row kernel). Same
  class as round 19's "ablations that change the data are not ablations".
- **Before trusting any `PROBE` number, `stat -c %s build/vk_spv/eedi3_16_row.spv`**:
  round 20 lost a whole round of recorded numbers to a `>=` guard compiling 4-7 to
  an empty kernel.
- **The benchmark's EEDI3 mask follows the clip's depth, like based_aa**; both
  depths now take the same no-conversion path (16.1), with one deliberate
  documented divergence (floats above ~8.42e6).
- **A structured mclip was not reproducible — FIXED (16.2)**, and the cause was not
  a race but a `break` that skipped a write. General rule: **a skip justified by
  what a region *reads* must also be checked for what it *writes*.** Regression test
  `test_eedi3_mclip_long_mask_off_prefix`, mutation-verified.

## Debug env vars (verified against `src/eedi3.cpp`)

- `VSFEEL_EEDI3_NOREBAR` — force staging + H2D DMA instead of host-direct ReBAR.
- `VSFEEL_EEDI3_VOUTDEV` — `=0` forces `vout` to staging (EEDI3H default: device-local).
- `VSFEEL_EEDI3_DSTHOST` — `=1` per-frame host-pointer output import (default off).
- `VSFEEL_EEDI3_AATIGHT` — `=0` restores the pre-round-21 EEDI3AA compose.
- `VSFEEL_EEDI3_VCLDS` — `=1` forces the LDS vcheck ping-pong back (global is default).
- `VSFEEL_EEDI3_VPARA` — vcheck form: 0 = serial row walk (A/B control), 1..6 =
  parallel with that many Jacobi steps. Default 6 (bit-exact on the test surface).
- `VSFEEL_EEDI3_QUEUES` — queue cap override (default `min(num_streams, queue_count)`).
- `VSFEEL_EEDI3_COPY` — 0..7 bit mask: bit0 NT raw gather, bit1 NT kept rows, bit2 NT blit.
- `VSFEEL_EEDI3_MASKFUSE` — `=0` A/Bs the pre-round-19 two-pass mask path.
- `VSFEEL_EEDI3_PAIR` — `=0` disables the fused kept+interp column gather.
- `VSFEEL_EEDI3_PADPAR` — `=0` disables the pad parity skip (full pad build).
- `VSFEEL_EEDI3_BLITCONTIG` — one contiguous copy instead of per-row strided copies.
- `VSFEEL_EEDI3_RAWSTAGE` — raw gather into staging (implies the DMA path).
- `VSFEEL_EEDI3_TRACE` — host phase trace.
- `VSFEEL_EEDI3_HBENCH` / `_HFRAME` — host stage split; `HFRAME=N` samples frame N (`-1` periodic).
- `VSFEEL_EEDI3_GBENCH` — GPU stage-boundary timestamps at ns=1, mean deltas every 50 frames.
- `VSFEEL_EEDI3_PTRTRACE` — sclip plane-pointer trace (temporary diagnostic).
- `VSFEEL_EEDI3_{NOBLIT,NOSCLIP,NORAW,NOH2D,NOVC,NOPAD,NOXFER,NOXPOSE,NOCOMPOSE,NOMASKX}` — ablation opt-outs (`NOBLIT` errors on EEDI3AA).
- `VSFEEL_EEDI3AA_HBENCH` / `_HFRAME` — EEDI3AA host stage split.
- `EEDI3_PROBE` — CMake cache var: ablation level 0/1/2/3/4/5/6/7/8/9/10/12 (11, 13+ unused).
- `EEDI3_MAXW` — CMake cache var: LDS vcheck max width → `-DMAXW` + `-DEEDI3_MAXW_LDS` (default 4096).

Temporary diagnostics flagged for deletion before landing but still in the tree:
`NOXPOSE`, `NOCOMPOSE`, `NOMASKX`, `PTRTRACE`.
