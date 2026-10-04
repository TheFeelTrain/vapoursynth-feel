# EEDI3 — notes

Status: **shipped** — family-A semantics (eedi3m/eedi3vk2). Both depths are
bit-exact against eedi3vk2: the fp32 rolling sums are re-summed in the
reference's k order and the cost/relax/vcheck/cubic float expressions carry its
`precise` qualifiers. `hp` (which eedi3m only registers; eedi3vk2, vszip and
vszipcl implement it) is live and bit-exact too. EEDI3, EEDI3H (native
transposed-plane) and EEDI3AA (fused based_aa chain) share every kernel,
geometry and pipeline. Runs on the **R80 GPU API**: `clip/sclip/mclip:vnode:gpu` in and
`ffGPUOutput` out, one exec pool, and no host upload/download/gather/blit
machinery at all. The parallel vcheck is the default.

Benchmark defaults: 2000 f, real based_aa clip, 2x2160p, field=3, mdis=20,
vcheck=2. `MANGOHUD=0 uv run tools/benchmark.py --filter eedi3 vsfeel vszipcl`.

| clip | vsfeel | vszipcl | speedup |
|---|---|---|---|
| u16 | 472 | 56 | 8.4x |
| u16 (EEDI3H, same call surface) | 471 | 52 | 9.0x |
| u16 hp | 198 | 48 | 2.5x |

(The hp row and the fp32/eedi3vk2 arms are the pre-optimization figures; the u16
rows are `tools/benchmark.py --repeat 3`, 2000 f.)

An order-reversed same-session pair against the pre-optimization build measures
u16 470.4 → 469.6 fps (+0.0%): the vertical path is transfer bound, so its kernel
work does not show up here.

The hp row is a 600 f same-session pair of the same workload (`--eedi3-hp 1`);
there hp costs vsfeel **2.3x** (451 → 198 fps against eedi3vk2's 141 → 80 and
vszipcl's 62 → 48) because the direction set doubles (TPITCH 41 → 81, K 2 → 3),
the relax widens to ±2 and the `pbt` column grows 16 → 81 bytes (320 MiB at
4K).

- **The graded vertical EEDI3 is HOST<->DEVICE TRANSFER bound, not kernel
  bound.** Ablating every EEDI3 kernel from the recording (`VSFEEL_EEDI3_SKIP`)
  moves the graded 2x2160p run 2.35 → 2.15 ms/frame (+8.9%); the row kernel alone
  is +9.1%. A bare `GPUUpload → GPUDownload` of one 4K16 plane measures 1.89 ms,
  so the per-output-frame transfer bill (~50 MB: sclip every frame, clip+mclip
  every other, the output download) *is* the frame; the filter owns ~9%.
- **Batch size (`VSFEEL_EEDI3_BATCH`, default from a scratch target — 256 MiB,
  512 for EEDI3AA).** The knee is where the submit path stops being amortised by
  the GPU fill, and **it moves with the harness's environment**: with RADV's
  transfer queue on (what the benchmark sets) EEDI3 2x2160p wants B=2 (B=1 355 /
  **B=2 470** / B=3 463 / B=4 453 / B=8 339, 2000 f), with it off B=3. Sweep
  batches under the harness's own env.
- **`pbt` is 2-bit packed** when a lane owns exactly two directions (TPITCH
  33..64, i.e. mdis 17..31, which includes the default 20): each lane builds a
  nibble and only even lanes store, combining their odd neighbour's nibble
  through a `subgroupShuffleXor`, so no barrier and no write race. Column stride
  41 → 16 bytes, and the whole per-frame scratch at 2x2160p 225 → 120 MiB (pbt
  170 → 63). Bit-exact vs the pre-port build on the whole 16-config sweep
  (mdis 5/20/40 cover the K=1/K=2/K=3 paths). Reusing the scratch instead of
  allocating it per frame (`gpuExecRetain` free list) measured **neutral**, so the
  allocator is not a cost. hp cannot pack: its deltas span ±2.

## Implementation

Benchmark call: `MANGOHUD=0 uv run tools/benchmark.py --filter eedi3 vsfeel
vszipcl` (and `--filter eedi3aa`, `--filter eedi3h` where registered;
`--eedi3-hp 1` for the half-pel search).

Two GPU passes per plane: `ENTRY_PAD` expands eedi3m's mirrors in VRAM from the
tight kept-row upload; `ENTRY_ROW` does the DP + backtrack; `ENTRY_VCHECK`
finalises each interp row. EEDI3H adds `ENTRY_XPOSE` (16x16 tiled transpose with an
LDS stage, clip → R', sclip → B') and `ENTRY_COMPOSE`; EEDI3AA adds `ENTRY_ASSEMBLEV`
(vertical merge of the two sub-frames) and reuses `ENTRY_COMPOSE` with a
`comp_fuse` push constant.

### hp (half-pel search)

Spec constant 8 (`HPF`) = 2 doubles `TPITCH`/`CENTER` (u in half-pel units): the
s1/s2 windows and gates take the half-pel direction, s0 windows the half-pel rows
for odd u and the kept rows at half the offset for even u, the relax widens to ±2
with a halved gamma, and interpolate/vcheck get even/odd (2-/4-tap) forms.
`ENTRY_HPFILL` (one workgroup per written pad row, between the pad and row
phases) precomputes them into region `hp` (binding 8); hp's `pbt` deltas span ±2,
so its columns are full `TPITCH`.

- `ENTRY_ROW` is a subgroup-register DP: one 32-lane workgroup per interp row, each
  lane owning `K = ceil(TPITCH/SGSIZE)` consecutive directions in private registers,
  the ±1 relax window read from neighbouring lanes via `subgroupShuffleUp/Down`
  (zero per-column barriers). `pbt` stores **relative** predecessor deltas, so
  backtracking accumulates (`fpath[x] = fpath[x+1] + pbt[...]`).
- Rolling window sums: a column advance costs one new leading term instead of
  ~90 loads; state dies at a masked column and reseeds at the next interior one.
- Span-skip: lane 0 scans the packed mask words for the first set bit into shared
  `rowXmin`; a fully-masked row takes a parallel cubic loop, else the DP starts at
  `max(1,xmin)` with an analytic predecessor seed and the walk stops left of `xmin`.
- Memory path (R80): every input plane is read straight out of the core's GPU
  frame at its own pitch and every kernel writes into the output frame's own
  memory; the mask predicate/dilation and the horizontal transpose are GPU
  kernels too. The per-frame scratch (pad, hp, dst, pbt, dmap, cint, vout, bits,
  R', B', v, o0, rempty) is one `createGPUBuffer` per frame handed to the exec
  context with `gpuExecUsesBuffer`, so the pool reclaims it when the submission
  completes. One `Eedi3Job` per frame per sub-pass feeds `record_pass`, split into
  `kPrep`/`kHp`/`kRow`/`kVcheck`/`kTail` so a batch's frames are recorded with no
  barrier between their row dispatches. Output frames are batched
  (`d->batch_size`, `VSFEEL_EEDI3_BATCH`) and cached for the sibling `getFrame`
  calls; `d->width_pipes` deduplicates per-width pipelines. `num_streams` and
  `device_id` are registered no-ops: depth is the core's pool, the device is
  `core.set_vulkan_device`.

### vcheck contract (family A)

Runs once per plane **after** all interp rows exist (masked rows = cubic). Pad rows
`y=MARGIN_V+field .. height-MARGIN_V` step 2 → dst row `dy=y-MARGIN_V` steps 2; the
pad-row write gate ≡ `dy ∈ [2, dstH-3]`.

Taps at column x for interp row dy: `dst2p` = dst dy-2 (**previous interp row,
already vchecked**); `dst1p/dst1n` = dst dy∓1 (kept source rows, never vchecked);
`dst3p/dst3n` = source kept rows dy∓3 (from pad); `dst2n` = dst dy+2 (next interp
row, still plain); `dmap[x±width]` = neighbour interp rows' dmap.

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

**Structural consequence (the implementation may rely on this):** the row
dependency is **only** dy-2 (finalized) and dy+2 (still plain) → two separate
passes, or one pass holding the previous row's final tline in shared.

### Reference structure (five backends, same GPU, READ-ONLY)

| ref | type | mclip | io convention |
|---|---|---|---|
| `eedi3m` (ORIGINAL) | C++ AVX2 | per-plane, same-fmt→8bit Point | native int scale (params ×1<<(bits-8)); f32 beta/gamma/vthresh ÷255; alpha unscaled; cost3 → alpha÷3; **f64 DP** |
| `eedi3vk2` | Vulkan GLSL | same as clip (planes[]) | native int scale; f32 `precise` DP |
| `vszip` | CPU Zig SIMD | single Gray→Gray8, all planes | native int scale; alpha÷3 beta÷255 gamma÷255; NO per-column clamp |
| `vszipcl` | OpenCL | **none** | **u16 normalized [0,1]** (÷65535); f32 identity; mirror-pad full mdis |
| `vszipcu` | CUDA/HIP | none | u16 normalized [0,1]; same structure as vszipcl |

**Two semantic families.** **A = eedi3m + eedi3vk2:** clamp
`umax=min(x,W-1-x,mdis)`; relax `v ∈ [max(-umax2,u-1), min(umax2,u+1)]`,
`umax2=min(x-1,W-x,mdis)`; MARGIN_H=12; `pbackt` **absolute**; guard
`x>=|3dir| && x<=W-1-|3dir|` else avg2; tie-break ascending v, strict `<`.
**B = vszip + vszipcl:** full direction set per column via mirror pad
~3·mdis+nrad+8; `pbackt` **deltas**; center-pref tie-break; vszipcl's [0,1] u16 is
a **third numeric domain** (thousands of LSB vs A).

**Target.** Ground truth = **eedi3m** (user); vk2 shows A + f32 DP ≈ eedi3m.
vsfeel = family-A semantics, f32 DP: u16 native int scale, int32 SAD costs, f32
combine. Bar from round 2: **bit-exact vs eedi3vk2**; eedi3m = loose oracle.

Accuracy landscape, jpbd frame 100, 1920x1080, field=1, interp rows only (kept rows
identical everywhere):

| comparison | f32 max abs | u8 max LSB |
|---|---|---|
| vszipcl vs vszip | 1.2e-7 (ulp) | — |
| eedi3m vs vszipcl | 0.067 on 1e-4 of px | 17 on 3.1% |
| eedi3vk2 vs vszipcl | 0.012 | — |
| eedi3m vs eedi3vk2 | 0.067 on 1e-4 px | 17 on 0.003% |

### Constraints vsfeel implements

- `eedi3m` validation: field 0..3; !dh ⇒ processed planes even; dh ⇒ field≤1;
  alpha+beta≤1; nrad 0..3; mdis 1..40; vcheck 0..3; vthresh*>0 if vcheck>0;
  mclip/sclip same videoinfo+numFrames (sclip only when vcheck>0). `cost3` gates:
  s1 if `(u>=0&&x>=u2)||(u<=0&&x<W+u2)`; s2 if `(u<=0&&x>=-u2)||(u>=0&&x<W-u2)`;
  `s1=s2=s0` fallbacks.
- `eedi3m` DP/backtrack: `z=(double)ppT[mdis+v] + (double)(gamma*|u-v|)` capped
  `FLT_MAX*0.9` on the f32 store; `fpath[x]=pbackt[tpitch*x+mdis+fpath[x+1]]`
  (column x's pbackt written during column x+1's DP); `copyMask` = per interp line
  last-scan dilation, `minmdis=min(W,mdis)`. vsfeel: MARGIN_H=12, MARGIN_V=4,
  BT_TILE=32, `RING_CAP=7` — `src/eedi3.comp`.

## Performance

`VSFEEL_EEDI3_GPUTIME=1` timestamps the phases inside the submission's own
command buffer (it waits that submission out, so it serializes the pipeline: a
diagnostic, never a benchmark config). 2x2160p u16, GPU-resident input, per
frame: prep 0.40 ms, **row 1.61**, vcheck 0.03, blit 0.03, hp ~0. The row kernel
is 79% of the GPU time and the GPU ~87% busy, so it is ~2/3 of that frame.

Its cost per column per lane, same-session `EEDI3_PROBE` variants at gc3
(µs/batch, 3195.8 shipped): PROBE=13 (no pad load/cvt) 2666.4 = **16.6%**;
PROBE=14 (one leading term, not three) 2933.5 = **8.2%**; PROBE=15 (no ring
shift) 2812.5 = **12.0%**; PROBE=16 (ring_sum of one term) 3089.5 = **3.3%**.
So it is **ALU-issue bound (~1 IPC), not load bound**: f32 io, which drops every
u16→f32 convert but doubles the pad bytes, makes it 27% *slower*. The ring shift
register is the largest addressable item; the three-window redundancy is worth
8.2% but needs a 2|u|-deep delay line (~90 registers).

## Historical

- **2026-10-03 — hp (half-pel search), bit-exact against eedi3vk2.** `hp` was
  an accepted no-op (eedi3m registers it that way: "only full pel is
  implemented"); eedi3vk2, vszip and vszipcl implement it, so vsfeel does too.
  Ported from eedi3vk2's `HPF == 2` paths (doubled direction set, the hpfill
  precompute, the half-pel s0 window and cost combine, the ±2 relax, the halved
  gamma, the even/odd interpolate and vcheck forms). Two traps: the host's pbt
  stride had to double with hp (a half-sized region corrupted the frame), and
  the reference's float 2-tap average in the *even* interpolate branch carries
  the integer `+1` (u16 must not add it: its store already rounds half-up). The
  mclip fixups are replicated verbatim, including the MDIS-based lane indices
  that in hp land outside the reachable set for mdis ≥ 3. Bit-exact vs eedi3vk2
  hp=1 on u16 (mdis 1..40, nrad 0..3, vcheck 0..3, field 0..3, dh, sclip, mclip
  right/left/all) ; f32 too, except nrad=3 at mdis ≥ 25 (1.19e-7, the parallel
  vcheck's Jacobi residual). hp costs 2.3x the full-pel row.
- **2026-10-02 — f32 bit-exactness, found by the clip change.** On real content
  the f32 path lost its ~1 ulp band to DP argmin flips worth up to 4.9e-2, and
  u16 lost bit-exactness by 1 LSB on a few pixels per frame; every failing
  config had `vcheck > 0`. Two mechanisms, both in `ENTRY_ROW`/`ENTRY_VCHECK`:
  eedi3vk2's `precise` qualifiers were missing, so ACO contracted
  `alpha*(s0+s1+s2)+beta*|u|`, the relax's `gamma*|dd|+ext` and the vcheck's
  `a0/a1/a2`/blend into FMAs an ulp off (and `cubic4` on essentially every
  interpolated pixel); and the rolling window sums were accumulated
  incrementally where the reference re-sums the term ring in k order
  (`orderedSum`), which drifts by an ulp per column. Diffing the two shaders
  expression by expression is what found both; the ring re-sum costs −1.2% (459.3
  → 453.7 fps on the graded chain, inside the run spread). The `mclip` cubic
  test's own oracle was also wrong (`p3`/`n3` read as the same row); corrected to
  the integer cubic, which the masked region reproduces exactly.
- **Compat args `opt`/`ucubic`/`cost3` are registered accepted no-ops.** They
  select behaviour vsfeel always runs (AVX2-class path, cubic fill, three-window
  costs), so any value is byte-identical to the default
  (`test_eedi3_compat_args_are_accepted_noops`); `hp` used to fail dispatch
  (vsaa forwards it). No perf change.
- **The R80 port's own cost** (same-session A/B against the pre-port build, 2
  reps, graded medians): EEDI3 **612 → 384 fps**, EEDI3H 417 → 401, EEDI3AA
  155 → 148.
- **Undefined packed-pbt read in the K=2 DP loop**: the paired predecessor byte
  was stored during the per-direction loop and read the not-yet-initialized
  second delta. No perf change.

Rounds in order. Perf totals from rounds 2–13 are **void** (zero-mask path, below);
the correctness fixes, accuracy proofs and mechanisms in those rounds survive and
are kept. Superseded detail is deleted, not archived.

- **The row kernel's subgroup requirements are checked, not assumed** — it indexes
  lanes by `gl_SubgroupInvocationID` (so it needs exactly 32-lane subgroups) and
  moves data with shuffle/shuffle-relative. Only BASIC is mandatory in Vulkan, so
  both are required at creation; a device without them gets an error instead of a
  pipeline the driver may mis-execute.

### Pre-round-14 — the benchmark's mask was broken

The harness passed an already-scaled threshold to `Morpho.binarize_mask`, which
re-scales from the 32-bit range → 65535 → **mask 100% zero**, so every row took the
fully-masked early-out (real `vsaa`: `scale_mask(60, 8, 32)` = 15420). Measured
(700 f, ns=8): all-zero 593 fps vs vszipcl 203, real mask 274 / 202.

**Void perf record (do not quote):** rounds 2–13's totals, standings and
stream/queue knees, and round 10's "host is the wall / the DP is −1%" ladder.

What survives, by mechanism:

- **u16 bit-exact vs eedi3vk2** (round 2 onward) on every shared config — field
  0/1/2/3, dh, mdis 3..40, nrad 0..3, vcheck 0..3, custom alpha/beta/gamma/vthresh,
  sclip, mclip Gray8/16/32. f32 ~1 ulp (7.45e-9); gamma=5 + vcheck ≤5.45e-3 (tol
  5e-3). Residual flips vs eedi3m (13..1500 px of 115200, max ~3276 LSB) match
  eedi3m-vs-vk2's own count → eedi3m's f64 ordering.
- **Correctness**: the dh/field>1 segfault (`newVideoFrame2` got a null plane array;
  dst sized at source height); the nrad≥2 u16 divergence — the int path needs
  `ip=floor((t1+t2+1)*0.5)` and integer `v`, not the f32 midpoint, or 0.5 on odd
  sums flips near-tie argmins; the `s2` gate keeps vk2's `x<width-u2`.
- **Dropped `ucubic`/`cost3`** (round 3): no other backend exposes them; the GPU
  family hardcodes `cost3=true` (alpha/=3, cost=s0+s1+s2) and `ucubic=true`.
- **sclip output semantics** (round 4): under field>1 sclip describes the OUTPUT
  (2N frames; 2x height under dh) — what `based_aa` supplies via
  `Interleave([s,s])`. based_aa's real call is `field = tff+2` = 3 even on
  progressive content.
- **The vcheck must not read the row it is writing**: vszipcl's 1-barrier
  global-read model races in place (a lane writing `dst[r][x]` read by another
  computing `dst[r][x±dirc]`) — up to 28 LSB at mdis=20, nondeterministic. vszipcl
  only escapes by writing a separate out buffer.
- **`build_bmask_row` 6.11 → 0.28 ms/frame (22x)** (round 10): the scalar
  last-propagation dilation is a serial chain and dilations compose, so it collapses
  to O(log mdis) whole-array 64-bit shift-OR passes. Gotchas: the accumulator must
  span `width+mdis` bits or the right edge loses coverage; clear bits past `width`;
  the shift form equals the scalar only when `width >= 2*mdis`.
- **The in-graph mask conversion was a whole extra pass** (round 11). Native 16-bit
  masks read directly: zimg's full-range 16→8 is monotonic `round(v*255/65535)`, so
  `converted != 0` is exactly `v >= 129` (verified over all 65536 values), packed via
  `bmask_bits16` (xor 0x8000 → signed `cmpgt` → `movemask`). **`cmpgt` is strict, so
  the constant is `128 - 32768`, not `-32639`** — the off-by-one silently tests
  `v >= 130` and only a near-threshold input exposes it. Float and 10/12/14-bit masks
  keep the reference conversion.
- **Harness memory** (round 12): a bimodal fp32 result was `make_aa_vpy`'s 48 GB
  `max_cache_size` plus the benchmark's decoded-frame list being **additive**, past
  62 GB. Lower the fp32 Python cap, **never** `max_cache_size`.

### Rounds 5–13 — pre-port host path (deleted by the R80 port)

Mechanism only; none of this code ships, and none of it is a "current shape".

- **`DEVICE_LOCAL|HOST_VISIBLE|COHERENT` staging written with ordinary stores
  collapsed to 27.6 fps** (RADV's coherent device memory is uncached). Origin of
  "host-visible VRAM is poison", which is **wrong for NT stores**.
- **Rolling-window costs** (round 5): the single largest kernel win, re-measured at
  **+76%**. Bit-exact for integer pad (sums < 2^24).
- **Subgroup-register DP rewrite** (round 6): A/B 219.4 vs 220.3 fps — the
  masked-column *iteration*, not the DP, is the row cost.
- **Native u16 pad, NT copies, packed bmask** (round 7): pad halves CPU writes
  (16.6 → 8.3 MB H2D); `movntdqa` faults unaligned, so NT copies need
  `row_bytes%32==0`.
- **Span-skip** (round 7, shader-only, bit-exact) — see Implementation.
- **Round 8**: the GPU pad kernel is KEPT; the GPU bmask kernel is REVERTED (+7.3 MB
  H2D + 2 launches for CPU work that was already hidden); `ENTRY_VCOPY` is KEPT — it
  drops ~690 of 1080 barriers and proved barriers cost ~40 ns, not ~500 ns.
- **`VSFEEL_EEDI3_COPY`** bit0/1/2 = NT upload gathers / NT blit / blit load flavor:
  **m3 (NT load+store on both) wins in situ** — the isolated microbenchmark says the
  opposite and does not transfer.
- **ReBAR direct upload** (**+14%**): a `DEVICE_LOCAL|HOST_VISIBLE|COHERENT`
  per-resource `up_dev`, CPU-written by the NT path, deletes the H2D and its barrier.
  The descriptor pool needs one set per *role* (pad kernel b0 = raw upload, row
  kernel b0 = built pad), or a path silently redirects writes out of bounds.
- **Direct-to-frame host import**: bit-exact, **LOSS, default OFF**. Only the import
  must be page-aligned (4096), not the binding (64). **Never use a multi-region
  `vkCmdBufferCopy` for strided row copies** (1080 regions of 7680 B: 710 fps vs
  326), and per-frame host-pointer imports are a **GPU** cost on RADV.
- **EEDI3H was first `Transpose → EEDI3 → Transpose`** with a 25-test transpose
  oracle (bit-exact u16 and f32); superseded by the native implementation, but the
  oracle and the vsaa integration survive. Gotcha: after `mapConsumeNode` into an
  args map, forget the pointer — `freeMap` releases it.

### Rounds 14–19 — the first honest cost model, on the pre-port build

- **Round 14**: honest ladder (900-frame runs, ns=8): **vcheck+vcopy 18-27%**, pad
  ~8%, raw gather ~8%, sclip gather ~7%, blit 4-5%, ReBAR upload **+14%** (kept).
  Pad parity skip +0.8%: every read pad row has parity `(field+1)&1`.
  Structured-mclip nondeterminism found here (fixed in round 16.2).
- **Round 15 — the row kernel is the dominant cost** (overturns round 10):
  walking `pbt` directly instead of staging the full TPITCH through LDS was
  **+9-13%**, because the serial lane-0 walk consumes one byte per column and
  staging pulled ~41x the needed data through the store→load dependency. Rolling
  window under mclip **+76%**. A two-dispatch DP/backtrack split was **NEUTRAL**
  (the phase boundary costs exactly what removing the same-WG store→load hazard
  saved). The u16 (+77%) vs fp32 (+21%) asymmetry is Amdahl: **the fp32 row kernel
  is entirely hidden**.
- **Round 16 — native float mask** (`mclip_native32`): based_aa passes the mask in
  the clip's format. The reference conversion's nonzero boundary is
  **`v > fl(0.5/255)`** — strict, since `>=` is off by one ULP (`0x3b008081` → byte
  0, `0x3b008082` → byte 1); verified over 63 cases with modal outputs over 5 runs.
  One deliberate divergence: above ~8.42e6 the reference conversion overflows int32
  and emits byte 0, while the native path treats those as nonzero, matching
  eedi3vk2.
- **Round 16.2 — the structured-mclip bug was a `break` that skipped the write.**
  The backtrack tile loop `break`ed for tiles entirely left of `xmin`, correctly
  skipping the *walk* but also the **cubic `dst` write** — every column below
  `xmin` kept uninitialized memory. Hence the structured-mask trigger and the
  all-zero/all-white/noise-mask cases being deterministic. Fix: write the vertical
  cubic for such tiles with the whole workgroup and `continue`. Perf-neutral.
  Agreement with eedi3vk2 over 4 real 4K frames: 22758 → 5902 → **3** differing px,
  max 1 LSB; real-chain determinism 3-5 distinct outputs in 8 runs → **8/8
  identical**. `test_eedi3_mclip_long_mask_off_prefix` is mutation-verified.
- **Rounds 17-18 — EEDI3H went native.** `std.Transpose` ≈ **0.91 ms/pass** at
  2x2160p u16 (16.6 MB read + 16.6 MB write, ~36 GB/s CPU memory path); the four
  passes were essentially all of EEDI3H's 2.16x penalty. The same EEDI3 pipeline on
  the transposed plane (`Eedi3HCreate` sets `d->horiz = true`, kernel dims swapped)
  took it to 0.76 ms and 1.55x. Traps: unused `std.Transpose` nodes are **pruned**;
  dropping `sclip` as a proxy for fewer transposes flips `HAS_SCLIP == 0` and makes
  the row kernel compute `cint` everywhere (258 vs 489 fps); ordinary stores into
  uncached ReBAR VRAM are catastrophic; **`vout` must be device-local**; the mask
  gather must key off the **mask's** depth, not the clip's.
- **Round 19 — host-path tuning.** Fused mask path (`gather_mask_bitmat`, +6.6%)
  and fused kept+interp gather (`gather_columns_pair`, +5.3%, exploiting based_aa's
  `Interleave([clip, clip])` handing out one plane pointer). Traps: **loop order
  matters more than byte count** (k-outermost keeps 64 row streams open and
  re-reads the mask frame per block, 2x slower), and the mask cost is the READ, not
  the ALU. Dead ends: `compose_tight` CPU merge (−4.8%), `NOMASKX`'s +33% was stale
  bits. Method: with a fixed variant order an **inert knob moved 5%** — session
  drift exceeds the effects; reverse the order on odd reps and grade only those.
- **Why EEDI3H cannot be closed further**: the mask read is 16.6 MB vs 8.3 (a
  transposed plane needs all `height` rows) and compose+blit is 49.8 MB vs 33.2
  (the output must be assembled from transposed interp values *and* kept columns).
  Every rearrangement — CPU merge, spread stores, SDMA D2H, `vout` in dev_buf, host
  import — measures worse or neutral in situ. The way out is the fused chain:
  EEDI3AA.

### Rounds 20–21 — the walk chain is dead; the vcheck verdict reverses

- **Round 20 — the harness bug that voided round 15's probe numbers**: `ENTRY_ROW`'s
  guard was `#if PROBE >= 3` instead of `== 3`, so levels 4-7 returned immediately
  (1656 B vs 43720 B). New levels: 7/8 = an extra walk (f-dependent / independent),
  9 = all columns to one address, 10 = 8 unrolled 4x, 12 = faithful no-row-kernel.
- **Breaking the walk's serial load chain is worth exactly zero** — 453.9 / 451.4 /
  457.0 fps for the three variants, and PROBE=6 is a dead tie with real (532.6 vs
  530.6). Issue/sector-throughput bound, not latency bound; 3840 dependent L1 loads
  per row are already overlapped. **2-bit pbt packing: ceiling ~5%, and the store is
  not bandwidth bound** — PROBE=9 is the direct proof (1/40th the traffic on one
  dirty line is *slower*, because repeated writes to one dirty line serialise).
- **Ablate the row kernel with `PROBE=12`, never `PROBE=2`/`5`**: those return with
  `dmap` stale, so the vcheck takes its `dirc == 0` fast branch and the row kernel
  is overstated by ~9 points on EEDI3AA (real 98.19 vs PROBE=2 150.39 = 1.53x;
  faithful PROBE=12 89.76 vs 72.00 = 1.25x).
- **The environment is bimodal**: the same binary and vpy measured EEDI3AA 97.7 /
  90.9 / 79.8 fps and vertical 530/495/451/410 while sclk 2304-2338 MHz, `mclk`
  1249, `fclk` 2000 stayed pinned, `gpu_busy_percent` 100, junction 86-90 C, vspipe
  CPU <0.3%, MemAvailable flat. Not clocks, thermals, host CPU or harness memory.
  **Effects under ~5% are not measurable here.**
- **Round 21 — the vcheck default flips to the global-read form.** The LDS
  ping-pong removes one global read per `dirc != 0` pixel, but its walk must visit
  **every** row and ends with an extra full-width flush, while the global form skips
  fully-masked rows via `ENTRY_VCOPY` — and ~2/3 of AA rows are fully masked.
  Order-reversed medians (1500 f, ns=8, 4 reps): EEDI3 vertical 498.3 → **513.6**
  (+3.1%), EEDI3H horizontal 340.9 → **378.4** (+11.0%). `VCHECK_LDS` now defaults
  to 0; the LDS pipeline is still built and selectable with `VSFEEL_EEDI3_VCLDS=1`.
  Zipcl's H/V ratio is ~0.91; the reversal is row-mask-density dependent.
- **Round 21 — EEDI3AA's horizontal compose merges in VRAM** (+3.0%, bit-exact): the
  two assembled horizontal planes used to go through staging for a CPU 50/50 merge.
  `ENTRY_COMPOSE` gained a `comp_fuse` push constant (sub-pass 0 parks O_0 in the
  device-local `o0` region, sub-pass 1 reads it back and averages exactly like
  `std::Merge`: `(a+b+1)>>1` u16, `0.5a+0.5b` f32). 89.7 → 108.8 fps with the
  round-21 changes together.
- The remaining EEDI3H gap at ns=8 (0.69 ms) is compose 0.31 + blit 0.28 + ~0.1 host.
  The compose is a 16.6 MB transpose writing directly to host memory at the PCIe line
  rate; a VRAM write + SDMA D2H moves the same bytes at the same rate. **Zipcl is not
  a counterexample**: its absolute horizontal overhead is 0.48 ms vs 0.67, but its
  vertical baseline is 2.5x slower so the ratio looks worse.
- Re-confirmed dead ends: `COPY=7` (cached blit load) 387.4 → 377.1 horizontal;
  host-pointer import still catastrophic; `QUEUES=8` still wins; `NOPAD`/`NOXPOSE`
  stay unusable as ablations (they change the row kernel's input and branch mix).

### Rounds 22-26 — cross-cutting correctness and probe hygiene

One bullet each; no perf change in any of them (the README's rule for cross-cutting
passes). All bit-identical to the previous build unless stated.

- **22 validation hardening**: flush/invalidate through the shared `mapped_range`
  helper (EEDI3's duplicated per-plane entries collapsed to one);
  `VK_EXT_subgroup_size_control` enabled only when advertised; extensions enumerated
  once; `apiVersion` checked; `vkCreateComputePipelines` takes
  `pipeline_cache_lock`; `allocate_memory` refuses rather than relaxing a
  `DEVICE_LOCAL|HOST_VISIBLE` request; `FramePool::emplace()` so a creation error is
  torn down instead of leaking.
- **23 NT-store alignment**: `deint_row_u16/f32` gated the streaming store on the
  **element index**, not the pointer, so any destination row whose cell length was
  not a multiple of 32 bytes (EEDI3H's transposed kept-row cells: 1260 B u16 /
  2520 B f32 at a 630-px source) silently became a cached store into the uncached
  VRAM window. Predicate dropped; `nt` alone selects `_mm256_stream_*`.
  **Perf-neutral in practice** (AA geometry, 630-px luma, EEDI3H field=3, mclip,
  800 f, ns=8, interleaved: +0.4%/+0.6%) — full-width AVX stores coalesce in the WC
  buffer, unlike the narrow scalar stores behind the old "27.6 fps" reading.
  Bit-identical output at 630/638/640 px.
- **24 four latent defects**: `VkPipeline destroyed[8*3]` had zero headroom
  (EEDI3AA's worst case is exactly 24) → `std::vector` + `std::find` dedup;
  `import_plane_host_memory` bound at `addr & (align-1)` without querying the
  imported buffer's `VkMemoryRequirements` → now queries and falls back to the CPU
  blit on a short region or bad `memoryTypeBits`; `field > 1` doubled `numFrames`
  unconditionally, turning a `-1` unknown-length source into `-2` → `if (numFrames >
  0)` (also in NNEDI3); `RAWSTAGE=1` alone was garbage (needed `NOREBAR=1`) and
  `NOBLIT` was a silent no-op on EEDI3AA → now a create-time error.
- **25 probe cleanup**: GBENCH indexed slots by `tag_id & 1`, so EEDI3, EEDI3H and
  EEDI3AA's vertical pass all accumulated into slot 0 and "stage N" meant different
  work per graph → four slots and an explicit slot argument. `MAXW` had two unlinked
  copies (the host's `MAXW_LDS` claimed CMake passed `-DMAXW` but the rule did not),
  so the LDS-fit check and `tfloat tlineSh[2][MAXW]` could drift → `EEDI3_MAXW` is
  now the single source, default 4096. Per-frame flush vectors became a reusable
  `Eedi3Resource::mapped_ranges`. Bit-identical to `8e65f1f` across 14 configs.
- **26 probe corruption**: the PROBE 7/8/10 "fake store" kept `f2` alive with
  `if (f2 == pc.pbt_base) dmap[rempty_base + r] = 0`, but `pbt_base` is a byte offset
  while `f2` is a sum of byte-valued steps, so firing cleared the per-row empty flag
  the vcheck reads — changing the vcheck's branch mix and the output. Replaced by a
  read-only sink into `rowXmin`: PROBE=7 row SPIR-V 43744 → **44432 B** (BITS=16), so
  the walk is still compiled in, and the output is byte-identical to PROBE=0 (sha256
  `283717477cead12f`). Warning baseline: `-Wall -Wextra -Wshadow` gave 191 hits, all
  but 2 being `-Wmissing-field-initializers` on the Vulkan designated-initializer
  convention; that one is silenced for the plugin and the baseline enforced in
  CMakeLists.
- **Batch-cache races returned `nullptr` with no error**: `getFrame` checked the
  cache and then took the frame under two separate locks, so a sibling batch could
  evict it in the gap, and the producer took its own frame after publishing, where a
  concurrent publish's eviction could win it. The claim and take are one
  `eedi3_take_or_claim` now and `eedi3_publish` hands `first` back under the same
  lock. No perf change.

### Rounds 27-30 — the parallel vcheck, the fused AA submit, and hardening

- **27 — the serial row walk is gone from the shipped path; the vcheck is parallel
  by default.** Row r takes `d2p` from the *un-vchecked* dst row r-1, so no vcheck
  write feeds another row: the pass runs one workgroup per interp row
  (`grid (1, rows, 1)`) and needs no vcopy. Same-session A/B, 5 order-reversed
  2000-frame pairs (ns=8): EEDI3 **497.96 → 628.68 fps, +26%**; EEDI3AA (1000-frame
  reps) **108.3 → 160.2 fps, +48%**. The drift is a blend decision flip that decays
  geometrically: u16 max (of 65535) is 4157 on 0.072% of pixels at level 1, 580 on
  0.018% at level 3, 36 on 0.003% at level 5, 5 on 0.0007% at level 6. **Default is
  level 6**, the lowest bit-exact on the whole tested surface (level 3 is the first
  that is not); `VSFEEL_EEDI3_VPARA=0` restores the serial walk as the A/B control,
  1..6 pin a level, and `test_eedi3_parallel_vcheck_matches_serial` pins the two
  arms together. The ladder costs no extra dispatches: one shared `vc_blend` and a
  macro-generated driver per d2p source.
- **28 — EEDI3AA's two horizontal submissions fused into one.** Both AA horizontal
  sub-passes read only the merged frame, which the first fence has finished, so they
  need neither the host nor a second fence between them: one CB, one
  `submit_with_fence`, one wait; both parities are pre-gathered with the
  `gather_columns_pair` form, and the parity-independent mask bits lose their
  duplicate build. ns=1 same frame **21.63 → 19.47 ms**; the hWait drop was only
  ~0.9 ms, not the 6 ms the model predicted — most of it is compute the GPU still has
  to do. ns=8 interleaved 2000 f: 141.3 → 144.8 and 132.4 → 144.5; fused won or tied
  every pair. A few percent, not 10%. Bit-exact vs the two-submission control (still
  in-tree as `VSFEEL_EEDI3AA_HFUSE`): 15 GRAY geometries x u16/f32 + 7 multi-plane
  cases, 0 mismatches. Bounded out: pre-recording the CBs (≤0.07%) and the pair
  gather alone (neutral).
- **29 — row-kernel levers.** **SGSIZE 64 / K=1 is +2% on EEDI3AA, neutral on
  EEDI3, not shipped** (a `-D`, a second row module and its device-limit handling for
  ~2% on one workload). The four 1D dispatches folded into Y
  (`ID.y * NumWorkGroups.x + ID.x`) because they reach ~130k workgroups at 8K, over
  the 65535-per-dimension limit; verified bit-identical with `VSFEEL_LIMIT_GRID_X=4`.
  Shaderstats: 96 VGPRs, 0 spills. The later `PROBE` attribution (Performance)
  supersedes this round's "latency has nothing to hide behind" reading: the kernel
  runs at ~1 IPC, i.e. it is issue bound, which is also why the K=1 shape bought
  nothing.
- **30 — creation-time hardening and variant-gated pipelines.** `dh` now requires
  every plane — a `planes` subset made a CPU output frame that failed per frame;
  scratch over 2 GiB (int32 region bases) and non-finite `alpha`/`beta`/`gamma`/
  `vthresh*` (NaN is false against every range check) are rejected at creation.
  Partial pipeline sets are owned from the first create: a failed later kernel
  abandoned a stack struct with no destructor, so a 128-invocation device leaked 2
  `VkPipeline`s that validation named at `vkDestroyDevice`. Pipelines are created per
  key's `horiz`, not `d->horiz` — a vertical EEDI3 built xpose/compose/assemble,
  EEDI3H built blit, and every key built both mask dilation forms, for kernels the
  invocation cannot dispatch. Dead code removed: the host `alloc` stage's structurally
  zero sample, `src_h`, `mclip_native16/32`, and the row kernel's unreachable
  `xd == 1` masked arm (`bmask` is a ±MDIS dilation, so `bmask[0] | bmask[1]` set
  implies `bmask[1]`).

- **2026-10-03 — EEDI3/EEDI3AA cost model and the transfer wall.** Added
  `VSFEEL_EEDI3_GPUTIME` (in-CB phase timestamps) and `VSFEEL_EEDI3_SKIP` and
  attributed the whole frame (Performance). The graded *vertical* number is a PCIe
  budget, not a kernel one: ablating every kernel moves it 2.35 → 2.15 ms/frame and a
  bare 4K16 upload+download measures 1.89 ms, so its kernels are 8.9% of the frame.
  Shipped: `PAD_STRIDE` as a compile-time constant (the host's `cfg.pad_stride` is
  `align16(WIDTH+24)`, so the pad pitch no longer rides a push constant), `roll_push`
  sharing the column `x+NRAD+1` triple between s1 and s2 (18 → 16 taps),
  `combine_cost`'s centre taps hoisted to the column, and the EEDI3AA batch knee
  re-swept (B=3 → **B=4**, +4.8%; the sweep must run under the harness's RADV env or
  the answer is B=3). Order-reversed same-session pairs: EEDI3 469.1 → 469.3
  (+0.0%), EEDI3AA 183.1 → 198.7 (**+8.5%**), 971 tests pass. Dead ends: a per-frame
  scratch free list through `gpuExecRetain` (neutral), a float pad to delete the
  u16→f32 converts (27% *slower*: load bytes beat the converts), the deep-interior
  hoist (neutral), the walk/interpolate split (−2.3%, Do not retry). The probes also
  hoisted the DP relax's `gamma*|dd|` to a bare `gamma` — valid only at full-pel (hp's
  `dd = ±2` is `2*gamma`), which failed the 12 hp tests; restored, no perf change.

## Open work

- **`eedi3h_vszipcl_loose` is a family-gap sanity bound, not a tolerance**: 0.06 /
  16384 on this clip (3.9-4.0% of pixels differ, p99.9 ≤ 294 LSB, max 4297). Only
  gross errors (wrong axis, broken composition) blow the fraction up to ~1.0.

- **Row kernel register pressure**: `RING_CAP` is not it. Trimming the fixed
  bound to `2*NRAD+1` leaves the compiled row kernel **byte-identical** at
  nrad=1/2/3 (VGPR 96/96/120, 16/16/12 subgroups/SIMD, no spills): the
  `if (k >= RN) break` guard is dead once the driver specializes `NRAD`, the ring
  is register state, not LDS (1 024 B), and at nrad=3 `RING_CAP == RN` exactly, so
  the lever is the `K=2` direction state. (Spec constants *can* size arrays here —
  `notes/NLMEANS.md` ships it.) ACO raises VGPRs for load ILP deliberately: a
  lower count with new spills or schedule damage is a regression.
- **SGSIZE 64 / K=1** — +2% on EEDI3AA only; not shipped (round 29).
- **hp's row cost**: three structural items ride on the doubled window work —
  the pbt column is 81 bytes at full TPITCH (hp's ±2 deltas would fit 3 bits),
  K = 3 keeps 69 ring registers live per lane, and hpfill is its own phase.
- **Kill the ring shift register** — the one large row-kernel item left (PROBE=15:
  12.0% of the kernel, ~8% of EEDI3AA). Shape: a circular ring of `RING_CAP` slots
  whose window is read from a rotating base, which needs the column loop unrolled by
  RN with the base compile-time per copy (any runtime index spills the ring to
  scratch). The base must then advance on EVERY column, masked ones included, so
  `roll_seed` has to write all RN slots at the current phase — it already writes all
  of them, so that is a rotation of the write indices, not a new invariant.
- **The three-window redundancy** (PROBE=14: 8.2%) is real but needs a 2|u|-deep
  delay line per direction: ~45 registers at mdis=20 for the worst lane, i.e. ~90 for
  K=2 against the 96 the kernel uses today. Only reachable if something else gives up
  its registers first.
- `ENTRY_PAD` div/mod by `pad_stride`, `ENTRY_VCOPY`/`ENTRY_BLIT` div by `WIDTH`: a
  2D dispatch removes them. Per-plane passes, not the row kernel.
- **Fuse the vcheck across planes** (one launch, `gl_WorkGroupID.y = plane`, as
  vszipcl does) — YUV-only, cannot move the Gray flagship, real for colour AA.
- **The graded vertical number is a transfer budget, not a kernel budget.** ~50 MB
  per output frame crosses PCIe (sclip every frame, clip+mclip every other, the
  output download); the filter's kernels are 8.9% of it. The one lever inside the
  filter is that `sclip` is `Interleave([sclip, sclip])` of the same image as `clip`
  in every based_aa chain, so its upload is redundant — but skipping it needs
  `sclip:vnode:all`, a host-pointer/test comparison, and a real fallback upload for
  the case where the two planes differ. Measured cost of the whole sclip leg
  (`vcheck=0`, which also drops the vcheck kernel): 469 → 545 fps, so the ceiling is
  ~16% and the fallback path is what makes it expensive.
- **Accuracy-for-speed items are void on mechanism**: fp16 pad/cost storage has no
  traffic to remove (pad is native u16, DP costs live in registers) and adaptive
  `nrad`/`mdis` trades accuracy for ~0. Any accuracy relaxation still requires
  measuring the drift on the noise clip and updating `tests/test_eedi3.py` in the
  same change.

### Do not retry

*(stale)* marks a verdict measured on the degenerate pre-round-14 config — kept as
the reason a variant failed, not as a current number.

- **Breaking the walk's serial load chain** by lookahead, ±1 candidate windows, or a
  layout making the address f-independent: 453.9 / 451.4 / 457.0 fps — identical —
  and PROBE=6 is a dead tie with real (532.6 vs 530.6). **4x-unrolling the walk with
  merged `tileF` stores** is a wash too (8 order-reversed 2000-frame pairs, median
  ratio 0.99). The walk is issue bound, not latency bound.
- **Ablating the row kernel with `PROBE=2`/`5`/`4`/`6`**: 2 and 5 return with `dmap`
  stale (the vcheck takes its `dirc == 0` fast branch, overstating the row kernel by
  ~9 points on EEDI3AA — use `PROBE=12`), and the old round-15 4/5/6 numbers are void
  because a `>=` guard compiled them to an empty kernel (1656 B vs 43720 B).
- **`pbt` store traffic**: DP+store is 4.8% of the vertical frame and the 170 MB store
  runs at ~1.9 TB/s (L2/port, not DRAM). PROBE=9 proves bytes are not the cost —
  1/40th the traffic on one dirty line is *slower*. Packing also adds ballots to an
  issue-bound kernel.
- **Splitting DP and backtrack into two dispatches** (round 15): neutral (vc0 322 vs
  326, vc2 295 vs 295) — the boundary costs exactly what removing the same-WG
  store→load dependency saved. **Splitting the walk out to parallelise it (one lane
  per ROW) lost too**: 189.8 vs 194.1 fps on the 2x2160p AA workload (same-session
  interleaved medians, reverted). The old form already ran hundreds of 32-lane
  workgroups concurrently, so "31 of 32 lanes idle" was never the limiter; the split
  only added a dispatch, a barrier and a global `dmap` round-trip.
- **Per-frame `VK_EXT_external_memory_host` imports** (default off): the import is
  GPU work on RADV (VM map/unmap + TLB sync per 16.6 MB BO). Arms: staged 608 |
  `vout` in dev_buf 588 | import the GPU never touches 487 | import+blit 390. Re-test
  only if RADV's import path gets cheaper.
- **Multi-region `vkCmdCopyBuffer` for strided row copies** (~13 us *per region* in
  situ), and **`vout` in dev_buf + SDMA D2H** (`VSFEEL_EEDI3_VOUTDEV=1`, still a
  loss: 289 vs 313-323; 327 vs 375 with vcheck off). **Lazy `cint`/`sclip` load in
  vcheck** is void by inspection: `cint_s` is both the `dirc == 0` result and the
  `dirc != 0` blend operand, so every `do_row` pixel reads it.
- **`LDS ping-pong tlineSh`**: the verdict has flipped twice (round 14 +3%, round 21
  +11% the other way) and both are real — it is workload-dependent on row-mask
  density, so re-measure rather than inheriting either.
- **(stale) batch**: a single contiguous blit instead of per-row strided copies (525
  vs 555); plain `memcpy` instead of the NT load/store pair anywhere in the frame path
  (blit 6.1 → 10.4 ms/frame; vc0 481 → 401); skipping the pbt global stores
  (`PROBE=1`); a 2-subgroup x 2-row 64-lane workgroup for `ENTRY_ROW` (regressed in
  every config; NOT the untried one-subgroup 64-lane / K=1 shape); `subgroupcoherent`
  on pbt (259 → 229); `restrict` on pbt (214 → 201); `[[unroll]]` on the K-direction
  loops (227 → 196); dropping `HOST_CACHED` from staging (collapsed to 43 fps — NT
  loads need WB/WC memory).
- **Queue cap** (`VSFEEL_EEDI3_QUEUES`, pre-port, gone with the one-queue R80 API):
  3 loses ~20%, 4/6/8 tie.
- **`core.max_cache_size` in the benchmark harness**: lowering it collapses *every*
  plugin (vsfeel 61 vs 220, vszipcl 39 vs 200) because the timed region starts
  re-running the decode/mask chain. Shrink the harness's own frame cache instead.

### Method rules

- **Grade sweeps at >=900 frames** (a 700-frame queue sweep reported +14% for
  `queues=4`; at 900 it was a tie), and grade the median per-pair ratio over >=4
  order-reversed 2000-frame pairs: effects under ~5% are not measurable on this box.
  The vertical arm is *not* less noisy than EEDI3AA — it looked stable for an hour,
  then swung 20%.
- **Never compare a number recorded in a different command** — interleave A/B in one
  session, and in the same *environment*: the EEDI3 batch knee is at 2 with RADV's
  transfer queue and at 3 without it, and a sweep run in the wrong env answers the
  wrong question.
- **Check `CMAKE_HOME_DIRECTORY` in a second build tree before trusting an A/B against
  it**: a copied/worktree `CMakeCache.txt` can still point at the main checkout, so
  `tools/install.sh` silently rebuilds the main tree and both arms are the same binary.
- **An isolated microbenchmark proposes; only an in-situ same-session A/B decides.**
- **Prove the stage split with the ablation ladder, not theory**, and make each rung
  reproduce what the consumer reads (PROBE=2/5 leave `dmap` stale → cheaper vcheck →
  overstated row kernel). Additive costs of similar size mean a shared memory path;
  one dominant item means a loop. `VSFEEL_EEDI3_SKIP` is the live ladder;
  `VSFEEL_EEDI3_GPUTIME` reads the per-phase GPU split directly.
- **Before trusting any `PROBE` number, `stat -c %s build/vk_spv/eedi3_16_row.spv`**:
  a `>=` guard once compiled 4-7 to an empty kernel (1656 B).
- **The benchmark's EEDI3 mask follows the clip's depth, like based_aa**; both depths
  take the same no-conversion path, with one deliberate documented divergence (floats
  above ~8.42e6).
- **A skip justified by what a region *reads* must also be checked for what it
  *writes*** (the structured-mclip bug). Regression test
  `test_eedi3_mclip_long_mask_off_prefix`, mutation-verified.

## Debug env vars

- `VSFEEL_EEDI3_VCLDS` — `=1` forces the LDS vcheck ping-pong back (global is default).
- `VSFEEL_EEDI3_GPUTIME` — `=1` stamps each phase inside the submission's command
  buffer and reports the split. It waits each submission out, so it serializes the
  pipeline: a diagnostic, never a benchmark config.
- `VSFEEL_EEDI3_SKIP` — comma-separated `mask,pad,row,vcheck,tail` to leave a kernel
  out of the recording. Changes the output by design; the GPU-cost ablation ladder.
- `VSFEEL_EEDI3_VPARA` — vcheck form: 0 = serial row walk (A/B control), 1..6 =
  parallel with that many Jacobi steps. Default 6 (bit-exact on the test surface).
- `VSFEEL_EEDI3_BATCH` — output frames recorded per submission (default: a
  ~256 MiB scratch target, 512 for EEDI3AA, capped by the core's VRAM allowance;
  clamped 2..8, which lands on 2 at 2x2160p for EEDI3 and 4 for EEDI3AA, 8 at
  1080p; drops to 1 when the allowance cannot hold two frames). `=1` is the A/B
  control.
- `VSFEEL_EEDI3_TRACE` — one-shot banner: VRAM accounting, per-plane region layout,
  spec constants per geometry.
- `VSFEEL_EEDI3_TIMING` — per-frame host stage split (acquire/alloc/record/submit).
- `VSFEEL_EEDI3_SYNC` — additionally wait each submission out and report its wall time
  (serializes the pipeline; for GPU-time measurements at depth 1).
- `VSFEEL_EEDI3_NOCLEAR` — EEDI3AA's frame path only: skip the per-submission
  scratch clear (0 fill).
- `VSFEEL_EEDI3_POISON` — EEDI3AA only: `<hex>[:<region>]` overlays a pattern on
  the scratch, or one named region, before the passes. Changes output by design.
- `EEDI3_PROBE` — CMake cache var: ablation level 0/1/2/3/4/5/6/7/8/9/10/12 (11, 13+ unused).
- `EEDI3_MAXW` — CMake cache var: LDS vcheck max width -> `-DMAXW` + `-DEEDI3_MAXW_LDS` (default 4096).
