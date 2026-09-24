# EEDI3 — notes

Status: **shipped** — family-A semantics (eedi3m/eedi3vk2), f32 DP, u16 bit-exact
vs eedi3vk2. EEDI3, EEDI3H (native transposed-plane, no `std.Transpose` nodes) and
EEDI3AA (fused based_aa chain) share every kernel, geometry and pipeline. Runs on
the **R80 GPU API**: `clip/sclip/mclip:vnode:gpu` in and `ffGPUOutput` out, one
exec pool, and no host upload/download/gather/blit machinery at all. The parallel
vcheck is the default.

Benchmark defaults: 2000 f, real based_aa clip, 2x2160p, field=3, mdis=20,
vcheck=2. `MANGOHUD=0 python3 tools/benchmark.py --filter eedi3 vsfeel vszipcl`.

| clip | vsfeel | vszipcl | speedup |
|---|---|---|---|
| u16 | 384 | 214 | 1.8x |
| fp32 | 246 | 186 | 1.3x |

Interleaved same-session A/B against the pre-port build (2 reps, `--repeat 2`,
graded medians): EEDI3 **612 → 384 fps (−37%)**, EEDI3H 417 → 401 (−4%),
EEDI3AA 155 → 148 (−4.5%). `--gpu-cache` (input pre-uploaded) on the port:
EEDI3 380 → 417, EEDI3H 400 → 488, EEDI3AA 149 → 146.

- **The remaining EEDI3 gap is the R80 API's single compute queue.** The core
  creates exactly one compute queue (`VSVulkanCoreHandles.computeQueueIndex`; the
  transfer-queue commit 1be682f2 adds more *transfer* queues, which only changed
  the upload/download legs — measured +5% on EEDI3H and +4% on EEDI3AA, nothing on
  the vertical path). The row kernel launches only `rows` workgroups of a
  latency-bound scan: 1080 waves at 2x2160p leaves the GPU under-occupied, so
  throughput comes from running several frames' rows at once. The pre-port filter
  used up to 8 Vulkan queues (forcing it to one drops it 987 → 416 fps at 1080p,
  i.e. the whole difference is cross-queue overlap). Consecutive *dispatches
  inside one command buffer* do overlap (1080p: one row dispatch 2.25 ms, ×2
  1.45 ms/frame, ×4 1.10), so the port records a **batch of output frames per
  submission, phase by phase** (all pads, then all rows with no barrier between
  them, then all vchecks, then all tails) and caches the frames the sibling
  `getFrame` calls then take.
- **Batch size (`VSFEEL_EEDI3_BATCH`, default from a ~256 MiB scratch target,
  512 for EEDI3AA).** Swept on the graded 2x2160p workload: EEDI3 B=2 406,
  B=4 389, B=8 282 fps; EEDI3AA B=2 142, **B=4 157**, B=8 103. The knee is *not*
  memory — B=8 loses just the same with a 3.7x smaller scratch (mdis=5), and
  raising `VS_VULKAN_MAX_VRAM_MB` changes nothing. It is the submit path:
  `gpuExecSubmit` costs 346 us/frame at B=4 against 670 at B=8, i.e. it grows
  faster than the frame count, and a batch that drains quickly (the masked rows
  early-out) reaches the point where that dominates sooner. That is also why the
  packed pbt below *lowered* the knee from 4 to 2.
- **`pbt` is 2-bit packed** when a lane owns exactly two directions (TPITCH
  33..64, i.e. mdis 17..31, which includes the default 20): each lane builds a
  nibble and only even lanes store, combining their odd neighbour's nibble
  through a `subgroupShuffleXor`, so no barrier and no write race. Column stride
  41 → 16 bytes, and the whole per-frame scratch at 2x2160p 225 → 120 MiB (pbt
  170 → 63). Bit-exact vs the pre-port build on the whole 16-config sweep
  (mdis 5/20/40 cover the K=1/K=2/K=3 paths). The stores were worth ~10% of the
  frame ungated (PROBE=4, no pbt store, 406 vs 367 fps); packing recovers about a
  third of that and moved the batch knee down, for ~7% on the graded vertical
  workload (374 at the old 1 GiB target → 403).
- **Next lever if EEDI3 must reach parity:** the queue, not the kernel. Nothing in
  the R80 API offers a second compute queue, so the remaining ~1.5x would need the
  row kernel to fill the GPU by itself — e.g. splitting each row's column walk
  across workgroups with a boundary fixup.

## Implementation (current)

Benchmark call: `MANGOHUD=0 python3 tools/benchmark.py --filter eedi3 vsfeel
vszipcl` (and `--filter eedi3aa`, `--filter eedi3h` where registered).

Two GPU passes per plane: `ENTRY_PAD` expands eedi3m's mirrors in VRAM from the
tight kept-row upload; `ENTRY_ROW` does the DP + backtrack; `ENTRY_VCHECK`
finalises each interp row. EEDI3H adds `ENTRY_XPOSE` (16x16 tiled transpose with an
LDS stage, clip → R', sclip → B') and `ENTRY_COMPOSE`; EEDI3AA adds `ENTRY_ASSEMBLEV`
(vertical merge of the two sub-frames) and reuses `ENTRY_COMPOSE` with a
`comp_fuse` push constant.

- `ENTRY_ROW` is a subgroup-register DP: one 32-lane workgroup per interp row, each
  lane owning `K = ceil(TPITCH/SGSIZE)` consecutive directions in private registers,
  the ±1 relax window read from neighbouring lanes via `subgroupShuffleUp/Down`
  (zero per-column barriers). `pbt` stores **relative** predecessor deltas, so
  backtracking accumulates (`fpath[x] = fpath[x+1] + pbt[...]`).
- Rolling window sums: a column advance costs one new leading term (~9 loads)
  instead of ~90; state dies at a masked column and reseeds at the next interior one.
- Span-skip: lane 0 scans the packed mask words for the first set bit into shared
  `rowXmin`; a fully-masked row takes a parallel cubic loop, else the DP starts at
  `max(1,xmin)` with an analytic predecessor seed and the walk stops left of `xmin`.
- Memory path (R80): every input plane is read straight out of the core's GPU
  frame at its own pitch and every kernel writes into the output frame's own
  memory; the mask predicate/dilation and the horizontal transpose are GPU
  kernels too. The per-frame scratch (pad, dst, pbt, dmap, cint, vout, bits, R',
  B', v, o0, rempty) is one `createGPUBuffer` per frame handed to the exec
  context with `gpuExecUsesBuffer`, so the pool reclaims it when the submission
  completes. One `Eedi3Job` per frame per sub-pass feeds `record_pass`, which is
  split into `kPrep`/`kRow`/`kVcheck`/`kTail` phases so a batch's frames can be
  recorded with no barrier between their row dispatches. Output frames are
  batched (`d->batch_size`, `VSFEEL_EEDI3_BATCH`) and cached for the sibling
  `getFrame` calls; `d->width_pipes` still deduplicates per-width pipelines.
  `num_streams` and `device_id` are registered no-ops (never read): depth is
  the core's pool, the device is `core.set_vulkan_device`.

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
  BT_TILE=32, `RING_CAP=7` (=2*NRAD+1 for NRAD≤3) — `src/eedi3.cpp:71`,
  `src/eedi3.comp:92`.

## Historical

Rounds in order. Perf totals from rounds 2–13 are **void** (zero-mask path, below);
the correctness fixes, accuracy proofs and mechanisms in those rounds survive and
are kept. Superseded detail is deleted, not archived.

- **The row kernel's subgroup requirements are checked, not assumed** — it
  addresses `gl_SubgroupInvocationID` as a lane index (so it needs exactly 32-lane
  subgroups, requested when the device offers them) and moves data between lanes
  with shuffle and shuffle-relative. Only BASIC is mandatory in Vulkan, so both
  operations are now required at creation; a device without them gets a clear
  error instead of a pipeline the driver may mis-execute.

### Pre-round-14 — the benchmark's mask was broken

The harness passed an already-scaled threshold to `Morpho.binarize_mask`, which
re-scales from the 32-bit range → 65535 → **mask 100% zero**, so every row took the
fully-masked early-out (real `vsaa`: `scale_mask(60, 8, 32)` = 15420). Measured
(700 f, ns=8): all-zero 593 fps vs vszipcl 203, real mask 274 / 202.

**Void perf record (do not quote):** rounds 2–13's totals, standings and
stream/queue knees — 74.6 → 107 → 155-167 → 363 → 398 → 604 @8s — plus round 7's
"streams plateau at 8", round 10's "host is the wall / the DP is −1%", round 10's
ladder and the ReBAR +29%. Rounds 14/15/20 supersede the cost model.

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
- **Cached staging + `pad_dev` mirror** (round 5), still the shipped shape.
  `DEVICE_LOCAL|HOST_VISIBLE|COHERENT` staging written with ordinary stores
  **collapsed to 27.6 fps** — RADV's coherent device memory is uncached, so the
  ~16.6 MB/frame CPU pad write went uncached. This is the origin of "host-visible
  VRAM is poison", which is **wrong for NT stores** (round 14).
- **Rolling-window costs** (round 5): the single largest kernel win, honestly
  re-measured at **+76%** in round 15. Bit-exact for integer pad (sums < 2^24).
- **The vcheck must not read the row it is writing**: vszipcl's 1-barrier
  global-read model races in place (a lane writing `dst[r][x]` read by another
  computing `dst[r][x±dirc]`) — up to 28 LSB at mdis=20, nondeterministic. vszipcl
  only escapes by writing a separate out buffer.
- **Subgroup-register DP rewrite** (round 6): A/B 219.4 vs 220.3 fps (comparison
  valid though absolutes are void). Correctness preserved, speed identical — the
  masked-column *iteration*, not the DP, is the row cost.
- **Native u16 pad, NT copies, packed bmask** (round 7): pad halves CPU writes
  (upload 16.6 → 8.3 MB, H2D and pad-read bytes); `copy_stream_out`/`_read` NT copies
  need `row_bytes%32==0` (`movntdqa` faults unaligned); packed bmask rows upload 8x
  smaller and the shader burst-copies 120 words to shared.
- **Span-skip** (round 7, shader-only, bit-exact) — see Implementation. The masks it
  relies on are real: 53% of mask rows and 64% of interp rows are fully masked in
  the AA workload; non-empty rows average a 22% span.
- **Round 8 GPU-kernel outcomes:** the GPU pad kernel is KEPT; the GPU bmask kernel
  is REVERTED (+7.3 MB H2D + 2 launches to free CPU work that was already hidden);
  `ENTRY_VCOPY` + empty-row flag is KEPT (drops ~690 of 1080 barriers, and proved
  barriers cost ~40 ns not ~500 ns — the vcheck is *traffic*-bound).
- **`build_bmask_row` 6.11 → 0.28 ms/frame (22x)** (round 10): the scalar
  last-propagation dilation is a serial chain and dilations compose, so it collapses
  to O(log mdis) whole-array 64-bit shift-OR passes. Gotchas: the accumulator must
  span `width+mdis` bits or the right edge loses coverage; clear bits past `width`;
  the shift form equals the scalar only when `width >= 2*mdis`.
- **`VSFEEL_EEDI3_COPY`** bit0/1/2 = NT upload gathers / NT blit / blit load flavor:
  **m3 (NT load+store on both) wins in situ** at ns=8 (400-417 fps) and ns=4
  (330-332) — the isolated microbenchmark says the opposite and does not transfer.
- **ReBAR direct upload is the shipped path** (round 14: **+14%**): a
  `DEVICE_LOCAL|HOST_VISIBLE|COHERENT` per-resource `up_dev`, CPU-written by the NT
  path and read directly, deletes the H2D and its barrier. Two silent-corruption
  bugs: the descriptor pool needs one set per *role* (pad kernel b0 = raw upload, row
  kernel b0 = built pad), and the pad set's b8 must stay `pad_dev` or the pad kernel
  overwrites its own input.
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
- **Direct-to-frame host import** (round 13): built, bit-exact, **LOSS, default
  OFF**. Only the import must be page-aligned (4096), not the binding (64).
  Reusable: **never use a multi-region `vkCmdBufferCopy` for strided row copies**
  (1080 regions of 7680 B: 1 region 710 fps vs 1080 regions 326), and per-frame
  host-pointer imports are a **GPU** cost on RADV (VM map/unmap + TLB sync per
  16.6 MB BO).
- **EEDI3H was first `Transpose → EEDI3 → Transpose`** (round 9) with a 25-test
  transpose oracle (bit-exact u16 and f32); superseded by round 18's native
  implementation, but the oracle and the vsaa integration survive. Gotcha: after
  `mapConsumeNode` into an args map, forget the pointer — `freeMap` releases it.

### Round 14 — first honest cost model

- Honest ladder (900-frame runs, ns=8): **vcheck+vcopy 18-27%**, pad ~8%, raw gather
  ~8%, sclip gather ~7%, blit 4-5%, ReBAR upload **+14%** (kept).
- LDS ping-pong vcheck re-tested (round 6's "flat-to-worse" was degenerate):
  interleaved 900-frame pairs global mean 295 vs LDS mean **304** (+3%, tighter) →
  default then, reversed again in 21. Bit-exact vs eedi3vk2 and vs the global path.
- Pad parity skip **+0.8%** (4/4 pairs): every read pad row has parity
  `(field+1)&1`; `VSFEEL_EEDI3_PADPAR=0` restores the full build. Queue cap 3 →
  256-269 fps, 4/6/8 → 294-322 (tie); default stays 8.
- Structured-mclip nondeterminism found here (fixed 16.2).

### Round 15 — row kernel is the dominant cost (overturns round 10)

Baseline (2000 f, ns=8, real mask): vc2 264.1, vc0 308.5, vszipcl 181.5, eedi3vk2
125.2. `PROBE=2` 564.3 / 693.6 — but it leaves `dmap` stale and so overstates the
row kernel (round 20).

- **Walk reads `pbt` directly instead of staging the full TPITCH through LDS:
  +9-13%** (vc0 326→368, vc2 295→329). The serial lane-0 walk consumes one byte per
  column, so staging pulled ~41x the needed data through the store→load dependency.
- **Rolling window under mclip: +76%**, the round's big win (vc2 290.3/290.5 →
  513.5/512.7, interleaved 2000 f). Round 6 gated it because its "no gain" was
  measured with zero unmasked columns.
- Two-dispatch DP/backtrack split **NEUTRAL** (vc0 322 vs 326, vc2 295 vs 295): the
  phase boundary costs exactly what removing the same-WG store→load hazard saved. Do
  not retry without a different overlap story.
- The u16 (+77%) vs fp32 (+21%) asymmetry is Amdahl: fp32's no-row-kernel floor is
  339 vs the real 340, i.e. **the fp32 row kernel is entirely hidden**.

### Round 16 — native float mask; the structured-mclip bug was a stale `dst`

- **Float (GrayS) masks folded natively** (`mclip_native32`); based_aa passes the
  mask in the clip's format. The reference conversion's nonzero boundary is
  **`v > fl(0.5/255)`** — strict, since `>=` is off by one ULP (`0x3b008081` → byte
  0, `0x3b008082` → byte 1); verified over 63 cases with modal outputs over 5 runs.
  One deliberate divergence: above ~8.42e6 the reference conversion overflows int32
  and emits byte 0, while the native path treats those as nonzero, matching
  eedi3vk2. **+2% fp32**.
- **Structured-mclip nondeterminism = a `break` that skipped the write.** The
  backtrack tile loop `break`ed for tiles entirely left of `xmin`, correctly skipping
  the *walk* but also the **cubic `dst` write** — every column below `xmin` kept
  uninitialized/recycled memory. Hence the structured-mask trigger (needs a whole
  32-column tile skipped) and the all-zero/all-white/noise-mask cases being
  deterministic. Fix: write the vertical cubic for such tiles with the whole
  workgroup and `continue`. Perf-neutral.
- Agreement with eedi3vk2 over 4 real 4K frames: 22758 → 5902 → **3** differing px,
  max 1 LSB; real-chain determinism 3-5 distinct outputs in 8 runs → **8/8
  identical**. `test_eedi3_mclip_long_mask_off_prefix` (black LEFT half) is
  mutation-verified. **Lesson: a skip justified by what a region *reads* must be
  checked for what it *writes*.**
- Medians of 3 x 2000 f: u16 **518** vs vszipcl 200 = 2.59x; fp32 **306** vs 202.

### Rounds 17-19 — EEDI3H native, then host-path tuning

- `std.Transpose` ≈ **0.91 ms/pass** at 2x2160p u16 (16.6 MB read + 16.6 MB write,
  ~36 GB/s CPU memory path); the four passes were essentially all of EEDI3H's 2.16x
  penalty. Ceiling if free: ~420-470 fps. Two traps: unused `std.Transpose` nodes
  are **pruned** (a 4-node chain measured identical to a 2-node one), and dropping
  `sclip` as a proxy for fewer transposes is invalid — it flips `HAS_SCLIP == 0`,
  making the row kernel compute `cint` everywhere (258 vs 489 fps).
- **Option A implemented** (round 18): the same EEDI3 pipeline on the transposed
  plane, so row/vcheck/vcopy/pad are untouched and the transpose oracle still passes.
  `Eedi3HCreate` is a thin wrapper with `d->horiz = true`; kernel dims are the source
  dims swapped; `dh` doubles the transposed height (= output *width*), so `out_w`/
  `out_h` are tracked separately. Interleaved: EEDI3 501.9, old EEDI3H 233.2, **new
  EEDI3H 360.9 = 1.55x**; penalty 2.29 → 0.76 ms. Holds across configs (mdis 3
  1.68x, mdis 20 1.56x, mdis 40 1.24x, vcheck 0 1.45x).
- Round-18 findings that cost time: ordinary stores into uncached ReBAR VRAM are
  catastrophic and the packed dilation bits were landing there — moving them to the
  cached staging mirror took vertical 474.9 → 492.2 fps; **`vout` must be
  device-local** (staging 328 → 360 fps); the compose output needed its own binding
  (b10) or moving vout dragged the output to VRAM; the mask gather must key off the
  **mask's** depth, not the clip's; the direct-to-frame import re-measured far worse
  than round 13 (−59%).
- **Round 19 host-path tuning (+15.2%, order-reversed 6 reps x 1000 f):**
  fused mask path (`gather_mask_bitmat`, **+6.6%**) thresholds 16 mask elements into
  16 predicate bytes, transposes the 16x16 tile in registers, and movemasks straight
  into the transposed bit row (~17 MB vs ~29 MB); fused kept+interp gather
  (`gather_columns_pair`, **+5.3%**) exploits based_aa's `Interleave([clip, clip])`
  handing out the identical plane pointer, so both parities come from one 64-byte
  chunk; aligned parity selects (`deint_row_*`/`nz_bytes_*`) load from element 0 and
  select parity in the deinterleave, so `first == 1` no longer straddles two lines.
  Traps: **loop order matters more than byte count** (k-outermost keeps 64 row
  streams open and re-reads the mask frame per block, 2x slower), and the mask cost is
  the READ not the ALU.
- Round-19 dead ends: CPU merge of the kept columns (`compose_tight`, −4.8% over 8
  order-reversed reps); k-outermost mask tiling (2x slower); **ablations that change
  the data are not ablations** (`NOMASKX` showed +33% by keeping stale bits).
- **Why EEDI3H cannot be closed further**: the mask read is 16.6 MB vs 8.3 (a
  transposed plane needs all `height` rows) and compose+blit is 49.8 MB vs 33.2 (the
  output must be assembled from transposed interp values *and* kept columns). Every
  rearrangement — CPU merge, spread stores, SDMA D2H, `vout` in dev_buf, host import
  — measures worse or neutral in situ. The way out is the fused chain: EEDI3AA.
- Method: with a fixed variant order an **inert knob moved 5%** — session drift
  exceeds the effects. Reverse the order on odd reps; grade only those.

### Round 20 — pbt-packing / walk-chain levers are DEAD; probe harness was broken

- **Harness bug**: `ENTRY_ROW`'s guard was `#if PROBE >= 3` instead of `== 3`, so
  levels 4-7 returned immediately — a **1656-byte empty kernel** vs 43720 B for
  PROBE=0. All round-15 `PROBE=4/5/6` numbers are void. Fixed; the shipping PROBE=0
  binary is byte-identical. New levels: 7 = real walk + an extra discarded walk over
  the previous row's pbt (f-dependent), 8 = same with an f-independent address,
  9 = DP with every column stored to the same 41 bytes, 10 = 8 with a 4x unroll,
  12 = faithful no-row-kernel.
- Current vertical cost model (3 interleaved reps, within-arm spread <0.5%):

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
  *slower* because repeated writes to one dirty line serialise.
- **EEDI3AA row kernel ~25%, not 35%**: `PROBE=2` returns with `dmap` stale, so the
  vcheck takes its `dirc == 0` fast branch (real 98.19 vs PROBE=2 150.39 = 1.53x;
  faithful **PROBE=12** 89.76 vs 72.00 = 1.25x). Use PROBE=12 for row-kernel
  ablations.
- **The environment is bimodal**: the same binary and vpy measured EEDI3AA 97.7 /
  90.9 / 79.8 fps and vertical 530/495/451/410 while sclk 2304-2338 MHz, `mclk`
  1249, `fclk` 2000 stayed pinned, `gpu_busy_percent` 100, junction 86-90 C, vspipe
  CPU <0.3%, MemAvailable flat. Not clocks, thermals, host CPU or harness memory.
  **Effects under ~5% are not measurable here.**

### Round 21 — vcheck verdict REVERSED; EEDI3AA compose fused into VRAM

- **The vcheck default flips to the global-read form.** The LDS ping-pong removes
  one global read per `dirc != 0` pixel, but the LDS walk must visit **every** row and
  ends with an extra full-width flush, while the global form skips fully-masked rows
  via `ENTRY_VCOPY` — and ~2/3 of AA rows are fully masked. Order-reversed medians
  (1500 f, ns=8, 4 reps): EEDI3 vertical 498.3 → **513.6** (+3.1%), EEDI3H horizontal
  340.9 → **378.4** (+11.0%), H/V ratio 0.684 → **0.737**. `VCHECK_LDS` now defaults
  to 0; the LDS pipeline is still built and selectable with `VSFEEL_EEDI3_VCLDS=1`.
  Zipcl's H/V ratio is ~0.91; the reversal is row-mask-density dependent.
- **EEDI3AA horizontal compose merges in VRAM** (+3.0%, bit-exact). It used to
  assemble two full horizontal planes into staging (16.6 MB PCIe write each) and have
  the CPU 50/50-merge them; `NOCOMPOSE` priced the two dispatches at **+14%** of the
  frame. `ENTRY_COMPOSE` gained a `comp_fuse` push constant: sub-pass 0 writes its
  assembled plane into a device-local `o0` region, sub-pass 1 reads it back,
  assembles O_1, averages exactly like `std::Merge` (`(a+b+1)>>1` u16, `0.5a+0.5b`
  f32) and writes the merged plane to staging, so the CPU does a plain row blit.
  89.7 → 108.8 fps (+21.3%) with both round-21 changes.
- The remaining EEDI3H gap at ns=8 (0.69 ms) is compose 0.31 + blit 0.28 + ~0.1
  host. The compose is a 16.6 MB transpose writing directly to host memory at the
  PCIe line rate; a VRAM write + SDMA D2H moves the same bytes at the same rate.
  **Zipcl is not a counterexample**: its absolute horizontal overhead is 0.48 ms vs
  0.67, but its vertical baseline is 2.5x slower so the ratio looks worse.
- Re-confirmed dead ends: `COPY=7` (cached blit load instead of NT) horizontal
  387.4 → 377.1; per-frame host-pointer import still catastrophic; `QUEUES=8` still
  wins (526.8/394.5 vs 519.9/387.2 at cap 4); `NOPAD`/`NOXPOSE` remain unusable as
  cost ablations because they change the row kernel's input and branch mix.

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
  VRAM window. Predicate dropped; `nt` alone selects `_mm256_stream_*`. **Perf-neutral
  in practice** (AA geometry, 630-px luma, EEDI3H field=3, mclip, 800 f, ns=8,
  interleaved: same-order pairs +0.4%/+0.6%) — full-width AVX stores coalesce in the
  WC buffer, unlike the narrow scalar stores behind the old "27.6 fps" reading.
  Bit-identical output at 630/638/640 px.
- **24 four latent defects**: `VkPipeline destroyed[8*3]` had zero headroom
  (EEDI3AA's worst case is exactly 24) → `std::vector` + `std::find` dedup;
  `import_plane_host_memory` bound at `addr & (align-1)` without querying the
  imported buffer's `VkMemoryRequirements` → now queries and falls back to the CPU
  blit on a short region or bad `memoryTypeBits`; `field > 1` doubled `numFrames`
  unconditionally, turning a `-1` unknown-length source into `-2` → `if (numFrames >
  0)` (also in NNEDI3); `RAWSTAGE=1` alone was garbage (needed `NOREBAR=1`; it now
  selects the DMA path itself) and `NOBLIT` was a silent no-op on EEDI3AA → now a
  create-time error.
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

### Round 27 — the vcheck is parallel by default; a Jacobi ladder controls drift

- **The serial row walk is gone from the shipped path.** Row r takes `d2p` from the
  *un-vchecked* dst row r-1, so no vcheck write feeds another row: the pass runs one
  workgroup per interp row (`grid (1, rows, 1)`) and needs no vcopy. Same-session
  A/B, 5 order-reversed 2000-frame pairs on the honest path (ns=8): EEDI3 **497.96 →
  628.68 fps, +26%**; EEDI3AA (1000-frame reps) **108.3 → 160.2 fps, +48%**.
- **The drift is a blend decision flip, and extra Jacobi steps decay it
  geometrically.** At `a == 1` the output is `cint_s` whatever `d2p` is, so feeding
  the predecessor row's own (one-step-less) parallel value reproduces the serial value
  at those pixels. Level N = N-1 extra steps. Drift vs the serial walk, real jpbd
  frames 300-305, benchmark config:

  | level | EEDI3 fps | u16 max (of 65535) | fp32 max |
  |---|---|---|---|
  | serial | 498.0 | — | — |
  | 1 | 631.3 | 4157 on 0.072% px | 0.088 on 0.108% px |
  | 3 | 635.7 | 580 on 0.018% | 0.021 on 0.037% |
  | 5 | — | 36 on 0.003% | 0.0017 on 0.012% |
  | 6 | 628.7 | 5 on 0.0007% | 2.8e-4 on 0.006% |

  (The fps column mixes two adjacent sessions; the vertical path pays nothing for the
  steps, EEDI3AA ~5 points.)
- **Default is level 6**, the lowest that is bit-exact on the whole tested surface:
  all 236 EEDI3/EEDI3H/EEDI3AA tests pass unmodified, u16 vs eedi3vk2 stays 0 and
  fp32 stays at the serial ulp on the noise clip. Level 3 is the first that is not.
  `VSFEEL_EEDI3_VPARA=0` restores the serial walk as the A/B control; 1..6 pin a
  level. `test_eedi3_parallel_vcheck_matches_serial` pins the two arms together.
- The ladder costs no extra dispatches: one shared `vc_blend` and a macro-generated
  driver per d2p source. The LDS form is untouched and still selected by
  `VSFEEL_EEDI3_VCLDS=1`; the parallel level wins when both are set.

### Round 28 — EEDI3AA's two horizontal submissions fused into one

Both AA horizontal sub-passes read only the merged frame, which the first fence has
finished, so they need neither the host nor a second fence between them. One CB, one
`submit_with_fence`, one wait; both parities are pre-gathered with the
`gather_columns_pair` form so the single submit has both uploads, and the mask bits
(parity-independent) lose their duplicate build. `num_streams` unchanged.

- ns=1 same frame: **21.63 → 19.47 ms**; second fence window 6.05 → 5.17. The hWait
  drop is only ~0.9 ms, not the 6 ms the model predicted — most of it is compute the
  GPU still has to do. ns=8, interleaved order-reversed 2000 f: 141.3 → 144.8 (6
  pairs), 132.4 → 144.5 (5 pairs); fused won or tied every pair. A few percent, not
  10%.
- Bit-exact vs the two-submission control, still in-tree as `VSFEEL_EEDI3AA_HFUSE`:
  15 GRAY geometries x u16/f32 + 7 multi-plane cases, 0 mismatches, including the
  fused GPU merge (`comp_fuse` == `_mm256_avg_epu16`).

Bounded out on the way: pre-recording the CBs (`record` 0.01-0.02 ms/frame =
≤0.07%); the pair gather alone (neutral, 21.933 → 21.881 ms); not building AA's
vertical `xpose`/`compose` (premise false — AA's *horizontal* sub-pass dispatches
`xpose`, and `create_pipeline` with a null module returns a pipeline that faults at
dispatch, not creation).

### Round 29 — row-kernel levers: SGSIZE 64 is +2% on AA only; the walk is not the lever

`VSFEEL_EEDI3_GBENCH=1` stage marks (`s0..s4` = pass start/xpose/pad/row/vcheck): row
kernel **s2 = 3.485 of 3.86 ms** at 1080p ns=1 — ~90% of the GPU frame.

- **Tuned targets yield to the budget** — the batch target and NLMeans' u4a ring
  are measured optimums, now capped by the core's VRAM limit; no perf change here.
- **SGSIZE 64 / K=1: +2% on EEDI3AA, neutral on EEDI3, not shipped.** EEDI3 585.9 →
  580.7 (ns=8, 2000 f, 4 pairs); EEDI3AA 142.1 → 145.8 and 142.8 → 145.6 (two runs,
  4 pairs each), the 64-lane arm winning 7 of 8. Not shipped: ~2% on one workload is
  at this box's resolution, for a compile-time `-D`, a second row module, a second
  pipeline cache and device-limit handling.
- **The walk has no chain to fix.** `idx` is already absolute in the DP; only the
  store relativizes it (`cRel[j] = idx - u`), so storing `idx` already makes the
  walk's loads f-independent — exactly PROBE 8, which ties PROBE 7 (451.4 vs 453.9)
  and PROBE 10 (457.0). Issue/sector-bound, not latency-bound. Walk+interp =
  **0.241 ms = 11.9%** of the vertical frame.
- **Row kernel** (`RADV_DEBUG=shaderstats`): 96 VGPRs, 0 spills, 3609 instructions,
  Latency 71374 vs InvThroughput 10659. At 96 VGPRs only ~2-3 waves are resident, so
  latency has nothing to hide behind — and nothing spills, which is why the K=1 shape
  bought nothing. 376 `s_waitcnt` with **171 inter-wait gaps of 2 instructions**; the
  137 `v_dual_*` pack fine, so it is dependency, not scheduler. 485 `v_cvt_f32_i32`
  are a red herring (not issue-bound).

## Open work

- **Row kernel register pressure**: `RING_CAP` is not it. Trimming the fixed
  bound to `2*NRAD+1` leaves the compiled row kernel **byte-identical** at
  nrad=1/2/3 (VGPR 96/96/120, 16/16/12 subgroups/SIMD, no spills): the
  `if (k >= RN) break` guard is dead once the driver specializes `NRAD`, and the
  ring is register state, not LDS (1 024 B). At nrad=3 `RING_CAP == RN` exactly,
  so the remaining lever is the `K=2` direction state. (Spec constants *can* size
  arrays here — `notes/NLMEANS.md` ships it.) ACO raises VGPRs deliberately for
  load ILP, so a lower count with new spills or schedule damage is a regression.
- **Split the walk into its own dispatch, one lane per ROW.** Today one lane of a
  32-lane workgroup runs the walk while the kernel is issue-bound, so this is a
  *lane-utilisation* change, not a chain fix (round 20/29). Ceiling 11.9% of the
  frame against a ~5% measurement floor, so expected value is a point or two. Shape:
  the row dispatch is `grid (1, rows, 1)` with rows on the grid's y, so a row-per-lane
  pass cannot ride it — it needs `grid (ceil(rows/SGSIZE), 1, 1)` with `row =
  gl_GlobalInvocationID.x` and no barriers, publishing directions to a global buffer
  the interpolate pass reads instead of `tileF`, plus re-deriving the `xmin` and
  rightmost-column special cases. Gate on the bit-exact oracle before measuring.
- **SGSIZE 64 / K=1** — +2% on EEDI3AA only; not shipped (round 29).
- `ENTRY_PAD` div/mod by `pad_stride`, `ENTRY_VCOPY`/`ENTRY_BLIT` div by `WIDTH`: a
  2D dispatch removes them. Per-plane passes, not the row kernel; unmeasured.
- BT_TILE as a spec constant swept {16,32,64}; hoisting the duplicated `cubic_float`
  evaluations; fixed-per-invocation floats as spec constants; parallelising the serial
  span scan. All bounded by the row kernel's ~17% share.
- **Fuse the vcheck across planes** (one launch, `gl_WorkGroupID.y = plane`, as
  vszipcl does) — YUV-only, cannot move the Gray flagship, real for colour AA.
- Re-run the host/GPU ladder before assuming which side binds: after round 28 the
  ns=1 host stages are ~19.5 ms against ~55.9 ms/stream end-to-end (ns=8), so compute
  dominates now.
- **Accuracy-for-speed items are void on mechanism**: fp16 pad/cost storage has no
  traffic to remove (pad is native u16, DP costs live in registers); adaptive
  `nrad`/`mdis` trades accuracy for ~0; dropping the sclip upload when the caller
  passes the same clip is undetectable through a node-pointer test. Any accuracy
  relaxation still requires measuring the drift on the noise clip and updating
  `tests/test_eedi3.py` in the same change.

## Do-not-retry (mechanism, not verdict)

*(stale)* marks a verdict measured on the degenerate pre-round-14 config — kept as
the reason a variant failed, not as a current number.

- **All round-15 `PROBE=4/5/6` numbers are void**: those levels compiled to an empty
  row kernel (1656 B vs 43720 B), which is why "store nothing", "walk another row"
  and "skip the backtrack" all land within 1.4% of each other.
- **Breaking the walk's serial load chain** by lookahead, ±1 candidate windows, or a
  layout making the address f-independent: 453.9 / 451.4 / 457.0 fps — identical —
  and PROBE=6 is a dead tie with real (532.6 vs 530.6). Issue/sector-throughput
  bound, not latency bound.
- **2-bit `pbt` packing**: DP+store is 4.8% of the vertical frame and the 170 MB
  store runs at ~1.9 TB/s (L2/port, not DRAM). PROBE=9 proves bytes are not the cost
  — 1/40th the traffic on one dirty line is *slower*. Packing also adds 4
  `subgroupBallot`s + a vector store to an issue-bound kernel.
- **4x-unrolling the walk with merged `tileF` stores**: bit-exact and 56/56 tests,
  but a wash over 8 order-reversed 2000-frame pairs (median ratio 0.99).
- **Ablating the row kernel with `PROBE=2` or `PROBE=5`**: they return with `dmap`
  stale, so the vcheck takes its `dirc == 0` fast branch and the row kernel is
  overstated by ~9 points on EEDI3AA. Use `PROBE=12`.
- **Splitting DP and backtrack into two dispatches** (round 15): neutral (vc0 322 vs
  326, vc2 295 vs 295) — the boundary costs exactly what removing the same-WG
  store→load dependency saved. The open-work item splits to *parallelise the walk*, a
  different thing.
- **Per-frame `VK_EXT_external_memory_host` imports** (`VSFEEL_EEDI3_DSTHOST=1`,
  default off): the import is GPU work on RADV (VM map/unmap + TLB sync per 16.6 MB
  BO), and 8 streams saturate the GPU on mapping. Arms: staged 608 | `vout` in
  dev_buf no import 588 | import the GPU never touches 487 | import+blit 390; leaking
  imports is worse (269). Re-test only if RADV's import path gets cheaper.
- **Multi-region `vkCmdCopyBuffer` for strided row copies** (~13 us *per region* in
  situ), and **`vout` in dev_buf + SDMA D2H** (`VSFEEL_EEDI3_VOUTDEV=1`, still a
  loss: 289 vs 313-323; 327 vs 375 with vcheck off).
- **Lazy `cint`/`sclip` load in vcheck**: void by inspection — `cint_s` is both the
  `dirc == 0` result and the `dirc != 0` blend operand, so every `do_row` pixel reads
  it.
- **`LDS ping-pong tlineSh`**: the verdict has flipped twice (round 14 +3%, round 21
  +11% the other way) and both sides are real — it is workload-dependent on row-mask
  density, so re-measure rather than inheriting either verdict if the mask mix changes.
- **(stale) batch**: a single contiguous blit instead of per-row strided copies (525
  vs 555); plain `memcpy` instead of the NT load/store pair anywhere in the frame path
  (blit 6.1 → 10.4 ms/frame; vc0 481 → 401); skipping the pbt global stores
  (`PROBE=1`); a 2-subgroup x 2-row 64-lane workgroup for `ENTRY_ROW` (regressed in
  every config; NOT the untried one-subgroup 64-lane / K=1 shape); `subgroupcoherent`
  on pbt (259 → 229); `restrict` on pbt (214 → 201); `[[unroll]]` on the K-direction
  loops (227 → 196); dropping `HOST_CACHED` from staging (collapsed to 43 fps — NT
  loads need WB/WC memory).
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
  not clocks, thermals, host CPU or harness memory). Grade the median per-pair ratio
  over >=5 order-reversed 2000-frame pairs (`tmp/probeab.py`, `tmp/probeabaa.py`);
  treat a single-pair <5% claim as unproven. The vertical arm is *not* less noisy than
  EEDI3AA — it looked stable for an hour, then swung 20%.
- **Never compare a number recorded in a different command.** The same committed
  binary measured 264 fps early in round 15 and 290 later; interleave A/B in one
  session (`tmp/ab.sh`), or use modal outputs over many runs when comparing *outputs*.
- **Check `CMAKE_HOME_DIRECTORY` in a second build tree before trusting an A/B against
  it**: a copied/worktree `CMakeCache.txt` can still point at the main checkout, so
  `tools/install.sh` silently rebuilds the main tree and both arms are the same binary.
- **An isolated microbenchmark proposes; only an in-situ same-session A/B decides.**
- **Never use timestamp queries for kernel attribution here** — the env-gated ablation
  ladder answered host-vs-GPU faster and more robustly; an early query-pool attempt
  hung the queue. (`GBENCH` is the one exception, and only for stage boundaries at
  ns=1.)
- **Prove the stage split with the ablation ladder, not theory**: the
  `VSFEEL_EEDI3_*` opt-outs (`NOBLIT`, `NOSCLIP`, `NORAW`, `NOH2D`, `NOVC`, `NOPAD`,
  plus `NOREBAR`) at ns=8, and `HBENCH` + `HFRAME` for the host split. Additive costs
  of similar size mean a shared memory path; one dominant item means a loop.
- **An ablation that removes a producer must reproduce what the consumer reads**
  (`PROBE=2/5` leave `dmap` stale → cheaper vcheck → overstated row kernel). Same
  class as round 19's "ablations that change the data are not ablations".
- **Before trusting any `PROBE` number, `stat -c %s build/vk_spv/eedi3_16_row.spv`**:
  round 20 lost a whole round of recorded numbers to a `>=` guard compiling 4-7 to an
  empty kernel.
- **The benchmark's EEDI3 mask follows the clip's depth, like based_aa**; both depths
  now take the same no-conversion path, with one deliberate documented divergence
  (floats above ~8.42e6).
- **A skip justified by what a region *reads* must also be checked for what it
  *writes*** (round 16.2's structured-mclip bug). Regression test
  `test_eedi3_mclip_long_mask_off_prefix`, mutation-verified.

## Debug env vars (verified against `src/eedi3.cpp`)

- `VSFEEL_EEDI3_VCLDS` — `=1` forces the LDS vcheck ping-pong back (global is default).
- `VSFEEL_EEDI3_VPARA` — vcheck form: 0 = serial row walk (A/B control), 1..6 =
  parallel with that many Jacobi steps. Default 6 (bit-exact on the test surface).
- `VSFEEL_EEDI3_BATCH` — output frames recorded per submission (default: the
  measured target capped by the core's VRAM limit, ~1 GiB of scratch clamped 2..8,
  4 at 2x2160p; 1 when the budget cannot hold two frames). `=1` is the A/B control.
- `VSFEEL_EEDI3_TRACE` — one-shot banner: VRAM accounting, per-plane region layout,
  spec constants per geometry.
- `VSFEEL_EEDI3_TIMING` — per-frame host stage split (acquire/alloc/record/submit).
- `VSFEEL_EEDI3_SYNC` — additionally wait each submission out and report its wall time
  (serializes the pipeline; for GPU-time measurements at depth 1).
- `EEDI3_PROBE` — CMake cache var: ablation level 0/1/2/3/4/5/6/7/8/9/10/12 (11, 13+ unused).
- `EEDI3_MAXW` — CMake cache var: LDS vcheck max width -> `-DMAXW` + `-DEEDI3_MAXW_LDS` (default 4096).
