# EEDI3 — implementation, accuracy landscape & port plan

Status: **shipped and tuned through round 18; EEDI3H is now native (transposed
staging + two GPU passes), 1.55x faster than the old composition.** This is the
single note for this filter. It is ordered in rounds — earlier rounds are the
port and the history, the last rounds are the current truth — and it ends with
the short-form "remaining work / do-not-retry / method rules" section that is
the first thing to read before touching EEDI3 again. Jump to `## Round 16` for
the cost model and the fixed correctness bug, `## Round 17` for the original
EEDI3H transpose analysis, and `## Round 18` for what was implemented and what
is left.

## References available

| ref | type | EEDI3 funcs | mclip | io convention |
|---|---|---|---|---|
| `eedi3m` (HolyWu, ORIGINAL) | C++ | EEDI3 | per-plane same-fmt→8bit Point conv | **native int scale**, params ×(1<<(bits-8)); f32 beta/gamma/vthresh ÷255; alpha not scaled; cost3 → alpha÷3; **double intermediate in DP relax** |
| `eedi3vk2` | Vulkan GLSL | EEDI3 | same as clip (planes[]) | native int scale (same as eedi3m); f32 `precise` DP |
| `vszip` | CPU Zig SIMD | EEDI3, EEDI3H | single Gray→Gray8, drives all planes | native int scale; alpha÷3 beta÷255 gamma÷255 etc; NO per-column clamp |
| `vszipcl` | OpenCL | EEDI3, EEDI3H | none | **u16 normalized to [0,1]** (÷65535); f32 identity; mirror-pad full mdis |
| `vszipcu` | CUDA/HIP | EEDI3, EEDI3H | none | u16 normalized [0,1] (×1/65535); same structure as vszipcl |

## Empirics (real clip /home/encode/test/jpbd.mkv frame 100, 1920x1080, field=1)

f32 max abs diff (interp rows only; kept rows identical):
- vszipcl vs vszip ≈ 1.2e-7 (same semantic family, ulp-level)
- eedi3m vs vszipcl ≈ 0.067 on 1e-4 of pixels; eedi3vk2 vs vszipcl ≈ 0.012
- eedi3m vs eedi3vk2 ≈ 0.067 on 1e-4 px... (see u8 below: they agree on 99.997%)

u8 max LSB diff:
- eedi3m vs eedi3vk2: 17 on 0.003% pixels   (≈ same family)
- eedi3m vs vszip: 17 on 3.3% pixels
- eedi3m vs vszipcl: 17 on 3.1%

**Conclusion: two semantic families.**
- Family A = eedi3m + eedi3vk2: per-column direction clamp `umax=min(x,W-1-x,mdis)` (tables only cover |u|≤umax at column x), DP relax over v∈[max(-umax2,u-1), min(umax2,u+1)] where `umax2=min(x-1,W-x,mdis)`; horizontal pad mirror MARGIN_H=12 (only ±3 taps + direction stays ≤ distance-to-edge, so 12 suffices); pbackt stores ABSOLUTE predecessor directions (not deltas); interpolate guard `x>=|3dir| && x<=W-1-|3dir|` else avg2; tie-break = ascending v, strict < (left-most wins).
- Family B = vszip/vszipcl: full direction set at every column via mirror pad of ~3*mdis+nrad+8; pbackt DELTAS; center-pref first tie-break.

u16 native-scale DP costs exceed 2^24 (gamma*256 accumulates over width) → f32 running path costs lose exactness; even eedi3vk2(f32 DP) vs eedi3m(f64 DP) flips ~0.01% pixels at u16, a few LSB. vszipcl's [0,1] normalized u16 is a THIRD numeric domain (rounds differently) → thousands of LSB vs family A.

## Target decision

Ground truth = **eedi3m** (user); eedi3vk2 demonstrates family-A + f32 DP ≈ eedi3m everywhere (u8 0.003%, u16 0.01% px flips). So vsfeel implements **family-A semantics** with an f32 DP, matching eedi3m's combine + relax order. Measure against eedi3m primarily, eedi3vk2 second.

u16: vsfeel runs native int scale (like eedi3m/eedi3vk2), samples as u16 in the pad, cost SADs as int32, combine in f32. Accept tiny flip fraction vs eedi3m; target ≈ eedi3vk2's agreement level.

## eedi3m key code facts (from EEDI3.cpp / EEDI3.h on GitHub)

- MARGIN_H=12, MARGIN_V=4; pad frame = (W+24) x (dstH+8), native sample type, kept-parity rows at 2-row pitch, mirror margins both axes.
- validate: field 0..3; !dh ⇒ every processed plane's height even; dh⇒field≤1; alpha,beta∈[0,1], alpha+beta≤1; gamma≥0; nrad 0..3; mdis 1..40; vcheck 0..3; vthresh*>0 if vcheck>0; planes dedup; mclip must isSameVideoInfo + same numFrames; convert mclip to same-family 8-bit via std.Expr(x 0.5-) only when YUV float, then resize.Point; sclip only validated when vcheck>0 (same videoinfo+frames). opt 0..3.
- cost (cost3, int pixel): umax=min(x,W-1-x,mdis); u in [-umax..umax]; u2=2u; s0,s1,s2 int; s1 only if (u>=0&&x>=u2)||(u<=0&&x<W+u2); s2 only if (u<=0&&x>=-u2)||(u>=0&&x<W-u2); s1=s2=s0 fallbacks (`s1>=0?s1:(s2>=0?s2:s0)`, float: >-FLT_MAX). ip=(src1p[x+u]+src1n[x-u]+1)/2 int. ccost=tpitch*x+mdis+u.
- DP col x=1..W-1: umax as above; umax2=min(x-1,W-x,mdis); u in -umax..umax: idx=0,bval=FLT_MAX; for v=max(-umax2,u-1)..min(umax2,u+1): z=(double)ppT[mdis+v] + (double)(gamma*|u-v|)  [gamma float; product in float]; ccost=min(z,FLT_MAX*0.9) f32; if ccost<bval {bval=ccost; idx=v}; pT=min((double)bval + (double)tT[u], FLT_MAX*0.9)→f32; piT[mdis+u]=idx (absolute v of PREVIOUS column). bmask !bmask[x]: x==1 ⇒ copy tT costs & zero piT; else copy ppT row & prev piT row + fixups at mdis∓pumax (pumax=min(x-1,W-x)); pumax<mdis: piT[mdis-pumax]=1-pumax; piT[mdis+pumax]=pumax-1.
- backtrack: fpath[W-1]=0; fpath[x]=pbackt[tpitch*x + mdis + fpath[x+1]] for x=W-2..0 — note pbackt col x = piT written during column x+1's DP.
- interpolate: bmask&&!bmask[x] ⇒ cubic 4tap or avg2, dmap=0. else dir=fpath[x]; if ucubic && x≥|3dir| && x≤W-1-|3dir| ⇒ cubic on rows at ±dir,±3dir else avg2 ±dir. int rounding (9*(...)+8)/16 clamp[0,peak]; float exact.
- vCheck: per interp line top→bottom; gate `y>=6 && y<height-6` (dst coordinates on pad rows: y iterates pad-row index MARGIN_V+field+2r... careful: y is the pad row; uses dstp offsets into the OUTPUT frame). reads dst2p=dstp-2rows (already-vchecked prior), dst2n=dstp+2rows (not yet), src±3 from PAD rows, dmap[x±W]. cint = sclip or cubic on (dst1p,dst1n,dst3p,dst3n) rows. Blend a from mdiff0/mdiff1/vthresh; tline= (1-a)*dstp[x]+a*cint; int cast truncation toward zero; float no clamp. memcpy row back (after computing whole row into tline).
- copyPad row formulas (kept rows 2-row pitch, off=1-field; dh: every row, off as given, pitch 1) + mirror loops top/bottom using MARGIN_V*2-y and y-c.
- copyMask(dh,mclip): mcp rows = mclip rows (dh ? all : every-2 starting `field`); bmask built per interp line r from mcp row r: last-scan dilation with minmdis=min(W,mdis); `last` init -666999 etc.

## eedi3vk2 extra / differences (for cross-check)

- requires subgroup shuffle; per-row two-subgroup WG, no WG barrier per column; uses rolling sums (non-mclip) vs full recompute (mclip); bt tile = 32 (16 when hp&&!mclip); scaledGamma=gamma*(hp?0.5:1); interpolate hp odd-dir uses pairs; vcheck keeps ping-pong tline in shared, quantized (int) rows.
- hp handling fully implements half-pel (dirs u in [-2mdis..2mdis] in half-pel units, rows hp1p... precomputed via hpfill kernel cubic4 0.5625/0.0625).

## vsfeel environment notes

- only u16 (int) and f32 inputs. devices: RX 7900XTX RDNA3.
- build: cmake+glslc → spirv_binaries.h via src/gen_spirv_header.py (list args + emit name), CMakeLists adds custom commands. install: cp build/libvsfeel.so → $(vapoursynth plugins)/vsfeel/. MUST rebuild+copy before tests/bench.
- pool plumbing (vsfeel.h): FramePool<T> ticket semaphore init `pool.semaphore.current=num_streams-1`, reserve/push one per stream; destroy_common; submit_with_fence (serialize on queue lock); checkVK returns set_error.
- record_command_buffer pre-recorded per resource; plane config dedup; staging upload/download, sometimes kd_download/host-direct. Tests pattern in test_dfttest.py; noise clip 640x360 24f.
- benchmark/bench.py FILTERS registry + build fn per plugin.

## vCheck exact semantics (EEDI3.h, family A) — REVISED

vCheck is called once per plane AFTER all interp rows were interpolated (dst
holds the full interp set, incl. masked rows = cubic). `srcp` = pad row ptr +
srcStride*MARGIN_V... precisely: in filter_avx2, `_srcp += srcStride * field` right
before vCheck, and vCheck's srcp arg = `_srcp` (which had been advanced by
`srcStride*MARGIN_V` before the row loop) + srcStride*field... At vcheck entry in
EEDI3.h: dstp = dst row `field` (the interp base), srcp = pad ptr + MARGIN_V+field
rows... loop `for y = MARGIN_V+field; y < height-MARGIN_V; y+=2` over PAD rows
(so dst row dy = y - MARGIN_V increments by 2). Gate `y>=6 && y<height-6` on the
PAD row index (pad height = dstH+8), i.e. dst row dy = y-MARGIN_V in
[6-MARGIN_V=2, height-6-MARGIN_V = dstH+8-6-4 = dstH-2) => dy in [2, dstH-3]
(effective). Inside the gate for row dy (an interp dst row):
- dst3p/src3n are PAD rows at MARGIN_V+(dy±3)+... (kept rows: dy±3 is kept
  parity); dst2p = dst row dy-2 (the PREVIOUS interp row, ALREADY vchecked),
  dst1p = dst row dy-1 (a KEPT row = source copy), dst1n = dst row dy+1 (kept),
  dst2n = dst row dy+2 (the NEXT interp row, NOT yet vchecked).
- cint = sclip? scpp[x] : cubic over (dst1p,dst1n,dst3p,dst3n) [vertical 4-tap].
  For u16 the cubic is (9*(dst1p+dst1n)-(dst3p+dst3n)+8)/16 CLAMPED [0,peak];
  floats unclamped. NOTE: dst3p/dst3n here read PAD rows = the SOURCE ±3 rows
  (copy of kept row dy±3 of the source). dst1p/dst1n also = source kept rows.
- the pass conditions (dirc==0; max(dirc*dirt,dirc*dirb)<0 || (dirt==dirb==0);
  x±|dirc| out of [0,width)) all store tline[x]=cint directly.
- else it=(dst2p[x+dirc]+dstp[x-dirc]+1)/2, ib=(dstp[x+dirc]+dst2n[x-dirc]+1)/2,
  vt=|dst2p[x+dirc]-dst1p[x+dirc]|+|dstp[x+dirc]-dst1p[x+dirc]|,
  vb=|dst2n[x-dirc]-dst1n[x-dirc]|+|dstp[x-dirc]-dst1n[x-dirc]|,
  vc=|dstp[x]-dst1p[x]|+|dstp[x]-dst1n[x]|, d0=|it-dst1p[x]|, d1=|ib-dst1n[x]|,
  d2=|vt-vc|, d3=|vb-vc|, mdiff0/1 combine by vcheck mode (int (d+d+1)/2 for
  mode 2; float /2). a0=mdiff0*rcpVthresh0, a1, a2=max((vthresh2-|dirc|)*rcp,0),
  a=min(max(a0,a1,a2),1); tline=(1-a)*dstp[x]+a*cint; int store truncates to
  pixel_t; float pass-through (may exceed range). Then memcpy(tline -> dst row).
- So the row dependency is ONLY dy-2 (the previous interp dst row already
  finalized). dmap[x±width] = interp-row r-1/r+1 dmap rows. Rows dy-1/dy+1 are
  the untouched source kept rows (from pad/dst kept rows). dy+2 read only at
  x+dirc and x-dirc (next interp row, not yet vchecked, plain interp value).

This means vCheck over dst can be structured as: two separate passes after all
rows interpolated, OR as a single pass where rows carry the previous row's
final tline in shared and rows dy+2 read the still-plain dst (b1). The dst row
dy is only overwritten after ALL reads of it as a "dy+2" of row dy-2 and as
"dstp" of itself... it cannot be overwritten before row dy+2 was computed
(reads dstp[x±dirc]). Since vCheck processes rows top-down and overwrites row
dy AFTER row dy+2 was read as its "2n"... wait row dy+2 reads THIS row (dy) as
its dstp! So eedi3m CANNOT overwrite row dy before processing dy+2: hence the
memcpy-back per row: row dy's tline is written to the dst only at the END of
the dy+2 iteration... Actually eedi3m writes row dy (memcpy dstp after the full
tline computed) while dstp still points at row dy, THEN dstp += 2 rows. Row dy
is read as dst2p by row dy+2's vcheck AFTER dy was already overwritten (that's
exactly the intended serial chain: dy+2 sees the vchecked dy). But row dy is
also read as `dstp[x±dirc]`/`dstp[x]`/it/ib/vt/vb by vcheck of row dy+2 and as
`dst1p/dst1n`?? NO dst1p/dst1n are KEPT rows (dy±1 of the current row) = the
source's rows; kept rows are never vchecked. And row dy is read as dst2p[x+dirc]
by row dy+2 (already vchecked — desired), and dy+2's own dstp[x±dirc] reads row
dy+2 (itself, plain) — good. So per-row in-place is fine EXCEPT that reading
row dy+2's tline needs dst rows dy+2's CURRENT (plain) interp values while row
dy was already overwritten. In-place top-down with a shared tline ping-pong
works: maintain tlineSh of the previous row only; current row computed from:
prev-tline (row dy-2, vchecked), dst rows (row dy plain for self/it; row dy+2
plain for 2n), kept pad rows dy±1,dy±3. Write row dy only after row dy+2 is
done reading it? Row dy+2 reads row dy as dst2p → must be the VCHECKED row dy.
So ordering: iterate r; compute tline[r]; THEN when we move to r+1, we may
overwrite dst row of r-1 (that is r+1's dst2p needs vchecked row r? wait
dst2p of row r+1 = dst row (dy+2)-2 = row r — yes vchecked row r; and row r+1's
own dstp = row r+1 plain (not yet vchecked); it/ib read row r+1 (plain) at
x±dirc and row r (vchecked) at x+dirc for it etc. Actually for row r+1:
dst2p = row r (vchecked tline), dstp = row r+1 (plain, current), dst2n = row
r+2 (plain, next). So we must keep row r's vchecked tline in shared until row
r+1 is done; row r may then be written back to dst. That matches my shared
ping-pong structure. One caveat: 2n reads at r+1 need row r+2's PLAIN dst
values (b1 before vcheck writes). Since we only write rows r-1 and earlier back
to dst, row r+2 is always still plain when read. And r+1's own dstp (self) is
still plain (not yet written). 

Correct.


## IMPLEMENTATION STATUS (round 1 working)

Host: src/eedi3.cpp (EEDI3VK namespace com.thefeeltrain.vsfeel), shaders src/eedi3.comp
(ENTRY_ROW per-interp-row WG + ENTRY_VCHECK single-WG serial walk). CMake +
gen_spirv_header.py + vsfeel registration done. vsfeel.cpp now enables
Vulkan 1.2 features storageBuffer8BitAccess/uniformAndStorageBuffer8BitAccess/
shaderInt8 via VkPhysicalDeviceVulkan12Features chain.

Row kernel architecture (MVP): one WG per interp row, lsz_row = max(TPITCH, 32)
(the DP owns one direction/lane; backtrack/interpolate tiles are BT_TILE=32
columns wide — an lsz of only TPITCH left columns >= TPITCH unwritten: classic
bug, fixed). Shared parity ping-pong path costs pco[2][TPITCH], barrier per
column. pbt rows (absolute predecessors) written to global, backtracked tile by
tile (BT_TILE=32) in shared, then interpolated by lanes 0..n-1. Column W-1 gets
dir 0 (vertical cubic) via lane 0 before the tile loop.

Measured agreement (noise_24f GRAY16 640x360, 230400 px):
- vsfeel == eedi3vk2 BIT-EXACT on every pixel tested (vcheck 0/1/2/3, several
  mdis/nrad) — max diff 0 over interp rows.
- vsfeel vs eedi3m (AVX2 float DP): <=1 LSB except a tiny float-rounding flip
  set (13 px at mdis5; 147 px at mdis20/nrad3) with large deltas where the DP
  argmin flips direction. eedi3vk2 shows the IDENTICAL flip vs eedi3m — i.e.
  vsfeel reproduces the eedi3vk2 float DP exactly and diverges from eedi3m only
  in the documented f32-DP-vs-f64-DP rounding manner.
- vcheck changes 47 px vs vc0 on this clip and matches eedi3vk2 vc2 exactly.
- f32: eedi3m vs vsfeel max ~0.024 on few px (DP flips), vc2 ~0.024 on 7px.

Tolerances for tests: u16 <= 1 LSB vs eedi3m (allow isolated few-px flips),
f32 few e-2 bounded by measurement; vsfeel vs eedi3vk2 can be EXACT (0 LSB)
for u16 — a much stronger oracle where eedi3vk2 is installed.

## IMPLEMENTATION STATUS (round 2 — final MVP, tests + bench)

Fixes since round 1:

1. **dh / field>1 SEGFAULT — root cause:** `newVideoFrame2` was passed
   `planeSrc = nullptr` when dh, but VS's frame ctor dereferences the array
   unconditionally (`planeSrc[i]` in vscore.cpp) → segfault on the first dh
   frame. Also the dst frame was allocated with the SOURCE height while dh
   doubles it. Fix: always pass a non-null `fr[]`/`pl[]` (fresh-alloc `nullptr`
   entries for processed planes and for every plane under dh; `src` passthrough
   only for unprocessed planes when !dh, dims then match). Result: dh=1
   field 0/1 and field=3 (doubling) all BIT-EXACT vs eedi3m/eedi3vk2.

2. **u16 divergence at nrad>=2 (frame-dependent) — root cause: `ip` rounding
   in conn_cost.** eedi3m's integer path computes `ip=(t1+t2+1)>>1`
   (round-half-up) and `v` as an INTEGER difference; eedi3vk2 INT path is the
   equivalent `floor((a+b+1)*0.5)`. My shader used the plain float midpoint
   `(t1+t2)*0.5`, which differs by 0.5 on odd sums → `v` off by up to 1 → flips
   near-tie argmins (seen as ~70-400 px on frame 11 at mdis20/nrad3 where
   eedi3m==eedi3vk2 but vsfeel differed; caught only because the round-1 probes
   tested frame 0). Fix in src/eedi3.comp conn_cost (BITS==16 branch): integer
   `ip=floor((t1+t2+1)*0.5)`, `v=|c1-ip|+|c2-ip|`. VERIFIED: independent Python
   integer DP with this ip is bit-exact vs eedi3m row 219; shader is now
   bit-exact vs eedi3vk2 on EVERY config incl. previously-divergent ones.

3. **s2 gate:** verified eedi3m AVX2 also uses `x < width + u2` for s2Flag (not
   the corrected `width - u2` that eedi3vk2's connectionCostFull and my shader
   use) — but only on border columns; interior dominates and both refs still
   agree at borders, so the eedi3vk2-corrected gate is fine (bit-exact vs vk2).

Measured agreement after fixes (noise_24f GRAY16/GRAYS 640x360):
- u16: vsfeel == eedi3vk2 BIT-EXACT (max 0) on EVERY shared config: field 0/1,
  dh, mdis 3..40, nrad 0..3, vcheck 0..3, custom alpha/beta/gamma/vthresh,
  sclip=src, mclip (Gray8/16/32 masks → vsfeel converts to Gray8; vk2 wants
  the clip format; both then agree bit-exactly).
- f32: vsfeel vs eedi3vk2 ~1 ulp (7.45e-9) on most configs; a few corners
  (gamma=5 + vcheck) show isolated DP argmin flips ≤ 5.45e-3 (tests use 5e-3).
- vsfeel vs eedi3m residual flip sets (13..1500 px of 115200, max ~3276 LSB)
  are IDENTICAL in count to eedi3m-vs-eedi3vk2 flip sets → not bugs, they are
  eedi3m's f32/f64-DP ordering vs the shared GPU float DP. USER DECISION:
  eedi3vk2 bit-exactness is the bar; eedi3m is only a loose sanity oracle.

Tests: tests/test_eedi3.py (48 tests) — determinism 1/4 streams, multi-stream
== single, parallel-load consistency (16/32), vspipe pipelined no-hang via
bench subprocess (16/32 × streams), all-frames finite/in-range, reference
sweep vs eedi3vk2 in ONE subprocess (u16 exact / f32 tol; crash → skip),
cost3/ucubic loose eedi3m sanity (< 2% px), mclip Gray8 matches vk2 same-format
exactly + auto-conversion g16/g32 == g8 + masked region == vertical cubic,
YUV passthrough chroma bit-identical (16/32), dh doubles height, field>1
doubles frames with kept parity, full validation rejection matrix.
Full suite: 328 passed.

Benchmark: benchmark/bench.py got an `eedi3` FilterSpec (args field/mdis/nrad/
alpha/beta/gamma/vcheck/vthresh0/1/2, num_streams default 4; build covers
vsfeel/eedi3m/eedi3vk2/vszipcl; input depth(get_y(clip),16)). PLUGINS gained
eedi3m/eedi3vk2 entries.

Baseline (real clip jpbd.mkv 1080p, u16, 500 frames, num_streams=4):
- vcheck=2 (like the refs' defaults): vsfeel 276 fps | eedi3vk2 293 |
  vszipcl 555 | eedi3m 78
- vcheck=0: vsfeel 308 | eedi3vk2 398 | vszipcl 774 | eedi3m 76
vsfeel trails eedi3vk2 by ~6-23% and vszipcl by ~2x → SPEED PHASE next
(see plan below).

## Speed phase plan (not yet started)

Architecture today: pad + bmask + sclip are staged host-visible (written via
CPU copy each frame into the mapped staging buffer that the ROW kernel reads
DIRECTLY as its pad/bmask/sclip buffers) and read back via an interp-rows-only
download; the row kernel does per-column workgroup barriers with shared parity
ping-pong path costs (pco[2][TPITCH]); one WG per interp row; int8 pbt written
straight to a DEVICE buffer; vcheck = one serial WG per plane.

Likely wins (in order), modeled on eedi3vk2 + vszipcl:
1. Move pad upload to a DEVICE-LOCAL buffer via vkCmdCopyBuffer (H2D once per
   frame; today the shader reads pad/bmask/sclip straight off host-visible
   staging = PCIe reads in the inner DP loop). bmask can stay host-side if
   only read once per row (cheap) or move too.
2. Replace per-column WG barriers + shared ping-pong with the register +
   subgroup-shuffle DP (eedi3vk2 style: 2 subgroups per WG, K=ceil(TPITCH/
   SG) lanes per thread, subgroupShuffle* to gather the u±1 neighbor window,
   rolling cost sums for non-mclip). This removes ~W barriers per row.
3. vcheck: today one WG serializes the whole row walk; consider per-row WGs
   with a single producer of each row's tline... (eedi3vk2 keeps one WG but
   overlaps via ping-pong in shared).
Measure each with the benchmark (real clip) + RADV_DEBUG=shaderstats; only
end-to-end fps medians over hundreds of frames count.

## Benchmark semantics (based_aa mirror)

The benchmark/bench.py `eedi3` entry mirrors vsaa.based_aa's EEDI3 usage so
fps numbers reflect a real AA workload, not a bare field interpolation:

- settings = the based_aa defaults used when no antialiaser is given
  (vsaa/funcs.py:206-213): alpha=0.125, beta=0.25, gamma=40, nrad=2, mdis=20,
  vcheck=2, vthresh=(12,24,4), field=1.
- input: luma of the real clip at GRAY16, Point-upscaled 2x (2160p) — the user
  chose a simple Point upscale over ArtCNN for the supersampling step; the 2x
  frames (and the 2x mask frames) are what the RAM cache holds.
- mclip: the vsaa.based_aa mask chain — Prewitt edgemask -> binarize(mask_thr=60,
  scaled to depth) -> box_blur(Maximum) — Point-upscaled 2x. Passed as mclip to
  vsfeel and eedi3vk2 (both support it); vszipcl/vszipcu have no mclip, so they
  get sclip only. sclip = the 2x input clip (based_aa always passes sclip=ss).
- implementation: FilterSpec gains `aa=True`; bench_filter routes real-clip
  runs to make_aa_vpy (new), which decodes+caches the first --cache-frames 2x
  ss and mask frames (capped at 500, ~16.6 MB/frame each) before vspipe times,
  so the timed region is only the EEDI3 call. --synthetic (BlankClip, hang
  tests) keeps a plain 1x EEDI3 call.

Measured (jpbd.mkv real content, 300 frames, num_streams=4, 2160p GRAY16):
- vsfeel 183.8 fps | eedi3vk2 198.4 | vszipcl 207.6 | vszipcu 130.4
(the mclip path is the interesting one vs vszipcl: vsfeel+eedi3vk2 run the
edge-directed DP only on ~18% masked-on pixels; vszipcl has no mask.)

## Why the AA benchmark "levels out" near ~200 fps (and what it really measures)

Investigation (probe scripts, real jpbd.mkv 2x Point GRAY16, 200 frames,
num_streams=4): the ~200 fps clustering across backends is NOT an artifact of
the upscale/mask bleeding into the timed region, and NOT cache inflation.

Verified:
- The eval/decode/upscale/mask/cache preamble adds ~0.00s to vspipe's Output
  timer (a 100-frame 2x cache preamble vs none: 0.00s both). The AA setup is
  genuinely outside the timed region.
- No output-cache inflation: a CONSTANT BlankClip at 2x (every frame identical
  => maximum possible downstream cache reuse) is the SLOWEST vsfeel case
  (54.9 fps), not the fastest. Repeated inputs do not inflate fps.
- vsfeel fps vs mask threshold (fraction of pixels where the edge-directed DP
  runs, binarized mask after Prewitt->boxblur-max):
      mask thr=60 (vsaa default, ~18% edge)  -> 167 fps
      mask thr=0  (whole mask on => DP 100%) -> 57 fps
      no mclip (DP on 100%)                   -> 64 fps
      bare (no sclip, no mclip)               -> 70 fps
- vszipcl has NO mclip support, so it does the full DP on 100% of pixels and
  still hits ~194 fps on the same real content.

Conclusion: the tight spread (~184/198/208 vsfeel/eedi3vk2/vszipcl in the AA
bench) is because mclip-capable backends only run the expensive DP on the
~18% edge pixels (cheap vertical cubic elsewhere) while vszipcl's lead comes
from a ~3x faster full-DP kernel. Under the mask everyone is bound by the same
~18% edge subset, compressing the spread.

The benchmark's honest read for the speed phase:
- vsfeel unmasked full-DP kernel: ~64 fps (2x 2160p u16) = ~3x slower than
  vszipcl's ~194 fps full-DP.
- eedi3vk2 unmasked: ~100 fps (standalone 2x BlankClip), ~108 fps @ mdis20.
- mclip gives vsfeel ~167 fps in the AA workload (~2.6x over its unmasked
  speed), which is exactly what vsaa.based_aa exploits.

ULTIMATE GOAL (user-stated): combine vszipcl-class per-pixel full-DP kernel
speed with the mclip fast-path boost. vsfeel alone among the four backends has
mclip AND a portable GPU kernel, so reaching vszipcl's per-pixel rate would put
vsfeel at ~3x vszipcl in the AA workload (vszipcl can't use a mask at all):
  target ~500+ fps in the AA benchmark (194 x ~2.6 mclip boost).
That is the north star for the speed phase.

## Round 3: dropped ucubic / cost3 user parameters

Requested by the user ("drop the ucubic and ucost3 parameters. It seems like
none of the 3 others actually use those for anything, right?") — VERIFIED:

- eedi3vk2 registerFunction arg list: no ucubic/cost3 (always on internally).
- vszipcl / vszipcu `eedi3_sig`: no ucubic/cost3.
- vsaa EEDI3 dataclass: `ucubic`/`cost3` are `Never = cast(Never, MISSING)`
  deprecated — get_deint_args strips them from kwargs with a RuntimeWarning.
- Only CPU eedi3m exposes them.

So the GPU family hardcodes cost3=true (alpha /= 3, cost sums s0+s1+s2) and
ucubic=true (4-tap cubic interpolation along the direction, avg2 only as the
out-of-frame fallback). vsfeel matched eedi3vk2 bit-exactly only under those
settings; the switches were dead surface.

Changes:
- src/eedi3.comp: removed spec constants 6/7 (COST3, UCUBIC); renumbered
  `local_size_x_id` 8/9 -> 6/7. conn_cost always returns
  alpha*(s0+S1+S2)+beta*|u|+rw*v; rightmost-column and masked pixels always
  use cubic_float; direction interp = cubic when 3*|dir| taps are in-frame,
  else avg2 (guard fallback, required regardless of ucubic).
- src/eedi3.cpp: removed ucubic/cost3 from Eedi3Data, WidthKey (==), RowSpecData
  (10 -> 8 entries, offsets 0..28), arg parsing, registration args string, and
  the `if (d->cost3)` guard around `d->alpha /= 3.0f` (now unconditional, with
  a comment). `opt` stays accepted-but-ignored for eedi3m parity.
- tests/test_eedi3.py: removed the 5-case cost3/ucubic-close-to-eedi3m
  parametrization (params no longer exist); header note updated (eedi3m no
  longer an oracle for those params).

Verified: `ucubic`/`cost3` now rejected as unknown args; default-param output
still bit-exact vs eedi3vk2 (max diff 0); test_eedi3.py 43 passed; full suite
323 passed.

## Round 4: field=3 (double-rate AA) benchmark + sclip output-semantics fix

The user asked whether based_aa uses field=2 not field=1. Spy on the real
EEDI3.Backend.EEDI3 call from vsaa EEDI3.antialias() (progressive input):
`field = tff + double_rate*2` = 1 + 2 = **3** (double_rate defaults True even
for AA on progressive content), then `std.Merge(clip[::2], clip[1::2])` folds
the doubled output back. So based_aa's default EEDI3 call is field=3.

bench.py change (per user: "skip the merge, make it field=3 and call that
close enough"): eedi3 FilterSpec field default 1 -> 3; the benchmark times the
field=3 double-rate EEDI3 call alone (the merge is std work, not EEDI3 work).

**BUG FOUND + FIXED (src/eedi3.cpp):** under field>1, vsfeel validated sclip
against the PRE-doubling vi: it accepted an N-frame sclip (wrong — requests
sclip frame n >= N out of range on real content) and rejected the 2N-frame
sclip that eedi3m/eedi3vk2/vszip all require. The references treat sclip as
describing the OUTPUT: 2N frames under field>1 (based_aa provides exactly that
via Interleave([s, s])), 2x height under dh. Fix: build an out_vi copy
(numFrames*2 when field>1 with INT32_MAX/2 guard, height*2 when dh) and
validate sclip against it via isSameVideoInfo + numFrames. Verified: N-frame
sclip rejected under field=3; 2N-frame accepted, bit-exact vs eedi3vk2
(max diff 0) for field1/field3/dh with and without mclip.

bench.py make_aa_vpy/bench_aa now take eedi3_field and build the doubled
sclip (core.std.Interleave([clip, clip]) when field>1, exactly like based_aa)
so the field=3 AA path validates and runs (measured vsfeel 159.6 vs eedi3vk2
174.7 fps @ 60 frames, field=3 mdis=20 defaults). mclip stays N frames
(requested at sn = n/2 — correct).

tests/test_eedi3.py additions (43 -> 50):
- sclip under field>1 must be 2N frames (N rejected; 2N bit-exact vs vk2)
- sclip under dh must be 2x height (1x rejected; 2x bit-exact)
- sclip content actually drives vcheck (shifted-noise sclip, bit-exact vs vk2,
  and differs from the no-sclip run)
- mclip x field 2/3 and x dh combos bit-exact vs vk2
- output props: _FieldBased=0 progressive; duration halved under field>1
  (cross-multiply compare vs the source duration)
- field=2 added to the vk2 reference sweep (was only 0/1/3)

## Round 5 (speed phase): accuracy policy RELAXED

USER DECISION (verbatim intent): "I don't care about bit exactness. As long as
it's 'close enough' that's fine. I just want it to be FAST. You can sacrifice
some accuracy and cut some corners if it means big speed gains."

This UNLOCKS porting vszipcl's kernel structure (strip-batched cost fill, K=2
fused DP, sliding register h-windows, local-staged backtrack) and aggressive
fast-math/relaxed ordering without staying family-A-bit-exact vs eedi3vk2.
Tests currently assert bit-exact (tol 0) vs eedi3vk2 on u16 -> they must be
re-baselined to "close enough" tolerances after the rewrite (measured first).

### Speed baseline (field=3 AA bench, full-DP mclip off, 200 frames, 2160p GRAY16)
- BEFORE any perf work: vsfeel 74.6 | eedi3vk2 103.5 | vszipcl 204.4 fps
- ROOT CAUSE #1 found: the row kernel read pad/bmask/sclip from the SYSTEM-RAM
  staging buffer (memory type 5 = HOST_VISIBLE|HOST_COHERENT|HOST_CACHED, heap
  0) on every cost/DP access -> PCIe reads from GTT.
- Attempted fix (WRONG, reverted): allocate staging DEVICE_LOCAL|HOST_VISIBLE|
  HOST_COHERENT (type 3/4 on this 7900XTX). Collapsed to 27.6 fps: RADV's
  coherent device memory has NO write-combining, so the CPU's ~16.6 MB/frame
  pad write went through a slow uncached path. eedi3vk2 gets WC via VMA
  HOST_ACCESS_SEQUENTIAL_WRITE which vsfeel's allocate_memory cannot express.
- CORRECT fix (shipped): keep CPU writes in system-RAM staging (fast cached
  host writes) and add a device-local `pad_dev` mirror buffer; one H2D
  vkCmdCopyBuffer(staging[0,upload_total) -> pad_dev) + transfer->compute
  barrier at the start of record_command_buffer; bind pad/bmask/sclip
  (bindings 0/4/5) to pad_dev. Kernels now read pad from VRAM.
- AFTER fix: vsfeel 107.1 | eedi3vk2 106.1 | vszipcl 207.4 fps -> vsfeel now
  matches eedi3vk2 exactly. All 50 EEDI3 tests still pass (bit-exact held).
- REMAINING GAP: vszipcl 207 is ~2x vsfeel/vk2 107. Its lead is the KERNEL
  ARCHITECTURE (strip-batched cost fill hoisted out of the barrier-locked DP,
  K=2 fused columns = 1 barrier/2 cols, sliding register h-windows cutting
  fill loads 2.3-3x, local-staged serial backtrack), NOT memory. Port that
  structure into the vsfeel GLSL row kernel next.

### Round 5 continued: perf probes
- vsfeel kernel is DP-bound: fps scales mdis 5/10/20/40 -> 167/159/114/81 (vcheck=0
  full-DP 2160p). Not fixed-overhead bound.
- vcheck=0 full-DP split: vsfeel 116.6 | eedi3vk2 143.6 | vszipcl 237.7.
  eedi3vk2's subgroup register DP (no per-col WG barriers, rolling sums) beats
  our barrier-per-column kernel by ~23%; vszipcl's strip-batched LDS DP
  (cooperative cost fill + K=2 fused columns + rolling register h-windows)
  beats both by ~1.65x over eedi3vk2.
- GPU: RDNA3 subgroup size 64 native, size-control lets us request 32;
  shuffle/shuffle-relative supported. eedi3vk2 forces 32 (2 rows/WG = 64 thrs).
- vszipcl has NO mclip at all (its 238 is pure full-DP). vsfeel must keep the
  mclip cost-skip to retain the AA-workload advantage.

### Round 5: conn_cost is the bottleneck (measured)
Decisive probe: replacing conn_cost with a constant (keeping the exact DP loop
+ barriers + backtrack) lifts vsfeel vcheck=0 full-DP from 116 -> 278 fps.
=> the connection-cost evaluation is ~70% of row-kernel time, NOT the
per-column barriers/DP. Cost is recomputed per (lane-direction, column): each
eval is 3 window sums x (2*nrad+1) abs-diff terms over 4 pad rows ~= 90 pad
loads, done for every direction lane at every column.

Fix direction (matches vszipcl/eedi3vk2): rolling window sums per direction
lane — each new column x adds ONE new leading A-term per window (6 loads for
the +u/-u pair vs 15) instead of a full recompute; the 3 s0/s1/s2 sums per
lane become ring-buffer rolls. Also amortizes global pad reads.

### Round 5: rolling-window costs — 116 -> 212 fps full-DP (vcheck=0)
Decisive probe: zeroing conn_cost (keeping the exact DP/barrier/backtrack)
lifted vcheck=0 full-DP from 116 -> 278 fps => conn_cost evaluation was ~70%
of row-kernel time. Implemented per-lane ROLLING window costs in the shader:
s0/s1/s2 for direction u are (2*nrad+1)-term window sums; advancing one column
costs ONE new leading-term eval (9 pad loads) instead of the full recompute
(~90 pad loads). Helpers rowA / roll_seed / roll_push / combine_interior added
before ENTRY_ROW; the DP loop rolls when the lane's column is "interior"
(x in [2|u|, WIDTH-1-2|u|], where both s1/s2 cost gates always hold) and falls
back to conn_cost at the left/right ramps and after masked columns.
- u16 output STILL BIT-EXACT vs eedi3vk2 (max diff 0): pad values are exact
  integers < 2^24 so window sums are exact regardless of accumulation order.
- f32: ~1 ulp on defaults; at the degenerate alpha+beta=1 (rw=0) corner a rare
  DP argmin flip appears (22/115200 px, max 0.038, frame 23) -> test tolerance
  for that case bumped 1e-6 -> 0.05. All other f32 cases stay <= 1 ulp.
- MEASURED (2160p GRAY16 field=3):
    full-DP vcheck=0:  vsfeel 212 | vszipcl 238  (was 116)
    full-DP vcheck=2:  vsfeel 164 | vszipcl 198 | eedi3vk2 105 (was 107/105)
    AA mclip ON vc0:   vsfeel 235 | vszipcl 245   (was 116-ish before? now great)
    AA mclip ON vc2:   vsfeel 155 | vszipcl 205   <- vcheck is NOW the bottleneck
- Lesson: with the row kernel fast, the vcheck serial row walk dominates the
  mclip workload (235 -> 155 fps = 80 fps cost). vszipcl's vcheck does one
  barrier per row reading the prev modified row from GLOBAL L2 (not a shared
  ping-pong), and it fused all planes into one launch. Ours uses a shared
  tlineSh[2][WIDTH] ping-pong (30 KB LDS at 3840 width) + 2 barriers/row,
  mirroring eedi3vk2 (which is slow here). NEXT: rewrite vcheck to vszipcl's
  single-barrier global-read model and consider batching planes.
- Also fixed a latent test bug: the reference-compare subprocess lacked
  .copy() on its ctypes frame views (phantom diffs possible under frame
  recycling) and f32_order.py-style probes that view (h, stride/4) as float32
  are WRONG (must slice tight bytes first). tests/test_eedi3.py compare script
  now .copy()s both views.

### Round 5: vcheck global-read experiment (REVERTED to shared ping-pong)
Attempted to drop the 30 KB shared tlineSh[2][WIDTH] ping-pong in vcheck for
vszipcl's global-read model (1 barrier/row, d2p from global dst). The naive
in-place version (write vcheck result to dst row r immediately) RACES: within
one row, lane A writing dst[r][x] can be read by lane B computing dst[r][x']
(x' = x +- dirc) in the same grid-stride loop -> nondeterministic, up to 28
LSB at mdis=20. vszipcl avoids this by writing to a SEPARATE out buffer and
reading taps from the untouched row-kernel snapshot dst. vsfeel writes in
place, so it needs the shared ping-pong (or host-side scratch buffer). Reverted
to the ping-pong, now with ONE barrier per row (compute into tlineSh[cur],
barrier, flush prev row to dst). Correct + bit-exact, but ~147 fps in AA
(~same as the 2-barrier version): the 30 KB LDS ping-pong itself is the cost,
not the barrier count. A proper fix = host-side scratch out buffer (vcheck
writes there, host blits back) like vszipcl -- deferred, ~10% upside.

### Round 5 FINAL numbers (2160p GRAY16 field=3, num_streams=4)
- AA mclip ON (based_aa workload):   vsfeel ~152-162 | vszipcl ~158-204 (high
  run-to-run variance) | eedi3vk2 ~194 | vszipcu ~129
- full-DP mclip OFF:                 vsfeel ~165  | vszipcl ~164-195 | eedi3vk2 ~100
=> vsfeel went from 75 fps (slowest, pre-round-5) to ~150-165 fps, AT PARITY
with vszipcl on full-DP and solidly beating eedi3vk2 (~60%) everywhere.
BIGGEST WIN by far: rolling-window connection costs (116 -> 212 fps vcheck=0
full-DP). H2D pad_dev copy fixed GTT-reads (75 -> 107).
- tests/test_eedi3.py: 50/50 pass. u16 bit-exact vs eedi3vk2 retained
  (rolling sums exact for integer-valued pad). f32 ~1 ulp except the rw=0
  degenerate corner (alpha+beta=1): 22/115200 px flip up to 0.038 -> that
  case's tolerance 0.05. Compare subprocess .copy()s its ctypes views now.

### Round 5 (step 3): vcheck vout model done; mclip row kernel is next target
Real AA bench (2000 frames, mclip on, vc2): vsfeel 155 | eedi3vk2 188 | vszipcl
202. Split at vc0 (mclip on): vsfeel 232 | eedi3vk2 317 | vszipcl 239. So:
- vcheck kernel costs ~77 fps (232->155); the vout rewrite (write vchecked rows
  to a scratch b7 buffer, read taps from untouched dst, no LDS ping-pong, 1
  barrier/row) is IN but only helped ~10 (was 144 -> 155).
- eedi3vk2's ROW kernel is the real gap at mclip-on: 317 vs our 232. Cause:
  eedi3vk2's DP is per-lane REGISTER state with subgroup shuffles for the
  +-1 neighbor window — ZERO workgroup barriers in the column loop. Masked
  columns are nearly free (cur[j]=pp[j] register copy + fixups + one btBuf
  write). Our shared-pco DP does 2 workgroup barriers + a full pco/pbt row
  copy PER COLUMN even for masked columns -> ~3150 masked columns x barrier.
- NEXT: port eedi3vk2's subgroup-register DP into our ENTRY_ROW (64-lane WG =
  2 subgroups x 32, one row per subgroup, dispatch rows/2, requiredSubgroupSize
  32, pbt as RELATIVE deltas, bmask packed into shared bits, rolling cost
  windows kept). Host changes: glslc subgroup extensions already needed;
  add VkPipelineShaderStageRequiredSubgroupSizeCreateInfo + dispatch rows/2
  + spec wgSize/sgSize. This is the big mclip win.

## Round 6: subgroup-register DP rewrite of ENTRY_ROW — IN but NO mclip win

Goal was eedi3vk2's 317 fps at mclip-on vc0 (ours 232). Ported its exact
structure into our ENTRY_ROW (all uncommitted):

- WG is now ONE 32-lane subgroup per interp row (not 2x32); host dispatch
  stays (1, rows); lsz_row spec const -> 32; row pipeline gets
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo(32) (dfttest.cpp ~675
  precedent; vsfeel.cpp already enables subgroup size control, min=32 max=64).
- DP in per-lane registers: each lane owns K = ceil(TPITCH/32) directions;
  pp[K]/pRel[K] + rolling sums s0/s1/s2[K] + rings r0/r1/r2[K][RING_CAP=7] +
  rolled[K]; +-1 predecessor window gathered via subgroupShuffleUp/Down into
  ext[K+2]; NO workgroup barriers in the column loop.
- pbt switched from ABSOLUTE predecessors to RELATIVE deltas (idx - u, int8);
  backtrack accumulates f += pbt[CENTER+f] (eedi3vk2 convention).
- Masked columns: cur=pp copy + fixups at CENTER+-pumax, no barrier.
- bmask packed once per row into shared bmaskSh[(WIDTH+31)/32] bits.
- glslc target raised to --target-env=vulkan1.3 for the EEDI3 comps (needs
  SPIR-V 1.3 for subgroup ops + memory model). Extensions + memory model
  pragma only under `#if defined(ENTRY_ROW)` (vcheck kernel untouched).

RESULT (bench AA, 4 streams, real jpbd 2x):
- vc0 mclip ON:  ~228-230 (was 232) vs eedi3vk2 ~312-318  -> NO GAIN
- vc2 mclip ON:  ~160-166 (was 155) vs eedi3vk2 ~174-191 -> tiny gain
- vc0 mclip OFF (full DP): ~176-181 vs eedi3vk2 ~137-151 -> we WIN (as before)
Tests: 50/50 pass, u16 still bit-exact vs eedi3vk2, f32 band unchanged.

DEAD ENDS / NON-GAINS (do NOT retry without new evidence):
1. Private arrays sized from spec-constant-derived K compile fine with glslc
   2026.3 (probed) — not a blocker.
2. Per-lane rolling kept alive WITH mclip (rolled on unmasked interior cols):
   no gain. Mirrored eedi3vk2 exactly — rolling ONLY when HAS_MCLIP==0, full
   conn_cost recompute on unmasked columns with mclip (lets the ~72 ring VGPRs
   go so masked columns run at small footprint): ~212 (was ~220) vc0 mclip-on.
   Still ~85 fps behind vk2.
3. bmaskSh shared packed bits (vk2 does this): barely moved vc0 mclip-on
   (228 vs 230). The per-column masked test is NOT the bottleneck.
4. Serial per-frame ms probes (get_frame in a loop) MISLEAD: at 4 streams
   they showed vsfeel no-mclip 24 ms vs vk2 9 ms/frame — a pure latency
   artifact of non-parallel get_frame. Only trust end-to-end fps from
   benchmark/bench.py (vspipe, parallel streams). The actual AA bench shows
   vsfeel BEATS vk2 at no-mclip.

WHY the port didn't close the mclip gap (hypotheses, untested):
- vk2 no-mclip full-DP is FASTER than its mclip case in our hands (9 vs 17 ms
  serial probes) yet its bench fps mclip-on vc0 (317) >> no-mclip (~137)?? The
  two probes disagree because serial ms != parallel fps; do not trust the
  serial comparisons. Real benchmark facts: mclip-on vc0 gap is ~85 fps
  (228 vs 313). Our row kernel at 2160p full-DP is NOT the limiter (we beat
  vk2 there). Suspects: (a) with mclip our unmasked-column cost recompute
  (conn_cost ~90 pad loads each) is what eedi3vk2 also does, but their
  whole-WG occupancy may be higher (smaller footprint) so the ~18%-unmasked
  columns cost less wall time; (b) the column loop still has a serial
  dependency through pp that a 32-lane WG can't hide as well as vk2's 64-lane
  2-subgroup WG (2 rows interleaved hide DP latency + their rolling
  seedSum/orderedSum uses fp32 exact re-sum, not running sums). (c) LDS/barrier
  differences are ruled out (bmaskSh + no barriers).

### Round 6 follow-up probes (mclip row kernel floor analysis)
Measured with benchmark/bench.py real AA chain (2x 3840x2160 GRAY16, field=3
mode, vc0 = row kernel only), 4 streams, multiple runs:
- real shader vc0 mclip-on: ~258-264 fps (vs eedi3vk2 ~320-325)  [row gap ~60]
- real shader vc2 mclip-on: ~167-177 fps (vs eedi3vk2 ~191)
- serial get_frame probes are UNRELIABLE (non-parallel latency artifacts:
  24ms vs 9ms shown at 4 streams while bench shows us BEATING vk2 no-mclip).
  Trust only bench.py fps.
DECOMPOSITION PROBES (each a hacked shader build, installed + benched vc0
mclip-on, then reverted):
1. Zero the unmasked-column conn_cost (mclip path): 235 -> 241 fps. The
   connection-cost recompute on unmasked columns is NOT the bottleneck
   (~10 fps).
2. Force EVERY column to the masked-carry path (no DP at all): -> 245-256 fps.
   The serial 3840-column carry loop itself floors ~256 fps.
3. Also drop the masked pbt global writes: -> 256 fps. pbt stores ~10 fps.
=> The masked-carry serial loop (~256 fps) is the mclip row-kernel wall; the
   whole DP + costs add only ~10-20 fps on top. vk2 reaches ~320. Conclusion:
   our 32-lane/1-row WG carry loop cannot be cheapened further by removing
   work; the gap is structural (per-column loop iteration cost / occupancy /
   wavefront layout). Do NOT keep hacking the carry path.
NON-GAINS logged (do not retry): subgroupcoherent on pbt (b2) made it WORSE
(259 -> 229, reverted; pbt is written per column by 96 lanes and read back by
the same subgroup — the qualifier hurt codegen/throughput on RADV);
bmaskSh shared packed bits (moved 230->228, negligible).

OPEN HYPOTHESES for the 258-vs-320 structural gap (untested):
(a) vk2's WG = 64 lanes = 2 subgroups x 32, each subgroup its OWN row
    (dispatch (rows+1)/2); ours = 32-lane WG, 1 row (dispatch (1, rows)).
    Occupancy math looks equivalent but the 2-row WG may schedule/ILP better
    or halve per-WG overhead (barrier cost amortized over 2 rows).
(b) vk2's DP relax window gathers ext[] BEFORE the masked test and reuses
    it; branchy mclip vs non-mclip compiles to separate code paths.
(c) vk2 stages the 4 source rows in SHARED memory (common.glsl P()/Pf() read
    LDS), ours reads pad from the device-local buffer directly. On unmasked
    columns conn_cost issues ~90 global pad loads; vk2 hits LDS.
Next experiment if continuing: convert ENTRY_ROW to vk2's exact 64-lane /
2-row-per-WG dispatch (sg = gl_SubgroupID, r = WorkGroupID.y*2+sg, rowExists
guard, shared arrays [2][...], bmaskSh[2][...]) and compare; also test
staging pad rows in LDS. Expected ceiling ~320 fps for vc0 mclip-on.

### Round 6 (2): 2-row-per-WG layout REJECTED (regression)
Converted ENTRY_ROW to eedi3vk2's exact 64-lane WG (2 subgroups x 32, sg =
gl_SubgroupID, r = WorkGroupID.y*2+sg, shared arrays [2][...], dispatch
(1,(rows+1)/2), rowExists early-return). Results on the real AA chain (4
streams, 300 frames):
- vc0 mclip-on: 216 vs eedi3vk2 322  (WORSE than 1-row's 258)
- vc2 mclip-on: 170 vs 192
REVERTED to the 1-row 32-lane WG (faster in every config). Hypothesis (a)
falsified — do not retry.
Caveat: heavy run-to-run AND run-session variance on vc0 (191-264 fps for the
identical 1-row binary across separate bench invocations; GPU clock state).
Head-to-head in the SAME invocation is stable. Judge only same-invocation
pairs, or median of 3 same-session runs. Flagship vc2 mclip-on stable:
vsfeel ~151-153 vs eedi3vk2 ~185-187 (the subgroup-register rewrite gained us
155 -> ~152?? No: pre-rewrite flagship was ~155; post-rewrite ~151-153 at
vc2 — essentially FLAT; the register port was a wash on the flagship
workload, though vc0 row-only went 232 -> ~235-258 (small gain, noisy).

CURRENT STATE: ENTRY_ROW = subgroup-register DP, 1 row per 32-lane WG,
requiredSubgroupSize=32, relative-delta pbt, bmaskSh packed bits. All 50
tests pass, u16 bit-exact. Row kernel vc0 mclip-on ~235-258 vs vk2 ~320-325;
vcheck kernel is CHEAPER than vk2's. Remaining gap is the row kernel's
masked-carry serial-column floor (probe floor ~256 = we are AT the floor; DP
adds <10 fps). Since vk2 exceeds our measured floor with the same structure,
its per-column iteration is cheaper in ways probes did not isolate; recorded
hypotheses in the previous section. Next lever candidates (not yet tried):
row kernel reading pad through LDS staging of the 4 source rows (hypothesis
(c)), and rechecking whether the two-row approach would win if the DP were
not rolled (untested since vc0 is DP-light under mclip).

### Round 6 (3): CORRECTED numbers at 1000+ frames + probing lesson
All earlier sub-300-frame probes were NOISY (GPU clock state swings the same
binary 191-264 fps between invocations). At 1000 frames (bench.py, real AA
chain, 4 streams), stable same-invocation pairs:
- vc0 mclip-on:  vsfeel ~227-231 | eedi3vk2 ~299-312   (row-only gap ~75)
- vc2 mclip-on:  vsfeel ~151-155 | eedi3vk2 ~187-190   (flagship)
Redo of the all-masked-carry floor probe at 1000 frames: 213 fps, i.e. the
carry serial loop IS the entire row cost and the DP adds ~nothing measurable;
the "floor" measured ≈ the real shader's own number in the same clock state.
=> The row kernel cannot be made faster by removing DP work. Both the real
shader and the carry-only shader sit ~213-231 depending on clock state; vk2
~300. Per-column iteration differences (subgroupcoherent pbt: 259->229;
restrict pbt: 214->201; 2-row WG: ->216) ALL regressed or were flat. Do NOT
retry those. The gap is in per-column codegen/issue that we have not isolated;
further row-kernel micro-tuning is low value.
NEXT-STEP recommendation: attack the vcheck kernel instead — at vc2 the
flagship spends ~75 fps of its ~150 on vcheck (row-only vc0 ~230 -> vc2 ~152),
and vk2's vcheck is also ~75 (325 -> 190) but from a higher row base. Our
vcheck uses a serial single-WG row walk with 1 barrier/row; vk2 does the same
but its per-row parallel work spans 1024 lanes grid-strided. Compare vcheck
costs in the SAME invocation before optimizing (see prior section numbers:
our vc0->vc2 delta 235->167 = 68fps; vk2 325->197 = 128fps — ours is already
cheaper in absolute terms, so vcheck is NOT our deficit; the row kernel is).

### Round 6 (4): more row-kernel micro-variants REJECTED (all regressed)
- `[[unroll]]` on the K-direction loops (needs GL_EXT_control_flow_attributes):
  vc0 227 -> 196. The unrolled body is huge (K=3 x TPITCH path) and hurts
  I-cache / scheduling. Reverted.
- `restrict` on pbt: 214 -> 201. Reverted.
- subgroupcoherent pbt (already logged): 259 -> 229. Reverted.
=> vk2's ~300 vs our ~195-230 at vc0 mclip-on is NOT reachable by these
micro-qualifier/codegen tweaks. Variance caveat: same binary swings 195-230
across invocations, so treat all single-run deltas <10% as noise. We have not
found the source of vk2's per-column advantage; no more row-kernel
micro-tuning without a profiling tool (rocprofv3 / RADV_DEBUG=perfetto) to
see occupancy/issue-stall counters.

### Round 6 (5): DEFINITIVE finding — subgroup rewrite == barrier DP, gap is NOT the DP
Fair A/B, same session, 1000 frames, vc0 mclip-on (real AA chain):
- OLD barrier shader (shared pco parity ping-pong): 219.4 fps
- NEW subgroup-register shader (registers + shuffles): 220.3 fps
- eedi3vk2 both runs: ~310-330
=> The register/subgroup rewrite is performance-IDENTICAL to the barrier DP
it replaced. All the vk2-style machinery (register DP, shuffles, relative
pbt, bmaskSh, no barriers) bought ZERO. It did preserve correctness (50/50).
- mdis=3 (TPITCH=7, K=1, DP nearly free), mclip on: vsfeel 259 | vk2 412
  => even with NO DP work, vk2 is 1.6x faster on the masked workload. The gap
  lives in the masked-column iteration / memory path, NOT the DP width or
  structure.
- no-mclip full-DP vc0 (real 2x content): vsfeel 184 | vszipcl 237 | vk2 140.
  => vsfeel beats vk2 WITHOUT mclip; vszipcl leads everything at full DP.
FULL STANDINGS (1000f, real AA 2x chain, 4 streams):
  vc0: vsfeel 222 | vszipcl 237 | eedi3vk2 308 | vszipcu 174
  vc2: vsfeel 168 | vszipcl 197 | eedi3vk2 182 | vszipcu 116
SHADERSTATS mdis3-mclip row: vsfeel 24 VGPR / 8408 B code; vk2 48 VGPR /
6684 B. vk2's masked loop compiles leaner despite more registers.
NEXT DIRECTIONS (ranked): (1) the masked-column iteration cost itself —
compare ISA of the per-column masked path (vk2 packs bmask bits and tests via
shared uint with scalar/restrict qualifiers; ours plain). (2) vszipcl beats us
at BOTH vc0 (237) and vc2 (197) WITHOUT mclip: it does not pay our mclip
masked loop at all. If mclip's masked fast path cannot beat vszipcl's full-DP
cost on this mask density (~18% unmasked), the mclip machinery is the wrong
place to compete; measure a no-mclip vsfeel run at the same settings to see
how much mclip actually buys (earlier: full-DP no-mclip vsfeel 184 = SLOWER
than its own mclip 222, so mclip DOES help vsfeel ~38 fps; vszipcl 237
full-DP is still above both).

## Round 7 (speed phase): host overhaul + span-skip — vsfeel TAKES THE LEAD

Starting point (2000f flagship field=3 AA, 4 streams): vsfeel 167 | vk2 197 |
vszipcl 192. End point: vsfeel 363 @8 streams (242 @4) | vszipcl 211 | vk2 196.

### 7.1 Mask stats (real AA workload, jpbd 2x mask, mdis=20)
- 53% of all mask rows fully empty; 64% of INTERP rows fully masked; non-empty
  rows average 22% dilated span. The DP/carry/backtrack on those rows is pure
  waste (output = vertical cubic, dmap 0).

### 7.2 Span-skip (src/eedi3.comp ENTRY_ROW, shader-only, BIT-EXACT)
- After the bmask pack: lane 0 scans packed words for the first set bit ->
  shared rowXmin (all lanes). Fully-masked row (xmin >= WIDTH): parallel cubic
  loop over all lanes, early return (no DP, no pbt, no backtrack).
- Else the DP loop starts at max(1, xmin) with an analytic predecessor seed:
  masked columns only CARRY costs (cur = pp, values never change), so pp = 0
  is exact; pRel gets the deterministic fixup pattern (last-hit-wins,
  -sign(u) iff |u| <= xmin-2 or the degenerate right-ramp clause, else -u).
- Backtrack tile loop breaks at the first tile fully left of xmin; the
  leftmost partial tile walks only down to xmin and interps [x0, xmin) as
  cubic directly (no unwritten-pbt/tileF reads anywhere).
- VERIFIED bit-exact vs eedi3vk2 on real content (vc0/vc2, field=3, mclip).
- SURPRISE: flagship barely moved (167 -> 170). The row kernel was NOT the
  limiter — the HOST was. (Zero-mask vc0: 279 vs real-mask 211: the fast path
  fires, but fixed costs dominate.)

### 7.3 Host was the bottleneck (VSFEEL_EEDI3_HBENCH probe, KEPT in tree)
- vc0 single-stream split: cpu_stage 4-5.5ms (pad build ~2ms + bmask ~2ms) +
  blit ~1.8ms + gpu_submit_wait ~2ms (cold clocks). Serial ~8ms/frame.
- Reference: vk2's GPU profile (EEDI3VK2_PROFILE=1): main=2.9ms vcheck=2.2ms
  copy=0.3ms per frame — GPU-bound with host overlapped. Our vcheck is ~4x
  cheaper (~0.5-1.2ms); our deficit was row+H2D+D2H + host.

### 7.4 u16-native pad (host + shader, BIT-EXACT)
- Pad stored/transferred as u16 for 16-bit input (was float): halves CPU
  writes, upload (16.6 -> 8.3MB), H2D copy, and kernel pad-read bytes.
  float(int(pad[])) widening in pad_get (a direct float(uint16) needs the
  int16-arithmetic ext/feature; int(uint16) already compiles under the 16-bit
  storage ext). All values < 2^16 so float() is exact -> bit-identical math.
- NOTE: vk2 also uploads float pad (its common.glsl: "the CPU upconverts to
  float32"), so this is a vsfeel-only upload/H2D advantage, not parity.

### 7.5 Streaming host copies (copy_stream_out/read, sibling precedent)
- Pad rows + vertical margins + sclip gather: copy_stream_out (NT stores;
  pad/staging never re-read by CPU). Blit: copy_stream_read for interp rows
  (staging->frame; guarded by row_bytes%32==0 since movntdqa faults
  unaligned) + copy_stream_out per kept row (replaces vsh::bitblt).
- After 7.4+7.5: cpu_stage ~2.8-3.7ms, blit ~1ms, gpu wait ~1.2ms (cold).

### 7.6 Streams: 8 is the sweet spot (2000f flagship)
- 4 streams: 242 | 8 streams: 349 | 12 streams: 341 (plateau).
- Competitors do NOT scale: vk2 197->196, vszipcl 207->211 (GPU-bound).
- VRAM cost per stream ~200MB (mostly pbt); 8 streams ~= 1.6GB, fine on 24GB.
- DEFAULTS UNCHANGED (still 4 streams): 4s ~242-250 still beats 211/196.
  Revisit default 4 -> 8 after further opts (needs VRAM/overlap re-tune).

### 7.7 Bit-packed bmask (host + shader, BIT-EXACT)
- build_bmask_row ORs bits into a zeroed word row (was byte stores): 8x
  smaller upload (4 -> 0.5MB/frame); shader burst-copies 120 words to shared
  instead of bit-packing bytes. bmask_base is now a WORD offset.
- BUG (caught by tests, fixed): bmask_base was left as a byte offset (4x too
  far) -> 5 mclip tests failed (gray8 same-format 16/32, field2/3, dh).
  tests/test_eedi3.py 50/50 + real-content bit-exactness re-verified after.
- Flagship @8s: 349 -> 363. Row bounds @8s vc0: zero-mask 440 / real 427 /
  full-DP no-mclip 235 (row kernel nearly at its cubic floor; remaining frame
  time is fixed costs + vcheck).

### 7.8 Memory types (7900XTX/RADV, recorded for future zero-copy work)
- heap0 = 33GB sysRAM, heap1 = 24GB VRAM. Host-visible VRAM types (3/4) are
  UNCACHED (no WC-exposed type) -> the old 27fps collapse stands; ReBAR-style
  read-upload-directly (vk2's VMA strategy) is NOT available to us. The H2D
  mirror + cached-RAM staging stays. (One-shot VSFEEL_MEMDUMP probe removed.)

### 7.9 Standings (2000f flagship field=3 AA, same-session)
- 8 streams: vsfeel 363 | vszipcl 211 | eedi3vk2 196  (+72% over fastest ref)
- 4 streams (defaults): vsfeel ~242-250 | vszipcl ~207 | eedi3vk2 ~197
- u16 output still BIT-EXACT vs eedi3vk2 everywhere (the bar from round 5).

### NEXT (post-checkpoint, ranked)
1. vcheck empty-row split: 1080 serial barriers (~0.5ms) dominate our vcheck;
   fully-masked rows (64%, dmap 0, no d2p need) -> parallel copy kernel, serial
   walk only over unmasked rows (~390 barriers). Est +10%.
2. H2D-elimination A/B: bind staging directly for pad/bmask/sclip (skip copy
   + barrier). DISTINCT bytes are small (12MB, cache-absorbed); old +43% H2D
   win predates rolling+mclip-skip — re-measure. Est +8%, risk: GTT slowness.
3. vout->staging direct (vc>0): drop the vout D2H copy+barrier. Est +7%.
4. bmask zero-row fast path on CPU (u64 OR-reduce + memset for the 64% empty
   rows; only matters if CPU re-limits at higher overlap).
5. Default streams 4 -> 8 + VRAM/overlap re-tune. No-mclip strip-batch port
   (secondary: we already beat vk2 without mclip).

### Probe notes (do NOT lose)
- KEPT: VSFEEL_EEDI3_HBENCH (host cpu_stage/pad/bmask + gpu_submit_wait +
  blit split, frame sn==0). tmp/hbench.vpy + tmp/hbench_nomask.vpy drivers.
- REVERTED (hung the queue, low GPU, stalled runs): Vulkan timestamp queries
  (vkCmdResetQueryPool + vkCmdWriteTimestamp ALL_COMMANDS + query-pool read).
  NEVER re-add without checking timestampValidBits + stage validity. The
  failure mode is a full stall, not an error.
- tmp/aa_zero.vpy (AA_NS streams, AA_MASK zero/real) + tmp/aa_full.vpy
  (no-mclip full-DP) isolate row-kernel bounds via vspipe.

## Round 8 (speed phase): GPU upload kernels + completion overhaul — 398fps

Starting point: 363fps @8s (round 7 checkpoint, committed). End: 398fps @8s
(vc0 @8s ~492). All bit-exact vs eedi3vk2, 50/50 tests throughout (failures
noted where they caught real bugs).

### 8.1 GPU pad kernel (ENTRY_PAD) — KEPT
- CPU now gather-copies tight kept rows only (streaming, ~0.3ms); H2D moves
  raw (8.3MB ~= old built 8.8MB, no byte growth); ENTRY_PAD expands eedi3m's
  exact mirrors in VRAM (1 thread/pad-element, 256-wide grid).
- Exactness details: vertical mapping replays copyPad in PAD space (top
  qy = 2*MV-qy; bottom qy -= 2+2*off+4k with y0 = pad_h-MV+off); kk clamped
  to [0, rows) (interp-parity rows unread); sx mirrors + clamp (stride-padding
  lanes unread; row-kernel reads provably stay in [MV-3, MV+W+2]).
- Verified: no-mclip full-DP bit-exact + full suite (dh/field2-3/YUV/f32 all
  pass — the mapping holds for every geometry).

### 8.2 GPU bmask kernel (ENTRY_BMASK) — IMPLEMENTED, BIT-EXACT, THEN REVERTED
- Window-OR formulation (out[x] = any mask in [x-mdis, x+mdis]) == the host's
  last-propagation dilation (proved equivalent incl. edges); one thread/word.
- BUG caught by spancheck (~1% px): kernel indexed the upload by mask row
  (mrow) instead of upload row (r) — diffs only on odd rows >= 1081 gave it
  away (r=540 reads mrow 1081 > uploaded 1080). Upload and kernel must agree
  the gather already applied the mapping. Fixed -> bit-exact, 50/50.
- REVERTED ANYWAY: 363 -> 290fps. +7.3MB H2D (maskraw) + 2 launches while the
  CPU it freed was already hidden at 8 streams. GPU-side != faster when the
  freed resource isn't the limiter. Code fully removed (pad kernel stays).

### 8.3 H2D elimination / GTT-direct — REVERTED (178fps)
- With kernels reading staging directly (no copy, no barrier): 290 -> 178.
  GPU reads of cached-RAM staging snoop-stall per line; the bmask window
  re-reads thrash under pbt streaming traffic. The old +43% H2D win stands:
  on this platform everything reused goes through VRAM. Reverted.

### 8.4 Uncached staging — REVERTED (43fps!)
- Dropping HOST_CACHED (all CPU access is NT/DMA anyway): 178 -> 43fps.
  movntdqa NT LOADS from UC memory serialize (~100ns each). NT needs WB or
  WC; UC kills it. Staging MUST stay cached. Reverted; lesson recorded.

### 8.5 Hybrid (KEPT): GPU pad + CPU packed bits + H2D = 386fps
- H2D 16.8MB (raw 8.3 + sclip 8 + bits 0.5). CPU ~1.3ms (gathers + bits +
  blit), mostly hidden. Bit-exact, 50/50.

### 8.6 vcheck empty-row split (ENTRY_VCOPY + rempty) — KEPT (+1%: 386 -> 390)
- Row kernel lane 0 writes 1 flag byte/row (dev_buf region, accessed via b3's
  whole-range int8 descriptor — no new binding). VCOPY = parallel copy pass
  (own module/pipeline/grid rows*W/256, only dispatched with mclip) writing
  vout = cint/sclip (do_row) or dst (!do_row) for empty rows. Walk `continue`s
  on the flag (uniform read -> legal barrier skip), dropping ~690 of 1080
  serial barriers.
- Result DISPROVED the barrier-cost model: removing 690 barriers saved
  ~0.03ms -> barriers are ~40ns, not ~500ns. vcheck (~0.53ms) is
  traffic-bound (~100MB: dst taps + pad + cint/sclip + vout), and the split
  saves barriers, not bytes. Kept anyway: exact, helps mask-heavy content,
  zero cost without mclip (copy skipped, flag never set).

### 8.7 vout-direct (no D2H for vc>0) — KEPT (390 -> 398)
- b7 descriptor -> staging; vout_base indexes the download region; the D2H
  copy + compute->transfer barrier are skipped for vc>0 (kept for vc==0 where
  dst still needs downloading). GPU streaming writes are posted/pipelined, so
  this removes a serial copy + sync at zero byte cost. Bit-exact, 50/50.
  (dev_buf vout region retired but kept allocated to avoid layout churn.)

### 8.8 Standings + next
- Flagship vc2 @8s: vsfeel 398 | vszipcl ~211 | eedi3vk2 ~196 (~+90% over
  fastest ref). vc0 @8s ~492 (row kernel at its floor; frame = fixed + vcheck).
- Per-stream footprint: dev ~200MB (pbt dominates) + pad_dev ~9MB + staging
  ~33MB. 8 streams ~= 1.9GB VRAM.
- NEXT: (a) H2D bytes (raw 8.3 + sclip 8 = 16.3MB) are now the biggest serial
  chunk — sclip is workload-duplicated content but can't be assumed; (b) SIMD
  / zero-skip CPU bmask (bytes stay; only if CPU re-limits); (c) default
  streams 4 -> 8 + re-tune; (d) no-mclip strip-batch (secondary).

## Round 9: EEDI3H (horizontal) — pure composition, 25 tests

Requested as a simple maintainable pair to EEDI3. Implementation: EEDI3H is
Transpose -> EEDI3 -> Transpose wired with invokes in Eedi3HCreate
(src/eedi3.cpp) — NO new kernels, NO new host paths, NO instance state. All
validation/numerics/mclip/sclip/dh/field behavior is EEDI3's, applied to the
transposed geometry exactly like vszipcl's EEDI3H (field selects
transposed-row parity, dh doubles transposed height = original width).
Arg string shared verbatim; userData carries the VSPlugin* for the inner
invoke. Ownership rule (bit us once during dev): after mapConsumeNode into an
args map, FORGET the pointer — freeMap releases it; manual freeNode after
that is a double-free (fixed before commit).

Oracle (tests/test_eedi3h.py, 25 tests, all pass):
- transpose-oracle BIT-EXACT (u16 AND f32) on field 0/1/2/3, dh, mdis 3..40,
  nrad 0..3, vcheck 0..3, alpha/beta/gamma corners, mclip, sclip (incl. 2N
  field>1 form), YUV planes — holds by construction (same kernels), guards
  against drift.
- vszipcl.EEDI3H loose sanity (family gap like vertical): ~1.7% px differ,
  max few hundred LSB (measured 594, frac>1 0.02%); bounds 5% / 4096.
- validation: odd width rejected (!dh), dh doubles width, field>1 doubles
  frames + halves duration, masked region == horizontal cubic (mirror of the
  vertical guarantee), dims/props, determinism 1-vs-4 streams.
- Synthetic smoke: EEDI3H ~= EEDI3 speed class (transpose overhead negligible;
  constant-clip numbers not representative, no H bench built — follow-up).
- vsaa/vs-jetpack integration (vsfeel/backend.py): foreign-backend protocol
  needs `supports_mclip` (read by based_aa) + EEDI3/EEDI3H dispatch methods;
  added `supports_mclip = True` and `supports_h = True` to FeelBackend.
  GOTCHA: the loaded `vsfeel` Python package is the site-packages COPY, not
  the repo (scripts run from other dirs miss repo edits) — sync backend.py
  there after changing it (same as the .so copy step). Verified based_aa
  (mclip path) and horizontal antialias (EEDI3H path) produce real frames.

## Full-suite "hang" investigation — NOT an EEDI3 regression
The full test suite appeared to hang (>120s). Root cause: the dfttest file
alone takes ~127s (61 tests), which exceeds the harness's 120s tool cap —
it was never hung, just slow. Verified on the pre-EEDI3 commit ed7ddea
(git stash -> checkout -> clean rebuild -> full tests/test_dfttest.py):
all 61 dfttest tests pass in 127s. EEDI3 is not implicated. Restored the
stash exactly (worktree.diff == worktree-restored.diff). Run the full suite
in the background with a >300s timeout (e.g. timeout 900 python3 -m pytest
tests/ -q).

## Round 10 (speed phase): host CPU path was the real wall — +15% at ns=4, +6% at ns=8

Starting point: 395-405 fps @8s (committed round-9 state). The round-8/9 notes
called vcheck (94 fps) the top target and the row kernel "at its floor". Both
were re-measured this round and both are now known to be WRONG as priorities.

### 10.1 Measurement discipline (the enabling step)
- `tmp/ab.sh A.so B.so reps frames ns` / `tmp/abns.sh`: interleaved same-session
  A/B of two `.so` builds (copies each into the plugin dir, alternates runs).
  Same binary swings a few % between invocations, so only interleaved pairs
  over >=1000 frames are trusted. Every number below is such a pair.
- `tmp/rep_eedi3.sh`: vc0/vc2 interleave (vc0 is the row-only normalizer).
- `tmp/sweep_copy*.sh` / `tmp/sweep_copy_ns.sh`: in-binary knob sweeps.
- `tmp/cpupath2.cpp`: standalone microbench of the CPU staging work at the
  benchmark geometry (3840x2160 GRAY16 -> 1080 interp rows, mdis=20), plus a
  randomized fuzz of the bmask builder against a brute-force dilation oracle.
- The probe now sets EEDI3_PROBE (CMake cache) and reports record/raw/bits/
  sclip separately.

### 10.2 Host CPU path IS the wall (NOT the kernels, NOT vcheck)
Row-kernel ablation (`EEDI3_PROBE=2`: return after the mask pack/span scan,
skipping the entire DP + backtrack + interpolate):
  PROBE=2  vc0 496 | PROBE=0  vc0 490  -> the whole DP/backtrack/interp is ~-1%
  (i.e. free; fully hidden behind the host).
vc0 vs vc2 with the PROBE=0 build: 490 vs 415 -> vcheck is 15% of the frame.
=> at 8 streams the frame is 85% fixed + host, and the GPU kernels are not the
   limiter at all. This overturns the round-8/9 priority list.

### 10.3 build_bmask_row was 6 ms/frame (single largest CPU cost)
The scalar last-propagation dilation scan is a serial dependency chain. The
same window-OR has a closed bit form: out[x] = OR_{t=0..2mdis} (b<<mdis)[x+t],
and a box dilation composes (dil_a ∘ dil_b == dil_(a+b)), so it collapses to
O(log mdis) whole-array 64-bit shift-OR passes:
  1. pack bits with `_mm256_cmpeq_epi8` + `movemask` (32 columns/instruction);
  2. B = b << mdis (bit word shift, masked to the same width);
  3. dilate toward lower x by 2*mdis: radius doubling, acc |= acc >> s;
  4. store as packed uint32 words; clear the bits past `width`.
Gotchas found by the fuzz (KEEP THESE IN MIND if touching it):
- `!` does not compile on an int spec constant in GLSL; but more importantly
  the *accumulator must span width+mdis bits*: B = b << mdis puts set bits up
  to width-1+mdis, and words past (width+63)/64 are needed or the right edge
  silently loses coverage (W=122 mdis=7 exposed it).
- The bits past `width` ended up set where the scalar left them zero (the
  scalar never ORs past width). The shader never reads them, but clear them so
  the output is byte-reproducible.
- The shift form equals the scalar only when width >= 2*mdis (the scan's two
  coverage loops are contiguous then); narrower rows keep the scalar path.
  Validation caps mdis at 40 so the scalar fallback is nearly unreachable.
MEASURED: 6.11 -> 0.28 ms/frame (22x) in the microbench, and +15% end-to-end
at ns=4 (283 -> 326 fps) / +6% at ns=8 (390 -> 405).

### 10.4 Copy flavor: NT load/store (copy_stream_*) beats memcpy IN SITU
The standalone microbenchmark says plain memcpy is 3x faster than the NT
store path (0.14 vs 0.51 ms/frame for the raw gather, 0.75 vs 1.05 for the
blit). IN SITU it is the OPPOSITE: a memcpy build measured blit 6.1 -> 10.4
ms/frame and vc0 481 -> 401 fps. Adding a runtime knob (VSFEEL_EEDI3_COPY:
bit0 upload gathers, bit1 blit, bit2 blit load flavor) and sweeping all 8
combinations same-session:
  ns=8: m0 331 | m1 377 | m2 365 | m3 **400-417** | m5 ~360 | m7 ~408
  ns=4: m0 266-284 | m1 313-327 | m2 275 | m3 **330-332**
=> the existing NT load+store pair (m3) is best everywhere; the "NT loads are
   bad" microbench conclusion does NOT transfer (the in-situ concurrency and
   the VRAM-staging layout change the answer). Lesson repeated from AGENTS.md:
   microbenchmarks propose, only the same-session end-to-end pair decides.
   The knob is kept as a durable tuning knob.

### 10.5 vcheck: parallel rewrite is a WASH, and costs accuracy -> REVERTED
Rewrote ENTRY_VCHECK as a flat rows*WIDTH dispatch taking `pr` (d2p) from the
plain dst row r-1 instead of the previously-vchecked row (round-5 accuracy
policy permits it). Result: vc2 397 vs 396 baseline = NEUTRAL, while the f32
reference comparison moved from <=1 ulp to 1.2e-3 and the vcheck tests had to
be re-baselined. Since it buys nothing measurable, it was REVERTED and the
serial walk + ENTRY_VCOPY split stay (tests 50/50, u16 still bit-exact).
Context for why: vc0 490 vs vc2 415 says vcheck is 15% of the frame and ~1 WG
per plane already hides inside the other streams' work at 8 streams.

### 10.6 Queue cap (VSFEEL_EEDI3_QUEUES) — no win, kept as a knob
Swept 1/2/3/4/6/8 at ns=8: 343 / 399 / 414 / 413 / 418 / 416. Floors at 3
(one queue per stream is fine here); default unchanged.

### 10.7 Stream count knee (measured, 1500f)
ns: 4 -> vc2 332 / vc0 480; 6 -> 399/493; 8 -> 410/502; 10 -> 418/504;
12 -> 419/503; 16 -> 413/498. Knee ~10; kept at 8 (user decision: 2% between
8 and 10 is not worth the VRAM).

### 10.8 Host-path ablation ladder (env-gated, all default off; KEPT)
VSFEEL_EEDI3_NOBLIT / NOSCLIP / NORAW / NOH2D / NOVC each skip one host stage
(output is garbage; diagnostics only). At ns=8/1000f on the bmask+kept build:
  base 431 | no blit 473 | no sclip 448 | no raw 501 | no h2d 494 |
  no vcheck 443 | no blit+sclip 509 | all off 718
=> every remaining host stage costs roughly the same (30-70 fps each) and NONE
   of them is individually dominant; they are additive traffic. This is the
   signature of a shared resource (host memory bandwidth / store buffers)
   rather than a single hot loop. Confirmed independently: the machine does
   ~55-60 GB/s aggregate for 8-thread NT copies and ~75 GB/s single-thread
   (tmp/bw.cpp), and EEDI3's copies already run at ~4-6 GB/s per stream
   (~40 GB/s aggregate at 8 streams) = at the wall.
=> REDUCING BYTES is the only remaining host lever, and exactly one duplicate
   existed (below).

### 10.9 Kept-row write merged into the upload gather (KEPT, +3.5%)
The tight kept-row gather (src -> upload) and the destination's kept-row copy
(src -> dst) read the SAME source rows in the SAME order. The second read was
8.3 MB/frame of DRAM traffic for nothing. Both are now written in the gather
loop while the source row is hot; the blit loop only handles interp rows.
MEASURED: 415 -> 431 fps (+3.5%), tests 50/50.

### 10.10 ReBAR direct upload — the big one (KEPT, +29%)
The old path wrote the 17 MB/frame upload into cached system-RAM staging and
then paid a vkCmdCopyBuffer H2D plus a transfer->compute barrier. An earlier
round (8.3, 7.8) tried binding staging directly and collapsed to 178 fps, and
allocating staging as host-visible device-local collapsed to 27.6 fps — so
"host-visible VRAM is poison here" was recorded as a lesson. THE LESSON WAS
WRONG, or rather incomplete: those attempts used ordinary cached stores. The
7900XTX's host-visible device-local types (3/4) are UNCACHED, so cached stores
to them are pathological — but NON-TEMPORAL stores bypass the cache and are
fast. nnedi3 has shipped exactly this (NT stores into a ReBAR-mapped
host-visible VRAM buffer + `_mm_sfence()`, no H2D at all).

Implemented: a per-resource `up_dev` buffer allocated
DEVICE_LOCAL|HOST_VISIBLE|COHERENT, mapped, and written by the CPU with the
existing NT path; the pad / row / vcheck kernels read it directly. No H2D
copy, no transfer barrier, and the upload bytes stop competing with the
download for system-RAM bandwidth. `_mm_sfence()` before submit drains the
store buffer. VSFEEL_EEDI3_NOREBAR=1 restores the old path.

Two BUGS found and fixed while porting (both silent-corruption class):
1. The descriptor pool had maxSets == num_streams; the ReBAR path needs a
   SECOND descriptor set per resource (the pad kernel's binding 0 is the raw
   upload while the row kernel's binding 0 is the built pad — the exact
   aliased-binding hazard AGENTS.md warns about). Pool now allocates 2 sets
   per stream. Symptom was vkAllocateDescriptorSets failing.
2. When building the pad kernel's set, binding 8 (its built-pad OUTPUT) must
   stay pad_dev; copying the row set's infos[] wholesale pointed b8 at up_dev,
   so the pad kernel wrote its output into the input buffer. Symptom: luma
   mismatched by 65535/0 on isolated pixels. Tests caught it immediately.

MEASURED: 431 -> 555 fps in interleaved A/B (+29%). Tests 50/50, u16 still
bit-exact vs eedi3vk2 on every config.

### 10.11 State after round 10 (checkpoint)
- vc2 @8s: 513-570 fps (bench.py same-session pair: vsfeel 512.9 vs
  eedi3vk2 193.5 vs vszipcl 180.1) = **~2.65x the fastest reference**.
- vc0 @8s: ~690-750 fps (row kernel + host only).
- Stream knee re-swept after ReBAR (1500f): ns 4/6/8/10/12/16 =
  465/541/571/576/587/569. Knee ~12; DEFAULT KEPT AT 8 (within 3% of 12, and
  the user preferred 8 for VRAM sanity). Re-sweep after any further byte
  reduction.
- Tests: 50/50, u16 bit-exact vs eedi3vk2 (nothing in this round traded
  accuracy — the parallel-vcheck experiment was reverted).
- Durable knobs added: VSFEEL_EEDI3_QUEUES (no win, floors at 3),
  VSFEEL_EEDI3_COPY (NT flavor sweep; 3 = NT loads+stores is best),
  VSFEEL_EEDI3_NOREBAR (H2D fallback), VSFEEL_EEDI3_NOBLIT/NOSCLIP/NORAW/
  NOH2D/NOVC (ablation ladder), EEDI3_PROBE (shader ablation cmake var).
- NEXT TARGETS (ranked, all still unclaimed):
  1. The remaining host copies are at host-bandwidth parity, so the only
     lever left is BYTES. sclip (8.3 MB/frame) is the largest single upload
     and is workload-duplicated content; mclip bits are already 0.5 MB.
     vout currently writes to staging and is downloaded; a ReBAR download
     buffer (GPU writes to host-visible VRAM, CPU reads) is the mirror image
     of 10.10 and is UNTRIED — that would remove the D2H-side staging
     competition entirely.
  2. vcheck is ~15% (throughput/traffic, not latency — see 10.5).
  3. The row kernel needs no further work — do not micro-tune it.
- PENDING IDEA (not yet tried): `pbt` (WIDTH x TPITCH int8 per row, ~170 MB at
  mdis=20) is written to VRAM and read back by the SAME subgroup right after
  the DP loop. PROBE=1 (zero the stores) measured NO gain, but the FOOTPRINT
  is real: a per-row ring of slots (or LDS staging) would cut VRAM per stream
  from ~200 MB, which only matters if the stream count becomes VRAM-limited.

## Round 11: the mask conversion was a whole extra graph pass (+8%)

Starting point 558 fps @8s (round-10 state). End 604-638.

### 11.1 The in-graph mclip conversion is expensive
`Eedi3Create` forced every non-Gray8 mask through
`SetFrameProps(_Range=1) -> resize.Point -> Gray8` so the kernel could test
`byte != 0`. For the AA benchmark (and for any 16-bit clip, where based_aa
scales the mask to the clip depth) that is a FULL EXTRA FRAME PASS in the
graph, running inside the timed region every frame:
    read 16.6 MB + write 8.3 MB per frame at the 2x-2160p bench geometry.
Probe (tmp/aa_mb.vpy, mask converted to Gray8 BEFORE the RAM cache so the
conversion leaves the timed region): 699 fps vs 566 for a Gray16 mask = +23%
upper bound. (Careful: that probe is an upper bound, not the achievable win —
it also moved the conversion out of the timed region.)

### 11.2 Native 16-bit mask path (KEPT, +8%)
zimg's full-range 16->8 reduction is monotonic and round(v*255/65535), so
`converted != 0` is EXACTLY `v >= 129`. Verified exhaustively over all 65536
u16 values (tmp/thresh_full.py: 0 mismatches, max output 255). So for a Gray16
integer mask the conversion node is pure overhead: read the u16 mask directly
(with only the needed kept-parity rows touched) and pack `v >= 129`.
`build_bmask_row` gained a `mask16` variant using `bmask_bits16`
(xor 0x8000 to make the compare signed, cmpgt, movemask, then a 5-step
even-bit compress 32->16). Everything else (float, 10/12/14-bit masks) keeps
the reference conversion.
MEASURED: 558 -> 602 fps @8s in an interleaved A/B (+8%). 75/75 tests pass.

**BUG CAUGHT (off-by-one in the compare constant).** `cmpgt` is STRICT, so the
constant must be `128 - 32768`, not `129 - 32768`; the latter silently tests
`v >= 130`. A binary mask (0/65535) cannot see it, and neither can a
uniformly-random mask, but a mask whose values cluster around the threshold
does. Found only by a purpose-built oracle: same inputs through the OLD binary
and the NEW binary and a bit-compare (tmp/mask_regress.py), over mask
distributions {binary, values 120..139, uniform random, sparse} x 6 configs.
Lesson: for a semantic-preserving rewrite, build the near-threshold input
explicitly — the existing tests' binary masks would never have caught this.
After the fix the new path is BIT-IDENTICAL to the old on all 24 cases.

### 11.3 Copy-flavor knob swept again (VSFEEL_EEDI3_COPY, 8 combos)
bit0 = upload gathers, bit1 = blit, bit2 = blit load flavor. Same-session:
   ns=8: m0 331 | m1 377 | m2 365 | m3 **400-417** | m5 ~360 | m7 ~408
   ns=4: m0 266 | m1 313 | m2 275 | m3 **330**
=> NT load + NT store (m3) stays best everywhere; a plain-memcpy build is
   SLOWER in situ (0.14 vs 0.51 ms/frame in the isolated microbenchmark, but
   blit 6.1 -> 10.4 ms/frame and vc0 481 -> 401 fps in place). The isolated
   microbenchmark proposes, only the in-situ pair decides.

### 11.4 Things that did NOT work (measured, do not retry blindly)
- **`vout` to dev_buf + SDMA D2H** instead of writing staging directly:
  553 vs 568 fps. Round 8.7's choice still wins even with the host now cheap.
  (Kept as VSFEEL_EEDI3_VOUTDEV, default off.)
- **A single contiguous blit** instead of 1080 strided row copies: 525 vs 555.
  The blit is traffic-bound, not pattern-bound.
- **Queue cap** (VSFEEL_EEDI3_QUEUES) 1/2/3/4/6/8 = 343/399/414/413/418/416:
  floors at 3, no win, default unchanged.
- **Skipping the pad kernel** (VSFEEL_EEDI3_NOPAD): only +3% (554 -> 571), so
  the pad builder is not worth eliminating. The pad kernel costs ~3%.
- **Host-pointer import for a direct-to-`dst` GPU write** would remove the
  entire download+blit (the biggest single ladder item, see 11.5), but
  `minImportedHostPointerAlignment = 4096` on this RADV setup while
  VapourSynth frame planes are only 64-byte aligned (measured). Importing
  would mean aligning down to a page that may precede the allocation.
  VK_EXT_external_memory_host is now enabled and the alignment is queried and
  stored (`VK_Device::host_import` / `host_pointer_alignment`) so a future
  attempt starts from a known state.

### 11.5 Ladder after round 11 (ns=8, 1000f, all env-gated)
  base 604 | no blit 725 | no sclip 664 | no raw 702 | no h2d 609 |
  no vcheck 666 | no blit+sclip 774 | all off 813
So: blit ~121 fps (20%), raw ~99 (16%), vcheck ~62 (10%), sclip ~60 (10%),
h2d ~5 (0%, ReBAR already removed it). Every host stage costs about the same
and they are additive -> shared-resource (host memory path) bound.
CPU traffic per output frame is now ~67 MB (src kept 8.3 read; upload 17.1
write; mask 8.3 read; sclip 8.3 read; staging 8.3 read; dst 16.6 write).
The Ryzen 7950X does ~54 GB/s aggregate with 8 threads and ~101 GB/s with 2
(tmp/bw.cpp) — so at 8 streams we are around 75% of the practical ceiling, not
saturated. `all off` = 813 is the GPU/kernel floor.

### 11.6 Stream-count knee (1500f, round-11 build)
ns 2/3/4/6/8/10/12/16/20/24/32 = 286/360/435/556/592/611/634/635/612/634/633.
Knee ~12-16 (plateau). Interleaved 8 vs 12 = 615 vs 638 (+4%).
DEFAULT KEPT AT 8 per the user's explicit preference; 12 is the fastest.

### 11.7 Open issue: rare nondeterminism (PRE-EXISTING, not from this round)
In the 24-instance stress test (tmp/mask_regress.py) the case
`binary | field=0, mdis=3, nrad=3, vcheck=3` produced two different outputs
across runs of the SAME binary (1 in ~4 runs). It reproduces with the round-10
binary too, so it predates round 11. NOT reproducible in 90 sequential
evaluations of that exact config in one process (tmp/nondet.py, ns=1/4/8),
nor in the test suite. So it needs many concurrent EEDI3 instances (shared
device/queues) to surface. Worth chasing: a race in a filter is a real bug.
Repro: run tmp/mask_regress.py several times with the same .so and diff.

## Round 12: benchmark-harness memory bug (important for trustworthy numbers)

The fp32 AA numbers were bimodal: the SAME binary measured 143 / 188 / 206 /
221 / 246 fps on consecutive runs while the GPU-bound reference (vszipcl)
stayed flat at 186-203 on every run. That is the AGENTS.md
"one plugin swings, the other stays flat" signature, so it was triaged rather
than averaged away.

ROOT CAUSE (harness, not the filter): `make_aa_vpy` sets
`core.max_cache_size = 1024*48` (48 GB) AND the benchmark separately pins a
Python-side list of decoded 2x frames. Those two caches are ADDITIVE. At 2x
2160p fp32 a frame is ~33 MB (luma) + ~16.6 MB (mask), so the 250-frame fp32
cap (chosen by the old `aa_cap` heuristic) plus VS's own cache ran past the
62 GB box and the resulting memory pressure hit the CPU-bound plugin
(vsfeel) while leaving the GPU-bound one (vszipcl) untouched.

FIX: keep `max_cache_size` at 48 GB (lowering it to 8 GB collapses EVERY
plugin -- measured vsfeel 61 vs 220 fps and vszipcl 39 vs 200, because the
timed region starts re-running the decode/mask chain) and instead cut the
fp32 Python cache cap 250 -> 120 frames (~6 GB). After the fix, three
consecutive 2000-frame fp32 runs: vsfeel 254.7 / 246.5 / 260.4 vs vszipcl
202.7 / 201.5 / 201.4 -> stable and ~1.25x.
LESSON: a bimodal measurement is a bug to chase, not noise to average; and
when tuning a benchmark's memory, change ONE cache at a time.

## Round 12: honest README numbers (medians of 3 x 2000 frames, ns=8, real AA)

| depth | vsfeel | vszipcl | speedup |
|---|---|---|---|
| u16  | 613 (591/621/613) | 211 (213/194/211) | 2.90x |
| fp32 | 255 (255/247/260) | 202 (203/202/201) | 1.26x |

Earlier single-run README figures (568/204 and 216/187) were taken from one
run each and the fp32 one was inflated by the cache bug above; the numbers
above are medians with the fixed harness.

## Round 13: direct-to-frame output via VK_EXT_external_memory_host — the
## last big idea, built, bit-exact, and a measured LOSS (kept default-OFF)

This was the direct-to-frame idea (2a) in the old, now folded-in idea list: the upload gather (~16%)
and the dst blit (~20%) exist only because kernels cannot address VapourSynth's
own frame memory, so importing a plane as a Vulkan buffer should delete both. It was
blocked on `minImportedHostPointerAlignment = 4096` vs VapourSynth's 64-byte
plane alignment; the resolution is that only the *import* has to be page
aligned, the *buffer binding* may sit at the plane's offset inside it
(`vkGetBufferMemoryProperties.alignment` is 64 here, and the plane offset is a
multiple of it). Groundwork + implementation:

- `import_plane_host_memory()` in `src/eedi3.cpp`: page-aligns the plane
  pointer down, imports `[page, page + round_up(offset+plane_bytes, page))`
  with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT`, picks the
  first HOST_VISIBLE|HOST_COHERENT type RADV reports for the pointer (type 5,
  sysRAM heap 0 — the same type the staging buffer uses) and binds at the
  offset. Every byte the GPU touches is inside the plane.
- New `ENTRY_BLIT` kernel in `src/eedi3.comp` (per depth, in `CMakeLists.txt`)
  spreads the tight interp rows from dev_buf into the imported frame at the
  frame's own row pitch and interp parity; one blit descriptor set per plane
  (`desc_set_blit[]`) is re-pointed at that plane's import every frame.
- `dst_host` (env `VSFEEL_EEDI3_DSTHOST`, **default 0**) selects it; it forces
  the pre-existing `vout` -> dev_buf layout (`vout_in_dev()`), so the fallback
  is exactly the old VOUTDEV path. `VSFEEL_EEDI3_NOXFER` = import without the
  blit (garbage output) is the ablation that separates the two costs.
- 50/50 eedi3 tests (bit-exact vs eedi3vk2) pass with it ON **and** OFF.

### It works, and it is bit-exact — but the import is not free

Isolated probes (scratch in tmp/hostimport/, gitignored like these notes):
import works and GPU
writes are CPU-visible; import+destroy of a fresh 16.6 MB region costs
0.43 ms host-side with no memory pressure, ~2.2 ms with 20 GB of ballast.
1 contiguous transfer region = 0.38 ms, 1080 strided row regions = 0.47 ms.

In situ (2x-upscaled jpbd AA bench, field=3, 2000f, ns=8, same-session):

| harness cache | staged blit | direct (import+blit) |
|---|---|---|
| 50 frames  | 641 | **697** |
| 150 frames | 611 | 458 |
| 500 frames | 613 | 390 |
| real bench.py, 500 | 603 | 372 |

Synthetic BlankClip 3840x2160 (no harness cache), ns=8: staged 633,
import-no-blit 787, direct 720. ns=2/cache=150: staged 287, direct 324.

### The mechanism (measured, not inferred)

The hbench probe was extended to split `submit` from `fence_wait` and to
sample periodically (`VSFEEL_EEDI3_HFRAME=-1`): `vkQueueSubmit` is 0.1-0.35 ms
and the import is 0.1-0.4 ms at every cache size, but `fence_wait` grows
1.8-3.5 ms (cache 50) -> 6-22 ms (150) -> 12-31 ms (500) while
`gpu_busy_percent` reads 95-100% during the run. A per-frame import therefore
puts *GPU-visible work* (VM page-table map/unmap + TLB sync for a 16.6 MB BO)
on the GPU, and 8 streams doing it every frame saturates the GPU.

Controlled arms at cache=500, ns=8 (the decisive table):

| arm | fps |
|---|---|
| staged CPU blit | 608 |
| staged + `vout` in dev_buf, **no import** | 588 |
| per-frame import, GPU never touches the imported buffer (NOXFER) | 487 |
| per-frame import + blit kernel | 390 |

So the import alone costs ~20% and the blit kernel ~16% more; the 20% CPU
blit it removes cannot pay for either. It only wins where the CPU, not the
GPU, is the limiter: light frame-buffer churn (harness cache 50) or low stream
counts (ns=2 +13%). Leaking the imports instead of freeing them is worse (269
fps at cache=150): the cost tracks the number of live pinned BO mappings, so
an import *cache* would not fix it either.

### Two reusable findings for other filters

- **Never use a multi-region `vkCmdCopyBuffer` for strided row copies.** 1080
  regions of 7680 B measured 0.47 ms in an isolated probe but ~13 us *per
  region* in situ (ns=8): the direct path went 710 fps with 1 region vs 326
  with 1080, same destination. A compute blit kernel costs ~1 ms of frame
  latency for the same 8.3 MB. (The same probe showed GPU stores to system RAM
  at ~25 GB/s vs ~279 GB/s to VRAM.)
- **Per-frame host-pointer imports are a GPU cost on RADV/amdgpu**, not a host
  cost. If a future filter wants zero-copy frame access, it needs the import
  to be stable across frames (a fixed set of buffers), which VapourSynth's
  per-frame allocation does not give.

Status: code kept in tree, **default OFF**, `dst_host` documents the numbers.
Re-test if RADV's import path gets cheaper or if a future API (e.g. a
stable-address frame pool) makes one import last many frames.

---

## Round 14: the benchmark's mask was a no-op — fixing it changes the whole cost model

### The bug (found while sizing round-13 leftovers)

`benchmark/bench.py::make_aa_vpy` built the mclip with

```python
mask = Morpho.binarize_mask(mask, scale_mask(60, 8, 16))   # WRONG
```

but `Morpho.binarize_mask` re-scales its `midthr` argument **from the 32-bit
float range** to the clip's format (`scale_mask(t, 32, clip)` inside
vsmasktools). Passing an already-16-bit-scaled 15420 therefore becomes
65535, nothing passes, and **the mask came out 100% zero**. Real `vsaa` does
`scale_mask(mask_thr, 8, 32)` (`vsaa/funcs.py:171`), i.e. threshold 15420.

Measured (700f, num_streams=8, same session):

| bench config | vsfeel | vszipcl |
|---|---|---|
| `scale_mask(60,8,16)` (all-zero mask, as shipped) | 593 | 203 |
| `scale_mask(60,8,32)` (real based_aa mask) | 274 | 202 |

With an all-zero mask `xmin >= WIDTH` for **every** row, so the row kernel
takes its parallel-cubic branch, the DP/backtrack never runs, vcheck skips
every row and vcopy copies sclip into vout: the entire GPU pipeline is dead
code and EEDI3 degenerates into "kept rows from src + interp rows from
sclip". That also explains the round-10 result "deleting the DP changed fps
by -1%": the DP was never executing in the graded benchmark.

**The recorded cost model in rounds 10-12 describes the degenerate path.**
Fixed in `benchmark/bench.py` (one line + comment); the numbers below are the
honest path.

### Honest cost model (real mask, 3840x2160 GRAY16, field=3, mdis=20, vcheck=2, ns=8)

Env-gated ablation ladder, 900-frame runs, same session:

| stage | cost |
|---|---|
| vcheck + vcopy | **~18-27%** |
| pad | ~8% |
| raw gather | ~8% |
| sclip gather | ~7% |
| blit | ~4-5% |
| ReBAR upload (vs `NOREBAR`) | **+14%** (keep it) |

So on the honest path the **vcheck is the largest single item**, the host
copies are each small, and the former headline numbers (~20% blit, ~16%
upload gather) do not transfer. Anything measured only in the degenerate
config must be re-measured.

### LDS ping-pong vcheck (round-6 "flat-to-worse" dead end, RE-TESTED: +3%)

`eedi3vk2`'s vcheck carries the modified previous row in shared memory
(`tlineSh[2][MAXW]`) and writes it back one row late, so the d2p term never
reads global memory. vsfeel read it back from `vout` — which is host-visible
**staging** when `vout_dev` is off, i.e. a GTT read per `dirc != 0` pixel.
The old "LDS is slower (the 30 KB of LDS was the cost)" verdict was measured
in the degenerate config where the vcheck does nothing.

Implemented as `ENTRY_VCHECK -DVCHECK_LDS=1` (new `vcheck_lds` pipeline,
`MAXW=4096`, chosen only when `width <= MAXW` and
`maxComputeSharedMemorySize >= 2*MAXW*4`; `VSFEEL_EEDI3_VCLDS=0` forces the
global form). The LDS walk carries **every** row, so it needs no ENTRY_VCOPY
dispatch and the host skips it on that path.

Measured, real mask, 4 interleaved pairs of 900-frame runs:

| arm | fps |
|---|---|
| global d2p | 294 / 304 / 289 / 293 (mean 295) |
| LDS d2p | 302 / 304 / 304 / 305 (mean 304) |

**+3% and tighter variance; kept as the default.** Bit-exact: 50/50
`tests/test_eedi3.py` vs eedi3vk2, plus byte-identical against the global
path over 20 real 3840x2160 frames with an all-zero mask (all rows empty →
vcopy vs inline) and an all-white mask (all rows non-empty).

### Pad parity skip (implemented: +0.8%, bit-exact)

`pad_get` is only ever called at pad rows `MARGIN_V + dst_row` with
`dst_row = field + 2r +/- {1,3}`, so every **read** pad row has parity
`(field+1)&1`; the other parity is written and never read. `ENTRY_PAD` now
returns early for it (`pc.pad_skip_parity`, `VSFEEL_EEDI3_PADPAR=0` restores
the full build). Real mask, 4 interleaved pairs: full 303.8/302.7/301.9/304.2
vs skip 306.4/306.1/305.3/304.4 (**+0.8%**, 4/4 pairs). Verified bit-exact
over 20 real frames on both the all-zero and all-white mask configs, and by
the test suite. (The whole pad kernel is only ~8%, hence the small win.)

### Queue cap re-sweep (the round-12 "floors at 3, no win" is config-specific)

On the honest path `VSFEEL_EEDI3_QUEUES` = 3 → 256-269 fps, 4/5/6/8 →
294-322 (all within noise of each other, 4 ≈ 6 ≈ 8). Default
`min(num_streams, queue_count)` = 8 stays. First-pass numbers at only 700
frames said 4 was +14%; at 900 frames that vanished — the short run was
riding the clock ramp. **Grade queue/stream sweeps at >=900 frames.**

### Round-13 leftovers re-measured on the honest path

The verdicts and their mechanisms now live in the consolidated "Remaining work /
do-not-retry" section at the end of this file; what belongs here is only the
measurement that changed from the pre-round-14 record:

- **`vout` in dev_buf + SDMA D2H (`VSFEEL_EEDI3_VOUTDEV=1`)**: 289 vs base
  313-323, and 327 vs 375 with vcheck off. Still a loss; the round-11 verdict
  survives the config change.
- **Host micro-opts**: `record_command_buffer` measures 0.004-0.012 ms/frame
  (0.2% of a 3.3 ms frame) and `vkQueueSubmit` 0.06-0.07 ms — the reason the
  prerecord/`vkResetCommandBuffer`/single-bind variants stay unimplemented.
- **Shader micro-opts and the accuracy-for-speed vcheck**: not attempted this
  round; both are open in sections A2/A1 below, and both need a fresh
  measurement on the honest path before any old verdict (row kernel hidden;
  unvchecked d2p neutral) is inherited.

### OPEN BUG (pre-existing, not from this round): nondeterministic output with a structured mclip

vsfeel EEDI3's output is **not reproducible run-to-run** on real content with
a real (structured) edge mask. Reproducer: the masked bench vpy
(`tmp/bench_masked_vsfeel.vpy`) run twice through `vspipe -e 9` and compared:
~0.5-1.1 M differing bytes. Evidence it is vsfeel, not the content or the
harness:

- the same chain's `clip`, `mclip` and `sclip` outputs are byte-identical
  run-to-run (the harness caching trick is deterministic);
- **`field=3, mclip, vcheck=0` is nondeterministic** (so the row kernel's
  `dst` alone differs — the vcheck is not involved);
- `num_streams=1` is nondeterministic (rules out cross-stream/resource races);
- `VSFEEL_EEDI3_NOREBAR=1` (old staging+DMA upload) is nondeterministic
  (rules out host-write visibility into host-visible VRAM);
- **no mclip, an all-zero mask and an all-white mask are all deterministic**
  — the trigger is a mask with both zero and nonzero regions;
- **`eedi3vk2` is deterministic on the identical chain and parameters**, and
  the 50 EEDI3 tests (noise clip, half-white/half-black mask) pass bit-exactly
  against it, so the test suite does not cover the trigger.

So it is a GPU-side race or uninitialized read inside the row kernel's
masked-column path, data-dependent on the mask pattern. One hypothesis was
tested and **falsified**: adding `subgroupMemoryBarrierBuffer()` before the
`pbt` store->`tileSh` load barrier (the round-trip is only ordered by an
execution barrier, which carries SubgroupMemory semantics under the Vulkan
memory model) left the non-reproducibility unchanged (883 K vs 1.06 M
differing bytes), so the pbt round-trip is not it. Other candidates not yet
tested: the last column (`n = min(BT_TILE, WIDTH - 1 - x0)` means column
`WIDTH-1` is never written for non-empty rows — stale `dst`/`dmap`, stable
per run but wrong), the `fcarry` walk, and the `bmaskSh` span scan. This
should be fixed before the graded benchmark's output is trusted; the *fps*
numbers are unaffected (same work every frame).

---

## Round 15: the row kernel is the dominant cost on the honest path (A2 answered)

Baseline re-established at the benchmark's own defaults (bench.py `--filter eedi3`,
2000 frames, ns=8, real based_aa mask, 3840x2160 GRAY16, field=3, mdis=20,
vcheck=2), same session:

| plugin | fps |
|---|---|
| vsfeel vc2 | **264.1** |
| vsfeel vc0 | 308.5 |
| vszipcl | 181.5 |
| eedi3vk2 | 125.2 |

(1.45x vszipcl, 2.11x eedi3vk2 on the honest path. The ~2.9x headline of
round 12 was the degenerate all-zero-mask path.)

**A2 (row-kernel ablation) is now answered, and it overturns round 10.**
Built with `-D EEDI3_PROBE=2` (ENTRY_ROW returns right after the mask
pack/span scan: no DP, no backtrack, no interpolate, only the rempty flag):

| arm | vc2 | vc0 |
|---|---|---|
| PROBE=2 (no row kernel) | **564.3** | **693.6** |
| real (PROBE=0) | 264.1 | 308.5 |

So on the honest path the row kernel (DP + backtrack + interpolate) costs
**more than half of the vc0 frame** (vc0 308 -> 694 without it), and the
vcheck is 264 -> 308 (~14%). Round 10's "deleting the DP changed fps by -1% /
the row kernel is fully hidden" was measured with the round-14 all-zero mask,
where every row takes the parallel-cubic early-return branch and the DP never
executes. **The row kernel, not the vcheck and not the host, is the top lever
on the honest path.** All row-kernel micro-verdicts in this file that were
recorded before round 14 (round 6 especially: rolling-with-mclip, the
carry-loop floor, 2-row WG, subgroup/barrier equivalence) were measured in
that degenerate config and must be re-measured before being inherited.

NOTE the PROBE=2 vc0 694 is host-bound (694 x 67 MB/frame ~= 46 GB/s, the
practical host ceiling), so 694 is a floor, not the true "no row kernel"
number; the row kernel's share is therefore at least the 308 -> 694 ratio.

### 15.A Row-kernel cost decomposition (probes, combined kernel, mdis=20)

Added probe levels 4 (skip the pbt stores entirely), 5 (return after the DP),
6 (backtrack reads the PREVIOUS row's pbt, i.e. same kernel, no same-workgroup
store->load dependency). 2000f, ns=8, honest mask, same session:

| arm | vc0 | vc2 |
|---|---|---|
| real (coupled) | 308 | 264 |
| PROBE=4 no pbt stores (backtrack reads stale pbt) | 728 | - |
| PROBE=5 DP + stores, no backtrack | 726 | - |
| PROBE=6 no same-WG dependency | **718** | **606** |

The DP+stores alone and the backtrack alone are each nearly free, but together
they cost 2.4x. The interaction is the `pbt` store->load round trip **inside
one workgroup**: the wave must make its own byte stores visible before the
backtrack's loads can issue, and only PROBE=6 (reads a row written by a
different workgroup) escapes it. Everything else about the row kernel was
already at the floor.

### 15.B Two-dispatch split of DP and backtrack: NEUTRAL (dead end)

Split `ENTRY_ROW` into a DP pass and a new `ENTRY_BT` backtrack pass (new
shader entry, CMake variant, host pipeline + dispatch + barrier) to move the
pbt read across a dispatch boundary. Output stayed correct (bit-exact vs
`eedi3vk2`; the one-pixel real-content delta was the pre-existing race below),
and it measured **322 vc0 / 295 vc2 versus 326 / 295 for the coupled kernel** —
i.e. the dispatch/phase boundary costs exactly what removing the same-WG
dependency saved. Variant kept only as `src/eedi3.comp` history in
`tmp/eedi3_split_direct.comp`; do NOT retry the split without a different
overlap story.

### 15.C Walk reads pbt directly instead of via LDS staging: +9-13% (KEPT)

The backtrack's staging loop loaded the full `TPITCH` (41) bytes of *every*
column into LDS (157 KB/row), but the serial lane-0 walk consumes exactly
**one** byte per column (`CENTER + f`). Staging therefore pulled ~41x more
data through the DP's store->load dependency for nothing. The walk now reads
`pbt[pbt_row + x*TPITCH + CENTER + f]` straight from global; the tile loop is
kept (it still bounds the serial chain and lets the workgroup interpolate each
tile in parallel). Measured: vc0 326 -> 368, vc2 295 -> 329.

### 15.D Rolling-window costs under mclip: +76% — THE BIG ROUND-15 WIN (KEPT)

Round 6 recorded "per-lane rolling kept alive WITH mclip: no gain" and gated
the rolling path behind `HAS_MCLIP == 0`. That measurement was taken with the
all-zero mask, i.e. with **zero unmasked columns**, so the rolling path never
executed. With the rolling gate removed (both paths use
`roll_seed`/`roll_push`/`combine_interior`), unmasked columns cost one 9-load
seed plus 9-load pushes instead of a ~51-load full `conn_cost` recompute per
direction. 50/50 EEDI3 + 25/25 EEDI3H tests pass, and the mclip tests assert
`max diff == 0` against `eedi3vk2` (field 1/2/3 and dh), so the rolling sums
stay bit-exact for integer pad (sums of exact integers < 2^24, any order).

Interleaved same-session A/B over 2000 frames at ns=8, vc2:

| arm | rep1 | rep2 |
|---|---|---|
| committed round-14 build | 290.3 | 290.5 |
| round 15 (15.C + 15.D) | **513.5** | **512.7** |

### 15.E Round-15 final standings (benchmark defaults: 2000f, ns=8, honest mask)

| plugin | fps | vs fastest ref |
|---|---|---|
| **vsfeel vc2** | **512.9** | **2.55x vszipcl / 3.85x eedi3vk2** |
| vszipcl | 201.2 | |
| eedi3vk2 | 133.1 | |
| vsfeel vc0 | 615.3 | |

Session-to-session drift is large (the *same* committed binary measured 264 at
the start of this round and 290 later), so only interleaved A/B ratios are
trustworthy; the +76% is an interleaved pair, the absolute fps are one run.

### 15.F Open bug status: the structured-mclip nondeterminism is FREQUENT

**RESOLVED in round 16.2** — it was a stale-`dst` write: the backtrack's tile
loop `break` skipped the cubic write for every column left of `xmin`. Kept here
because the *characterisation* below was accurate and is what made the cause
findable; see 16.2 for the mechanism and the fix.

Still present at the time of round 15, and worse than the round-14 note implied. 8 consecutive runs of
the same binary on the real chain (2 frames, `tmp/aa_m20.vpy`) produced three
distinct outputs for the committed build and **five** distinct outputs for the
round-15 build; the differences are tens of pixels per frame with deltas up to
a few hundred LSB (DP argmin flips). It is timing-sensitive — the faster build
seems to hit it more often — and it predates every change in this round
(the committed build produced 22758 differing px vs `eedi3vk2` over 4 real
frames while round 15 produced 5902, so round 15 did not make agreement
worse). **The test suite's noise-clip mclip oracle is bit-exact and passes, so
the test suite does not see it.** Fixing this is now the top *correctness*
item; the fps numbers are unaffected.

### 15.G Why round 15 gains far more at u16 (+77%) than at fp32 (+21%)

Measured because the asymmetry looks like the round-14 degenerate-mask artifact.
It is not: it is Amdahl's law over two genuinely different fixed-cost floors.
Same session, `--bits` sweeps, vc0 (row kernel only) and vc2, honest mask:

| depth | r14 vc2 | round-15 vc2 | no-row-kernel floor (PROBE=2 vc0) | round-15 vc0 |
|---|---|---|---|---|
| u16 | 290 | 513 | 726 | 615 |
| fp32 | 250 | 303 | **339** | 340 |

Frame-time arithmetic (vc2):
- u16: old frame 3.45 ms, of which the row kernel was **1.68 ms (49%)**; new
  frame 1.95 ms, of which the row kernel is **0.18 ms (9%)**. The floor (host +
  pad + mask + vcheck traffic) is 1.77 ms = 91% of the new frame.
- fp32: the no-row-kernel floor is 339 fps = 2.95 ms and the real build measures
  340 fps, i.e. **at fp32 the row kernel is now completely hidden behind the
  floor**. There is nothing left to win in the fp32 row kernel; the frame is
  ~100% floor.

Why the fp32 floor is 2.1x heavier in fps terms:
1. every byte in the path doubles (luma pad, upload gathers, sclip, dst
   download, vcheck taps). The round-14 ladder said those stages cost about the
   same each and are shared-resource bound, so doubling them nearly halves fps.
2. **the fp32 benchmark's mask is GRAY32 float**, so `Eedi3Create` inserts the
   reference `SetFrameProps(_Range=1) -> resize.Point -> Gray8` node
   (`src/eedi3.cpp`: `mclip_native16` is set only for 16-bit *integer* masks).
   That is a whole extra 2x-2160p pass in the graph, inside the timed region,
   every frame — exactly the cost round 11 deleted for GRAY16. The u16 run's
   GRAY16 mask takes the native path and pays nothing.

CONSEQUENCE for the next fp32 attempt: do NOT tune the row kernel. Either
(a) extend the native-mask fold to float masks (done in round 16 — see below),
or (b) attack the doubled bytes (the round-14 ladder items), because at fp32
they are now the entire frame.

---

## Round 16: native float mask + the structured-mclip bug was a stale-`dst` write

### 16.1 Float masks are handled natively (+2% at fp32, and structurally correct)

`based_aa` passes `mclip` in the CLIP's format (`vsaa/funcs.py:232`: the mask is
built from `luma` and follows its depth), so a float clip gets a **GrayS**
(32-bit float) mask — which `eedi3vk2` consumes directly. vsfeel only folded
Gray16 integer masks natively (`mclip_native16`), so every float mask went
through the reference `SetFrameProps(_Range=1) -> resize.Point -> Gray8` node: a
whole extra full-frame graph pass that based_aa does not require.
`mclip_native32` now folds it too (`src/eedi3.cpp`), with the exact predicate
measured rather than assumed:

- the reference conversion's nonzero boundary is **`v > fl(0.5/255)`**. `fl(0.5/255)`
  is `0x3b008081` and its converted byte is 0; the next float up `0x3b008082`
  gives byte 1. So it is a **strict** compare — `>= 0.5f/255` is off by one ULP.
  Exactly round 11's `cmpgt` lesson, in a different guise.
- `tmp/float_thresh_probe.py` finds the boundary against the real conversion;
  `tmp/floatmask_regress.py` compares old vs new binaries over 63 cases
  (binary, a ramp/checker of consecutive floats straddling the boundary, random,
  sparse, out-of-range) using **modal outputs over 5 runs** (a single run can
  land on a race outcome).
- ONE DELIBERATE DIVERGENCE: above ~8.42e6 the reference conversion's
  `v*255+0.5` overflows int32 and zimg's saturating SIMD convert emits byte 0.
  The native path treats such values as nonzero, matching `eedi3vk2` (which
  reads the float mask directly) instead of an unrelated library's overflow
  artifact. Masks are documented as `[0,1]`; based_aa's are exactly 0.0/1.0.
- Measured: **+2% at fp32** (302.8 -> 308.3, 301.4 -> 307.4, same-session
  pairs). Small because the fp32 frame is floor-bound (15.G), and because the
  host now reads a 4-byte mask instead of the 1-byte converted one, which
  offsets part of the removed pass. The win is structural as much as numeric:
  both depths now have the same mask path.

### 16.2 THE STRUCTURED-MCLIP NONDETERMINISM: a `break` that skipped `dst`

The bug tracked as "open" since round 14 is fixed, and it was not a race: the
backtrack's tile loop did

```glsl
for (x0 = rightmost_tile; x0 >= 0; x0 -= BT_TILE) {
    if (x0 + n <= xmin) break;   // tile entirely left of the first DP column
    ...
    if (lane < n) { if (x < xmin) { cubic; store_dst_val(...); } ... }
}
```

`xmin` is the first column whose mask bit is set, i.e. the first column the DP
runs on. The `break` correctly skips the *walk* for tiles entirely left of it —
but it also skipped the **write**. Every column below `xmin` therefore kept
whatever `dst` already held: uninitialized memory on the first frame, a
recycled resource's previous contents afterwards. That is why

- the trigger was a *structured* mask: it needs `xmin` far enough right that a
  whole 32-column tile is skipped;
- an all-*zero* mask is deterministic (mask off everywhere -> `xmin = WIDTH` ->
  the parallel-cubic branch writes the row), an all-*white* mask is
  deterministic (`xmin = 0` -> no tile is skipped), and the noise-clip test
  mask is deterministic (`_right_half_mask` is white on the LEFT, so
  `xmin = 0` — the existing tests structurally could not see it);
- it reproduced at `num_streams=1` and with `NOREBAR` (neither is involved),
  and `eedi3vk2` was deterministic on the same chain;
- it was timing-sensitive in apparent frequency: how much of the skipped
  prefix was left stale only matters where the stale bytes happened to differ.

FIX: when a tile is entirely left of `xmin`, write the vertical cubic for it
with the whole workgroup (no walk, no shared memory, so no barrier needed) and
`continue` instead of `break`. Performance-neutral: interleaved A/B 510.6/511.5
(fixed) vs 515.2/518.5 (unfixed) — within noise, no regression.

**One fix, three measurements that were previously unexplainable:**

| comparison on 4 real 3840x2160 frames | differing px vs `eedi3vk2` |
|---|---|
| round-14 committed build | 22758 |
| round-15 build | 5902 |
| round-16 (this fix) | **3** (max 1 LSB) |

and real-chain determinism went from 3-5 distinct outputs in 8 runs to
**8/8 identical**. The remaining 3 px at <=1 LSB are the documented
f32-DP-vs-f64-DP ordering family.

Test: `test_eedi3_mclip_long_mask_off_prefix` (black LEFT half, white right, so
`xmin` is mid-frame) asserts 3 independent instances agree AND that the
mask-off prefix equals the all-zero-mask run's prefix. Mutation-verified: it
fails when the `break` is restored and passes with the fix.

LESSON: the round-14 note called this "a GPU-side race or uninitialized read,
data-dependent on the mask pattern" and listed hypotheses about the analytic
seed, `fcarry` and `bmaskSh`. The actual mechanism was a loop `break` whose
correctness argument ("nothing left of xmin reads directions") was true about
*reads* and silently false about *writes*. When a skip is justified by what a
region reads, check what it writes.

### 16.3 Round-16 standings (benchmark defaults, medians of 3 x 2000f, ns=8)

| depth | vsfeel | vszipcl | speedup |
|---|---|---|---|
| u16 | 518 (512/518/520) | 200 (200/190/200) | 2.59x |
| fp32 | 306 (307/305/306) | 202 (203/202/202) | 1.52x |

Tests: 81 pass (79 + the 2 new float-mask cases and 2 new prefix cases).

---

## Round 17: EEDI3H's four `std.Transpose` passes are the whole gap (documented, NOT implemented)

Investigated on request; the user chose to record it rather than change code, so
**nothing below is implemented**. It is the clearest remaining EEDI3-family win.

### 17.1 Measurement (real based_aa AA chain, 400 frames, ns=8, same session)

| chain | fps | ms/frame |
|---|---|---|
| EEDI3 (vertical, no transposes) | 488.6 | 2.05 |
| **EEDI3H (4 transposes + EEDI3)** | **226.6** | **4.41** |
| transposes only, no filter (`T(T(clip))`) | 424.0 | 2.36 |
| nothing (chain floor) | 1881 | 0.53 |

Driver: `tmp/hbench.py` (builds the bench.py AA vpy with a substituted chain).

- `std.Transpose` costs **~0.91 ms** per pass at 2x2160p u16: `tx2 - none` =
  1.83 ms for two passes = 16.6 MB read + 16.6 MB write each, i.e. **~36 GB/s**,
  which is the CPU memory path (AGENTS.md measures ~40-54 GB/s aggregate here).
- EEDI3H is **2.16x slower than EEDI3**, and the four transposes account for
  essentially all of it (EEDI3H - EEDI3 = 2.37 ms ≈ 4 x 0.59 ms with overlap).
- Ceiling if the transposes became free: EEDI3H -> ~2.1 ms -> **~420-470 fps
  (~1.9x)**, i.e. matching EEDI3's own rate.

### 17.2 Two measurement traps found while doing this

- **Unused `std.Transpose` nodes are pruned.** `tx4` (transposing clip, mclip,
  sclip, output) measured identical to `tx2`, because only the clip transposes
  feed the output; the mclip/sclip transposes were dead and VS evaluated
  neither. To measure N transposes in context they must all be on the output
  path (EEDI3H is the easy way: the inner EEDI3 consumes all of them).
- **Do NOT compare EEDI3H against EEDI3 with `sclip`/`mclip` dropped** as a
  proxy for "fewer transposes": `vert_noaux` measured 258 fps versus 489 for
  `vert`, i.e. *removing* sclip made it ~2x SLOWER. Dropping sclip sets
  `HAS_SCLIP == 0`, which makes the row kernel compute and store `cint` for
  every pixel instead of the vcheck reading sclip. The algorithm changes, so
  the comparison is meaningless.

### 17.3 Options (ranked; effort/risk noted)

**A. Fold the transpose into passes that already move the data (best, most
invasive).** No extra passes at all:
  - input clip: `ENTRY_PAD` already writes every pad element through a mirror
    mapping, so a horizontal mode is swapped mirror roles plus a **shared-memory
    tile** (a naive swap makes the raw reads stride-`WIDTH`, one transaction per
    lane). Margins: the built pad has MARGIN_H=12 columns and MARGIN_V=4 rows;
    the transposed space needs 12 *rows* (direction taps) and 4 columns, so the
    pad must be built with a uniform margin (12 both ways) or the transposed
    read under-runs at the edges. The transposed pad cannot alias the built pad
    (dims are asymmetric, H != W), so it is genuinely a second buffer.
  - output: one GPU pass folded into the download/blit instead of a CPU
    `std.Transpose` (the default path is row-kernel dst -> D2H -> host blit, so
    the transpose belongs between dst and D2H, or in the blit's addressing).
  - mclip: needs no 16.6 MB transpose at all — the host already reduces it to
    **packed bits** (1 MB at 4K), so the bit matrix is what gets transposed
    (~1/8 the traffic). Doing it well wants a 32x32 bit-tile transpose rather
    than scattered per-bit writes.
  - sclip: read with swapped addressing instead of materializing a transposed
    copy (the vcheck is the only consumer).

**B. Keep the composition, move the transposes to the GPU.** VRAM is ~300 GB/s
vs ~36 on the CPU host path, so ~0.11 ms per pass instead of 0.91. Mechanical,
but still pays for 4 passes (~0.4 ms) and needs the plugin to own the
transposes instead of composing `std.Transpose` nodes.

**C. Clip in/out only.** Fold just the two unconditional clip transposes,
leaving the mclip/sclip graph nodes. Smallest change, roughly +30%.

### 17.4 Notes for whoever implements it

- `transpose_plane()` at `src/eedi3.cpp:63` is **dead code** (defined, never
  called): a 64x64-blocked out-of-place transpose with NT stores. It looks like
  a leftover from an earlier host-side EEDI3H design. It is the natural helper
  if any host-side (or staged) transpose is wanted, but at ~36 GB/s it is not
  the win — the GPU is. *(Round 18: it was also **wrong** — the destination
  ranges overlap — and has been rewritten as a verified 16x16 SSE2 byte
  transpose; it is now live in the EEDI3H mask path.)*
- The safety net is already built: `tests/test_eedi3h.py` (25 tests) asserts
  `EEDI3H(x) == Transpose(EEDI3(Transpose(x)))` **bit-exactly** for u16 and f32
  across field 0-3, dh, mclip, sclip and YUV planes, so a rewrite is verifiable
  rather than hopeful.
- Re-measure with `tmp/hbench.py 400` (it prints the floor and the transpose-only
  arms alongside, so the comparison stays honest).

---

## Round 18: EEDI3H implemented natively (transposed staging + 2 GPU passes)

Round 17's option A, implemented. EEDI3H is no longer a composition; it is the
same EEDI3 pipeline with the plane transposed, so **the row/vcheck/vcopy/pad
kernels are untouched** (no algorithm duplication) and the 25-test bit-exact
transpose oracle still passes unchanged.

### 18.1 How it works

- `Eedi3HCreate` is now a thin wrapper calling the shared create path with
  `d->horiz = true`; there are no std.Transpose nodes and no new filter.
- Geometry: the kernel plane is the transpose of the source plane
  (`width = source height`, `height = source width`), so `rows`, `pad_stride`,
  `pad_height` and every region size are computed exactly as before from those
  swapped dims. `dh` doubles the transposed height, i.e. the output *width*, so
  `out_w`/`out_h` are kept separately from the kernel dims.
- Host gather: the source ROWS are read and the needed COLUMN parity is written
  compactly (`K[y][k] = src[y][first + step*k]`, `step = 2` for the kept/interp
  parity, `step = 1` under `dh` because every input column survives). SIMD
  deinterleave (`deint_even_u16/f32`, 32 source columns -> 16 kept), NT stores
  when the destination is the uncached ReBAR VRAM.
- `ENTRY_XPOSE` (new, per bit depth): 16x16 tiled transpose `K -> R'` with an
  LDS stage, so both the load and the store are coalesced. Run twice per plane
  (clip -> R', sclip -> B').
- `ENTRY_COMPOSE` (new, per bit depth): `O[y][2k+p] = (p == field) ?
  vout[k][y] : R'[k][y]` -- transposes both sources back into frame order and
  writes the *whole* output plane (kept and interpolated columns) into the
  staging download; the CPU blit is then a plain row copy.
- The pad builder now reads its source at **binding 9** (the transposed raw)
  instead of binding 0. Vertically binding 9 is created pointing at the same
  buffer as binding 0, so the pad kernel needs no second variant.
- The mask is deinterleaved to a 0xFF/0x00 byte matrix, transposed with a real
  16x16 byte transpose, and fed to the existing `build_bmask_row`; the vcheck
  reads the transposed sclip (B') through binding 5.

### 18.2 Measurements (real based_aa AA chain, 1000 f, ns=8, interleaved)

| chain | fps | ms/frame |
|---|---|---|
| EEDI3 (vertical) | 501.9 | 1.99 |
| old EEDI3H (`T(EEDI3(T(x)))`) | 233.2 | 4.29 |
| **new EEDI3H** | **360.9** | **2.77** |

Config sweep, new vs old: mdis 3 -> 394.5/234.4, mdis 20 -> 364.1/232.7,
mdis 40 -> 285.2/230.0, vcheck 0 -> 443.0/304.7. Stream knee is still **8**
(295.9 @4, 345.2 @6, 358.1 @8, 356.5 @10, 345.5 @12); the queue cap knob
(`VSFEEL_EEDI3_QUEUES`) still does not help (q=4 ties the default, q<=2 loses
~50%). Full suite: 449 passed.

### 18.3 Findings that cost time (all measured, all kept in the code)

- **Ordinary stores into the uncached ReBAR VRAM are catastrophic — and the
  packed dilation bits were being written there** (both paths). At 4K that is
  only ~0.27-0.52 MB, but one PCIe transaction per word made the mask stage the
  single largest host stage (probe: `bits` 1.66 ms vertical, 7.9 ms horizontal).
  Writing the bits into the *cached staging mirror* instead (binding 4 points at
  staging under ReBAR) took the vertical path 474.9 -> 492.2 fps as a side
  effect and cut the horizontal mask stage's dilate from ~2.9 ms to ~0.34 ms.
- **`transpose_plane()` was dead code AND wrong.** Its 64x64 "blocked" loop
  wrote `xn` contiguous destination elements at `dst[x0*dstride + y0+y]`, i.e.
  overlapping ranges; for `dstride != 1` the result was garbage (verified with a
  5x4 unit test). It is now a correct 16x16 SSE2 byte transpose (its 4th stage
  must pair `t[i]` with `t[i+8]`, not `t[i+1]`), with a scalar fallback for
  wider elements. Never trust an unexercised helper; the oracle would have
  caught it if the mask tests had used a non-uniform mask.
- **The mask gather must key off the MASK's depth, not the clip's**
  (`mclip_native16/32`, else Gray8). Reading a Gray8 mask as u16 happened to
  pass the mdis=5 half-plane test and failed at mdis=20.
- **vout must be device-local for EEDI3H** (new default, `VSFEEL_EEDI3_VOUTDEV=0`
  still forces staging): the compose pass reads vout back, and reading it over
  PCIe from the staging download cost 328 -> 360 fps (+10%).
- **The compose output needed its own binding (b10).** With one binding for both
  vout and the assembled plane, moving vout to VRAM dragged the output there too
  (unreadable by the CPU).
- **The direct-to-frame import path is much worse than the notes' old number.**
  Re-measured on the current binary: vertical 505.3 -> 205.3 fps with
  `VSFEEL_EEDI3_DSTHOST=1` (-59%, versus -36% when round 13 measured it). The
  horizontal mode disables it outright.
- Temporary ablations (`VSFEEL_EEDI3_ABL`) attributed the remaining penalty:
  compose ~0.53 ms, xpose ~0.10 ms (before the vout fix). They are removed.

### 18.4 What is left (not attempted)

- The compose pass still writes 16.6 MB of assembled plane to the staging
  download every frame (the whole point of the pass), and that GPU->system-RAM
  write is the largest single remaining item. Writing only the interpolated
  columns (8.3 MB) and merging on the CPU with the source row costs +8.3 MB of
  host traffic for -8.3 MB of PCIe, i.e. roughly a wash -- not done.
- The mask path runs two passes (deinterleave+threshold, then a byte transpose)
  where a fused deinterleave+transpose would save one 4.15 MB write + read. The
  bit-level variant (0.5 MB matrices) would save ~14 MB but needs a bit
  transpose and a bit-input dilation.
- The xpose passes are ~0.1 ms; merging them into one dispatch would only save
  launch overhead.

Files: the gather helpers are at `src/eedi3.cpp` (`gather_columns`,
`gather_mask_u8`, `transpose16x16_u8`), the pass plumbing is in
`record_command_buffer`/`vsfeel_eedi3_create`, the new kernels are
`ENTRY_XPOSE`/`ENTRY_COMPOSE` at the end of `src/eedi3.comp` and the CMake
`VK_EEDI3_UP_ENTRIES` list.

---

## Round 19: EEDI3H host-path tuning (+15%) and the pivot to a fused AA call

Goal was to close the EEDI3H-vs-EEDI3 gap from round 18. It did not close it —
the remaining gap is structural (below) — but three host-path changes landed,
**+15.2% on EEDI3H at ns=8 with the vertical arm flat as a control** (measured
same-session, order-reversed, 6 reps × 1000 f: horiz base 384.9 fps, `PAIR=0`
365.6, `MASKFUSE=0` 361.1, both off 334.2; vert 534.5 / 534.8 / 535.1 / 531.4).

### 19.1 The fused mask path (`gather_mask_bitmat`, +6.6%)

EEDI3H needs the mask transposed (row k = source COLUMN `field + 2k`). The old
path built a `rows × width` byte matrix with `gather_mask_u8`, transposed it with
`transpose_plane` (a 16x16 SSE2 byte transpose) into a second matrix, and only
then packed/dilated it — ~29 MB of traffic for the 0.5 MB of bits the row kernel
reads. The fused form thresholds 16 mask elements into 16 predicate bytes,
transposes the 16x16 byte tile **in registers**, and movemasks it straight into
the 64-bit word of the transposed bit row: ~17 MB, and `build_bmask_row`'s
shift+dilate tail is shared through `bmask_dilate_store` /
`build_bmask_row_from_bits`.

Two things cost time and are worth keeping in mind:
- **Loop order matters more than the byte count.** With `k` (16-column blocks)
  outermost, the gather keeps 64 row streams open and re-reads the whole mask
  frame per block — 2x slower than the shipped `y` outermost / 4x16-row-group
  form, which keeps only 16 streams live (`0x1DB4000`-scale pointer arithmetic
  is not the issue; the prefetcher is).
- **It is the READ, not the ALU.** At ns=1 the stage is ~1.6 ms for 16.6 MB
  (~10 GB/s), of which the byte transpose is ~0.3 ms. Cutting the transpose
  algorithm cannot pay; cutting the read can, and cannot be done (a transposed
  plane inherently needs every source row).

The old path is kept behind `VSFEEL_EEDI3_MASKFUSE=0` for A/B.

### 19.2 Fused kept+interp column gather (`gather_columns_pair`, +5.3%)

When the sclip frame **is** the clip frame (based_aa's
`Interleave([clip, clip])`, which vsapi hands out as the identical plane
pointer — verified with a temporary pointer trace), the kept columns (parity
`off`) and the interp columns (parity `field`) are the two halves of the same
64-byte row chunk, so one pass produces both and the second full-frame read
disappears. `VSFEEL_EEDI3_PAIR=0` restores the two-pass form. Requirements:
`!dh`, the two NT flags equal, dst rows 32-byte aligned, `rows` a multiple of the
vector width — the call site checks and falls back.

### 19.3 Aligned parity selects (`deint_row_*` / `nz_bytes_*`, neutral-to-positive)

Every gather used to offset the source pointer by `first` (0 or 1) and read
`step*k`, so with `first == 1` **every** 32-byte load straddled two cache lines.
Loading from element 0 and selecting the parity with the deinterleave
(`deint_odd_*` added) keeps the loads aligned, and for `step == 2` it makes the
last 16-column block exactly in bounds (the scalar tail is then only for
unaligned rows). Vertical is unchanged (control), horizontal is equal-to-better;
kept for the alignment and the simpler bounds.

### 19.4 Dead ends measured this round (mechanism, not verdict)

- **CPU merge of the kept columns** (`compose_tight`: compose writes interp-only
  tight, CPU interleaves with the kept source columns) — **−4.8%** over 8
  order-reversed reps (385.5 → 366.9). The extra 16.6 MB source re-read plus the
  merge ALU costs more than the 8.3 MB PCIe + 8.3 MB VRAM it saves. The
  GPU-assembled whole plane (`ENTRY_COMPOSE` as shipped) is the right structure.
- **k-outermost mask tiling** (19.1) — 2x slower, do not reorder back.
- **Ablations that change the data are not ablations.** `NOMASKX` (skip the mask
  gather, keep dilating stale bytes) showed +33% and led nowhere: the stale bits
  change which columns are masked, so the row kernel's DP work changes too.
  Round 17.2's warning applies to every `NORAW`/`NOPAD`/`NOXPOSE`-style probe.

### 19.5 Why EEDI3H cannot be closed further (the pivot)

After 19.1–19.3 the residual per-frame difference to EEDI3 is, in bytes:

- **mask read 16.6 MB vs 8.3** — the transposed plane needs all `height` rows to
  build mask columns; the vertical path needs only the `field`-parity rows.
- **compose + blit 49.8 MB vs 33.2** — the output plane must be assembled from
  transposed interp values *and* the kept columns; every rearrangement of that
  (CPU merge, spread stores, SDMA D2H, `vout` in dev_buf, host-pointer import)
  measures worse or neutral in situ.

Both are properties of "transpose the data, reuse the vertical kernels". The
workable way out is to stop transposing per filter call and fuse the **whole
based_aa chain** (vertical call + merge + horizontal call + merge) into one
plugin call, where the mask/sclip/source are read once for all four sub-passes.
That is specced in **`notes/EEDI3AA.md`** — measured chain today 98.9 fps vs
509.6/374.0 for its two constituent calls, projected 1.6–1.8x.

Measurement rule learned the hard way this round: with a **fixed variant order**
inside each rep, an *inert* knob on the vertical arm moved 5% — session drift is
larger than the effects. `tmp/abl.py` now reverses the variant order on odd reps;
grade only on those. Tools: `tmp/abl.py` (env A/B), `tmp/ab2.py` (two .so A/B),
`tmp/agg.py` (stage split at ns=8), `tmp/split.py` (ns=1), `tmp/fused_probe.py`
(the chain), `tmp/merge_semantics.py` (the chain's Merge formula).

Temporary diagnostics still in the tree (delete before landing):
`VSFEEL_EEDI3_NOXPOSE`, `VSFEEL_EEDI3_NOCOMPOSE`, `VSFEEL_EEDI3_NOMASKX`,
`VSFEEL_EEDI3_PTRTRACE`. Keep the `maskx=`/`bmask=` hbench split.

---

## Round 20: the pbt-packing / walk-chain levers are measured DEAD, and the probe
## harness was silently compiling to an empty kernel

Prompted by "the two biggest untried levers are 2-bit pbt packing (170 MB/sub-pass
written and re-read) and breaking the walk's serial load chain". Both are now
answered by measurement, and both answers are negative. **The round-15 cost model
does not describe the current kernel.**

### 20.0 The probe harness bug (FIXED): PROBE 4-7 compiled to nothing

`ENTRY_ROW`'s entry guard was `#if PROBE >= 3` instead of `== 3`, so every level
above 3 returned immediately and the whole kernel was dead code. Measured proof:
`glslc -DBITS=16 -DPROBE=5 -DENTRY_ROW` produced a **1656-byte** SPIR-V against
**43720 bytes** for PROBE=0. So the round-15/15.A ladder (`PROBE=4/5/6` = 728 /
726 / 718 vs 308 real) is a measurement of an *empty* kernel for 4, 5 and 6 —
which is why the odd ordering (each of DP-only, stores-only and no-coupling
measuring the same large number) never made sense. Fixed to `PROBE == 3`; the
shipping PROBE=0 binary is **byte-identical** before and after (md5
`3f53fffe2b81…`), and 214 EEDI3/EEDI3H/EEDI3AA tests pass.

New diagnostic levels added while re-deriving the model (all compile-time, all
documented in `src/eedi3.comp`): **7** = real walk + one extra discarded walk over
the *previous* row's pbt (f-dependent address); **8** = the same extra walk with an
f-INDEPENDENT address; **9** = keep the DP but write every column to the same 41
bytes (no store traffic); **10** = 8 with an explicit 4x unroll (4 independent
loads in flight); **12** = the faithful "no row kernel" ablation (see 20.3).

### 20.1 Current vertical cost model (the honest numbers)

EEDI3 vertical, benchmark defaults (2000 frames, ns=8, real based_aa mask,
2x2160p GRAY16, field=3, mdis=20, vcheck=2), 3 interleaved reps in a *stable*
window (within-arm spread <0.5%):

| build | fps | frame | attributable |
|---|---|---|---|
| real (PROBE=0) | 495.3 | 2.019 ms | |
| PROBE=2 no top-to-bottom tail | 594.4 | 1.682 ms | **whole row kernel 0.337 ms = 16.7%** |
| PROBE=5 DP+stores, no walk/interp | 562.4 | 1.778 ms | **walk+interp 0.241 ms = 11.9%** |
| PROBE=9 DP, ~zero store traffic | 514.0 | 1.946 ms | *slower than real* (see 20.2) |

Walk isolation, separate stable window (within-arm spread <0.5%):

| build | fps | read |
|---|---|---|
| real (PROBE=0) | 492.5 | |
| **PROBE=6** real walk reads the PREVIOUS row's pbt | **532.6** vs 530.6 real | **no gain: the same-workgroup store->load dependency is worth 0** |
| PROBE=7 real + extra f-DEPENDENT walk | 453.9 | one extra walk costs 8.5% of the frame |
| PROBE=8 real + extra f-INDEPENDENT walk | 451.4 | **identical to 7** |
| PROBE=10 real + extra unrolled MLP walk | 457.0 | **identical to 7/8** |

### 20.2 What that means for the two proposed levers

**Breaking the walk's serial load chain is worth exactly zero.** PROBE 7 vs 8 vs
10 differ by <1%: making the load address f-independent, and even issuing four
independent loads back-to-back with an explicit unroll, costs the same as the
serial f-dependent chain. The walk is **issue/sector-throughput bound, not
latency bound** — 3840 dependent L1 loads per row are already overlapped. There is
nothing to recover by speculating on the +-1 direction window, by lookahead, or by
any address-decoupling layout. (Independently corroborated: PROBE=6, which removes
the same-workgroup store->load hazard entirely, is a dead tie with the real
kernel — the round-15 "the coupling costs 2.4x" is gone after the 15.C/15.D
rewrites.)

**2-bit pbt packing has a ceiling of ~5%, and the store is not bandwidth bound.**
The whole DP+store stage is 4.8% of the vertical frame (probe 2 -> 5: 0.091 ms for
170 MB written = 1.9 TB/s, i.e. L2/port bound, not DRAM). PROBE=9 makes the point
directly: writing every column to the *same* 41 bytes — 1/40th the traffic — is
**slower** than the real scattered store (0.264 ms vs 0.096 ms of frame time),
because repeated writes to one dirty line serialise. Bytes are not the cost; the
store instruction/L1 port is. Packing replaces the two byte-stores per lane per
column with four `subgroupBallot`s plus one 16-byte store, i.e. *more* instructions
in a kernel that is already issue bound, to save traffic that is not the
bottleneck. Net expectation: a wash or a loss — not worth the layout rewrite.

### 20.3 EEDI3AA: 25-35% row kernel, and why the ablation overstates it

Fused EEDI3AA, same clip (5 interleaved reps, within-arm spread 1.3%): real 98.19
fps vs PROBE=2 150.39 fps = **1.53x, row kernel 35% of the output frame** — double
the vertical path's share, which is what makes the levers look attractive there.

But PROBE=2 returns with **`dmap` stale**, and the vcheck takes its cheap
`dirc == 0` branch when dmap reads 0, so PROBE=2 under-counts the vcheck and
overstates the row kernel. The faithful form is the new **PROBE=12**, which fills
`dmap` with the real pattern (0 on masked columns, nonzero on DP columns) before
returning: 89.76 fps vs 72.00 real in a noisy window = 1.25x, i.e. **~25%**. Any
future row-kernel/vcheck ablation must use PROBE=12, not PROBE=2/5, or it is
comparing against a cheaper vcheck.

### 20.4 The measurement environment degraded through the session

Same binary, same vpy: AA throughput measured 97.7 / 90.9 / 79.8 fps on three
runs, and vertical 530 / 495 / 451 / 410, while **all clocks stayed pinned**
(sclk 2304-2338, mclk 1249, fclk 2000 MHz), `gpu_busy_percent` = 100 throughout,
junction 86-90 C, vspipe's own CPU <0.3%, and MemAvailable flat (38-39 GB). So it
is neither clocks, thermal throttling, host CPU nor harness memory — the same
binary genuinely ran 20% slower for a whole 2000-frame run. Consequence: **effects
below ~5% are not measurable here without many interleaved pairs**; grade on the
median per-pair ratio from `tmp/probeab.py` / `tmp/probeabaa.py` (both order-
reversed, both 2000-frame), never on a single number.

### 20.5 What is actually left in the row kernel

The only real inefficiency left is **lane utilisation**: the walk is a serial chain
executed by 1 of the 32 lanes (the other 31 idle), and it is throughput/issue
bound, so its cost is ~8% of the vertical frame *because the wave spends issue
slots on one lane*. The fix is structural, not a layout change: split the walk
into its own dispatch with **one lane per row** (1080 independent chains instead of
1080 sequential ones), writing directions into the already-existing `dmap`, and
leave the coalesced tile-interpolate where it is. Projected from the 8% walk share:
~5-9% end-to-end, at the cost of two extra dispatches per sub-pass (the 15.B
DP/backtrack split was neutral *with the serial walk kept*, so the dispatch
boundary alone is known to be affordable). Not attempted — it is a real
restructure of `ENTRY_ROW` plus host dispatch/barrier plumbing, and it should be
prototyped behind the `PROBE` harness before anything lands.

---

# Remaining work, do-not-retry, and method rules (consolidated)

This is the short form: the first thing to read before touching EEDI3 again.
The *evidence* for everything here is in the rounds above — round 14 for the
current cost model, what it shipped and the open nondeterminism bug; earlier
rounds for individual verdicts. It is deliberately not a second copy of those
numbers: where an item's rationale is a measurement, the round is named.

**Two standing caveats on every verdict recorded before round 14.** The graded
benchmark's `mclip` was 100% zero until round 14, which made the row kernel's
DP/backtrack, vcheck and vcopy dead code; anything measured there describes that
degenerate path. And a recorded dead end decays (round 14 promoted two items
back into the tree), so re-check a verdict against the mechanism and the current
code before inheriting it.

## A. What is left

**A1. Parallel vcheck (the one lever with real upside).** Feed the d2p term
from the unvchecked `dst[r-1]` instead of the vchecked previous row. Rows then
become independent, so the serial 1080-barrier walk turns into a per-row
parallel pass and ENTRY_VCHECK's whole cost is the ceiling — that is the `NOVC`
arm, **+18-27%** on the honest path. The cost is a real accuracy change; the
one measurement ever taken of it (round 11: 1.2e-3 drift vs the f32 reference)
came from the degenerate config where it measured *neutral*, so it must be
re-measured. Prerequisite: a drift policy on the noise clip agreed with the
user, plus `tests/test_eedi3.py` tolerance updates.

**A2. Row-kernel optimisation — round 20 CLOSED the two remaining items as
measured dead ends; the current row kernel is only ~17% of the vertical frame
(~25% of the EEDI3AA frame).** The round-10 result "deleting the DP changes fps
by -1%" was measured with an all-zero mask, i.e. with the DP never executing.
Round 15 re-ran `-DEEDI3_PROBE=2` on the honest config and DONE both big items
(both KEPT):
  - **rolling window costs *with* mclip** — the single biggest win of the
    round, **+76%** end-to-end (15.D). Round 6's "no gain" was void.
  - **the walk reads pbt directly instead of staging the whole tile through
    LDS** — +9-13% (15.C).
  - **the two-dispatch DP/backtrack split is NEUTRAL** — the phase boundary
    costs what the same-WG store->load hazard saved (15.B). Do not retry.
  Round 20 re-derived the model on the current kernel (the round-15 probe ladder
  was invalid — see 20.0) and killed the rest:
  - **Break the walk's serial load-latency chain — MEASURED WORTH ZERO (20.1/20.2).**
    An extra walk whose load address is f-INDEPENDENT costs exactly what the
    f-dependent serial walk costs (453.9 vs 451.4 fps, identical within 0.5%),
    and even an explicitly 4x-unrolled MLP form is the same (457.0). PROBE=6
    (same-workgroup store->load hazard removed) is also a dead tie with the real
    kernel (532.6 vs 530.6). The walk is issue/sector-throughput bound, not
    latency bound: no lookahead, speculation or address-decoupling layout can
    recover anything. Do not retry.
  - **2-bit pbt packing — ceiling ~5%, and the store is not bandwidth bound
    (20.2).** DP+store is 4.8% of the vertical frame; the 170 MB/sub-pass store
    runs at ~1.9 TB/s (L2/port, not DRAM), and PROBE=9 shows that a *1/40th*
    traffic store pattern which collides on one dirty line is SLOWER than the
    real scattered one. Packing replaces 2 byte-stores/lane/column with 4
    `subgroupBallot`s + 1 vector store into an already issue-bound kernel. Do not
    implement it.
  - **Unrolling the walk (4x, merged tileF stores) — measured a wash** (8 reps:
    455.4 vs 445.6 fps, median pair ratio 0.99). Not landed.
  Remaining levers, in order:
  - **Split the walk into its own dispatch with one lane per ROW** (20.5): the
    walk's cost is that 31 of 32 lanes idle through a serial chain. 1080
    independent chains instead of 1080 sequential ones, directions written to the
    existing `dmap`, tile-interpolate left coalesced where it is. PROJECTED
    ~5-9% (vertical), NOT attempted — a real `ENTRY_ROW` + host dispatch
    restructure; prototype behind the PROBE harness first.
  - **SGSIZE 64 / K=1** (`TPITCH = 41 <= 64` at mdis 20): one full native
    wavefront per row, one direction per lane, the r0/r1/r2 rings collapse to
    21 VGPRs instead of ~42, likely better occupancy at this mdis. NOT yet
    tried (round 6's 2-row-WG regression was a different shape — see B).
  - BT_TILE as a spec constant swept {16,32,64}; hoisting the duplicated
    `cubic_float` evaluations; the fixed-per-invocation floats as spec
    constants; parallelising the serial span scan. (All bounded by the row
    kernel's ~17% share.)
  - Note the row kernel is now fast enough that the vcheck (~14%) and the host
    stages matter again; re-run the ladder before assuming anything.

**A3. Small / niche.** Fuse vcheck across planes (one launch,
`gl_WorkGroupID.y = plane`, as vszipcl does) — YUV-only, so it cannot move the
Gray flagship benchmark, but it is a real win for colour AA. Not attempted.
Nothing else in the host path is worth doing: `record_command_buffer` is
0.004-0.012 ms/frame (~0.2%), so prerecording per field parity,
`vkResetCommandBuffer` and binding the descriptor set once cannot pay for
themselves, and `vkQueueSubmit` is 0.06-0.07 ms.

**A2c. EEDI3H host-path tuning — DONE (round 19), and the remaining gap is
structural.** Fused mask path (+6.6%) + aliased-sclip pair gather (+5.3%) +
aligned parity selects: **+15.2% on EEDI3H at ns=8** (see round 19). What is
left is the 2x mask read amplification and the compose/blit output round trip,
both inherent to reusing the vertical kernels on transposed data. The next real
lever is fusing the whole `based_aa` chain into one call — specced in
`notes/EEDI3AA.md`.

**A2b. EEDI3H's four `std.Transpose` passes — DONE (round 18).** EEDI3H is
now the same EEDI3 pipeline run on the transposed plane: nothing in the EEDI3
kernels changed, the host gathers the kept/interp source *columns* instead of
rows, one GPU transpose (`ENTRY_XPOSE`) puts them in the pad builder's own
layout, and one GPU pass (`ENTRY_COMPOSE`) reassembles the output plane.
Measured on the real AA chain (1000 f, ns=8, same-session interleaved):
EEDI3 501.9, old EEDI3H (`T(EEDI3(T(x)))`) 233.2, **new EEDI3H 360.9 fps — 1.55x**,
i.e. the penalty fell from 2.29 ms to 0.76 ms (67% of it removed). Holds across
configs (mdis 3: 1.68x, mdis 20: 1.56x, mdis 40: 1.24x, vcheck 0: 1.45x).
Remaining gap is the compose pass (~0.4-0.5 ms, mostly its 16.6 MB GPU->staging
write) plus the extra host traffic; details, mechanisms and the dead ends are in
**round 18**.

**A4. Accuracy-for-speed (the round-5 authorisation) — the remaining items are
void, on mechanism:**

- **fp16 pad/cost storage**: no traffic to remove. The pad is native u16 already
  and the DP costs live in registers, so there is no 2-byte -> 1-byte win.
- **Adaptive `nrad`/`mdis` by local activity**: nothing to buy. `mdis` affects
  `pbt` size (VRAM only) and the row kernel's inner loop; `nrad` is purely
  row-kernel cost. This trades accuracy for ~0. (The old justification — "the
  row kernel is not on the critical path" — was degenerate-config and is now
  false, but the mechanism argument stands: both are real per-pixel algorithm
  parameters, so relaxing them changes the output everywhere.)
- **Dropping the sclip upload when the caller passes the same clip**: undetectable
  in practice. `based_aa` (and therefore the benchmark) passes
  `sclip = Interleave([clip, clip])`, which a node-pointer test cannot see
  through; and when the frames are genuinely aliased the only saving is the
  row-copy half of the gather.
- Anything that relaxes accuracy still requires measuring the drift on the noise
  clip and updating `tests/test_eedi3.py` in the same change.

## B. Do-not-retry (mechanism, not verdict)

Entries marked **(stale)** were measured on the degenerate pre-round-14 config;
they are kept as the reason a variant failed, not as a current number.

- **ALL of the round-15 `EEDI3_PROBE=4/5/6` numbers are void** — the entry guard
  was `#if PROBE >= 3`, so those levels compiled to an empty row kernel (1656 B
  of SPIR-V against 43720 B for PROBE=0). Fixed to `== 3` in round 20; the
  shipping PROBE=0 binary is byte-identical. Anyone re-running the round-15/15.A
  ladder today measures the same empty kernel for 4, 5 and 6 — which is exactly
  why "storing nothing", "walking another row" and "skipping the backtrack all
  land within 1.4% of each other" (728/726/718) in that table.
- **Breaking the walk's serial load chain**, by any of lookahead, +/-1 candidate
  windows, or a packed layout that makes the load address f-independent (round
  20, 20.1/20.2): the extra-walk A/B gives 453.9 (f-dependent) vs 451.4
  (f-independent) vs 457.0 (explicit 4x unroll / MLP) fps — all identical, and
  PROBE=6 (same-workgroup store->load dependency removed) is a dead tie with the
  real kernel (532.6 vs 530.6). The walk is **issue/sector-throughput bound, not
  latency bound**, so there is no chain latency to hide. Mechanism: 4 independent
  loads in flight buy nothing, which can only be true if the loads were already
  overlapped.
- **2-bit `pbt` packing**: the DP+store stage is 4.8% of the vertical frame and
  the 170 MB/sub-pass store runs at ~1.9 TB/s, i.e. L2/port bound, not DRAM.
  **PROBE=9 is the direct proof that bytes are not the cost**: storing every
  column to the *same* 41 bytes (1/40th the traffic, same instruction count) is
  *slower* than the real scattered store (514.0 vs 562.4 fps) because repeated
  writes to one dirty line serialise. Packing also replaces 2 byte-stores per
  lane per column with 4 `subgroupBallot`s + one 16-byte store, i.e. more
  instructions in an issue-bound kernel. Do not implement without a different
  mechanism argument.
- **4x-unrolling the walk with merged `tileF` stores**: bit-exact and 56/56 tests
  pass, but measured a wash over 8 order-reversed 2000-frame pairs (median pair
  ratio 0.99). Not landed; the code was removed again.
- **Ablating the row kernel with `PROBE=2` or `PROBE=5`**: they return with `dmap`
  stale, so the downstream vcheck sits on its `dirc == 0` fast branch and the row
  kernel's share is overstated by ~9 points on EEDI3AA (35% vs 25%). Use the
  round-20 **`PROBE=12`** (dmap filled with the real pattern) for any future
  no-row-kernel measurement.
- **Splitting the DP and backtrack into two dispatches** (round 15, 15.B):
  neutral (vc0 322 vs 326, vc2 295 vs 295) — the dispatch/phase boundary costs
  exactly what removing the same-workgroup `pbt` store->load dependency saved.
  NOTE (round 20): that "dependency is worth 2.4x (PROBE=6)" half of the entry is
  void (PROBE=6 = PROBE=0 on the current kernel); the split's own neutrality was
  measured on the pre-15.C/15.D kernel and would need a re-test *with the walk
  kept serial*. The round-20 proposal (20.5) is a split that parallelises the
  walk, which is a different thing.
- **Per-frame `VK_EXT_external_memory_host` imports** (round 13, kept default-off
  behind `VSFEEL_EEDI3_DSTHOST=1`). The direct-to-frame path is bit-exact and
  wins at low churn / low stream counts, but the import is GPU work on
  RADV/amdgpu (VM map/unmap + TLB sync per 16.6 MB BO), not host work: staged 608
  | `vout` in dev_buf no import 588 | import the GPU never touches 487 | import
  + blit 390. Leaking imports is worse, so a cache would not fix it. Re-test only
  if RADV's import path gets cheaper or VapourSynth exposes stable-address frames.
- **Multi-region `vkCmdCopyBuffer` for strided row copies** (round 13): ~13 us
  *per region* in situ (1080 regions of 7680 B = ~14 ms of frame latency) versus
  ~1 ms for the compute blit kernel that replaced it.
- **`vout` in dev_buf + SDMA D2H** (`VSFEEL_EEDI3_VOUTDEV=1`): re-measured on the
  honest path, still a loss (289 vs 313-323; 327 vs 375 with vcheck off). The D2H
  copy costs more than the GTT reads it removes.
- **Lazy `cint`/`sclip` load in vcheck**: void by inspection — `cint_s` is the
  `dirc == 0` result *and* the `dirc != 0` blend operand, so every `do_row` pixel
  reads it.
- **`LDS ping-pong tlineSh`** (round 6, "flat-to-worse, the 30 KB of LDS was the
  cost"): **(stale)** — with the vcheck on the critical path it is **+3% and is
  now the shipped default** (round 14). The strongest example of why this list
  carries dates and mechanisms.
- **Parallel vcheck via unvchecked `d2p`** (round 11, "neutral while costing
  accuracy"): **(stale)** — neutral because the vcheck did nothing. Promoted to
  **A1**.
- **A single contiguous blit** instead of per-row strided copies: 525 vs 555 (stale).
- **Plain `memcpy` instead of the NT load/store pair** anywhere in the frame
  path: regressed in situ (blit 6.1 -> 10.4 ms/frame; vc0 481 -> 401) (stale).
- **Skipping the pbt global stores** (`EEDI3_PROBE=1`): no gain (stale: measured
  in the degenerate config, where the DP never ran so the stores never
  happened; re-measure on the honest path — see A2).
- **2-subgroup x 2-row 64-lane workgroup** for `ENTRY_ROW`: regressed in every
  config (round 6) (stale). NOTE this is NOT the same as the untried
  **one-subgroup 64-lane / K=1** shape now listed in A2.
- **`subgroupcoherent` on `pbt`** (259 -> 229), **`restrict` on `pbt`**
  (214 -> 201), **`[[unroll]]` on the K-direction loops** (227 -> 196) (stale).
- **Dropping `HOST_CACHED` from staging**: collapsed to 43 fps — NT loads need
  WB/WC memory (stale).
- **Rolling window connection costs *with* mclip**: ~~no gain~~ — **REVERSED in
  round 15, +76% and now the shipped default** (15.D). The old "no gain" was
  measured with an all-zero mask, i.e. with zero unmasked columns to roll over.
  The clearest example in this file of a stale verdict costing real speed.
- **Queue cap** (`VSFEEL_EEDI3_QUEUES`): 3 loses ~20%, 4/6/8 tie; the default
  `min(num_streams, queue_count)` = 8 stays and the knob stays for tuning.
- **Serial per-frame `get_frame` millisecond probes**: misleading at >1 stream.
  Use `VSFEEL_EEDI3_HBENCH` with `VSFEEL_EEDI3_HFRAME` instead.
- **`core.max_cache_size` in the benchmark harness**: lowering it collapses
  *every* plugin (vsfeel 61 vs 220 fps; vszipcl 39 vs 200) because the timed
  region starts re-running the decode/mask chain. Shrink the harness's own frame
  cache instead.

## C. Method rules that replaced earlier advice

- **Grade sweeps at >=900 frames.** A 700-frame queue sweep reported +14% for
  `queues=4`; at 900 frames it was a tie (clock ramp).
- **An isolated microbenchmark proposes; only an in-situ same-session A/B
  decides** (the memcpy-vs-NT-store case, `AGENTS.md`).
- **Never use timestamp queries for kernel attribution here.** The env-gated
  ablation ladder answered the host-vs-GPU question faster and more robustly; an
  early query-pool attempt hung the queue.
- **Prove the stage split with the ablation ladder, not theory**: the
  `VSFEEL_EEDI3_*` opt-outs (`NOBLIT`, `NOSCLIP`, `NORAW`, `NOH2D`, `NOVC`,
  `NOPAD`, plus `NOREBAR`) at ns=8, and `VSFEEL_EEDI3_HBENCH` +
  `VSFEEL_EEDI3_HFRAME` for the host split. Additive costs of similar size mean
  a shared memory path; one dominant item means a loop.
- **A structured mclip was not reproducible — FIXED in round 16.2.** The cause
  was not a race: the backtrack's tile-loop `break` skipped the *write* for
  every column left of the first DP column, leaving `dst` stale (uninitialized
  on frame 0). Real-chain determinism is now 8/8 identical and agreement with
  `eedi3vk2` over 4 real 4K frames went 22758 -> **3** differing px (max 1 LSB).
  Regression: `test_eedi3_mclip_long_mask_off_prefix` (mutation-verified).
  The general rule it teaches: **a skip justified by what a region *reads* must
  also be checked for what it *writes*.**
- **The benchmark's EEDI3 mask follows the clip's depth, like based_aa.** A
  float clip gets a GrayS float mask; vsfeel folds it natively (round 16.1), so
  both depths now take the same no-conversion mask path. One deliberate,
  documented divergence: floats above ~8.42e6 are nonzero to the native path
  but convert to byte 0 through the old reference conversion (a zimg int32
  overflow artifact); `eedi3vk2` agrees with the native path.
- **Session-to-session drift is large on this box** (the same committed EEDI3
  binary measured 264 fps early in round 15 and 290 later). Never compare a
  number to a number recorded in a different command; always interleave A/B in
  one session (`tmp/ab.sh`) or use the modal output over many runs when
  comparing *outputs* (a single run can land on any race outcome).
- **Round 20: the drift is a bimodal STATE, not noise, and it is not clocks,
  thermals, host CPU or harness memory.** The same binary + same vpy measured
  EEDI3AA 97.7 / 90.9 / 79.8 fps on three consecutive isolated runs while
  sclk 2304-2338 MHz, mclk 1249, fclk 2000 were pinned, `gpu_busy_percent` = 100
  for the whole run, junction 86-90 C, vspipe's own CPU <0.3% and MemAvailable
  flat at 38-39 GB. Vertical swung 410-530 over the same period. So: **effects
  under ~5% are not measurable on this box**, and a "win" of that size from a
  single pair is noise. Grade on the median per-pair ratio over >=5
  order-reversed 2000-frame pairs (`tmp/probeab.py`, `tmp/probeabaa.py`), and
  treat a claim of <5% as unproven. Corollary: the vertical EEDI3 arm is *not*
  less noisy than EEDI3AA — it looked stable for an hour and then swung 20%.
- **Before trusting any `PROBE` number, sanity-check the built SPIR-V size.**
  Round 20 found levels 4-7 silently compiling to a 1656-byte empty kernel
  (against 43720 B for PROBE=0) because the entry guard was `>=` rather than
  `==`. `stat -c %s build/vk_spv/eedi3_16_row.spv` after each probe build would
  have caught it instantly — one command, and it invalidated a whole round of
  recorded numbers. Do this for every probe variant.
- **An ablation that removes a producer must also reproduce what the consumer
  reads.** `PROBE=2`/`PROBE=5` skip the row kernel and therefore leave `dmap`
  stale; the vcheck then takes its `dirc == 0` fast branch, so those arms
  under-count the vcheck and overstate the row kernel's share (35% vs 25% on
  EEDI3AA). `PROBE=12` fills `dmap` with the real pattern first. Same class of
  error as round 19's "ablations that change the data are not ablations".

---

## Round 21: the LDS vcheck verdict REVERSED, and the EEDI3AA compose fused into VRAM

Goal: close the EEDI3 vs EEDI3H gap (the user's based_aa EEDI3AA chain calls both,
so EEDI3H's horizontal-stage overhead shows up there directly). Zipcl's H/V
ratio is ~0.91 while vsfeel's was 0.70, which is what prompted the request.

### 21.1 The vcheck default flips to the global-read form (+11% EEDI3H, +21% EEDI3AA)

Round 14 made `ENTRY_VCHECK`'s shared-memory d2p ping-pong the default because it
removes one global read of the previous row per `dirc != 0` pixel. **On the
current kernel that verdict is inverted.** The LDS walk *must* visit every row
(each row has to advance the ping-pong) and ends with an extra full-width flush
loop per row; the global form skips fully-masked rows via `ENTRY_VCOPY`, and on
the AA workload ~2/3 of the rows are fully masked. Same-session order-reversed
medians (1500 f, ns=8, 4 reps):

| arm | LDS (old default) | global (new default) |
|---|---|---|
| EEDI3 (vertical) | 498.3 | **513.6** (+3.1%) |
| EEDI3H (horizontal) | 340.9 | **378.4** (+11.0%) |
| H/V ratio | 0.684 | **0.737** |

Fused EEDI3AA, same-session A/B, 1200 f x 4 order-reversed reps:
**89.7 -> 108.8 fps (+21.3%)** with both changes (see 21.2). The LDS pipeline
is still compiled and selectable with `VSFEEL_EEDI3_VCLDS=1`.

### 21.2 The EEDI3AA horizontal compose now merges in VRAM (bit-exact, +3%)

The fused filter assembled two full horizontal planes into staging (16.6 MB PCIe
write each) and then had the CPU 50/50-merge them (reads 33.2 MB, writes 16.6).
`NOCOMPOSE` ablation: the two compose dispatches cost **+14%** of the frame
(104.6 -> 119.4 fps), the single largest remaining horizontal item.

`ENTRY_COMPOSE` gained a `comp_fuse` push constant: sub-pass 0 writes its
assembled plane into a device-local `o0` region (binding 1, whose `dst_base` is
free because the compose reads `vout` when `vcheck > 0`), and sub-pass 1 reads
that plane back, assembles O_1, averages exactly like `std.Merge`
(`(a+b+1)>>1` for u16, `0.5a+0.5b` for f32) and writes the merged full plane
straight to the staging download. The CPU then does a plain row blit instead of
the merge. That removes one full-plane PCIe write and the 16.6 MB merge read.
`VSFEEL_EEDI3_AATIGHT=0` restores the old form; vcheck == 0 falls back
automatically (there `dst_base` still addresses the row kernel's interp rows).

Verified **bit-exact** against the old path on 30 real 4K frames (497 MB dumped
from each arm, identical) and by `tests/test_eedi3aa.py` (133 passed; the u16
cases assert exact equality against the oracle). Measured +3.0% (111.0 vs 107.8,
4 order-reversed pairs).

### 21.3 What the gap is made of now, and why it cannot be closed further

`VSFEEL_EEDI3_GBENCH` (new, see 21.4) at ns=1 gives the per-submission split
(horiz, mdis=20, vcheck=2): xpose 0.05 ms, pad 0.04, row ~2.7, vcheck ~2.7,
**compose 0.62**; vertical: pad 0.05, row ~2.9, vcheck ~2.5. At ns=8 the
remaining EEDI3H gap (0.69 ms, 513.6 vs 378.4) decomposes as compose 0.31 ms +
blit 0.28 ms + ~0.1 host.

The compose is a 16.6 MB transpose writing **directly to host memory**, measured
at 27 GB/s == the PCIe 4.0 x16 line rate; a VRAM write + SDMA D2H moves the same
bytes at the same rate (and the vertical `vout`-in-dev_buf experiment already
lost). It cannot be made cheaper except by writing fewer bytes, and every
"write the interp half only" variant trades the 8.3 MB PCIe write for an 8.3 MB
host read (compose_tight: read src again in the blit -> measured -4.8% in round
19) or an 8.3 MB host write (K mirror in staging) -- a wash. Unlike EEDI3AA's
fused filter, standalone EEDI3H has only ONE composed plane, so there is no
second kept half to substitute for the source read.

**Zipcl is not a counterexample.** Its absolute horizontal overhead is 0.48 ms
(full-plane GPU transpose in + transpose out + D2H); vsfeel's is 0.67 ms but its
vertical baseline is 2.5x faster, so the *ratio* looks worse. There is no
transposed-plane implementation on this box with a ~0 overhead: the transpose is
mathematically required (the DP walks the interpolation axis, the output needs
the perpendicular layout) and the row kernel's stores are only coalesced in the
transposed layout.

### 21.4 Tooling: `VSFEEL_EEDI3_GBENCH` (committed this round)

The GPU stage profiler round 4.5 claimed was durable but was never in the tree is
now real: `VSFEEL_EEDI3_GBENCH=1` creates one timestamp query pool per resource
and writes a `BOTTOM_OF_PIPE` mark at every `record_pass` stage boundary
(pass start / after xpose / after pad / after row / after vcheck-vcopy / after
tail), reading back after the fence and printing the mean stage deltas every 50
frames, tagged `single` / `aa-v` / `aa-h`. It is zero-cost when unset. **Get the
`timestampPeriod` arithmetic right** (10 ns/tick on Navi31): the tick delta must
be *multiplied* by the period, then divided by 1e6 to reach ms
(`ticks * period / 1e6`); the first version divided by the period instead and
printed nonsense. Multi-stream marks include other streams' interleaved work,
so attribute kernels only in the ns=1 trace.

### 21.5 Dead ends re-confirmed this round

- `VSFEEL_EEDI3_COPY=7` (cached load in the blit instead of the NT load):
  horizontal 387.4 -> 377.1, vertical flat. The NT load stays.
- Per-frame `VK_EXT_external_memory_host` import for the output frame is still
  catastrophic on the current driver: vertical 504.2 -> 215.7 fps (measured
  again this round). Do not re-derive.
- Queue caps: `QUEUES=8` (default) still wins (526.8/394.5 vs 519.9/387.2 at
  cap 4) after the vcheck change.
- `VSFEEL_EEDI3_NOPAD`/`NOXPOSE` remain unusable as cost ablations: they change
  the row kernel's input and therefore its branch mix (the horizontal "pad
  +7.3%" reading was mostly that, not the pad kernel -- gbench puts pad+xpose at
  under 0.1 ms).

## Round 22 — cross-cutting hardening

- Every flush/invalidate range goes through the shared `mapped_range` helper in
  `vsfeel.h` (offset rounded down, size rounded up to `minNonCoherentAtomSize`,
  `VK_WHOLE_SIZE` when the allocation size is unknown or the rounded end would
  overrun it). EEDI3's identical per-plane flush entries (`[0, upload_total)`
  pushed once per processed plane, twice in the file) collapsed to one range.
  The `!coherent` invalidate ranges over `v_offset` / `out_offset` / `dl_offset`
  keep their per-plane shape and are now atom-aligned.
- `VK_EXT_subgroup_size_control` is enabled only when the physical device
  advertises it (or the device is Vulkan 1.3+, where it is core); the device
  extension list is enumerated once for all three optional extensions;
  `apiVersion` is recorded and `create()` reports
  `"EEDI3 requires Vulkan 1.3 (device reports X.Y)"`; all
  `vkCreateComputePipelines` calls take `pipeline_cache_lock`; and
  `allocate_memory` refuses a `DEVICE_LOCAL|HOST_VISIBLE` request instead of
  silently relaxing it to host memory.
- No behaviour change here (`api_version=1.4`, extension present, ReBAR present,
  staging coherent), so no measurement; verified with the full test suite plus
  `VSFEEL_DBG=1` showing the recorded values.

## Round 23 — NT-store alignment predicate in the host gather

`deint_row_u16` / `deint_row_f32` gated their streaming store on
`(k & 15) == 0` / `(k & 7) == 0` — the *element index*, not the pointer. The
scalar head loop already aligns `d`, so after it `d + k` is 32-byte aligned by
construction and every body iteration advances exactly 32 bytes; the predicate
was therefore true only when the head length happened to be a multiple of the
vector width. On any destination row whose cell length is not a multiple of 32
bytes (EEDI3H's transposed kept-row cells are `rows * elem`: at a 630-px source
that is 1260 B for u16 and 2520 B for f32) the head is 10 / 6 elements and
*every* body store silently fell back to `_mm256_storeu_*` — a cached store into
the uncached host-visible VRAM window that the whole ReBAR upload design targets.
Predicate dropped: `nt` alone now selects `_mm256_stream_*`. The 1920-px
flagship was unaffected (rows are a multiple of 32 bytes, head always 0).

Measured on the AA geometry (jpbd -> 630px luma -> 2x Point, EEDI3H field=3,
mclip, 800 frames, ns=8, interleaved old/new binaries, 3 rounds of 2 reps;
harness: `tmp/bench_eedi3h_crop.py`):
old 799.9/784.1/798.9, new 803.1/824.9/803.4 fps. Same-order pairs are +0.4%
and +0.6%; the middle pair is +5% but order-confounded. Perf-neutral in
practice, and the ablation says why: `VSFEEL_EEDI3_NORAW=1` (skip the raw
gather) is worth only +2.6% at this width, and `VSFEEL_EEDI3_COPY=0` (force
cached stores on every gather) costs ~1% — full-width 32-byte AVX stores
coalesce in the write-combining buffer, unlike the narrow scalar stores behind
the old "27.6 fps" observation. Output bit-identical at 630/638/640 px.

## Creation error path

The per-stream `Eedi3Resource` is created into the pool via
`FramePool::emplace()`, so an error return inside the creation loop is torn down
by `~Eedi3Data` (buffers, device memory, mapped windows, command pool, query
pool, fence) instead of leaking it. Correctness-only; the EEDI3 suites pass.

## Round 24 — four latent-defect hardenings (correctness-only)

1. **Pipeline dedup array.** `VkPipeline destroyed[8 * 3]` feeding
   `destroyed[nd++]` had zero headroom: EEDI3AA's worst case (four `WidthKey`s x
   six non-null pipelines, 24) exactly fills it. Now a `std::vector` with
   `std::find` dedup, so a fifth key or a new pipeline kind cannot smash the
   stack.
2. **Host-pointer import.** `import_plane_host_memory` bound at
   `addr & (align-1)` without ever reading the imported buffer's own
   `VkMemoryRequirements`. It now queries them and rejects (falls back to the
   CPU blit) when the offset is not a multiple of `mem_req.alignment`, when the
   allocation region is below `mem_req.size`, or when `mem_req.memoryTypeBits`
   rejects the chosen host-visible type. Latent only on this driver
   (`minImportedHostPointerAlignment` already satisfies the buffer alignment).
3. **Unknown-length sentinel.** `field > 1` multiplied `numFrames` by 2
   unconditionally, so a `-1` unknown-length source became `-2` at
   `createVideoFilter`. Guarded with `if (numFrames > 0)` in EEDI3's sclip
   validation, EEDI3's output vi, and NNEDI3's output vi. No installed source
   plugin here reports `-1` (BestSource/FFMS2 always resolve a count, including
   on a truncated FFV1 mkv), so the sentinel arithmetic is verified by a
   standalone compile of the exact guard (`tmp/wo20_unknown_len.cpp`): `-1`
   stays `-1`, `0` stays `0`, known lengths still double, the
   `> INT32_MAX/2` overflow guard is unchanged.
4. **Diagnostic-flag consistency.** `VSFEEL_EEDI3_RAWSTAGE=1` wrote the raw
   gather into staging while the kernels still read the raw upload from
   `up_dev`, so alone it was garbage (it needed `NOREBAR=1`); it now selects the
   DMA path itself. `VSFEEL_EEDI3_NOBLIT` was a silent no-op on EEDI3AA (the
   final merge is not skippable), so the combination now fails loudly at
   creation instead of pretending.

Verified: `test_eedi3.py + test_eedi3h.py + test_eedi3aa.py` 214 passed,
`test_nnedi3.py` 47 passed; known-length `field>1` still doubles 24 -> 48 for
EEDI3/EEDI3H/NNEDI3; `DSTHOST=1` still imports bit-identically to the default
path. No performance measured; every change is on a cold or latent path.
