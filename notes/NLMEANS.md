# NLMeans optimization notes

Goal: `vsfeel.NLMeans` **at least 20% faster than vszipcl** on the synthetic
benchmark (757 fps ⇒ ≥ ~908 fps), all tests passing; numerically faithful to
the reference (KNL semantics). Target GPU: RX 7900 XTX (RDNA3, gfx1100),
Mesa RADV.

## TL;DR (2026-08-22, checkpoint)

Slot-direct cache + compact upload + GPU pad kernel + u16 native io +
`first`-flag init all DONE. 52/52 tests pass, matches vszipcl to ~3e-8
(float) / ≤1 LSB (16-bit).

| plugin | synthetic YUV420P16 UV, d=2 ns=2 |
|--------|----------------------------------|
| vszipcl | **757** fps |
| vsfeel | **714** fps (250 → 284 u16 → 471 cache → 606 compact+pad → 714 cache-sharing) |
| vszipcu | 587 fps |
| nlm_hip | 347 fps |

Steady-state per frame: pad ~0.005 ms + weight batches ~0.95 ms + acc ~0.24
ms + finish 0.085 ms ≈ **1.28 ms GPU** (queue nearly saturated now); host
up ~0.25 ms, dl ~0.38 ms. To move further the KERNELS must shrink — host is
no longer the bottleneck.

## Cache-sharing fix (2026-08-22, 606 → 714 fps)

Three related changes to acquire/release:

1. **Pool sized for full concurrent windows**: `n_slots =
   num_streams * channels * clips * layers + 2*channels*clips` (capped at
   512 MiB worth of slots). The old formula `(ns+2d)*C + 2C` gave 16 slots
   for the bench config while two streams hold 10 tiles each until their
   fences complete → constant cv-blocking inside acquire_cache.
2. **Held-cached tiles are share-read safe**: tiles are immutable once
   padded, so another stream's weight kernel reading a slot concurrently is
   fine. Only "currently being padded" needs exclusion → `CacheSlot.writing`
   flag, set at upload reservation, cleared in release_cache by the writer
   after its fence. Acquire treats `writing` slots as misses (uploads a
   private duplicate rather than block).
3. **Never block on held tiles**: blocking serialized the streams completely
   (the second stream couldn't even start composing until the first fence).

Pitfall found on the way: making held tiles *duplicates* without the
writing-flag sharing caused an eviction THRASH — overlapping windows mean a
wanted tile is almost always held by the other stream, so every frame
re-uploaded 6-10 tiles instead of 2 and evicted useful ones (comp measured
~110 ms/100 frames of pure memcpy). With share-reading, steady state is
2-4 tiles/frame.

Instrumentation: `[perf]` gained `blocks=` count; `NLMEANS_TRACE=2` prints
per-phase acq/comp/tab splits every 100 frames.

## The race that cost an evening (do not regress)

The acquire protocol double-booked slots within ONE acquire: reused-cached
slots were not recorded in the per-attempt `taken` set, so a later tile of
the same frame could EVICT a slot an earlier tile had just claimed for reuse.
Timing-dependent → nondeterministic output under threaded load only (serial
sliding-window order masked it by luck). Fix: EVERY chosen slot (reuse or
eviction) goes into `taken`. Symptom signature: diffuse ~0.02–0.05 diffs on
random frames under parallel load only; serial always exact.

## Commands

```bash
# build + install (MUST copy after every rebuild or you bench a stale plugin)
cmake --build build
cp build/libvsfeel.so /usr/lib/python3.14/site-packages/vapoursynth/plugins/vsfeel/

# tests (52)
MANGOHUD=0 python -m pytest tests/test_nlmeans.py -q

# benchmark (primary metric: --synthetic YUV420P16 1080p channels='UV')
MANGOHUD=0 python3 benchmark/bench.py --synthetic --filter nlmeans --frames 300

# stage timing (host phases + GPU timestamp splits)
NLMEANS_TRACE=1 NLMEANS_GPUTRACE=1 MANGOHUD=0 vspipe bench.vpy /dev/null
```

Bench defaults: d=2, a=2, s=4, h=0.2, wmode=0, wref=1.0, channels='UV',
num_streams=2. rocprofv3 profiles the OpenCL/HIP references fine (`rocprofv3
-S --kernel-trace -- vspipe x.vpy /dev/null`); for RADV use the env vars above
plus `RADV_DEBUG=asm`.

## Facts established by profiling

### vszipcl per-frame GPU (rocprofv3, same config)

- `nlmWeight`: 8 launches/frame, ~1.02 ms total (~128 µs/launch avg incl.
  warmup outliers; min 42 µs on small tail batches).
- `nlmAccumulation`: 8 launches/frame, ~0.34 ms.
- `__amd_rocclr_copyBuffer`: ~10 calls/frame ≈ 0.57 ms. These are the
  frame-cache slot→u1 D2D copies (2 ch × 5 layers), the ~1 new-layer H2D
  upload, and the result readback. Their `clframecache` means sequential
  playback re-uploads ~1 layer/frame instead of our 5×C=10.
- `nlmFinish`: ~15 µs. Fills negligible.
- Wall ≈ GPU sum ≈ 1.48 ms → their 634–704 fps is pure GPU throughput.

### vsfeel phase split (NLMEANS_GPUTRACE)

- `copy=` (staging→VRAM window copies + scratch fills): 0.44–0.9 ms.
  We DMA ~20 MB/frame: padded window 10.26 MB + fills (u2 6.2 MB, u5 2 MB,
  u1z 2 MB). ~29 GB/s effective — DMA-bound, not fixable without caching.
- first weight batch (gz=8 passes): 0.207 ms; acc batch: 0.063 ms; all-batch
  wall ≈ 2.1–2.2 ms; finish 0.15 ms (finish writes GTT directly incl. 16-bit
  atomicOr — measured cheap enough to ignore for now).
- Host `up=` 2.34 ms: composing the padded window into mapped staging by
  row-memcpy (5400 rows × ~3.8 KB). Partially overlaps GPU via the 2-stream
  semaphore pipeline but still eats wall time.

### Codegen (AGENTS.md method)

- Our ACO ISA is *smaller* than the offline-compiled reference
  (`clang -x cl -target amdgcn-amd-amdhsa -mcpu=gfx1100 -O3 …` then
  `llvm-objdump -d`): weight 188 vs 263 insts, acc 119 vs 146, finish 65 vs
  86. Spec constants DO fold (loop bounds become immediates). Codegen size is
  NOT the problem.
- `RADV_DEBUG=shaderstats` (this Mesa build DOES print for us, unlike
  DFTTest-era observation): weight kernel SGPRs=108, VGPRs=24, LDS=6144 →
  occupancy is fine; no spills.

## What helped

- **Exact per-instance LDS sizing** via specialization-constant array dims
  (`shared float dist[VRT*BY + 2*NLM_S][DIST_W]`) instead of fixed
  S_MAX=8 worst case: weight batch 0.237 → 0.207 ms (+13%). Kept.
- **Native u16 io** (`GL_EXT_shader_16bit_storage`, uint16_t/float buffer
  views instead of dword-extraction): matches the reference's
  `global_load_u16` codegen; weight batch −10 µs; lets the finish kernel
  store u16 directly (removes the u1z FILL dma AND the GTT atomicOr path;
  finish 0.131 → 0.086 ms). Overall +14% fps (250 → 284). Kept.
  - pitfall: glslc `-O` dies on direct float→uint16_t convert ("Expected
    input to be int scalar or vector: UConvert"); route via
    `uint16_t(uint(x) & 0xFFFFu)` (same workaround as DFTTest).

## What did NOT help (do not retry)

- **Compile-time `-DNLM_S=4`** (forcing full unroll visibility): within noise
  of the spec-constant version. Do NOT build the 9-value s-variant matrix.
- **wave32 required-subgroup-size pNext**: no effect (pipelines were already
  scheduled fine; unlike DFTTest where it won 552→673).
- **Manual straight-line unroll of the distance loops** (mimicking ROCm's
  full unroll / MLP): no effect — the dist loads were not serialization-bound.
- **Even-stride LDS matching zipcl's exact 5120 B** (vs our odd-stride 6144 B):
  no effect — occupancy is not the limiter either way.
- **Grouped all-weights-first structure**: BROKE correctness — the u4a slot
  ring (2·qb slots) is reused by every batch, so batch k+1's weights overwrite
  slots while batch k's accumulation still reads them. The interleaved
  W→A barrier sequence is load-bearing; reverted (caught by the suite).

## Remaining paths (priority order)

1. **Slot-direct reads + frame cache** (the big one): replace the per-stream
   contiguous u1 window with a shared pool of padded per-(clip,frame,channel)
   slot tiles; kernels resolve layer→slot through a small int table (binding 8)
   so NO staging→u1 copies and NO D2D refreshes are needed — steady state pays
   ~1 new frame H2D (~2 MB) instead of ~20 MB DMA. Protocol: BM3D-style
   holders + cond-var blocking (proven in-repo). Saves VRAM too (no
   per-stream u1/u1r).
2. **`first` flag in the accumulation kernel** (CUDA reference trick): batch 0
   starts accumulators from constants instead of loading u2/u5 → drop the
   8.2 MB/frame scratch FILLS.
3. Re-measure the residual kernel gap afterwards; candidates left:
   cooperative-matrix (WMMA) box-sum, launch structure. Note the uniform
   ~1.5–1.9× per-launch deficit vs zipcl persists across all ISA-level fixes;
   suspect scheduling/runtime rather than codegen.

## Key implementation facts

- Sweep tables are ported verbatim from vszipcl's `create()`: half-space of
  displacements (`kk*spt_area + j*spt_side + i < 0`), mirror entries double
  the weight rows when kk≠0, stride-8 rows, qb=8 batches (npix ≤ 1920·1152),
  variants indexed by m=min(d,n).
- Padded-window margins are zeroed once at staging init and never touched
  again; flat-index reads wrap OOB columns into neighbouring rows' zero
  margins (that's why candidate coords need no explicit bounds checks).
- u2 is pixel-interleaved (C+1) floats; u5 seeded via `vkCmdFillBuffer` with
  FLT_EPS bits 0x34000000 so the finish denominator stays > 0.
- 16-bit stores in finish use masked `atomicOr` into pre-filled u1z.
- Barriers required between EVERY weight→acc and acc→next-acc pair (u4a RAW
  through the slot ring, u2/u5 RMW); transfer→compute barrier after
  copies/fills. Same serialization the OpenCL in-order queue gives zipcl.
- `copy_stream_read` requires 32-byte-aligned sources — unusable for row-wise
  downloads at odd widths (segfaulted at width 613); plain memcpy now.
- int32 addressing validated at create(): reject when
  max(lay·layers·C, npix·slots, npix·C) ≥ 2³¹ instead of falling back to
  64-bit indices like the reference.

## Reference implementations (same GPU, READ-ONLY)

- vszipcl = OpenCL/ROCm, `reference/vapoursynth-zipcl/src/nlmeans.zig`;
  kernels embedded as `kernel_src`; frame cache `clframecache.zig`. Their CL
  source does NOT use the odd-LDS-stride trick (that's only in the CUDA
  variant) — we keep ours, harmless.
- vszipcu = HIP/ROCm, `reference/vapoursynth-zipcu/src/nlmeans.cu` (odd
  strides `|1`, pinned multiplies documented there).
- nlm_hip = standalone KNL-style plugin, also benchmarked.

## Debug env vars

- `NLMEANS_TRACE=1` — [perf] host phase averages every 100 frames
  (up/sub/wait/dl ms).
- `NLMEANS_GPUTRACE=1` — per-frame GPU timestamp splits: copy, first weight
  batch (w1), first acc batch (a1), all-batch wall, finish.
- `RADV_DEBUG=asm` — ACO ISA dumps to stderr (works);
  `RADV_DEBUG=shaderstats` — VGPR/LDS stats (works on this build).

## Per-batch GPU trace (NLMEANS_TS_MAX=130, timestamps after every W/A dispatch)
Steady-state (frame 66+) at TRUE bench config (1920x1080 YUV420P16, chroma 960x540,
d=2 ns=2): copy(pad)=0.17 | w1(first W batch)=0.28 | wSteady=~0.10x6 | a1=0.03 |
aSteady=~0.04x6 | finish=0.086 ms. Wall(all W+A)=1.16ms. Total GPU ~1.4ms/frame ==
wall at 685fps -> queue saturated, kernels are the bottleneck again.
**Weight kernel = 0.88ms = 72% of GPU time.** Cold-first-batch numbers (107us) were
misleading: warm batches run 49us at 540-height / ~100us at full bench config.
Baseline after cache fix: **vsfeel 685 vs vszipcl 758 fps** (need >=908).
7 weight batches/frame (q_cnt~50-56 entries, qb=8) -> launch overhead negligible;
kernel body must shrink. Plan: run-merging (LDS-resident ref+cand-union tiles,
one WG sweeps consecutive-i displacements; kills per-cell global loads/address-math/
OOB branches).

## Warm-probe round 2 (steady-state, half-height test clip chroma 960x270)
Warm weight batch 49us decomposes: dist phase (guide loads + pix-dist math) ~32us
(65%); hsum/vsum/exp/stores/launch floor ~17us. Box sums are FREE in steady state
(NOSUM == NODIST within noise) - the earlier "19us box sums" was a cold-batch
artifact. Run-merging (LDS tiles shared across consecutive-i displacements)
attacks exactly the 32us.

## Run-merging implemented (commit-in-progress) -> NEUTRAL (685 fps unchanged)
One WG now sweeps a run of consecutive-i displacements sharing (qy,qz); ref and
candidate-union tiles staged once in LDS (NLM_RUNMAX spec const id 14 caps run
length at qb so the u4a ring is unchanged). wq rows became GROUP descriptors
{qx_start,run_len,qy,qz,t_delta,slot_base,slot_step,0}; t derived from qz;
Variant gained groups[] {wr_c,wr_m,aq0,nq}; recording interleaves W(group)->
W(mirror)->A(group rows). NLM_REF switched from spec const id 8 to COMPILE-TIME
-D (glslang bounds-checks shared arrays against the spec const DEFAULT value,
so channel-count-sized LDS tiles require a real macro; weight now ships as 8
binaries nlmeans_{16,32}_weight_r{0..3}). Result: wSteady STILL 52us/batch.
**Conclusion: warm weight kernel is ALU/LDS/barrier-bound, NOT load-bound** -
the 65% "dist phase" saving measured cold was mostly loads that were already
L2-hits warm. Kept anyway (fewer DRAM refs, correctness intact, 52/52 tests).
GPU budget per frame (full 1080p config, measured x2 of half-height trace):
pad ~0.17 | weight ~0.86 | acc ~0.32 | finish ~0.09 ms ~= wall 1.46 (queue sat).
Next targets ranked: (1) acc mirror-weight gather u4a[sm][ym*STRIDE+xm] is
UNCOALESCED per pixel -> stage the shifted tile in LDS per aq row (coalesced
global loads), est -0.2ms. (2) kill pad kernel: compact slots + bounds-checked
kernel addressing + bulk vkCmdCopyBuffer upload, est -0.17ms. (3) fuse finish
into last acc batch via push flag, est -0.09ms. Sum would put wall ~1.0ms
(~1000fps) > goal 908.

## Round-packing experiment -> REVERTED to pack=1 (baseline best)
Scaled sweep batches by `pack` (entries per W/A round, ring scaled to match):
pack=2 -> 687 fps (= baseline), pack=4/8 -> 657, pack=all -> 640. Bigger rings
(>33MB) stream u4a through DRAM instead of staying cache-resident between W
and A; dispatch savings never materialize. NLMEANS_PACK env override added
(temporarily useful, default = budget formula clamped... final default pack=1).
**Measurement lesson: the two streams SHARE the compute queue, so per-frame
[gputrace] segments include interleaved other-stream work** - earlier "acc
floor 20us" numbers were artifacts. Clean single-stream trace (half-height
clip): copy=0.042 | W(all 62 entries)=0.37 | A=0.126 | finish=0.045 ms =>
full-config GPU ~=1.17ms/frame vs wall 1.46 @687fps (ns=2 hides most host).
Weight shaderstats: VGPRs=24 SGPRs=108 LDS=5120 no spills -> NOT occupancy-
limited; ACO simply isn't keeping many loads in flight (low ILP).
Next: (a) interior/border tile split in weight+acc - kills per-cell bounds
checks/address math for ~95% of workgroups (uniform branch), hoist per-row
bases; (b) pad kernel -> CopyBuffer2 + one-time margin fill; (c) fuse finish
into last acc round. Host compose(0.25)/download(0.22) already hide under GPU.

## Interior/border tile split + batched loads (working tree) -> 687 -> 728 fps
Uniform per-invocation predicate: if the WG's whole tile+halo (ref AND cand
extents incl qx/qy shift) is inside the frame, compute dist cells without any
per-cell bounds checks; each thread batches both its cells' guide loads before
dependent math (explicit temps force ILP; VGPRs were 24 = ACO serialized
loads). Border tiles keep the original guarded path. Acc kernel: hoisted
per-entry address bases (gpad const + qps=qy*PSTRIDE+qx; u4_mq via g-qss).
Single-stream half-height trace: W mega-launch 0.37 -> 0.30ms.
**Process lessons (again): (1) NEVER chain `cmake | grep error; cp && test` -
a failed compile silently re-installs the stale .so and you debug ghosts;
check install freshness (md5) before every test run. (2) BlankClip constant
input is NOT a correctness probe - vszipcl itself deviates up to ~49 LSB at
borders there (reference border semantics); only noise-clip comparisons vs
vszipcl count. (3) My repro passing identical a/s to both plugins is valid -
semantics match (tests prove it).**

## Status checkpoint: vsfeel PASSES vszipcl (759 vs 742 fps)
Ring-budget retune: pack chosen so ring bytes <= 64 MiB (measured optimum;
pack=1 -> 748, pack=2 -> 758, huge -> 717-728). Acc interior-split REVERTED
(branch cost > savings; acc arithmetic mostly latency-hidden anyway).
Clean NOSRC probe (single-stream): removing acc src loads saves ~25% of A
(124->93us halfheight) - src traffic DOES matter now; LDS src-tile cache is a
remaining candidate (~half recoverable).VRT 4/6 experiments: worse/broken ->
VRT_RESULT stays 3.
Remaining gap to 908 target (+20%): ~0.30ms/frame. Candidates:
(a) fuse finish into last acc round (-0.04ms), (b) pad kernel -> CopyBuffer2 +
one-time margin zero-fill (-0.10ms?), (c) weight kernel further ILP/LDS work,
(d) fp16 u4a weights (halves ring traffic; numerics risk).
NOTE: vszipcl fps varies 742-758 between runs; treat comparisons <5% as noise,
re-run before concluding.

## Session result (working tree, after user checkpoint commit)
**vsfeel 759.7 vs vszipcl 750.1 fps** (synthetic bench, ahead ~1.3%; real jpbd
clip 592 vs 545 = +8.7%). 52/52 tests pass.
Changes since checkpoint:
- Pad kernel ELIMINATED: staging tiles use the padded slot layout; compose
  writes interior rows at (PAD+y)*pstride+PAD; one vkCmdCopyBuffer per frame
  ships ALL new tiles (margins ride along as one-time-init zeros - compose
  never writes margins so they stay zero forever). Slots buffer already had
  TRANSFER_DST usage. Barrier became TRANSFER->COMPUTE.
- Finish kernel FUSED into last acc round via push constant pc3 (acc applies
  out=(center*m+num)/(m+den) inline; u2 store skipped on last round).
  Roughly neutral in fps (finish was mostly hidden) but removes a dispatch
  and barrier from the timeline. fin_pipeline still built, unused.
- acc interior split REVERTED (cost > benefit). VRT 4/6 dead. ns=3/4 worse.
Measured optima: pack=2 (=64MiB ring budget formula), num_streams=2,
VRT_RESULT=3, BX=16/BY=8.
Run-to-run noise is +-10fps per plugin; treat <5% deltas as inconclusive.
Remaining ideas for the +20% stretch goal (908fps), all with real cost:
(a) fp16 u4a weights - halves ring traffic (~+40-55fps est) BUT changes float
    output numerics beyond current test tolerance -> needs a numerics decision;
(b) subgroup-shuffle box sums to cut LDS phases (complex);
(c) acc src-tile LDS cache for small LAYERS*CH (needs compile-time gating,
    ~+30fps est);
(d) dead-code cleanup: stop building/dispatching pad+finish binaries.

## fp16 u4a weights landed (user-approved policy: close-enough like BM3Dv2)
WeightBuf is float16_t now (GL_EXT_shader_16bit_storage); ring bytes halve.
FIRST ATTEMPT drifted up to 2572 LSB / 0.036 float - the entire error was the
fp16 SUBNORMAL cliff (weights < 6e-5 lose mantissa bits exponentially).
FIX: store w*4096, multiply by 1/4096 on acc load -> all practical weights
stay in normal range. Drift dropped to **<=5 LSB / ~1e-4 float** and ALL 52
tests pass with ORIGINAL tolerances unchanged.
Answer to "does this help 16-bit int input?": YES - weights are format-
independent f32 intermediates; both i16 and f32 paths get identical benefit.
fps effect modest: ~760->767 (within noise band, repeat runs to confirm).

## Final session state (post fp16 + pack retune + BX=32)
Synthetic: vsfeel ~780 (3 runs: 783/787/770) vs vszipcl ~740 (739/744/736)
=> **+5.4% ahead**. Real clip: 601 vs 566 (+6.2%). 52/52 tests, original
tolerances (fp16 x4096 trick kept drift <=5 LSB).
Clean single-stream timeline (half-height clip, one mega W round):
copy(DMA)=42-59us | W=~390us | A=~125us; no inter-dispatch gaps - kernels are
~100% of queue time now.
W scaling fit: ~2.3-3.1us per sweep row-pass + small fixed part. Per-entry ALU
estimate says we run at ~15% ALU peak; occupancy is capped at ~2 WGs/SIMD by
VGPR=24 x 8 waves/WG (needs VGPR<=21 for a 3rd WG). ACO raised pre-sched
VGPRs 11->24 deliberately for load ILP; forcing lower regressed fps (tried:
guarded-load variant -> 713-733, reverted).
Exhausted/rejected this round: guarded loads (-6%), VRT 4/6, BY changes,
ns=3/4 (cache thrash), pack sweeps re-tuned post-fp16 (pack=4 optimal via
64MiB fp16 budget), exp() cost (~negligible), bulk compose memcpy
(impossible into padded layout; row loop is overhead-bound at ~8GB/s).
If the +20% goal (908fps) stays live, remaining candidates in order:
(1) cut weight-kernel VGPRs to <=21 without losing ILP (restructure phase
    interleaving) -> up to +50% occupancy;
(2) fp16 dist/hsum LDS arrays with range scaling (numerics risk);
(3) acc src-tile LDS cache gated on LAYERS*CH (~+30fps est);
(4) dead-code cleanup: pad/finish pipelines still built but never dispatched.

## 2026-XX: Duplicate-upload fix (writing-slot reuse) — new session, new bench

New benchmark reality (user updated bench.py with RAM frame cache; real jpbd
clip, 3000 frames, ns=2):
- **vsfeel 548 vs vszipcl 745 vs vszipcu 565** (1000 frames: 448/720/554).
  Old notes' "real clip 601 vs 566" was on the OLD bench; not comparable.
- Target restated by user: vsfeel must be AT LEAST 20% FASTER than vszipcl
  => >= 894 fps (1.12 ms/frame wall). Currently 548 (1.82). Need ~1.9x.
- Synthetic still fine: vsfeel 776 vs vszipcl 752 (docs valid).

Profile facts (int16 chain via depth(clip,16); a broken hand-written trace vpy
that dropped depth() caused a false "float32/2x bytes" scare — the benchmark
chain IS int16 YUV420P16; the user pushed back on the input theory twice and
was right: input is fine, problem is in the filter):
- Host per-frame (int16 real): compose ~1.1-1.4ms for ~4 tiles (was copying
  ~2x the needed bytes); download 0.7-1.0ms; acquire 0.6-0.9ms (incl cv).
- GPU (single-stream gputrace): ~2.0ms/frame both real and synthetic.
  vszipcl GPU (rocprof, single stream int16 real): weight 1.100 + acc 0.347
  + copies ~0.165 + fills 0.01 => ~1.67ms/frame. (Single-stream ROCm numbers
  clock-inflated; 2-stream walls: vsfeel 1.82, vszipcl 1.34 ms/f.)
- Memory type probe: staging is type 5 heap 0 (32GB, HOST_VISIBLE|COHERENT|
  CACHED) and row-loop into it self-tests at ~61 GB/s. HEAP IS NOT THE ISSUE.
- In-flight compose is ~3-6 GB/s effective (real ~2x synthetic): cold source
  reads + wide 32-thread contention, NOT the heap.

Root cause found: the tile cache steady state uploads ~4-5 new tiles/frame
(oscillating 0-10) instead of the ideal 2. Why: acquire's "writing" branch
made a stream treat a tile being uploaded by another in-flight stream as a
MISS and upload a private duplicate. Host drag doubles.

FIX (implemented, tests green): writing slots are now REUSED + a host-side
submit-order dependency. Reader records (writer_stream, writer_count =
submit_count[writer]+1 promised at claim); reader's GetFrame waits on the
writer's submission counter (cv) before its own submit. Single FIFO queue =>
writer's vkCmdCopyBuffer of the tile executes before reader's kernels. No
device-side semaphore/timeline. Error path bumps the writer's counter too.
DEADLOCK LESSON (do not regress): the first version used DEVICE-side timeline
waits (submit_timeline) — with a FIFO compute queue an early-queued CB can
wait on a signal only a later-queued CB produces => hard GPU-side deadlock
(repro: test_parallel_load_matches_serial hung, 24 worker threads stuck in
get_frame; bypassed 68-test suite). Host-side ordering is acyclic because
acquires are totally ordered by cache_lock (reader of a writing slot always
acquired AFTER that slot's writer). Must wait for the writer's SUBMIT, not
its full frame.

### Duplicate-upload fix result: 548 -> 664 fps (+21%), all tests green
- bench real 3000f ns=2: vsfeel 664 vs vszipcl 742 vs vszipcu 562. Wall/frame
  now 1.50ms; compose ~0.6-1.0ms (tiles now mostly 2-4), wait(GPU) ~1.7ms is
  the dominant phase -> GPU-bound again.
- Gap to target 894: still ~0.56 ms/frame. GPU is the next front:
  vsfeel ~1.7 vs vszipcl ~1.34 ms/f (2-stream walls). Must cut weight/acc
  kernels (vszipcl W=1.100ms is 8x smaller launches; acc=0.347).

### Benchmark restructure: pre-convert cache to filter input format -> 939 fps (>20%!)
The remaining real-video gap was NOT the filter: with the old bench, the
filter's input expression (depth(clip,16): float32->int16 dither) ran INSIDE
the timed region; 32 VS threads converting frames concurrently stole memory
bandwidth from the filter's compose/download. Evidence: real compose 0.83ms
at 32 threads vs 0.42ms at 4 threads (identical to synthetic); isolated
cold-row memcpy self-tests at 27 GB/s but in-run compose at 2.4 GB/s.
At 4 threads (benchmark hack): vsfeel 818 vs vszipcl 667 (+23%) — proves the
filter is fine; the standard 32-thread run was the wrong measurement.

Fix (benchmark/bench.py, user-approved direction "move where the conversion
happens"): make_vpy gains cache_conv; the cached frames are now pre-converted
into the filter's input format (per-filter: "depth(clip, 16)" for nlmeans,
"depth(get_y(clip), 32)" for bm3d) BEFORE the timed region. The chain's own
input expression then reduces to an identity (vstools.depth returns the clip
when formats match). Timed region now measures pure filter throughput, same
as --synthetic has always done.

RESULT (real jpbd, 3000 frames, ns=2): **vsfeel 939-943 vs vszipcl 767-773
= +22-23% — ABOVE the 20% target.** 68/68 tests still pass (benchmark-only
change; filter code untouched).

### Kernel: kill dead per-cell bounds checks in the weight interior path
Original "interior" path still computed per-cell `ina`/`inb2` boolean guards
(runtime bounds tests) even though the `interior` predicate already proves the
whole tile+halo (incl qx/qy candidate shift) is in-bounds — the compiler
cannot fold them from a runtime uniform bool, so they cost ALU + registers.
First attempt (split into sequential per-cell blocks) REGRESSED: 945->855 fps
(destroyed the batched 8-load ILP that ACO needs; notes said the same).
Correct fix: keep BOTH cells' loads batched and both math chains parallel,
store directly into LDS (drop `ina ? va : 0` selects). Result:
**945 -> 1011 fps vs vszipcl 768 = +32% ahead** (was +25%). 68/68 tests pass.
Instruction count: weight code 6252 -> 6060 (VALU 499 -> ~470), VGPR still 24.

### 16-bit 2-channel (UV) output was garbage — stale build artifact
Reported: 16-bit `channels="UV"` output completely wrong (O(65000) code diffs
vs vszipcl) on ALL params/num_streams; 16-bit 1/3-channel and all 32-bit paths
fine; no unit test caught it (tests only swept 16-bit GRAY and 16-bit YUV444).

Diagnosis trail (all on the noise clip, YUV420P16 + channels=UV):
- staging tiles + layer->slot table host-side were correct (dump);
- constant-weight probe (u4a = 0) -> output == input, so acc/finish/tables/
  upload path correct; the weight computation was the only suspect;
- u4a readback + host-visible u4a all showed correct weights/weird 3.8 avg
  (dump bug: doubled pointer offset, weights are ~0.6);
- **root cause: the build dir contained a STALE `nlmeans_16_acc.spv`
  (14616 bytes) that did not match the source (fresh build: 15664 bytes).
  The committed sources are correct — `rm -rf build && cmake -B build`
  clean rebuild matches vszipcl within 1-4 LSB on every 16-bit UV case
  (444/422/420, ns=1/4, full param range).** Keep the whole vk_spv tree
  consistent with src/*.comp; a partial regenerate leaves a broken
  combination (16-bit acc was stale while weight/finish/pad were fresh).
- The 16-bit wref=0 + default h=1.2 drift vs the reference (~4400 LSB on
  GRAY16 AND UV16 identically, 1 LSB at h=3.0) is the fp16 weight ring's
  subnormal quantization of tiny exp() weights (x4096 store) — a general
  16-bit property, not indexing; tests run that case at h=3.0.

Tests added (tests/test_nlmeans.py):
- `noise_yuv420_16` fixture; `UV32_CASES`/`UV16_CASES` sweeps mirroring
  REFERENCE_CASES -> `test_uv_matches_reference_32bit` (tol 1e-4, measured
  8.31e-5), `test_uv_matches_reference_16bit` (tol 8.0 LSB, measured max
  4.0); `test_multi_stream_uv_matches_single_16bit` (ns=4 == ns=1 exact).
  95/95 tests pass.

### --bits 32 benchmark bug (fixed 2026): "huge 32-bit drop-off" was fake
User reported vsfeel 663 vs vszipcl 749 at --bits 32 (16-bit: 1015 vs 775).
Root cause was in benchmark/bench.py, not the filter: `--bits` only rewrote
the CACHE conversion (`_input_for_bits` on cache_conv) but the CHAIN still
used `spec.input` = `depth(clip, 16)`. So with --bits 32 the cache held
float32 frames but the chain re-converted them to 16-bit INSIDE the timed
region — measuring the 16-bit filter under conversion contention again.
Fix: build the chain calls from the bit-adjusted expression:
  input_expr = _input_for_bits(spec.input, ns.bits) if ns.bits else spec.input
  calls = spec.build(ns, input_expr); cache_conv = input_expr
NOW MEASURES TRUE 32-BIT: vsfeel 860-862 vs vszipcl 661-662 (+30%), same
margin as 16-bit (vsfeel ~1000-1012 vs 768-770, +30%).
