# NNEDI3 — vsfeel port (Vulkan)

## References

- Primary: `reference/VapourSynth-nnedi3vk` (Vulkan, `nnedi3vk.NNEDI3`).
- Secondary: `reference/vapoursynth-zipcu` CUDA/HIP (`vszipcu.NNEDI3`, runnable
  here via the local HIP build at
  `/home/encode/test/vapoursynth-ziphip/zig-out/lib/libvszipcu.so`).
- CPU original `core.nnedi3.nnedi3` (znedi3) also installed.

## Reference agreement (measured 2026-09-04, noise_24f.mkv → YUV420P8, field=1)

- **nnedi3vk vs vszipcu: BIT-EXACT** (maxdiff=0 on frames 0/5/11, all 3 planes).
- nnedi3vk vs CPU znedi3: maxdiff=5, meandiff≈0.01–0.02, ~1% pixels differ
  (CPU float evaluation-order differences; ignored per project direction —
  **nnedi3vk is the ground truth**).
- Weights files are md5-identical in both references
  (`5c97e25c4a7277d06d3e3851373f1065`, 13574928 bytes).

Implication: the GPU math is fully determined (same weights, same formulas),
so a correct from-scratch port should land within ~1 LSB of nnedi3vk, and a
subgroup-cooperative predict kernel could plausibly go bit-exact later.

## Algorithm (as implemented by both references)

Per processed plane (output W×H, interp rows = H/2):

1. **Field extract** (host): kept field rows (every other input row from
   `parity`, or all rows in `dh` mode) packed into a W×rows buffer.
2. **Pad**: replicate to `(W+48)×(rows+6)` with clamped edges
   (`f = clamp(i-(MARGIN_V-fp))`, `c = clamp(x-MARGIN_H)`, `fp = 1-parity`).
   nnedi3vk pads on the CPU; vszipcu has a GPU pad kernel.
3. **Prescreen** (pscrn=0 skips): per interp pixel, small NN decides
   cubic-sufficient vs needs-predictor.
   - pscrn=1 (old): 12×4 window, 3 layers (4/4/4 neurons), one pixel/thread.
   - pscrn≥2 (new): 16×4 window, 2 layers (4/4), four pixels/thread.
   - Accepted pixels get cubic `(-3,19,19,-3)/32` on pad rows r+1..r+4.
   - Rejected pixel indices compacted into a list (atomics) for predict.
   - Prescreener math is strict non-FMA (`precise` / `-fmad=false`); the
     decision must match bit-for-bit or the pixel sets diverge.
4. **Predict**: per listed pixel, XDIM×YDIM window (nsize selects 8×6 … 48×6,
   8×4 … 32×4), mean/variance normalize, 2-layer NN (softmax + Elliott
   activations, nns = 16…256 neurons per type), wae5 softmax blend, `qual`
   passes averaged. Predictor MACs are explicit FMA (`__fmaf_rn` / `fma()`),
   reductions are subgroup butterfly adds. `exp(clamp(s,-80,80))`.
5. **Interleave** (host): kept lines copied from source, interp lines from
   the predictor output; `_FieldBased=0`, `_Field` deleted, duration halved
   for double-rate (field>1: 2 output frames per input frame, parity flips
   on odd outputs; `_FieldBased` prop drives parity when present).

Weight blob layout (`nnedi3_weights.bin`, f32 LE, sequential):
`ps_old(kernel_l0[4][48], bias_l0[4], kernel_l1[4][4], bias_l1[4],
kernel_l2[4][8], bias_l2[4])`, then 3× `ps_new(l0[256] stored transposed,
bias_l0[4], l1[16], bias_l1[4])`, then for etype 0..1, nns-sel 0..4,
nsize 0..6: q1+q2 × (softmax[nns·fs], elliott[nns·fs], softmax_bias[nns],
elliott_bias[nns]) — non-selected combos skipped as `4·nns·fs+4·nns` floats.
Host prep: prescreener mean-subtract ÷ pixel_half (0.5 float, peak/2 int);
model mean-subtract (per-neuron means + cross-neuron mean filter + bias mean).
Predictor upload layout per qual pass: weight pairs `(sm,el)` as
`pdW[(q·fs+k)·N+p]`, biases `pdB[(q·2N+p)] = (smB, elB)`.

## vsfeel design (own structure, not a nnedi3vk mirror)

- `src/nnedi3.comp`, five entry points, BITS=16/32:
  - `ENTRY_PAD`: clamp pad kernel (own trivial kernel, cf. vszipcu's).
  - `ENTRY_PRESCREEN`: one thread per pixel group (P=1 old pscrn, P=4 new)
    doing cubic taps + prescreener verdict inline; accepted pixels get cubic
    stored, rejected indices compacted into a list with one atomicAdd per
    128-thread workgroup (subgroup-aggregated totals in shared memory).
  - `ENTRY_PREDICT`: cooperative predictor over the list (pscrn>0) or all
    pixels (pscrn=0): one 32-lane subgroup per PXP pixels (8 when
    NNS<=64 and FS<=128, else 4), window stats via subgroup adds, GEMV from
    shared-memory tile, wae5 blend; fixed 2048-group direct dispatch with a
    GRIDPX-strided loop tail (count-bounded, always correct).
  - `ENTRY_COUNT`: 1-thread `groupsX = ceil(count/16)` derivation (created
    but not dispatched yet — indirect launch is still env-gated experimental).
  - `ENTRY_ASSEMBLE`: interleave kept field rows + packed interp rows into
    the full frame, written straight into the staging download area
    (kernel-direct download, no D2H copy).
- `src/nnedi3.cpp`: vsfeel-style host (FramePool, staging upload, device-local
  field/pad/dst/list/indirect buffers, persistent-mapped weight buffers, one
  CB per frame: H2D → pad → prescreen → predict → assemble). Field extract +
  interleave... (interleave now on GPU via assemble). Spec-constant dims.
- Weights: `src/nnedi3_weights.bin` committed (13.5 MB), objcopy-embedded.
- Args: field/dh/planes/nsize/nns/qual/etype/pscrn/device_id/num_streams.
  Depths: 16-bit int + 32-bit float (8-bit/f16 rejected with a clear error).

## Log

- 2026-09-04: agreement check (vk==cu bit-exact), fused-MVP design + build.
  MVP accuracy 46/46 pass (16-bit within 1 LSB, f32 within ~3e-8); speed
  ~1130 fps vs nnedi3vk ~2095 fps (~0.54x, jpbd.mkv 1080p GRAY16 field=1).
- 2026-09-05: prescreen-list + cooperative-predict split landed (uncommitted).
  Two correctness bugs found and fixed during stabilization:
  1. Predict GRIDPX loop tail was `#if 0`-disabled (timing-experiment
     leftover) — only the first 32k pixels computed. Re-enabled.
  2. `indCount` (indirect-struct word at byte offset 12) was never reset —
     the per-frame FillBuffer cleared only bytes 0..12 (groupsX/Y/Z). Counts
     accumulated across frames on the reused resource (LIFO pool): frame 11
     read 157+134=291, frame 23 read 157+134+5173=5464. Stale list entries
     made predict overwrite cubic-accepted pixels with predictor values
     (sparse, large, history-dependent diffs). Fixed with a third FillBuffer
     (offset 12, size 4). Mechanism verified via VSFEEL_NNEDI3_COUNT
     readbacks; order-swap probes (frame 23 alone vs after 0+11) now agree.
  After fix: 46/46 pass. Missing-`field`_arg create-path abort observed in a
  probe (omitted optional arg → terminate in Nnedi3Create); tests always pass
  field explicitly so out of scope for now, flagged for later.
  **(2026-09-16, WO-01) FIXED.** Registering `"field:int:opt;"` while reading
  it with a null error pointer meant VapourSynth took VS_FATAL_ERROR ->
  `fprintf` + `std::terminate`; the process died with SIGABRT (exit 134) and
  `try/except` never ran. `field` is now registered required (`"field:int;"`,
  matching EEDI3), so omitting it is a normal catchable `vs.Error`
  ("NNEDI3: argument field is required"). Regression test:
  `tests/test_nnedi3.py::test_nnedi3_requires_field`.

- **(2026-09-16, WO-02) FIXED: prescreen dispatch grid under-covered the
  frame.** Host computed `pre_grid_x = ceil(width*rows / (P*128))`, but
  `nnedi3.comp` groups pixels per ROW as `ceil(width/P)` (r = gid/xg), so the
  exact requirement is `ceil(rows*ceil(width/P)/128)`. Whenever `P` does not
  divide `width` (P=4 for the default pscrn>=2) the last row's tail groups
  were never dispatched, and since the D2H copy ships the whole packed interp
  region, **uninitialized VRAM reached the output**. Measured before the fix:
  YUV420P16 1924x1080 (chroma 962) -> **182 garbage pixels per chroma plane**,
  every frame, values 0..65527 (the correct output is constant 30000); luma
  1924 (divisible by 4) and pscrn=0/1 were unaffected. Fix follows the
  reference (`nnedi3vk.cpp:1121-1123`, threads = rows*ceil(width/P)):
  `groups_per_row = ceil(width/P); pre_grid_x = ceil(rows*groups_per_row/128)`.
  The suite was green because every fixture is 640 wide (chroma 320).
  Added the 630/638 case to `tests/test_geometry.py` (vsfeel vs nnedi3vk,
  GRAY16 pscrn=2, frames 0..2): **bit-exact, 0 codes** at 630/638; a probe also
  covered 640/626/610 (all 0). Repro kept in `tmp/wo02_repro.py`
  (constant-clip tail scan, made deterministic by reverse frame order / pool
  poisoning) and `tmp/wo02_oracle.py`.

- Accuracy (still current, same 46 tests): 16-bit within 1 LSB of nnedi3vk
  everywhere tested (0 on most content; isolated 1-code rounding flips from
  serial-vs-butterfly reduction order — mechanism, not a bug). f32 within
  ~3e-8 (bound 1e-6). Covers field 0/1/2/3, dh, planes subsets, nsize 0..6,
  nns 0..4, qual 1/2, etype 0/1, pscrn 0..4, GRAY16/YUV420P16/GRAYS,
  determinism, 1-vs-4 streams, parallel load, props (2x frames, 2x height,
  _FieldBased=0, no _Field).
- Old speed (fused MVP, jpbd.mkv 1080p GRAY16 field=1, 2 streams): vsfeel
  ~1130 fps vs nnedi3vk ~2095 fps (~0.54x). The split above is the speedup.
- MVP limits (future work): 8-bit/f16 input rejected; dh + planes-subset
  zeroes interp lines on skipped planes (reference leaves them
  uninitialized); default num_streams=2 (no knee sweep yet).

## Reference orchestration (vszipcu HIP path, read 2026-09-05)

`process()` per GRAY frame is FULLY SERIAL on one stream: H2D field (2 MB,
stream2) → event → stream waits → pad → memset → prescreen → predict →
sync ×2 → D2H dst (2 MB) → sync. Host packs field BEFORE stream acquire,
interleaves kept/interp AFTER release (spool = streams+2, same as our
snapshot design). No cross-kernel overlap at all — yet wall 430µs.

Their GPU (rocprof): H2D ~129 + pad ~32 + pre ~53 + pred ~81–122 + D2H ~87
≈ 380µs total. Ours (query pool): h2dpad ~95 + pre ~45 + pred ~185 + asm
~22 ≈ 350µs. SAME total! So kernels are NOT the gap — host + submit is:
our pack alone is ~175µs for 2 MB (12 GB/s — way too slow for memcpy;
should be ~60–80). Suspect: NT stores (`copy_stream_out` + `_mm_sfence`)
on the GTT upload mapping stall. NEXT: pack microbenchmark (NT vs memcpy
vs cached) + check upload memory type actually landed in GTT/WC.

## Plan (own implementation, vsfeel plumbing)

1. **Zero-copy upload** (this): split staging — upload side VRAM-mapped
   (`DEVICE_LOCAL|HOST_VISIBLE|COHERENT`, NT stores, GPU reads at full
   speed), download side stays GTT. CPU pre-pads (row-clone); delete
   field_buf, H2D copy, pad kernel + pipeline. Prescreen/predict bind the
   upload staging as their pad source. [DOING]
2. **Indirect predict**: dispatch the (already-written) count kernel after
   prescreen, launch predict with `vkCmdDispatchIndirect`, stride the GRIDPX
   loop off device-read `indGroupsX`; drop prescreen's TEMPORARY atomicMax
   and the fixed-2048 grid. [NEXT]
3. Transfer-queue async D2H: only if 1+2 still trail (needs queue-family
   probing + cross-queue sync; deferred).

## GPU timestamp truth (2026-09-05, own query-pool probe vs NNEDI3VK_PROFILE)

Their profiler on jpbd 1080p: prescreen 50µs, predict 63µs, copy 92µs
(GPU 205µs; wall 430µs). Ours (same content): pre 80µs (fills+prescreen+
count), pred 178µs, copy 86µs. Copy at parity; prescreen ~1.6x; **predict
2.8x is the kernel gap**.

Isolated so far (each refuted or banked):
- Window-pass fusion 3→1 global + register tiling: banked (+7% wall).
- `restrict` on weight buffers → scalar loads (their ISA pattern): banked,
  prescreen barely moved (-14µs) — L2 weight traffic was not the bottleneck.
- Predict ISA 2804→1405 (single-path, exact grids): banked (+12% wall).
  Ours now smaller than theirs (1633) with identical math — gap persists.
- PXP compile-time folding, 64-thread groups, 256-thread prescreen: all
  neutral (reverted). Coopvec tensor path: NOT engaged (their coopvec=False
  still predicts in 67µs).
- Dense rate 2ns/px both (pscrn=0 full-frame 2056µs/1M px); sparse 11k px
  costs us ~150µs (14ns/px) vs their ~40µs over dense rate. The sparse
  penalty is orchestration (list/count/indirect/tiny-grid), not math.

## Decision: fused kernel (own design, neither reference fuses)

One thread per pixel group (P=1 old, P=4 new, shared L0): verdict inline,
rejected pixels predicted inline with serial math + neuron chunking (tile
32 → bounded registers for all NNS), cubic for accepted. Writes packed
interp to device dst (D2H + host interleave unchanged, zero-copy upload
unchanged). Eliminates: list buffer traffic, count kernel, indirect
launch, fills, sparse-grid inefficiency. Expected GPU: verdict-full-grid
~50 + inline-predict ~22 + D2H 86 ≈ 160 vs split ~315.
Accuracy: serial (non-butterfly) reduction order, same formulas — the fused
MVP already proved ≤1 LSB / 1e-6 on this exact tradeoff.

## Session status 2026-09-05 (STOPPED, tree RED — read before touching)

Working tree (uncommitted, `git diff --stat`: comp 501+/-, cpp 804+/-):
`python -m pytest tests/test_nnedi3.py -q` → **28 passed, 18 failed**.
Do NOT benchmark this tree; fix correctness first.

### What landed and is banked (each verified 46/46 at the time)

1. Zero-copy upload: CPU pre-pads into VRAM-mapped upload staging
   (`DEVICE_LOCAL|HOST_VISIBLE|COHERENT`, NT stores), prescreen/predict bind
   it as pad source. field_buf/H2D/pad-kernel deleted. Interleave split into
   kept-pass (`memcpy` from source) + interp-pass (`copy_stream_read`).
2. Indirect predict: count kernel derives `groupsX = ceil(count/PPG)` after
   prescreen; `vkCmdDispatchIndirect`, device-strided loop; prescreen's
   TEMPORARY atomicMax removed.
3. Predict single-path (exact grids, no GRIDPX loop tail): ISA 2804→1405,
   +12% wall. `restrict` on weight buffers (scalar loads). Window-pass
   fusion + register tiling (+7% wall).
4. UPTO/count/timestamp/bench probes are TEMPORARY env-gated code still in
   tree (`VSFEEL_NNEDI3_UPTO/COUNT/TSTAMP/BENCH`); query pool per resource.
   Pipeline dedup key widened (geometry + pscrn/xdim/ydim/nns/qual).
   Per-frame CB re-record kept (cheap, keeps COUNT-debug pool use simple);
   record now happens AFTER the CPU pre-pad with a host-write→shader-read
   barrier in the CB plus `_mm_sfence()` after the NT stores.

### Fused kernel experiment: tried, measured, REVERTED

`ENTRY_FUSED` (verdict inline + serial chunked-neuron predict, no list, no
subgroups) built, passed 46/46, but GPU timestamps showed **pred 464µs vs
split 178µs** (verdict now runs on the full grid + rejected pixels diverge
across threads; ISA 2344 insns, FMA-starved: 222 fmac vs 339 mul + 484 add).
Wall 824 fps vs split ~1960. Fully reverted (shader block, CMake entry,
header-gen args, host module/pipeline/CB). A staged chunk-32→8 neuron-tile
edit was discarded with the revert, never built — mentioned only so nobody
re-derives it; it would not have fixed the verdict-full-grid cost anyway.

### Current failure (the thing to fix next)

- pscrn=0 and pscrn=1 pass everywhere; **pscrn=2/3/4 fail** on noise content
  (e.g. frame 23: maxdiff ~4300, ~900–1000 px >1 LSB; frames 0/1/4/7/11/13
  pass, 16–23 fail). jpbd blank clip passes (content-dependent).
- Wrong pixels are ALL on interp (odd) rows, kept rows exact (maxdiff 0);
  ~half the wrong values equal the source pixel → prescreener verdict flips
  (cubic-accepted pixels getting predictor values or vice versa), not a
  predict-math error.
- Frame-23 result varies across fresh instances (ndiff 40…927) but rereads
  of one instance are stable → nondeterministic GPU-side input to the
  verdict, not host math (host parse/pack is deterministic; pscrn=0/1 read
  the same pad and pass).
- Refuted so far: NT-store tearing (`memcpy` isolation: identical failures),
  missing sfence (added, no change), missing host barrier (added, variance
  pattern changed but failures persist), pipeline-dedup key (widened, single
  Gray plane never dedups anyway), count-reset bug (counts read back exact:
  pscrn=1 frame23 count=8658, pscrn=2 count=1003 — both stable and sane).
- Open hypotheses (unproven, in priority order):
  1. The shared-L0 refactor (`prescreenL0New` once per group + `verdictNew`
     vs committed per-pixel `needPredictNew`): mathematically identical ops,
     but the committed MVP passed 46/46 with per-pixel floats — revert the
     verdict to per-pixel form as a one-shot bisect.
  2. Prescreen compaction race under the zero-copy upload (128-thread groups
     + `sgCounts[4]` assume exactly 4×32-lane subgroups; device reports
     max subgroup 64 — if any dispatch runs wider subgroups, totals
     undercount and stale dst pixels survive). pscrn=1 shares the compaction
     code and passes, which weakens but does not kill this (contention
     differs: 8658 vs 1003 entries).
  3. Something in the fused-revert left the split path inconsistent (the
     last green was single-path+indirect+zero-copy at ~1960 fps/2-stream;
     the restore touched CB order, key, sfence, barrier). Bisect host
     changes one at a time — each is a small revert.
- Suggested next step: hypothesis 1 first (10-minute shader-only bisect,
  no host churn). If green, re-derive the shared-L0 form carefully or keep
  per-pixel verdict (L0 recompute ×4 costs little: verdict is not the
  bottleneck — predict is).

## Session 2026-09-05 continued — transfer queue → GPU pad → GPU assemble → ReBAR/predict-lane wins
(tree GREEN, 46/46 throughout unless noted; real-world bench defaults per
user: `field=3 dh=0 nsize=0 nns=4 qual=2 etype=0 pscrn=4`, jpbd 1080p GRAY16)

### What landed since (all verified 46/46, best-of-3, same-session pairs)

1. **Count kernel folded into prescreen** (`src/nnedi3.comp` ENTRY_PRESCREEN:
   `atomicMax(groupsX)` off the `atomicAdd` return value — the nnedi3vk
   reference pattern — replacing the 1-thread count dispatch + 2 barriers;
   `ENTRY_COUNT` shader retained but undispatched). +~90 fps at 2 streams
   (1689 → 1775 official).
2. **Zero-copy ReBAR upload** (upload staging `DEVICE_LOCAL|HOST_VISIBLE|
   COHERENT`, types 3/4; CPU packs tight field rows with NT stores; pad +
   assemble shader-read it directly — H2D DMA + field_buf deleted;
   descriptor binding 0 rebound up_staging→pad). h2dpad 95 → 17µs
   (H2D ~80µs gone, pad ~12–17µs remains). +~300 fps at 2 streams
   (1775 → 2078 official; 2210 best-of-3). Matches nnedi3vk's
   `PREFER_DEVICE` upload buffer.
3. **Kept lines pre-acquire** (reference pattern: kept pass memcpy'd from
   src BEFORE `pool.take()`, interp scattered from staging post-fence;
   parity hoisted above take; t_kept now pre-acquire). +~100 fps at
   4 streams (4-stream 2029 → 2200 best-of-3); 2-stream ~1985–2010
   (interp scatter NT vs memcpy both profiled: NT 161–163µs wins over
   memcpy 261–269µs — scattered 2KB rows still prefer NT here because the
   sfence cost sits in t_pack, already paid; do NOT re-derive blindly).
   NOTE 2026-09-06: a cleaner KEPTPOST A/B re-test (pre vs post-fence
   full-frame copy, best-of-3 @2 streams) showed 2006 pre vs 2025 post —
   NEUTRAL. The reference's win comes from its rb-slot pool releasing the
   stream early, not from the copy placement itself; our fence already
   rides the copy CB so placement doesn't move the needle. No change.
4. **PXP=4 split variant** (earlier): `predict_n4` (`-DPXP=4`, own module +
   `use_pxp8()` host select) for wide networks (NNS=64..256, FS>128);
   pred 319 → 230µs at the time (now ~185µs with the rest).

### 2026-09-06 follow-ups: pack re-tune + over-launch (tree GREEN 46/46)

1. **Pack writer re-tune on ReBAR: NT kept.** `PACKMEMCPY` A/B on the new
   uncached-VRAM target: 1-stream NT 1075 vs memcpy 1027 best-of-3 (pack
   143 vs 160µs); 2-stream within noise (2057 vs 2042). The old GTT verdict
   (NT wins) still holds on ReBAR. Probe removed, NT + sfence stays.
2. **Predict over-launch: REFUTED, indirect stays.** `PREDIRECT=1`
   (full-grid direct in list mode, early-exit bounded): pred 268 vs 223µs
   GPU, 1892 vs 2055 fps best-of-3 at 2 streams — SLOWER. The exiting
   subgroups still pay window-gather + occupancy before their firstPix
   check. Probe branch removed (comment records the numbers); the
   `pred_grid_direct_x` path stays for pscrn=0 only, where it is required.

### 2026-09-06 continued — the push past parity (tree GREEN 46/46 throughout)

Radical round, each step measured best-of-3/median-of-5 same-session pairs
(real-world defaults, jpbd 1080p GRAY16). Starting point: 2s ~0.79x,
4s ~0.83x.

1. **GPU assemble deleted, D2H packed interp only.** Assemble re-read kept
   lines from ReBAR, rewrote them to asm, and D2H'd 4MB of which the host
   used only the 2MB interp half. Now: single pred→transfer barrier, D2H
   the packed interp dst directly (2MB), host scatters from tight rows.
   Instant +8%: 2s 0.79→0.93, 4s →0.95. (asm_buf still allocated but
   undispatched — dead VRAM ~4MB/stream, cleanup later.)
2. **Pack reads dst kept rows (cache-hot) instead of src.** Non-dh only
   (dh pack is contiguous from src; dst would be strided). Saves the 2MB
   DRAM source re-read; 4s →0.96/0.99 across runs.
3. **PXP=4 small-tile variant (`predict_n4s`, `-DPXP=4 -DSHSTRIDE=64u`).**
   The fixed `shTile[4*288]` (18KB, shaderstats LDS 15360B) throttled
   occupancy for small windows; FS<=64 networks (nsize 0/4/5 incl. the
   bench default FS=48) now use 256 vec4s (4KB, LDS 4096B). Same ISA
   (1674). 4s →0.99.
4. **Transfer queue deleted, always inline.** Same-family xfer measured
   SLOWER at every depth (extra submit + timeline + sync > 2MB overlap):
   NOXFER A/B +2-3% at 2s and 4s. Deleted copy_pool/cmd, timeline,
   record_copy_buffer, xfer selection, per-frame vector allocs. 1s
   0.80→0.87, 2s →0.96, 4s →1.05 (first lead!).
5. **Pad kernel fused into window reads.** `loadPad` now clamps the tight
   ReBAR field directly (`fp=1-parity` from push word3; prescreen/predict
   pushes carry parity, B=up_elem). Deletes pad dispatch + barrier (+20us
   GPU); clamp ALU costs ~3-7us on pre/pred (L2-hit traffic). 46/46 first
   try. 2s →0.977.
6. **REQUIRE_FULL_SUBGROUPS on pre/predict** (like the reference; control
   flow is subgroup-uniform so safe). 2s →0.985.
7. **Redundant host barrier deleted.** vkQueueSubmit already orders pack
   (before submit + sfence) before device work — the reference records
   none. Kept defensively since zero-copy debugging; determinism green
   without it. 4s →1.062.
8. **Pre-recorded parity CBs, UPTO deleted.** Two CBs per resource
   recorded once at create (swap idiom — record writes resource.cmd);
   GetFrame just selects. UPTO (TEMPORARY stage-truncation) incompatible
   with pre-record, served its purpose, deleted. ~flat (record was
   already overlapped) but simpler per-frame path.
9. **Timestamp slot trim (5→4 queries).** Deleted the marker timestamp
   (each write costs a bubble); slots now 0=top 1=pre 2=pred 3=copy,
   `t_ts_copy` replaces `t_ts_asm`. 2s distributions overlap ref first
   time (0.984).

### Perf state (2026-09-06 final, median-of-5 unless noted)

- 1s: vsfeel ~1330 vs ref ~1550 (0.86x).
- 2s: vsfeel ~2320 vs ref ~2360 (~0.985x, distributions overlap).
- 4s: vsfeel ~2640-2680 vs ref ~2520 (**~1.05x — ahead**).
- 6s: vsfeel ~2765 vs ref ~2560 (~1.08x).
- 8s: vsfeel ~2814 vs ref ~2584 (~1.09x, still climbing; ref plateaus
  ~2540 from 4s).
- Reference profiled (NNEDI3VK_PROFILE, 1s): pre 34 / pred 179 / copy 89.
  Ours: pre ~55 / pred ~195-200 / copy ~89 (D2H at exact parity; compute
  +20us each — VGPR 192 predict / 96 prescreen occupancy vs ref, accepted
  as the residual; host bytes already minimal).
- Host (1s): pack ~170 + kept ~335 + interp ~87 + submit ~18 ≈ 600 serial;
  the kept 2MB strided memcpy (6GB/s effective) is the single biggest
  item and hardware-bound. Further wins need fewer host bytes
  (VK_EXT_external_memory_host zero-copy — high risk, not attempted) or
  more overlap (streams — done, see defaults).
- **Defaults set at the knee (user cap: ≤4): filter `num_streams` 2→4,
  bench nnedi3 `default_streams` + build fallback 2→4.** Out-of-box:
  ~2640 vs ref default-2s ~2360 (+12%); same-4s h2h +5%. Per-stream
  ~18MB VRAM (incl. dead asm+pad ~8MB — cleanup would drop to ~10MB).

### Perf state (best-of-3, 2026-09-05 post-ReBAR)

- 1 stream: vsfeel ~986 vs nnedi3vk ~1625 (0.61x). Host: pack ~150–160 +
  kept ~320–365 + submit/record ~35 ≈ 520 serial; GPU ~260
  (h2dpad ~17 + pre ~40 + pred ~185 + asm ~21). wait ~607–625 (1-stream
  wait ≈ GPU + DMA since nothing overlaps).
- 2 streams: vsfeel ~1985–2010 vs ref ~2500–2540 (~0.79x).
- 4 streams: vsfeel ~2200 vs ref ~2643 (~0.83x).
- Reference profiled (NNEDI3VK_PROFILE, 1-stream): prescreen 34µs,
  predict 179µs, copy 89µs. Ours: pre ~40, pred ~185, asm ~21, pad ~17 —
  **kernels at parity** (pred within 3%, prescreen within 18%). The
  remaining gap is HOST-side (pack + kept + interp ≈ 500–600µs serial
  vs their overlapped pre-acquire copies) + launch structure, not math.

### Reference orchestration (read, not mirrored yet — deltas vs ours)

- vszipcu HIP (`reference/vapoursynth-zipcu/src/nnedi3.zig::process`):
  FULLY SERIAL per frame (H2D → pad → prescreen → predict → sync ×2 →
  D2H → sync) but pack+interleave with plain memcpy on 2MB buffers.
  Predict over-launches `ceilDiv(w*rows,16)` blocks (sparse frames exit
  via `firstPix >= npix`); NO indirect/count/atomicMax anywhere.
  NO transfer queue (single stream2 for H2D/D2H, p==0 compute on stream).
  Their wall wins come from small host copies, not overlap.
- nnedi3vk Vulkan (`recordAndSubmit`): push descriptors per plane (no
  descriptor sets), dedicated DMA family (transferQueue != queue on their
  device; on 7900XTX NO dedicated transfer family exists — families are
  GFX(1q)/COMP(4q)/VID-dec/VID-enc — so their split also rides a
  same-family queue here), readback slots pooled separately
  (numRbSlots = numStreams+2, PREFER_HOST cached) so a stream never
  stalls behind a frame still copying rows out. Predict WG sizing:
  `subgroupsPerWG = min(4, maxWG/sgSize)`, `predictWG = sg*subgroups`,
  indirect via `atomicMax(groupsX)` in prescreen (folded above).
  Upload: `PREFER_DEVICE` ReBAR (mirrored above).

### Rules learned (do not re-derive)

- **7900XTX has NO dedicated DMA family** (vulkaninfo: GFX 1q, COMPUTE 4q,
  video-dec, video-enc, sparse). Our "transfer queue" is a second
  same-family compute queue — 2-stream NOXFER A/B showed +75 fps
  (1755→1830), so it helps slightly, but it is NOT the reference's DMA
  engine. Do not chase queue topology further.
- **UPTO + TSTAMP together HANG**: truncated CBs never write queries 2..4
  and the WAIT_BIT readback blocks forever (killed two vspipe runs).
  Run UPTO benches with BENCH only, never TSTAMP.
- **Interp scatter prefers NT stores here** (161µs vs memcpy 261–269µs):
  the sfence cost is already paid in t_pack, so per-row NT has no extra
  fence penalty. Re-measure if the pack path changes.
- (Older rules below: barriers per dispatch pair, copy_stream_read
  alignment, PXP shared-footprint, YUV push-layout atomicity — all still
  hold.)

### What landed (all verified 46/46)

1. **Transfer-queue split submit** (`src/nnedi3.cpp`: per-resource
   `copy_pool`/`copy_cmd`/timeline/`xfer_queue`; `record_copy_buffer` does
   per-plane D2H + TRANSFER→HOST barrier on a second same-family queue;
   `submit_timeline` compute-signals/copy-waits, fence rides the copy;
   `VSFEEL_NNEDI3_NOXFER` fallback; teardown extended). NOXFER A/B showed
   copy overlap delta ≈ 0 — the D2H was never the bottleneck.
2. **Decoupled host overlap**: after invalidate, snapshot staging to a heap
   buffer + early `pool.give_back` when `num_streams > 1` (single-stream
   copies in place); interp/frame pointer switched to snapshot base.
   Exactly-once rule: every `set_error` path must run BEFORE the early
   give_back (resource is moved-from there); final give_back conditional.
3. **GPU pad kernel** (replaces zero-copy CPU pre-pad): CPU packs tight
   field rows (no margins) into GTT upload with NT stores
   (`copy_stream_out`, measured faster than memcpy: ~300 vs ~440µs pack at
   2 streams); CB does H2D DMA → device-local field → pad kernel →
   device-local pad. Prescreen/predict window reads now run at full VRAM
   speed. Descriptors rebound 0=field, 1=pad (was up_staging ×2).
   synth-input proof: pred 7µs blank (list ~empty) vs 190µs real.
4. **GPU assemble**: new device-local `asm_buf` full frames; assemble entry
   (`A=field B=dst C=asm`, parity push) interleaves on GPU; D2H now ships
   full frames (4 MB, was 2 MB packed interp); host does ONE memcpy per
   plane + early-give_back snapshot. `ENTRY_ASSEMBLE` was already compiled
   (usage flag `STORAGE_BUFFER_BIT` on staging was pre-added for it) —
   only needed dispatch + barrier wiring.
5. **SKIPIL hang fix**: timing path skipped copies but never returned the
   resource → pool drained after `num_streams` frames. Now SKIPIL still
   snapshots-skips but always give_backs. (SKIPIL long runs also trip an
   unrelated `radv/amdgpu: CS cancelled, context lost` — garbage-out
   timing path only, not investigated.)

### Rules learned (do not re-derive)

- **Every producer→consumer dispatch pair needs an explicit barrier.**
  The assemble-missing-predict-barrier bug failed 21 tests with ~927–970
  wrong interp pixels (odd rows only, ~half equal to source = stale dst);
  adding shader-write→shader-read barrier → 46/46. Same mechanism as the
  earlier prescreen→count barrier. Back-to-back dispatches order NOTHING.
- **`copy_stream_read` faults on unaligned SOURCE** (uses
  `_mm256_stream_load_si256` = aligned load; heap vector + staging offsets
  are not 32B-aligned) → SIGSEGV inside libvsfeel. Snapshot now does manual
  head-align + `_mm256_loadu` + NT store + sfence; dst write uses
  `copy_stream_out` (unaligned-load variant). Staging offsets are 32-aligned
  but the map base is not — never assume source alignment.
- **PXP=16 restructure REVERTED (root-caused)**: making PX fixed 16 with
  per-pixel shared stride 32 (`shBase = warp*SHSTRIDE + px*32`, SHSTRIDE left
  at 288) overflows `shTile[4*288]` for warp≥1
  (warp*288 + 15*32 + 31 > 1152) → OOB shared writes → fresh-instance
  nondeterminism (maxdiff ~3000, thousands of px) + 26 failed. Baseline
  (stash test) deterministic → bug was in the experiment, not the machine.
  Lesson: shared-tile footprint must be re-derived (≤1152 vec4s) for ANY
  PXP/stride change; fresh-instance determinism test (`nondet3.py` pattern:
  two instances, same process, frame 23) catches it in seconds.

### Perf state (same-session pairs, 2 streams unless noted)

- vsfeel ~1420–1460 vs nnedi3vk ~2500–2560 (was ~1830 vs ~2500 before GPU
  pad/assemble — REGRESSED at 2 streams; see analysis). 4 streams:
  ~1600 vs ~2700. 1 stream: ~890–917 vs ~1650–1760. 8: ~1637. 16: ~1675
  (pack 2195 + kept 7345 blow up — host copies don't scale past ~8).
- GPU timestamps, 5-point (h2dpad/pre/pred/asm): 1-stream real content:
  **h2dpad ~95–97, pre ~45–46, pred ~183–190, asm ~22–23**.
  Decomposition: UPTO=1 (H2D+pad+prescreen+count, no predict) wall 1073 vs
  full 890–917 → predict adds only ~150µs wall at 1 stream. UPTO=2 ≈ UPTO=1
  (count is noise). So per-frame GPU ≈ 95 + 45 + 190 + 22 = 350µs, but
  wall-between-UPTO diffs says predict contributes ~150 and H2D+pad+pre
  ~800?? Inconsistent — overlap hides GPU behind host (wait 700 ≫ GPU
  350). Host bill at 1 stream: pack 175 + kept(snapshot+frame) ~180–195 +
  submit/record ~35 ≈ 400µs serial-ish; wait (fence) 700 covers GPU+DMA.
- Reference (rocprof, qual=2 real): H2D ~129 (2 MB field @ ~16 GB/s),
  pad ~32, prescreen ~53, predict ~81–122, D2H ~87 (2 MB). Their GPU total
  ≈ 380µs vs ours ≈ 350µs — SAME total! But their wall is 2x better
  (430 vs 900µs) → the difference is HOST-side + overlap, not kernels:
  they pack+interleave with plain memcpy on 2 MB buffers while doing GPU
  on streams; our host does NT pack (175) + snapshot (NT 4 MB) + frame
  copy (4 MB) + record/submit per frame, and our 2-stream overlap only
  reaches ~1.35x (1444/2 per stream... actually 2-stream wall 1444 vs
  1-stream 900 = 1.6x scaling — decent but from a worse base).
- D2H note: old copy(ts) ~20–22µs for 4 MB (= 190 GB/s, impossible over
  PCIe) → the xfer-queue timestamp overlaps compute; NEVER trust it.
  NOXFER A/B earlier showed overlap delta ≈ 0.
- **pscrn=0 dense is at parity** (ours ~177 vs ref ~191 fps, 1-stream;
  GPU pred 4742µs both over the full 1M-px grid) → the math is fine, the
  sparse path (prescreen+count+indirect+assemble+extra barriers) is the gap.

### Next (hypothesis order)

1. **Predict over-launch**: ref launches `ceilDiv(w*rows,16)` blocks
   REGARDLESS of count (over-launch + `firstPix >= npix` early-exit); we
   launch exact `ceil(count/PPG)` indirectly (e.g. groupsX=16 for
   count=1003). Tiny dispatches can't fill 96 CUs → latency-bound. Cheap
   test: direct full-grid dispatch, keep count-bounded early exit.
2. **Prescreen 140→50**: same VRAM window source now, still 2.8x. Compare
   ISA against ref pattern (their prescreen: scalar `float v[EPL]`, warp
   shuffles, no shared); check our VGPR/occupancy + `precise` chains.
3. Reconsider total work: GPU pad+assemble ADDED a pad kernel, an assemble
   kernel, 2 extra barriers, 2 MB H2D + 2 MB extra D2H vs nnedi3vk's
   zero-copy pre-pad + 2 MB D2H. If 1+2 close the kernel gap but wall still
   trails, the remaining lever is host copies (pack + snapshot + frame
   copy ≈ 1000µs at 2 streams) — e.g. assemble directly into mapped
   staging (no D2H, no snapshot), or kept-line host path with fewer passes.

### Probes kept in tree (TEMPORARY, env-gated — remove after tuning)

`VSFEEL_NNEDI3_BENCH/TSTAMP/UPTO/COUNT/SKIPIL/SPLITIL(dead after assemble —
fused copy has no split path)/NOXFER/UPGTT(dead after GPU pad — upload is
always GTT now)`, per-resource query pool. `tmp/` scratch: `probe2.vpy`,
`q2_vs.vpy`, `p0vs.vpy`, `a1.vpy`, `synth_a.vpy`, `nondet*.py`, `asm_*.txt`
(stale: pre-pad ISA), `ns_tmp.vpy`, `ss.vpy`.

### Scalar-GEMV experiment (STASHED, not in tree — correctness bug open)

Per-pixel scalar `accS[8]/accE[8]` + broadcast wae5 (`nn` loop,
`subgroupBroadcast` per neuron; needed `GL_KHR_shader_subgroup_ballot` on
the predict entry). Compiles, runs, but:
- Speed: predict ISA **47357 instructions** (was 1674 — the `nn`-loop
  `subgroupBroadcast` + NNS-iteration exploded codegen; ACO unrolled the
  NNS=256 loop with per-iteration broadcasts). Slower, not faster. REJECT
  this shape even if corrected.
- Correctness: deterministic per-run but wrong on PPL>1 networks
  (nns=2/3/4 → thousands of px, odd rows; PPL==1 exact). Root cause OPEN:
  the per-pixel `accS[px]` accumulation looks right vs the reference
  (each lane sums its own neuron slice over all k, then...). The missing
  piece: with PPL>1, lane L's `accS[px]` holds only slice t=L/32 partials,
  and the wae5 `nn` loop reads `accS[px]` on the OWNER lane for neuron nn —
  but owner lane `nn&31` holds slice `t = nn/32`, NOT the full sum. The
  `subgroupAdd(accS[px])` pre-pass was tried (all lanes full sums) but then
  EVERY lane's broadcast carries the full sum for every nn — double counts
  PPL times... no wait, that gives the right value on every lane (full sum
  + bias), which IS correct for wae5 (each neuron needs full k-sum). That
  version failed on ALL nns though (even PPL==1) → suggests the
  subgroupAdd placement broke the `[[unroll]]` macro scoping (adds ran
  outside the px-guard?) or a stale build was measured. Unresolved —
  re-derive from the committed vec4 form, don't patch the stash blindly.
- Stash: `git stash list` → "wip scalar gemv" (comp only; cpp PREDIRECT
  probe also stashed? No — PREDIRECT is in tree, env-gated off by default).

### YUV multi-plane corruption (FIXED 2026-09-05, 46/46 green)

`test_nnedi3_yuv_matches_reference` failed on chroma whenever ≥2 planes ran
in one sequence. Root cause: shader/cpp mismatch — cpp wrote per-plane
indirect structs (push 5 words + `d_base`, per-plane fills at
`ind_offset`), but the shader had been reverted to HEAD form (bare
`indCount`/`indGroupsX` globals = plane 0's struct for every plane).
Fix: re-applied `d_base` shader side (`indBuf[]`, `d_base` push word,
per-plane atomic/count/predict reads). Lesson: shader and host push
layouts must change atomically; the 4-word→5-word mismatch compiled AND
passed GRAY (single plane → offset 0, 5th word ignored) so only
multi-plane tests caught it. Any future push change needs a YUV run, not
just GRAY.

### Pack writer: NT stores win end-to-end (trust bench, not microbench)

C microbench (`tmp/pack_c.c`, heap→heap 540×3840B rows): memcpy 41µs vs
NT 65µs vs contiguous 32µs. Real upload mapping (HOST_CACHED GTT):
NT `copy_stream_out` + sfence ≈ 300µs pack at 2 streams vs memcpy ≈ 440µs.
Opposite! The heap microbench lacks the DMA snoop/writeback behavior of
the real path. Rule: pack-writer decisions come from `bench.py` + `t_pack`.
Memory types (7900XTX): type 2 = GTT uncached, 5/6 = GTT cached
(+HOST_CACHED, heap 0 sysRAM — our upload), 3/4 = VRAM-mapped ReBAR
(DEVICE_LOCAL+visible+coherent — the old zero-copy upload). GPU pad
removed the need for VRAM-mapped CPU writes.

### h2dpad decomposition (5-point timestamps: h2dpad/pre/pred/asm)

1-stream real: h2dpad ~95 + pre ~45 + pred ~185 + asm ~22 ≈ 350µs GPU.
Split probes: NOH2D (pad reads stale field) → pad = 12µs; NOHPAD (H2D
then prescreen reads UNPADDED field — garbage math) hangs the GPU (context
lost; expected, timing-only path). dh scaling: 2 MB field → 95µs,
4 MB field (dh=1) → 172µs. So the H2D copy itself ≈ 80µs for 2 MB
(~25 GB/s, plausible PCIe) and pad ≈ 12–15µs. Reference H2D ~129µs for
the same 2 MB — we are FASTER there. The kernels sum the same (~350 vs
~380); wall differs (900 vs 430) → host-side + submit, not GPU.

## Validation hardening (cross-cutting pass)

The upload flush (`up_total`) and download invalidate (`download_total`) now go
through the shared `flush_range`/`invalidate_range` helpers, which round the
range to `minNonCoherentAtomSize`/`VK_WHOLE_SIZE` as Vulkan requires (previously
offset 0 + an arbitrary size, invalid on a non-coherent memory type). Creation
also preflights the recorded `apiVersion` and reports
`"NNEDI3 requires Vulkan 1.3 (device reports X.Y)"` instead of an opaque
pipeline-creation failure. No behaviour change on this device; full suite green.

## Creation error path

The per-stream `Nnedi3Resource` is created into the pool via
`FramePool::emplace()`, so an error return inside the creation loop is torn down
by `~Nnedi3Data` (buffers, device memory, mapped windows, command buffers, query
pool, fence) instead of leaking it. Correctness-only; `test_nnedi3.py` passes.
