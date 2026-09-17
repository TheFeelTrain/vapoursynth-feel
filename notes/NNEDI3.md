# NNEDI3 — notes

*vsfeel's Vulkan port of nnedi3vk (refs `VapourSynth-nnedi3vk` / `vapoursynth-zipcu` / CPU `znedi3`).*

Status: **shipped**, `num_streams = 4` (`1..32`), ~18 MB VRAM/stream. Scoreboard
is median-of-5 same-session pairs, jpbd 1080p GRAY16, bench defaults
(`field=3 dh=0 nsize=0 nns=4 qual=2 etype=0 pscrn=4`); nnedi3vk in the ref column:

| ns | vsfeel | ref | ratio |
|---|---|---|---|
| 1 | 1330 | 1550 | 0.86 |
| 2 | 2320 | 2360 | 0.985 |
| 4 (shipped) | 2650 | 2520 | **1.05** |
| 6 | 2765 | 2560 | 1.08 |
| 8 | 2814 | 2584 | 1.09 |

- Bench `default_streams = 4` too, so out-of-box is ~2640 vs ref default-2s ~2360.
- Remaining gap is all 1-stream: kernels at parity (pre 55 / pred 195-200 / copy
  89 µs vs ref 34 / 179 / 89), our host bill ~600 µs serial.
- Agreement: nnedi3vk vs vszipcu **BIT-EXACT** (maxdiff 0, frames 0/5/11, all
  planes, noise_24f → YUV420P8 field=1); vs CPU znedi3 maxdiff 5, ~1% px (CPU
  float ordering) — **nnedi3vk is ground truth**.
- Accuracy, 25 tests: 16-bit within **1 LSB** (isolated 1-code flips from
  serial-vs-butterfly reduction order), GRAYS within **~3e-8** (bound 1e-6). Sweeps
  field 0/1/2/3, dh, planes subsets, nsize 0..6, nns 0..4, qual 1/2, etype 0/1,
  pscrn 0..4 across GRAY16/YUV420P16/GRAYS, plus determinism, 1-vs-4 streams,
  parallel load and props.
- `PAD`/`ASSEMBLE`/`COUNT` compile but are **never dispatched**; `pad_buf` +
  `asm_buf` stay allocated (~5 MB/stream at 1080p) — cleanup candidate. Weights
  `src/nnedi3_weights.bin`, 13,574,928 B, md5 `5c97e25c4a7277d06d3e3851373f1065`.

## Algorithm (both references)

Per plane (output W×H, interp rows = H/2): **field extract** (kept rows per
`parity`, all in `dh`) → **pad** to `(W+48)×(rows+6)`,
`f = clamp(i-(MARGIN_V-fp))`, `c = clamp(x-MARGIN_H)`, `fp = 1-parity`,
`MARGIN_H=24`, `MARGIN_V=3` → **prescreen** → **predict** → **interleave**.

- **Prescreen** (skipped at pscrn=0): a small NN picks cubic vs predictor. pscrn=1:
  12×4 window, 3 layers (4/4/4), 1 px/thread; pscrn≥2: 16×4, 2 layers (4/4),
  4 px/thread. Accepted → cubic `(-3,19,19,-3)/32` on rows r+1..r+4; rejected
  indices compacted into a list by atomics. Strict non-FMA (`precise`) — the
  verdict must match bit-for-bit or the pixel sets diverge.
- **Predict**: per listed pixel an XDIM×YDIM window (nsize 8×6…48×6, 8×4…32×4),
  mean/variance normalize, 2-layer NN (softmax + Elliott, nns 16…256), wae5 blend,
  `qual` passes averaged. Explicit FMA, subgroup butterfly reductions,
  `exp(clamp(s,-80,80))`.
- **Interleave**: `_FieldBased=0`, `_Field` deleted, duration halved; `field>1`
  doubles the rate (parity flips on odd outputs; `_FieldBased` drives parity when
  present). Weight blob (f32 LE):
  `ps_old(kernel_l0[4][48], bias_l0[4], kernel_l1[4][4], bias_l1[4],
  kernel_l2[4][8], bias_l2[4])`; 3× `ps_new(l0[256] transposed, bias_l0[4],
  l1[16], bias_l1[4])`; then etype 0..1 × nns-sel 0..4 × nsize 0..6: q1+q2 ×
  `(softmax[nns·fs], elliott[nns·fs], softmax_bias[nns], elliott_bias[nns])`,
  unselected combos skipped as `4·nns·fs+4·nns` floats.
- Host prep: prescreener ÷ pixel_half (0.5 f32, peak/2 int), model mean-subtract;
  upload per qual pass `pdW[(q·fs+k)·N+p]` = (sm,el), `pdB[q·2N+p]` = (smB, elB).

## Implementation (what ships)

`src/nnedi3.comp` — five entries, BITS=16/32, plus `predict_n4` (`-DPXP=4`) and
`predict_n4s` (`-DPXP=4 -DSHSTRIDE=64u`):

- `ENTRY_PRESCREEN` — 128 threads, one per pixel group (P=1 at pscrn=1, else 4);
  cubic taps + verdict inline, cubic store or list compaction (one `atomicAdd` per
  workgroup). Clamps the field for its own window reads and **maintains the indirect
  width itself** (`atomicMax(groupsX)` off the `atomicAdd` return) — no count
  dispatch. Grid `ceil(rows*ceil(width/P)/128)`.
- `ENTRY_PREDICT` — 128 threads = 4 subgroups, one per PXP pixels; PXP = 8 when
  `ceil(nns/32) <= 2 && fs <= 128` else 4; shared tile `shTile[4*SHSTRIDE]` (288,
  or 256 for `n4s`). Subgroup-add window stats, GEMV from the shared tile, wae5
  blend. Indirect off the prescreen count for pscrn>0, direct grid for pscrn=0.
- `ENTRY_PAD`/`ENTRY_COUNT`/`ENTRY_ASSEMBLE` — compiled and pipeline-created but
  **never dispatched**.

`src/nnedi3.cpp`: `FramePool` of `Nnedi3Resource` (ODR-unique); host-mapped ReBAR
upload staging (`DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`), device-local
dst/list/indirect buffers, persistent-mapped weights; descriptors 0-8 per stream;
5 push words `{list, up, dst, parity, d_base}`.

- **Two pre-recorded CBs per resource** (one per parity, at create). Per frame:
  CPU pack → submit → host scatter. CB per plane: indirect-struct reset (list mode
  only) → prescreen → barrier → indirect predict → barrier → inline
  `vkCmdCopyBuffer` of the packed interp dst → host barrier: no H2D, no GPU pad, no
  assemble, no transfer queue.
- Pack: tight field rows via `copy_stream_out` (NT) + `_mm_sfence()`, reading the
  kept rows out of `dst` for non-dh (copied from `src` pre-acquire, cache-hot);
  kept lines are copied **before** `pool.take()` to overlap other frames' GPU work.
- Args: field/dh/planes/nsize/nns/qual/etype/pscrn/device_id/num_streams; 16-bit
  int + 32-bit float only (8-bit/f16 rejected); `field:int;` is **required**.

## Historical

### 2026-09-04 — MVP: fused build, bit-exact with vszipcu, ~1130 fps vs ~2095 (~0.54x), 2 streams

### 2026-09-05 — prescreen list + cooperative predict split

- Two bugs fixed in the new split. Predict's GRIDPX loop tail was `#if 0`-disabled
  (only 32k px computed), and the indirect count word (offset 12) was never reset,
  so it accumulated across frames on the reused resource (frame 23 = +5173) and
  stale list entries let predict overwrite cubic-accepted pixels. Third
  `vkCmdFillBuffer`, verified by COUNT readbacks + order-swap probes. Then 46/46.
- **Fused kernel tried, REVERTED.** `ENTRY_FUSED` (verdict inline + serial
  chunked-neuron predict, no list/subgroups) passed 46/46 but timestamps showed
  **pred 464 µs vs split 178 µs** — verdict on the full grid, FMA-starved ISA
  (222 fmac vs 339 mul + 484 add), wall 824 vs ~1960 fps; a chunk-32→8 neuron tile
  would not have fixed the verdict-full-grid cost.
- Session ended RED (28 passed / 18 failed; pscrn=2/3/4 only, interp rows only,
  nondeterministic across fresh instances); later fixed, shared-L0 was the suspect.

### 2026-09-05 continued — transfer queue → GPU pad → GPU assemble

Each verified 46/46; **all were later deleted**. **Count folded into prescreen**
(`atomicMax` off `atomicAdd`): +~90 fps at 2s. **Zero-copy ReBAR upload** (H2D +
`field_buf` deleted): h2dpad 95 → 17 µs, **+~300 fps at 2s** (1775 → 2078
official, 2210 best-of-3). **Kept lines pre-acquire**: +~100 fps at 4s (2029 →
2200), and a cleaner pre-vs-post-fence re-test was neutral (2006 vs 2025) — the
reference's win is its rb-slot pool releasing the stream early. **Transfer-queue
split submit**: NOXFER A/B showed copy-overlap delta ≈ 0, so D2H was never the
bottleneck. **GPU pad kernel**: pack NT ~300 µs vs memcpy ~440 µs at 2s (synth
proof: pred 7 µs empty vs 190 µs real). **GPU assemble** + **decoupled host
overlap** (snapshot + early `give_back`): the missing predict→assemble barrier
failed 21 tests with 927-970 wrong interp pixels (odd rows, ~half = source, i.e.
stale dst); **SKIPIL fix** had the timing path skip copies without returning the
resource, draining the pool.

### 2026-09-06 — the push past parity

Each step best-of-3/median-of-5 same-session, real-world defaults; start 2s 0.79x,
4s 0.83x. Delete GPU assemble, D2H packed interp only (it shipped 4 MB of which the
host used 2 MB): +8% (2s → 0.93); pack reads kept rows from `dst` not `src`
(4s → 0.96-0.99). `predict_n4s`: the fixed `shTile[4*288]` (18 KB, LDS 15360 B)
throttled small-window occupancy, so FS≤64 now uses 256 vec4 (4 KB, LDS 4096 B);
4s → 0.99. **Transfer queue deleted, always inline** (submit + timeline + sync cost
more than 2 MB of overlap): 1s 0.80 → 0.87, 2s 0.96, 4s 1.05 (first lead). **Pad
fused into window reads** (`loadPad` clamps the tight field, `fp = 1-parity`):
deletes the pad dispatch + barrier (+20 µs GPU), clamp ALU 3-7 µs.
`REQUIRE_FULL_SUBGROUPS` on prescreen/predict; redundant host barrier deleted
(`vkQueueSubmit` orders the pack); pre-recorded parity CBs; UPTO deleted; timestamp
slots 5 → 4 (0=top 1=pre 2=pred 3=copy).

### Measured dead ends (mechanism + why it lost)

- **Predict over-launch REFUTED.** Full-grid direct with early exit: pred 268 vs
  223 µs GPU, 1892 vs 2055 fps at 2s — exiting subgroups still pay window-gather +
  occupancy before the `firstPix` check.
- **PXP=16 restructure REVERTED.** Per-pixel shared stride 32
  (`warp*288 + 15*32 + 31 > 1152`) overflows `shTile[4*288]` for warp≥1 → OOB
  shared writes → fresh-instance nondeterminism (maxdiff ~3000, 26 failed); rule:
  re-derive the shared footprint (≤1152 vec4) for any PXP/stride change.
- **Pack writer: NT wins — judge on `bench.py`, not a microbench.** An isolated heap
  microbench said memcpy 41 vs NT 65 µs; both real mappings said the opposite (GTT:
  NT+sfence ~300 vs memcpy ~440 at 2s; ReBAR: 1s NT 1075 vs memcpy 1027).
- **`copy_stream_read` faults on an unaligned SOURCE** (`_mm256_stream_load_si256`
  is an aligned load) → SIGSEGV; never assume source alignment. Not used here.
- **UPTO + TSTAMP together HANG**: a truncated CB never writes queries 2..4, so the
  `WAIT_BIT` readback blocks forever; run stage truncation with BENCH only.
- **Every producer→consumer dispatch pair needs an explicit barrier** — back-to-back
  dispatches order nothing. Prescreen→predict needs SHADER_WRITE → SHADER_READ *and*
  INDIRECT_COMMAND_READ; predict→copy SHADER_WRITE → TRANSFER_READ.
- **YUV multi-plane: shader and host push layouts must change atomically.** Chroma
  corrupted with ≥2 planes in one sequence — the host wrote per-plane indirect
  structs (5th push word `d_base`) while the shader read plane 0's globals. The
  4→5-word mismatch compiled and passed GRAY (single plane → offset 0), so only a
  YUV run catches it.
- **Scalar-GEMV stash: rejected.** Per-pixel accumulation + `subgroupBroadcast` wae5
  exploded predict ISA to **47357** (from 1674) and was wrong for PPL>1, root cause
  never found; re-derive from the committed vec4 form, never patch the stash.
- **7900XTX has no dedicated DMA family** (GFX 1q / COMPUTE 4q / video-dec/encode):
  the "transfer queue" was a second same-family compute queue, worth +75 fps
  (1755 → 1830) at 2s but net negative with submit/timeline cost. **pscrn=0 is at
  parity** (ours 177 vs ref 191 fps 1-stream, pred 4742 µs both over the full
  1M-px grid), so the math is fine; the sparse path (prescreen + list + indirect) is
  the gap. **D2H timestamps were bogus** while the transfer queue existed —
  copy(ts) ~20-22 µs for 4 MB, impossible over PCIe.

### Reference orchestration (read, not mirrored)

vszipcu HIP `process()` is fully serial per frame (pack before acquire, interleave
after release) and over-launches `ceilDiv(w*rows,16)` blocks with a `firstPix` early
exit — no indirect/count/atomicMax, no transfer queue, 430 µs. nnedi3vk uses push
descriptors, pools readback slots separately (`numRbSlots = numStreams+2`), sizes
`subgroupsPerWG = min(4, maxWG/sgSize)`, and uploads with `PREFER_DEVICE` ReBAR. Our
kernels match theirs in total (1-stream jpbd 1080p: ref H2D 129 / pad 32 / pre 53 /
pred 81-122 / D2H 87 ≈ 380 µs vs ours ≈ 350 µs), so the gap was never kernel math;
H2D was ours to lose (~80 µs for 2 MB vs their 129).

## Open work

- **1-stream host bill** (~600 µs: pack ~170 + kept ~335 + interp ~87 + submit ~18).
  The kept 2 MB strided memcpy runs at ~6 GB/s effective and is hardware-bound;
  further wins need fewer host bytes (`VK_EXT_external_memory_host`, not attempted).
- **Prescreen is ~2.8x the reference** (140 vs 50 µs when measured): compare ISA
  against their pattern (scalar `float v[EPL]`, warp shuffles, no shared), check
  VGPR/occupancy and the `precise` chains. **Delete the dead pad/assemble/count
  pipelines and their VRAM** (~5 MB/stream at 1080p) — no test can regress, the
  paths are unreachable.
- **MVP limit**: `dh` + a planes subset zeroes interp lines on skipped planes (the
  reference leaves them uninitialized).

## Debug env vars (all in `src/nnedi3.cpp`)

| env | what it does |
|---|---|
| `VSFEEL_NNEDI3_TSTAMP` | per-frame GPU timestamps: 0=top 1=pre 2=pred 3=copy |
| `VSFEEL_NNEDI3_BENCH` | per-frame host phase averages printed at teardown |
| `VSFEEL_NNEDI3_COUNT` | read back the prescreen pixel count |
| `VSFEEL_NNEDI3_SKIPIL` | skip the download copies (timing only, garbage out) |
| `VSFEEL_NNEDI3_QUEUES=N` | override the queue cap (default 2) |

`VSFEEL_NNEDI3_PREDIRECT` survives only as the measured-numbers comment on the
over-launch refutation; `UPTO`, `SPLITIL`, `NOXFER`, `UPGTT` no longer exist.

## Cross-cutting hardening

- Flush/invalidate use the shared `mapped_range` helpers
  (`minNonCoherentAtomSize`/`VK_WHOLE_SIZE` rounding); a creation error is torn down
  by `~Nnedi3Data` via `FramePool::emplace()`; creation preflights `apiVersion`
  instead of failing opaquely at pipeline creation.
- `field` is registered required: as `:opt;` with a null error pointer it took
  `VS_FATAL_ERROR` → `std::terminate` (SIGABRT 134) that `try/except` could not
  catch. Test `tests/test_nnedi3.py::test_nnedi3_requires_field`.
- Prescreen grid under-coverage: the host used `ceil(width*rows/(P*128))`, but the
  shader groups per row, so the exact need is `ceil(rows*ceil(width/P)/128)`; when P
  did not divide width the last row's tail was never dispatched and **uninitialized
  VRAM reached the output** (YUV420P16 1924x1080: 182 garbage chroma px/frame,
  0..65527). Matches `nnedi3vk.cpp:1121-1123`; 630/638 is bit-exact in
  `tests/test_geometry.py` (640/626/610 also 0 codes). Unknown length: `field > 1`
  doubled the `-1` sentinel to `-2`, now guarded with `numFrames > 0` (same fix as
  EEDI3); `field=2/3` still doubles 24 → 48.
