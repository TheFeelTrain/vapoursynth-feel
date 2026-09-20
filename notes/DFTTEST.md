# DFTTest — notes

Status: **shipped.** Verified against `src/dfttest.{cpp,comp}` and
`CMakeLists.txt`. Target GPU: RX 7900 XTX (RDNA3, gfx1100), Mesa 26.2 RADV.

- **Slot-direct frame cache** — each source frame is reflect-pad'd once into a
  shared device-local slot buffer; fused reads the slots **in place** via
  `slot_base[7]` push constants (`-1` = direct-padded fallback slice). The D2D
  copies and their submit were deleted.
- **Timeline semaphores per slot** — one timeline sem per slot, a fresh signal
  value per generation, non-destructive waits; any number of readers can wait.
- **ReBAR host-direct upload** (`up_buf`, memoryType 3/4) with a separate
  `pad_set`; fused's binding-1 writes stay on staging.
- **Reclaim via `frame_gen`, no host waits** (drain-on-fence and mutex+cv
  waits both deadlock).
- `effective_streams = max(num_streams, 2)`; `slot_count = max(eff+3, tw)`.
- Fused is pre-recorded per radius in a **branchless `FUSED_DIRECT`** variant
  plus a mixed variant for the fallback slices; host picks per frame.
- Perf, 500f cached real-clip medians: **ns=1 1215 vs 864 (+41%)**, **ns=4
  1325 vs 1297 (+2%)**. README rows: u16 1445 vs 909 (1.59x), f32 1102 vs 628
  (1.75x).
- Perf, 1000f jpbd GRAY16 same-session pairs (ns=1, 3 order-reversed reps):
  **1506 vs 1345 fps** for the tight fused transpose, +12% (vszipcl 871).
- VRAM: 310 MiB at ns=1, 600 MiB at ns=4 (was 1182 MiB/instance before the
  in-flight floor moved 8 → 2).

## Implementation

### Cached-submit shape

- Slot buffer: `slot_count = max(effective_streams + 3, tw)` slots **per plane**
  (binding 4), `slot_plane_bytes = pw*ph*bytes`.
- Per-frame submits: pad CBs at claim time under `slot_lock`; then the
  pre-recorded `[fused (+col2im)]` CB, fence = `resource.fence`. `cmd2` (the old
  copy CB) is unused, kept for layout stability.
- `cmd_pad` is a per-(plane,t) CB ring (`tw*num_planes` per resource) — a CB
  must never be re-recorded while its previous submission still executes.
- Pipeline: `ENTRY_PAD_SLOT`, `ENTRY_PAD_DIRECT`, `ENTRY_FUSED`, `ENTRY_COL2IM`.
  The old `ENTRY_PAD` (t-loop pad into binding 3) is **gone from the shader
  source** and from CMake.
- CMake emits `pad_slot`/`pad_direct`/`col2im` per bit depth plus
  `fused_r{radius}` and `fused_direct_r{radius}` (radii 1..4, 2 bits).

### Protocol (all under `d->slot_lock`)

- `SlotState { gen; committed[{res_id, frame_gen}]; sem; signal; }` indexed
  `plane*K + s`; `ResMeta` holds a fence and an atomic `frame_gen` advanced on
  resource re-acquire.
- Per (plane,t): `idx = clamp(n-radius+t, 0, numFrames-1)`,
  `which = n - max(0, idx-radius)`. Clamped boundaries map two slices to the
  same (slot, which), so **waits are deduped**.
- **Peek**: `st.gen == idx` → reader; commit and wait on `st.sem` at `which`.
- Else **padder**: upload `src[t]` to own staging + per-op flush, re-take the
  lock, re-check (someone may have claimed in between), else claim:
  - **Reclaim rule:** the slot is free iff `gen == -1` or every committed
    reader's `frame_gen` has advanced past its commit gen (resource reuse
    happens-after that frame's fence wait) — no host waits.
  - `st.gen = idx`, `committed = {self}`, `signal += 1`, then **submit the pad
    immediately** (`submit_timeline`, signals `st.sem`, fence=NULL).
- **Fallback** (all K slots busy — out-of-order completion at ns>1, or radius 3
  where a window spans 7 sources against K=5): `ENTRY_PAD_DIRECT` pads into the
  per-resource padded buffer, read same-queue-ordered. ns=1 + radius 1 never
  hits it (4 distinct sources < 5 slots).
- **Pad and fused must share a queue.** A padder adds no `st.sem` wait for its
  own pad: a fresh slot is visible only through same-queue submission order plus
  the fused CB's head barrier. Moving the pads off that queue needs the wait
  added explicitly, and the direct-pad fallback has no semaphore at all, so it
  cannot move off the compute queue without new sync.

### Why it is deadlock/stall-free (do not "simplify" away)

- **No queue stall:** RADV will not run a later submit while an earlier one on
  the same queue waits on an un-signaled semaphore. Submitting the pad inside
  the claim's `slot_lock` section means every reader commits after the claim,
  so its wait is queued behind the signal.
- **No host drains:** two concurrent padders would each wait on the other's
  fence. `frame_gen` reclaim needs none.
- **Timeline, not binary:** a fresh signal value per generation with
  non-destructive waits, so chained instances cannot starve a consumer.
- Lock nesting is exactly `slot_lock → queue_lock` (`submit_pad_op`).
- **This box cannot do DEVICE-side waits**: `vkCmdWaitSemaphores`/
  `vkCmdSignalSemaphore` are absent from the stripped loader. SUBMIT-level
  timeline semaphores work (device feature + `VkTimelineSemaphoreSubmitInfo`).

### Key facts

- **Spatial buffer = `num_blocks*256` floats, center slice only.** Fused writes
  `spatial_base + block_id*256`; col2im reads
  `spatial_base + ((i*hn+j)*16+off_y)*16+off_x`. A mismatched stride zeroed the
  bottom half — this is what fixed the 3.4 GB → 1.2 GB thrash (236 → 801 fps).
- 16-bit IO is plain `uint16_t[]` storage; glslc `-O` bug on `OpUConvert` to
  16-bit → `uint16_t(uint(x) & 0xFFFFu)`.
- `create_pipeline()`: spec constants `filter_type` and `zmean` built from an
  explicit `(id, value)` list; wave32 via `VK_EXT_subgroup_size_control` pNext.
- `num_queues = resolve_queue_cap(effective_streams, queue_count,
  "VSFEEL_DFFTEST_QUEUES", 2)` — resolved from **effective_streams**, not the
  user's `num_streams`, so the knob is reachable at the shipped default. Two
  queues measured **+7.0%** over one (3×1000 same-session pairs: 1362.7/1377.1/
  1367.4 vs 1298.5/1267.3/1278.3).
- Fused: `SUB_BLOCKS=8` (128-thread WGs), `td[TD_SZ]` register-resident,
  `subgroupBarrier()`, one 288-float (1 152 B) transpose window per sub-block →
  **10 240 B LDS** per workgroup. Measured on the shipped GRAY16 config: radius
  0/1 VGPR 120 and **12 subgroups/SIMD**; radius 2 VGPR 192 with 20 KB scratch,
  8; radius 3 VGPR 256 with 28 KB scratch, 5. Re-measure rather than quote any
  of it.
- Benchmark defaults: `ftype=0`, `sigma=8`, `sosize=12`, `tbsize=3`, `swin=0`,
  `twin=7`, `sbeta=2.5`, `tbeta=2.5`, `zmean=1`, `f0beta=1.0`.
- Bounds: every offset pushed to the shader is `int32`, so creation bounds each
  region separately (`tw*pad_elems`, `upload_bytes`, `padded_bytes`,
  `slot_plane_bytes`, `nblk*256` per plane, the `*_sum` aggregates, and
  `upload_total + download_sum` which `dst_base` is built from). 8K 16-bit +
  `num_streams=32` used to wrap `slot_base[]` negative.

### Pitfalls (each cost a session)

- `load_upload()` adds `pc.src_base`; C++ must set it to `cfg.upload_offset`
  ONLY (the shader adds `pad_t0*up_slice`). Double-adding made plane t read
  t+t planes ahead into the download region (small correlated diffs, invisible
  at tbsize=1).
- `VSFEEL_DFFTEST_TRIVIAL=1` still submits empty pad CBs so reader waits don't
  stall.
- A non-coherent ReBAR upload type is rejected (falls back to staging) instead
  of being accepted and never flushed.
- The host timing probe used to count `t2-t1` (pool pop) as "upload" and drop
  the real `t2→t3`; now `acquire = t0..t2`, `upload = t2..t3`, and the stages
  sum to `avg_total`.

## Historical

### Speed run (ReBAR + slot-direct + IM2COL SR)

Warm in-command-buffer timestamps (`VSFEEL_DFFTEST_QBENCH=<frame>`, q0=head,
q1=post-barrier, q2=fused end, q3=col2im end) vs vszipcl rocprofv3 averages
(200f BlankClip GRAY16 1080p, default args):

| stage | vszipcl | vsfeel before | vsfeel after |
|---|---|---|---|
| pad | 24 µs | ~150 µs host upload + GTT read | host memcpy → ReBAR VRAM |
| D2D copies | 3×11.5 µs | 22 µs | **gone** (fused reads slots) |
| fused | 384 µs | 473 µs | **448 µs** (IM2COL SR) |
| col2im | 368 µs | 266 µs | 226–260 µs |

- **ReBAR upload**: CPU writes VRAM directly with plain memcpy — NT stores
  measured slower here (31 vs 64 GB/s) — and pad reads VRAM not GTT. Synth
  989→1150, real-clip 1014→1172 median (+15%).
- **Slot-direct fused**: `slot[t]` via `slot_base[7]`, D2D copies + copy submit
  deleted, fused submit carries the timeline waits at COMPUTE stage. With
  IM2COL, real-clip 1172→1215, ns=4 1100→1325.
- **IM2COL strength reduction** (hoist `row0`/`wrow0`, per-j `+pw`/`+16`):
  fused 473→448 µs. First attempt regressed to 904 µs — the `sb>=0` branch
  if-converted into doubled u16 loads (48→96), de-dualized ALU and 143 extra
  `s_waitcnt`. Fix: the branchless `FUSED_DIRECT` variant + a mixed variant,
  host picks per frame; +8 SPIR-V blobs (58→66 args).
- Dead code removed: `record_copy_cb`, the NOWAIT probe, the old GPU_BENCH copy
  stage (now pad / pad+fused / pad+fused+col2im).

### VRAM: in-flight floor 8 → 2

`effective_streams = max(num_streams, 8)` allocated 8 full resources regardless
of `num_streams`: at 1080p GRAY16 one resource was ~142 MiB (tw=3 padded slices
12.4 MiB + `num_blocks*256` float spatial 129.7 MiB) plus slots → 1182
MiB/instance (the reported 1.16 GiB; ×5 chained ≈ 6 GB). Sweep (real +
synthetic, tbsize 1/3/7, 16/32-bit, ns=1/4, chains ×1/×3/×5): **S=2 keeps
≥98.5% of S=8 throughput everywhere** (real ns=1 994 vs 1000; ns=4 1096 vs
1101; chain ×5 207 vs 199 — S=2 sometimes faster, less contention). S=1 cannot
overlap host work with GPU (~640 vs ~1000). Fix: `max(num_streams, 2)`. New
footprint 310 MiB (ns=1), 600 MiB (ns=4), YUV420 460 MiB, GRAY32 350 MiB. ~85%
of the remainder is the algorithm-inherent spatial working set (one 16×16 float
tile per block per in-flight frame); shrinking needs tiling or fp16, both
risky — not pursued.

### Chained-instance deadlock (fixed by timeline semaphores)

Chaining ≥3 DFTTest instances hung with the GPU idle: the slot protocol
signalled **binary** semaphores, one signal per generation, and chaining makes
an upstream frame get processed ~N times (eval-time probes + the streaming pass
re-requesting in-flight frames), so N-1 copies waited on the one signal and
every consumer past the first waited on an already-consumed semaphore. RADV
stalled the queue, `vkQueueSubmit` blocked host-side holding `queue_lock`, and
the pad chain wedged permanently.

Fix: each slot owns ONE timeline semaphore with a per-generation signal
*value*; waits are non-destructive.
`VkPhysicalDeviceTimelineSemaphoreFeatures` is chained into
`VkDeviceCreateInfo`. Also fixed: the copy submit shared one
`VkPipelineStageFlags` across `waitSemaphoreCount > 1` (out-of-bounds read).

Verified: `tmp/dlt_progress.py` N=3 on a 34072-frame clip OLD = exit 124, zero
frames, FIXED ≈317 fps (N=4 ~286, N=5 ~203); tbsize=3 N=1..4 =
613/336/255/187; a blank chain N=3 GRAY16 1920×1080 LEN=20000 stalled at
9k–14k frames before and completes after; single-frame digests bit-exact OLD vs
FIXED for N=1/2 × tbsize 1/3.

### Correctness pass (slot rollback, UB, debug submits)

- **Failed pad submit wedged the queue forever**: the claim set `gen`,
  `committed` and bumped `signal` before `submit_pad_op`; a failed submit left
  the slot claimed with a generation it never signalled, so a later reader
  waited on a value that never arrives. Fix: roll back (`gen = -1`, clear
  `committed`) under `slot_lock`. Reproduced with `VSFEEL_DFTTEST_FAILPAD=N`:
  pre-fix frame 0 errored then frame 1 hung (>45 s); post-fix frames 1/2 pass.
- **`st.signal = ++st.signal` was UB** (unsequenced read/write,
  `-Wsequence-point`); now `st.signal += 1`.
- **Debug submits bypassed `submit_with_fence`/`queue_lock`** on `DUMP_PAD` and
  re-implemented it on `QBENCH`; both now use the shared helper. `slot_buf`
  dropped unused transfer flags; `padded_buf` swapped `TRANSFER_DST` for the
  `TRANSFER_SRC` the dump uses.
- **`DUMP_PAD` copied 46464 B into a 32768 B staging buffer**
  (`VUID-vkCmdCopyBuffer-size-00116`), clobbering the download region — on this
  box it **reset the GPU**. Fix: a third staging region reserved at creation
  (only when `DUMP_PAD` is set), the copy targets `upload_total +
  download_total`, an oversized copy is refused, and a whole-range invalidate
  covers a non-coherent staging type. `VSFEEL_DFTTEST_DUMP_PATH` is mandatory.
  Verified under `VK_LAYER_KHRONOS_validation`: 0 errors, dump 46464 B, output
  digest byte-identical with and without the dump.
- `f0beta` rejects NaN; the `slocation/ssx/ssy/sst` `mapGetFloatArray` error is
  checked before the result is dereferenced; flush/invalidate ranges go through
  the shared helpers; DFTTest reports `"requires Vulkan 1.3 (device reports
  X.Y)"` instead of an opaque pipeline error.

### Shared-plumbing refactor — ODR COMDAT hazard

dfttest/gaussblur/bilateral each defined their own `struct VK_Resource` at
namespace scope with different sizes while instantiating
`FramePool<VK_Resource>` from the shared header. Identical mangled template
names let the linker COMDAT-fold `give_back`/`take`/`push` into ONE built with
the wrong `sizeof` (80-byte stride while dfttest's was 152), so the vector's
`_M_finish` moved by the wrong amount and `items.back()` landed mid-item. Only
frames with 2+ concurrent processings hit it; single-frame digests passed; it
crashed `test_dfttest_deterministic` as a SIGSEGV with a corrupted pool item.
Fix: unique per-filter names — `DFTTestResource`, `GaussBlurResource`,
`BilateralResource` (`NLStream`, `Bm3dStream` never collided). Any filter-local
type used to instantiate a shared template must have a unique name.

### Wins (cumulative, ns=1 synthetic fps)

72 → 303 fused unroll (`td[]` register-resident) → 457 CPU/GPU pipelining
(`effective_streams` 8) → 470 AVX2 streaming upload → 472 uint16 storage +
`SUB_BLOCKS=8` (16 is worse, ~400) → 552 `filter_type` spec constant + dispatch
grid fix → 673 wave32 pNext → 801 center-slice spatial buffer (3.4 GB → 1.2 GB)
→ **1009 frame cache** (+22% vs vszipcl 829). Fused codegen (float-literal FFT
constants, ZMEAN spec constant, gf hoist): 3378 → 2613 instr, 690 → 468 µs
(vszipcl 441).

## 2026-09-20 — fused transpose window tightened, LDS 18 432 → 10 240 B (+12%)

The fused kernel's per-sub-block transpose scratch was the reference's
`2*16*17 = 544` floats. The four barrier-separated phases that share it need
only two row layouts — 17-float rows for the 16x16 real transposes, 18-float
rows for the 9x16 float ones — and both fit in one 288-float window (max
offset 287). Every fused variant's LDS drops to 10 240 B. Radius 0 keeps VGPR
120 and goes 6 → 12 subgroups/SIMD on the LDS alone; radius 1 drops VGPR
240 → 120 (6 → 12); radius 2 drops 240 → 192 (6 → 8); radius 3 is unchanged at
256/5, scratch-bound.

jpbd 1080p GRAY16, 1000 frames, 3 order-reversed same-session pairs, ns=1:
**1345 → 1506 fps**, every pair +6% … +13%; ns=4, two pairs: 1417/1410 →
1465/1615. fp32 is not a regression either (`--bits 32`, two pairs: 1058.7/
1058.8 → 1095.9/1099.4). Data movement only, so output stays bit-exact: 70/70
`test_dfttest.py` (tbsize 1/3/5/7 vs vszipcl), 800/800 full suite. No `-D`
variant was needed for the trim — see `notes/NLMEANS.md` for why a
specialization constant can size an array on this toolchain.

## 2026-09-20 — second-queue transfer split measured neutral (reverted)

Both `VSFEEL_DFFTEST_XFER` prototypes were bit-exact and then reverted, because
neither moved the graded median:

- **D2H split**: col2im wrote a device-local result buffer and a pre-recorded
  `vkCmdCopyBuffer` on queues 2/3 (compute on 0/1) moved it to staging, with the
  frame fence on the copy. 5 order-reversed pairs, jpbd 1080p GRAY16, 2000
  frames, median of 3: **1161.56 → 1161.55 fps** at ns=4.
- **Pad split**: pads on the second queue instead of the compute queue. Requires
  an explicit `st.sem` wait for the frame's *own* pad added to the fused — without
  it the fused could read an unwritten slot (intermittent, 1 of 6 trials in a
  repeat harness, output off by up to 1630 codes). With
  the wait: ns=4 **1321.97 → 1317.04**, ns=8 **1352.39 → 1355.44** (3 reps × 3
  × 1000-frame medians); ns=1 was too bimodal (1010/1235, ±19%) to call.

Mechanism: the frame is fused-kernel-bound (448 µs against col2im 226-260 µs and
a ~24 µs pad), and shader stores to GTT are posted, so both legs already overlap
the neighbouring stream's compute. A second queue adds a submit plus a timeline
wait and buys nothing; the ns=4 gap to vszipcl is fused codegen, not SDMA.

## Open work

- **Fused codegen residue (448 vs 384 µs)**: remaining IM2COL/window ALU,
  pointer-walk strength reduction (base + increment per j), ACO dual-issue
  packing (`v_dual_mov` 128 vs 18).
- **Fused occupancy: the LDS half is resolved, the scratch half is not.** The
  radius-0/1 variants were **LDS-bound**, not VGPR-bound: at VGPR 120 the old
  18 432 B window held 6 subgroups/SIMD, and halving the window to 10 240 B at
  the same VGPR took them to 12. Radius 2/3 still **spill 20/28 KB of scratch**
  at VGPR 192/256 and stay at 8/5, so *those* are the low tier that remains —
  the donor there is the `td[TD_SZ]` register block, not LDS (the old
  LDS-pad-probe plan is no longer the first step). Any VGPR-reduction variant
  needs a same-session A/B, not a theory: forcing VGPRs down regressed on
  NLMeans.
- Col2im (260 µs) is already 2× faster than both references — leave it.

## Tests

- `test_dfttest.py`: 30 tests locally (~2 s); a HANG is a bug, so always run it
  under `timeout 20`.
- `test_dfttest_vspipe_pipelined_no_hang` (ns=1/4, 32/16-bit): runs the real
  `benchmark.py --synthetic --frames 200` through vspipe with a 120 s timeout.
  vspipe's pipelined reader + prefetch activates frames 11+ ahead — the workload
  that stalls the queue under a batched pad submit. Pure-Python sequential
  `get_frame` does not reproduce it.
- `test_dfttest_vspipe_pipelined_chained_no_hang_16bit` (3 chained instances):
  synthetic GRAY16 1920×1080, **LEN=200 plus an eval-time probe** on each
  intermediate node. The probe is the reload trigger — without it the buggy
  build only stalled after 9k–37k frames (timing-dependent, flaky).
- Suite counts across the fixes: 30 → 35/35 (chained test) → 61/61 → 187
  (shared-plumbing migration) → 600 passed (queue-cap/emplace work).
- `test_dfttest_frame_request_order_matches_serial` drives the node in six
  request orders (repeat/reverse/far/random/thrash) on a fresh instance against
  the serial run: exact (0.0 diff) at tbsize=3, ns=4, and on 1–2-frame clips.
  Sequential `get_frame` never re-requests an in-flight frame, so this is the
  pattern the shared `conftest.assert_temporal_order_consistent` exists to add.

## Do not retry

- **Second-queue transfer split** (`VSFEEL_DFFTEST_XFER`, both the D2H copy and
  the pad variant): neutral at every graded config, see the 2026-09-20 round.
  The transfer legs already overlap neighbouring compute; the bottleneck is the
  fused kernel.
- **Host-side cache synchronization in any form**: mutex+cv ordered-submission
  waits starve the VS worker pool; drain-on-fence gives circular waits between
  concurrent padders. The cache must be sync-free: GPU ordering via submit order
  + submit-level semaphores, reclaim via `frame_gen`.
- **Batched pad submit after the ops loop** (the original bench hang); **fixed
  per-slot binary semaphores across generations** (stale-signal corruption);
  **any binary-semaphore slot signalling under chained instances** (permanent
  queue stall).
- LDS-slice fused restructure (2× slower, LDS-bound; reverted `3589644`).
- `SUB_BLOCKS=16`; DMA upload to a device raw_buf (453 fps); float-internal
  buffers + a pack kernel (buggy); `RADV_PERFTEST=cswave32` (driver-global, not
  shippable); `OpExecutionMode SubgroupSize` injection (ignored); glslc `-Os`/
  spirv-opt (no-op); `RADV_DEBUG=llvm` (not in release Mesa).
  (`RADV_DEBUG=shaderstats` **does** print for DFTTest on current Mesa — the
  older claim that it prints nothing is stale, and it is how the fused kernel's
  VGPR/LDS/occupancy numbers in Open work were measured.)
- **Ranked-intermediate depth cut** (BM3D's per-window-list win): no
  application — the chain is im2col → spatial DFT → temporal DFT → filter →
  IDFT, with no candidate list. The producer-vs-consumer check on the LDS
  transposes is clean: the first transpose stores rows 0..15 at columns 0..8 and
  the read consumes columns 0..8 of all 16 rows (lanes 0..8 store, all 16 read);
  the second writes and reads 16x16. No dead stores to delete.
- Diagnosing the cache corruption as an out-of-order race — it was the double
  temporal offset (`src_base`).

## Debug env vars

- `VSFEEL_DFFTEST_QBENCH=<frame>` — one-shot warm in-CB timestamps for that
  frame (q0 head, q1 post-barrier, q2 fused end, q3 col2im end).
- `VSFEEL_DFFTEST_GPU_BENCH=N` — one-shot cold stage timing on frame 0.
- `VSFEEL_DFFTEST_TRACE=1` — per-frame protocol trace (claim/read/reclaim/
  submit).
- `VSFEEL_DFFTEST_TIMING=1` — host phase averages (`acquire`/`upload`/`wait`).
- `VSFEEL_DFFTEST_FORCEPAD` — pad every op (bypass slot reuse).
- `VSFEEL_DFFTEST_UPDIRECT=0` — staging + H2D copy instead of ReBAR upload.
- `VSFEEL_DFFTEST_FUSEDDIRECT=0` — force the mixed fused variant.
- `VSFEEL_DFFTEST_DUMP_PAD=n` + `VSFEEL_DFFTEST_DUMP_PATH=file` — dump the
  padded buffer after frame n (path mandatory).
- `VSFEEL_DFFTEST_FAILPAD=N` — fault injection: fail the Nth slot pad submit.
- `VSFEEL_DFFTEST_STREAMS=N` — override `effective_streams`.
- `VSFEEL_DFFTEST_TRIVIAL=1` — skip kernels (still submits empty pad CBs).
- `VSFEEL_DFFTEST_QUEUES=N` — override the queue cap (default 2).
- `VSFEEL_DFFTEST_DBG=1`, `VSFEEL_DFFTEST_SGSIZE=N`,
  `VSFEEL_DFFTEST_SGSIZE_INVALID` — plumbing/diagnostics.
- `RADV_DEBUG=asm` — ACO ISA to stderr (works).
