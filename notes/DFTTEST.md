# DFTTest optimization notes

Goal: `vsfeel.DFTTest` ≥20% faster than vszipcl (the faster reference), all
tests passing, VRAM ≤ references. Target GPU: RX 7900 XTX (RDNA3, gfx1100),
Mesa 26.2 RADV. A ~21 GB local model is resident on the GPU (~1.6–4.3 GB
free) — keep the footprint small.

## Speed run 2026-09-08 (qbench + ReBAR + slot-direct + IM2COL SR)

Warm timestamp queries (`VSFEEL_DFTTEST_QBENCH=[frame]`, q0=head, q1=post-
barrier, q2=fused end, q3=col2im end; fused CB carries its own reset so no
cross-CB races; one-shot armed frame only) vs vszipcl rocprofv3 averages
(200f BlankClip GRAY16 1080p, default args):

| stage | vszipcl | vsfeel before | vsfeel after |
|-------|---------|---------------|--------------|
| pad (1 new src) | 24 µs | PCIe-bound (~150 µs host upload + GTT read) | host memcpy→ReBAR VRAM + VRAM read |
| D2D copies | 3×11.5 µs | 22 µs | **gone** (fused reads slots) |
| fused | 384 µs | 473 µs | **448 µs** (IM2COL addr strength-reduction) |
| col2im | 368 µs | 266 µs (already faster) | 226–260 µs |

1. **ReBAR upload** (`up_buf` per resource, DEVICE_LOCAL|HOST_VISIBLE|
   COHERENT = memoryType 3/4; CPU writes VRAM directly with plain memcpy —
   NT stores measured slower here, 31 vs 64 GB/s — pad reads VRAM not GTT;
   separate `pad_set` descriptor set because fused's binding-1 writes must
   stay on staging; `VSFEEL_DFTTEST_UPDIRECT=0` opts out): synth
   989→1150, real-clip 1014→1172 median (+15%).
2. **Slot-direct fused** (fused reads `slot[t]` via new `slot_base[7]` push
   constants, −1 = direct-padded fallback slice in `padded_buf`; D2D copies
   + copy submit deleted; fused submit carries the slot timeline waits at
   COMPUTE stage; head shader-write→shader-read barrier for same-queue
   visibility; `slot_count = max(eff+3, tw)` so radius-3 windows fit):
   combined with #3, real-clip 1172→1215, ns=4 1100→1325.
3. **IM2COL strength reduction** (hoist `row0`/`wrow0`, per-j `+pw`/`+16`):
   fused 473→448 µs. First attempt regressed to 904 µs: the `sb>=0` branch
   if-converted into doubled u16 loads (48→96) + de-dualized ALU + 143
   extra s_waitcnt (ISA-compared). Fix: `FUSED_DIRECT` variant (branchless
   `load_fused_direct`) + mixed variant; host picks per frame
   (`VSFEEL_DFTTEST_FUSEDDIRECT=0` forces mixed). +8 SPIR-V blobs
   (2 bits × 4 radii; CMake + gen_spirv_header 58→66 args).
4. Dead code removed: `record_copy_cb`, NOWAIT probe, old GPU_BENCH
   copy stage (now pad / pad+fused / pad+fused+col2im).

Result (500f cached real-clip medians): **ns=1: 1215 vs 864 (+41%)**,
**ns=4: 1325 vs 1297 (+2%)**. Suite 61/61. ns=4 gap is structural (notes
item: vszipcl overlaps H2D/D2H on SDMA full-duplex; ours serializes on the
compute CP — needs a dedicated transfer queue + 2× staging ping-pong).

## VRAM: in-flight floor 8 → 2 (2026-09-08)

Default `effective_streams = max(num_streams, 8)` allocated 8 full resources
regardless of `num_streams`: at 1080p GRAY16 one resource is ~142 MiB VRAM
(tw=3 padded slices 12.4 MiB + `num_blocks*256` float spatial 129.7 MiB;
staging is GTT, not VRAM) plus `(eff+3)` 4.1 MiB slots → 1182 MiB/instance
(the reported 1.16 GiB; ×5 chained = ~6 GB). Sweep (RX 7900XTX, cached
real-clip + synthetic BlankClip, tbsize 1/3/7, 16/32-bit, ns=1/4, chains
×1/×3/×5): S=2 keeps ≥98.5% of S=8 throughput everywhere (real-clip ns=1:
994 vs 1000 fps; ns=4: 1096 vs 1101; chain ×3: 337 vs 330; chain ×5: 207 vs
199 — S=2 is sometimes *faster*, less contention). S=1 cannot overlap host
work (upload/record/submit) with GPU work (~640 vs ~1000 fps), hence the
floor of 2, not 1. Fix: `max(num_streams, 2)` + comment in `DftCreate`;
`VSFEEL_DFTTEST_STREAMS` override kept. New footprint: 310 MiB (ns=1,
+18% vs vszipcl at 911 fps), 600 MiB (ns=4), YUV420 460 MiB, GRAY32
350 MiB. Bench ns=1 after: vsfeel 1072 vs vszipcl 911. Suite: 61 passed.
Remaining 85% of the footprint is the algorithm-inherent spatial working
set (one 16×16 float tile per block per in-flight frame); shrinking it
needs tiling or fp16, both risky — not pursued.

## TL;DR (2026-08-18; VRAM fix 2026-09-08 above)

Padded-source frame cache is **DONE for ns=1 and ns>1** (in the working tree,
uncommitted; 30/30 tests pass):

| plugin | ns=1 | ns=4 |
|--------|------|------|
| vszipcl | 829 | 1198 |
| vsfeel | **1009** (+22%) | 1073 (+30% vs pre-cache 826; still −10% vs vszipcl) |
| vszipcu | 500 | 1076 |

Steady-state per-frame GPU work with the cache: **pad 167 µs (1 new source) +
3 D2D copies 42 µs + fused 450 µs + col2im 260 µs ≈ 920 µs** (1210 µs without
the cache: 3 pads 500 µs). GPU-bound: the kernels saturate all 96 CUs, so
queue count adds nothing — only reduced per-frame GPU work does.

## Chained-instance deadlock (2026-08-26)

**User bug:** chaining DFTTest instances deadlocks at 3+ ("1 or 2 works but 3
or higher does not") — vspipe hangs with the GPU idle. Root cause: the slot
protocol signalled **binary** semaphores, one signal per generation. Chaining
N instances makes an upstream frame get processed ~N times (eval-time probes +
the streaming pass re-request frames still in flight), so N-1 copies wait on
the one signal; every consumer past the first waits on an already-consumed
semaphore. RADV does not execute a later submit while an earlier one on the
same queue waits on an un-signaled semaphore — the queue stalls,
`vkQueueSubmit` blocks host-side holding queue_lock, and the pad chain wedges
(padder holds slot_lock waiting for queue_lock; ticket semaphore starves):
permanent deadlock. (GDB: all workers on one instance blocked in
`ticket_semaphore::acquire` / slot_lock; padder thread at `submit_pad_op`'s
queue_lock; queue-lock holder inside `vkQueueSubmit` → drmSyncobjTimelineWait
→ ioctl.)

**Fix — timeline semaphores (Vulkan core 1.2):** each slot owns ONE timeline
semaphore with a per-generation signal *value*; the pad signals value V and
any number of copies wait on V. Timeline waits are non-destructive, so
repeated processings can never starve a second consumer. The per-generation
binary sem create/destroy loop is gone; signalling moves into `st.signal`
(`st.signal += 1` at claim). `VkPhysicalDeviceTimelineSemaphore-
Features` is chained into `VkDeviceCreateInfo` in `get_device()` (the app
already requests 1.4). Also fixed along the way: the D2D copy submit shared
one `VkPipelineStageFlags` across `waitSemaphoreCount > 1` — an
out-of-bounds read; now a per-wait vector.

**Verified (MANGOHUD=0):**
- Original repro (`tmp/dlt_progress.py`, N=3, 34072-frame jpbd.mkv): OLD =
  exit 124, ZERO frames; FIXED = ~317 fps steady. N=4 ~286, N=5 ~203.
- Temporal window (tbsize=3): N=1..4 = 613/336/255/187 fps, no hangs.
- Self-contained blank chain (N=3, GRAY16 BlankClip 1920×1080, LEN=20000,
  vspipe): OLD stalls/crashes around frame 9k–14k; FIXED completes all 20000
  frames steady at ~512 fps — codified as
  `test_dfttest_vspipe_pipelined_chained_no_hang` (8 MB cache, LEN=12000,
  ~23 s on the fixed build).
- Single-frame digests (frame 5, tests/noise_24f.mkv): FIXED == OLD
  bit-exact for N=1/2 × tbsize 1/3 (3651e55a…, 527375cc…, 4383e3ff…
  and 80f35632…).
- Suite: 35/35 (34 + the new chained test).

## Correctness pass 2026-09 (slot rollback, UB, secondary defects)

- **Failed pad submit wedged the queue forever.** The claim set `gen`,
  `committed` and bumped `signal` *before* `submit_pad_op`; if that submit
  failed, the slot claimed a generation it never signalled, a later reader
  committed to it, and its fused submit waited on a value that never arrives —
  RADV will not run a submit behind such a wait, so the queue (and host) hung.
  Fix: on failure, roll the slot back (`gen = -1`, clear `committed`) under
  `slot_lock` so the next frame re-pads it. Reproduced with the new fault
  injection `VSFEEL_DFTTEST_FAILPAD=N` (fails the Nth *slot* pad submit; unset =
  zero effect): pre-fix frame 0 errored then frame 1 hung (>45 s); post-fix
  frames 1/2 complete. Repro: `tmp/wo09_repro_failpad.py`.
- **`st.signal = ++st.signal` is UB** (unsequenced read/write of one scalar;
  `-Wsequence-point` at `dfttest.cpp:1407`). Now `st.signal += 1`. A compiler
  could have stored the old value, making readers wait on the previous
  generation's value (stale slot read).
- **Debug submits bypassed `submit_with_fence`/`queue_lock`** on `DUMP_PAD` and
  re-implemented it on `QBENCH`; both now go through the shared helper with
  checked `vkBegin/EndCommandBuffer`. `slot_buf` dropped its unused transfer
  flags; `padded_buf` swapped `TRANSFER_DST` for the `TRANSFER_SRC` the dump
  copy actually uses. A non-coherent ReBAR upload type is now rejected (falls
  back to staging) instead of being accepted and never flushed. The host timing
  probe accumulated `t2-t1` (pool pop) as "upload" and silently dropped the
  real `t2 -> t3` upload/pad-stage; now `acquire = t0..t2`, `upload = t2..t3`,
  so the stages sum to `avg_total`. Spec-constant entries are built from an
  explicit `(id, value)` list (the positional form delivered `filter_type` as
  constant 2 when `filter_type < 0, zmean >= 0`).
- **`DUMP_PAD` copied 46464 B into a 32768 B staging buffer** (one padded plane
  into upload+download) — `VUID-vkCmdCopyBuffer-size-00116`, and it clobbered
  the download region, so the dumped frame's output was corrupt. On this box it
  **reset the GPU**. Fix: staging reserves a third region for the dump
  (creation-time, only when `DUMP_PAD` is set), the copy targets
  `upload_total + download_total`, the host reads from there, a hard bound
  refuses any copy that would not fit, and whole-range invalidate covers a
  non-coherent staging type. `VSFEEL_DFTTEST_DUMP_PATH` is now mandatory (the
  old hardcoded `/tmp/opencode/pad_dump.bin` fallback is gone). Verified under
  `VK_LAYER_KHRONOS_validation`: 0 validation errors, dump 46464 B, and the
  output frame digest is byte-identical with and without the dump. Repro:
  `tmp/wo21_dump_check.py`.
- Suite 61/61 at each rebuild.
- **Cross-cutting hardening pass.** Every offset pushed to the shader is an
  `int32`, so creation now bounds each region in the unit the shader uses:
  `tw*pad_elems`, `upload_bytes`, `padded_bytes`, `slot_plane_bytes` and
  `nblk*256` per plane, the `upload_sum`/`download_sum`/`padded_sum`/
  `spatial_sum` aggregates *and* `upload_total + download_sum` (that pair is
  what `dst_base` is built from), plus `slot_total` for the slot cache — 8K
  16-bit + `num_streams=32` used to wrap `slot_base[]` negative and the
  fused-direct variant has no fallback guard. `f0beta` now rejects NaN, the
  `slocation/ssx/ssy/sst` `mapGetFloatArray` error is checked before the result
  is dereferenced, and all flush/invalidate ranges go through the shared
  `flush_range`/`mapped_range` helpers (offset rounded down and size rounded up
  to `minNonCoherentAtomSize`, clamped with `VK_WHOLE_SIZE`). `DFTTest` also
  reports `"requires Vulkan 1.3 (device reports X.Y)"` instead of an opaque
  pipeline error. Verified in `tmp/verify_validation.py`; the atom-alignment change
  is a no-op on this coherent device.


## Commands

```bash
# build + install (MUST copy after every rebuild or you bench a stale plugin)
cmake --build build
cp build/libvsfeel.so /usr/lib/python3.14/site-packages/vapoursynth/plugins/vsfeel/

# tests (30; ~2 s; a HANG = bug — always run with a timeout)
MANGOHUD=0 timeout 20 python -m pytest tests/test_dfttest.py -q

# benchmarks (primary metric: --synthetic BlankClip GRAY16 1080p; the real
# clip /home/encode/test/jpbd.mkv is decode-capped ~631 fps — smoke test only)
MANGOHUD=0 python3 benchmark/bench.py --synthetic --frames 500 --filter dfttest vsfeel vszipcl vszipcu
MANGOHUD=0 python3 benchmark/bench.py --synthetic --frames 500 --num-streams 4 --filter dfttest vsfeel vszipcl vszipcu
```

Bench defaults: ftype=0, sigma=8, sosize=12, tbsize=3, swin=0, twin=7,
sbeta=2.5, tbeta=2.5, zmean=1, f0beta=1.0. Test input: `tests/noise_24f.mkv`
(24f 640×360 noise); `_COMPARE_SCRIPT` fetches frames out of order (0, 11, 23)
— intentional, must keep working. rocprofv3 **cannot** profile Vulkan/RADV —
use `VSFEEL_DFTTEST_GPU_BENCH` for stage timings.

## The frame cache (current design)

Each *source* frame of the tw-frame window is reflect-pad'd **once** into a
shared device-local **slot buffer**, then reused across the window via D2D
copies into each frame's padded planes. Steady state: 1/3 of frames pay
upload+pad, 2/3 pay a 14 µs D2D copy.

- Slot buffer: `K = effective_streams + 3` slots **per plane** (5 at the
  default `effective_streams = 2`), `slot_plane_bytes = pw*ph*bytes`.
  Binding 4. 1080p GRAY16: 5 × 4.12 MB ≈ 21 MB/plane.
- Per-frame submits: pad CBs are submitted **at claim time** (below); then
  `[cmd2: D2D copies + T→C barrier]` (timeline waits = deduped (slot sem,
  signal value) pairs, fence=NULL) and `[cmd: pre-recorded fused+col2im]`
  (fence=resource.fence). Worker waits resource.fence.
- `cmd_pad` is a **per-(plane,t) CB ring** (`tw*num_planes` per resource): a
  CB must never be re-recorded while its previous submission still executes.

### State and protocol (all under `d->slot_lock`)

```cpp
struct SlotState {          // index = plane*K + s
    long long gen = -1;     // source idx currently in the slot
    std::vector<std::pair<int,long long>> committed;  // readers: {res_id, frame_gen}
    VkSemaphore sem {};     // ONE TIMELINE sem per slot, kept for the slot's lifetime
    uint64_t signal {};     // value the current generation's pad signals; every
                            // reader of that generation waits on this value
};
struct ResMeta {            // stable alloc, res_meta[res_id]
    VkFence fence {};
    std::atomic<long long> frame_gen {0};   // advances when the resource is re-acquired
};
```

Per (plane, t): `idx = clamp(n-radius+t, 0, numFrames-1)`,
`which = n - max(0, idx-radius)`. Clamped boundaries map two slices to the
same (slot, which) → **waits are deduped** (a double wait on one binary sem
deadlocks).

- **Peek**: `st.gen == idx` → **reader**: `st.committed.push_back({res,
  my_gen})`; op waits on `st.sem[which]`.
- Else **padder**: upload `src[t]` → own staging + per-op
  `vkFlushMappedMemoryRanges` (must precede the pad submit), re-take the lock,
  re-check (if someone claimed in between → reader, upload wasted), else
  **claim**:
  - **Reclaim rule (no host waits):** the slot is free iff `gen == -1` or
    every committed reader's `res_meta[rid].frame_gen` has advanced past its
    commit gen — resource reuse happens-after the frame's fence wait, so an
    advanced gen means that reader's copy is done.
  - Destroy the old `st.sem`, create **tw fresh semaphores** (no stale
    signals across generations), `st.gen = idx`, `st.committed = {self}` (the
    padder is also a reader).
  - **Submit the pad immediately, still under slot_lock** (nested queue_lock):
    one-dispatch CB (`ENTRY_PAD_SLOT`, `pc.padded_base=slot_base,
    pc.src_base=upload_offset, pc.pad_t0=t`), `pSignalSemaphores = st.sem`,
    fence=NULL.
  - **Fallback** (all K slots busy — with out-of-order completion at ns>1,
    or radius 3 where one frame's window spans 7 sources against K=5 slots):
    `ENTRY_PAD_DIRECT` pads straight into the per-resource
    padded buffer; no copy, no sems (fused reads it same-queue-ordered).
    ns=1 + radius 1 never hits this: the in-flight set is a consecutive
    2-window (ticket released at frame end, in-order completion on one
    queue) → 4 distinct sources < 5 slots, so a slot is always reclaimable.

### Why this is deadlock/stall-free (do not "simplify" away)

- **No queue stall (the original bench hang):** RADV does not execute a later
  submit while an earlier submit on the same queue is blocked on an
  un-signaled semaphore — the queue stalls, `vkQueueSubmit` blocks host-side,
  all workers freeze (GPU idle). The old design batched the pad submit after
  the ops loop, so a reader's copy submit (sem wait) could land before the
  padder's pad submit (signal) → hang. Now the pad is submitted inside the
  claim's slot_lock section: every reader commits after the claim, so its
  copy submit is always queued after the pad submit. Same queue: signal
  before wait. Cross-queue: independent queues, the signal's queue always
  proceeds.
- **No host drains:** draining the old generation (host-wait on old readers'
  fences) deadlocks — two concurrent padders each drain the other's fence
  (circular), and a mutex+cv ordered-submission wait starves the VS worker
  pool. Reclaim via `frame_gen` needs no host waits at all.
- **No stale signals / no second-consumer starvation:** one timeline
  semaphore per slot; each generation signals a fresh value
  (`st.signal += 1` at claim) and ANY number of readers wait on
  that value non-destructively — the chained-instance deadlock (2+
  consumers of one binary signal) is structurally impossible. Destroying
  the slot semaphore at reclaim is safe: reclaim only happens after every
  committed reader's resource advanced, i.e. its copy is done.
- Lock nesting is exactly `slot_lock → queue_lock` (in `submit_pad_op`); no
  other path takes them in that order.
- **This box cannot do DEVICE-side waits:** `vkCmdWaitSemaphores`/
  `vkCmdSignalSemaphore` are absent from the (stripped) Vulkan loader —
  `vkGetDeviceProcAddr` returns NULL (verified against the loader binary's
  strings; the ICD likely supports them but is unreachable through this
  loader). SUBMIT-level timeline semaphores DO work: they need only
  `VkPhysicalDeviceTimelineSemaphoreFeatures` (a device feature) plus the
  `VkTimelineSemaphoreSubmitInfo` pNext on `vkQueueSubmit` — no loader
  entry points.

### Pitfalls (each cost a debugging session)

- `load_upload()` adds `pc.src_base`; C++ must set `pc.src_base =
  cfg.upload_offset` ONLY (the shader adds `pad_t0*up_slice`). Double-adding
  the temporal offset made plane t read t+t planes ahead (t=2 read past the
  upload region into the download region → small correlated diffs; tbsize=1
  looked perfect).
- `VSFEEL_DFTTEST_TRIVIAL=1` still submits (empty) pad CBs so reader waits
  don't stall.
- The D2D copy submit must size `pWaitDstStageMask` per wait — sharing one
  `VkPipelineStageFlags` value when `waitSemaphoreCount > 1` reads out of
  bounds (fixed together with the timeline rework).
- `ENTRY_PAD` (the old t-loop pad into binding 3) is dead: the C++ side and
  the CMake entry are removed; the shader entry still sits in dfttest.comp
  (not compiled) pending a shader-source cleanup pass.

## Regression test

`test_dfttest_vspipe_pipelined_no_hang` (ns=1/4) runs the real
`benchmark/bench.py --synthetic --frames 200 --filter dfttest vsfeel` through
vspipe with a 120 s timeout. vspipe's pipelined reader + prefetch activates
frames 11+ ahead while older frames are in flight — the workload that stalls
the queue under the old batched-pad-submit design (pure-Python sequential
`get_frame` does NOT reproduce it). A hang fails the test; a reintroduced
batched pad submit + submit-level wait pattern is caught here.

Companion regression: `test_dfttest_vspipe_pipelined_chained_no_hang_16bit`
streams 3 chained instances through vspipe — the user's deadlock shape.
Upstream frames get processed again while a previous processing is still in
flight; with binary slot sems the second consumer wedges the queue (see the
chained-instance section above). The fixed build completes the run at a
steady fps; a hang (or a non-zero vspipe exit) fails the test. Uses a
synthetic GRAY16 BlankClip (no external media): 8 MB cache, **LEN=200 plus an
eval-time probe**.

**Reload trigger (2026-08-28):** the old test (LEN=12000, no probe) was
flaky — the blank-chain stall position is timing-dependent (observed 9k-37k
frames on this box across days; the buggy build even completed 12000). The
reliable trigger is an eval-time frame probe: wrapper libraries
(vsdenoise's `check_progressive` -> `FieldBased.from_video`) *sample a frame
of each intermediate node while the script evaluates*, and the streaming
pass then re-requests those frames "while still in flight" — the binary-sem
condition fires at frame 0. Reproducer (self-contained, no wrapper):

    BlankClip GRAY16 1080p, LEN=200; for 3x: DFTTest(sigma=7, tbsize=1);
    clip.get_frame(0) after each instance; vspipe -p.

Verified against the pre-fix build (2c58074, rebuilt): hangs 3/3 at zero
frames for LEN=100/200/500/1000 (probe [0] alone suffices); the fixed build
completes 3/3 in ~2 s. The old 12000-frame variant took ~26 s on the fixed
build and could NOT reliably catch the bug on this box anymore.

## Remaining paths (priority order)

1. **~~ns=4 gap: fused reads the slots directly~~ DONE 2026-09-08** (see
   Speed-run section above): per-t slot bases via `slot_base[7]` push
   constants, D2D copies + copy submit deleted, branchless FUSED_DIRECT
   variant for the all-slots fast path.
2. **ns=4 SDMA overlap** (the remaining gap, 1325 vs 1297: vszipcl runs
   H2D/D2H on separate copy engines full-duplex while kernels run; ours
   serializes everything on the compute CP): a dedicated transfer queue +
   2× staging ping-pong, pads/uploads on DMA while fused runs. Untried.
3. **Fused codegen residue (448 vs 384 µs):** remaining IM2COL/window ALU,
   (pointer-walk strength reduction: base + increment per j instead of
   recomputing `((i)*ph + iy*bs + j)*pw + ix*bs + lane`), ACO `v_dual_mov`
   (128 vs 18) / dual-issue packing. ACO 2613 instr vs vszipcl 2546 (both 2
   wave32/SIMD).
 3. Col2im (260 µs) is already 2× faster than both references — leave it.
 4. Optional hygiene: remove the dead `ENTRY_PAD` entry from dfttest.comp
    (C++/CMake side already cleaned).

## Key implementation facts

- **Spatial buffer = `num_blocks * 256` floats (center slice only).** Fused
  writes `pc.spatial_base + block_id*256`; col2im reads
  `pc.spatial_base + ((i*hn+j)*16+off_y)*16+off_x`. A mismatched stride
  zeroes the bottom half (this is what fixed the 3.4 GB → 1.2 GB VRAM
  thrash, 236 → 801 fps).
- 16-bit IO: plain `uint16_t[]` storage (`GL_EXT_shader_16bit_storage`,
  `storageBuffer16BitAccess=VK_TRUE`; glslc `-O` bug on `OpUConvert` to
  16-bit → `uint16_t(uint(x) & 0xFFFFu)`).
- `create_pipeline()`: spec constants constant_id=1 filter_type,
  constant_id=2 zmean (pad/col2im pass -1); wave32 via
  `VK_EXT_subgroup_size_control` pNext on all kernels (the 552→673 win).
- `effective_streams = max(num_streams, 2)` (resource pool / ticket = 2
  in-flight by default; was 8 — see VRAM note below). Per-resource ReBAR
  `up_buf` (upload_total, host-mapped VRAM) + `pad_set` (binding 1 = upload
  buffer; `desc_set` binding 1 stays staging for fused's download writes);
  `num_queues = min(num_streams, queue_count)` (≤4).
- Fused: SUB_BLOCKS=8 (128-thread WGs), `td[TD_SZ]` register-resident,
  `subgroupBarrier()`, ~240 VGPR → 2 wave32/SIMD.
- Stripped Vulkan headers/loader on this box (see above): also no
  `vkGetDeviceProperties`.

## Reference implementations (same GPU, READ-ONLY)

- **vszipcl** = OpenCL/ROCm (`reference/vapoursynth-zipcl/`): 128-thread WGs,
  16-lane tiles, TILES_PER_GROUP=8, 17-stride shared-memory transpose.
  Pad-once frame cache (`clframecache.zig`): reuse via
  `clEnqueueCopyBuffer`, OpenCL **command-level** events (a later command can
  run while an earlier one waits — the queue-stall freedom this Vulkan loader
  can't give); slots = `tw + max(num_streams, n_threads) + 2*radius`.
- **vszipcu** = HIP/ROCm (`reference/vapoursynth-zipcu/src/dfttest.cu`):
  `__syncwarp`, `__shfl_sync` DC broadcast, same cache concept.
- Kernel math is identical across all three; ROCm LLVM does NOT beat ACO —
  the differentiator is the frame cache + stream structure.
- ACO/OpenCL comparison: ours `RADV_DEBUG=asm` (ACO on stderr); vszipcl build
  the .cl offline (`clang -x cl -target amdgcn-amd-amdhsa -mcpu=gfx1100 -O3
  -cl-std=CL1.2 -cl-denorms-are-zero`, prefix from `dfttest.zig
  genKernelPrefix`; artifacts in `/tmp/opencode/`), `llvm-objdump -d`; count
  instr / s_mov / ds / VGPR / FP-op mix (a 1.3× instr gap ≈ 1.5× time).

## Wins (cumulative, ns=1 synthetic fps)

72 → 303 fused unroll (td[] register-resident) → 457 CPU/GPU pipelining
(effective_streams 8) → 470 AVX2 streaming upload → 472 uint16 storage +
SUB_BLOCKS=8 (16 is worse, ~400) → 552 filter_type spec constant + dispatch
grid fix → 673 wave32 pNext → 801 center-slice spatial buffer (3.4 GB →
1.2 GB) → **1009 frame cache** (+22% vs vszipcl 829). Fused codegen
(float-literal FFT constants, ZMEAN spec constant, gf hoist): 3378 → 2613
instr, 690 → 468 µs (vszipcl 441).

## What did NOT help (do not retry)

- LDS-slice fused restructure (2× slower, LDS-bound; reverted `3589644`).
- SUB_BLOCKS=16, DMA upload to device raw_buf (453), float-internal buffers +
  pack kernel (buggy), `RADV_PERFTEST=cswave32` (driver-global, not
  shippable), `OpExecutionMode SubgroupSize` injection (ignored), glslc
  `-Os`/spirv-opt (no-op), `RADV_DEBUG=llvm` (not in release Mesa),
  `RADV_DEBUG=shaderstats` (prints nothing).
- **Host-side cache synchronization in any form:** mutex+cv ordered-submission
  wait (deadlocks the VS worker pool by starving lower frames), drain-on-fence
  (circular waits between concurrent padders). The cache must be sync-free:
  GPU ordering via submit order + submit-level binary sems, reclaim via
  frame_gen.
- **Batched pad submit after the ops loop** (the bench hang), **fixed
  per-slot binary semaphores across generations** (stale-signal corruption),
  and **any binary-semaphore slot signalling under chained instances**
  (2+ consumers per signal → permanent queue stall; this deadlock is what
  the timeline rework fixes).
- Diagnosing the cache corruption as an "out-of-order race" — it was the
  double temporal offset (see pitfalls).

## Debug env vars

- `VSFEEL_DFTTEST_GPU_BENCH=N` — one-shot stage timing on frame 0 (cold).
- `VSFEEL_DFTTEST_TRACE=1` — per-frame protocol trace (claim/read/reclaim/
  submit lines), plus the one-shot `[dfttest] up_direct=` line at creation
  (also shown with `VSFEEL_DFTTEST_DBG=1`).
- `VSFEEL_DFTTEST_FORCEPAD` — pad every op (bypass slot reuse).
- `VSFEEL_DFTTEST_DUMP_PAD=n` + `VSFEEL_DFTTEST_DUMP_PATH=file` — dump the
  padded buffer after frame n. The path is now required (no hardcoded default).
- `VSFEEL_DFTTEST_FAILPAD=N` — fault injection: fail the Nth slot pad submit,
  to exercise the claim-rollback path.
- `VSFEEL_DFTTEST_STREAMS=N` — override effective_streams.
- `VSFEEL_DFTTEST_TRIVIAL=1` — skip kernels. `VSFEEL_DFTTEST_DBG=1`,
  `VSFEEL_DFTTEST_SGSIZE=N` — plumbing.
- `RADV_DEBUG=asm` — ACO ISA to stderr (works).

## Shared-plumbing refactor (2026-08-26/27) — ODR COMDAT hazard, fixed

All five filters now share inline plumbing from vsfeel.h: `FramePool<T>`
(ticket semaphore + mutex-guarded LIFO of per-frame resources; take/give_back),
`destroy_common(dev, r)` (frees cmd/pool/fence/staging), `submit_with_fence`,
`submit_timeline` (waits/values/stages + optional signal value + optional
fence, all under the queue's lock), and `trace_on(env)`.

**CRITICAL ODR FINDING (crashed test_dfttest_deterministic as a SIGSEGV with
a corrupted pool item; `resource.id` garbage, `_M_finish` misaligned by 72
bytes = sizeof(FramePool)):** dfttest/gaussblur/bilateral each define their
own `struct VK_Resource` at namespace scope with different field sets (and
sizes), yet instantiate `FramePool<VK_Resource>` from the shared header. The
mangled template names are identical across the three TUs, so the linker
folds (COMDAT) the `give_back`/`take`/`push` instantiations into ONE — built
with the *wrong* `sizeof(VK_Resource)` (e.g. 80-byte stride from another
filter while dfttest's is 152) — the vector's `_M_finish` then moves by the
wrong amount, `items.back()` lands mid-item, and reads produce garbage
handles. Only frames that ran more than one concurrent processing (2+ frames
in flight — always true in streaming) hit it; single-frame digests passed.

Fix: renamed the per-filter structs to unique names — `DFTTestResource`
(dfttest.cpp), `GaussBlurResource` (gaussblur.cpp), `BilateralResource`
(bilateral.cpp). nlmeans (`NLStream`) and bm3d (`Bm3dStream`) never collided.
Rule going forward: any filter-local type used to instantiate a shared
template in vsfeel.h must have a unique name (or be renamed when structs
are added).

Verified on the fully-migrated build (md5 6c09249a, installed): GRAY32
crash repro (48 get_frames, deterministic) runs clean; original chained
deadlock repro N=3/4/5 streams at 328/244/197 fps (no hang, no exit-124);
full suite 187 passed (112s); per-filter benches match baseline (see
tmp/baseline_bench.txt): gaussblur ~800, bilateral ~895, dfttest ~790,
nlmeans ~425, bm3dv2 ~154 fps.

dfttest submit migration detail: frame-start `vkResetFences` removed — the
fence resets now happen inside submit_with_fence/submit_timeline under the
queue lock at the fused submit; copies are fence-less timeline-waits.

## Queue cap and creation error path

`resolve_queue_cap` was called with `d->num_streams` (1 at the shipped default)
while the loop creates `effective_streams` (= max(num_streams, 2)) resources, so
the cap collapsed to 1 and `VSFEEL_DFFTEST_QUEUES` could never exceed it. It now
resolves from `effective_streams`: with the knob unset the two in-flight streams
go to two queues, `VSFEEL_DFFTEST_QUEUES=1` puts both on queue 0, and `=2/4`
selects two (verified with a temporary `num_queues` print, since removed).
Same-session interleaved A/B at the shipped default, 3 x 1000 frames
(`bench.py --filter dfttest vsfeel`): two queues 1362.7/1377.1/1367.4 fps vs
one queue 1298.5/1267.3/1278.3 — **+7.0%**, so the README row was updated. The
per-stream resource is also created into the pool via `FramePool::emplace()`, so
a creation error is torn down by `~DftData` instead of leaking. Full suite 600
passed.
