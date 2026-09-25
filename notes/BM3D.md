# BM3Dv2 — notes

**shipped.** Vulkan port of vszipcl's BM3D (fused mode: one estimation pass per
output frame, temporal aggregation over the stack window). Numerically faithful
to vszipcl; see the tolerance policy in `tests/test_bm3dv2.py`.

Current correctness and performance fixes: all BM3D radii use exact per-candidate
SSD scores with a flattened candidate partition; per-slot witness clearing is
ordered with estimation, and radius-0 fallback uses its private source slot.
BM3D tests pass; the 1000-frame jpbd benchmark is 766.5 fps for the exact
sigma=0.7, radius=2, bm_range=9, ps_range=4, block_step=8 config.

Current design:

- Two kernels: `bm3d.comp` (block match + group + collaborative transform,
  one warp of 32 lanes = 4 sub-groups of 8 lanes, one 8x8 block each) and
  `bm3d_agg.comp` (temporal aggregation over the TW = 2r+1 stack slices).
- Estimation accumulates with hardware buffer float atomics where the device
  has them (`VK_EXT_shader_atomic_float` + its float2 companion, and
  `shaderBufferFloat32AtomicAdd` set — the plain `...Atomics` load/store/exchange
  bit is a separate feature and not enough; RADV gates both at GFX11). Everywhere
  else the same kernel's `-DNO_FLOAT_ATOMICS` build runs the reference's own
  `atom_add_f` CAS loop; the host picks per device, `VSFEEL_BM3D_CAS=1` forces
  it (see Historical for the cost).
- The spatial and temporal scans use a flattened candidate partition; every
  lane evaluates the same direct SSD, then each subgroup merges its local top-K.
  Adjacent lanes cover adjacent candidate origins for better source-load locality.
- Candidate SSDs compare each source patch against the same fixed 8x8 reference
  patch; reuse of shifted SSD column sums is invalid because it shifts the fixed
  reference columns too.
- Temporal per-window lists are only PS_NUM deep; the top-k merge proof is in
  Historical. The full spatial list stays 8 deep.
- Degenerate paths: `sigma < FLT_EPSILON` passes the plane through (a source
  copy), like the installed references' `PROC_MASK`.
- **Everything goes through the core's exec pool** (`createGPUExecPool` /
  `gpuExecAcquire` / `gpuExecSubmit`); the filter owns no command pool, timeline
  or fence. The estimation is one submission per recomputed window position (a
  full-recompute frame has up to 2r+1 of them) so no single submission can reach
  Windows' TDR watchdog on a slow card (`VSFEEL_BM3D_SPLIT=0` reverts), then one
  aggregation submission whose output plane the pool publishes.
- The estimate stacks and source ring stay private VRAM buffers. Cross-frame
  handoff is a per-slot *submitted* ready flag: a reader waits host side until
  its writers' estimations are submitted, then a full pipeline barrier at the
  start of its first command buffer supplies the execution and memory dependency
  (a barrier's first scope is every earlier command in submission order on the
  queue, so no per-frame semaphore wait is needed). A reader never holds a
  recording context while waiting, so the pool's ring cannot deadlock.
- Runs on the R80 GPU API: `clip:vnode:gpu` in and out with `ffGPUOutput`, so
  the core's `GPUUpload`/`GPUDownload` cross the bus and a consumer waits on the
  plane's producer pair; device choice, queue locking and the buffer pool are the
  core's.
- `num_streams` and `device_id` are registered no-ops (never read) -- depth is
  the core's, device choice is `core.set_vulkan_device`; the cache depth is a
  fixed two (`kInflightFrames`).

Performance — 1080p GRAY32, jpbd, `tools/benchmark.py -f bm3dv2 vsfeel vszipcl
bm3dvk`, 1000 frames × 3 interleaved, sigma 0.7, radius 2, bm_range 16,
ps_range 7, block_step 4:

| | fps | note |
|---|---|---|
| vsfeel | **332.8** | exec-pool port; different benchmark config from the current working-tree scan |
| bm3dvk | 54.8 | R80 reference |
| vszipcl | 41.6 | |

vsfeel is 6.1x the faster reference. The first R80 API port (2026-09-20) cost
4-6% on this CPU-sink benchmark; the exec-pool port (2026-09-23) is **+1%** in
three interleaved A/B runs against the raw-submit build -- see Historical.

## Implementation notes that the code alone does not show

- **The block-matching search is the whole filter.** At r=2/step 4 it is ~85%
  of the estimation kernel; the collaborative transform + patch loads + all
  133.8 M float atomics together are only ~0.4 ms/frame, and the zero-fill plus
  the ring copies are ~0.1 ms. Nothing else is worth tuning (see the ablation
  table under Historical).
- **The aggregation was a PCIe store, not work** (pre-port): ~0.32 ms at
  TW = 1, 5 and 9 alike, the 8.3 MB result going to GTT at ~26 GB/s, which is
  why vectorising it to `vec4` changed nothing. The R80 port writes the output
  plane in place, so that PCIe cost now lives in `GPUDownload` -- see the port
  round.
- **LDS is a trap for this kernel.** Three separate attempts to move
  kernel-live data into shared memory (the resolved group, the per-lane scan
  lists, the direction-invariant centres) all raised ACO's VGPR count from 216
  to 240 and dropped occupancy from 7 to 6 waves/SIMD, losing 5-90%. The
  mechanism is the dynamic LDS addressing, not the memory traffic. Do not
  retry an LDS variant here; the win came from *reducing the data* (list depth,
  packed coordinates) instead.
- The 8-lane transposes and the group-8 reduction are `subgroupShuffleXor`
  butterflies (register-only, no LDS, no barrier) and reproduce the
  reference's exact reduction tree.
- `extractor_exp`'s `(x + E) - E` pre-rounding must stay under GLSL `precise`:
  RADV folds the pair to `x` when E is a known spec constant, which makes the
  parameter inert (fixed, see Historical).

## Historical

Chronological; each entry keeps the mechanism, not the story.

- **The CAS fallback's retry bound was below what the kernel can contend.**
  `res_add`'s compare-exchange loop is capped so a stuck retry cannot reset the
  device (the reference's unbounded `do/while`); one res element receives at most
  `8 * ceil(8 / block_step)^2` adds, so the old 32 was exactly the bound *only at
  the tuned block_step=4* and short of it for every smaller step (512 at 1).
  Measured with a NaN-store probe: block_step=1 exhausted it (1074 NaN pixels
  over the clip's frames); against the hardware-atomic arm, the old bound put 16
  of 4096 pixels 1e-5..6.1e-5 out, the derived 512 lands at 7e-9 (add order
  alone). `test_bm3dv2_cas_fallback_holds_at_small_block_step` pins it.
  The last-resort store now lands on a freshly re-read value (bounded, and
  unreachable for every supported block_step) and `gpu_make_buffer` refuses a
  buffer over `maxStorageBufferRange`, which BM3D's 190 MiB estimate cache
  exceeds before the other filters' scratch do.

- **2026-09-23 — exec-pool port: +1%.** Replaced BM3D's per-stream
  command pools, timelines and raw `gpu_submit` with the core's exec pool, and
  deleted `FramePool`/`ticket_semaphore`/`gpu_submit` from `vsfeel.h`. The
  blocking change: the pool allocates timeline values at submit, so the cache's
  pre-reservation `(timeline, value)` writer pairs became per-slot *submitted*
  flags. Cross-frame visibility and execution ordering now ride a leading
  `vkCmdPipelineBarrier` in the reader's first submission rather than per-frame
  semaphore waits; the reader also waits for a writer's *submission* where the
  old code waited for completion, which is safe because a barrier's first
  synchronization scope is every earlier command in submission order on that
  queue. The estimation-before-aggregation overlap, the TDR split, the cache
  sizing and the radius-0 private slot are unchanged. Cross-frame waits happen
  with no recording context held and the acquisition order makes the wait graph
  acyclic, so the pool ring cannot deadlock. Three independent interleaved
  same-session A/B runs of 3 rounds x 3 reps (1080p jpbd GRAY32 r=2, 1000
  frames): 335.1/332.3/332.0 vs 335.9/335.3/335.7, 334.0/331.0/330.7 vs
  333.6/334.1/333.9 and 333.2/332.1/333.3 vs 331.2/336.6/336.8 fps (old vs
  new), i.e. +1.0%, +0.9% and +1.0% on the medians. 71/71 BM3D tests; full
  suite green (732 passed; also green with `VSFEEL_BM3D_SPLIT=0` and
  `VSFEEL_BM3D_CACHE=1`).

- **2026-09-20 — R80 GPU API port: -4..6% on a CPU sink, upload free, download
  the whole cost.** Same-session pairs, 1080p GRAY32 r=2 1000 frames x3:
  314.43 -> 297.04 and 302.87 -> 290.94 fps. Mechanism: the core's
  `downloadPlanes` reads a plane in place only when it is
  `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED`, and a discrete card's frame planes
  are write-combined, so a CPU consumer pays a full-frame DMA into cached
  staging plus the host copy -- where the pre-port build wrote its result
  straight into host-visible memory from the aggregation kernel. The upload is
  free because `uploadPlanes` does have that path (one memcpy into the frame's
  VRAM plane, no submission); per 1080p GRAY32 frame, single vspipe runs of an
  otherwise empty graph: 0.276 ms cache floor, 0.921 download only, 0.899
  upload + download. BM3D with `--gpu-cache` (no upload stage, 3x medians):
  295.85 vs 295.92 fps, i.e. unchanged.
  Dead ends, all slower than the shipped version:
  - Own host staging for a CPU input, -12.8% (273.2 vs 314.4): a pooled host
    buffer plus a DMA into the ring loses to writing the frame's plane directly.
  - Writing the source ring directly when `createGPUBuffer` returns it
    host-visible recovered most of that (299.8) but needed a second IO path for
    the same result.
  - Cached vs uncached staging and NT stores changed nothing.
  Two correctness bugs, both fixed: releasing a cache slot at the next use of
  that stream deadlocked `acquire_cache` (release at submit instead -- the one
  compute queue already orders a later recompute after the aggregation), and a
  pass-through path that ran after `acquire_cache` without releasing its
  reservations hung the next frame. Benchmark-harness traps found here are
  commented in `tools/benchmark.py` (`--gpu-cache`), not repeated.

- **Block matching reused shifted SSD columns incorrectly.** For radius 0–2,
  each candidate's cached source-column errors were compared against shifted
  columns of the fixed reference patch, changing candidate ranks. All radii now
  use exact SSD scores with a flattened candidate partition; on the jpbd
  benchmark config above, 647 → 766.5 fps; 75/75 BM3D tests pass.
- **Concurrent first-use tag clearing could erase witnesses.** `tags_cleared`
  raced across parallel callbacks, and a whole-buffer clear could submit after
  another estimate wrote tags. Each estimate now clears only its exclusively
  reserved slot in the same ordered command buffer.
- **Radius-0 aggregation fallback used a frame-derived ring slot.** Radius 0
  reserves private slots that need not equal `frame % ring`; fallback now uses
  the reservation's slot. No performance change intended.
- **2026-09-25 — the aggregation trusted slices that were not this frame's.**
  `acc/acw` never checked that the TW slices it summed were the output frame's
  own contributions, so a zero-filled slot divided by zero (black band), a partly
  accumulated one averaged short (dim band), and a recycled one contributed
  another frame's patch groups (the band that matched no frame). The estimation
  now witnesses each slice (`tags[slot * TW + z]` in `bm3d.comp`) and the
  aggregation skips any slice whose witness is not the frame it intends,
  falling back to the source pixel when none survive (`bm3d_agg.comp`,
  expectation computed in `bm3d.cpp`); 73/73 reference tests pass, and with the
  witnesses zeroed the output is bit-exact the source. No perf change.

## Round: the block-match scan (2026-09-21, +64% end to end)

Everything above is a whole-kernel time from a warm single-request
`VSFEEL_BM3D_GPUTRACE=1` run (`-r 1`, 300 frames, 1080p GRAY32 defaults), which
settles to ±0.2% on repeat. Kernel ms is the metric for kernel work: the fence
is depth-invariant (5.4 → 5.6 ms over radius 1..8) and fps is flat from ns=2 to
ns=8, so the GPU is the wall. Graded fps is quoted separately.

1. **Four-wide `scan_row` optimization — superseded by the SSD correctness fix.**
   Its timings described the old shifted-column reuse and are not valid for the
   corrected exact per-candidate SSD path; do not quote them for current code.
2. **Pack the candidate's (x, y) into one word — 3.96 → 3.72 ms.** The insert
   shifts three arrays per position instead of four; x gets 16 bits and y 15,
   which is why creation now rejects dimensions above 65535x32767.
3. **Temporal per-window lists are PS_NUM deep — 3.72 → 2.85 ms, VGPR 216 →
   192, subgroups/SIMD 7 → 8.** `merge_group` only ever hands `fe[0..PS_NUM-1]`
   to `ginsert8` and only the first PS_NUM merged entries are read back as the
   next window's centres, so each lane only needs its own top-PS_NUM: a k-way
   merge's k-th output is always the k-th smallest of the union, and a global
   top-k element is inside its lane's top-k. With PS_NUM=2 the insert shifts one
   element instead of seven and the list costs six registers instead of
   twenty-four. The spatial list stays 8 deep (its group of 8 is the filter's
   actual output group). `ps_num` 1..8 all pass the reference sweep.
4. **Dead ends, with mechanism.**
   - The resolved group in LDS (Stage A): 5.36 ms, VGPR 240, 6 waves.
   - The per-lane scan lists in LDS with a bank-conflict-free `[k][lane]`
     layout and a real branch on the insert: 7.27 ms, VGPR 240, 6 waves. The
     branch fires as intended; the register cost of the dynamic addressing
     swamps the 60 instructions it saves.
   - The direction-invariant centres in LDS: 3.88 vs 3.72 ms, VGPR 240.
   - `reduce8` (reduce the sliding window from explicit terms instead of
     shifting a `col[8]`): byte-identical code, no change, reverted.
   - `vec4` aggregation: 0.330 vs 0.327 ms, i.e. neutral — see the PCIe note.
5. **Ablation ladder** (same kernel, default windows): full 2.884 ms;
   `NOESTIMATE=1` 2.488 (so patch load + transform + all atomics = 0.40 ms);
   `NOSEARCH=1` 0.741 — but NOSEARCH also makes every candidate tie, so it
   suppresses the temporal inserts too and cannot be read as "the spatial
   search costs 2.1 ms". The honest split comes from window sweeps:
   0.0032 ms per candidate per lane for `bm_range`, 0.0030 for `ps_range`, and
   a ~1.4-2.0 ms intercept at tiny windows (row-init column batches, merges,
   the estimate phase, the 83 MB/slot fill and the ring copies).

**Stream knee, re-swept after the restructure.** ns = 1/2/3/4/6/8 =
243.4/312.8/311.9/310.1/312.3/313.0 fps (800f × 2). A careful 2000f × 3 pass
gives 314.5 / 316.0 / 316.8 for ns = 2/4/8, and direct `vspipe` runs give
ns=2 the edge at 2-3 requests (318.4 vs 316.5) while ns=4 leads by 0.7-1.0%
only at vspipe's own ~32-request default (313.6 vs 310.6). The plateau is at 2;
2 was chosen as the default for the 332 MiB it saves (1107 vs 1440 MiB at
1080p r=2 — the estimate stack itself scales as `tw + ns + 2r`, so it is 23%
off the total, not just off the per-stream part).

**Per-instance VRAM**, printed by `VSFEEL_BM3D_VRAM=1` at creation. Two
formulas cover it: `src = src_ring * clips * pe` with `src_ring = 4r + ns`, and
`res = res_cap * tw * 2 * pe` with `res_cap = tw + ns + 2r`. 1080p r=2:
ns=2 1107.4 MiB, ns=4 1439.6 MiB, ns=8 2104.1 MiB; the shared estimate stack is
79% of that at ns=2 and 64% at ns=8 (both it and the per-stream staging scale
with the count), and the per-stream part is staging + dst only. The staging
right-sizing (4r+1 slots instead of `4r+num_streams`) removed 23.7 MiB per
stream, 95 MiB at ns=4. At r=4/ns=8 the same figure is 5.3 GiB for the basic
estimate and 7.0 GiB for the final (Wiener) pass, which is why the 32-bit `res`
addressing guard exists.

**The estimate cache is sized for the in-flight working set by default;
`VSFEEL_BM3D_CACHE=1` restores a window of seek margin.** An in-flight
frame needs the stacks of centre frames `[n-r, n+r]`, so the fixed
two-stream depth spans `2 + 2r` slots — that is `res_cap` now. The
default used to add `tw` more so an out-of-order (seek) request finds a warm
slot instead of waiting in the acquire; that margin costs `tw/(ns+2r+tw)` of the
largest buffer, which is what made radius 4 fail to allocate on an 8 GiB card.
1080p equivalents (banner run at 640x360, exactly x9 in `pe`) r=2 ns=2 1107 ->
712 MiB, r=4 ns=2 3132 -> 1851 MiB, r=4 ns=2 final 3544 -> 2263 MiB, i.e.
-36..41% of the whole instance. In-order throughput is unchanged: 1000f 1080p
GRAY32 r=2 ns=2, three interleaved pairs, 299.9/296.9/299.2 fps slack vs
299.3/298.8/297.6 fps working set. All 75 BM3D tests pass either way, including
the six request orders and the seek-collision case.

**Correctness of the round.** 75/75 BM3D tests and 800/800 repo tests pass.
No numerical change was intended or observed: the equivalence argument for the
PS_NUM-depth list is exact (same (e, sq) merge order), and the packing and
unrolling are order-preserving. `test_bm3dv2_matches_reference` spans
radius 0..4 against vszipcl at the documented tolerances.

**2026-09-20 — runs where buffer float atomics are missing (BM3D was
AMD-RDNA3-only).** RADV gates `shaderBufferFloat32AtomicAdd` at GFX11 and the
Windows driver has no such feature on Polaris, so creation failed on RX 580,
Vega and RDNA1/2 alike. `bm3d.comp` gained a `-DNO_FLOAT_ATOMICS` build
(`bm3d_cas.spv`, auto-selected, `VSFEEL_BM3D_CAS=1` forces it) running zipcl's
own `atom_add_f` CAS loop: core `atomicCompSwap` only, so nothing about it is
driver-specific, and one fp32 add per round keeps it bit-identical to the
hardware path at `extractor_exp=8` (75/75 tests). Cost, 1000f 1080p GRAY32 r=2
ns=2 interleaved pairs: 314.8/312.5/314.9 vs 260.0/260.4/259.8 fps, +17%.

**2026-09-20 — the estimate cache sizes to its working set by default.** `res_cap`
was `tw + ns + 2r` where the in-flight working set is only `ns + 2r`, so every
instance allocated `tw` slots for a margin only out-of-order requests consume —
which is what made radius 4 fail to allocate on an 8 GiB card. Dropping it is
-36..41% of total VRAM at no in-order cost; `VSFEEL_BM3D_CACHE=1` restores the
margin for seek-heavy graphs. Numbers in the VRAM paragraph.

**2026-09-20 — the direct-upload path needs a real ReBAR heap, not just the
memory type.** An RX 580/Windows reports heap 0 = 7936 MiB device-local and heap
2 = 256 MiB device-local, with the `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`
type in that aperture: staging taken from it failed with
`VK_ERROR_OUT_OF_DEVICE_MEMORY` while VRAM sat empty (1080p r=2; the Wiener pass
stages 284.8 MiB). `rebar_available(dev, bytes)` now requires the backing heap
to be a quarter of the largest device-local heap, then probes the whole
per-instance staging once before the stream loop; the DB machine is unchanged.

## Open work

- **RX 580/Windows: the long-submission TDR is bounded, awaiting a field run.**
  The freeze hits frames whose estimation recomputes every window position, with
  no player involved — not the seek, and not the GPU-timing probe
  (`timestamp_bits=64`, so those timestamps were valid usage). `bm_range=4 tr=1`
  and `NOSEARCH=1` both make it disappear, so it tracks the search's total
  duration in one submission; the estimation now submits per recomputed position.

- **The spatial search is the remaining kernel cost.** Its per-lane list must
  stay 8 deep, so its insert is ~14 instructions per candidate more expensive
  than the temporal one. Dropping the sequence index from the spatial list
  (recovering `sq = (cy-top)*rw + (cx-left)` from the packed coordinates at
  merge time, which is exact) is worth ~2% of the frame and was not taken.
- **Dead plumbing.** `merge_group`'s `ms` output and the spatial `gseq` array
  are unused tails of the port (`ms`/`fs` were already dead before this round).
- **The row-init column batches** (8 `col_ssd` per row per lane, 21 rows per
  lane per frame) are ~15% of kernel instructions and are inherent to the
  sliding window. The spatial scan also wastes ~18% of its wave iterations on
  the final row, where only 1 of 8 lanes is active.
- The estimate phase's 0.40 ms has not been decomposed into transform vs
  atomics. The CAS build bounds it (below): its +0.66 ms/frame for a
  read-modify-write loop means the two `atomicAdd`s are most of that 0.40 ms.
- **No cheaper atomics exist on the older devices that need the fallback.**
  GFX8–10 have no `buffer_atomic_add_f32`, so CAS *is* the floor there;
  `shaderSharedFloat32AtomicAdd` (which RADV does expose from GFX8) cannot help,
  because a group's 8 matched patches land anywhere in the plane — there is no
  per-workgroup LDS tile to accumulate into.
- `sigma` is scaled per plane but only luma is processed; the YUV path copies
  chroma. No measurements needed unless a user asks.

**Do not re-derive** (measured, not theory):

- Any LDS restructure — see the mechanism above.
- Queue cap: uncapped is best or tied (table above).
- `#pragma unroll`: glslc ignores it in GLSL and ACO already unrolls
  fixed-trip loops; the four remaining backward branches are the variable-trip
  scans, which is why the unrolling is manual.
- A different SSD accumulation order (e.g. a running window sum instead of the
  per-candidate 8-term tree): the reference-comparison tolerance is already
  0.0079 against a 0.01 bound at the defaults, so nothing may perturb which
  blocks match.

**Method rules**

- BM3D reference comparisons cover match-selection-sensitive inputs only
  incompletely: add structured horizontal patterns before changing SSD math.
- Grade kernel changes on `VSFEEL_BM3D_GPUTRACE=1` at `-r 1` over a few hundred
  frames (repeat once: the printed value settles to ±0.2%), then confirm fps
  with an interleaved `tools/benchmark.py` pair over 1000+ frames. A one-shot
  `-r 1` fps figure mixes in the ~1.4 ms host path and moves for unrelated
  reasons.
- The GPU timestamp accumulator had two bugs until this round — it added
  `uint64_t * float` into an `atomic<uint64_t>` (truncating) and divided ns by
  `1e3` while printing "ms", so every figure it printed before 2026-09-21 is
  1000x off. Use the accumulator, not the old numbers.
- Never chain build → install → test; use `tools/install.sh` (hash-verified).

## Debug env vars

All flags are `VSFEEL_BM3D_<FLAG>`, routed through the
`env_flag`/`env_int`/`env_str` helpers in `vsfeel.h`; `TRACE` and `DUMP` are
cached at creation, not read per frame.

- `VSFEEL_BM3D_TRACE=1` — acquire/submit/wait trace.
- `VSFEEL_BM3D_TIMING=1` — host-stage split per frame (cached at creation).
- `VSFEEL_BM3D_VRAM=1` — creation-time VRAM budget (pooled buffers only now).
- `VSFEEL_BM3D_SPLIT=0` — one estimation submission per frame (pre-TDR-split).
- `VSFEEL_BM3D_CACHE=1` — add the seek margin back to the estimate cache
  (0, the default, sizes it for the working set; see the VRAM paragraph for the
  +36..41%).
- `VSFEEL_BM3D_CAS=1` — force the atomicCompSwap aggregation build on a device
  that has buffer float atomics (A/B measurement only).
- `VSFEEL_BM3D_NOSEARCH=1` / `VSFEEL_BM3D_NOESTIMATE=1` — ablation knobs; both
  are vsfeel inventions, not reference behaviour, and NOSEARCH distorts the
  temporal search as well (see the ablation note).
- `VSFEEL_BM3D_DUMP=1` / `VSFEEL_BM3D_GPUTRACE=1` — slot dump / GPU timestamps.
  The probe waits each instrumented frame's aggregation out (the query pool is
  shared, so the path serializes) and reads `vkGetQueryPoolResults` on the host;
  it averages over 50 frames and never runs in a benchmark.
- `VSFEEL_BM3D_HD`, `VSFEEL_BM3D_QUEUES` — **gone** with the pre-R80 transfer and
  queue-selection code.

