# BM3Dv2 — notes

**shipped.** Vulkan port of vszipcl's BM3D (fused mode: one estimation pass per
output frame, temporal aggregation over the stack window). Numerically faithful
to vszipcl; see the tolerance policy in `tests/test_bm3dv2.py`.

Current design:

- Two kernels: `bm3d.comp` (block match + group + collaborative transform,
  one warp of 32 lanes = 4 sub-groups of 8 lanes, one 8x8 block each) and
  `bm3d_agg.comp` (temporal aggregation over the TW = 2r+1 stack slices).
- Estimation accumulates with hardware buffer float atomics where the device
  has them (`VK_EXT_shader_atomic_float`; RADV gates it at GFX11). Everywhere
  else the same kernel's `-DNO_FLOAT_ATOMICS` build runs the reference's own
  `atom_add_f` CAS loop; the host picks per device, `VSFEEL_BM3D_CAS=1` forces
  it (see Historical for the cost).
- The spatial search scans the (2*bm_range+1)² window with a per-lane row
  partition and a **sliding column-SSD window**; each sub-group lane keeps its
  own top-8, which `merge_group` 8-way-merges.
- Each temporal direction/t step scans PS_NUM PS_RANGE windows around the
  previous step's matches. **Its per-lane list is only PS_NUM deep** — the merge
  only ever consumes that many (`merge_group(PS_NUM, …)`) and only the first
  PS_NUM entries are read back as the next centres, which is a provable
  equivalence and the single largest win in this file (below).
- The scan body is unrolled four candidates wide; the four new column loads
  issue before any reduce/insert consumes them.
- Degenerate paths: `sigma < FLT_EPSILON` passes the plane through (a source
  copy), like the installed references' `PROC_MASK`.
- Per-frame resources live in `FramePool<Bm3dStream>`; cross-frame handoff is
  per-stream timeline semaphores plus `res_holders` reservation tokens.
- Upload staging is host-visible VRAM (ReBAR) on the default path
  (`VSFEEL_BM3D_HD=0` opts out), sized for the 4r+1 slots a record actually
  uploads.
- `num_streams` default **2**: the GPU saturates at two in-flight frames.

Performance — 1080p GRAY32, jpbd, `tools/benchmark.py -f bm3dv2 vsfeel vszipcl`,
1000 frames × 3 interleaved, sigma 0.7, radius 2, bm_range 16, ps_range 7,
block_step 4:

| | fps | GPU est. kernel | GPU agg | VRAM (r=2) |
|---|---|---|---|---|
| vsfeel | **313.7** (311-316 over runs; 313.6 at ns=4) | 2.86 ms | 0.33 ms | 1107 MiB |
| vszipcl | 46.5 | — | — | — |

Before this round the same command gave 190.5 fps at ns=4 (kernel 4.94 + agg
0.33 ms). Speedup over vszipcl went 4.6x → 6.7x, absolute fps +65%.

## Implementation notes that the code alone does not show

- **The block-matching search is the whole filter.** At r=2/step 4 it is ~85%
  of the estimation kernel; the collaborative transform + patch loads + all
  133.8 M float atomics together are only ~0.4 ms/frame, and the zero-fill plus
  the ring copies are ~0.1 ms. Nothing else is worth tuning (see the ablation
  table under Historical).
- **The aggregation is PCIe-bound, not work-bound.** Its time is ~0.32 ms at
  TW = 1, 5 and 9 alike; the 8.3 MB result store goes to GTT over PCIe at
  ~26 GB/s. Vectorising it to `vec4` (one thread per 4 columns) changes
  nothing, so it is left vectorised purely for the lower instruction count.
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

- **2026-09-16 — creation safety.** Reject an estimate stack past the kernel's
  signed 32-bit `res` addressing (`res_cap * tw * 2 * pe`; 4K r=4 ns=4 is
  11.7 GiB), require a known positive frame count, and flush `NOSEARCH`'s
  synthetic group instead of letting the aggregation read uninitialized LDS
  (`test_bm3dv2_nosearch_matches_search_on_constant_clip`).
- **2026-09-16 — parameter validation.** `bm_range`/`ps_range` [1, 8192]
  (int32 `(2r+1)²`), `extractor_exp` [-126, 127], `num_streams` 1..32, `sigma`
  rejects NaN/inf, `rpGeneral` whenever `radius > 0`.
- **2026-09-16 — same-queue timeline ordering.** A reader device-waits on the
  timelines of the streams that filled its result slots, and on a shared queue
  that wait can sit in the FIFO ahead of the signal RADV will not run past; each
  frame now publishes its estimation *seq* under `cache_lock` and the reader
  host-waits for the writers' **submission** before submitting its aggregation
  (`test_bm3dv2_seek_collision_single_queue`).
- **2026-09-16 — fp32 aggregation, sigma skip, gputrace gating.** `bm3d_agg`
  divided in `double` (1/16 rate, hard `shaderFloat64` need); `sigma[0] <
  FLT_EPSILON` passes the plane through, which removes the 0/0 Wiener NaN;
  `gpu_trace` cached at creation.
- **2026-09-16 — frame error path.** A failed frame handed its stream back with
  GPU work in flight, so a successor could re-record the buffers and reuse slots
  the running kernels read; the error path now drains the queue and resets the
  fence.

**2026-09-19 — ReBAR upload staging (+4.6%).** Per-stream staging moved from
GTT to `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`, so the CPU writes VRAM and
the ring copy is a device-local read; the store form flips with the allocation
(NT stores for GTT, cached stores for the write-combined window, no flush for
the coherent one). 178.2 vs 170.3 fps at 1000f 1080p GRAY r=2 ns=4, 3
interleaved pairs.

**2026-09-19 — subgroup shuffles for the 8-lane exchanges (+6%).**
`transpose_pack8` and `reduce_group8` became register-only `subgroupShuffleXor`
butterflies. The barriers were already free (a 32-lane workgroup is one wave, so
all 64 `barrier()`s compiled away); the cost was the LDS round-trip. 179.0 →
190.9 fps at r=2 ns=4. VGPR 256 → 216, LDS 6144 → 4096 B, subgroups/SIMD 5 → 7.
Shuffle masks are < 8, so this stays correct on a wave64 device.

**2026-09-20 — queue cap stays uncapped.** `VSFEEL_BM3D_QUEUES` {1,2,4} ×
`num_streams` {4,8} at 2 and 4 concurrent requests: uncapped is best or tied in
every cell. Unlike Bilateral (+16% at cap 2) BM3D is not queue-starved — ~88%
of the frame is fence, so a shared queue has no drain bubble to fill.
Re-measured on the current binary: 309.9 / 313.0 / 313.1 / 312.4 fps for cap
1 / 2 / 4 / uncapped. Unchanged.

**`extractor_exp` was a silent no-op — fixed.** Host wiring was fine and the
SPIR-V kept `(x + E) - E`, but RADV folds that pair once E is a known spec
constant (`RADV_DEBUG=asm` was instruction-identical for E=0 and E=20). The
pair now computes under `precise` inside `if (EXTRACTOR != 0.0)`; the `if`
folds at pipeline creation, so E=0's ISA is unchanged. Rule: a
spec-constant-guarded `(x + E) - E` idiom needs `precise` or RADV folds it.

**Slot-direct cache consumption was already in place.** Both caches are read in
place: `bm3d.comp` indexes the source ring through `slot_base(z)`, and
`bm3d_agg.comp` reads the estimate stacks through the per-slice push-constant
`bases[9]`. The only D2D copy in the filter is `staging -> src_buf`, the upload
leg.

## Round: the block-match scan (2026-09-21, +64% end to end)

Everything above is a whole-kernel time from a warm single-request
`VSFEEL_BM3D_GPUTRACE=1` run (`-r 1`, 300 frames, 1080p GRAY32 defaults), which
settles to ±0.2% on repeat. Kernel ms is the metric for kernel work: the fence
is depth-invariant (5.4 → 5.6 ms over radius 1..8) and fps is flat from ns=2 to
ns=8, so the GPU is the wall. Graded fps is quoted separately.

1. **Four-wide sliding scan (`scan_row`) — 4.943 → 3.96 ms.** A rolled loop
   leaves the next candidate's eight loads behind the previous candidate's
   ~60-instruction insert, and every wave stalls on the same `s_waitcnt` at the
   same time. Issuing four candidates' new-column loads before any of them is
   consumed fixes that: 2-wide reached 4.168, 4-wide 3.963, 8-wide 4.093
   (register pressure starts to bite). Same operations in the same order, so
   bit-identical.
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
frame needs the stacks of centre frames `[n-r, n+r]`, so `num_streams`
concurrent frames span `num_streams + 2r` slots — that is `res_cap` now. The
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
AMD-RDNA3-only).** RADV gates `shaderBufferFloat32AtomicAdd` at GFX11
(`radv_physical_device.c`) and the Windows driver has no such feature on
Polaris, so BM3D refused creation — RX 580, Vega, RDNA1/2 alike. `bm3d.comp`
gained a `-DNO_FLOAT_ATOMICS` build (`bm3d_cas.spv`, auto-selected, forced with
`VSFEEL_BM3D_CAS=1`) whose aggregate stores run the CAS loop zipcl's own
`atom_add_f` has always used: core `atomicCompSwap` only, so nothing about it is
driver-specific, and one fp32 add per round keeps it bit-identical to the
hardware path at `extractor_exp=8` (75/75 BM3D tests either way). Cost, 1000f
1080p GRAY32 r=2 ns=2 interleaved pairs: 314.8/312.5/314.9 (hardware) vs
260.0/260.4/259.8 (CAS) fps, +0.66 ms/frame or ~17%.

**2026-09-20 — the estimate cache now sizes to its working set (default), not
the working set plus a window of seek slack.** `res_cap` was
`tw + ns + 2r` while the in-flight working set is `ns + 2r`, so every instance
allocated `tw` slots for a margin only out-of-order requests consume; radius 4
on an 8 GiB card failed to allocate because of them. Dropping the margin is
-36..41% of total VRAM at no in-order cost, so it became the default;
`VSFEEL_BM3D_CACHE=1` restores it for seek/scrub-heavy graphs, where a warm
slot beats blocking in the acquire. Numbers in the VRAM paragraph.

**2026-09-20 — the ReBAR probe now tests the heap, not the memory type.**
`DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` also exists without Resizable BAR,
backed by the PCIe aperture (typically 256 MiB) instead of VRAM, so staging
taken from it failed with `VK_ERROR_OUT_OF_DEVICE_MEMORY` while VRAM was empty
(RX 580/Windows, 1080p r=2, both passes). `rebar_available(dev, bytes)` gates on
the backing heap size, then probes the whole per-instance staging once before
the stream loop and falls back to GTT; no change on the DB machine (type 3 is in
the 24 GiB device-local heap).

## Open work

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

Standardised on `VSFEEL_BM3D_<FLAG>`; the pre-standardisation `BM3D_<FLAG>`
spelling still works for one release (new name wins). All route through the
`env_flag`/`env_int`/`env_str` helpers in `vsfeel.h`; `TRACE` and `DUMP` are
cached at creation, not read per frame.

- `VSFEEL_BM3D_TRACE=1` — acquire/submit/wait trace.
- `VSFEEL_BM3D_TIMING=1` — host-stage split per frame (cached at creation).
- `VSFEEL_BM3D_VRAM=1` — creation-time VRAM budget (shared + per-stream).
- `VSFEEL_BM3D_QUEUES=N` — queue cap override.
- `VSFEEL_BM3D_HD=0` — force the GTT staging + PCIe upload.
- `VSFEEL_BM3D_CACHE=1` — add the seek margin back to the estimate cache
  (0, the default, sizes it for the working set; see the VRAM paragraph for the
  +36..41%).
- `VSFEEL_BM3D_CAS=1` — force the atomicCompSwap aggregation build on a device
  that has buffer float atomics (A/B measurement only).
- `VSFEEL_BM3D_NOSEARCH=1` / `VSFEEL_BM3D_NOESTIMATE=1` — ablation knobs; both
  are vsfeel inventions, not reference behaviour, and NOSEARCH distorts the
  temporal search as well (see the ablation note).
- `VSFEEL_BM3D_DUMP=1` / `VSFEEL_BM3D_GPUTRACE=1` — slot dump / GPU timestamps.
