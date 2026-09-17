# EEDI3AA — notes

Status: **shipped** — `core.vsfeel.EEDI3AA` (`src/eedi3.cpp:4939`), on the shared
`EEDI3`/`EEDI3H` argument string: the exact `based_aa` chain
`Merge(H(Merge(V(x))))` in **one plugin call, two submits**, both 50/50 merges
folded into the kernels. Bit-exact vs the two-call chain for u16, a few ulp for
f32. Specialised `dh=false`, `field>1`, single-rate. The `vsfeel/vsaa.py`
wrapper, the `eedi3aa` benchmark entry (1000 f, ns=8) and `tests/test_eedi3aa.py`
ship.

Scoreboard, real based_aa clip (jpbd 2x Point → 3840x2160 GRAY16, 600 f, ns=8):

| arm | fps |
|---|---|
| vsfeel `EEDI3AA` (fused) | 96.3 |
| vsfeel two-call chain (order-reversed A/B, 6×600 f) | 98.9 (fused 100.2 → **1.01–1.07x**) |
| vszipcl chain (best reference) | 45.3 |
| vszipcu / eedi3vk2 chain | 25.8 / 28.1 |

**≈2.1x the best reference chain.** The brief's 1.6–1.8x projection did not
materialise (Historical).

## Semantics that must be reproduced exactly

Measured, not assumed — do not re-derive from intuition.

- **`std.Merge(a,b)` weight 0.5** (`tmp/merge_semantics.py`, random GRAY16, 512
  samples): u16 `(a+b+1)>>1` (0 mismatches; `(a+b)>>1` 263 wrong, half-even
  131); f32 `0.5f*a + 0.5f*b` bitwise. The fused merge is **integer u16, after**
  the vcheck's quantisation.
- **Parity** (`src/eedi3.cpp:2502-2512`): `field = d->field & 1`, overridden by
  the frame's `_FieldBased` (TOP→1, BOTTOM→0), then `(n&1) ^ field`. Vertical
  sub-frames take the input's `_FieldBased`; horizontal ones take `field & 1`
  with **no override**, because their input `v` is always
  `_FieldBased=PROGRESSIVE` (`src/eedi3.cpp:2776`) — wrong here is a silent
  one-parity-wide error.
- **Merge pairing:** output frame `k` pairs sub-frames `2k`/`2k+1`, both from
  input frame `k` (`sn = n/2`).
- **Props:** N frames at the input's fps, `_FieldBased` progressive; the fused
  filter must **not** halve `_DurationNum` the way each chained call does.
- **Additive only:** EEDI3/EEDI3H and their tests stay untouched. Out of scope
  (falls back to the chain): `direction != BOTH`, `double_rate=False`,
  `transpose_first`, a `Deinterlacer` sclip, unsupported formats.

## Implementation

- `ENTRY_ASSEMBLEV` (vertical merge, one dispatch per plane) and `ENTRY_COMPOSE`
  with `comp_fuse` 1/2, where sub-pass 0 parks its plane in device-local `o0`
  and sub-pass 1 reads it back and averages — `src/eedi3.comp:1491,1412`.
- `record_pass(planes, horiz, second, tail, direct)` + `PassTail` +
  `record_h2d_copy`: one CB records several sub-passes, so EEDI3/EEDI3H keep
  their old shape; it opens with a full compute barrier so pbt/dst/built-pad/R'
  are reused between sub-passes.
- `vsfeel_eedi3_create` `aa` mode: vertical geometry in `planes`, horizontal in
  `aplanes`; byte-identical regions (raw/sclip/dst/vout/pbt/dmap/cint) shared,
  only the differing ones doubled (raw2/sclip2/bits2/dst2/vout2/out2/v).
- `Eedi3AaGetFrame` (`src/eedi3.cpp:2437`): CB1 = vertical sub-pass 1 (tail
  none) + 2 (assemble-v); host gathers `v` for both horizontal sub-passes; CB2 =
  horizontal sub-pass 1 + 2 (compose, VRAM-fused); CPU does a plain row blit.
- **Two extra descriptor sets** (AA needs both geometries' views at once, which
  `d->horiz` cannot express): `desc_set_h` for horizontal row/vcheck/compose
  (b0/b5/b9 = pad_dev) and `desc_set_xp` for xpose + the horizontal pad builder
  (b0 = upload, b8/b9 = pad_dev) — `src/eedi3.cpp:864-873`. Aligning them was
  the biggest bug: with the wrong set, b9 was the upload buffer while the code
  used pad_dev-relative offsets, so under ReBAR the horizontal stage read/wrote
  out of bounds (all-zero output); non-ReBAR masked it, which is why
  `VSFEEL_EEDI3_NOREBAR=1` was bit-exact and the default was not.
- `vsfeel/vsaa.py`: `EEDI3(vsaa.deinterlacers.EEDI3)` overrides `antialias` to
  emit one `EEDI3AA` call, else `super()`; `vsfeel.EEDI3` is a PEP-562 lazy
  re-export, so `import vsfeel` never needs vsaa.
- `benchmark/bench.py`: reference arms are the `vsaa` EEDI3 antialiaser itself
  (`should_h`/`supports_mclip`/the `Interleave([s,s])` sclip/`field = tff + 2`
  all come from vsaa, so they cannot drift); the fused arm calls
  `core.vsfeel.EEDI3AA` directly.

## Historical

**Cost model** (2x-2160p, mdis=20, vcheck=2, ns=1, steady state, four sub-passes
/ two submissions) from the `VSFEEL_EEDI3_GBENCH` profiler:

| stage | x4 / frame | share |
|---|---|---|
| vcheck (serial row walk) | ~11 ms | **47%** |
| row kernel (DP + backtrack + interpolate) | ~9.7 ms | **42%** |
| compose / assemble-v | 1.87 ms | 7.6% |
| xpose + pad builder | ~0.5 ms | <2% |
| **total** | **~23.2 ms** | |

`gpu_busy_percent` polls 100 for an ns=8 run and the wall matches the total: the
frame **is** GPU work. `timestampPeriod` is 10 ns/tick on Navi31 (assuming 1.0
under-reported every stage 10x).

- **1.33x is the ceiling for vcheck-side work.** `VSFEEL_EEDI3_NOVC` (deletes
  vcheck + vcopy) is the only large lever: 84 → 111 fps at ns=8 on the 2x-2160p
  synthetic; the other 53% is the EEDI3 algorithm. Round 2's host-split
  conclusion was wrong — "58 ms of 72 ms is fence waits" is uninterpretable at
  ns=8 (those waits include the other seven streams); the truth is GPU-bound.
- **`NOPAD`/`NOXPOSE` are worth ~0%** (ns=1: 30.86 → 31.01 → 30.98 fps); an
  earlier +15%/+17% quote was clock noise. Do not spend anything there.
- **V-then-H is a data dependency, not a choice**: H's cost windows are computed
  on the vertically interpolated values, and fusing both DPs into one kernel is
  not expressible (H needs the vertical walk's committed result over a
  neighbourhood while that walk is column-sequential).
- **The two sub-passes within a direction are independent** (V0 even rows, V1
  odd, same frame, results only merged), serialized **only** by shared scratch
  (`pad_dev`, pbt, R'). Giving each its own would overlap them but would **not**
  raise ns=8 throughput — the GPU is saturated; it is an ns=1 latency win.
  Streams already reach that ceiling: ns=4..32 measured 75.9 / 87.4 / 88.6 /
  90.5 / 90.6 / 90.4 fps (flat from 8 up).
- **Row kernel is ~25%, not 35%.** Real based_aa clip, 2000 f, ns=8, 5
  interleaved reps (within-arm spread 1.3%): fused 98.2 → 150.4 fps with the row
  kernel deleted, but that arm leaves `dmap` stale so vcheck takes its cheap
  `dirc == 0` branch; the faithful ablation (`PROBE=12`, real dmap) is ~1.25x.
- **The LDS vcheck IS engaged at the benchmark width**: gate is
  `use_lds = lds_ok && key.width <= MAXW_LDS`, `MAXW_LDS = 4096`
  (`src/eedi3.cpp:758,4322`) — a **column count**, not bytes; 3840 ≤ 4096.
  `ENTRY_VCHECK` nonetheless defaults to the global-read (empty-row-skipping)
  form (`VSFEEL_EEDI3_VCLDS=1` restores the LDS ping-pong), and the two
  horizontal planes are 50/50-merged inside `ENTRY_COMPOSE` in VRAM
  (`VSFEEL_EEDI3_AATIGHT=0` restores the two-plane + CPU form; vcheck 0 falls
  back). Together, round 6: fused 89.7 → 108.8 fps (1200 f × 4 order-reversed
  reps), bit-exact vs the old path (30 real 4K frames, 497 MB/arm
  byte-identical); `bench.py --filter eedi3aa` then measured ~121 fps at ns=8
  (vszipcl 48.6, eedi3vk2 36.2).
- **Dead end — GPU K compaction (`ENTRY_ASSEMBLEK`).** One kernel merging
  src+vout *and* emitting both compacted column matrices was **bit-exact for
  luma but left every chroma plane untouched by the dispatch**, in ReBAR and
  non-ReBAR alike; ramp/constant probes showed plane 0 written and planes 1/2
  never written, and forcing plane 0's pipeline, reversing dispatch order,
  disabling the pipeline cache and enabling validation changed nothing. Never
  diagnosed; reverted. If retried, first prove the dispatch runs for a *second*
  plane (a 5-line ramp probe settles it).
- **Dead end — CPU merge / DSTHOST.** CPU merge of the kept columns (compose
  writes interp-only, CPU interleaves): **−4.8%** over 8 order-reversed reps
  (385.5 → 366.9 fps on EEDI3H), because the extra 16.6 MB source re-read +
  merge ALU costs more than the 8.3 MB PCIe + 8.3 MB VRAM it saves.
  Direct-to-frame host-pointer import (`VSFEEL_EEDI3_DSTHOST=1`): 534 → 216.6 fps
  (**−59%**) — the per-frame import/destroy is a global VM cost a pointer cache
  would not remove.
- **Dead end — mask gather / pbt / vcheck.** `gather_mask_bitmat` k-outermost
  order: 64 concurrent row streams defeat the prefetcher, ~2× slower than the
  y-outermost 16-row group. The mask gather's ALU is not where the time is
  (16.6 MB of mask reads is inherent; the byte transpose is ~0.3 ms/frame at
  ns=1 vs ~0.8 ms of read). 2-bit `pbt` packing and breaking the walk's serial
  load chain are both measured dead on the current kernel (`notes/EEDI3.md`
  round 20); `PROBE=4` ("no pbt store") is not a valid pbt probe — it
  short-circuits the whole row kernel (v_row 2.9 → 0.005 ms), ablating the DP
  rather than the stores. A parallel vcheck reading the *un*modified previous
  row is rejected: it changes the output and the AA suite is a bit-exact oracle.

### Round history

- **Round 2 — implemented**, 585/585 tests green. The 1.6–1.8x projection
  failed: worth 1.0–1.07x over the vsfeel chain, only the two `std.Merge` nodes
  and the intermediate frame materialisation (~9%).
- **Round 3 — benchmark fidelity.** The arms were a hand-rolled based_aa
  reimplementation assuming the opposite of the plugins' real capabilities:
  vszipcl and vszipcu both register `EEDI3H`, support no mclip; eedi3vk2 has no
  EEDI3H but does support mclip. Noise clip (640x360 → 2x Point, 16 frames,
  ns=8): **eedi3vk2 129.6 fps without mclip vs 190.2 with it (~+47%)**. Fixed
  by driving the arms through the `vsaa` antialiaser itself (145 tests passed).
- **Rounds 4/5 — cost model, then corrections.** The profiler gave the table
  above and superseded round 2; it was never committed, and "the LDS vcheck is
  off at 3840" was wrong (see the `MAXW_LDS` bullet).
- **Round 7 — correctness.** `Eedi3AaGetFrame` never *invalidated* the two
  GPU-written staging regions it reads (merged `v`, the composed planes), only
  flushed them, so a non-coherent device would gather a stale `v`; mirrored
  EEDI3's invalidate per fence wait. No perf change; no-op here (133/133 with a
  forced-`coherent=false` build, zero VUIDs).
- **Round 8 — hardening.** The shared `mapped_range`/`flush_range`/
  `invalidate_range` helpers replaced all hand-built 32-byte-aligned
  `VkMappedMemoryRange`s (`src/vsfeel.h:196`), and the shared
  `deint_row_u16/f32` streaming-store predicate now tests the pointer, not the
  element index — `aa_gather_horizontal` inherited both.
- **Superseded design brief**: its buffer-reuse map, implementation order, reuse
  map, testing plan, benchmarking plan and risk list are all realised in code,
  and its `VSFEEL_EEDI3AA_QUEUES` knob never existed (the cap is
  `VSFEEL_EEDI3_QUEUES`).

## Open work

- **A parallel vcheck is the only path past 1.33x** (47% of the frame), and it
  changes the shared EEDI3 core, which would speed up EEDI3/EEDI3H by the same
  factor and leave the ratio where it is. The walk is sequential because each
  row's blend reads the *modified* previous row via `pr`; the untried fix is a
  width-independent form (previous row in registers via subgroup shuffles, or a
  tiled LDS window) — pure constant-factor, identical output, safe to land. Even
  a 2x vcheck is ~1.2x end-to-end; 1.5x needs it *plus* the row kernel.
- **Row kernel / `pbt`**: both named levers are measured dead (above); the one
  remaining structural idea is a walk dispatch with one lane per row (~5–9%
  projected, not attempted). `pbt` is `W*rows*tpitch` int8 = 170 MB per sub-pass
  at 4K, written and re-read by the backtrack — any experiment needs its own
  probe, not `PROBE=4`.
- **No GPU stage profiler in the tree**; restoring it is the prerequisite for
  attributing anything. `_NOVC`/`_NOXPOSE`/`_NOCOMPOSE`/`_NOMASKX` and
  `_PTRTRACE` are diagnostics slated for deletion before landing.

## Debug env vars

All are `VSFEEL_EEDI3_*`; there is no `..._EEDI3AA_*` namespacing, and the queue
cap is `VSFEEL_EEDI3_QUEUES`, not `..._EEDI3AA_QUEUES`.

- `..._AA_HBENCH` + `..._AA_HFRAME=<n>` — host-stage split (vGather / vRec /
  vWait / hGather / hRec / hWait / merge / total), shape of EEDI3's `_HBENCH`.
- `..._GBENCH` — GPU stage timestamps (one query pool per submission; a shared
  pool never becomes available), read host-side after the fence; not in the tree.
- `..._QUEUES=N` queue cap; `..._NOREBAR=1` force the H2D path; `..._VOUTDEV` /
  `..._COPY` the vout DMA form and non-temporal load/store bits (bit0 upload
  gathers, bit1 final blit); `..._AATIGHT=0` two-plane + CPU merged compose;
  `..._VCLDS=1` LDS ping-pong vcheck; `..._MASKFUSE=0`/`..._PAIR=0` the shared
  EEDI3H mask-fuse and aliased-pair-gather ablations, applied to the AA gathers.
- Shared ablation knobs the AA path honours: `_NORAW`, `_NOSCLIP`, `_NOVC`,
  `_NOXPOSE`, `_NOCOMPOSE`, `_NOMASKX`, `_NOPAD`, `_NOH2D`, `_NOXFER`, `_PADPAR`,
  `_RAWSTAGE`, `_BLITCONTIG`, `_DSTHOST` (`_NOBLIT` is rejected); `_PTRTRACE` and
  `_TRACE` are diagnostics.

## Tests

`tests/test_eedi3aa.py` — **133 tests**: u16 exact oracle over field 2/3, mdis
3/20/40, nrad 0..3, vcheck 0..3, alpha/beta/gamma corners, Gray8/16/32 mclip
present/absent, aliased and distinct sclip; the same for f32 under a 1e-6 bound
(measured max 5.96e-8); `_FieldBased` progressive/TFF/BFF input; YUV420 all
planes and `planes=[0]`; determinism, multi-stream and parallel load; props
(N frames, input fps, `_FieldBased` progressive) vs the chain; input validation.
`tests/test_python_backend.py` adds 3 wrapper tests. Whole suite **603 passed**;
collected EEDI3-family total **214** (`test_eedi3.py` 40 + `test_eedi3h.py` 11 +
133 + the wrapper group).
