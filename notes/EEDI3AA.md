# EEDI3AA — notes

Status: **shipped** — `core.vsfeel.EEDI3AA` (`Eedi3AaCreate`, `src/eedi3.cpp`), on the shared
`EEDI3`/`EEDI3H` argument string: the exact `based_aa` chain
`Merge(H(Merge(V(x))))` in **one plugin call, one submit**, both 50/50 merges
folded into the kernels. Runs on the R80 GPU API (`vnode:gpu` in/out, one exec
pool) and batches output frames per submission like `EEDI3` (see
`notes/EEDI3.md`); `field`, `mdis`, `nrad` and `vcheck` are unchanged. Bit-exact vs the two-call chain for u16, a few ulp for
f32. Specialised `dh=false`, `field>1`, single-rate. The `vsfeel/vsaa.py`
wrapper, the `eedi3aa` benchmark entry (1000 f, ns=8) and `tests/test_eedi3aa.py`
ship.

Scoreboard, real based_aa clip (jpbd 2x Point → 3840x2160 GRAY16, 600 f, ns=8):

| arm | fps |
|---|---|
| vsfeel `EEDI3AA` (fused, pre-port build) | 96.3 |
| vsfeel `EEDI3AA` (R80 port, interleaved A/B) | 159.2 vs 153.2 (**+4%**) |
| vsfeel two-call chain (order-reversed A/B, 6×600 f) | 98.9 (fused 100.2 → **1.01–1.07x**) |
| vsfeel `EEDI3AA` (batch knee re-swept, 4×1000 f order-reversed) | 183.1 → **198.7 (+8.5%)** |
| vszipcl chain (best reference) | 45.3 |
| vszipcu / eedi3vk2 chain | 25.8 / 28.1 |

**≈2.1x the best reference chain.** The brief's 1.6–1.8x projection did not
materialise (Historical).

## Implementation

### Semantics that must be reproduced exactly

Measured, not assumed — do not re-derive from intuition.

- **`std.Merge(a,b)` weight 0.5** (`.scratch/merge_semantics.py`, random GRAY16, 512
  samples): u16 `(a+b+1)>>1` (0 mismatches; `(a+b)>>1` 263 wrong, half-even
  131); f32 `0.5f*a + 0.5f*b` bitwise. The fused merge is **integer u16, after**
  the vcheck's quantisation.
- **Parity** (`Eedi3GetFrame`'s `field`, `Eedi3AaGetFrame`'s `base`/`fh0`):
  `field = d->field & 1`, overridden by the frame's `_FieldBased` (TOP→1,
  BOTTOM→0), then `(n&1) ^ field`. Vertical sub-frames take the input's
  `_FieldBased`; horizontal ones take `field & 1` with **no override**, because
  their input `v` is a progressive scratch region, not a frame
  (`Eedi3AaGetFrame`, the `fh0`/`fh1` comment) — wrong here is a silent
  one-parity-wide error.
- **Merge pairing:** output frame `k` pairs sub-frames `2k`/`2k+1`, both from
  input frame `k` (`Eedi3AaGetFrame`, one `Eedi3Job` per sub-pass per frame).
- **Props:** N frames at the input's fps, `_FieldBased` progressive; the fused
  filter must **not** halve `_DurationNum` the way each chained call does
  (`Eedi3GetFrame` halves it under `d->field > 1`; `Eedi3AaGetFrame` only sets
  `_FieldBased`).
- **Additive only:** EEDI3/EEDI3H and their tests stay untouched. Out of scope
  (falls back to the chain): `direction != BOTH`, `double_rate=False`,
  `transpose_first`, a `Deinterlacer` sclip, unsupported formats.

- `ENTRY_ASSEMBLEV` is the vertical merge (one dispatch per plane) and
  `ENTRY_COMPOSE` carries `comp_fuse` 1/2: sub-pass 0 parks its plane in the
  device-local `o0` region, sub-pass 1 reads it back and averages —
  `ENTRY_ASSEMBLEV`/`ENTRY_COMPOSE` in `src/eedi3.comp`, the `comp_fuse` arms in
  the compose entry point.
- `record_pass(d, cmd, jobs, njobs, phase)` over `PassPhase`
  `kPrep`/`kRow`/`kVcheck`/`kTail`; one `Eedi3Job` per frame per sub-pass
  (`Eedi3Job`/`PassPhase`/`record_pass`, `src/eedi3.cpp`).
- `Eedi3AaGetFrame`: one exec context and one CB for the whole batch. Four
  sub-pass groups run in order — vertical v0, vertical v1 (`kAssembleV`),
  horizontal h0, horizontal h1 (`kCompose`) — each `kPrep` → barrier → `kRow` →
  barrier → `kVcheck` → barrier → `kTail`, with a barrier between groups
  (`Eedi3AaGetFrame`'s four-sub-pass loop). All four share the frame's one
  scratch buffer (the `GpuBuffer scratch` in `Eedi3AaGetFrame`); the vertical
  merge writes the intermediate `v` into the scratch's `v` region and the
  horizontal pass reads it from there, so nothing is gathered to the host and
  there is no CPU blit.
- Output is `newGPUVideoFrame` when every plane is processed, else
  `newVideoFrame2` sharing the unprocessed planes from the source
  (`Eedi3AaGetFrame`).
- The scratch is filled with 0 once per submission so a pass reading a region it
  never wrote gets the benign value; `VSFEEL_EEDI3_NOCLEAR=1` opts out and
  `VSFEEL_EEDI3_POISON` overlays a chosen pattern (`eedi3_fill_scratch` /
  `eedi3_poison_scratch`, called from `Eedi3AaGetFrame`).
- `vsfeel_eedi3_create` `aa` mode: vertical geometry in `planes`, horizontal (the
  transpose) in `aplanes`, sized up front for the larger of the two; every region
  the two geometries share is placed once at `max(V, H)` and they alias it
  (`vsfeel_eedi3_create`'s `d->aa` / `aplanes` branch).
- `vsfeel/vsaa.py`: `EEDI3(vsaa.deinterlacers.EEDI3)` overrides `antialias` to
  emit one `EEDI3AA` call when a `_fusable_geometry`/`_fusable_format` check
  passes, else `super()`; `vsfeel.EEDI3` is a PEP-562 lazy re-export, so
  `import vsfeel` never needs vsaa.
- `tools/benchmark.py`: reference arms are the `vsaa` EEDI3 antialiaser itself
  (`should_h`/`supports_mclip`/the `Interleave([s,s])` sclip/`field = tff + 2`
  all come from vsaa, so they cannot drift); the fused arm calls
  `core.vsfeel.EEDI3AA` directly.

### Tests

`tests/test_eedi3aa.py` — **133 tests**: u16 exact oracle over field 2/3, mdis
3/20/40, nrad 0..3, vcheck 0..3, alpha/beta/gamma corners, Gray8/16/32 mclip
present/absent, aliased and distinct sclip; the same for f32 under a 1e-6 bound
(measured max 5.96e-8); `_FieldBased` progressive/TFF/BFF input; YUV420 all
planes, `planes=[0]` (chroma passthrough) and `planes=[1]` (chroma-only must take
the fused merge vs the chain); odd processed-plane width and odd subsampled
chroma width rejected; determinism, multi-stream and parallel load; props
(N frames, input fps, `_FieldBased` progressive) vs the chain; input validation.
`tests/test_python_backend.py` adds the wrapper cases (fused == chain, the
`direction != BOTH` fallback, backend selection, odd-geometry fallback,
forwarded parameters the plugin does not declare). Whole
suite **803 passed** via `tools/test.sh`.

## Performance

**Read the scoreboard with the screen quiet.** The frame is GPU bound, and an
active display (KWin compositing, or this page in a browser) holds
`gpu_busy_percent` at 12-18 even at idle, which comes straight off the top: the
same build and batch measured 192.5 fps with the monitor off and 171.3 with it on
(`notes/METHOD.md`). The scoreboard's numbers are screen-quiet.

`VSFEEL_EEDI3_GPUTIME=1` stamps every phase of every sub-pass inside the
submission's command buffer (it waits that submission out, so it serializes the
pipeline: a diagnostic, never a benchmark config). 2x2160p u16, mclip+sclip,
batch 4, per batch of ~3 frames:

| sub-pass | prep | row | vcheck | tail |
|---|---|---|---|---|
| v0 (vertical, parity 0) | 1971* | 3729 | 128 | - |
| v1 (vertical, parity 1) | 291 | 3625 | 132 | 172 (assembleV) |
| h0 (horizontal) | 409 | 3009 | 135 | 73 (compose) |
| h1 (horizontal) | 416 | 2989 | 135 | 178 (compose+fuse) |

\* v0's prep also carries the command buffer's prologue: the wait for the input
upload's producer pairs, which is PCIe time, not compute. The real prep is ~291
us per sub-pass.

So the **row kernel is 77% of the GPU time**, prep 18%, vcheck 3%, tail 2.4%; the
GPU is ~88% busy and the frame is GPU bound (removing the input uploads with
`--gpu-cache` does not help). Ablating the row kernel end to end takes the graded
run 172 -> 281 fps, i.e. **the row kernel is 39% of the frame**, which is where
the batch and row-kernel work below pays off. The kernel itself is attributed in
`notes/EEDI3.md`'s Performance section; the short version is that it is
ALU-issue bound at ~1 IPC, not load bound.

## Historical

- **2026-10-02 — the `oracle_32bit[case4-*]` failures were the EEDI3 core, not
  the AA chain.** They measured 0.002 against the 1e-6 oracle because the fused
  chain's f32 EEDI3 sub-passes lost DP argmins to missing `precise` qualifiers
  and incremental window sums; both are fixed in `src/eedi3.comp` and the case
  is back at 0. The premise (which pixels the distinct sclip leaves alone) was
  never content-specific.

**Cost model** (pre-R80; 2x-2160p, mdis=20, vcheck=2, ns=1, steady state, four
sub-passes / two submissions) from the `VSFEEL_EEDI3_GBENCH` profiler:

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
  `p.vcheck_lds = !p.vcheck_para && want_lds && d->have_lds && key.width <= MAXW_LDS`,
  `MAXW_LDS` (`src/eedi3.cpp`, from `-DEEDI3_MAXW_LDS`) — a **column count**, not
  bytes; 3840 ≤ 4096. `ENTRY_VCHECK` nonetheless defaults to the global-read
  (empty-row-skipping) form (`VSFEEL_EEDI3_VCLDS=1` restores the LDS ping-pong),
  and the two horizontal planes are 50/50-merged inside `ENTRY_COMPOSE` in VRAM
  (vcheck 0 falls back). Together, round 6: fused 89.7 → 108.8 fps (1200 f × 4
  order-reversed reps), bit-exact vs the old path (30 real 4K frames, 497 MB/arm
  byte-identical); `benchmark.py --filter eedi3aa` then measured ~121 fps at ns=8
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

- **2026-10-03 — the batch knee moves to 4 and the row kernel is priced.** Re-swept
  the batch under the harness's own RADV env (the transfer queue is what sets the
  knee): B=3 189 / **B=4 198** / B=5 197 / B=6 175 fps, so the auto rule's target
  went 512 → 768 MiB with the cap at `budget/12`. With the shared core's row-kernel
  work (notes/EEDI3.md), an order-reversed 4×1000 f same-session pair measures
  183.1 → **198.7 fps (+8.5%)**; 971 tests pass. The GPUTIME attribution (Performance)
  is new: it is what showed the row kernel at 77% of the GPU and the input-upload
  wait hiding inside v0's prep, i.e. that this filter is GPU bound where the
  vertical one is transfer bound.

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
- **Round 7 — correctness (pre-R80; the host staging path is gone).**
  `Eedi3AaGetFrame` never *invalidated* the two
  GPU-written staging regions it reads (merged `v`, the composed planes), only
  flushed them, so a non-coherent device would gather a stale `v`; mirrored
  EEDI3's invalidate per fence wait. No perf change; no-op here (133/133 with a
  forced-`coherent=false` build, zero VUIDs).
- **2026-09-25 — the per-submission scratch is cleared before any pass runs**
  (`src/eedi3.cpp`), so a pass reading a region it never wrote in that submission
  gets 0, the benign value for every flag in it, instead of recycled pool
  contents. Measured inert on healthy frames (no pixel changes); `..._NOCLEAR=1`
  restores the old behaviour. No perf change.
- **Round 8 — hardening (superseded by the R80 port).** The hand-built
  32-byte-aligned `VkMappedMemoryRange`s and the `deint_row_u16/f32`
  streaming-store predicate were removed with the upload/download path; no live
  code to preserve.
- **Superseded design brief**: its buffer-reuse map, implementation order, reuse
  map, testing plan, benchmarking plan and risk list are all realised in code.
  Its `VSFEEL_EEDI3AA_QUEUES` knob never existed, and neither does the
  `VSFEEL_EEDI3_QUEUES` cap it named — the core owns the one compute queue.
- **2026-09-29 — odd AA plane geometry rejected.** `a.rows = in_w / 2` left the
  last column of an odd processed plane width unwritten (the compose kernel
  stores only columns `2k`/`2k+1`), i.e. recycled VRAM every frame; the
  create-time parity guard now checks both axes for `d->aa`, and `vsfeel/vsaa.py`
  falls back to the chain for odd geometry. No perf change.
- **2026-09-29 — the fused 50/50 merge is per plane.** The fuse flag was read
  once from `planes[0].o0_bytes`, so a `planes` subset omitting plane 0
  (`planes=[1]`) dispatched `comp_fuse=0` and wrote one unmerged horizontal
  sub-pass; each plane now reads its own `cfg.o0_bytes`. Measured 1229 codes off
  the chain at 16-bit, `planes=[1]`, mdis=5, nrad=1, frame 0 of the noise clip,
  before the fix. No perf change.
- **2026-09-29 — the wrappers drop parameters the plugin does not declare.**
  vs-jetpack's EEDI3 dataclass forwards its own keys (`hp` at the time, which no
  vsfeel entry registered), and VapourSynth rejects unknown keywords before
  dispatch, so every wrapper path failed to build. `FeelBackend._dispatch` and
  the fused path now filter kwargs against the plugin's own `__signature__`
  (`vsfeel/backend.py`); the argument string later grew
  `hp`/`ucubic`/`cost3`/`opt` as accepted no-ops, so the filter is now defensive.
  No perf change.

## Open work

- **A parallel vcheck is the only path past 1.33x** (47% of the frame), and it
  changes the shared EEDI3 core, which would speed up EEDI3/EEDI3H by the same
  factor and leave the ratio where it is. The walk is sequential because each
  row's blend reads the *modified* previous row via `pr`; the untried fix is a
  width-independent form (previous row in registers via subgroup shuffles, or a
  tiled LDS window) — pure constant-factor, identical output, safe to land. Even
  a 2x vcheck is ~1.2x end-to-end; 1.5x needs it *plus* the row kernel.
- **Row kernel / `pbt`**: both named levers are measured dead (above). The row
  kernel is 39% of the graded frame, and its own attribution (`notes/EEDI3.md`,
  Performance) leaves exactly one large item: the ring shift register (PROBE=15,
  12.0% of the kernel). Reaching it needs the column loop unrolled by RN with a
  compile-time ring base, which is a hand unroll and a rotation of the `roll_seed`
  write indices.
- **Batch knee, re-measured under the harness's RADV env** (the transfer queue is
  what sets it): EEDI3AA 2x2160p B=3 189 / **B=4 198** / B=5 197 / B=6 175 fps.
  The auto rule's 512 MiB scratch target landed on 3, so it is 768 MiB now and the
  budget cap is `budget/12`; both are tuned constants, not invariants.

## Debug env vars

All are `VSFEEL_EEDI3_*`; there is no `..._EEDI3AA_*` namespacing. EEDI3AA reads
the shared core's knobs plus its own `..._NOCLEAR`/`..._POISON`.

- `..._BATCH=<n>` — output frames per submission (default from the scratch
  target, see `notes/EEDI3.md`); `=1` is the A/B control.
- `..._GPUTIME=1` — per-phase GPU timestamps inside the
  submission's own command buffer (serializes the pipeline; diagnostic only).
- `..._SKIP=<kernels>` — omit `mask,pad,row,vcheck,tail` from the
  recording, to price a kernel end to end. Changes the output.
- `..._VPARA=<n>` — vcheck form: 0 = serial row walk (A/B control), 1..6 =
  parallel with that many Jacobi steps (default 6).
- `..._VCLDS=1` — force the LDS vcheck ping-pong back (global reads are default).
- `..._NOCLEAR=1` — skip the per-submission scratch clear.
- `..._POISON=<hex>[:<region>]` — fill the scratch, or one region (`pad`, `dst`,
  `dmap`, `rempty`, `vout`, `o0`, `v`, …), with a pattern, so two runs differ
  exactly on the pixels that depend on unwritten scratch. Changes output by
  design.
- `..._TIMING=1` — per-frame host stage split (acquire/alloc/record/submit).
- `..._SYNC=1` — additionally wait each submission out and report its wall time.
- `..._TRACE=1` — one-shot creation banner: VRAM accounting, per-plane region
  layout.
- `EEDI3_PROBE` / `EEDI3_MAXW` — CMake cache vars: shader ablation level and the
  LDS vcheck max width (`-DMAXW`, `-DEEDI3_MAXW_LDS`); see `notes/EEDI3.md`.
