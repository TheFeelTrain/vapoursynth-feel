# DFTTest — notes

Status: **shipped on the R80 GPU API.** Verified against `src/dfttest.{cpp,comp}`.
Target GPU: RX 7900 XTX (RDNA3, gfx1100), Mesa 26.2 RADV.

- `clip:vnode:gpu` in, `ffGPUOutput` out: the core owns every transfer and the
  filter records one submission per output frame into one exec pool (no
  filter-owned staging, timelines, fences or descriptor sets).
- u16 / f16 / f32 IO. The half path stages native `float16_t` in the padded
  plane and widens for the f32 work; half is a storage width, not a precision
  (Implementation).
- Per output frame and processed plane: one `pad` dispatch writes every temporal
  slice, then per stripe of block rows `fused` → barrier → `col2im` straight
  into the output plane. The stripe ring keeps the block handoff in L2
  (Performance). Scratch is two `createGPUBuffer` allocations per frame, handed
  to the context.
- **Runs need `RADV_EXPERIMENTAL=transfer_queue`** (mechanism:
  `notes/BILATERAL.md`), which `tools/benchmark.py` forces.

Scoreboard — jpbd 1080p GRAY16, 3000 frames, medians of 3 through
`tools/benchmark.py --filter dfttest`:

| mode | vsfeel | vszipcl | ratio |
|---|---|---|---|
| `--cpu-cache --download` (CPU cache) | 1984.9 | 887.4 | 2.24x |
| `--download` (resident GPU input) | 2013.6 | 851.1 | 2.37x |
| `--synthetic` BlankClip | 2001.8 | 895.0 | 2.24x |

IO depth, jpbd 1080p, 3000 frames, medians of 3, one session
(`--format u16|f16|f32`; the f16 clip carries the u16 clip's content):

| io | vsfeel | vszipcl |
|---|---|---|
| u16 | 1984.9 | 887.4 |
| f16 | 1997.6 | no f16 path |
| f32 | 1221.7 | 605.3 |

- Neither reference implements an f16 DFTTest, so the half path is graded
  against the f32 path on the losslessly widened clip, not against vszipcl.

- The three modes agree within 1.5%: the frame is the kernels, and with the
  transfer queue the transfers ride along instead of adding a stage.
- Temporal radius, same session and medians of 3: 2566.2 fps at tbsize 1,
  1984.9 at 3, 1328.6 at 5, 910.6 at 7. That is 67% and 46% of the tbsize-3
  number where vszipcl keeps 75% and 34% (Implementation has the cube layout
  that bought it).

## Implementation

### Frame path (`dft_gpu_frame`, `src/dfttest.cpp`)

- `arInitial` requests the `2*radius+1` source frames; `arAllFramesReady` takes
  them, makes the output frame (`newGPUVideoFrame` when every plane is
  processed, `newVideoFrame2` sharing the unprocessed planes from the centre
  frame otherwise) and acquires one exec context.
- One recording per output frame, per processed plane: `pad` (one dispatch,
  every temporal slice, reading the tw source frame planes through a descriptor
  array) → barrier → per stripe of block rows `fused` → barrier → `col2im`
  (straight into the output frame plane) → barrier. Then `gpuExecReadsFrame`
  per distinct source, `gpuExecWritesPlane` per processed plane,
  `gpuExecSubmit`.
- Per-frame scratch is two `createGPUBuffer` allocations per plane — the padded
  window (`tw*pw*ph*bytes`) and the block-buffer ring (`spatial_rows*hn*256*4`,
  63 MB at 1080p) — handed to the context with `gpuExecUsesBuffer`.
  Allocation is measurably free (aliasing the ring onto the padded window
  changes the frame time by <1%). Depth is the core's exec ring (2..8
  contexts), not `num_streams`.
- Push descriptors with seven bindings: `wt` (scalar view), `padded`, `spatial`,
  `dst`, the tw source planes as one descriptor array (binding 4, pad only), and
  two vector views of `wt` (5 = vec2 for the window_freq pairs, 6 = vec4 for the
  transposed window). Binding 4's descriptor count is the widest compiled
  variant (7), not the instance's tw, because every radius's pipelines are
  created against the one layout.
- `num_streams` and `device_id` are registered no-ops (never read): depth is
  the core's call, device choice is `core.set_vulkan_device`.

### The block-row ring (`dft_plan_spatial`)

- A full-frame block buffer is `num_blocks*256*4` bytes, 136 MB at 1080p:
  `fused` writes it and `col2im` reads it back every frame, and it does not fit
  the 96 MB Infinity Cache, so it streams through VRAM. That traffic is the
  filter's largest single cost (aliasing the buffer to 1 MB: +32%).
- Block rows are processed in stripes and the handoff stays in a ring of
  `spatial_rows` rows indexed by `block row & (spatial_rows-1)`: the largest
  power of two whose bytes fit `kSpatialRowBudget` (64 MiB) and at least
  `lookback+1`. `lookback = (16-block_step)/block_step` is how far back
  `col2im`'s window reaches into the previous stripe; `stripe_rows =
  spatial_rows - lookback` (61 rows = 3 stripes at 1080p).
- Per stripe: `fused` writes rows `[row_base, row_end)`, `col2im` emits the
  output rows whose block support is complete (padded `y < row_end*block_step`)
  and the rows left over by the previous stripe are still in the ring, then the
  trailing barrier protects the slots the next `fused` overwrites
  (`spatial_rows` is exactly `stripe_rows + lookback`).
- `vn <= spatial_rows` is one stripe and behaves like a full-frame layout, so
  small frames pay neither the extra dispatches nor the barriers.

### Kernels (`src/dfttest.comp`)

- `ENTRY_PAD` writes every temporal slice of the padded plane in one dispatch:
  each invocation folds its (x, y) once with `n-1 - abs(n-1 - abs(v))` and
  issues tw loads/stores, slice t through `src_bufs[t]` with the compile-time
  index the unroll produces.
- `ENTRY_FUSED` is im2col + window + 3D DFT + frequency filter + inverse,
  writing the centre temporal slice of each block into the ring; it loads the
  window four j at a time from the transposed copy (`vec4`) and window_freq one
  `(re, im)` pair at a time (`vec2`).
- The transformed cube is 9 x-bins x 16 y-bins x tw slices, and every one of
  the 16 lanes holds 9 of the 144 bins: lanes 0..8 keep their own x-bin's
  y = 0..8 (`x = lane, y = k`), lanes 9..15 take one whole y-bin each
  (`y = lane, x = k`), so a lane's cube is `18*tw` floats and the bin's table
  slot is `y*9 + x` under a single select (no division). The y-bins the workers
  do not keep are pooled through the transpose buffer once per slice, which
  costs 2 LDS round trips per slice. See Performance for why 9 bins per lane
  instead of 16 per 9 lanes.
- `ENTRY_COL2IM` overlap-adds into the output plane with its stride, reading the
  window from the scalar table (its offsets are per pixel, so nothing
  vectorizes there).
- `RADIUS` (0..3) is a compile-time define so the temporal loops unroll and
  `td[]` stays in registers; `FILTER_TYPE` and `ZMEAN` are specialization
  constants (the dead filter branches and their divisions vanish); wave32 is
  requested. `SUB_BLOCKS=8` (128-thread workgroups) and the 288-float
  bank-conflict-free transpose window (10 240 B LDS) remain the sweep winners.
- Each 16-lane tile exchanges `trans` across `subgroupBarrier()`, whose scope is
  **one subgroup**, so the size in use has to be a multiple of 16 — wave32 is
  requested when the device offers it, and when it does not, the driver's default
  is accepted only if it is a multiple of 16. Otherwise creation fails with that
  reason instead of running tiles that span two subgroups.
- FFTW codelets, window/sigma table math, filter formulas and the reflect-pad
  rule are unchanged from the ported reference; only the buffers the data comes
  from and goes to changed.
- 27 SPIR-V blobs (`col2im` × 3 depths, `pad_r0..3` × 3, `fused_r0..3` × 3):
  pad and fused are per radius because pad's slice count and fused's temporal
  unroll are both compile-time.

### f16 IO (`IO_F16`, `src/dfttest.comp`)

- The pad plane stages native `float16_t` (2 bytes, half the padded bytes) and
  the fused kernel widens it. Widening a half is exact and the `*255` domain
  scale is exact in f32 too (an 11-bit mantissa shifted by 8), so the fused
  values are the f32 path's on the widened clip; `spatial` stays f32, so the
  only rounding the half path adds is the col2im store's round-to-nearest-even.
- Graded against the f32 path on the losslessly widened clip (Point GRAYH →
  GRAYS is a container change): worst 0.95 half ulp relative to the output
  magnitude over the whole parameter surface, and ≥ 99.99% of samples
  bit-identical to the f32 result rounded to half (`tests/test_dfttest.py`).
  The last-bit f32 difference between the two SPIR-V modules comes from the
  half → float widening being contracted into a mixed-precision fma, the same
  residue `notes/GAUSSBLUR.md` documents.
- The half path selects different blobs, so `DftCreate` keys the variant on
  `half` rather than `bits` (both are 16). `tests/test_dfttest.py` also adds a
  630-px cropped half case (1260-byte rows, 630-byte chroma rows) and
  `tests/test_validation.py` a `DFTTest_f16` validation run.

### What the port deleted

`SlotState`/`slot_lock`/`frame_gen`, the per-slot timeline semaphores, the
ReBAR host-direct upload buffer, staging + download buffers, the `FramePool` of
per-stream resources, per-resource fences, the pad command-buffer ring, the
`pad_slot`/`pad_direct` split, the `FUSED_DIRECT` pair, and the dump-pad /
force-pad / fail-pad probes.

## Performance

- **The L2-resident block handoff is the win: +24% at the default config**
  (jpbd 1080p GRAY16, 3000 frames, medians of 3, one session: 1515.9 → 1875.2
  fps). Ladder: aliasing the block buffer to a 1 MB footprint +32%, a 128-row
  ring (63 MB) +27.6%, a 64-row ring (32 MB) +29%; shipped with the 63 MB ring
  and 3 stripes (+18.5%), because the 5-stripe version's extra dispatches and
  barriers ate 7 points of it.
- **One pad dispatch instead of 3–7 (+3.35%)**: the tw source planes are one
  descriptor array read with a compile-time slice index. Deleting the pad pass
  entirely and folding branchless reflection into fused was 4% *slower* — the
  reflection costs more than the pass it replaces — and the padded plane is a
  lossless copy (u16/f16 round-trip exactly, f32 keeps its bits), so there was
  nothing else to gain there.
- **The table loads are not the cost: 0.73%.** Ablating the window weights and
  window_freq reads measured 15.5% of the frame, but vectorizing them (144 → 60
  loads per thread through a transposed window copy and vec2/vec4 views of the
  table buffer) returned 0.7%: the cost is the per-bin arithmetic (window
  multiply, the zero-mean shift's two subtracts and two adds), not the loads.
- **Radius 2/3: the cube did not fit, and spreading it over all 16 lanes fixed
  it (6.2x at tbsize 5, 6.7x at 7, and +6.8% at 3).** The cube is live between the
  three transforms, so at 32*TW floats per lane (160/224) it went past the
  register budget: ACO gave the r2 module 216 VGPRs and spilled 640 B with 734
  scratch ops (r3: 256 and 896 B / 1026), which cost 3–6x — the AMD OpenCL
  reference compiles the same algorithm to 256 VGPRs with *no* private segment,
  and it keeps 75%/34% of its tbsize-3 fps. Holding 9 bins per lane instead of
  16 per 9 lanes (Implementation) drops the cube to 90/126 floats: r2/r3 now
  compile to **191 VGPRs with zero scratch** (4788/6912 instructions, down from
  6888/10215) and the whole frame lands at 1328.6/910.6 fps. The same change
  also fills the 7 idle lanes during the temporal phase, which is where the
  +6.8% at tbsize 3 comes from.
- **Half-precision spatial staging is rejected on accuracy, not speed.** It
  measured +4.6%, but it adds a half rounding before the overlap-add and breaks
  the u16 ≤1 LSB and f16/f32 oracles. The ring captured the same DRAM mechanism
  without it.
- **`SUB_BLOCKS` re-swept after the ring: 8 still wins.** 4 is 1.65% worse (u16)
  and 0.59% (f16), 16 is 1.15% and 0.55% worse.
- **Full-fp16 arithmetic is rejected: it measured 5.6% SLOWER, not 30%
  faster.** A throwaway `-DF16_MATH` build made `td[]`/`trans` half and the
  FFTW codelets compute in half: 1438.1 vs 1524.1 fps, and the parity oracle
  collapsed (66% exact on the default config, worst 7.8e-3). Mechanism: a half
  scalar op is not cheaper here (the f16 module's fused_r1 VALU 1710 → 2109,
  the extra count the f32 window/filter terms' converts) and dual-issue packing
  collapsed (VOPD 562 → 15); the LDS/scratch halving and 12 → 16 waves/SIMD at
  radius 0 did not pay for it. Half only wins as a storage width, which is what
  ships.
- **f16 is +0.6% over u16 and is a storage width, not a precision.** 1997.6 vs
  1984.9 fps on the same content (jpbd 1080p, 3000-frame medians): half halves
  the padded plane (26 → 13 MB at 1080p) and its pad stores, and the parity
  oracle holds. The f16 module's fused_r1 codegen is within a few percent of
  the f32 module's (VGPR 120, 12 waves/SIMD; VALU 1710 vs 1777).
- **CPU in / CPU out** is core-inserted upload+download (both free: ReBAR memcpy
  ahead, SDMA download). `--download` is −1% (core planes read in place) and
  chained instances +4%, the residual being the pad dispatch and the ring
  barriers the chain still pays.
- **Fused codegen residue is deliberately left**: remaining IM2COL/window ALU
  and pointer-walk strength reduction; col2im already runs ~2x the references.

## Historical

The pre-R80 design and every round that shaped it, kept for the mechanisms:

- **The R80 port's scoreboard is void.** It read 1452.6/1376.0 → 1361.3/1360.3
  (CPU cache), 1182.3 → 1350.3 (`--download`) and 1376.2 → 1364.7
  (`--synthetic`) fps, and "1.57x vs vszipcl", all against the pre-fix
  benchmark harness (`notes/METHOD.md`). The banner's table replaces them; the
  mechanism it recorded (the port's gap was the transfer) still holds.

- **Creation-failure leak** (correctness-only, no perf change).
  `createVideoFilterEx2` returns `nullptr` without running the free callback when
  the node constructor throws (the defensive `isValidVideoInfo` path), so the
  instance released just before the call leaked its buffers, its exec pool and
  its node reference. The `nullptr` arm now releases through `DftFree`; BM3D and
  Bilateral have the same fix.
- **`env_flag` accepted `"false"`** (correctness-only, no perf change). Only
  `"0"` and the empty string disabled a flag, so
  `VSFEEL_DFTTEST_SGSIZE_INVALID=false` still forced the invalid subgroup size and
  failed creation (and every `env_flag`-gated ablation, e.g. BM3D's `NOSEARCH`,
  read as on). It now rejects `"false"` like the debug/trace parsers; pinned by
  `test_dfttest_env_flag_false_is_disabled`.
- **Dead creation-time stride probe removed** (correctness-only, no perf change).
  It allocated a full-size CPU frame at creation to bound the *output* pitch for
  32-bit addressing, while the frame path re-reads the pitch anyway and never
  bounded the *source* pitches at all. The probe frame is gone and the bound now
  runs per frame on the real output and source pitches (`dft_offsets_fit`), so it
  covers more than before. `any_process`/`any_plane` were dead too (`planes`
  cannot be an empty array without the `empty` marker).
- **Slot-direct frame cache** — each source frame reflect-padded once into a
  shared device-local slot, fused reading the slots in place via `slot_base[7]`,
  reclaim gated on a per-resource `frame_gen` and published on per-slot
  *timeline* semaphores. Worth +22% then; the port replaced it with the core's
  own GPU frame cache plus re-padding, which costs ~26 µs/instance.
- **Semaphore protocol** — the cache first used one binary semaphore per pad
  generation, which deadlocked at 3+ chained instances (an upstream frame
  processed twice left the second consumer waiting on a consumed signal); a
  timeline semaphore with one signal value per generation, non-destructive
  waits, fixed it. All of it is gone with the cache.
- **ReBAR host-direct upload** — plain `memcpy` into host-visible device-local
  memory, pad reading VRAM instead of GTT (+15%). The measured contradiction
  (`memcpy` 64 GB/s vs NT stores 31 GB/s here, the reverse in an earlier
  session) was never resolved; the port removed the question.
- **Fused `IM2COL` branch** — the `sb >= 0` slot/padded branch if-converted into
  doubled u16 loads and +143 `s_waitcnt`; the fix was a branchless
  `FUSED_DIRECT` variant plus a mixed one, host-picked per frame. The port has
  one addressing mode, so there is nothing to pick.
- **Second-queue transfer split** (pre-R80, neutral) and **a second
  compute-family queue for the R80 core's transfers** — both the minimal form
  (downloads on compute-family index 1, uploads left on compute) and the full
  one (uploads index 1, downloads index 2), built from `reference/vapoursynth`:
  measured **neutral**, 1192/1194 fps with and without on DFTTest and 1699 vs
  1707 on Bilateral. Both were neutral for the same reason: a second queue of
  the *compute* family is the same engine as the compute queue, so the copy
  still competed with the dispatches. Only the dedicated transfer family, which
  is what `RADV_EXPERIMENTAL=transfer_queue` exposes, moves it off. The patch
  trees were deleted and the stock core is what runs here.
- **ODR COMDAT hazard** — filters that instantiate the shared `FramePool<T>`
  with a filter-local struct of the same name but different size silently
  corrupt the pool. No `FramePool` here any more; the rule lives in `AGENTS.md`.
- **The subgroup size was assumed, not checked** — the fused kernel used to take
  whatever default the device reported. Measured on lavapipe (the one device here
  whose subgroups are 8 lanes wide, `VSFEEL_DFTTEST_SGSIZE=8` to reproduce the old
  selection) it still matched gfx1100 to 3.7e-9 (one ulp), because a software
  backend implements the subgroup-scoped barrier as a workgroup one; on hardware
  with real 8-lane subgroups the same selection is a race, so the requirement is
  now enforced at creation. `src/vsfeel.h` also gained the two checks the request
  was missing: size control actually enabled, and subgroups per workgroup within
  `maxComputeWorkgroupSubgroups`.
- **Non-finite window vs `fsum`** — mode 4 (`swin`/`twin` 4) overflows its
  Bessel series to `inf/inf = NaN` for `|beta| ≳ 4e11`, and NaN makes every
  `lo != 0.0` test in the Shewchuk sum true, so the expansion grew one slot per
  window value and wrote up to 704 doubles past `partials[64]`. Creation now
  rejects a non-finite window; `fsum` bails to NaN at the array bound. No perf
  change.
- Dead ends that stay dead: `SUB_BLOCKS=16` (worse), the LDS-slice fused
  restructure (2x slower, LDS-bound), a host-side cache with mutex/cv
  ordered-submission waits (starves the worker pool).
- **`yuv_passthrough_*` was a test bug, not a filter bug (2026-10-02).** The YUV
  clip was built from the untrimmed 300-frame source while the Gray fixture is 24
  frames, so frame 23 (the fixture's last) saw a different temporal window; both
  sides now come from `source_clip()`. No perf change.

- **2026-10-22 — the block handoff goes through L2 (+24%).** Both passes
  streamed a 136 MB block buffer through VRAM every frame (aliasing it to 1 MB
  measured +32%). Block rows now run through a 63 MB ring in 3 stripes
  (+18.5%), `pad` writes every slice in one dispatch (+3.35%) and the table
  loads are vectorized (+0.73%): 1515.9 → 1875.2 fps (jpbd 1080p GRAY16, 3000
  frames, medians of 3). Rejected after measuring: half-precision staging
  (+4.6%, accuracy), removing the pad pass (−4%, reflection costs more),
  `SUB_BLOCKS` 4/16 (both worse).

- **2026-10-23 — the cube spread over all 16 lanes (6.2x at tbsize 5).** The
  transformed cube (9 x-bins x 16 y-bins x tw slices) lived as 32*tw floats per
  lane in the 9 lanes that own an x-bin, which ACO could not keep in registers:
  r2 spilled 640 B with 734 scratch ops and r3 896 B with 1026, for 208/131 fps
  where the reference keeps 75%/34% of its tbsize-3 fps. Every lane now holds 9
  bins (18*tw floats) and the surplus y-bins are pooled through the transpose
  buffer once per slice; r2/r3 compile to 191 VGPRs with no scratch and land at
  1328.6/910.6 fps (same-session A/B: 6.2x and 6.7x), plus +6.8% at tbsize 3
  from the now-full 16 lanes (1984.9 fps; jpbd 1080p GRAY16, 3000 frames,
  medians of 3).

## Open work

- **Radius 2/3 retention** (46–67% of the tbsize-3 fps against vszipcl's
  34–75%). The cube is now spill-free at 191 VGPRs, which is still 7 waves
  against 10 at tbsize 3: the remaining gap is occupancy, and closing it needs
  the cube spread over 32 lanes per block rather than 16 (18 bins per block
  over 16 lanes is 2*18*tw floats; over 32 lanes it would be 9*tw). That is a
  full lane-remapping rewrite of the transposes, not a tuning change.
- **Sigma-array reads** (non-scalar `ssystem` paths only): `sigma[(t*16+y)*9+x]`
  is read with stride 9, so a `[x][t][y]` transpose would make each thread's 144
  reads contiguous and take the same vec4 treatment the window got. Not the
  default path, hence not done with it.
- **The stripe floor.** 6 dispatches and 6 barriers per plane stand between the
  32 MB-ring ceiling (+29%) and the shipped +18.5%; a column-direction stripe
  would fit 2 stripes at 1080p (72 MB ring) if the reindexing is worth ~2-3%.
- `pad`/`col2im` are 32x8 (256 invocations, the plugin's largest workgroup):
  a device at the Vulkan minimum of 128 cannot run DFTTest at all and now says
  so at creation (`tests/test_device_limits.py`). The local size is a literal in
  the shader and the grid is sized from it, so a smaller variant means a `-D`
  plus the matching grid math in `DftCreate` — worth doing only if such a device
  turns up. NLMeans, EEDI3's copy kernels, NNEDI3's keep/pad and BM3D's
  aggregation are in the same position.

### Do not retry

- **Half arithmetic in the fused kernel.** 5.6% slower and less accurate, for
  the VALU/VOPD mechanism in Performance; half pays only as a storage width.
- **In-filter download alternatives.** A second *compute-family* queue for the
  copy (neutral: same engine), the core's host-visible direct-read path with
  streaming loads (fast at 1080p, collapses past ~8 MB), host-cached plane
  memory (much worse), and size-gated streaming (helps DFTTest, does not
  generalize) — all measured and rejected. The fix was the SDMA opt-in.

## Debug env vars

- `VSFEEL_DFTTEST_GPUTRACE=<frame>` — one-shot warm GPU timings for that frame
  (`pad`, then the *first stripe's* `fused`/`col2im`, then the whole frame);
  defaults to frame 100 and requires the queue family to report timestamp bits.
- `VSFEEL_DFTTEST_TIMING=1` — per-frame host stage averages
  (`acquire`/`record`/`submit`).
- `VSFEEL_DFTTEST_TRACE=1` — one line per submitted frame.
- `VSFEEL_DFTTEST_VRAM=1` — per-in-flight-frame scratch banner.
- `VSFEEL_DFTTEST_SGSIZE=N`, `VSFEEL_DFTTEST_SGSIZE_INVALID` — force (or
  deliberately break) the requested subgroup size; both bypass the multiple-of-16
  requirement above on purpose, and an unsupported size is still rejected when the
  pipeline is created.
- `RADV_DEBUG=asm`, `RADV_DEBUG=shaderstats` — ACO ISA and VGPR/LDS/occupancy.
