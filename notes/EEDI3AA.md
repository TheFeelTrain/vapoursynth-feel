# EEDI3AA — notes

Status: **implemented and shipped.** `core.vsfeel.EEDI3AA` exists
(`src/eedi3.cpp` registers it on the `EEDI3`/`EEDI3H`/`EEDI3AA` create path,
`src/eedi3.comp` carries `ENTRY_ASSEMBLEV` + `ENTRY_COMPOSE`), the `vsaa`
wrapper ships, and `tests/test_eedi3aa.py` covers it. It replaces the
`vsaa.based_aa` chain of four EEDI3 sub-passes plus two `std.Merge` nodes with
**one plugin call and one fused VRAM compose**; it is bit-exact against the
two-call chain for u16 and within a few ulp for f32.

Current measured state: `benchmark/bench.py --filter eedi3aa` ~**121 fps at
ns=8** (vszipcl 48.6, eedi3vk2 36.2). The round-by-round design record follows
in the order it happened, ending at round 8; the original handoff text
("Status: designed, not implemented") is preserved under `## Historical` at the
end of this file.

The one-line summary: `vsaa.based_aa` runs **four** EEDI3 sub-passes per output
frame as two chained filter calls plus two `std.Merge` nodes. Fusing that whole
chain into one plugin call lets the mask, the sclip, the source frame and the
intermediate merged frame be read once instead of two-to-four times, and lets
the two merges happen inside the assemble kernel. Projected win on the real
based_aa path: **~1.6–1.8×** (measured chain today: 98.9 fps; see below), versus
the +15% that round-19 EEDI3H tuning delivered.

---

## 1. What based_aa does today (this is the spec)

`vsaa/deinterlacers.py` (reference copy at `reference/vs-jetpack/vsaa/`, and the
installed `/usr/lib/python3.14/site-packages/vsaa/` matches it here):

- `class EEDI3(SuperSampler)` — line 492.
- `EEDI3.antialias` — line 755. The loop (780–796) is the whole story:

```python
sclip, mclip = kwargs.pop("sclip"), kwargs.pop("mclip")
if sclip and self.double_rate:
    sclip = core.std.Interleave([sclip, sclip])     # ONCE, shared by both passes
tff = fallback(kwargs.pop("tff", self.tff), True)
for y in (VERTICAL, HORIZONTAL):                    # transpose_first=False
    horizontal = (y == HORIZONTAL)
    clip = self._interpolate(clip, tff, self.double_rate, False,
                             horizontal=horizontal, sclip=sclip, mclip=mclip, **args)
    if self.double_rate:
        clip = core.std.Merge(clip[::2], clip[1::2])
```

- `SuperSampler._interpolate` — line 838: `func(clip, tff + double_rate*2, dh, ...)`,
  where `func` is `backend.EEDI3` or `backend.EEDI3H` (lines 831/835).
  based_aa defaults: `double_rate=True`, `dh=False`, so **field = 3** for both
  passes; `tff` True unless overridden.
- **Consequences that define the fused filter:**
  1. VERTICAL runs first; the HORIZONTAL pass consumes the *merged vertical
     output*, not the original clip: `out = Merge(H(Merge(V(x))))`.
  2. `sclip` and `mclip` are the *same clips* for both passes. `mclip` is
     single-rate and is indexed by `sn = n/2` inside the plugin, so both field
     sub-passes of one output frame read the *same* mask frame. `sclip` is the
     interleaved (`Interleave([s,s])`) clip, so sclip[2k] and sclip[2k+1]
     usually alias (same plane pointer — verified with
     `VSFEEL_EEDI3_PTRTRACE=1`).
  3. `based_aa` (funcs.py, ~line 207+) calls `antialiaser.antialias(ss, **aa_kwargs)`
     once; the two calls are entirely inside `antialias`.
  4. The direction loop is sequential, so the two directions cannot be run
     side-by-side on the same input — fusion means "one filter call that does
     all four sub-passes", not "one kernel for both directions".

### Measured today (real AA bench clip, 600 requested frames, ns=8, `tmp/fused_probe.py`)

| what | fps | note |
|---|---|---|
| one `vsfeel.EEDI3` call (bench.py's graded chain) | 509.6 | 1.96 ms per sub-frame |
| one `vsfeel.EEDI3H` call | 374.0 | 2.67 ms per sub-frame |
| **exact based_aa chain (V→Merge→H→Merge)** | **98.9** | 10.11 ms per output frame |

Sum-of-parts = 2×1.96 + 2×2.67 = 9.27 ms, so the two Merge nodes + the V0/V1/v
frame materialisation cost ~0.84 ms (~9%). Reproduce with
`python3 tmp/fused_probe.py 600` (needs the working tree; it builds the vpy
through `benchmark/bench.py`).

---

## 2. Semantics that must be reproduced *exactly*

These were verified, not assumed — do not re-derive them from intuition.

1. **`std.Merge(a, b)` with the default weight 0.5** (`tmp/merge_semantics.py`,
   full-range random GRAY16, 512 samples; GRAYS float):
   - u16: `out = (a + b + 1) >> 1` — 0 mismatches. `(a+b)>>1` is wrong (263
     mismatches); half-even is wrong (131).
   - f32: `out = 0.5f*a + 0.5f*b` exactly (bitwise).
   - Therefore the fused filter can be **bit-exact**, and the merge must be done
     in *integer* u16 arithmetic after the vcheck's quantisation (see 3.4).
2. **sub-frame → parity mapping.** For output sub-frame `n` (0-based within the
   doubled stream), the plugin computes `field` exactly as
   `Eedi3GetFrame` does today (`src/eedi3.cpp` lines 2003-2015):
   `field = d->field & 1`, then `_FieldBased` of the *frame being processed*
   overrides it (TOP→1, BOTTOM→0), then `if (d->field > 1) field = (n&1) ^ field`.
   With `d->field = 3`, progressive input, `tff=1`: sub-frame 0 has interp parity
   **1**, sub-frame 1 has interp parity **0**. The sub-frame's *kept* parity is
   `1 - field` (rows/columns copied from the source, not interpolated).
3. **The horizontal pass's `_FieldBased` override must NOT be applied.** Its
   input in the chain is the merged vertical output, and every EEDI3 output
   frame gets `_FieldBased = VSC_FIELD_PROGRESSIVE` (`src/eedi3.cpp` line 2455),
   so the horizontal sub-passes see PROGRESSIVE and take no override. The
   vertical sub-passes DO apply the *input clip's* `_FieldBased`. Getting this
   wrong is a silent one-parity-wide error.
4. **Merge pairing:** output frame `k` of the whole chain pairs sub-frames
   `n = 2k` and `n = 2k+1` of the same pass, both derived from input frame
   `k` (`sn = n/2`).
5. **fps / frame count / props:** the chain outputs N frames at the *input's*
   fps (verified: `vspipe --info` on the chain shows 600 frames @ 24000/1001,
   same as the input; each EEDI3 call doubles the rate and the merge/node
   bookkeeping brings it back). The fused filter emits N frames, so it should
   simply pass the input's fps through and set `_FieldBased` progressive.
   Concretely `Eedi3GetFrame` *halves* `_DurationNum` when `field > 1` — that is
   how each chained call doubles the rate; the fused filter must NOT do that
   (copying that block verbatim is the easy mistake). Verify props equality
   against the chain (protocol: `vspipe --info`).
6. **Plugins must not change output for the reference path.** The fused filter
   is additive; EEDI3/EEDI3H and their tests stay untouched.

### Explicitly out of scope (fall back to the two-call chain)

`direction != BOTH`, `double_rate=False`, `dh=True`, non-GRAY/YUV formats the
plugin already rejects, and any case the wrapper cannot map (see §5).

---

## 3. The fused design

One plugin function, e.g. `core.vsfeel.EEDI3AA(clip, field=3, nrad, mdis, alpha,
beta, gamma, vcheck, vthresh0..2, sclip, mclip, num_streams)`, called once per
output frame, doing four sub-passes in **two submits**. It returns the exact
`Merge(H(Merge(V(clip))))` frame (N frames total, unchanged rate).

Recommended structure per output frame `k` (input frame `sn = k`):

```
host   gather everything except v's rows  (shared across all 4 sub-passes)
       inputs: src = clip[k], mask = mclip[k] (one frame, both parities),
       sclip = sclip[2k] and sclip[2k+1] (the interleaved 2N-frame node;
       usually the same plane pointer -> one read)
CB#1   per sub-frame n in {2k, 2k+1} sequentially reusing buffers:
         pad build -> row kernel -> vcopy/vcheck      (vertical, source = clip[k])
       then: assemble-v kernel  (needs src + vout[0] + vout[1])
submit#1, wait
host   pair-gather v's rows once -> K_v for BOTH column parities (one read)
CB#2   per sub-frame n in {2k, 2k+1} sequentially reusing buffers:
         xpose(K_v[n]) -> R_v' ; pad build ; row kernel ; vcopy/vcheck
       then: merged compose kernel (needs R_v', vout[0], vout[1]) -> staging
submit#2, wait
host   plain row copy staging -> dst  (kept + interp already assembled, merged)
```

### 3.1 Vertical merge (assemble v)

With interp parities `f0` (n=2k) and `f1 = 1-f0` (n=2k+1):

```
V0[y] = (y&1 == f0) ? interp0(y) : src[y]
V1[y] = (y&1 == f1) ? interp1(y) : src[y]
v[y]  = (V0[y] + V1[y] + 1) >> 1          (u16; f32: 0.5*(V0+V1))
```

i.e. every row of v is `(src[y] + interp_of_that_row + 1) >> 1`; the interp value
comes from vout[f0] or vout[f1] depending on the row's parity. The kept value is
the *source frame row* (`V0`/`V1` kept rows are copied verbatim from src by the
existing host path). **The interp values must be the quantised u16 the vcheck
produced** (vcheck already quantises via `vc_quant`), and the average is integer.

Row indexing (vertical, `dh=false`: kernel plane width = frame width,
`rows = height/2`):

```
sub-frame n writes interp row r to output row y = f_n + 2r   ->  vout_n[r][x]
kept rows are y = (1-f_n) + 2r and equal src[y][x]
assemble: for output row y with (y&1) == f_n -> interp = vout_n[(y-f_n)/2][x]
          otherwise                         -> src[y][x]
```

`src[y]` does not need a separate full-frame upload: the two compacted
per-parity uploads together *are* the frame, so row y comes from the upload of
the sub-frame whose kept parity matches `y&1` (verify the row base math against
`gather_columns(..., step=1 ...)` / the vertical `frame_copy_out` block).

### 3.2 Horizontal input (v) layout

v must be in **host memory** for the horizontal column gathers (they deinterleave
source rows). Write v to the staging region (one full-plane write, 16.6 MB at the
bench geometry) — the same place EEDI3H's `out_offset` lives today.

Then the host does ONE `gather_columns_pair`-style pass over v's rows producing
the kept-column compactions for **both** column parities (`K_v[f0]`, `K_v[f1]`,
complementary halves = the whole frame, 16.6 MB written into the VRAM upload).
That is exactly the round-19 aliased-sclip trick, now applied to the v pair.

### 3.3 Horizontal merge (final compose)

For sub-frame n the horizontal interp columns are parity `f_n` and the kept
columns are `v[q]` at the other parity. Define `H0`, `H1` as the two assembled
frames; the chain merges them:

```
out[q] = (H0[q] + H1[q] + 1) >> 1
       = ( v[q] + (q&1 == f0 ? vout_h0(q) : vout_h1(q)) + 1 ) >> 1
```

so ONE compose kernel with both `vout` buffers and `R_v'` (the transposed kept
columns, which equal `v[q]`) writes the already-merged plane into staging:

```
out[y][2k+p] = ( R_v'[k][y] + (p == f0 ? vout0[k][y] : vout1[k][y]) + 1 ) >> 1
```

(u16; f32 `0.5*(a+b)`). This replaces both the second `std.Merge` and the chain's
full-plane staging round trip per sub-frame.

Column indexing (horizontal: kernel plane width = frame height,
`rows = frame width/2`): interp column `q = 2k + f_n` holds `vout_n[k][y]`; kept
column `q = 2k + (1-f_n)` holds `v[y][q] == R_v'[k][y]` (the compose already has
`R_v'` at `pc.pad_base`, and `WIDTH`/`rows` swap roles exactly as EEDI3H does
today). `ENTRY_COMPOSE` in
`src/eedi3.comp` is the starting point — add the average and the second vout
source (binding 7 is already the vcheck output; a second binding is needed for
the other field's vout, or two push-constant bases into the same buffer).

### 3.4 Buffer reuse (keeps VRAM near today's)

The two sub-frames of a pass can run *sequentially inside one command buffer*
with a barrier between (the row kernel and vcheck already need barriers), so
`pad`, `pbt`, `dst`, `dmap`, `cint`, `rempty` are all reusable per sub-frame.
Only what is consumed simultaneously must be doubled:

- `vout` for both fields (+1 buffer) — the assemble/compose kernels average them;
- `pbt` is the big one: `rows*width*tpitch` int8 = **170 MB per sub-frame** at the
  bench geometry today. Reuse it across the two sub-frames of a pass; do not
  allocate 2× or per-stream VRAM balloons (ns=8).
- uploads (K, sclip, bits) for both parities; the mask bit matrices (see 3.5);
  the staging v plane and the final output plane.

### 3.5 Shared host gathers (the actual win)

Only the input side needs new code; all of it already exists in round-19 form:

| data | chain cost per output frame | fused |
|---|---|---|
| mask frame | vertical 2×8.3 MB + horizontal 2×16.6 MB ≈ **50 MB read** | **16.6 MB read**, one pass builds vertical bits for both parities (`build_bmask_row` on the parity rows) *and* both transposed bit matrices (`gather_mask_bitmat`, extended to emit even+odd parity in the same tile — the aligned deinterleave already produces both) |
| sclip frame | vertical 2×8.3 + horizontal 2×16.6 ≈ **50 MB read** + uploads | **16.6 MB read** (or 8.3 with the aliased pair trick when sclip[2k]==sclip[2k+1]), one pass emits the parity rows (vertical) and the parity columns (horizontal) |
| src clip | 2 × 8.3 MB read (one per vertical field, complementary row parities) | **16.6 MB read once**, upload once (either the full frame, or the two compact parities as today — same bytes) |
| intermediate v | V0/V1 frames + `Merge` + v frame ≈ **80 MB** write/read | one staging write (16.6) + one pair-gather read (16.6) |
| merges | two `std.Merge` nodes, ~two graph nodes | inside the assemble/compose kernels |
| submits | 4 plugin calls + 2 graph nodes | 2 submits, 1 call |

Byte model: chain ≈ 450 MB per output frame (≈44.5 GB/s at 98.9 fps — it is
sitting on the memory wall, which is why the single-call fps figures do not
compose). Fused ≈ 250–300 MB ⇒ projected **150–180 fps (~1.6–1.8×)**. Validate
with a prototype before committing to the whole integration.

### 3.6 Suggested implementation order

1. **Correctness first, no sharing.** Add `EEDI3AA` that just runs the four
   sub-passes the way the chain does (two full plugin-equivalent passes inside
   one call, with v materialised in staging and the two merges as GPU passes).
   Get `tests/test_eedi3aa.py`'s oracle bit-exact before touching performance —
   every later bug is then in the sharing, not in the semantics.
2. **Then share the gathers** (§3.5): one mask pass for all four bit sets, one
   sclip read, src read once, v pair-gathered once. Re-run the oracle.
3. **Then tune**: streams knee (`num_streams`), queue cap
   (`VSFEEL_EEDI3AA_QUEUES`), `COPY` modes, buffer reuse (§3.4) — always with
   the real AA chain and the order-reversed A/B loop (§7).
4. **Then the wrapper** (§5) and the benchmark entry (§7), and only then
   consider landing: the existing EEDI3/EEDI3H path must stay untouched.

Registration anchor: `vsfeel_register_eedi3` at the bottom of `src/eedi3.cpp`
(~line 3755) registers `EEDI3`/`EEDI3H` with a shared argument string; add
`EEDI3AA` there (same args, `field` required, `sclip`/`mclip` optional) plus an
`Eedi3AaCreate` next to `Eedi3HCreate` (~line 3745).

---

## 4. Reuse map (`src/eedi3.cpp`, round 19 state)

- `gather_columns_pair` / `gather_pair_row_u16` / `gather_pair_row_f32` — one
  pass produces both column parities (aliased sclip). Reuse for the v pair.
- `gather_mask_bitmat` — fused transposed mask bits (y-outermost, 4×16-row
  groups, aligned loads). Extend it to emit both parities.
- `build_bmask_row` / `bmask_dilate_store` / `build_bmask_row_from_bits`.
- `deint_even/odd_u16`, `deint_even/odd_f32`, `deint_even_u8`, `nz_bytes_*` —
  parity selection by deinterleave, loads always 32-byte aligned.
- `ENTRY_XPOSE`, `ENTRY_COMPOSE`, `ENTRY_PAD` (with `pad_skip_parity`),
  `ENTRY_ROW`, `ENTRY_VCHECK`/`ENTRY_VCOPY`, `ENTRY_BLIT`.
- `PlaneConfig`, `Eedi3Resource`, `FramePool`, `submit_with_fence`,
  `submit_timeline`, `destroy_common` (`src/vsfeel.h`).
- Env knobs to keep: `VSFEEL_EEDI3_MASKFUSE`, `VSFEEL_EEDI3_PAIR`,
  `VSFEEL_EEDI3_COPY`, `VSFEEL_EEDI3_QUEUES`, `VSFEEL_EEDI3_NOREBAR`,
  `VSFEEL_EEDI3_VOUTDEV`, `VSFEEL_EEDI3_HBENCH` + `VSFEEL_EEDI3_HFRAME`.
  One env name per filter: use `VSFEEL_EEDI3AA_*` for anything new.
- Diagnostic knobs added during round 19 that should be deleted before landing:
  `VSFEEL_EEDI3_NOXPOSE`, `VSFEEL_EEDI3_NOCOMPOSE`, `VSFEEL_EEDI3_NOMASKX`,
  `VSFEEL_EEDI3_PTRTRACE` (see §10).

### 4.1 Footprint: what is new vs reused (no fork)

`src/eedi3.cpp` is 3786 lines, `src/eedi3.comp` 1349. The fused filter is an
extension of the same files, not a second EEDI3: `horiz` and `field` are already
parameters everywhere they matter.

**Reused unchanged**
- All seven kernels (`ENTRY_ROW/VCHECK/VCOPY/PAD/XPOSE/COMPOSE/BLIT`); EEDI3 and
  EEDI3H already share 100% of them. No new DP/backtrack/vcheck math.
- Every host gather is a free function over `(ptr, stride, first, step)`:
  `gather_columns`, `gather_columns_pair`, `gather_mask_u8`,
  `gather_mask_bitmat`, `build_bmask_row`. They take raw pointers, so the
  horizontal pass can be fed the internally produced `v` buffer with no new
  code, and both parities are already expressible via `first = 0/1`.
- Pipeline creation + `WidthKey` cache, `PlaneConfig`/`Eedi3Resource`/
  `FramePool`, `submit_with_fence`, ReBAR upload, pipeline cache.

**Refactor, do not duplicate (~300 lines touched)**
- `record_command_buffer` (line 1552) already takes `field`; make it take the
  plane set (`const std::array<PlaneConfig, MAX_PLANES> &`) instead of reading
  `d.planes[]`, and split its tail into a small mode
  (`none` / `assemble-v` / `merged-compose` / existing blit-D2H). It then
  records exactly one sub-pass.
- `Eedi3GetFrame` (line 1933): factor out the field derivation (lines
  2003-2015), the gather loops and the submit/wait/blit sequences. The fused
  frame handler is new orchestration over those helpers.
- `vsfeel_eedi3_create`'s layout block (~150 lines): offsets need a sub-frame
  dimension. Cleanest: group `PlaneConfig` into `struct PassConfig { PlaneConfig
  plane[MAX_PLANES]; }` and pass it down, so `pbt` (170 MB/sub-frame at 4K) is
  reused sequentially instead of doubled. This is the one place a copy-paste
  fork would be tempting and would rot.

**New (~400-500 lines)**
- The fused frame handler (2 submits, intermediate `v`, 4 sub-passes) — the bulk,
  and it is orchestration, not algorithms.
- Two small shader entries: the `v` assembler (`v = (src + vout_n + 1) >> 1`) and
  the merged compose (add the average + a second `vout` source to
  `ENTRY_COMPOSE`) — ~40-60 GLSL lines + CMake + two pipeline handles.
- Registration (~15 lines, same argument string) and the Python wrapper (§5).

**Step 1 needs no new shader code at all**: run the four sub-passes as four
`record_command_buffer` calls, merge `V0/V1` and `H0/H1` with a host loop over
the staging buffers, and feed the horizontal gathers from the internal `v`
buffer. That is bit-exact immediately and already removes the two `std.Merge`
nodes and the intermediate VS frames (the measured ~9% floor). Only then fold a
pass's two sub-passes into one command buffer and move the merges onto the GPU.

Because the fused filter calls the same helpers, the existing
`tests/test_eedi3.py` + `tests/test_eedi3h.py` (81 tests) stay a live regression
net for the shared code.

---

## 5. Integration (no vs-jetpack changes)

`vsfeel/backend.py` already duck-types the vsaa backend. Add a subclass of the
vsaa EEDI3 *antialiaser* that overrides `antialias()`:

```python
# vsfeel/backend.py (or a lazily-imported vsfeel.vsaa module)
from vsaa.deinterlacers import EEDI3 as _VsaaEEDI3   # import lazily!

class EEDI3(_VsaaEEDI3):
    def antialias(self, clip, direction=_VsaaEEDI3.AADirection.BOTH, **kwargs):
        if (direction == _VsaaEEDI3.AADirection.BOTH and self.double_rate
                and <format/params supported>):
            import vapoursynth as vs
            sclip, mclip = kwargs.pop("sclip"), kwargs.pop("mclip")
            if sclip is not None:                     # exactly the base class
                sclip = vs.core.std.Interleave([sclip, sclip])
            tff = fallback(kwargs.pop("tff", self.tff), True)
            return vs.core.vsfeel.EEDI3AA(            # field = tff + double_rate*2
                clip, tff + 2, dh=False,
                sclip=sclip, mclip=mclip,
                **self.get_deint_args(**kwargs))
        return super().antialias(clip, direction=direction, **kwargs)
```

- Usage: `based_aa(clip, antialiaser=vsfeel.EEDI3(backend=vsfeel.Backend, ...))`.
  `based_aa`'s `isinstance(antialiaser, EEDI3)` checks pass because it is a
  subclass; `.sclip`/`.mclip`/`.backend`/`.supports_mclip` come from the base.
- Keep `import vsfeel` working without vsaa installed: import vsaa lazily
  (inside the module/class construction), mirroring how `backend.py` already
  imports vsrgtools/vsdenoise inside `__call__`.
- The existing `FeelBackend` (with `EEDI3`/`EEDI3H` entry points) stays as-is for
  `backend=` users who are happy with the two-call chain.
- Fall back to `super().antialias(...)` for `direction != BOTH`,
  `double_rate=False`, `dh`, or unsupported formats.

---

## 6. Testing plan

**Oracle (bit-exact, mandatory).** Build the chain out of the *same plugin* so
the oracle isolates the fusion from any plugin difference:

```python
def _oracle(clip, **kw):
    v  = core.vsfeel.EEDI3(clip, field=3, **kw)
    vm = core.std.Merge(v[::2], v[1::2])
    h  = core.vsfeel.EEDI3H(vm, field=3, **kw)
    return core.std.Merge(h[::2], h[1::2])
assert frames_equal(core.vsfeel.EEDI3AA(clip, field=3, **kw), _oracle(clip, **kw))
```

- Assert **exact** equality (max diff 0) for u16 and f32; per AGENTS.md, measure
  before relaxing anything and document any mechanism if it ever cannot be exact.
- Cases: `tff` 0/1, `_FieldBased` progressive/TFF/BFF input, `mclip` present and
  absent, `sclip` present and absent (HAS_SCLIP=0 path → the vertical row kernel
  computes `cint`), `sclip` aliased (`Interleave([s,s])`) and non-aliased, the
  `Interleave([clip, clip])` frame-pair case, `vcheck` 0..3, `mdis` 3/20/40,
  `nrad` 0..3, `front`/`back` alpha/beta/gamma corners, YUV420 planes (subsampled
  chroma), cropped/padded frames (never assume tight pitch; `.copy()` ctypes
  arrays — see `tests/conftest.py`).
- Self-consistency: `num_streams=1` vs 8 identical, repeated runs identical
  (the round-16 stale-`dst` bug was a real nondeterminism), parallel-load
  consistency.
- Props: frame count, fps, `_FieldBased` equal to the chain (compare via the
  vpy `--info` or `get_frame` props in a subprocess).
- Keep the noise-clip rule: only `tests/noise_24f.mkv` comparisons prove
  correctness; constant/blank input hides border bugs.
- Full suite before/after: `python -m pytest tests/test_eedi3aa.py -q` plus the
  existing `tests/test_eedi3.py` and `tests/test_eedi3h.py` (nothing may regress).

---

## 7. Benchmarking

`benchmark/bench.py` currently grades a *single* EEDI3 call, so the fused filter
needs a new entry (e.g. `"eedi3aa"`, `aa=True`) whose builder emits
`core.vsfeel.EEDI3AA(clip, field=3, sclip=sclip, mclip=mclip, num_streams=8)`,
and a reference arm that runs the chain based_aa would run on each reference
plugin. **Do not hand-roll that chain** — build it from `vsaa`'s own EEDI3
antialiaser (`vsaa.deinterlacers.EEDI3`), which is the object `based_aa`
drives; see ROUND 3 below for why the hand-rolled form was wrong.
Grade with the real AA chain (`make_aa_vpy`) over ≥1000 frames.

Measurement discipline on this box (round 19 experience): session-to-session
drift is ±5% and larger than the effects being chased. Use the same binary with
env knobs and an **order-reversed interleaved** loop (`tmp/abl.py` now reverses
the variant order on odd reps — before that fix, an inert knob on the vertical
arm swung 5%). Existing scratch tools, all functional:
`tmp/abl.py` (env-knob A/B), `tmp/ab2.py` (two .so A/B), `tmp/agg.py` (aggregate
stage split at ns=8), `tmp/split.py` (ns=1 stage split),
`tmp/fused_probe.py` (chain vs single calls), `tmp/merge_semantics.py`.

---

## 8. Risks / open questions

1. **Sync and ordering.** Two submits, with the horizontal host gather in
   between. CB#1 must fence the vertical row-kernel/vcheck writes before the
   assemble-v kernel reads vout0/vout1 (a compute->compute barrier inside CB#1,
   as the existing vcheck barrier does). The host must then wait on CB#1's fence
   before gathering v from staging — that wait IS the 2-submit structure (the
   alternative, one CB, is impossible because the host gather sits inside the
   dependency). Reuse `FramePool` / `submit_with_fence`; add no new sync
   primitives, and keep the submit on the queue lock.
2. **`_FieldBased` asymmetry** (§2 item 3) is the easiest silent bug.
3. **Quantisation order** (§3.1 and §3.3): quantise the interp value first, then
   integer-average — matching the chain, where V0/V1/H0/H1 are u16 frames.
4. **Aliased sclip in the chain:** in the wrapper, sclip is `Interleave([s,s])`
   so frames 2k/2k+1 usually share a plane pointer. Detect it (pointer compare,
   as `gather_columns_pair` does) and fall back to reading both frames.
5. **VRAM:** the pbt buffers are 170 MB/sub-frame at 4K; reuse per sub-frame
   (§3.4). Budget ~2× today's per-stream VRAM at worst (2 vout + 2 upload
   parity sets); re-check `num_streams=8`.
6. **The fused filter is specialised**, not a general EEDI3: it hardcodes
   vertical-then-horizontal, double-rate, and the 50/50 merge. Document that in
   the plugin's argument docs.
7. **Do not silently change the reference path.** EEDI3/EEDI3H remain the
   drop-in filters; EEDI3AA is a new entry point.

---

## 9. Do-not-retry (measured this session, mechanism included)

- **CPU merge of the kept columns** (compose writes interp-only tight; the CPU
  interleaves with the kept source columns): measured **−4.8%** over 8
  order-reversed reps (385.5 → 366.9 fps on EEDI3H). The extra 16.6 MB source
  re-read plus the merge ALU costs more than the 8.3 MB PCIe + 8.3 MB VRAM it
  saves. The GPU-kept path (`ENTRY_COMPOSE` writing the whole plane) is right.
- **Direct-to-frame host-pointer import** (`VSFEEL_EEDI3_DSTHOST=1`): 534 → 216.6
  fps (−59%) when re-measured. The per-frame import/destroy is a global VM cost;
  a pointer cache would not remove it (round 13: an import the GPU never touches
  still costs 20%).
- **`gather_mask_bitmat` k-outermost loop order**: 64 concurrent row streams
  defeat the prefetcher and measure ~2× slower than the y-outermost, 16-row
  group form.
- **Micro-optimising the mask gather's ALU** is not where the time is: 16.6 MB of
  mask reads is inherent (a transposed plane needs frame columns), and the byte
  transpose is ~0.3 ms/frame at ns=1 against ~0.8 ms of read.
- **EEDI3H's remaining single-filter gap is small.** After round 19 (fused mask
  +6.6%, aliased pair gather +5.3%, together +15.2% at ns=8, vertical arm flat as
  the control) the residual is the 2× mask read amplification (16.6 vs 8.3 MB)
  and the compose's full-plane staging write. Both are structural; that is the
  reason to fuse at the based_aa level instead.

---

## 10. Round-19 state of the tree (uncommitted, tests green)

`src/eedi3.cpp` carries: the fused EEDI3H mask path (`gather_mask_bitmat` +
`build_bmask_row_from_bits`, knob `VSFEEL_EEDI3_MASKFUSE=0` to A/B), the
aliased-sclip pair gather (`gather_columns_pair`, knob `VSFEEL_EEDI3_PAIR=0`),
and aligned parity-selecting loads in `deint_row_*` / `nz_bytes_*`. Measured
same-session, order-reversed, 6 reps × 1000 f, ns=8: EEDI3H base 384.9 fps,
`PAIR=0` 365.6, `MASKFUSE=0` 361.1, both off 334.2; vertical 534.5/534.8/535.1/531.4
(control). `tests/test_eedi3.py` + `tests/test_eedi3h.py`: 81 passed.

Temporary diagnostics still in the tree and to be removed before landing:
`skip_xpose`, `skip_compose`, `skip_maskx` fields + their env vars and the
`h_tMaskMid`/`maskx=`/`bmask=` hbench split (keep the hbench split, it is a
durable probe; drop the three skip knobs), plus the `VSFEEL_EEDI3_PTRTRACE`
block in `Eedi3GetFrame`.

---

# ROUND 2 — IMPLEMENTED (this session)

Status: **implemented, 585/585 tests green.** `EEDI3AA` exists, is bit-exact
against the two-call chain for u16 (and within a few ulp for f32), the vsaa
wrapper ships, and the benchmark entry grades it against the reference chain.

## What landed

- `src/eedi3.comp`: new `ENTRY_ASSEMBLEV` (the vertical merge, one dispatch per
  plane) + `vout2_base`/`raw2_base` push constants.
- `src/eedi3.cpp`:
  - `record_command_buffer` split into `record_pass(planes, horiz, second,
    tail, direct)` + a `record_h2d_copy` helper + `PassTail` enum, so one
    command buffer can record several sub-passes and EEDI3/EEDI3H keep their
    exact old shape (81/81 of their tests still pass).
  - `vsfeel_eedi3_create` grew an `aa` mode: the vertical geometry stays in
    `planes`, the horizontal pass over the merged frame gets `aplanes`, and the
    two share every byte-identical region (raw/sclip/dst/vout/pbt/dmap/cint)
    with only the differing ones doubled (raw2/sclip2/bits2/dst2/vout2/out2/v).
  - `Eedi3AaGetFrame`: two submissions. CB1 = vertical sub-pass 1 (tail none) +
    sub-pass 2 (tail assemble-v); host column-gathers `v` for both horizontal
    sub-passes; CB2 = horizontal sub-pass 1 + 2 (tail compose); host 50/50
    merges the two composed planes into `dst`.
  - **Descriptor sets**: AA needs BOTH geometries' views at once, which
    `d->horiz` cannot express. Added `desc_set_h` (horizontal row/vcheck/
    compose: b0/b5/b9 = pad_dev) and `desc_set_xp` (xpose + horizontal pad
    builder: b0 = the upload buffer, b8/b9 = pad_dev). *This was the single
    biggest bug*: without them, `pad_set`'s b9 was the upload buffer while the
    code used pad_dev-relative offsets, and under ReBAR the horizontal stage
    read/wrote out of bounds -> all-zero output. Non-ReBAR masked it (there
    b9 happened to be pad_dev anyway), which is why `VSFEEL_EEDI3_NOREBAR=1`
    produced bit-exact output while the default did not.
- `vcopy`/`vcheck`/`compose` bind `row_set`, xpose/pad bind `pad_set`;
  `record_pass` starts with a full compute barrier so the sub-passes can reuse
  pbt/dst/built-pad/R'.
- `vsfeel/vsaa.py`: `EEDI3(vsaa.deinterlacers.EEDI3)` overriding `antialias` to
  emit one `EEDI3AA` call; falls back to `super()` for `direction != BOTH`,
  `double_rate=False`, `transpose_first`, a `Deinterlacer` sclip or an
  unsupported format. `vsfeel.EEDI3` is a PEP-562 lazy re-export, so `import
  vsfeel` still never needs vsaa.
- `benchmark/bench.py`: new `eedi3aa` filter (vsfeel emits the fused call, every
  reference emits the chain based_aa would run: native for plugins with an
  EEDI3H, the transpose fallback otherwise). `parse_args` de-duplicates shared
  CLI flags.

## Measured (RX 7900XTX, real based_aa clip: jpbd 2x Point -> 3840x2160 GRAY16)

| arm | fps |
|---|---|
| vsfeel `EEDI3AA` (fused) | 96.3 |
| vsfeel two-call chain (order-reversed A/B, 6x600f) | 98.9 (fused 100.2 -> **1.01-1.07x**) |
| vszipcl chain (best reference) | 45.3 |
| vszipcu chain | 25.8 |
| eedi3vk2 chain (transpose fallback: no EEDI3H installed) | 28.1 |

**The §5 projection of 1.6-1.8x did NOT materialise.** The fused filter is
worth ~1.0-1.07x over the vsfeel chain: it removes the two `std.Merge` nodes
and the intermediate frame materialisation (the ~9% the notes measured as the
chain's graph overhead) and nothing else. It is still **2.1x faster than the
best reference chain**, which is the headline number.

Why the byte model was wrong: the host/GPU split probe
(`VSFEEL_EEDI3AA_HBENCH`) shows the per-frame latency is dominated by the two
fence waits (58 ms of 72 ms at ns=8), not the host gathers (13.5 ms). Removing
all host work raises fps to ~190, but that ablation also zeroes the mask and
therefore removes the DP work (the fully-masked fast path), so it is not a
valid host-bound proof. The honest reading is **GPU-bound**: the four
sub-passes cost the same on the GPU whether they run inside one call or two,
and `sum-of-parts` (2x EEDI3 + 2x EEDI3H) already predicts the fused number.

## Dead end — GPU K compaction (`ENTRY_ASSEMBLEK`), do not re-derive

Attempted: replace the intermediate `v` frame entirely — one kernel that
merges src+vout *and* emits both compacted column matrices `K_off[y][k] =
v[y][off+2k]` the horizontal `ENTRY_XPOSE` consumes, removing the v
materialisation (16.6 MB GPU->host write) plus two full-frame host reads and
two full-frame host writes per frame (measured as ~half the frame cost by the
host-work ablation).

Result: **bit-exact for luma, wrong for every chroma plane, in ReBAR and
non-ReBAR alike.** Verified with a debug dump (the K region copied into the
output frame at the end of the frame):
- the K *values* were verified exact (`K_even`/`K_odd` == `vm[:, p::2]`) when
  the kernel wrote into the staging buffer;
- a ramp probe (`asm_k[k_base+idx] = idx`) and a constant probe
  (`= WIDTH`) showed plane 0 written correctly (640 everywhere) and planes 1/2
  **untouched by the dispatch** — not mis-indexed, not mis-offset, simply never
  written;
- forcing plane 0's pipeline for plane 1 and reversing the plane dispatch order
  changed nothing; no validation-layer error was reported; disabling the
  persistent pipeline cache changed nothing.

So the plane-1/2 dispatch of that entry point is dropped for a reason not
identified after extensive bisection. The whole experiment was reverted and
`ENTRY_ASSEMBLEV` restored. If it is retried: start by proving the dispatch
executes for a *second* plane (the ramp probe above is 5 lines and settles it
immediately) before measuring anything.

## Knobs / probes kept

- `VSFEEL_EEDI3_NOREBAR`, `VSFEEL_EEDI3_VOUTDEV`, `VSFEEL_EEDI3_COPY`,
  `VSFEEL_EEDI3_QUEUES` — unchanged, still apply to AA.
- `VSFEEL_EEDI3_NORAW` / `NOSCLIP` / `NOMASKX` / `NOXPOSE` / `NOCOMPOSE` now
  also gate the AA gathers/passes (durable ablation knobs; the AA path honours
  them).
- `VSFEEL_EEDI3AA_HBENCH` + `VSFEEL_EEDI3AA_HFRAME=<n>`: host-stage split
  (vGather / vRec / vWait / hGather / hRec / hWait / merge / total), same shape
  as EEDI3's hbench.
- `tmp/ab_aa.py <frames> <reps> <ns>`: order-reversed same-session A/B of the
  chain vs the fused call on the real based_aa vpy (the grading tool).
- `tmp/aa_sweep.py`, `tmp/aa_kdiff.py`: ad-hoc oracle sweeps used during
  development (superseded by `tests/test_eedi3aa.py`).

## Tests

`tests/test_eedi3aa.py` (133 tests): u16 exact oracle over field 2/3, mdis
3/20/40, nrad 0..3, vcheck 0..3, alpha/beta/gamma corners, Gray8/16/32 mclip
present/absent, aliased and distinct sclip; the same for f32 with a 1e-6 bound
(measured max 5.96e-8); `_FieldBased` progressive/TFF/BFF input; YUV420 all
planes and planes=[0]; determinism, multi-stream and parallel load; props
(N frames, input fps, `_FieldBased` progressive) against the chain; input
validation. Plus 4 vsaa-wrapper tests in `tests/test_python_backend.py`.
Full suite: **585 passed**.

---

# ROUND 3 — benchmark/test fidelity to based_aa (this session)

The `eedi3aa` benchmark's reference arms were a hand-rolled reimplementation of
based_aa's chain whose comments (and the `transposed()` helper) assumed the
**opposite** of the plugins' real capabilities: eedi3vk2 was described as
"native EEDI3 then EEDI3H" and vszipcl/vszipcu as "no EEDI3H, use Transpose".
The truth (probed on the installed plugins and confirmed in the references —
`reference/vapoursynth-zipcl/src/vszipcl.zig:30` and
`reference/vapoursynth-zipcu/src/plugin.zig:46` register `EEDI3H`;
`reference/VapourSynth-eedi3vk2` has none):

| plugin | `supports_h` | `supports_mclip` | based_aa path |
|---|---|---|---|
| vszipcl | True | False | native `EEDI3H`, sclip only |
| vszipcu | True | False | native `EEDI3H`, sclip only |
| eedi3vk2 | **False** | **True** | transpose -> `EEDI3` -> transpose, **with mclip** |

`plugin_caps()` did detect this at runtime, so vszipcl/vszipcu already graded
correctly; the functional damage was the fallback arm: it hardcoded
`sclip=sclip` and dropped mclip, which is exactly the backend eedi3vk2 *does*
support it on.

**Measured impact (noise clip 640x360 -> 2x Point, 16 frames, ns=8,
`bench.py --filter eedi3aa`): eedi3vk2 at 129.6 fps without mclip vs 190.2 fps
with it (~+47%).** So the old fallback arm graded eedi3vk2 at a speed based_aa
never runs it at — a material unfairness to the reference. (Output happens to
be bit-identical with vs without mclip once sclip is supplied: eedi3vk2's
shader only uses the mask to skip DP work — see the "with mclip the DP is much
cheaper" note in `reference/VapourSynth-eedi3vk2/src/shaders/eedi3.comp:32` —
so this was a throughput error, not an output error.)

**Fix:** `_eedi3aa_build` no longer reimplements the dispatch. The reference
arms are the `vsaa` EEDI3 antialiaser itself —
`_VsaaEEDI3(backend=_VsaaEEDI3.Backend.<member>, ...).antialias(clip, tff=...,
sclip=clip, mclip=mclip, num_streams=N)` — so `should_h` / `supports_mclip` /
the double-rate `Interleave([s, s])` sclip / the `field = tff + double_rate*2`
derivation all come from vsaa and cannot drift again. `--eedi3-field` is mapped
back onto `(tff, double_rate)`. The fused arm still calls
`core.vsfeel.EEDI3AA` directly and takes the benchmark's already-interleaved
2N `sclip` (which is what `vsfeel.vsaa.EEDI3` hands it). Synthetic runs
(`--synthetic`, only used by the hang tests) emit a maskless chain over the
BlankClip.

Verified:
- `python -m pytest tests/test_eedi3aa.py tests/test_python_backend.py -q` →
  **145 passed** (the test oracle already mirrors based_aa's FEEL dispatch; no
  test encoded the inverted assumption, so nothing there needed changing).
- Real-clip run: `bench.py --filter eedi3aa --frames 8 --cache-frames 24
  --clip tests/noise_24f.mkv vsfeel vszipcl eedi3vk2` → 800.7 / 237.4 / 167.2
  fps, all plugins executing.
- Synthetic run: 149.7 / 137.9 / 77.3 fps (vsfeel / vszipcl / eedi3vk2).

---

# ROUND 4 — the real cost model (GPU stage profiler), and why fusion cannot reach 1.5x

**The ROUND 2 conclusion was wrong and is superseded.** It said "the honest reading is
GPU-bound … the four sub-passes cost the same on the GPU whether they run inside one call
or two". The first half is right, the second half hides the actual question, and the
*evidence* it rested on (the `VSFEEL_EEDI3AA_HBENCH` host split, "58 ms of 72 ms is fence
waits") is uninterpretable at ns=8 because those waits include the other seven streams'
interleaved work.

## 4.1 A working GPU stage profiler (`VSFEEL_EEDI3AA_GBENCH`)

Added to `src/eedi3.cpp`. `rocprofv3` **cannot see RADV dispatches** (`--kernel-trace`
produces an empty output dir; only HIP/HSA traces appear), so in-command-buffer timestamps
are the only attribution available on this box.

- One `VkQueryPool` **per submission** (`resource.gpu_pool[2]`); a single pool shared by
  both submissions never became available (`vkGetQueryPoolResults` + `WAIT_BIT` blocked
  forever, and the command-buffer copy form read back all zeros).
- `gpu_mark()` writes a timestamp before the first dispatch and after every stage of
  `record_pass` (pad / row / vcheck / tail), reading back host-side after the fence with
  `vkGetQueryPoolResults` — the shape bm3d's `BM3D_GPUTRACE` already uses.
- **`timestampPeriod` is 10 ns/tick on Navi31, not 1.** Assuming 1.0 under-reported every
  stage by 10x; the first "GPU is 4.6 ms of a 32 ms frame" reading was that bug.
- Usage: `VSFEEL_EEDI3AA_GBENCH=1 vspipe ...`, one `[eedi3aa-gbench] v …` / `h …` line per
  submission. Ignore the first ~10 frames (clock ramp).

## 4.2 What the GPU is actually doing (2x-2160p, mdis=20, vcheck=2, steady state)

Per output frame (four sub-passes, two per submission), from `_profaa.vpy` (ns=1):

| stage | per sub-pass | x4 / frame | share |
|---|---|---|---|
| row kernel (DP + backtrack + interpolate) | 2.6–2.9 ms | ~9.7 ms | **42%** |
| vcheck (serial row walk) | 2.7–2.8 ms (v) / 2.8–3.4 ms (h) | ~11 ms | **47%** |
| assemble-v | 0.61 ms | 0.61 ms | 2.6% |
| compose | 0.63 ms | 1.26 ms | 5% |
| pad builder | 0.03 ms (v) / 0.08 ms (h) | ~0.2 ms | <1% |
| xpose transpose | folded into h_pad | ~0.3 ms | 1% |
| **total** | | **~23.2 ms** | |

Corroboration: `gpu_busy_percent` polls at **100** throughout an ns=8 run, and the
end-to-end frame time matches the profiler total — the frame *is* GPU work.

Consequences for the earlier ablations:
- `VSFEEL_EEDI3_NOPAD` / `NOXPOSE` are worth ~0% end-to-end (measured ns=1: 30.86 → 31.01 →
  30.98 fps). ROUND 2 quoted +15%/+17% for these from a single ns=1 run at 700 frames —
  that was clock noise, and the profiler explains why: **pad and xpose together are under
  2% of the frame.** Do not spend anything on them.
- `VSFEEL_EEDI3_NOVC` (removes vcheck + its vcopy) is the only large lever: **84 → 111 fps
  at ns=8** (1.33x) on the 2x-2160p synthetic, matching the profiler's 47%.
- **1.33x is the ceiling for any vcheck-side work.** Even deleting the vcheck entirely does
  not reach 1.5x, and the remaining 53% (row kernel + tails) is the actual EEDI3 algorithm.

## 4.3 "Can the horizontal and vertical passes run at the same time?"

**Vertically no, horizontally partly — and neither converts into throughput here.** This was
asked directly, so the answer is recorded with its mechanism:

1. **V then H is a data dependency, not an implementation choice.** based_aa's
   `EEDI3.antialias` (and therefore the graded chain) runs VERTICAL first and feeds the
   *merged vertical output* to the HORIZONTAL pass: `out = Merge(H(Merge(V(x))))`. The
   horizontal DP's cost windows are computed **on the vertically interpolated values**.
   Running H on the original clip instead is a different algorithm and not a valid
   replacement for the reference.
2. **Fusing them into one kernel is not expressible either.** The H direction needs the
   vertical DP's *committed* result (post-backtrack, post-vcheck blend) on a whole
   neighbourhood of source rows at once, while the vertical walk is itself column-sequential
   (left-to-right within a row across all directions). A "do both DPs at once" kernel has to
   either (a) wait for the vertical walk anyway, or (b) branch into a 2D search that computes
   substantially *more* work than V-then-H. There is no shared term between the two DPs to
   hoist.
3. **The two sub-passes within a direction are genuinely independent** (V0 interpolates the
   even rows, V1 the odd rows; both read the same frame, and V0's result is only *merged*
   with V1, never accumulated). They are the only real concurrency available. But they are
   serialized today only by **shared scratch** (`pad_dev`'s built pad, pbt, and EEDI3H's R'
   are reused between the two sub-passes, so `record_pass` opens with a full compute
   barrier). Giving each its own pad/R' and interleaving the dispatches would let them
   overlap — and it would **not** raise ns=8 throughput, because the GPU is already 100%
   busy. It is a latency win for ns=1, not a throughput win, so it does not move the graded
   number. Recorded so it is not re-derived: the concurrency ceiling is already reached by
   the stream-level parallelism (ns=4 → 32 measured: 75.9 / 87.4 / 88.6 / 90.5 / 90.6 / 90.4
   fps — flat from 8 up).

## 4.4 Why the fused filter's speedup is structural, not tunable

Every one of the four passes is *necessary* work: two EEDI3 runs (one per interp parity) for
the vertical direction, two for the horizontal, with the mask dilation, the sclip gather and
the backtrack state each genuinely needed by the pass that consumes them. Fusion removed the
things that were actually redundant (the two `std.Merge` graph nodes, the V0/V1/v frame
materialisation, the repeated mask read) and that is worth the measured **1.07x**. The
remaining redundancy is small by the byte model and ~0 by the profiler: pad+xpose+compose+
assemble are 8.6% of the frame between them.

**To move the needle further the change has to be inside the vcheck or the row kernel, and
either one changes the shared EEDI3 core — which would speed up `EEDI3`/`EEDI3H` by the same
factor and leave the 1.07x ratio exactly where it is.** That is the real tension: EEDI3AA's
justification is *not* that it can be made 1.5x faster than the chain by tuning, because
there is no redundant work left to remove. It would need one of:

- **a parallel vcheck formulation** (the only path that breaks the 1.33x ceiling): the walk
  is sequential because each row's blend reads the *modified* previous row via `pr`. The
  `VCHECK_LDS` ping-pong exists but only engages when `WIDTH*4 <= MAXW_LDS` (~6k), so at
  3840/2160 wide it is off. A width-independent form (keep the previous row in registers via
  subgroup shuffles, or a tiled LDS window) is the untried fix. It is a pure-constant-factor
  change: **identical output**, so it is safe to land. Estimate from the profile: vcheck is
  47%, so even a 2x faster vcheck is ~1.2x end-to-end and 1.5x needs it *plus* the row
  kernel.
- **2-bit pbt packing** (row kernel is 42%; pbt is `W*rows*tpitch` int8 = 170 MB/sub-pass at
  this geometry, written and re-read by the backtrack). A 4x traffic cut is the only big
  structural idea left in the row kernel. The existing `EEDI3_PROBE` variants cannot measure
  it: **`PROBE=4` ("no pbt store") short-circuits the entire row kernel** (v_row 2.9 →
  0.005 ms), so it ablates the DP, not the stores. Any pbt experiment needs its own probe.
- **accuracy relaxation** (parallel vcheck reading the *un*modified previous row): rejected —
  it changes the output, and the whole AA test suite is a bit-exact oracle.

## 4.5 Tooling left in the tree

- `VSFEEL_EEDI3AA_GBENCH=1` — the GPU stage profiler (durable; this is the tool that was
  missing, and it inverted the earlier host-vs-GPU conclusion).
- `tmp/aa_ab.py <frames> <reps> <ns> [chain|fused|both]` — same-session order-reversed A/B
  that **does not discard vspipe stderr** (unlike `tmp/ab_aa.py`), so probes work.
- `tmp/aa_one.py <frames> [ns] [w] [h] [vcheck] [mdis] [nrad]` — one synthetic geometry,
  for sweeps and stream-count curves.
- `tmp/_profaa.vpy`, `tmp/_profaa8.vpy` — pre-warmed 2x-2160p real-clip vpys (ns=1 / ns=8)
  used for every number above.

---

# ROUND 5 — corrections to rounds 4.4/4.5, and the row-kernel cost model re-derived

Two things in the round-4 text are wrong; the rest stands.

1. **`VSFEEL_EEDI3AA_GBENCH` is not in the tree.** Round 4.5 calls it durable,
   but `grep GBENCH src/eedi3.cpp` finds nothing — the profiler was never
   committed. Re-deriving the stage split needs the `-DEEDI3_PROBE` ablation
   ladder (`notes/EEDI3.md` round 20) or a fresh profiler; do not go looking for
   the env var.
2. **The LDS vcheck IS engaged at the benchmark width.** 4.4 says it "only
   engages when `WIDTH*4 <= MAXW_LDS` (~6k), so at 3840 wide it is off". The gate
   is `use_lds = lds_ok && key.width <= MAXW_LDS` with `MAXW_LDS = 4096`
   (`src/eedi3.cpp:747,4126`) — a **column count**, not bytes. 3840 <= 4096, so
   the shipped vcheck is the shared-memory ping-pong form at the graded geometry.
   There is no width-independence win sitting there.

**Row-kernel cost model, measured (2000 f, ns=8, real based_aa clip, 5
interleaved reps, within-arm spread 1.3%):** the fused `EEDI3AA` arm runs
98.2 fps; with the row kernel deleted it runs 150.4 fps. So the four row-kernel
sub-passes are **~35% of the output frame** (vs ~17% for a single vertical
EEDI3), which is what makes them look like the right target. **But that 35% is
overstated**: the `PROBE=2` arm leaves `dmap` stale, so the vcheck takes its
cheap `dirc == 0` branch. The faithful ablation (`notes/EEDI3.md` round 20,
`PROBE=12`, dmap filled with the real pattern) measures ~1.25x -> **~25%**.
Use `PROBE=12` for any future EEDI3AA row-kernel ablation.

Both of the row kernel's named levers were then measured dead on the current
kernel (2-bit `pbt` packing; breaking the walk's serial load chain) — see
`notes/EEDI3.md` round 20 for the numbers and mechanisms. The one remaining
structural idea inside the row kernel is a walk dispatch with one lane per row
(~5-9% projected, not attempted). For EEDI3AA specifically, the frame is
dominated by the vcheck + the two extra sub-passes, not by `pbt`.

---
# ROUND 6 — the vcheck default flip and the VRAM-fused horizontal compose (+21%)

Two shared/AA changes landed this round; both are documented with measurements
in `notes/EEDI3.md` round 21. Summary for this file:

- **`ENTRY_VCHECK` now defaults to the global-read (empty-row-skipping) form**,
  not the LDS ping-pong round 14 promoted. The LDS walk must visit every row and
  flush it through LDS; the global form lets `ENTRY_VCOPY` handle the ~2/3 of
  fully-masked rows. `VSFEEL_EEDI3_VCLDS=1` restores LDS. Same-session A/B,
  1200 f x 4 order-reversed reps: fused EEDI3AA **89.7 -> ~101 fps** for this
  change alone.
- **The two horizontal composed planes are 50/50-merged inside `ENTRY_COMPOSE`
  in VRAM** instead of being written to staging and averaged by the CPU. A new
  `comp_fuse` push constant makes sub-pass 0 park its assembled plane in a
  device-local `o0` region and sub-pass 1 read it back, assemble O_1, average
  (`(a+b+1)>>1` u16 / `0.5a+0.5b` f32) and write the merged plane straight to
  the staging download; the CPU then does a plain row blit. `VSFEEL_EEDI3_AATIGHT=0`
  restores the old form; vcheck == 0 falls back automatically. Bit-exact vs the
  old path (30 real 4K frames, 497 MB per arm byte-identical) and +3.0%.

Combined same-session A/B against the pre-round defaults:
**EEDI3AA 89.7 -> 108.8 fps (+21.3%)**, and `benchmark/bench.py --filter
eedi3aa` measures ~121 fps at ns=8 (vszipcl 48.6, eedi3vk2 36.2). All 214
EEDI3-family tests pass.

---
# ROUND 7 — invalidate GPU-written staging before the CPU reads it

`Eedi3AaGetFrame` reads two GPU-written staging regions but never invalidated
them: the merged vertical frame (`v_offset`, written by `ENTRY_ASSEMBLEV` via
binding 10) read by `gather_horizontal`, and the two composed planes
(`out_offset`/`out2_offset`, written by `ENTRY_COMPOSE`) read by the final
merge. Both reads were guarded only by the `coherent` flag, which
`allocate_memory` can drop — on a non-coherent device the horizontal pass would
gather a stale `v`. `Eedi3GetFrame` already had the matching invalidate; the AA
path had only the flush half.

Fix: mirror EEDI3's block after each fence wait — one `vkInvalidateMappedMemoryRanges`
over `{upload_total + download_total + v_offset, v_bytes}` per processed plane
after the vertical fence, and one over `out_offset`/`out2_offset` after the
horizontal fence.

No measurement: the block is behind `!coherent` and this box's staging is
coherent, so it is a no-op here. Verified behaviourally with a temporary forced-
`coherent=false` build: 133/133 `test_eedi3aa.py` pass and the Khronos
validation layer emits zero VUIDs (ranges stay inside the staging allocation).

---
# ROUND 8 — atom-aligned ranges + Vulkan 1.3 preflight

`VkMappedMemoryRange` offsets/sizes must be multiples of
`minNonCoherentAtomSize` (256 here), or run to the end of the allocation. All
five filters' hand-built 32-byte-aligned ranges were replaced by the shared
`mapped_range`/`flush_range`/`invalidate_range` helpers in `vsfeel.h`, which
round the offset down, the size up, and fall back to `VK_WHOLE_SIZE` when the
allocation size is unknown or the rounded end would overrun it. EEDI3/AA's
identical per-plane flush entries (three copies of `[0, upload_total)`) collapsed
to one.

Same pass: `VK_EXT_subgroup_size_control` is only enabled when advertised (or
the device is 1.3+, where it is core), the device extension list is enumerated
once, `VkPhysicalDeviceProperties::apiVersion` is recorded and checked before
the SPIR-V 1.6 filters create a pipeline ("requires Vulkan 1.3 (device reports
X.Y)"), every `vkCreateComputePipelines` takes `pipeline_cache_lock`, and
`allocate_memory` refuses rather than silently relaxing a
`DEVICE_LOCAL|HOST_VISIBLE` request. On this box nothing changes behaviourally
(`api_version=1.4`, extension present, staging coherent); full suite green.

The fused horizontal gather shares EEDI3's `deint_row_u16/f32`, whose
streaming-store predicate tested the element index instead of the pointer (so
row cells not a multiple of 32 bytes silently took cached stores into the
uncached ReBAR window). See `notes/EEDI3.md` round 23 — `aa_gather_horizontal`
gets the same fix for free.

---

## Historical

The original file handoff, kept verbatim as the design brief it was:

> Status: **designed, not implemented.** This file is the handoff for a fresh
> session. Read `notes/EEDI3.md` round 19 (the last section of that file's
> "Remaining work") for the state of the EEDI3H filter this builds on.


