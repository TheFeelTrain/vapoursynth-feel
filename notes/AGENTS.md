# Filter notes — conventions

**Binding for every file in this directory.** This file is injected
automatically whenever you read or edit anything under `notes/`. The fixed
shape, the budget and the rules that do not bend are requirements.

One file per filter (`BILATERAL.md`, `BM3D.md`, `DFTTEST.md`, `EEDI3.md`,
`EEDI3AA.md`, `GAUSSBLUR.md`, `NLMEANS.md`, `NNEDI3.md`), plus one cross-cutting
file (`METHOD.md`). They are **tracked** in git: the durable design record, not
scratch memory.

These files are read by people deciding what to try next. Length is not a
virtue and the history is not the point — the *current design*, the *measured
mechanisms*, and the *open questions* are. A change that needs three paragraphs
is rare; most need one bullet.

## The gate

`tools/notes_check.py` — the `notes` gate in `tools/lint.sh`, which CI runs on
every push — checks the mechanical half of this file: the title, the part names
and their order (for filter notes), the line budgets, the file index above, and
report IDs in the text. A note that fails it is a defect to fix, not a style
disagreement.

## Fixed shape (filter notes, in this order)

1. `# <Filter> — notes` — exact form, no suffix, no port tag.
2. **Status banner** (no heading) — the state (`shipped` / `iterating` /
   `designed`), the design that ships, and the scoreboard table. A reader who
   stops here must not be misled about what the code does today.
3. `## Implementation` — the structure that ships, derived from `src/`.
4. `## Performance` — optional; the mechanisms behind the current numbers.
5. `## Historical` — superseded rounds, in the order they happened.
6. `## Open work` — remaining paths, do-not-retry list, method rules.
7. `## Debug env vars` — the live knobs, one line each.

Use these part names even where a file has only a stub for a part, and no
others at the top level: anything else is a `###` subsection of the part it
belongs to (`### Do not retry` under **Open work**, `### Tests` under
**Implementation**). Consistency across filters is worth more than a slightly
better bespoke heading.

The scoreboard **table** lives in the status banner. `## Performance` explains
the mechanisms behind those numbers and never repeats the table.
Cross-cutting correctness passes (validation hardening, error-path fixes,
`sfence` ordering) are not rounds of their own: at most one bullet each, under
**Historical**.

## Cross-cutting notes

`METHOD.md` documents method that belongs to no single filter: the order to work
in (its `## Typical workflow`), comparing a vsfeel kernel against the reference
kernels, and the porting discipline. The fixed shape above is for filter notes; a
cross-cutting note carries its own `##` sections instead, with no status banner,
scoreboard or `## Historical`. The rest still binds it: the title form, the
writing style below, dead ends keeping their mechanism, and a 350-line budget on
the whole file — a filter note's live-part budget, which is what it has instead
of a history section. `tools/notes_check.py` holds it to exactly that.

## Writing style

- **Conclusion first.** Each section's first line says what was decided or
  measured; evidence follows. Never make the reader reach the end to learn.
- **Tables for numbers, bullets for mechanisms, prose for neither.** If a
  paragraph has more than ~4 lines, it should be a bullet list.
- **One entry per round or date**, inside `## Historical`: either a
  `### Rounds N–M — <claim>` / `### <date> — <claim>` heading, or a
  `- **<date> — <claim>.**` bullet. It states the result, not the topic.
- **Quote a number only with its config** (depth, geometry, streams, sigma,
  frames) and whether it was a graded median or a screen.
- **A correctness-only change gets one line**: what was wrong, what the fix is,
  and "no perf change". Do not narrate the debugging.
- **A perf change gets**: the mechanism, the before → after fps, the config,
  and whether it was re-measured after any later structural change.
- **Record the mechanism of a failure, not the story of finding it.** Steps
  that did not change the conclusion are noise and are the main source of bloat.

## Keeping it short

- Keep the whole file under ~700 lines and the **live part** — everything
  except the `## Historical` section — under ~350. Both are limits on new
  writing, not targets to grow into: when a file crosses one, compress rather
  than append.
- **Delete perf conclusions that were measured on a broken benchmark.** If the
  harness, input, or mask was wrong, the round's numbers are void: keep the
  accuracy proofs, correctness fixes and mechanisms, and replace the perf record
  with one explicit "void, do not quote" note. Do not leave a void number looking
  authoritative because it was once believed.
- Do not restate the algorithm, paste probe logs, or copy code; link to the
  source file and line instead.
- Do not re-derive what another filter's note already established. A one-line
  pointer ("same mechanism as `notes/BILATERAL.md`'s HD path") is enough.
- Superseded detail is deleted, not archived. `## Historical` keeps the
  *mechanism* of a dead end (one or two bullets) plus its replacement — not the
  full original section.
- Meta-commentary about the note itself ("historical:", "kept for the
  mechanism", "do not trust this") belongs only where a reader would otherwise
  act on the stale text. One marker, not a disclaimer per paragraph.

## Rules that do not bend

- **Top of file is the current truth.**
- **Append-only history.** Never rewrite an old round to match new code; mark
  it superseded and move it under `## Historical` instead of deleting it.
- **Dead ends keep their mechanism** — which configuration, why it lost — so a
  later variant is judged against the failed mechanism, not the verdict.
- **Cross-check every number and symbol against `src/` before quoting it.** A
  drifted note is worse than no note.

## Test helpers that cover several filters

Test infrastructure shared by more than one filter belongs in `tests/conftest.py`
and gets one line here, not a section per note.

- `assert_temporal_order_consistent(filter, params, tol, nframes=...)` — runs the
  node in six frame-request orders (forward/reverse/far/interleave/scramble/
  revisit) on a fresh instance each and compares to the serial run, under the
  subprocess timeout. Temporal caches (DFTTest slots, NLMeans tiles, BM3D
  ring/res) are order-sensitive by construction and sequential `get_frame` never
  re-requests an in-flight frame, so this is the request pattern vspipe actually
  produces. `nframes=1/2` builds the short-clip cases. Tolerance is 0 for the
  slot/tile caches; BM3D uses 1e-5 because its atomicAdd aggregation has a
  ~3e-8 ordering floor.
- `source_clip()` / the `nframes` spec field — every clip a test or comparison
  subprocess builds from the committed 300-frame file must be trimmed to
  `CLIP_FRAMES`, the length the fixtures expose. Otherwise a temporal filter's
  last frame sees a different window than the fixture it is compared against,
  and a frame-doubling filter (EEDI3/NNEDI3 `field>1`) doubles 300 frames
  instead of 24.

## Short vs long — worked examples

Correctness-only fix:

```markdown
### Round 7 — invalidate GPU-written staging before the CPU reads it
u16/f32 download read stale cache lines because the invalidate range was
missing on the cached path. Now invalidated per plane; no perf change.
```

A perf round that earned its space:

```markdown
### Round 10 — host path was the wall (+15% at ns=4)
CPU staging, not the kernel, was the limiter (upload gather 6.1 ms/frame).
Replaced the serial scan with a SIMD dilation: 6.1 → 0.28 ms/frame, ns=4
1530 → 1760 fps (jpbd 1080p GRAY16, 5000-frame medians). Re-swept ns: knee
8 → 12.
```
