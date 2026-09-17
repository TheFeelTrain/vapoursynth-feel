# Filter notes — conventions

One file per filter (`BILATERAL.md`, `BM3D.md`, `DFTTEST.md`, `EEDI3.md`,
`EEDI3AA.md`, `GAUSSBLUR.md`, `NLMEANS.md`, `NNEDI3.md`). They are **tracked**
in git: the durable design record, not scratch memory.

These files are read by people deciding what to try next. Length is not a
virtue and the history is not the point — the *current design*, the *measured
mechanisms*, and the *open questions* are. A change that needs three paragraphs
is rare; most need one bullet.

## Fixed shape (every file, in this order)

1. `# <Filter> — notes` (exact form; no "performance notes", no port tag).
2. **Status banner** — one of `shipped` / `iterating` / `designed`, the current
   design in at most ~10 lines of bullets, and the current scoreboard. A reader
   who stops here must not be misled about what the code does today.
3. **Implementation** — the structure that ships, derived from `src/`.
4. **Performance** — one table of the current graded numbers, with the config
   in the caption.
5. **Historical** — superseded rounds, in the order they happened.
6. **Open work** — remaining paths, do-not-retry list, method rules.
7. **Debug env vars** — the live knobs, one line each.

Use these part names even where a file has only a stub for a part. Consistency
across filters is worth more than a slightly better bespoke heading.

The **Performance** table lives in the status banner — do not repeat it as a
separate section. Cross-cutting correctness passes (validation hardening,
error-path fixes, `sfence` ordering) are not rounds of their own: at most one
bullet each, under **Historical**.

## Writing style

- **Conclusion first.** Each section's first line says what was decided or
  measured; evidence follows. Never make the reader reach the end to learn.
- **Tables for numbers, bullets for mechanisms, prose for neither.** If a
  paragraph has more than ~4 lines, it should be a bullet list.
- **One entry per round or date**, headed `## Round N — <claim>` or
  `## <date> — <claim>`. The heading states the result, not the topic.
- **Quote a number only with its config** (depth, geometry, streams, sigma,
  frames) and whether it was a graded median or a screen.
- **A correctness-only change gets one line**: what was wrong, what the fix is,
  and "no perf change". Do not narrate the debugging.
- **A perf change gets**: the mechanism, the before → after fps, the config,
  and whether it was re-measured after any later structural change.
- **Record the mechanism of a failure, not the story of finding it.** Steps
  that did not change the conclusion are noise and are the main source of bloat.

## Keeping it short

- Keep the whole file under ~500 lines, with the **live part** (banner through
  open work) under ~250. That matches the healthy files (`BILATERAL` 257,
  `BM3D` 224, `GAUSSBLUR` 101) and puts `DFTTEST` (514) right at the ceiling;
  it is a limit on new writing, not a target to grow into. Files over it
  (`EEDI3AA` 946, `NLMEANS` 702, `NNEDI3` 700) get their history compressed
  when touched, never appended to. `EEDI3.md` is the one accepted exception —
  it is the port's full accuracy record.
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

## Short vs long — worked examples

Correctness-only fix:

```markdown
## Round 7 — invalidate GPU-written staging before the CPU reads it
u16/f32 download read stale cache lines because the invalidate range was
missing on the cached path. Now invalidated per plane; no perf change.
```

A perf round that earned its space:

```markdown
## Round 10 — host path was the wall (+15% at ns=4)
CPU staging, not the kernel, was the limiter (upload gather 6.1 ms/frame).
Replaced the serial scan with a SIMD dilation: 6.1 → 0.28 ms/frame, ns=4
1530 → 1760 fps (jpbd 1080p GRAY16, 5000-frame medians). Re-swept ns: knee
8 → 12.
```
