#!/usr/bin/env python3
"""Check notes/<filter>.md against the conventions in notes/AGENTS.md.

notes/AGENTS.md is the authority on the notes: the fixed shape, the two line
budgets and the rules that do not bend. Half of that is prose no machine can
read; this is the other half, and it is what makes the conventions hold -- a note
that drifts out of shape fails here instead of quietly teaching the next
contributor that the rules are optional.

    uv run python tools/notes_check.py             # every note in notes/
    uv run python tools/notes_check.py notes/BM3D.md
    uv run python tools/notes_check.py --budget    # print both budgets per note

Checked, per note:

  * the title is exactly `# <Filter> — notes`;
  * the top-level parts are the fixed ones -- `## Implementation`,
    `## Performance` (optional), `## Historical`, `## Open work`,
    `## Debug env vars` -- under those exact names, in that order, with no other
    `##` heading anywhere (`###` subsections are free);
  * the whole file is under 700 lines, and the live part -- everything outside
    `## Historical` -- is under 350;
  * no report IDs (`WO-55`, `§I.11`) leaked in from a work order;
  * no trailing whitespace, and the file ends in exactly one newline.

And once for the directory: the file index in notes/AGENTS.md lists exactly the
notes that exist, so adding a filter cannot leave the index stale.

notes/AGENTS.md is not itself a note and is never checked against this shape.
Headings inside fenced code blocks are ignored, so the worked examples it
carries are not mistaken for real parts.
"""

import argparse
import re
import sys
from pathlib import Path

TITLE = re.compile(r"^# .+ — notes$")
PART = re.compile(r"^## (.+?)\s*$")
FENCE = re.compile(r"^\s*```")
TRAILING = re.compile(r"[ \t]+$")
INDEX_ENTRY = re.compile(r"`([A-Za-z0-9_.-]+\.md)`")

# The fixed shape, in order. `Performance` is optional but keeps its slot; every
# other part is required. notes/AGENTS.md is the authority on both lists.
SHAPE = ("Implementation", "Performance", "Historical", "Open work", "Debug env vars")
REQUIRED = ("Implementation", "Historical", "Open work", "Debug env vars")
HISTORICAL = "Historical"
TOTAL_MAX = 700
LIVE_MAX = 350

# Banned in code, comments and notes alike: they name a work order, not a thing
# a reader of the tree can look up.
REPORT_ID = re.compile(r"WO-\d+|§[IVX0-9]")

CONVENTIONS = "AGENTS.md"


def unfenced(lines):
    """Yield (line number, text) for every line outside a fenced code block."""
    fenced = False
    for number, line in enumerate(lines, start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield number, line


def parts(lines):
    """Yield (line number, name) for every top-level heading outside a fence."""
    for number, line in unfenced(lines):
        match = PART.match(line)
        if match:
            yield number, match.group(1)


def live_lines(lines, found):
    """Count the lines outside the `## Historical` section, the live part."""
    start = next((number for number, name in found if name == HISTORICAL), None)
    if start is None:
        return len(lines)
    end = next((number for number, name in found if number > start), len(lines) + 1)
    return len(lines) - (end - start)


def check_note(path):
    """Return the problems in one note as (line number, message) pairs."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    found = list(parts(lines))
    problems = []

    if not lines or not TITLE.match(lines[0]):
        problems.append((1, "title must be exactly '# <Filter> — notes'"))

    names = [name for _, name in found]
    for number, name in found:
        if name not in SHAPE:
            problems.append(
                (number, "unknown part '## %s'; the fixed parts are %s" % (name, ", ".join(SHAPE)))
            )
    for name in REQUIRED:
        if name not in names:
            problems.append((1, "missing required part '## %s'" % name))

    rank = -1
    for number, name in found:
        if name not in SHAPE:
            continue
        position = SHAPE.index(name)
        if position <= rank:
            problems.append(
                (
                    number,
                    "'## %s' is out of order; the fixed order is %s" % (name, ", ".join(SHAPE)),
                )
            )
        rank = max(rank, position)

    total = len(lines)
    if total > TOTAL_MAX:
        problems.append((total, "file is %d lines, over the %d-line budget" % (total, TOTAL_MAX)))
    live = live_lines(lines, found)
    if live > LIVE_MAX:
        problems.append(
            (
                1,
                "live part is %d lines, over the %d-line budget (compress, do not append)"
                % (live, LIVE_MAX),
            )
        )

    for number, line in unfenced(lines):
        if REPORT_ID.search(line):
            problems.append((number, "report ID in the text; notes name things, not work orders"))
    for number, line in enumerate(lines, start=1):
        if TRAILING.search(line):
            problems.append((number, "trailing whitespace"))

    if not text.endswith("\n"):
        problems.append((total, "file must end with a newline"))
    elif text.endswith("\n\n"):
        problems.append((total, "file must not end with a blank line"))

    return problems


def check_index(directory):
    """Return the problems with the notes index in notes/AGENTS.md."""
    conventions = directory / CONVENTIONS
    if not conventions.is_file():
        return [(0, "%s is missing; it is the authority these notes follow" % conventions)]
    listed = set(INDEX_ENTRY.findall(conventions.read_text(encoding="utf-8"))) - {CONVENTIONS}
    actual = {path.name for path in directory.glob("*.md")} - {CONVENTIONS}
    problems = []
    for name in sorted(actual - listed):
        problems.append((0, "%s exists but is not listed in %s" % (name, conventions)))
    for name in sorted(listed - actual):
        problems.append((0, "%s is listed in %s but does not exist" % (name, conventions)))
    return problems


def note_paths(targets):
    """Resolve the targets to note files, leaving the conventions file out."""
    paths = []
    for target in targets:
        paths.extend(sorted(target.glob("*.md")) if target.is_dir() else [target])
    return [path for path in paths if path.name != CONVENTIONS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="notes or directories of notes to check (default: this checkout's notes/)",
    )
    ap.add_argument(
        "--budget", action="store_true", help="print the line budgets instead of checking"
    )
    args = ap.parse_args()

    default_dir = Path(__file__).resolve().parent.parent / "notes"
    targets = args.paths or [default_dir]
    files = note_paths(targets)

    if args.budget:
        print("%-24s %6s %6s" % ("note", "lines", "live"))
        for path in files:
            lines = path.read_text(encoding="utf-8").splitlines()
            print("%-24s %6d %6d" % (path.name, len(lines), live_lines(lines, list(parts(lines)))))
        return 0

    failed = 0
    for path in files:
        for number, message in check_note(path):
            print("%s:%d: %s" % (path, number, message))
            failed = 1
    for target in targets:
        if not target.is_dir():
            continue
        for number, message in check_index(target):
            print("%s:%d: %s" % (target / CONVENTIONS, number, message))
            failed = 1

    conventions = next((target for target in targets if target.is_dir()), default_dir) / CONVENTIONS
    if failed:
        print(
            "\nnotes conventions: %s (checked by tools/lint.sh notes)" % conventions,
            file=sys.stderr,
        )
    else:
        print("%d notes match the conventions in %s" % (len(files), conventions))
    return failed


if __name__ == "__main__":
    sys.exit(main())
