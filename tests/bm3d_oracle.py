"""Scalar model of mawen's V-BM3D block matcher.

This is the independent oracle for BM3D's matcher: it follows the pinned CPU
source (``reference/VapourSynth-BM3D``: ``VBM3D_Base.cpp``'s ``BlockMatching``
and ``Block.h``'s ``BlockMatchingMulti`` / ``GenSearchPos``) at search steps 1,
which is the sampling BM3D fixes. It is written from the reference, not from
the shader, so a shared implementation mistake is unlikely; the SSD reduction
does reproduce the shader's float32 accumulation order, because a different
summation order can flip a near-tie.

The GPU side is observed through ``VSFEEL_BM3D_MATCHTRACE``, which records the
group the kernel actually selected.
"""

import numpy as np

MAX_GROUP = 8

# The host scales the user's 8-bit MSE threshold by normY before converting it
# to the shader's SSD domain: the CPU implementation multiplies its thMSE by
# the norm of its color matrix's luma row, and vsfeel's sigma factors carry the
# same constant. 0.7496149945138504 is sqrt(0.2126^2 + 0.7152^2 + 0.0722^2),
# the bt709 row the CPU defaults to at HD.
MATRIX_NORM_Y = 0.7496149945138504


def matching_threshold(th_mse):
    """The shader's inclusive SSD bound for a user-domain th_mse."""
    if th_mse <= 0:
        return np.float32(0.0)
    return np.float32(th_mse * MATRIX_NORM_Y * (64.0 / (255.0 * 255.0)))


def ssd(cur, plane, cx, cy):
    """Sum of squared differences in the reference's accumulation order.

    ``cur`` is an 8x8 block indexed ``[column][row]`` and ``plane`` a 2D plane.
    The SSE path loads one row at a time with lanes over x, so lane j takes
    column j and column j+4 of each row before the next row; the four lanes then
    fold left to right. Verified against the kernel's own numbers through
    ``VSFEEL_BM3D_MATCHTRACE``.
    """
    a = [np.float32(0.0)] * 4
    for i in range(8):
        for j in range(4):
            d = np.float32(cur[j, i] - plane[cy + i, cx + j])
            a[j] = np.float32(a[j] + np.float32(d * d))
            d = np.float32(cur[j + 4, i] - plane[cy + i, cx + j + 4])
            a[j] = np.float32(a[j] + np.float32(d * d))
    return np.float32(np.float32(np.float32(a[0] + a[1]) + a[2]) + a[3])


def clipped_window(sx, sy, radius, w, h):
    """The CPU's ``_SearchBoundary`` window at step 1: 8x8 origins inside."""
    return (
        max(sx - radius, 0),
        min(sx + radius, w - 8),
        max(sy - radius, 0),
        min(sy + radius, h - 8),
    )


def union_positions(seeds, radius, w, h):
    """The CPU's ``GenSearchPos`` union, in ascending (y, x) order."""
    pos = set()
    for sx, sy in seeds:
        left, right, top, bottom = clipped_window(sx, sy, radius, w, h)
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                pos.add((y, x))
    return sorted(pos)


def rank(entries):
    """Stable sort by error of a (y, x)-ordered scan."""
    return sorted(entries, key=lambda e: (e[0], e[1], e[2]))


def search_frame(cur, plane, seeds, ps_range, w, h, th_sse):
    out = []
    for y, x in union_positions(seeds, ps_range, w, h):
        d = ssd(cur, plane, x, y)
        if d <= th_sse:
            out.append((d, y, x))
    return rank(out)


def match_group(anchor, x, y, frames, radius, bm_range, ps_num, ps_range, th_mse):
    """Return ``(final_group, info)`` for the reference block (x, y) of ``anchor``.

    ``final_group`` is a list of ``(frame, x, y, error)`` in the matcher's group
    order; ``info`` carries the current-frame list and the per-frame lists of
    both temporal directions.
    """
    h, w = frames[anchor].shape
    th_sse = matching_threshold(th_mse)
    cur = np.asarray(frames[anchor][y : y + 8, x : x + 8].T, dtype=np.float32).copy()
    ref = (np.float32(0.0), y, x)
    if th_mse <= 0:
        return [(anchor, x, y, np.float32(0.0))], {"current": [ref], "frames": []}

    left, right, top, bottom = clipped_window(x, y, bm_range, w, h)
    same = []
    for yy in range(top, bottom + 1):
        for xx in range(left, right + 1):
            if xx == x and yy == y:
                continue
            d = ssd(cur, frames[anchor], xx, yy)
            if d <= th_sse:
                same.append((d, yy, xx))
    current = [ref] + rank(same)[: MAX_GROUP - 1]
    combined = [(anchor, xx, yy, d) for (d, yy, xx) in current]
    info = {"current": current, "frames": []}

    nf = len(frames)
    for direction in (-1, 1):
        seeds = [(xx, yy) for (_, yy, xx) in current[: min(ps_num, len(current))]]
        for step in range(1, radius + 1):
            f = anchor + direction * step
            if f < 0 or f >= nf:
                break
            matches = search_frame(cur, frames[f], seeds, ps_range, w, h, th_sse)
            matches = matches[:MAX_GROUP]
            for d, yy, xx in matches:
                combined.append((f, xx, yy, d))
            info["frames"].append((f, matches))
            seeds = [(xx, yy) for (_, yy, xx) in matches[: min(ps_num, len(matches))]]

    if len(combined) <= MAX_GROUP:
        return combined, info
    best = sorted(range(1, len(combined)), key=lambda i: (combined[i][3], i))
    return [combined[0]] + [combined[i] for i in best[: MAX_GROUP - 1]], info
