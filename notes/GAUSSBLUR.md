# GaussBlur — notes

Status: **iterating** — beats vszipcl, still trails vszipcu. Config: jpbd
1920x1080 GRAY16, σ=16 (radius 48 → two-pass `ENTRY_VERT`/`ENTRY_HORIZ` path),
1000 frames.

| ns | vsfeel | vszipcl | vszipcu |
|---|---|---|---|
| 4 (shipped) | **2577** | 1414 | 3431 |
| 8 | **2522** | 1425 | 3711 |

Knee is 4 (8 is flat), so the filter is GPU-side limited, not
overlap-limited. The remaining gap is to vszipcu's HIP path, not to OpenCL.
`num_streams` defaults to 1 in the plugin (the benchmark passes 4); per-stream
VRAM is ~4 MB src + ~4 MB dst + staging.

Bench: `MANGOHUD=0 python3 benchmark/bench.py --filter gaussblur
[--gauss-sigma X] [--num-streams N] vsfeel vszipcl vszipcu`.
Target GPU: RX 7900 XTX (gfx1100), RADV, ReBAR (`memoryType 3`).

## Implementation

Mirrors the Bilateral VRAM structure (see `notes/BILATERAL.md`):

- Per-stream device-local `src_buf` + `dst_buf` in VRAM holding **native
  `uint16_t`/`float` elements**, plus a host-visible cached GTT `staging`
  (upload + download regions).
- **Host-direct upload** (default; `VSFEEL_GAUSS_HD=0` opts out): `src_buf` is
  ReBAR host-mapped and the CPU memcpys each plane straight into VRAM — no H2D
  copy. Falls back to staging + in-CB `vkCmdCopyBuffer` when no host-visible
  device-local type exists; the flag is decided once before the resource loop.
- **Kernel-direct download** (default; `VSFEEL_GAUSS_KD=0` opts out): the blur
  kernels' coalesced stores write the GTT staging download region directly, so
  there is no D2H copy. With both on, per-frame GPU work is **kernel only**.
- Native `uint16_t`/`float` SSBOs: no shift/mask unpacking, no dword `atomicOr`
  packing, no per-frame `vkCmdFillBuffer`. Push constants are element offsets.
- **Packed pair stores** (`store_row_packed`, binding 4 `DstU32`): the
  horizontal pass writes R outputs as `R/2` 32-bit stores; `x0` is guaranteed
  even so each thread owns its dwords. The vertical pass writes float `tmp`.
- Workgroup `16x8`; queue cap 1 (`VSFEEL_GAUSS_QUEUES`).

## Reference structure (same GPU, READ-ONLY)

- **vszipcu (fastest):** device-local `d_src`/`d_dst` + float `d_tmp`. Per
  plane: async `memcpyHtoD` (dedicated copy stream) → kernels on the compute
  stream → async `memcpyDtoH`, then sync both. Still serial per frame, but the
  copies overlap the previous plane's kernels and run on SDMA engines
  full-duplex with compute.
- **vszipcl:** same shape with OpenCL pinned staging + `clFinish` per frame;
  already beaten.
- `rocprofv3` profile of vszipcu (BlankClip gray16 1080p, σ=16, ns=1, 200
  frames): vertical 118 µs, horizontal 177 µs, H2D 153 µs, D2H 152 µs —
  kernels 295 + copies 305, ≈ PCIe-bound.

## Open work

- **No HD/KD ablation is recorded for GaussBlur.** The +34%/+3% figures once
  quoted here were Bilateral's. Sweep `VSFEEL_GAUSS_HD` × `VSFEEL_GAUSS_KD` in
  one session before quoting a split.
- Replace `VSFEEL_GAUSS_GPU_BENCH` with the warm in-command-buffer timestamp
  form (eedi3/nnedi3/dfttest) before attributing any kernel-vs-copy split.
- Then sweep the workgroup shape across σ ∈ {4, 16, 40, 80}.

## Historical

- First structure (deleted in the VRAM rework): GTT staging only, kernels
  reading/writing sysmem over PCIe, 16-bit writes via dword `atomicOr` RMW with
  a per-frame `vkCmdFillBuffer` pre-clear, fully synchronous per frame, all 4
  compute queues at ns=4.
- VRAM rework (device-local `src`/`dst`/`tmp` + one H2D and one D2H
  `vkCmdCopyBuffer`): vsfeel **1808** (was 1191–1342) vs vszipcl 1412, vszipcu
  3489.
- **CPU reads from the VRAM BAR are ~1.3 GB/s NT / 0.1 GB/s memcpy** (64 MiB
  probe, memoryType 3) → mapped-VRAM staging is dead for the read path.
- **Plain memcpy into the mapped VRAM window beats NT stores** on upload: 22.5
  vs 11 GB/s in the probe, and in situ the WC window is why the shipped
  `copy_plane_out` passes no NT flag. This is the *opposite* of the
  EEDI3/NNEDI3 upload verdict — the path is buffer- and shape-dependent, so
  re-measure rather than inheriting it.
- rocprofv3 does **not** see RADV/Vulkan work, so kernel times must come from
  in-command-buffer timestamps or fps.

## Debug env vars

- `VSFEEL_GAUSS_HD=0` — staging + H2D copy instead of host-direct upload.
- `VSFEEL_GAUSS_KD=0` — VRAM dst + D2H copy instead of kernel-direct download.
- `VSFEEL_GAUSS_QUEUES=N` — override the queue cap (default 1).
- `VSFEEL_GAUSS_GPU_BENCH=N` — retained scratch probe: re-records the
  production CB and idles the device from a frame callback **without holding
  the other streams' queue locks**, so its per-stage numbers are not
  comparable to the in-CB timestamp form. Do not trust it.

## Validation hardening

Upload flush / download invalidate ranges go through the shared `mapped_range`
helper (`minNonCoherentAtomSize`/`VK_WHOLE_SIZE` rounding); no behaviour change
on this coherent device. On the staging path the upload is NT-stored, so an
unconditional `_mm_sfence()` now sits before `submit_with_fence`. Creation
errors are torn down by `~GaussData` via `FramePool::emplace()`, and the
frame-path `set_error` frees `dst`. All correctness-only; `test_gaussblur.py`
passes.
