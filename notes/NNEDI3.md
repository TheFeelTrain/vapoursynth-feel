# NNEDI3 — notes

*vsfeel's Vulkan port of nnedi3vk (refs `VapourSynth-nnedi3vk` / `vapoursynth-zipcu` / CPU `znedi3`).*

Status: **shipped on the R80 GPU API.** Design (current):

- `clip:vnode:gpu` in, `ffGPUOutput` out: the core owns every transfer
  (`std.GPUUpload`/`std.GPUDownload`); the filter owns its pipelines, three
  weight buffers and one exec pool — no staging, per-stream resources,
  download-slot pool, descriptor pool, fences, queue cap or queue choice.
  `num_streams` and `device_id` are registered no-ops (never read), and the
  device is chosen by `core.set_vulkan_device`.
- The kernels read the source plane in place (field row `f` → source row
  `DH ? f : 2f+parity` at the plane's own pitch) and write interpolated row `r`
  straight into the output plane at `2r + (1-parity)`. `ENTRY_KEEP` copies the
  kept rows and, on a skipped plane under `dh`, zero-fills the interpolated
  rows. `DH`/`ZERO` are spec constants (ids 9/10; 11 spec entries total).
- Per frame: one transient device-local scratch (`width*rows*4 B` list +
  16 B indirect struct per plane) handed to the recording context; five push
  words `{list_elem, src_stride, dst_stride, parity, ind_elem}`.
- Persistent VRAM: the three weight buffers (worst config qual2/nsize3/nns4 ≈
  1.2 MB, host-visible, device-local preferred) plus pipelines; transient
  4.2 MB scratch per in-flight frame (1080p GRAY16 `field=3`), bounded by the
  exec ring; src/output frames are core-owned.
- Runs need `RADV_EXPERIMENTAL=transfer_queue`, which `tools/benchmark.py`
  forces (mechanism: `notes/BILATERAL.md`).
- The filter sits at the API's transfer ceiling: a bare core
  `GPUUpload→GPUDownload` graph (same cached real-clip vpy, no filter, 5000
  frames) runs 2316 fps.

Scoreboard — jpbd 1080p GRAY16, bench defaults (`field=3 dh=0 nsize=0 nns=4
qual=2 etype=0 pscrn=4`), interleaved pre-R80/R80 rounds through
`tools/benchmark.py`, arm order alternated, medians:

| input to vsfeel | pre-R80 | R80 port | delta |
|---|---|---|---|
| CPU cache (benchmark default), 5000 f ×3, 4 rounds | 3034 | 2695 | **−11.2%** |
| `--gpu-cache` (resident GPU clip), 3000 f ×2, 3 rounds | 2156 | 2739 | **+27.0%** |

- References, same-session pairs: CPU cache (5000 f ×3) — vsfeel **2652**,
  nnedi3vk 2908 → **0.91x**; `--gpu-cache` (3000 f ×2) — vsfeel **2740**,
  nnedi3vk 2031 → **1.35x (#1)**. The port wins exactly where a resident chain
  is involved; the CPU-sink row keeps the pre-port reference shape inverted for
  the mechanism below.
- Mechanism of the CPU-sink delta (same as `notes/GAUSSBLUR.md` and
  `notes/BM3D.md`): the pre-R80 kernel stores *were* the download (D2H of the
  interpolated half only) and the NT ReBAR pack *was* the upload, so the old
  build moved one full-frame DMA less than the core's round trip and never ran
  `GPUDownload` at all. Forcing **one** core download into the old build's own
  chain (`--gpu-cache` row above) drops it 3034 → 2156 fps (−29%): the delta
  is the deleted transfer path, not the filter. Nothing in-filter can remove it
  under the `vnode:gpu` contract — do not chase it here.
- Kernels at parity: `VSFEEL_NNEDI3_TSTAMP=100` (GRAY16, warm frame) gives
  pre 76 / pred 192 / total 268 µs — pred against the pre-port uncontended
  195–200, and the pre-port segment includes keep + indirect fills where the
  old stamp was prescreen alone. The old inline copy stamp (89 µs D2H on the
  compute queue) is gone from compute entirely. Host split
  (`VSFEEL_NNEDI3_TIMING=1`, 5000 f): record 7.5 / submit 58.9 µs per frame —
  acquire's ~1000 µs mean is the exec ring wait, a sleep, not host work.
- Agreement: nnedi3vk vs vszipcu **BIT-EXACT** (maxdiff 0, frames 0/5/11, all
  planes, noise_24f → YUV420P8 field=1); vs CPU znedi3 maxdiff 5, ~1 % px
  (CPU float ordering) — **nnedi3vk is ground truth**.
- Accuracy (unchanged by the port, `tests/test_nnedi3.py` green under
  `cpu_node`): 16-bit within **1 LSB** (isolated 1-code flips from
  serial-vs-butterfly reduction order), GRAYS within **~3e-8** (bound 1e-6).
  Sweeps field 0/1/2/3, dh, planes subsets, nsize 0..6, nns 0..4, qual 1/2,
  etype 0/1, pscrn 0..4 across GRAY16/YUV420P16/GRAYS, plus determinism,
  1-vs-4 streams, parallel load, geometry tails and props. Full suite: 800
  green via `tools/test.sh` (one unrelated NLMeans `a=64,d=16` contention
  flake, passes serially — see `notes/NLMEANS.md`).
- Weights `src/nnedi3_weights.bin`, 13,574,928 B, md5
  `5c97e25c4a7277d06d3e3851373f1065`.

## Implementation

`src/nnedi3.cpp` + `src/nnedi3.comp`.

- Frame path (`Nnedi3GetFrame`): source frame → output frame
  (`newGPUVideoFrame`, or `newVideoFrame2` sharing unprocessed planes —
  **always fresh under `dh`**, since the keep kernel expands them there) →
  parity from `_Field`/`_FieldBased`/`field` (double-rate flips on odd `n`) →
  acquire, transient scratch (list mode only), record per plane: keep →
  fill the indirect struct `{0,1,1,0}` → transfer→compute barrier →
  prescreen → compute→(compute|indirect) barrier → indirect predict →
  declare reads/writes → submit → props (`_FieldBased=0`, `_Field` deleted,
  `field>1` halves duration). Plane regions are disjoint, so nothing else is
  ordered.
- Grids: prescreen `ceil(rows*ceil(width/P)/128)` (P=1 at pscrn=1 else 4 —
  grouping is per ROW, so this exact form is what covers non-divisible row
  tails); predict indirect off the prescreen count (prescreen keeps `groupsX`
  via `atomicMax` off its `atomicAdd` — no count kernel); direct grid
  `ceil(width*rows/(4*PXP))` for pscrn=0; keep grid linearized over x and y
  (`ID.y * groups.x * 256 + ID.x`) so a 4K double-rate plane stays under the
  65535-group x limit.
- Pipelines deduplicated per `{width, rows, pscrn, xdim, ydim, nns, qual,
  zero}`; predict module by the shader's own rule (PXP=8 when
  `ceil(nns/32)<=2 && fs<=128`, PXP=4 small-tile `SHSTRIDE=64` when `fs<=64`,
  PXP=4 otherwise). Prescreen/predict ask 32-lane full subgroups; the keep
  writer takes the driver's default.
- The predict module's tile is `4 * SHSTRIDE` vec4 rows: 16 KiB for PXP=8
  (256), 18 KiB for `n4` (288, every PXP=4 window), 12 KiB for `n4m` (192) and
  4 KiB for `n4s` (64). `n4m` takes over from `n4` only where the device cannot
  hold 18 KiB and the window is at most 192 rows, so the measured module still
  runs wherever it fits. The FS=96 `n4m` arm is bit-identical to `n4`
  (`tests/test_device_limits.py`, sha1 of the plane).
- Weights: blob parsed at creation (prescreener mean/scale, model
  mean-subtraction projected out), packed into three vec4/vec2-layout blobs
  and written through persistently mapped host-visible buffers + `_mm_sfence`.
- `field` is registered **required** (the `:opt` + null-error-pointer history
  killed the process); 16-bit integer and 32-bit float input only; `field>1`
  doubles `vi_out` with the unknown-length (`numFrames == -1`) guard.
- `vsfeel/vsaa.py`: `NNEDI3(vsaa.deinterlacers.NNEDI3)` overrides only
  `_deinterlacer_function` to return `core.vsfeel.NNEDI3` (field/dh and the
  `nsize`/`nns`/`qual`/`etype`/`pscrn` mapping come from the base class, so
  they cannot drift); `vsfeel.NNEDI3` is a PEP-562 lazy re-export alongside
  `EEDI3`, so `import vsfeel` never needs vsaa.

## Historical

Pre-R80 rounds below were measured on the deleted transfer path — their fps
figures are void as current-filter numbers (same marker as
`notes/BILATERAL.md`); mechanisms are kept.

- **The coop kernels' subgroup requirements are checked, not assumed** — prescreen
  and predict reduce across lanes with subgroup arithmetic and count lanes with
  ballot, and only BASIC is mandatory in Vulkan. Both are required at creation
  (ballot only when `use_list` builds a prescreen), and the 32-lane request is
  resolved against the device instead of being passed through unconditionally.
- **The 18 KiB predict tile had no smaller variant for a 16 KiB device** — `n4`
  is `4 * 288` vec4, more than a device may report, so `n4m` (`SHSTRIDE=192`,
  12 KiB) now covers every window but the 48x6 network and FS=288 reports
  `18432` against the limit instead of failing inside the driver. Every pipeline
  is measured against the device the same way (`gpu_create_pipeline`), so a
  128-invocation device gets "needs a 32x8x1 workgroup (256 invocations)" per
  kernel rather than a `vkCreateComputePipelines` failure; `tools/shader_limits.py`
  prints the per-variant numbers those checks are built from, and
  `VSFEEL_LIMIT_SHARED_MEMORY` / `VSFEEL_LIMIT_INVOCATIONS` clamp the reported
  limits down so the paths are testable on a GPU that fits everything
  (`tests/test_device_limits.py`).

### 2026-09-23 — R80 GPU API port

- Ripped: ReBAR upload staging + `copy_stream_rows` pack, the kept-line host
  memcpy (pre-`take` overlap), the download-slot pool and host interp
  scatter, two pre-recorded parity CBs + the per-frame D2H copy CB,
  `FramePool<Nnedi3Resource>`, the descriptor pool/sets, `upload_weights`
  staging H2D, `resolve_queue_cap`/queue assignment, `submit_with_fence` waits,
  and the `BENCH`/`COUNT`/`SKIPIL`/`QUEUES` knobs. Shader side: `loadPad`
  indexes the source plane directly (was: packed upload staging), `storeDst`
  writes the pitched output plane (was: packed device dst), `ENTRY_KEEP` added.
- Method: MVP first (one frame per submission, 57/57 reference tests green on
  the first build), then the interleaved A/B above; the `--gpu-cache` row and
  the bare-transfer control are what localized the delta to the core's
  `GPUDownload` rather than the filter.
- The port also deleted the legacy shared layer from `src/vsfeel.h`/
  `src/vsfeel.cpp` with NNEDI3 as its last user: `VK_Device`/`get_device`,
  `volkInitialize`, the legacy pipeline-cache path, `allocate_memory`,
  `copy_*`/`mapped_range` helpers, `retire_instance`, `submit_with_fence`,
  `submit_timeline`, `destroy_common`, `rebar_available`,
  `require_vulkan_1_3` and the `VK_EXT_device_fault` dump (the core's device
  has no extensions). `FramePool`/`ticket_semaphore` stay for BM3D.

### 2026-09-23 — output-frame batching measured flat, reverted

EEDI3's fix for the single-queue world (one submission carrying a batch of
frames, phase-recorded, claim/publish cache) was ported and swept
`VSFEEL_NNEDI3_BATCH` 1/2/4/8/auto at 3000 f ×3: **2689 / 2684 / 2682 / 2667 /
2654 — flat, B=1 nominally fastest.** Mechanism of the miss: unlike EEDI3's
latency-bound row kernel, one predict grid already saturates the compute
queue, and the wall is downstream (the core's `GPUDownload`, see the banner),
so submission structure cannot move it — confirmed by gpu_busy 81 % idle
pattern matching a download-bound pipeline, not a starved queue. Host
record+submit did fall ~66 → ~23 µs/frame, but with no fps gain the
machinery (claims, cache, multi-frame requests) was reverted as unneeded;
re-test before reintroducing it, together with whatever changed the transfer
floor.

### 2026-09-20 — download slot pool split (pre-R80, +7.7 % at ns=1)

The slot pool was separated from the streams so a stream returned at its
fence while the host scatter still read the slot: 1473 → 1587 fps ns=1
(5000-frame medians); neutral at ns≥4. Superseded — both pools are deleted.

### 2026-09-20 — dead pad/assemble/count path deleted (pre-R80)

`pad_buf`/`asm_buf`, three undispatched pipelines and their SPIR-V outputs
removed: `mem_info_vram_used` 60.1 → 34.9 MB around a 4-stream 1080p GRAY16
instance (24.1 MB freed). Mechanism kept: audit what a dispatch actually
touches, not what once existed — same rule that later killed `BENCH`/`COUNT`.

### 2026-09-05/06 — the pre-R80 structure was assembled one measurement at a time

Mechanisms still live in the shipped design: prescreen **list + indirect
launch** (fused kernel tried and reverted: pred 464 vs 178 µs — verdict on the
full grid loses); `atomicMax` folded the count kernel away; pad folded into
the window reads (`loadPad` clamps the field directly); exact indirect grid
kept (over-launch refuted below); `REQUIRE_FULL_SUBGROUPS` on the cooperative
kernels only; one-shot ReBAR upload (`NT stores + sfence`) — now the core's
upload; host kept-lines pre-`take` — now `ENTRY_KEEP`.

### Measured dead ends (mechanisms)

- **Predict over-launch REFUTED.** Full-grid direct with early exit: pred 268
  vs 223 µs, 1892 vs 2055 fps at 2 streams — exiting subgroups still pay the
  window gather; the exact indirect grid ships.
- **PXP=16 restructure REVERTED.** Per-pixel shared stride 32 overflowed
  `shTile[4*288]` for warp≥1 → OOB shared writes, fresh-instance nondeterminism
  (maxdiff ~3000). Rule: re-derive the shared footprint (≤1152 vec4) for any
  PXP/stride change; `SHSTRIDE` is always a `-D` so array and addressing cannot
  drift apart.
- **Every producer→consumer dispatch pair needs an explicit barrier** —
  back-to-back dispatches order nothing. Prescreen→predict needs
  SHADER_WRITE → SHADER_READ *and* INDIRECT_COMMAND_READ; the indirect-struct
  fill needs TRANSFER → COMPUTE (both ship as the two `VkMemoryBarrier2`
  helpers in `src/nnedi3.cpp`).
- **YUV multi-plane: shader and host push layouts must change atomically.**
  Chroma corrupted when the host's per-plane `ind_elem` outgrew the shader's
  globals — a 4→5-word mismatch compiled and passed GRAY (single plane →
  offset 0), so only a YUV run catches it (`test_nnedi3_yuv_matches_reference`).
- **Scalar-GEMV stash: rejected.** Per-pixel accumulation + broadcast exploded
  predict ISA to 47357 (from 1674) and was wrong for PPL>1; re-derive from the
  committed vec4 form, never patch the stash.
- **An isolated copy microbenchmark proposed the wrong pack writer** (heap said
  memcpy 41 vs NT 65 µs; both real mappings said the opposite) — judge copy
  paths on `benchmark.py`. The pack itself is deleted with the port; the rule
  stands (AGENTS "microbenchmark proposes, in-situ A/B decides").
- **pscrn=0 is at parity** (177 vs ref 191 fps 1-stream pre-port, pred ~4.7 ms
  both over the full grid): the math is fine; the sparse path was where the
  old gap lived. **7900XTX has no dedicated DMA family** — the old "transfer
  queue" was a second compute queue, worth ~75 fps, net negative (deleted);
  today `RADV_EXPERIMENTAL=transfer_queue` exposes the real transfer family the
  core downloads on.

### Cross-cutting hardening (pre-R80 fixes, still shipped)

- `field` registered required — `:opt` with a null error pointer took
  `VS_FATAL_ERROR` → `std::terminate`; test `test_nnedi3_requires_field`.
- Prescreen grid under-coverage: host must compute
  `ceil(rows*ceil(width/P)/128)` (per-row grouping), never
  `ceil(width*rows/(P*128))` — the tail of every non-divisible row was left
  unwritten and uninitialized VRAM reached the output (YUV420P16 1924x1080:
  182 garbage chroma px/frame). Covered by the `test_geometry.py` nnedi3 rows.
- Validation layer: `REQUIRE_FULL_SUBGROUPS_BIT` only on the cooperative
  kernels (prescreen/predict) — `gpu_create_pipeline` takes the flag
  explicitly; covered by `tests/test_validation.py`.
- `numFrames == -1` is the unknown-length sentinel: `field>1` doubling is
  guarded (`numFrames > 0`), same fix as EEDI3.

## Open work

- **CPU-sink row trails nnedi3vk (0.91x)** — structural: the references write
  CPU-native output with their own overlapped transfers while the API mandates
  a core `GPUDownload` for a CPU consumer. Same trade already shipped for
  GaussBlur (−6.1 %), BM3D (−4 %) and Bilateral (−0.1 %); the resident row is
  where this port wins (1.35x). Nothing left to optimize inside the filter.
- **Prescreen vs reference ~2.8x claim is pre-port** (40 µs gap on old
  stamps): re-measure against nnedi3vk's kernels on the current binary before
  quoting it.
- **MVP limit**: `dh` + a planes subset zeroes interp lines on skipped planes
  (the reference leaves them uninitialized) — kept deliberately, tests rely on
  vsfeel's defined behavior.

## Debug env vars (all in `src/nnedi3.cpp`)

| env | what it does |
|---|---|
| `VSFEEL_NNEDI3_TSTAMP=<frame>` | one-shot GPU stamps for that frame: top / prescreen done / predict done (default 100; gated on the queue family's `timestampValidBits`) |
| `VSFEEL_NNEDI3_TIMING=1` | per-frame host stage averages (acquire/record/submit/total) printed at teardown |

`BENCH`, `COUNT`, `SKIPIL` and `QUEUES` were deleted with the pre-R80
machinery; `TSTAMP` became a one-shot probe and `BENCH` was renamed `TIMING`.
