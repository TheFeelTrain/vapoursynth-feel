# GaussBlur — performance notes

Status: **iterating** — vsfeel 1808 fps vs vszipcl 1412 (beaten) vs vszipcu
3489. Baseline was 1342/1643/3083. Next target: eliminate the GPU-side H2D
copy via host-mapped VRAM upload.

## Context

Benchmark call: `MANGOHUD=0 python benchmark/bench.py --filter gaussblur`

- Input: `/home/thefeeltrain/Encode/test/jpbd.mkv` 1920x1080 YUV420P8, converted to GRAY16
  (`depth(get_y(clip), 16)`), 5000 frames cached in RAM, num_streams=4.
- sigma=16 → taps = ceil(16*6+1) = 97 → radius 48 > 32 → **two-pass large path**
  (ENTRY_VERT then ENTRY_HORIZ) on every frame.
- Target GPU: RX 7900 XTX (gfx1100), RADV. PCIe 4.0 x16. **ReBAR is enabled**
  (BAR0 = 32 GiB; Vulkan heap 1 = 24 GiB DEVICE_LOCAL, and memoryType[3] =
  DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT — i.e. VRAM mapped into the CPU
  address space).

## Current vsfeel implementation (baseline)

- Host-visible cached staging buffer (type 5: HOST_VISIBLE|HOST_COHERENT|
  HOST_CACHED, heap 0 = 31 GiB GTT/carve-out), one per stream, holding
  upload + download regions back to back. Kernels read/write **sysmem
  directly** (SSBO bound to staging), no device-local bounce.
- The 16-bit kernel unpacks u16 elements from u32 dwords (shift/mask per
  element) and stores results with **masked atomicOr** into shared dwords
  (because two adjacent lanes share a dword) — the download region is
  pre-cleared with vkCmdFillBuffer every frame.
- Large path: ENTRY_VERT writes float tmp (device-local buffer) then
  ENTRY_HORIZ reads tmp and writes 16-bit dst via atomicOr packs.
- Per-frame flow: memcpy VS frame → staging (streaming stores), submit,
  vkWaitForFences, memcpy staging → VS frame (streaming loads). Fully
  synchronous per frame; pool depth = num_streams.
- num_queues = min(num_streams, queue_count). RADV compute family exposes
  **4 queues** (family 1, compute+transfer) — with num_streams=4 all four
  queues are used, one stream each.

## Reference structures

- vszipcu (fastest): separate d_src/d_dst (device-local, HIP), d_tmp device
  float buffer. Per plane: async memcpyHtoD (cstream — a dedicated copy
  stream), kernel(s) on the compute stream, async memcpyDtoH, then
  **s.cstream.sync() + s.stream.sync()** (i.e. also fully synchronous
  per frame, but the H2D copy of each plane runs concurrently with kernels
  of previously-copied planes *within* the frame because of event
  chaining; D2H too).
- vszipcl: same shape with OpenCL pinned staging buffers + clFinish per
  frame.

## Facts gathered

- Queue families on RADV gfx1100: [0] gfx+compute (1 queue), [1] compute+
  transfer (**4 queues**), [2] video decode, [3] video encode.
- Vulkan memory types: [3] and [4] = DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT
  (heap 1, 24 GiB) — usable for host-mapped VRAM staging.
- Current staging (HOST_CACHED, heap 0) is GTT RAM, not VRAM.
- **rocprofv3 per-frame profile of vszipcu** (BlankClip gray16 1080p,
  sigma=16, ns=1, 200 frames): vertical_blur avg **118 µs**, horizontal_blur
  avg **177 µs**, H2D copy **153 µs**, D2H copy **152 µs**. Kernels 295 µs +
  copies 305 µs, overlapped via copy-stream/compute-stream pipelining and
  4 streams → 3400 fps ≈ 294 µs/frame ≈ **PCIe-bound** (8.3 MB/frame ≈ 28 GB/s).
- BlankClip gray16 throughput (pure filter, no decode):
  - ns=1: vsfeel 669 (σ16) / 738 (σ4) vs vszipcu 1621 / 1827, vszipcl 1594/1871
  - ns=4: vsfeel ~1270, vszipcu ~3150, vszipcl ~1440
- **ReBAR host-mapped VRAM bandwidth probe** (64 MiB, Vulkan memoryType 3):
  CPU→VRAM NT stores 11 GB/s, plain memcpy 22.5 GB/s (write-combining works);
  **VRAM→CPU read 1.3 GB/s NT / 0.1 GB/s memcpy — unusable for downloads**.
  → mapped-VRAM direct staging is dead for the read path; downloads must use
  SDMA copies into GTT.
- rocprofv3 does NOT see RADV/Vulkan work (only the HIP/ROCm runtime), so
  vsfeel kernel times must be measured via fps or RADV_DEBUG/asm dumps.
- RADV shaderstats of current large-path kernels (σ16: KLEN=97):
  vert: 178 instr, 24 VGPR, 9 VMEM clauses; horiz: 146 instr, 24 VGPR.
  Tiny kernels — the loop stayed rolled; not obviously bloated.

## Plan (validated by the numbers)

1. **Device-local VRAM src/dst buffers per stream**; kernels read/write VRAM
   only (they currently read/write GTT over PCIe, and 16-bit writes go
   through dword atomicOr RMW over PCIe — catastrophic). One big
   `vkCmdCopyBuffer` H2D (staging upload region → dev_src) at the head of the
   pre-recorded command buffer and one big D2H (dev_dst → staging download
   region) at the tail. This is exactly vszipcu's structure (their H2D runs
   on a separate copy stream; ours runs inside the same command buffer but
   on 4 different queues so frames overlap).
2. **16-bit SSBO elements** (`uint16_t[]` via GL_EXT_shader_16bit_storage /
   storageBuffer16BitAccess, already enabled device-wide in vsfeel.cpp):
   removes the per-element shift/mask unpacking, the dword packing, the
   atomicOr, and the per-frame vkCmdFillBuffer clear + barrier entirely.
3. Push constants become element offsets into the VRAM buffers (u16/f32).
4. Keep per-plane config/dedup, same math (FMA ascending-k, mirror) — output
   must stay bit-exact.

## Log

- (start) Baselines above. Reading code, profiling next.
- **VRAM rework done** (committed as "GaussBlur: Move frame io to
  device-local VRAM with DMA copies"): device-local src/dst/tmp + one big
  H2D and one big D2H vkCmdCopyBuffer per frame, native u16 SSBO elements
  (no atomicOr, no fill-clear, no shift/mask). Benchmark: vsfeel
  **1808 fps** (was 1191-1342), vszipcl 1412 (**beaten**), vszipcu 3489.
  BlankClip ns=1: 731 fps (was 669), ns=4: 1794 (was 1273). Probe added:
  `VSFEEL_GAUSS_GPU_BENCH=N` env (wired into GaussGetFrame, frame 0 only).
- **Probe numbers** (σ16 gray16 1080p, ns=1, N=100): full CB (copies+kernels)
  **715.5 µs/frame**, kernels only **306.7 µs** (= vert 118 + horiz 177 ≈
  vszipcu kernel parity). → the two DMA copies cost **~409 µs**
  (~20.3 GB/s, ~200 µs per direction) and barely overlap.
- Analysis: vszipcu's 287 µs/frame = kernels 295 µs (CUs) fully overlapped
  with copies 305 µs (HIP SDMA engines) — PCIe full-duplex. Our copies run
  on the CP inside the same command buffer/queue; overlap across frames on
  4 queues is only partial (aggregate 553 µs/frame at ns=4).
- Next: **remove the GPU-side H2D copy** — CPU memcpys the frame directly
  into host-mapped VRAM (ReBAR memoryType 3; measured 22.5 GB/s with plain
  memcpy, better than NT stores at 11 GB/s). GPU per frame becomes
  kernels + D2H copy only; CPU upload/download memcpys run on the VS
  worker threads in parallel with GPU work. D2H must stay a GPU copy
  (CPU reads from VRAM BAR are 1.3 GB/s — dead end, measured).

## Validation hardening (cross-cutting pass)

The upload flush and download invalidate ranges (per plane, at arbitrary
32-byte-aligned offsets) now go through the shared `mapped_range` helper, which
rounds them to `minNonCoherentAtomSize`/`VK_WHOLE_SIZE` as Vulkan requires. No
behaviour change on this coherent device; all `test_gaussblur.py` tests pass.

## NT-store ordering

Same gap as Bilateral: on the staging path (`host_direct_upload == false`) the
upload uses NT stores, and the submit that tells the GPU to read that window had
no `_mm_sfence()` before it. Added unconditionally just before
`submit_with_fence`. Correctness-only change, no fps effect expected;
`test_gaussblur.py` passes.
