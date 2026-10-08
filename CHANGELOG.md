# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- BM3Dv2 no longer runs about 100x slow on Nvidia GPUs
- BM3Dv2 uses the compare-and-swap accumulation on Nvidia GPUs

## [1.1.1] - 2026-10-05

### Fixed

- BM3D through vs-jetpack's `backend=vsfeel.Backend` raised "Use of invalidated Core" after a vsview script reload

## [1.1.0] - 2026-10-05

All filters now have full feature parity with other implementations and are more accurate.

### Added

- BM3Dv2 supports 16-bit integer input and output
- BM3Dv2 supports the joint 4:4:4 `chroma=True` mode for denoising chroma planes
- EEDI3, EEDI3H, and EEDI3AA implement `hp` (half-pel search), bit-exact with eedi3vk2

### Changed

- EEDI3AA is about 8% faster
- BM3Dv2 is about 5% faster with less run-to-run variance
- BM3Dv2 accepts up to `radius=16`, matching vszipcl
- BM3Dv2 now matches VapourSynth-BM3D output, diverging from bm3dcuda/vszipcl/bm3dvk

### Fixed

- EEDI3 and EEDI3H are now bit-exact with eedi3vk2 in fp32 (u16 already matched)
- NLMeans more closely matches vszipcl in fp32 and u16
- The pipeline cache is only rewritten when a process compiled something new

## [1.0.0] - 2026-10-01

The plugin has moved to the VapourSynth R80 GPU API. The core now owns the device, the queues and the transfers, and frames stay on the GPU from input to output

### Added

- macOS and arm64 wheels
- Capability checks at filter creation, so an unsupported device or argument is reported instead of failing later
- A shader limits check, which compares every compiled kernel against the workgroup and shared memory sizes the host declares for it
- Linting and code quality gates for the C++ host code, the GLSL kernels and the Python tooling, plus a CI job for the tests that need no GPU
- A `uv` development environment with a lockfile

### Changed

- All nine filters were ported to the R80 GPU API. The plugin no longer creates its own device, queues or host transfer path
- Frame caches are capped by the core's VRAM budget instead of fixed sizes
- Invalid arguments and unsupported input are rejected at creation time
- Python packaging was reworked. Wheel and sdist contents are pinned, and every build is verified before publishing

### Fixed

- Resource leaks on creation and frame error paths across every filter
- Frame cache races in BM3Dv2's rings and EEDI3's batched output, which could read a wrong or recycled frame
- EEDI3AA's per plane merge, and its handling of odd plane geometry
- Benchmark and test harness bugs that reported misleading results

## [0.2.2] - 2026-09-21

### Added

- `VSFEEL_DEBUG=1` prints a device banner and a full error trace. `VSFEEL_DEBUG=2` adds per frame traces and GPU timings

### Changed

- BM3Dv2 sizes its estimate cache to the working set instead of a fixed size
- Direct upload staging is only used when the GPU has a ReBAR heap

### Fixed

- BM3Dv2 could hang the driver on long submissions. The estimation now submits one window position at a time
- A BM3Dv2 cache slot could be overwritten by a stale writer
- `env_flag=0` was treated as enabled in some filters
- Subgroup support was assumed rather than checked in several creation paths
- Upload fixes for EEDI3 and BM3Dv2 on devices without a ReBAR heap
- GPU timing probes are only armed when the queue family reports valid timestamps
- Small fixes in DFTTest's fused transpose window and NNEDI3's download slots

## [0.2.1] - 2026-09-20

### Fixed

- BM3Dv2 now works on GPUs without buffer float atomics, using a compare and swap aggregation path

## [0.2.0] - 2026-09-20

### Added

- Tests for temporal frame order, cropped geometry, stream counts and resource boundedness. The suite now runs in parallel

### Changed

- The Vulkan requirement was lowered from 1.4 to 1.3
- BM3Dv2 matching is faster, with a wider scan, subgroup operations and uploads staged in host visible memory
- EEDI3's vertical consistency check runs in parallel, which is a large speedup
- EEDI3AA's two horizontal passes share one submission
- Shader, pipeline and buffer helpers are shared by all filters in `vsfeel.h`

## [0.1.0] - 2026-09-18

### Added

- First release: Bilateral, BM3Dv2, GaussBlur, DFTTest, NLMeans, EEDI3, EEDI3H, EEDI3AA and NNEDI3, all implemented as Vulkan compute shaders driven from C++
- A benchmark harness comparing every filter against the reference implementations
- vs-jetpack backend integration, so the filters can be used through the vs-jetpack wrappers
- Windows and Linux wheels, built and published by CI

[1.1.1]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/1.0.0...1.1.0
[1.0.0]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/0.2.2...1.0.0
[0.2.2]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/TheFeelTrain/vapoursynth-feel/releases/tag/0.1.0
