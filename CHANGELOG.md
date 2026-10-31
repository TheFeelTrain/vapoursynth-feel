# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- WNNM releases its input clips when a filter instance is destroyed
- WNNM no longer corrupts frames when its match cache cannot be allocated
- WNNM output no longer depends on concurrent same-address GPU writes the Vulkan memory model does not guarantee
- WNNM no longer rejects a configuration whose combined scratch exceeds the device's per-buffer storage range
- WNNM's split Gram reduction no longer depends on the driver's subgroup lane layout, and a device that cannot run it falls back to the fused kernel
- WNNM copies a 16-bit boundary frame into its conversion ring once per plane instead of once per clamped window position

## [v1.3.1] - 2026-10-30

### Changed

- WNNM's `radius` argument can be set per plane like `sigma`

### Fixed

- Bilateral rejects non-positive `block_x`/`block_y` instead of silently using the default shape
- BM3D follows a 16-bit clip's `_Range` like the reference instead of treating every integer clip as full range
- BM3D validates its block/search arguments even when every sigma is zero
- Box no longer terminates the host process when the output size is illegal for a subsampled format
- Box matches resize2 at tap boundaries on cropped and odd-sized sources
- DFTTest rejects a negative `f0beta` or `pmax` instead of returning an all-NaN frame

## [v1.3.0] - 2026-10-25

### Added

- `core.vsfeel.WNNM`, a Vulkan implementation of the WNNM patch denoiser that matches vszip's CPU filter
- WNNM accepts 16-bit integer and fp16 input and output natively

### Changed

- DFTTest is about 30% faster
- DFTTest with larger temporal windows (tbsize 5 and 7) is up to 6.7x faster

### Fixed

- Windows builds link against the current MinGW CRT

## [v1.2.0] - 2026-10-21

### Added

- `vsfeel.feel_aa`, a copy of `vsaa.based_aa` that keeps its processing entirely on the GPU
- `core.vsfeel.Box`, a Vulkan implementation of a box downscale
- All filters support fp16 input and output

### Changed

- GaussBlur is over 2x faster
- BM3D is about 18% faster
- Bilateral is about 15% faster

## [v1.1.2] - 2026-10-18

### Fixed

- BM3D no longer runs at 1 fps on Nvidia GPUs

### Changed

- BM3D is now registered as `core.vsfeel.BM3D` instead of `core.vsfeel.BM3Dv2`

## [v1.1.1] - 2026-10-05

### Fixed

- BM3D through vs-jetpack's `backend=vsfeel.Backend` raised "Use of invalidated Core" after a vsview script reload

## [v1.1.0] - 2026-10-05

All filters now have full feature parity with other implementations and are more accurate.

### Added

- BM3D supports 16-bit integer input and output
- BM3D supports the joint 4:4:4 `chroma=True` mode for denoising chroma planes
- EEDI3, EEDI3H, and EEDI3AA implement `hp` (half-pel search), bit-exact with eedi3vk2

### Changed

- EEDI3AA is about 8% faster
- BM3D is about 5% faster with less run-to-run variance
- BM3D accepts up to `radius=16`, matching vszipcl
- BM3D now matches VapourSynth-BM3D output, diverging from bm3dcuda/vszipcl/bm3dvk

### Fixed

- EEDI3 and EEDI3H are now bit-exact with eedi3vk2 in fp32 (u16 already matched)
- NLMeans more closely matches vszipcl in fp32 and u16
- The pipeline cache is only rewritten when a process compiled something new

## [v1.0.0] - 2026-10-01

The plugin has moved to the VapourSynth R80 GPU API. The core now owns the device, the queues, and the transfers.

### Added

- macOS and arm64 wheels
- Capability checks at filter creation, so an unsupported device or argument is reported instead of failing later

### Changed

- All nine filters were ported to the R80 GPU API. The plugin no longer creates its own device, queues or host transfer path
- Frame caches are capped by the core's VRAM budget instead of fixed sizes
- Invalid arguments and unsupported input are rejected at creation time
- Python packaging was reworked. Wheel and sdist contents are pinned, and every build is verified before publishing

### Fixed

- Resource leaks on creation and frame error paths across every filter
- Frame cache races in BM3D's rings and EEDI3's batched output, which could read a wrong or recycled frame
- EEDI3AA's per plane merge, and its handling of odd plane geometry
- Benchmark and test harness bugs that reported misleading results

## [v0.2.2] - 2026-09-21

### Added

- `VSFEEL_DEBUG=1` prints a device banner and a full error trace. `VSFEEL_DEBUG=2` adds per frame traces and GPU timings

### Changed

- BM3D sizes its estimate cache to the working set instead of a fixed size
- Direct upload staging is only used when the GPU has a ReBAR heap

### Fixed

- BM3D could hang the driver on long submissions. The estimation now submits one window position at a time
- A BM3D cache slot could be overwritten by a stale writer
- `env_flag=0` was treated as enabled in some filters
- Subgroup support was assumed rather than checked in several creation paths
- Upload fixes for EEDI3 and BM3D on devices without a ReBAR heap
- GPU timing probes are only armed when the queue family reports valid timestamps
- Small fixes in DFTTest's fused transpose window and NNEDI3's download slots

## [v0.2.1] - 2026-09-20

### Fixed

- BM3D now works on GPUs without buffer float atomics, using a compare and swap aggregation path

## [v0.2.0] - 2026-09-20

### Changed

- The Vulkan requirement was lowered from 1.4 to 1.3
- BM3D matching is faster, with a wider scan, subgroup operations and uploads staged in host visible memory
- EEDI3's vertical consistency check runs in parallel, which is a large speedup
- EEDI3AA's two horizontal passes share one submission
- Shader, pipeline and buffer helpers are shared by all filters in `vsfeel.h`

## [v0.1.0] - 2026-09-18

### Added

- First release: Bilateral, BM3D, GaussBlur, DFTTest, NLMeans, EEDI3, EEDI3H, EEDI3AA and NNEDI3, all implemented as Vulkan compute shaders driven from C++
- A benchmark harness comparing every filter against the reference implementations
- vs-jetpack backend integration, so the filters can be used through the vs-jetpack wrappers
- Windows and Linux wheels, built and published by CI

[v1.3.1]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/v1.3.0...v1.3.1
[v1.3.0]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/v1.2.0...v1.3.0
[v1.2.0]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/1.1.2...v1.2.0
[v1.1.2]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/1.1.1...1.1.2
[v1.1.1]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/1.1.0...1.1.1
[v1.1.0]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/1.0.0...1.1.0
[v1.0.0]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/0.2.2...1.0.0
[v0.2.2]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/0.2.1...0.2.2
[v0.2.1]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/0.2.0...0.2.1
[v0.2.0]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/0.1.0...0.2.0
[v0.1.0]: https://github.com/TheFeelTrain/vapoursynth-feel/releases/tag/0.1.0
