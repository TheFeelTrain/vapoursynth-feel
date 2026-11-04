# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Bilateral respects valid explicit `block_x` and `block_y` values on devices with tighter limits and matches vszipcl at very large radii

### Fixed

- BM3D plays on at full speed after a seek instead of recomputing a full temporal window for every frame
- BM3D applies limited-range `_Range` to RGB clips and no longer corrupts multi-plane output, leaks resources, or crashes on a missing frame
- DFTTest no longer corrupts tall frames with high spatial overlap or double-rounds float32 padding
- DFTTest and NNEDI3 report oversized memory requests at creation, and DFTTest also rejects negative sigmas and duplicate frequency locations
- EEDI3, EEDI3H, EEDI3AA, and NNEDI3 clear stale field tags, and EEDI3AA and its two-pass equivalent now agree exactly in 32-bit float
- NNEDI3 rejects `dh=True` with a `planes` subset and works more reliably across GPU drivers
- NLMeans uses the reference's rounding for 16-bit output
- A failed upstream frame is reported as an error instead of crashing
- Cache paths containing Windows separators work, and malformed numeric environment settings fall back to their defaults
- WNNM rejects frames that are too large for the device and no longer allocates unused memory
- The `feel_aa` wrapper validates `rfactor` and `pscale`, and uploads a host prefilter once instead of once per reader

## [v1.3.3] - 2026-11-04

### Changed

- NLMeans is 20-30% faster, especially on luma-only processing
- BM3D's `radius` argument can be set per plane like `sigma`

### Fixed

- All filters reject unknown-length or variable-format clips at creation, and failed creation no longer leaks resources
- BM3D rejects invalid arguments and frames that are too large
- DFTTest rejects invalid parameters and frames the device cannot process, and no longer reads partial data on some devices
- WNNM rejects very large frames and its trace mode no longer leaks memory

## [v1.3.2] - 2026-10-31

### Changed

- WNNM is about 16% faster when all three 1080p planes are processed

### Fixed

- WNNM no longer crashes, double-frees clips, or corrupts frames when caches or inputs fail on some drivers
- WNNM works on more GPUs, no longer rejects large configurations that fit, and handles 16-bit boundary frames more efficiently

## [v1.3.1] - 2026-10-30

### Changed

- WNNM's `radius` argument can be set per plane like `sigma`

### Fixed

- Bilateral rejects invalid `block_x` and `block_y` values instead of silently using defaults
- BM3D follows a 16-bit clip's `_Range` and validates its arguments even when every sigma is zero
- Box no longer crashes on illegal output sizes and matches resize2 when scaling cropped or odd-sized sources
- DFTTest rejects invalid filter parameters instead of returning NaNs

## [v1.3.0] - 2026-10-25

### Added

- `core.vsfeel.WNNM`, a Vulkan implementation of the WNNM patch denoiser
- WNNM accepts 16-bit integer and half-float input and output

### Changed

- DFTTest is about 30% faster, and up to 6.7x faster with larger temporal windows

### Fixed

- Windows builds use the current MinGW runtime

## [v1.2.0] - 2026-10-21

### Added

- `vsfeel.feel_aa`, a copy of `vsaa.based_aa` that keeps processing on the GPU
- `core.vsfeel.Box`, a Vulkan box downscale
- All filters support half-float input and output

### Changed

- GaussBlur is over 2x faster, BM3D about 18% faster, and Bilateral about 15% faster

## [v1.1.2] - 2026-10-18

### Changed

- BM3D is now registered as `core.vsfeel.BM3D` instead of `core.vsfeel.BM3Dv2`

### Fixed

- BM3D no longer runs at 1 fps on Nvidia GPUs

## [v1.1.1] - 2026-10-05

### Fixed

- BM3D through vs-jetpack's `backend=vsfeel.Backend` raised "Use of invalidated Core" after a vsview script reload

## [v1.1.0] - 2026-10-05

All filters now have full feature parity with other implementations and are more accurate.

### Added

- BM3D supports 16-bit integer input and output
- BM3D supports the joint 4:4:4 `chroma=True` mode for denoising chroma planes
- EEDI3, EEDI3H, and EEDI3AA implement `hp` (half-pel search), matching eedi3vk2

### Changed

- EEDI3AA is about 8% faster, and BM3D about 5% faster with less run-to-run variance
- BM3D accepts up to `radius=16` and now matches VapourSynth-BM3D output

### Fixed

- EEDI3 and EEDI3H now match eedi3vk2 exactly in 32-bit float
- NLMeans more closely matches vszipcl
- The shader cache is only rewritten when a process compiled something new

## [v1.0.0] - 2026-10-01

The plugin now uses the core's GPU device, queues, and transfers instead of managing its own.

### Added

- macOS and arm64 wheels, plus capability checks at filter creation

### Changed

- All nine filters use the core's GPU device, queues, and transfers
- Frame caches follow the core's GPU memory budget, and invalid input is rejected at creation

### Fixed

- Resource leaks, frame-cache races, and EEDI3AA plane-merge errors

## [v0.2.2] - 2026-09-21

### Added

- `VSFEEL_DEBUG=1` prints a device banner and a full error trace, while `VSFEEL_DEBUG=2` adds per-frame traces and GPU timings

### Changed

- BM3D sizes its cache to the working set, and upload staging is only used on GPUs that benefit from it

### Fixed

- BM3D no longer hangs on long jobs or reuses stale cache data
- Several device detection, upload, and timing probe fixes, plus small DFTTest and NNEDI3 fixes

## [v0.2.1] - 2026-09-20

### Fixed

- BM3D now works on more GPUs, including ones without float atomics

## [v0.2.0] - 2026-09-20

### Changed

- The Vulkan requirement was lowered from 1.4 to 1.3
- BM3D, EEDI3, and EEDI3AA are faster

## [v0.1.0] - 2026-09-18

### Added

- First release: Bilateral, BM3D, GaussBlur, DFTTest, NLMeans, EEDI3, EEDI3H, EEDI3AA, and NNEDI3
- A benchmark harness, vs-jetpack backend integration, and Windows and Linux wheels

[v1.3.3]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/v1.3.2...v1.3.3
[v1.3.2]: https://github.com/TheFeelTrain/vapoursynth-feel/compare/v1.3.1...v1.3.2
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
