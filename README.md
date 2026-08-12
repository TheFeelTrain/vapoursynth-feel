# vapoursynth-feel
Copyright© 2026 TheFeelTrain

GPU-accelerated filters in Vulkan for VapourSynth.

## Description
[Bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter) is a non-linear, edge-preserving and noise-reducing smoothing filter for images.

The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution.

Special thanks to [Kice](https://github.com/kice) for doing most of the work in previous implementation.

## Requirements

- GPU with Vulkan 1.3 (or 1.1 with `VK_KHR_maintenance4` / `VK_KHR_8bit_storage`) and `glslc` at build time

## Supported Formats

sample type: 8-16 bit integer or 32 bit float Gray/YUV/RGB input

## Usage

```python
core.vsfeel.Bilateral(clip, sigma_spatial=3.0, sigma_color=0.02, radius=0, num_streams=4, use_shared_memory=True)
```

- clip:
    The input clip.

- sigma_spatial: (Default: 3.0)
    Filter sigma in the coordinate space.
	Use an array to assign it for each plane. If "sigma_spatial" for the second plane is not specified, it will be set according to the sigma_spatial of first plane and sub-sampling.

- sigma_color: (Default: 0.02)
    Filter sigma in the color space.
	Use an array to assign it for each plane, otherwise the same sigma_color is used for all the planes.
	It will be normalized internally, so that for clips with different bit depths, the same values get similar results.

- radius: (Default: 0)
    Kernel window size. 0 = automatic calculatation based on "sigma_spatial".

- num_streams: (Default: 4)
    Number of command buffers submitted per frame, enables concurrent kernel execution and data transfer.

- use_shared_memory: (Default: True)
    Use on-chip shared memory to reduce bandwidth requirements on memory operations.

## Manual Compilation

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

The plugin is installed automatically into VapourSynth's plugin directory. To install manually:

```bash
cp build/libvsfeel.so "$(python3 -c 'import vapoursynth; print(vapoursynth.get_plugin_dir())')/vsfeel/"
```
