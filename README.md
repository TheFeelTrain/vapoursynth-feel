# vapoursynth-feel

GPU-accelerated filters in Vulkan for VapourSynth.

## Requirements

- GPU with Vulkan 1.3 (or 1.1 with `VK_KHR_maintenance4` / `VK_KHR_8bit_storage`) and `glslc` at build time

## Usage

### Bilateral

[Bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter) is a non-linear, edge-preserving and noise-reducing smoothing filter for images.

The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution.

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

### BM3Dv2

[BM3D](https://en.wikipedia.org/wiki/Block-matching_and_3D_filtering) is a block-matching and 3D collaborative filtering denoiser. Groups of similar blocks (across space and time) are stacked into 3D arrays, denoised together with a hard-thresholded 3D transform, and the overlapping estimates are aggregated into the output.

```python
core.vsfeel.BM3Dv2(clip, sigma=0.7, block_step=4, bm_range=16, radius=2, ps_num=2, ps_range=7, num_streams=4, extractor_exp=0)
```

- clip:
    The input clip. Only 32 bit float Gray or YUV input is currently supported; for YUV input only the luma plane is denoised and the chroma planes are passed through unmodified.

- sigma: (Default: 3.0)
    Noise standard deviation of the input. Use an array to assign it for each plane, otherwise the same value is used for all planes. Must be non-negative.

- block_step: (Default: 8)
    Step between block positions in pixels. Use an array to assign it for each plane. Must be in range [1, 8].

- bm_range: (Default: 9)
    Search range in pixels for block matching in the spatial domain. Use an array to assign it for each plane. Must be positive.

- radius: (Default: 0)
    Temporal radius, i.e. how many frames before and after the current frame are searched for matching blocks. Must be in range [0, 4].

- ps_num: (Default: 2)
    Number of predicted positions used to seed the search in each temporal frame. Use an array to assign it for each plane. Must be in range [1, 8].

- ps_range: (Default: 4)
    Search range in pixels around each predicted position in the temporal search. Use an array to assign it for each plane. Must be positive.

- num_streams: (Default: 4)
    Number of command buffers submitted per frame, enables concurrent kernel execution and data transfer.

- extractor_exp: (Default: 0)
    Exponent of the extractor used to bias the aggregation weight. 0 disables the extractor.

## Manual Compilation

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

The plugin is installed automatically into VapourSynth's plugin directory. To install manually:

```bash
cp build/libvsfeel.so "$(python3 -c 'import vapoursynth; print(vapoursynth.get_plugin_dir())')/vsfeel/"
```
