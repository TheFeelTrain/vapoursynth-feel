# vapoursynth-feel

GPU-accelerated filters in Vulkan for VapourSynth.

Primarily optimized for running on RDNA 3 with the RADV driver on Linux.

## Requirements

- GPU with Vulkan 1.3 (or 1.1 with `VK_KHR_maintenance4`) and `glslc` at build time

## Usage

### Bilateral

[Bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter) is a non-linear, edge-preserving and noise-reducing smoothing filter for images.

The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution.

```python
core.vsfeel.Bilateral(clip, sigma_spatial=3.0, sigma_color=0.02, radius=0, num_streams=4, use_shared_memory=True)
```

- clip:
    The input clip. Supports 16 bit integer and 32 bit float input in Gray, YUV or RGB color families.

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

### DFTTest

DFTTest is a 3D block-wise frequency-domain denoiser. Each 16x16 spatial block is stacked with temporally neighboring frames, transformed to the frequency domain, filtered, and transformed back with overlap-add. It reduces noise while preserving detail, with configurable spatial and temporal windowing and a choice of frequency-domain filter variants.

```python
core.vsfeel.DFTTest(clip, sigma=8.0, sosize=12, tbsize=3, swin=0, twin=7, zmean=1, num_streams=1)
```

- clip:
    The input clip. Supports 16 bit integer and 32 bit float input in Gray, YUV or RGB color families.

- ftype: (Default: 0)
    Frequency-domain filter variant: 0 = wiener, 1 = hard threshold, 2 = multiply, 3 = bandpass, 4 = power.

- sigma: (Default: 8.0)
    Noise standard deviation of the input. Use an array to assign it for each plane, otherwise the same value is used for all planes.

- sosize: (Default: 12)
    Spatial overlap in pixels. Must be in range [0, 15].

- tbsize: (Default: 3)
    Temporal block size in frames, i.e. how many frames before and after the current frame are processed together. Must be odd, in range [1, 7].

- swin: (Default: 0)
    Spatial window type, 0 to 11.

- twin: (Default: 7)
    Temporal window type, 0 to 11.

- zmean: (Default: 1)
    Apply the zero-mean correction (subtract the window frequency response scaled by the DC gain before filtering).

- f0beta: (Default: 1.0)
    Exponent used with `ftype=0`.

- planes:
    The planes to process. By default all planes are processed.

- num_streams: (Default: 1)
    Number of command buffers submitted per frame, enables concurrent kernel execution and data transfer. Must be in range [1, 32].

### GaussBlur

[Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) is a smoothing filter that blends each pixel with its neighbors, weighting nearby pixels more heavily than distant ones according to a bell-shaped (Gaussian) curve. It is commonly used to soften an image or reduce noise.

```python
core.vsfeel.GaussBlur(clip, sigma=0.5, num_streams=1)
```

- clip:
    The input clip. Supports 16 bit integer and 32 bit float input in Gray, YUV or RGB color families.

- sigma: (Default: 0.5)
    Blur sigma for each plane. Use an array to assign it for each plane, otherwise the second plane defaults to `sigma[0] / sqrt((1 << subSamplingH) * (1 << subSamplingW))` and the third to the second plane's value. Must be non-negative. A value below the machine epsilon copies the plane through unmodified.

- num_streams: (Default: 1)
    Number of command buffers submitted per frame, enables concurrent kernel execution and data transfer. Must be in range [1, 32].

### NLMeans

[Non-local means](https://en.wikipedia.org/wiki/Non-local_means) is a denoising filter that replaces every pixel with a weighted average of pixels across a search window, weighting them by the similarity of their surrounding patches. Searching across space and time suppresses noise while preserving detail.

```python
core.vsfeel.NLMeans(clip, d=2, a=2, s=4, h=0.2, wmode=0, wref=1.0, channels="auto", rclip=None, num_streams=2)
```

- clip:
    The input clip. Supports 16 bit integer and 32 bit float input in Gray, YUV or RGB color families.

- d: (Default: 1)
    Temporal radius, i.e. how many frames before and after the current frame are included in the search. Must be in range [0, 16].

- a: (Default: 2)
    Radius of the search window in pixels. The search covers (2a+1)^2 positions per frame pair. Must be in range [1, 64], and the window must fit within the frame.

- s: (Default: 4)
    Radius of the patch compared around every pixel when computing similarity. The patch contains (2s+1)^2 pixels. Must be in range [0, 8].

- h: (Default: 1.2)
    Filter strength. Higher values give more weight to less similar patches, producing stronger smoothing. Must be positive.

- wmode: (Default: 0)
    Weight transform applied to the normalized patch distance: 0 = exponential decay, 1 = linear, 2 = quadratic, 3 = eighth-power cutoff. Must be in range [0, 3].

- wref: (Default: 1.0)
    Weight multiplier for the center pixel itself relative to the found candidates. Must be non-negative.

- channels: (Default: "auto")
    Planes to process: "Y" denoises luma only, "UV" only chroma, "YUV" all planes jointly (requires 4:4:4), "RGB" all planes jointly with RGB distance weighting. "auto" picks based on the color family. Case-insensitive.

- rclip:
    Optional guide clip whose planes provide the patch content used to compute the distances, while the planes of the source clip are averaged. Must match the source clip's format and dimensions.

- num_streams: (Default: 1)
    Number of command buffers submitted per frame, enables concurrent kernel execution and data transfer. Must be in range [1, 32].

## vs-jetpack integration

The wheel ships a small Python module alongside the plugin that adds vsfeel as a backend for the [vs-jetpack](https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack) wrappers. 

vsjetpack is an optional dependency, only needed at runtime when you use these backends.

```python
from vsrgtools import bilateral, gauss_blur
from vsdenoise import bm3d, DFTTest, nl_means

import vsfeel

# Bilateral
blurred = bilateral(clip, ref, 3.0, 0.02, backend=vsfeel.Backend.Bilateral)

# BM3D
denoised = bm3d(clip, 0.7, tr=2, profile=bm3d.Profile.NORMAL, ref=ref, planes=0, backend=vsfeel.Backend.BM3D)

# DFTTest
dft = DFTTest(clip, backend=vsfeel.Backend.DFTTest).denoise({0.0: 16.0, 0.5: 8.0, 1.0: 0.0}, tr=1)

# GaussBlur
smooth = gauss_blur(clip, 1.5, backend=vsfeel.Backend.GaussBlur)

# NLMeans
denoised = nl_means(clip, h=0.2, tr=2, a=2, s=4, ref=ref, planes=[1, 2], backend=vsfeel.Backend.NLMeans)
```

## Manual Compilation

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

The plugin is installed automatically into VapourSynth's plugin directory. To install manually:

```bash
cp build/libvsfeel.so "$(python3 -c 'import vapoursynth; print(vapoursynth.get_plugin_dir())')/vsfeel/"
```
