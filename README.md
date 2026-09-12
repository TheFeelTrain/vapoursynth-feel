# vapoursynth-feel

GPU-accelerated filters in Vulkan for VapourSynth.

Primarily optimized for running on RDNA 3 with the RADV driver on Linux.

## Performance

| Filter    | u16 vsfeel | u16 ref | u16 speedup | fp32 vsfeel | fp32 ref | fp32 speedup |
|-----------|-----------:|--------:|------------:|------------:|---------:|-------------:|
| Bilateral | 1903       | 1484    | 1.28x       | 1250        | 763      | 1.64x        |
| BM3Dv2    | -          | -       | -           | 166         | 41       | 4.02x        |
| DFTTest   | 1268       | 871     | 1.46x       | 1053        | 625      | 1.69x        |
| EEDI3     | 301        | 206     | 1.46x       | 253         | 199      | 1.27x        |
| GaussBlur | 2516       | 1599    | 1.57x       | 1267        | 756      | 1.68x        |
| NLMeans   | 970        | 737     | 1.32x       | 846         | 651      | 1.30x        |
| NNEDI3    | 2907       | 2659    | 1.09x       | 1563        | 1396     | 1.12x        |

Reference columns are `vszipcl` for all filters except `nnedi3vk` for NNEDI3.

Each figure is the median of 3 `benchmark/bench.py --bits <16|32>` runs on an RX 7900 XTX.

## Common Arguments

`device_id=0` selects the GPU. `num_streams` sets how many frames the filter processes concurrently. 

More streams means more VRAM. Default values should be the most efficient.

All filters support 16-bit integer and 32-bit float input except for BM3Dv2, which is 32-bit only.

## Filters

### Bilateral

[Bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter) is a non-linear, edge-preserving and noise-reducing smoothing filter for images. The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution.

```python
core.vsfeel.Bilateral(clip clip[,
    clip    ref,                    # guide clip: weights from ref, values from clip
    float[] sigma_spatial=3.0,      # spatial blur reach; per-plane, chroma default = sigma[0]/sqrt((1<<ssw)*(1<<ssh))
    float[] sigma_color=0.02,       # edge sensitivity: how different pixels may be to blend; depth-normalized
    int[]   radius,                 # per-plane blur window radius; default max(1, round(sigma_spatial*3))
    int     use_shared_memory=1,    # faster on-chip kernel while the window fits; falls back otherwise
    int     block_x, int block_y,   # GPU thread-block dims; auto-tuned when unset (32 x 8, x16 when radius > 12)
    int     device_id=0, 
    int     num_streams=4])                   
```

### BM3Dv2

[BM3D](https://en.wikipedia.org/wiki/Block-matching_and_3D_filtering) is a block-matching and 3D collaborative filtering denoiser. Groups of similar blocks (across space and time) are stacked into 3D arrays, denoised together with a hard-thresholded 3D transform, and the overlapping estimates are aggregated into the output.

```python
core.vsfeel.BM3Dv2(clip clip[,
    clip    ref,                    # basic-estimate clip for the Wiener (final) pass
    float[] sigma=3.0,              # denoising strength per-plane; below FLT_EPSILON skips the plane
    int     radius=0,               # temporal search radius in frames, 0..4 (0 = spatial only)
    int[]   block_step=8,           # per-plane block grid spacing, 1..8; smaller = fewer artifacts, slower
    int[]   bm_range=9,             # per-plane spatial search radius in pixels (> 0)
    int[]   ps_num=2,               # motion-predicted candidates seeding each temporal search
    int[]   ps_range=4,             # search radius around each predicted candidate, in pixels
    int     extractor_exp=0         # aggregation weight bias; 0 = off, >= 3 = reproducible output
    int     device_id=0, 
    int     num_streams=4])       
```
32-bit float only. Chroma passes through unprocessed.

### DFTTest

DFTTest is a 3D block-wise frequency-domain denoiser. Each 16x16 spatial block is stacked with temporally neighboring frames, transformed to the frequency domain, filtered, and transformed back with overlap-add. It reduces noise while preserving detail, with configurable spatial and temporal windowing and a choice of frequency-domain filter variants.

```python
core.vsfeel.DFTTest(clip clip[,
    int     ftype=0,                # 0 = Wiener: mult by max((psd-sigma)/psd, 0) ** f0beta
                                    # 1 = hard threshold: zero the bin when psd < sigma
                                    # 2 = mult every bin by sigma
                                    # 3 = mult by sigma inside [pmin,pmax], else by sigma2
                                    # 4 = mult by sigma*sqrt(psd*pmax / ((psd+pmin)*(psd+pmax)))
    float   sigma=8.0,              # base denoising strength
    float   sigma2=8.0,             # ftype=3 only: strength outside [pmin,pmax]
    float   pmin=0.0,               # ftype=3/4 only: lower PSD bound
    float   pmax=500.0,             # ftype=3/4 only: upper PSD bound
    int     sbsize=16,              # spatial block size; must be 16 in this backend
    int     sosize=12,              # spatial overlap, 0..15; >50% needs (sbsize-sosize) | sbsize
    int     tbsize=3,               # temporal block size; ODD, 1..7 (1 = spatial only)
    int     swin=0,                 # spatial window: 0=hanning 1=hamming 2=blackman
    int     twin=7,                 # temporal window: 3=4-term b-harris 4=kaiser-bessel
                                    #   5=7-term b-harris 6=flat top 7=rectangular 8=bartlett
                                    #   9=bartlett-hann 10=nuttall 11=blackman-nuttall
    float   sbeta=2.5,              # kaiser-bessel beta (swin=4 only)
    float   tbeta=2.5,              # kaiser-bessel beta (twin=4 only)
    int     zmean=1,                # subtract the windowed mean before filtering
    float   f0beta=1.0,             # ftype=0 exponent (1.0 = plain Wiener, 0.5 = sqrt)
    float[] slocation,              # frequency-dependent sigma: [freq, sigma, freq, sigma, ...]
    float[] ssx,                    # slocation along the X axis only
    float[] ssy,                    # slocation along the Y axis only
    float[] sst,                    # slocation along time only
    int     ssystem=0,              # slocation scale: 0 = relative to block size, 1 = absolute
    int[]   planes,                 # planes to process; default: all
    int     device_id=0, 
    int     num_streams=1])
```

### EEDI3

EEDI3 is an edge-directed interpolator for deinterlacing and upscaling. For each missing pixel it searches candidate connection angles between the neighboring lines, picks the one minimizing a cost functional over local neighborhoods, and interpolates along that direction so edges stay sharp instead of stair-stepping.

```python
core.vsfeel.EEDI3(clip clip, int field[,   # and EEDI3H, same args, horizontal
    clip    sclip,                  # source for the vcheck comparison
    clip    mclip,                  # edge mask; fully-masked spans are skipped
    float   alpha=0.2,              # 0..1 (alpha+beta <= 1): weight given to connecting similar
                                    #   neighborhoods. Larger = more lines/edges connected
    float   beta=0.25,              # 0..1: weight given to the vertical difference created by
                                    #   the interpolation. Larger = fewer edges connected
                                    #   (1.0 = no edge directedness at all)
    float   gamma=20.0,             # >= 0: penalizes changes in interpolation direction.
                                    #   Larger = smoother interpolation field
    int     nrad=2,                 # radius used for neighborhood similarity, 0..3
    int     mdis=20,                # max connection radius, 1..40; larger connects shallower
                                    #   lines, but costs speed and risks artifacts
    int     vcheck=2,               # 0..3: strength of the vertical-consistency check
    float   vthresh0=32.0,          # vcheck thresholds; all must be > 0 when vcheck > 0
    float   vthresh1=64.0,
    float   vthresh2=4.0,
    int     dh=False,               # double-height output keeping every source line (no field extracted)
    int[]   planes,                 # planes to process; default: all
    int     device_id=0, 
    int     num_streams=8])
```
`field` 2/3 are the double-rate variants (not allowed with `dh=True`).

### GaussBlur

[Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) is a smoothing filter that blends each pixel with its neighbors, weighting nearby pixels more heavily than distant ones according to a bell-shaped (Gaussian) curve. It is commonly used to soften an image or reduce noise.

```python
core.vsfeel.GaussBlur(clip clip[,
    float[] sigma=0.5,              # blur strength; per-plane, chroma default = sigma[0]/sqrt((1<<ssw)*(1<<ssh))
    int     device_id=0, 
    int     num_streams=1])
```

### NLMeans

[Non-local means](https://en.wikipedia.org/wiki/Non-local_means) is a denoising filter that replaces every pixel with a weighted average of pixels across a search window, weighting them by the similarity of their surrounding patches. Searching across space and time suppresses noise while preserving detail.

```python
core.vsfeel.NLMeans(clip clip[,
    clip    rclip,                  # weights computed from rclip, values from clip
    int     d=1,                    # temporal radius; 0 = spatial only
    int     a=2,                    # search-window radius, 1..64
    int     s=4,                    # patch radius, 0..8
    float   h=1.2,                  # > 0: filtering strength
    int     wmode=0,                # weighting function of the patch distance x, 0..3:
                                    #   0 = exp(-x) | 1 = max(1-x, 0)
                                    #   2 = max(1-x, 0)**2 | 3 = max(1-x, 0)**8
    float   wref=1.0,               # >= 0: weight of the pixel itself
    string  channels="auto",        # planes to process, jointly for YUV/RGB; auto picks by format
    int     device_id=0, 
    int     num_streams=1])
```
`channels="YUV"` requires 4:4:4 so on subsampled clips run a `"Y"` pass and a `"UV"` pass instead.

### NNEDI3

NNEDI3 is a neural-network edge-directed interpolator for deinterlacing and upscaling. A small neural network predicts each missing pixel from its local neighborhood, while a prescreener network skips pixels that simple cubic interpolation already handles.

```python
core.vsfeel.NNEDI3(clip clip, int field[,
    int     nsize=6,                # predictor neighborhood, 0..6: 0=8x6 1=16x6 2=32x6
                                    #   3=48x6 4=8x4 5=16x4 6=32x4
    int     nns=1,                  # predictor neurons, 0..4: 0=16 1=32 2=64 3=128 4=256
    int     qual=1,                 # 1 or 2: number of predictor passes averaged
    int     etype=0,                # 0 = weights trained on absolute error, 1 = squared error
    int     pscrn=2,                # prescreener, 0..4: 0=off (predict every pixel)
                                    #   1=original, 2..4=new levels 0..2 (higher = fewer pixels
                                    #   left to cubic interpolation: slower, slightly better)
    int     dh=False,               # double-height output keeping every source line (no field extracted)
    int[]   planes,                 # planes to process; default: all
    int     device_id=0, 
    int     num_streams=4])
```
`field` 2/3 are the double-rate variants (not allowed with `dh=True`).

## vs-jetpack integration

The wheel ships a small Python module alongside the plugin that adds vsfeel as a backend for the [vs-jetpack](https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack) wrappers.

vsjetpack is an optional dependency, only needed at runtime when you use these backends.

```python
from vsrgtools import bilateral, gauss_blur
from vsdenoise import bm3d, DFTTest, nl_means

import vsfeel

# Bilateral
blurred = bilateral(clip, ref, 3.0, 0.02, backend=vsfeel.Backend)

# BM3D
denoised = bm3d(clip, 0.7, tr=2, profile=bm3d.Profile.NORMAL, ref=ref, planes=0, backend=vsfeel.Backend)

# DFTTest
dft = DFTTest(clip, backend=vsfeel.Backend).denoise({0.0: 16.0, 0.5: 8.0, 1.0: 0.0}, tr=1)

# GaussBlur
smooth = gauss_blur(clip, 1.5, backend=vsfeel.Backend)

# NLMeans
denoised = nl_means(clip, h=0.2, tr=2, a=2, s=4, ref=ref, planes=[1, 2], backend=vsfeel.Backend)
```

To also route the *internal* filters vs-jetpack calls on its own (e.g. the bilateral postfilter inside `vsaa.based_aa`), use the backend as a context manager:

```python
from vsaa import based_aa

with vsfeel.Backend():
    based = based_aa(clip, backend=vsfeel.Backend)
```

## Manual Compilation

### Full Python Package
```bash
pip install .
```

### Plugin-Only
```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build
```