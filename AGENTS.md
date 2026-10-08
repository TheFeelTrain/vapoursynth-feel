# AGENTS.md — vapoursynth-feel

Guidance for AI coding agents working on this repository. This file is about
the vsfeel project as a whole; per-filter implementation notes live next to
the filters themselves.

## What this project is

`vsfeel` is a VapourSynth plugin that provides GPU-accelerated video filters
implemented in **Vulkan** (GLSL compute shaders compiled to SPIR-V with
`glslc`, driven from C++ host code). Each filter is named after, and produces
results equivalent to, a reference implementation in the `reference/` folder.

The primary GPU this project is developed and tuned against is an **AMD Radeon
RX 7900XTX (RDNA3, gfx1100)**. Optimizations are targeted at that GPU, but
other configurations should be considered when possible.

## The `reference/` folder is READ-ONLY

The `reference/` directory contains the source of the reference
implementations (e.g. `vapoursynth-zipcl`, `vapoursynth-zipcu`,
`vs-dfttest2`, `VapourSynth-BM3DCUDA`).

- **Do not modify anything inside `reference/`.** It is for reading only, so
  you can port algorithms and understand expected behaviour.
- The references are the source of truth for **numerical correctness**: a
  vsfeel filter's output should match the reference output closely (exact or
  within a ulp / small tolerance of float rounding differences between
  backends).

## The goal

Make every vsfeel filter **faster than the reference implementations**. 
Concretely, the benchmark should show vsfeel beating the fastest
reference (usually vszipcl) by a comfortable margin.

Speed matters more than code size or elegance. Do not be afraid to rewrite a
filter wholesale if it makes it meaningfully faster, as long as it stays
correct and passes all tests.

When tuning, use the benchmark (below) to measure before/after, and treat the
GPUs documented here as the target. `MANGOHUD=0` should be set for every
benchmark run. It does not change results, it just suppresses extra messages
in the output.

## Comments and docstrings

Keep them short and only where the code is not self-explanatory: aim for 3
lines or less, say *why* rather than *what*, and do not restate the code. Test
docstrings are a few lines at most. Notes files (`notes/<filter>.md`) may be
longer, but do not pad them. Never cite work-order or report IDs (`WO-55`,
`§I.11`, `T9`) in code, comments, docstrings or notes.

## Changelog (CHANGELOG.md)

Keep a Changelog format, newest first, with an `## [Unreleased]` section for
what has landed since the last tag. Write the entry with the change, not at
release time. One bullet per user-visible change, under `Added`, `Changed`,
`Fixed` or `Removed`: one unwrapped source line that names the filter or
component first. Entries read like this:

```markdown
- BM3Dv2 accepts up to `radius=16` instead of 4, matching vszipcl
- A BM3Dv2 cache slot could be overwritten by a stale writer
- Upload fixes for EEDI3 and BM3Dv2 on devices without a ReBAR heap
```

- **Short and high level.** What a user notices, not how it works: no kernel or
  function names, no buffer budgets, no rationale, and no "Added support for ..."
  openers. A fix to the development tooling (`tools/`, the lint and notes gates, 
  the benchmark harness) is not an entry either. Strictly user-facing changes.
- **No em-dashes.** Use a comma, a full stop, or parentheses.
- **Only what the last release did differently.** An entry is read by someone
  upgrading from the last tag, so a bug in something that has only ever existed
  in the `[Unreleased]` section is not an entry: no release ever had it, and
  fixing it only makes the unreleased feature work as first documented.

The notes files carry the mechanism; the changelog carries the result.

## Environment (uv)

The dev environment is the project venv, not the system Python or the system
VapourSynth: `uv sync` builds `.venv` from `uv.lock` and `.python-version` (a
uv-managed CPython), and `tools/install.sh` installs the plugin into that venv's
plugin directory. The core is pinned to the `vapoursynth==81rc1` pre-release in
the `dev` group (the R81 changes move the benchmark; the published requirement
stays `vapoursynth>=80`). The `dev` group also supplies pytest/xdist, numpy,
vsjetpack and the reference plugins the suite and the benchmark compare against
(`vszipcl` from git plus `fmtconv` from JET's vs-wheels index, `eedi3vk2`, 
`nnedi3vk`, `bm3dvk`, `nlm_hip`, `knlmmeansvk`, `zsmooth`, `bestsource`, 
`edgemasks`, `resize2`, `descale`). `vapoursynth-bm3d` is pinned to the git
commit `reference/VapourSynth-BM3D` is checked out at, not the PyPI wheel: the
wheel is an older tree (its September commits were force-pushed away upstream),
and BM3Dv2 is graded against the reference source. That commit is a dangling
object on GitHub, so re-resolving needs it to still be fetchable; the local
checkout under `reference/` is the fallback. `uv sync --group lint` adds the
pinned clang-format/clang-tidy/ruff. Never `pip install` into this tree, and never rely 
on a system plugin being visible. `tools/*.sh` and `uv run` pick the venv up on 
their own. One-time per venv, run `.venv/bin/vapoursynth config`: the wheel's 
`vspipe` embeds CPython through VSScript, whose autodetection cannot find a 
`libpython` for uv's statically linked CPython, and the registration is what 
makes the benchmark and the vspipe hang tests work.

## Testing

Every filter needs **comprehensive unit tests** in `tests/`, run with pytest.
The committed `tests/bigbuckbunny_360p_grain.mp4` clip is the standard test 
input. The fixtures expose its first 24 frames (`CLIP_FRAMES`): the helpers 
iterate every frame they are handed, so the whole file would make a full run 
12x longer without covering anything new. A test that needs a longer run asks 
for one explicitly.

- Tests must verify **correctness against the reference behavior** and
  **self-consistency** (determinism across runs, multi-stream vs single-stream
  agreement, parallel-load consistency).
- The `tests/` folder has `conftest.py` with shared fixtures/helpers
  (`WIDTH`, `HEIGHT`, `CLIP_PATH`, `frame_to_ndarray`, ...).
- Always run the full test suite for the filter you touch before and after
  changes: `tools/test.sh tests/test_<filter>.py -q`.
- Run the whole suite with **`tools/test.sh`** (extra args are passed through,
  default target `tests`). It uses pytest-xdist (`dev` dependency group) with
  `--dist loadfile` and caps workers at 8. Do not raise the cap and
  do not drop `loadfile`: 16 workers fail with `vkQueueSubmit failed` (GPU
  exhaustion), and a plain `-n 8` is flaky because NLMeans `a=64, d=16`
  hard-recovers the GPU whenever another client shares it — file-level
  distribution stops that config overlapping another heavy file. See
  `notes/NLMEANS.md`. `VSFEEL_TEST_WORKERS=1 tools/test.sh` forces serial.
- **The suite loads the installed plugin, not `build/libvsfeel.so`** — build and
  install first, or you are testing the previous binary.
- A rewrite is only acceptable if all tests still pass.

### Reference-comparison coverage and tolerance policy

- **Sweep parameters against the reference** — every scalar parameter, every
  supported input depth, plus special paths (joint processing, guide clips,
  cropped frames). Crash-prone references run in a subprocess.
- **Measure before setting a tolerance**, then document mechanism and
  measured values. Tiers: `1e-6` for ulp-level float32 math; measurement-
  bounded bounds (a few e-3) where discrete decisions flip on rounding order;
  integer output in whole codes (`<= 1 LSB`); self-consistency stays exact.
- Read planes back stride-aware and `.copy()` ctypes arrays: both alias recycled
  frame memory and produce phantom nondeterminism.

## How benchmarking works

The benchmark is `tools/benchmark.py`. It is data-driven: every filter is one
entry in a `FILTERS` registry describing its CLI args, the input clip
expression, and a builder that maps each supported plugin to the vpy call that
runs it. Plugins are described separately in `PLUGINS`.

- Timing is done with `vspipe`, so results are comparable across plugins and
  with the earlier per-plugin scripts.
- Usage:
  - `uv run tools/benchmark.py` — all filters
  - `uv run tools/benchmark.py --filter <name>` — one filter
  - `uv run tools/benchmark.py --filter <name> vsfeel vszipcl` — a subset of
    plugins, to compare against references
  - `--frames N`, `--clip PATH` to control the run
  - `--gpu-cache` hands every arm device-resident frames, the chain a GPU
    filter actually runs in (a CPU filter pays the download it would pay
    there); `--no-download` crops each arm's output to 8x8, which takes
    vspipe's implicit output download out of the fps. Both together time a GPU
    filter's own compute; `notes/METHOD.md` has what the mode's residual costs.
- The default clip is `/home/encode/test/jpbd.mkv` (1920x1080, YUV420P8).
- By default the run **caches real frames in RAM**: the first `--cache-frames`
  (default 1000) frames are decoded while vspipe evaluates the script, and its
  fps figure only covers the output loop, so timing measures filter throughput
  on real content without the BestSource decode bottleneck (~630 fps). Frames
  loop when `--frames` exceeds the cached span, so temporal filters see a seam
  every N frames — fine for throughput, not for output inspection.

To benchmark a single filter against the references:

```bash
uv run tools/benchmark.py --filter dfttest vsfeel vszipcl
```

This prints fps for each plugin and ranks them. Compare vsfeel's fps against
the fastest reference. Judge optimizations on multiple runs over hundreds of
frames.

To compare two vsfeel builds or two env configs against each other, use the
benchmark's A/B mode instead of writing a scratch script:

```bash
uv run tools/benchmark.py --filter bm3dv2 --ab-so build/libvsfeel.so old/libvsfeel.so --ab-names new,old
uv run tools/benchmark.py --filter bm3dv2 --ab-b-env VSFEEL_BM3D_DERIVE=1 --ab-names legacy,derived
```

Each round measures both arms interleaved (alternating order, or ABBA with
`--ab-order abba`); `.so` copies are sha-verified and the original installed
binary is restored afterwards. Without explicit plugins only vsfeel runs.

Two-tier measurement keeps the iteration loop tight: screen candidates with a
fast custom `.vpy` + `vspipe`, and grade only on full `benchmark.py` same-session
pairs over 1000+ frames, as medians rather than single short bursts. The
measurement rules behind that are in `notes/METHOD.md`.

## Porting and tuning method

`notes/METHOD.md` holds the cross-cutting method, and is binding. Start at its
`## Typical workflow`; the rest is the detail: comparing a vsfeel kernel against
the reference kernels (profiling, ACO ISA and instruction-count diffs,
specialization constants) and the porting discipline (MVP first, bit-exact
oracles, ablation ladders, benchmark hygiene, keeping `notes/<filter>.md`
current).

## Shared plumbing (src/vsfeel.h)

All filters share an inline (zero-overhead, C++20) plumbing layer in
`src/vsfeel.h`. New filters must build on it — do not re-invent this wheel:

- **Record through the exec pool**: one command buffer per output frame from
  `gpuExecAcquire` / `gpuExecCommandBuffer`, declare inputs with
  `gpuExecReadsFrame` and outputs with `gpuExecWritesPlane`, then
  `gpuExecSubmit`; `gpuExecAbandon` on every error path before submit. The pool
  owns the ring, the timeline and the scratch lifetime and turns producer pairs
  into device-side waits, so the host never waits per frame. Copy the pattern
  from gaussblur/bilateral/nnedi3 (stateless) or bm3d/dfttest (cached/temporal).
- **Cross-frame caches stay possible without owning a submission**: the slot's
  ready flag is *submitted*, not a timeline value, and a full
  `vkCmdPipelineBarrier` at the start of the reader's first command buffer
  supplies the dependency. Never signal or hand-roll the pool's timeline; the
  worked mechanism is `src/bm3d.cpp`'s cross-frame ordering comment.
- **Debug flags** go through `vsfeel_debug_flag` / `vsfeel_debug_trace` /
  `vsfeel_debug_probe`, never `env_flag` directly, with one env name per filter.
  The levels, what each helper prints and what `VSFEEL_DEBUG=1|2` turns on are in
  the `Debug switches` block of `src/vsfeel.h`; a level-2 run is instrumented, so
  never benchmark it.
- **Gate a GPU-timing probe on `GPUDevice::timestamp_valid_bits`** (via
  `vsfeel_probe_timestamps`): a timestamp write on a queue family reporting 0 is
  invalid usage, and a driver that takes one anyway can hang the engine (a
  machine-wide freeze, not a lost device). Gate the *flag*, not only the pool.
- **Errors and the frame trail**: every `set_error` lambda calls
  `vsfeel_trace_error(filter, frame, message, d->gpu.get())` first (a null device
  is fine), and the frame path opens with `vsfeel_trace_frame_begin()` plus a
  `vsfeel_trace_mark(stage)` before each step, so the first error names the step
  that failed. Reporting through `vsapi->mapSetError` directly bypasses both.

**ODR rule (learned the hard way):** a filter-local struct used to instantiate
a shared template needs a **unique name per filter** (`Bm3dStream`, never
`VK_Resource`): the same mangled name with a different `sizeof` across TUs lets
the linker COMDAT-fold one instantiation over the others and corrupt in-flight
resources. It stands for any shared template a filter instantiates.

**What stays per-filter:** frame caches (DFTTEST slots, NLMeans tiles, BM3D
estimate/source rings), shaders + launch config, sync choreography and cache
sizing. Never a command pool, timeline or fence of your own.

## Building and installing

**Build with `tools/install.sh`. It builds *and* installs in a single command.**
There is no separate copy step and no reason to run `cmake` by hand — a
hand-built `libvsfeel.so` that was never copied into the plugin directory is
invisible to VapourSynth, so the tests and the benchmark quietly keep exercising
the previous binary.

```bash
uv sync                          # create/refresh the venv (first time, and after dependency changes)
tools/install.sh                 # configure + build + install, then hash-verify
tools/install.sh -h              # -b build dir, -p plugin dir, -c build type
```

That one command configures `build/` (Release) with CMake + `glslc` (the Vulkan
shader compiler), compiles the plugin, copies `libvsfeel.so` into
VapourSynth's plugin directory, and **fails unless the installed copy's sha256
equals the build's**. It asks `vapoursynth.get_plugin_dir()` where the running
Python loads plugins from, so the project venv installs into the venv's plugin
directory (the system tree is never touched); its headers are passed explicitly
as `VS_INCLUDE_DIR` so the build cannot pick up a distribution `pkg-config` file
for a different VapourSynth release instead. `VSFEEL_PYTHON` overrides the
interpreter, `VSFEEL_BUILD_DIR` and `VSFEEL_PLUGIN_DIR` the autodetection. A run
whose plugin directory already matches the build is a no-op.

The bare CMake rules are only for a compile-only check that must not touch the
installed plugin:

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Never chain build → install → test into one shell command.** If the compile
fails, the previous `.so` stays installed and the test run silently "passes"
against a binary that does not contain your change. Build and install with
`tools/install.sh`, then test as a separate command.

Vulkan headers (`Vulkan-Headers`) are pinned and fetched by `FetchContent` at
configure time, so no Vulkan SDK is needed to link — only `glslc` from one.
**Nothing links a Vulkan library**: the R80 GPU API hands the plugin the core's
dispatch table (`VSVulkanFunctions` via `getVulkanFunctions`), no `vk*` function
is called directly, and `VSVulkan4.h` defines `VK_NO_PROTOTYPES` before pulling
in `vulkan_core.h`, so a direct call cannot even be compiled. That is what lets
the Linux wheel be repaired to manylinux (`libvulkan.so.1` is on no manylinux
whitelist) and the Windows DLL need no import library. The plugin builds with
`-fvisibility=hidden` and exports only `VapourSynthPluginInit2`.

The SPIR-V shaders are compiled at build time and embedded into a generated C++
header (`spirv_binaries.h`) via `src/gen_spirv_header.py`. Adding a shader
variant is one line in the owning component's `VK_*_VARIANTS` table in
`CMakeLists.txt` — an `"<out>|<source>|<defs>"` entry naming the output, the
`.comp` file and every `-D`; `--target-env=vulkan1.4` is not in the table because
every variant uses it (the R80 core's device baseline). `add_spv_variant()`
turns each entry into the `glslc` rule and collects the outputs into
`VK_SPV_OUTPUTS`, which is generated, not edited by hand. The header script
(`--out <header> <spv...>`) derives symbol names from the filenames
(`<stem>.spv` → `<stem>_spv` / `<stem>_spv_size`), so it never needs editing for
new variants.

### How the shader pipeline fits together

Each `src/*.comp` file holds several entry points selected by `-D` defines
(e.g. `-DENTRY_FUSED -DRADIUS=1 -DBITS=16`); a CMake `foreach` loop compiles
one `.spv` per variant-table entry into `build/vk_spv/`, and
`gen_spirv_header.py` packs them all into `spirv_binaries.h` as `uint32_t`
arrays. The C++ side picks the arrays it needs (usually per bit depth / radius
at filter creation) and creates one `VkShaderModule` + `VkPipeline` per
variant. So a new fast-path variant means: an `#if` block in the `.comp`, one
entry in that component's `VK_*_VARIANTS` table, and a module/pipeline pair in
the filter's creation function — nothing else.

Every variant rule also depends on a generated `<out>.spv.flags` stamp holding
that variant's `glslc` arguments. Ninja would rebuild on a flag-only change by
itself, but the default Makefiles generator does not: without the stamp a
changed `-D` or `PROBE`/`MAXW` cache value leaves a stale
`.spv` in place and silently ships the old kernel (a stale probe kernel already
invalidated a round of measurements once). The stamp is written with
`file(GENERATE)` and content-hashed, so an unrelated `CMakeLists.txt` edit does
not recompile the shaders. Do not remove it.

### Persistent pipeline cache (makes creation and the test suite fast)

Compiling the compute shaders from SPIR-V dominates filter creation on RADV —
about 4.4 s for the first DFTTest variant in a process, versus ~0.11 s once the
driver has it. The plugin therefore keeps a `VkPipelineCache` seeded from and
flushed to a file:

- default location: `$XDG_CACHE_HOME/vsfeel/pipeline_cache_<pipelineCacheUUID>.bin`
  (`~/.cache/vsfeel/...`), with an automatic fallback to `$TMPDIR/vsfeel/` when
  the home cache is not writable (read-only home, sandbox, CI);
- `VSFEEL_PIPELINE_CACHE=<path>` overrides the file, `VSFEEL_PIPELINE_CACHE=0`
  disables the cache entirely (useful when checking that a shader change
  actually recompiles);
- the driver's `pipelineCacheUUID` is both the filename and part of the cache
  contents, so a driver/device change misses instead of feeding the driver
  incompatible data. The file is written via a unique temp file + rename, so
  concurrent test subprocesses cannot corrupt it;
- the write is **skipped when the blob is the size it was loaded at**, so a
  process that compiled nothing new leaves the file alone. Without that check
  every short-lived process rewrote it at exit: 25.8 GB of device writes for
  `tests/test_bm3dv2.py` alone, 0.2 GB after it.

Measured on the full suite: **~7.5 min cold → ~3.5 min for
the run that builds the cache → 142 s warm** (904 tests, 323 s before the write
check). The first run after a driver or `glslc` update pays the compile again by
design.

## Code quality (lint)

`tools/lint.sh` is the single entry point, and CI runs exactly it
(`.github/workflows/lint.yml`). The pinned clang-format/clang-tidy/ruff come
from the `lint` dependency group (`uv sync --group lint`), and the script puts
`.venv/bin` first on `PATH`, so a local run and CI use the same versions --
except `cppcheck`, which is a system package (apt on CI, the distro locally) and
whose findings do drift between releases. `lint.sh` prints each gate's version
so that difference is visible in the log; a finding that reproduces on neither
box is a version difference, not a defect. Six gates:

- **format** — `clang-format --dry-run --Werror` over `src/*.cpp` and `src/*.h`
  (`.clang-format`); `tools/lint.sh --fix` rewrites in place.
- **shaders** — the build's `shader-validate` target: every variant is compiled
  with `glslc -Werror` and each emitted module is checked with `spirv-val`
  against `vulkan1.4`. A shader that compiles but is not valid SPIR-V fails the
  build here (this is what caught a 16-bit constant needing `Int16`).
- **tidy** — `clang-tidy` over the compile database (`.clang-tidy`).
- **cppcheck** — over the same database; the one tool whose version is not
  pinned, so its findings can differ between a developer's box and CI.
- **ruff** — `ruff check` over `vsfeel/`, `tools/`, `hatch_build.py`,
  `src/gen_spirv_header.py` and `tests/` (see `[tool.ruff.lint]` in
  `pyproject.toml` for the curated check set).
- **notes** — `tools/notes_check.py` over `notes/*.md`, against
  `notes/AGENTS.md`: the fixed parts and their order, both line budgets, the
  file index, report IDs. It reads no build directory, so it also runs on a
  bare checkout.

The middle three need a configured `build/` — they read `compile_commands.json`
and `build/vk_spv/`, which `tools/install.sh` writes. A gate whose tool is
missing reports `skipped`; a gate whose build directory is missing fails,
because then it did not run.

After changing a shader or a `-D`, run `uv run python tools/shader_limits.py` for
the workgroup/LDS table; `--check` compares that table against the `GpuWorkgroup`
literals at each `gpu_create_pipeline` call site and fails on a mismatch. The lint
run does it for you (`tools/lint.sh shaders`), and CI runs both so the table lands
in the log next to the validation.

The `.comp` shaders are deliberately **not** clang-format'd: clang-format parses
GLSL as C++ and reflows the buffer blocks and the push-constant struct into
unreadable shapes. They are hand-formatted to the same 80 columns and gated by
the shader build instead; `.editorconfig` records the whitespace conventions for
both.

CI pins `clang-format`/`clang-tidy` to the version the tree was formatted with
(the PyPI wheels), because formatter output changes between LLVM majors and the
runner's apt packages are older. `VSFEEL_WERROR=ON` (a configure option, off by
default) turns the host warning baseline into an error; the lint job sets it.

## Use web searches

Use web searches often. If you feel like you are getting stuck, do not be
afraid to search for hints. Search even when you think you don't need to — it
is always better to have more information. Look up relevant topics such as
GPU/Vulkan/GLSL/RDNA3 performance, shader optimization techniques, and the
reference projects' own documentation and discussions.

## Scratch files

Do all scratch work (temporary scripts, probe outputs, intermediate artifacts)
in the repo's `.scratch/` folder instead of the system `/tmp`. Under the sandboxed
shell, `/tmp` is a fresh tmpfs per shell call, so files written there are
cleared between commands. Anything a later command needs must live under the
workspace (`.scratch/`).

## Commits

Do not make any commits yourself. If you want a commit or a checkpoint, stop
and ask the user to make it for you. Leave your changes staged/unstaged in the
working tree and describe what should be committed.
