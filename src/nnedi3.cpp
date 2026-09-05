#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

// _mm_sfence for ordering the non-temporal pad stores before submit.
#include <immintrin.h>

#include <vulkan/vulkan.h>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#include "vsfeel.h"
#include "spirv_binaries.h"

using namespace std::string_literals;

// ---------------------------------------------------------------------------
// NNEDI3 — intra-field interpolator.
//
// vsfeel's own implementation of the predictor/prescreener math both GPU
// references compute (nnedi3vk and vszipcu agree bit-exactly, see
// notes/NNEDI3.md). The host follows the usual vsfeel shape (FramePool,
// upload/download staging, device-local working buffers); the GPU runs a
// prescreen kernel (cubic + compacted rejected-pixel list), an indirect
// cooperative predict kernel over the list, and a DMA readback of the packed
// interp rows (host interleaves kept lines from the source).
// ---------------------------------------------------------------------------

// Window / network tables indexed by the filter arguments.
constexpr int NNEDI3_XDIM[7] { 8, 16, 32, 48, 8, 16, 32 };
constexpr int NNEDI3_YDIM[7] { 6, 6, 6, 6, 4, 4, 4 };
constexpr int NNEDI3_NNS[5] { 16, 32, 64, 128, 256 };

constexpr int MARGIN_H = 24;
constexpr int MARGIN_V = 3;

// Weight blob linked into the binary (see CMakeLists.txt objcopy rule).
extern "C" {
extern const uint8_t _binary_nnedi3_weights_bin_start[];
extern const uint8_t _binary_nnedi3_weights_bin_end[];
}

// ---------------------------------------------------------------------------
// Weight blob parsing
//
// Layout of the f32 blob (see notes/NNEDI3.md): old prescreener, three new
// prescreeners (layer-0 stored transposed), then etype x nns x nsize model
// pairs (qual 1 + qual 2). Only the selected model is retained.
// ---------------------------------------------------------------------------

struct PsOldWeights {
    float k0[4][48] {};
    float b0[4] {};
    float k1[4][4] {};
    float b1[4] {};
    float k2[4][8] {};
    float b2[4] {};
};

struct PsNewWeights {
    float k0[4][64] {};
    float b0[4] {};
    float k1[4][4] {};
    float b1[4] {};
};

struct ModelWeights {
    int xdim {}, ydim {}, nns {};
    std::vector<float> sm1, el1, sm_b1, el_b1;
    std::vector<float> sm2, el2, sm_b2, el_b2;
};

struct WeightReader {
    const float * data {};
    size_t count {};
    size_t pos {};

    bool read(float * dst, size_t n) {
        if (pos + n > count) {
            return false;
        }
        std::memcpy(dst, data + pos, n * sizeof(float));
        pos += n;
        return true;
    }

    bool skip(size_t n) {
        if (pos + n > count) {
            return false;
        }
        pos += n;
        return true;
    }
};

static double vec_mean(const float * v, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        acc += v[i];
    }
    return acc / static_cast<double>(n);
}

// Prescreener prep: subtract each layer-0 neuron's own mean and scale by
// 1/pixel_half, so the shader dots raw pixel values directly.
template <size_t W>
static void prescreener_prep(float (&k0)[4][W], double pixel_half) {
    for (int n = 0; n < 4; ++n) {
        const double m = vec_mean(k0[n], W);
        for (size_t k = 0; k < W; ++k) {
            k0[n][k] = static_cast<float>((k0[n][k] - m) / pixel_half);
        }
    }
}

// Model prep: project the per-neuron means and the shared mean filter out of
// the weights (one pass per qual), so the shader normalizes with v*mstd2 and
// re-adds the window mean only in the final blend.
static void model_prep_pass(std::vector<float> & sm, std::vector<float> & el,
                            std::vector<float> & sm_b, size_t fs, size_t nns) {
    std::vector<double> sm_means(nns), el_means(nns), mean_filter(fs, 0.0);
    for (size_t p = 0; p < nns; ++p) {
        sm_means[p] = vec_mean(sm.data() + p * fs, fs);
        el_means[p] = vec_mean(el.data() + p * fs, fs);
        for (size_t k = 0; k < fs; ++k) {
            mean_filter[k] += sm[p * fs + k] - sm_means[p];
        }
    }
    for (size_t k = 0; k < fs; ++k) {
        mean_filter[k] /= static_cast<double>(nns);
    }
    const double bias_mean = vec_mean(sm_b.data(), nns);
    for (size_t p = 0; p < nns; ++p) {
        for (size_t k = 0; k < fs; ++k) {
            sm[p * fs + k] -= static_cast<float>(sm_means[p] + mean_filter[k]);
            el[p * fs + k] -= static_cast<float>(el_means[p]);
        }
        sm_b[p] -= static_cast<float>(bias_mean);
    }
}

static std::optional<std::string> parse_weights(int nsize, int nns_sel, int etype,
                                                int pscrn, double pixel_half,
                                                PsOldWeights & ps_old,
                                                PsNewWeights & ps_new,
                                                ModelWeights & model) {
    const float * data = reinterpret_cast<const float *>(
        _binary_nnedi3_weights_bin_start);
    const size_t count = (reinterpret_cast<const uint8_t *>(
        _binary_nnedi3_weights_bin_end) -
        reinterpret_cast<const uint8_t *>(
        _binary_nnedi3_weights_bin_start)) / sizeof(float);
    WeightReader r { data, count, 0 };

    for (int n = 0; n < 4; ++n) {
        if (!r.read(ps_old.k0[n], 48)) return "weight blob truncated (ps_old l0)";
    }
    if (!r.read(ps_old.b0, 4)) return "weight blob truncated (ps_old b0)";
    for (int n = 0; n < 4; ++n) {
        if (!r.read(ps_old.k1[n], 4)) return "weight blob truncated (ps_old l1)";
    }
    if (!r.read(ps_old.b1, 4)) return "weight blob truncated (ps_old b1)";
    for (int n = 0; n < 4; ++n) {
        if (!r.read(ps_old.k2[n], 8)) return "weight blob truncated (ps_old l2)";
    }
    if (!r.read(ps_old.b2, 4)) return "weight blob truncated (ps_old b2)";

    PsNewWeights all_new[3] {};
    for (int i = 0; i < 3; ++i) {
        float l0s[4 * 64] {};
        float l1s[4 * 4] {};
        if (!r.read(l0s, 4 * 64)) return "weight blob truncated (ps_new l0)";
        if (!r.read(all_new[i].b0, 4)) return "weight blob truncated (ps_new b0)";
        if (!r.read(l1s, 4 * 4)) return "weight blob truncated (ps_new l1)";
        if (!r.read(all_new[i].b1, 4)) return "weight blob truncated (ps_new b1)";
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 64; ++k) {
                all_new[i].k0[n][k] = l0s[(k / 8) * 32 + n * 8 + k % 8];
            }
            for (int k = 0; k < 4; ++k) {
                all_new[i].k1[n][k] = l1s[k * 4 + n];
            }
        }
    }

    bool found = false;
    for (int m = 0; m < 2; ++m) {
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 7; ++j) {
                const size_t nns = NNEDI3_NNS[i];
                const size_t fs = static_cast<size_t>(NNEDI3_XDIM[j]) * NNEDI3_YDIM[j];
                if (m == etype && i == nns_sel && j == nsize) {
                    model.xdim = NNEDI3_XDIM[j];
                    model.ydim = NNEDI3_YDIM[j];
                    model.nns = static_cast<int>(nns);
                    model.sm1.resize(nns * fs);
                    model.el1.resize(nns * fs);
                    model.sm_b1.resize(nns);
                    model.el_b1.resize(nns);
                    model.sm2.resize(nns * fs);
                    model.el2.resize(nns * fs);
                    model.sm_b2.resize(nns);
                    model.el_b2.resize(nns);
                    if (!r.read(model.sm1.data(), nns * fs) ||
                        !r.read(model.el1.data(), nns * fs) ||
                        !r.read(model.sm_b1.data(), nns) ||
                        !r.read(model.el_b1.data(), nns) ||
                        !r.read(model.sm2.data(), nns * fs) ||
                        !r.read(model.el2.data(), nns * fs) ||
                        !r.read(model.sm_b2.data(), nns) ||
                        !r.read(model.el_b2.data(), nns)) {
                        return "weight blob truncated (model)";
                    }
                    found = true;
                } else {
                    if (!r.skip(4 * nns * fs + 4 * nns)) {
                        return "weight blob truncated (skip)";
                    }
                }
            }
        }
    }
    if (!found) {
        return "model not found in weight blob";
    }
    if (r.pos != r.count) {
        return "weight blob size mismatch";
    }

    if (pscrn == 1) {
        prescreener_prep(ps_old.k0, pixel_half);
    } else if (pscrn >= 2) {
        ps_new = all_new[pscrn - 2];
        prescreener_prep(ps_new.k0, pixel_half);
    }
    model_prep_pass(model.sm1, model.el1, model.sm_b1,
        model.sm1.size() / static_cast<size_t>(model.nns), model.nns);
    model_prep_pass(model.sm2, model.el2, model.sm_b2,
        model.sm2.size() / static_cast<size_t>(model.nns), model.nns);
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------

struct Nnedi3Plane {
    int width {};
    int height {};
    int rows {};
    int pad_stride {};
    int pad_h {};
    uint32_t pad_grid_x {};
    uint32_t pad_grid_y {};
    uint32_t pre_grid_x {};          // direct prescreen dispatch (threads)
    uint32_t pred_grid_x {};         // legacy serial direct grid (unused)
    uint32_t pred_grid_direct_x {};  // cooperative direct grid (pscrn==0)
    uint32_t asm_grid_x {};
    VkPipeline pad_pipeline {};
    VkPipeline pre_pipeline {};      // null when pscrn==0
    VkPipeline pred_pipeline {};
    VkPipeline cnt_pipeline {};      // null when pscrn==0
    VkPipeline asm_pipeline {};
    VkDeviceSize up_offset {};       // bytes in the upload staging (padded plane)
    VkDeviceSize download_offset {}; // bytes in the download staging (packed interp)
    VkDeviceSize list_offset {};     // bytes in the device list buffer (uints)
    int32_t pad_elem {};             // element offset of the pad in upload staging
    int32_t dst_elem {};
    int32_t list_elem {};            // uint element offset in the list buffer
};

struct Nnedi3Resource {
    VkBuffer staging {};       // download staging (GTT, DMA target + host reads)
    VkDeviceMemory staging_mem {};
    VkBuffer up_staging {};    // upload staging (VRAM-mapped: CPU pre-pads, GPU reads direct)
    VkDeviceMemory up_mem {};
    VkBuffer dst_buf {};
    VkDeviceMemory dst_mem {};
    VkBuffer list_buf {};        // rejected-pixel indices (device-local)
    VkDeviceMemory list_mem {};
    VkBuffer ind_buf {};         // {groupsX,1,1,count} indirect struct (device-local)
    VkDeviceMemory ind_mem {};
    VkCommandPool pool {};
    VkCommandBuffer cmd {};
    VkFence fence {};
    VkQueryPool query_pool {};   // TEMPORARY per-stage timestamps (remove after tuning)
    VkDescriptorSet desc_set {};
    VkQueue queue {};
    std::mutex * queue_lock {};
    uint8_t * map {};
    uint8_t * up_map {};
    uint32_t staging_type_index {};
    uint32_t up_type_index {};
};

struct Nnedi3Data {
    VSNode * node;
    const VSVideoInfo * vi;
    VSVideoInfo vi_out;

    int device_id, num_streams;
    int field;
    bool dh;
    int qual, pscrn;
    bool use_list;               // prescreen compacts a list (pscrn > 0)
    int peak, elem_bytes;
    int xdim, ydim, nns;
    bool process[3] { true, true, true };

    std::shared_ptr<VK_Device> device;
    VkDescriptorSetLayout set_layout {};
    VkPipelineLayout pipeline_layout {};
    VkDescriptorPool desc_pool {};
    VkShaderModule pad_module {};
    VkShaderModule pre_module {};
    VkShaderModule pred_module {};
    VkShaderModule cnt_module {};
    VkShaderModule asm_module {};
    VkBuffer ps_buf {};
    VkDeviceMemory ps_mem {};
    VkBuffer pdw_buf {};
    VkDeviceMemory pdw_mem {};
    VkBuffer pdb_buf {};
    VkDeviceMemory pdb_mem {};

    VkDeviceSize up_total {};
    VkDeviceSize download_total {};
    VkDeviceSize dst_total {};
    VkDeviceSize list_total {};
    std::array<Nnedi3Plane, 3> planes {};
    FramePool<Nnedi3Resource> pool;

    // Env-gated per-stage timing (VSFEEL_NNEDI3_BENCH): accumulated
    // nanoseconds + frame count, reported as per-frame averages.
    std::atomic<uint64_t> t_record {};
    std::atomic<uint64_t> t_pack {};
    std::atomic<uint64_t> t_gpu {};
    std::atomic<uint64_t> t_interleave {};
    std::atomic<uint64_t> t_frames {};
    // TEMPORARY GPU timestamps (VSFEEL_NNEDI3_TSTAMP): nanoseconds per stage.
    std::atomic<uint64_t> t_ts_pre {};
    std::atomic<uint64_t> t_ts_pred {};
    std::atomic<uint64_t> t_ts_copy {};
    std::atomic<uint64_t> t_ts_n {};
    ~Nnedi3Data() {
        if (trace_on("VSFEEL_NNEDI3_BENCH") && t_frames.load() > 0) {
            const double n = static_cast<double>(t_frames.load());
            fprintf(stderr,
                "[nnedi3-bench] per-frame us: record=%7.1f pack=%7.1f "
                "gpu=%7.1f downcopy=%7.1f (frames=%.0f)\n",
                t_record.load() / 1000.0 / n,
                t_pack.load() / 1000.0 / n, t_gpu.load() / 1000.0 / n,
                t_interleave.load() / 1000.0 / n, n);
        }
        if (trace_on("VSFEEL_NNEDI3_TSTAMP") && t_ts_n.load() > 0) {
            const double n = static_cast<double>(t_ts_n.load());
            fprintf(stderr,
                "[nnedi3-ts] GPU us: pre=%7.1f pred=%7.1f copy=%7.1f (frames=%.0f)\n",
                t_ts_pre.load() / 1000.0 / n, t_ts_pred.load() / 1000.0 / n,
                t_ts_copy.load() / 1000.0 / n, n);
        }
        if (!device) {
            return;
        }
        VkDevice dev = device->device;
        vkDeviceWaitIdle(dev);

        for (auto & resource : pool.items) {
            if (resource.map) {
                vkUnmapMemory(dev, resource.staging_mem);
            }
            if (resource.up_map) {
                vkUnmapMemory(dev, resource.up_mem);
            }
            destroy_common(dev, resource);
            if (resource.query_pool) {
                vkDestroyQueryPool(dev, resource.query_pool, nullptr);
            }
            const std::pair<VkBuffer *, VkDeviceMemory *> bufs[] {
                { &resource.up_staging, &resource.up_mem },
                { &resource.dst_buf, &resource.dst_mem },
                { &resource.list_buf, &resource.list_mem },
                { &resource.ind_buf, &resource.ind_mem },
            };
            for (auto [buf, mem] : bufs) {
                if (*mem) {
                    vkFreeMemory(dev, *mem, nullptr);
                }
                if (*buf) {
                    vkDestroyBuffer(dev, *buf, nullptr);
                }
            }
        }

        const std::pair<VkBuffer *, VkDeviceMemory *> wbufs[] {
            { &ps_buf, &ps_mem },
            { &pdw_buf, &pdw_mem },
            { &pdb_buf, &pdb_mem },
        };
        for (auto [buf, mem] : wbufs) {
            if (*buf) {
                if (*mem) {
                    vkUnmapMemory(dev, *mem);
                    vkFreeMemory(dev, *mem, nullptr);
                }
                vkDestroyBuffer(dev, *buf, nullptr);
            }
        }

        VkPipeline seen[15] {};
        int n_seen = 0;
        for (auto & plane : planes) {
            const VkPipeline ps[5] {
                plane.pad_pipeline, plane.pre_pipeline,
                plane.pred_pipeline, plane.cnt_pipeline,
                plane.asm_pipeline
            };
            for (VkPipeline p : ps) {
                if (!p) {
                    continue;
                }
                bool dup = false;
                for (int i = 0; i < n_seen; ++i) {
                    dup |= seen[i] == p;
                }
                if (!dup && n_seen < 15) {
                    seen[n_seen++] = p;
                    vkDestroyPipeline(dev, p, nullptr);
                }
            }
        }
        if (desc_pool) {
            vkDestroyDescriptorPool(dev, desc_pool, nullptr);
        }
        if (pipeline_layout) {
            vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
        }
        if (set_layout) {
            vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        }
        if (pad_module) {
            vkDestroyShaderModule(dev, pad_module, nullptr);
        }
        if (pre_module) {
            vkDestroyShaderModule(dev, pre_module, nullptr);
        }
        if (pred_module) {
            vkDestroyShaderModule(dev, pred_module, nullptr);
        }
        if (cnt_module) {
            vkDestroyShaderModule(dev, cnt_module, nullptr);
        }
        if (asm_module) {
            vkDestroyShaderModule(dev, asm_module, nullptr);
        }

        release_device(device);
    }
};

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

static std::variant<VkShaderModule, std::string> create_shader_module(
    const VK_Device & dev, const uint32_t * code, size_t code_size) {

    VkShaderModuleCreateInfo module_info {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = code_size,
        .pCode = code
    };

    VkShaderModule module;
    VkResult result = vkCreateShaderModule(dev.device, &module_info, nullptr, &module);
    if (result != VK_SUCCESS) {
        return "vkCreateShaderModule failed: "s + vk_result_string(result);
    }
    return module;
}

struct Nnedi3Spec {
    int32_t width, rows, pad_stride, peak, pscrn, xdim, ydim, nns, qual, use_list;
};

static std::variant<VkPipeline, std::string> create_pipeline(
    const VK_Device & dev, const Nnedi3Spec & spec,
    VkShaderModule module, VkPipelineLayout layout,
    uint32_t required_subgroup_size = 0) {

    std::array<VkSpecializationMapEntry, 10> entries {};
    for (uint32_t i = 0; i < 10; ++i) {
        entries[i] = { i, i * static_cast<uint32_t>(sizeof(int32_t)), sizeof(int32_t) };
    }

    VkSpecializationInfo spec_info {
        .mapEntryCount = static_cast<uint32_t>(entries.size()),
        .pMapEntries = entries.data(),
        .dataSize = sizeof(spec),
        .pData = &spec
    };

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = required_subgroup_size
    };

    VkPipelineShaderStageCreateInfo stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = required_subgroup_size ? &subgroup_size_info : nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = module,
        .pName = "main",
        .pSpecializationInfo = &spec_info
    };

    VkComputePipelineCreateInfo pipeline_info {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stage_info,
        .layout = layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
    };

    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(
        dev.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return "vkCreateComputePipelines failed: "s + vk_result_string(result);
    }
    return pipeline;
}

// Records the per-frame dispatch sequence: per plane (indirect-struct reset,
// prescreen + list compaction unless pscrn==0, predictor, D2H of the packed
// interp rows). The CPU pre-pads the field into VRAM-mapped upload staging
// that prescreen/predict read directly (zero-copy upload: no H2D, no pad
// kernel). All plane regions are disjoint, so dispatches of different planes
// may overlap.
static std::optional<std::string> record_command_buffer(
    const Nnedi3Data & d, Nnedi3Resource & resource) {

    VkDevice dev = d.device->device;

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr
    };

    if (vkBeginCommandBuffer(resource.cmd, &begin_info) != VK_SUCCESS) {
        return "vkBeginCommandBuffer failed";
    }
    // TEMPORARY per-stage timestamps, plane 0 only (remove after tuning)
    vkCmdResetQueryPool(resource.cmd, resource.query_pool, 0, 4);
    vkCmdWriteTimestamp(resource.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        resource.query_pool, 0);
    // Host-write -> shader-read visibility for the CPU-pre-padded upload
    // staging: the submit orders after the CPU pad, but cached/host-visible
    // writes need an explicit availability+visibility edge before the first
    // shader read of the padded planes.
    {
        VkMemoryBarrier host_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
        };
        vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_barrier,
            0, nullptr, 0, nullptr);
    }

    // Temporary per-kernel timing (VSFEEL_NNEDI3_UPTO=n ends the CB after
    // stage n: 1=prescreen, 2=count, 3=predict, 4=readback copy).
    const char * upto_env = getenv("VSFEEL_NNEDI3_UPTO");
    const int upto = upto_env ? atoi(upto_env) : 9;
    bool truncated = false;


    for (int plane = 0; plane < d.vi->format.numPlanes; ++plane) {
        if (!d.process[plane]) {
            continue;
        }
        const auto & cfg = d.planes[plane];

        if (d.use_list) {
            // reset this plane's indirect struct to {0,1,1,0}: groupsX=0,
            // groupsY/Z=1, and the prescreen-accumulated pixel count=0 (the
            // count word must be cleared — prescreen bumps it with atomics)
            vkCmdFillBuffer(resource.cmd, resource.ind_buf, 0, 4, 0);
            vkCmdFillBuffer(resource.cmd, resource.ind_buf, 4, 8, 1);
            vkCmdFillBuffer(resource.cmd, resource.ind_buf, 12, 4, 0);
            VkMemoryBarrier fill_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
            };
            vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fill_barrier,
                0, nullptr, 0, nullptr);
        }

        if (d.use_list) {
            // prescreen: cubic stores + rejected-pixel compaction
            const int32_t push[4] { cfg.list_elem, cfg.pad_elem, cfg.dst_elem, 0 };
            vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.pre_pipeline);
            vkCmdBindDescriptorSets(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
            vkCmdPushConstants(resource.cmd, d.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
            vkCmdDispatch(resource.cmd, cfg.pre_grid_x, 1, 1);
            if (upto <= 1) {
                truncated = true;
                break;
            }

            // Prescreen's atomicAdd writes to indCount must be visible to
            // the count kernel's read (Vulkan needs an explicit
            // availability+visibility edge between the two dispatches —
            // back-to-back dispatches do NOT order memory automatically).
            {
                VkMemoryBarrier pre_count_barrier {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
                };
                vkCmdPipelineBarrier(resource.cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 1, &pre_count_barrier, 0, nullptr, 0, nullptr);
            }
            // Derive this plane's indirect groupsX from the compacted count
            // (single thread), then launch the predictor indirectly — exact
            // sizing, no over-launch on sparse frames.
            vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.cnt_pipeline);
            vkCmdBindDescriptorSets(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
            vkCmdDispatch(resource.cmd, 1, 1, 1);
            if (upto <= 2) {
                truncated = true;
                break;
            }
            VkMemoryBarrier ind_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                 VK_ACCESS_INDIRECT_COMMAND_READ_BIT
            };
            vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &ind_barrier,
                0, nullptr, 0, nullptr);
        }

        // TEMPORARY timestamps, plane 0 only
        if (plane == 0) {
            vkCmdWriteTimestamp(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                resource.query_pool, 1);
        }
        // predictor: indirect launch off the prescreen-maintained count
        // (list mode) or fixed direct grid (pscrn=0, full coverage); the
        // shader strides its loop off the dispatched width, always correct
        {
            const int32_t push[4] { cfg.list_elem, cfg.pad_elem, cfg.dst_elem, 0 };
            vkCmdBindPipeline(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cfg.pred_pipeline);
            vkCmdBindDescriptorSets(resource.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                d.pipeline_layout, 0, 1, &resource.desc_set, 0, nullptr);
            vkCmdPushConstants(resource.cmd, d.pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
            if (d.use_list) {
                vkCmdDispatchIndirect(resource.cmd, resource.ind_buf, 0);
            } else {
                vkCmdDispatch(resource.cmd, cfg.pred_grid_direct_x, 1, 1);
            }
        }
        // TEMPORARY timestamps, plane 0 only
        if (plane == 0) {
            vkCmdWriteTimestamp(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                resource.query_pool, 2);
        }
        // Download: DMA the packed interp rows (device-local dst) to the
        // staging download area. Kept lines never cross the bus — the host
        // interleaves them straight from the source frame. One region per
        // plane; regions are disjoint so a single copy call suffices.
        {
            VkMemoryBarrier copy_barrier {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
            };
            vkCmdPipelineBarrier(resource.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &copy_barrier, 0, nullptr, 0, nullptr);
            const VkDeviceSize field_bytes = static_cast<VkDeviceSize>(cfg.width) *
                static_cast<VkDeviceSize>(cfg.rows) * static_cast<VkDeviceSize>(d.elem_bytes);
            const VkBufferCopy region {
                static_cast<VkDeviceSize>(cfg.dst_elem) * static_cast<VkDeviceSize>(d.elem_bytes),
                cfg.download_offset,
                field_bytes
            };
            vkCmdCopyBuffer(resource.cmd, resource.dst_buf, resource.staging, 1, &region);
        }
        // TEMPORARY timestamps, plane 0 only
        if (plane == 0) {
            vkCmdWriteTimestamp(resource.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                resource.query_pool, 3);
        }
    }

    if (vkEndCommandBuffer(resource.cmd) != VK_SUCCESS) {
        return "vkEndCommandBuffer failed";
    }

    (void)truncated;
    return std::nullopt;
}

static const VSFrame *VS_CC Nnedi3GetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    Nnedi3Data * d = static_cast<Nnedi3Data *>(instanceData);

    const int sn = d->field > 1 ? n / 2 : n;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(sn, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame * src = vsapi->getFrameFilter(sn, d->node, frameCtx);

        VSFrame * dst;
        if (!d->dh) {
            const VSFrame * plane_src[3] { nullptr, nullptr, nullptr };
            int planes[3] { 0, 1, 2 };
            for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                if (!d->process[plane]) {
                    plane_src[plane] = src;
                }
            }
            dst = vsapi->newVideoFrame2(
                &d->vi->format, d->vi->width, d->vi->height,
                plane_src, planes, src, core);
        } else {
            dst = vsapi->newVideoFrame(
                &d->vi_out.format, d->vi_out.width, d->vi_out.height, src, core);
        }

        auto resource = d->pool.take();

        auto set_error = [&](const std::string & error_message) {
            d->pool.give_back(std::move(resource));
            vsapi->setFilterError(("NNEDI3: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        };

        // Source field parity, mirroring the references: parity == 1 keeps
        // the bottom field. Double-rate flips parity on odd outputs.
        const int default_parity = (d->field == 0 || d->field == 2) ? 1 : 0;
        int parity;
        {
            int err;
            const VSMap * props = vsapi->getFramePropertiesRO(src);
            if (d->dh) {
                parity = static_cast<int>(vsapi->mapGetIntSaturated(props, "_Field", 0, &err));
                if (err) {
                    parity = default_parity;
                }
            } else if (d->field > 1) {
                const int field_based = static_cast<int>(
                    vsapi->mapGetIntSaturated(props, "_FieldBased", 0, &err));
                if (field_based == VSC_FIELD_BOTTOM) {
                    parity = 1;
                } else if (field_based == VSC_FIELD_TOP) {
                    parity = 0;
                } else {
                    parity = default_parity;
                }
                if (n & 1) {
                    parity = !parity;
                }
            } else {
                parity = d->field == 0 ? 1 : 0;
            }
            parity = !!parity;
        }
        const size_t bps = static_cast<size_t>(d->elem_bytes);
        const bool bench = trace_on("VSFEEL_NNEDI3_BENCH");
        const auto now = std::chrono::steady_clock::now;
        auto t_prev = now();
        auto bump = [&](std::atomic<uint64_t> & acc) {
            if (bench) {
                const auto t = now();
                acc.fetch_add(
                    static_cast<uint64_t>((t - t_prev).count()),
                    std::memory_order_relaxed);
                t_prev = t;
            }
        };

        VkDevice dev = d->device->device;
        uint8_t * map = resource.map;


        // Pre-pad the field rows into the VRAM-mapped upload staging
        // (zero-copy upload: prescreen/predict read it directly, no H2D).
        // Row-clone pattern: full-row streaming stores + replicated edges.
        // Write-combined memory: streaming stores, never re-read by the CPU.
        {
            const int fp = 1 - parity;
            for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }
                const auto & cfg = d->planes[plane];
                const uint8_t * srcp = vsapi->getReadPtr(src, plane);
                const ptrdiff_t src_stride = vsapi->getStride(src, plane);
                const uint8_t * fieldp = srcp + (d->dh ? 0 : parity * src_stride);
                const ptrdiff_t field_stride = src_stride * (d->dh ? 1 : 2);
                uint8_t * pad = resource.up_map + cfg.up_offset;
                const size_t row_bytes = static_cast<size_t>(cfg.width) * bps;
                const size_t pad_row_bytes = static_cast<size_t>(cfg.pad_stride) * bps;
                for (int i = 0; i < cfg.pad_h; ++i) {
                    int f = i - (MARGIN_V - fp);
                    f = f < 0 ? 0 : (f >= cfg.rows ? cfg.rows - 1 : f);
                    const uint8_t * line = fieldp + static_cast<ptrdiff_t>(f) * field_stride;
                    uint8_t * dst = pad + static_cast<size_t>(i) * pad_row_bytes +
                        static_cast<size_t>(MARGIN_H) * bps;
                    copy_stream_out(dst, line, row_bytes);
                    // replicate edge elements into the margins (element-typed)
                    if (bps == 2) {
                        const uint16_t l = reinterpret_cast<const uint16_t *>(line)[0];
                        const uint16_t r = reinterpret_cast<const uint16_t *>(line)[cfg.width - 1];
                        uint16_t * d16 = reinterpret_cast<uint16_t *>(pad) +
                            static_cast<size_t>(i) * cfg.pad_stride;
                        for (int x = 0; x < MARGIN_H; ++x) {
                            d16[x] = l;
                            d16[MARGIN_H + cfg.width + x] = r;
                        }
                    } else {
                        const float l = reinterpret_cast<const float *>(line)[0];
                        const float r = reinterpret_cast<const float *>(line)[cfg.width - 1];
                        float * d32 = reinterpret_cast<float *>(pad) +
                            static_cast<size_t>(i) * cfg.pad_stride;
                        for (int x = 0; x < MARGIN_H; ++x) {
                            d32[x] = l;
                            d32[MARGIN_H + cfg.width + x] = r;
                        }
                    }
                }
            }
        }
        bump(d->t_pack);
        // Order the non-temporal pad stores before the submit: an SFENCE
        // drains the CPU store buffer so no NT write is still in flight when
        // the GPU starts reading the padded planes.
        _mm_sfence();
        const bool up_coherent =
            !!(d->device->mem_props.memoryTypes[resource.up_type_index].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!up_coherent && d->up_total > 0) {
            VkMappedMemoryRange flush_range {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = resource.up_mem,
                .offset = 0,
                .size = d->up_total,
            };
            checkVK(vkFlushMappedMemoryRanges(dev, 1, &flush_range));
        }
        // Record after the pad lands, then make the CPU writes visible to
        // the GPU: vkQueueSubmit's host-happens-before covers the ordering,
        // but DEVICE_LOCAL staging needs an explicit flush+barrier, not just
        // the queue-latch. Host writes must complete before the shader reads
        // them — this barrier (host write -> shader read) is the visibility
        // edge the zero-copy upload was missing.
        checkVK(vkResetCommandPool(dev, resource.pool, 0));
        if (const auto err = record_command_buffer(*d, resource)) {
            return set_error(*err);
        }
        bump(d->t_record);

        checkVK(submit_with_fence(dev, resource.queue, resource.queue_lock,
            resource.cmd, resource.fence));

        checkVK(vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX));
        bump(d->t_gpu);
        // TEMPORARY GPU timestamps (remove after tuning)
        if (trace_on("VSFEEL_NNEDI3_TSTAMP")) {
            uint64_t ts[4] = {};
            if (vkGetQueryPoolResults(dev, resource.query_pool, 0, 4,
                    sizeof(ts), ts, sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS &&
                ts[3] >= ts[0]) {
                const double period =
                    static_cast<double>(d->device->limits.timestampPeriod);
                d->t_ts_pre.fetch_add(
                    static_cast<uint64_t>((ts[1] - ts[0]) * period),
                    std::memory_order_relaxed);
                d->t_ts_pred.fetch_add(
                    static_cast<uint64_t>((ts[2] - ts[1]) * period),
                    std::memory_order_relaxed);
                d->t_ts_copy.fetch_add(
                    static_cast<uint64_t>((ts[3] - ts[2]) * period),
                    std::memory_order_relaxed);
                d->t_ts_n.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (trace_on("VSFEEL_NNEDI3_COUNT")) {
            // one-off count readback: extra submit, perturbs timing
            VkCommandBuffer cb;
            VkCommandBufferAllocateInfo ainfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = resource.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };
            if (vkAllocateCommandBuffers(dev, &ainfo, &cb) == VK_SUCCESS) {
                VkCommandBufferBeginInfo binfo {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .pNext = nullptr,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                    .pInheritanceInfo = nullptr
                };
                if (vkBeginCommandBuffer(cb, &binfo) == VK_SUCCESS) {
                    VkBufferCopy r { 0, d->download_total, 16 };
                    vkCmdCopyBuffer(cb, resource.ind_buf, resource.staging, 1, &r);
                    vkEndCommandBuffer(cb);
                    submit_with_fence(dev, resource.queue, resource.queue_lock, cb, resource.fence);
                    if (vkWaitForFences(dev, 1, &resource.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS) {
                        uint32_t words[4] = {};
                        std::memcpy(words, map + d->download_total, 16);
                        fprintf(stderr, "[nnedi3-count] frame %d groupsX=%u count=%u\n",
                            n, words[0], words[3]);
                    }
                }
                vkFreeCommandBuffers(dev, resource.pool, 1, &cb);
            }
        }

        const bool dl_coherent =
            !!(d->device->mem_props.memoryTypes[resource.staging_type_index].propertyFlags &
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!dl_coherent && d->download_total > 0) {
            VkMappedMemoryRange inv_range {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = resource.staging_mem,
                .offset = 0,
                .size = d->download_total,
            };
            checkVK(vkInvalidateMappedMemoryRanges(dev, 1, &inv_range));
        }

        // Interleave: kept lines come straight from the source frame, interp
        // lines from the DMA'd packed rows in the staging download area.
        // Unprocessed dh planes (which the reference leaves uninitialized)
        // get kept lines + zeroed gaps deterministically.
        for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
            const auto & cfg = d->planes[plane];
            uint8_t * dstp = vsapi->getWritePtr(dst, plane);
            const ptrdiff_t dst_stride = vsapi->getStride(dst, plane);
            if (!d->process[plane]) {
                if (d->dh) {
                    const uint8_t * srcp = vsapi->getReadPtr(src, plane);
                    const ptrdiff_t src_stride = vsapi->getStride(src, plane);
                    const int w = vsapi->getFrameWidth(src, plane);
                    const int rows = vsapi->getFrameHeight(src, plane);
                    const size_t row_bytes = static_cast<size_t>(w) * bps;
                    for (int r = 0; r < rows; ++r) {
                        std::memcpy(dstp + (parity + 2 * r) * dst_stride,
                            srcp + r * src_stride, row_bytes);
                        std::memset(dstp + (1 - parity + 2 * r) * dst_stride,
                            0, row_bytes);
                    }
                }
                continue;
            }
            const uint8_t * srcp = vsapi->getReadPtr(src, plane);
            const ptrdiff_t src_stride = vsapi->getStride(src, plane);
            const uint8_t * interp = map + cfg.download_offset;
            const size_t row_bytes = static_cast<size_t>(cfg.width) * bps;
            const int height = vsapi->getFrameHeight(dst, plane);
            const int fp = 1 - parity;
            // Kept pass (cached source): one memcpy per kept row.
            if (!d->dh) {
                for (int y = parity; y < height; y += 2) {
                    std::memcpy(dstp + y * dst_stride, srcp + y * src_stride, row_bytes);
                }
            } else {
                for (int y = parity; y < height; y += 2) {
                    std::memcpy(dstp + y * dst_stride,
                        srcp + ((y - parity) / 2) * src_stride, row_bytes);
                }
            }
            // Interp pass (WC staging): non-temporal loads, never cached.
            for (int r = 0; r < cfg.rows; ++r) {
                copy_stream_read(dstp + (fp + 2 * r) * dst_stride,
                    interp + r * row_bytes, row_bytes);
            }
        }
        bump(d->t_interleave);
        d->t_frames.fetch_add(1, std::memory_order_relaxed);

        vsapi->freeFrame(src);
        d->pool.give_back(std::move(resource));

        VSMap * props = vsapi->getFramePropertiesRW(dst);
        vsapi->mapSetInt(props, "_FieldBased", VSC_FIELD_PROGRESSIVE, maReplace);
        vsapi->mapDeleteKey(props, "_Field");
        if (d->field > 1) {
            int err_num, err_den;
            int64_t dur_num = vsapi->mapGetInt(props, "_DurationNum", 0, &err_num);
            int64_t dur_den = vsapi->mapGetInt(props, "_DurationDen", 0, &err_den);
            if (!err_num && !err_den) {
                vsh::muldivRational(&dur_num, &dur_den, 1, 2);
                vsapi->mapSetInt(props, "_DurationNum", dur_num, maReplace);
                vsapi->mapSetInt(props, "_DurationDen", dur_den, maReplace);
            }
        }

        return dst;
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

static void VS_CC Nnedi3Free(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    Nnedi3Data * d = static_cast<Nnedi3Data *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}

// Upload-once helper for a constant weight buffer: creates a host-visible
// coherent+cached buffer, maps it persistently, and copies the bytes in.
static std::optional<std::string> upload_weights(
    Nnedi3Data & d, VkBuffer & buf, VkDeviceMemory & mem,
    const void * bytes, VkDeviceSize size) {

    VkDevice dev = d.device->device;

    VkBufferCreateInfo buffer_info {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = std::max<VkDeviceSize>(size, 4),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr
    };
    if (vkCreateBuffer(dev, &buffer_info, nullptr, &buf) != VK_SUCCESS) {
        return "vkCreateBuffer (weights) failed";
    }
    {
        const auto result = allocate_memory(*d.device, buf,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (std::holds_alternative<std::string>(result)) {
            return std::get<std::string>(result);
        }
        mem = std::get<AllocatedMemory>(result).memory;
    }
    void * wmap = nullptr;
    if (vkMapMemory(dev, mem, 0, std::max<VkDeviceSize>(size, 4), 0, &wmap) != VK_SUCCESS) {
        return "vkMapMemory (weights) failed";
    }
    std::memcpy(wmap, bytes, size);
    return std::nullopt;
}

static void VS_CC Nnedi3Create(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<Nnedi3Data>() };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);
    d->vi_out = *d->vi;

    int error;

    auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, ("NNEDI3: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    const auto & fmt = d->vi->format;
    const int bits = fmt.bitsPerSample;
    const bool depth_ok = (fmt.sampleType == stInteger && bits == 16) ||
                          (fmt.sampleType == stFloat && bits == 32);
    if (!depth_ok || d->vi->width <= 0 || d->vi->height <= 0 ||
        (fmt.colorFamily != cfGray && fmt.colorFamily != cfYUV && fmt.colorFamily != cfRGB)) {
        return set_error("only 16-bit integer and 32-bit float Gray/YUV/RGB input supported.");
    }

    d->peak = fmt.sampleType == stInteger ? (1 << bits) - 1 : 0;
    d->elem_bytes = bits / 8;

    d->field = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "field", 0, nullptr));
    if (d->field < 0 || d->field > 3) {
        return set_error("field must be 0, 1, 2, or 3.");
    }
    d->dh = !!vsapi->mapGetInt(in, "dh", 0, &error);
    if (d->dh && d->field > 1) {
        return set_error("field must be 0 or 1 when dh is true.");
    }

    for (int i = 0; i < 3; ++i) {
        d->process[i] = true;
    }
    const int num_plane_args = vsapi->mapNumElements(in, "planes");
    if (num_plane_args > 0) {
        for (int i = 0; i < 3; ++i) {
            d->process[i] = false;
        }
        for (int i = 0; i < num_plane_args; ++i) {
            const int p = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "planes", i, nullptr));
            if (p < 0 || p >= fmt.numPlanes) {
                return set_error("plane index out of range.");
            }
            if (d->process[p]) {
                return set_error("plane specified twice.");
            }
            d->process[p] = true;
        }
    }

    const int nsize = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "nsize", 0, &error));
    const int nsize_v = error ? 6 : nsize;
    const int nns_sel = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "nns", 0, &error));
    const int nns_v = error ? 1 : nns_sel;
    d->qual = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "qual", 0, &error));
    if (error) {
        d->qual = 1;
    }
    const int etype = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "etype", 0, &error));
    const int etype_v = error ? 0 : etype;
    d->pscrn = vsh::int64ToIntS(vsapi->mapGetIntSaturated(in, "pscrn", 0, &error));
    if (error) {
        d->pscrn = 2;
    }
    if (nsize_v < 0 || nsize_v > 6) {
        return set_error("nsize must be between 0 and 6 (inclusive).");
    }
    if (nns_v < 0 || nns_v > 4) {
        return set_error("nns must be between 0 and 4 (inclusive).");
    }
    if (d->qual < 1 || d->qual > 2) {
        return set_error("qual must be 1 or 2.");
    }
    if (etype_v < 0 || etype_v > 1) {
        return set_error("etype must be 0 or 1.");
    }
    if (d->pscrn < 0 || d->pscrn > 4) {
        return set_error("pscrn must be between 0 and 4 (inclusive).");
    }
    d->use_list = d->pscrn > 0;

    if (!d->dh) {
        for (int plane = 0; plane < fmt.numPlanes; ++plane) {
            const int ph = d->vi->height >> (plane > 0 ? fmt.subSamplingH : 0);
            if (d->process[plane] && (ph & 1) != 0) {
                return set_error("plane height must be mod 2 when dh is false.");
            }
        }
    }

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }
    if (device_id < 0) {
        return set_error("invalid device ID.");
    }

    int num_streams = vsh::int64ToIntS(vsapi->mapGetInt(in, "num_streams", 0, &error));
    if (error) {
        // Per-stream efficiency is the target (not scaling): default stays
        // low; pipelining across streams still helps throughput.
        num_streams = 2;
    }
    if (num_streams < 1 || num_streams > 32) {
        return set_error("num_streams must be 1..32.");
    }
    d->num_streams = num_streams;

    if (d->field > 1) {
        if (d->vi_out.numFrames > INT32_MAX / 2) {
            return set_error("resulting clip is too long.");
        }
        d->vi_out.numFrames *= 2;
        vsh::muldivRational(&d->vi_out.fpsNum, &d->vi_out.fpsDen, 2, 1);
    }
    if (d->dh) {
        d->vi_out.height *= 2;
    }

    // Weights: parse the embedded blob, prep, and pack the GPU layouts.
    PsOldWeights ps_old {};
    PsNewWeights ps_new {};
    ModelWeights model {};
    {
        const double pixel_half = fmt.sampleType == stFloat
            ? 0.5 : static_cast<double>(d->peak) / 2.0;
        if (const auto err = parse_weights(
                nsize_v, nns_v, etype_v, d->pscrn, pixel_half, ps_old, ps_new, model)) {
            return set_error(*err);
        }
    }
    d->xdim = model.xdim;
    d->ydim = model.ydim;
    d->nns = model.nns;
    const size_t fs = static_cast<size_t>(d->xdim) * d->ydim;
    const size_t nns = static_cast<size_t>(d->nns);
    const size_t num_q = static_cast<size_t>(d->qual);

    // prescreener vec4 blob: transposed layer-0 + inline deeper layers
    std::vector<float> ps_blob;
    if (d->pscrn == 1) {
        ps_blob.resize(63 * 4, 0.0f);
        for (int k = 0; k < 48; ++k) {
            for (int n = 0; n < 4; ++n) {
                ps_blob[k * 4 + n] = ps_old.k0[n][k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[48 * 4 + n] = ps_old.b0[n];
        }
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 4; ++k) {
                ps_blob[(49 + n) * 4 + k] = ps_old.k1[n][k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[53 * 4 + n] = ps_old.b1[n];
        }
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 4; ++k) {
                ps_blob[(54 + n * 2) * 4 + k] = ps_old.k2[n][k];
                ps_blob[(55 + n * 2) * 4 + k] = ps_old.k2[n][4 + k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[62 * 4 + n] = ps_old.b2[n];
        }
    } else if (d->pscrn >= 2) {
        ps_blob.resize(70 * 4, 0.0f);
        for (int k = 0; k < 64; ++k) {
            for (int n = 0; n < 4; ++n) {
                ps_blob[k * 4 + n] = ps_new.k0[n][k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[64 * 4 + n] = ps_new.b0[n];
        }
        for (int n = 0; n < 4; ++n) {
            for (int k = 0; k < 4; ++k) {
                ps_blob[(65 + n) * 4 + k] = ps_new.k1[n][k];
            }
        }
        for (int n = 0; n < 4; ++n) {
            ps_blob[69 * 4 + n] = ps_new.b1[n];
        }
    } else {
        ps_blob.resize(4, 0.0f);
    }

    // predictor weights: (softmax, elliott) pairs in [q][k][p] order
    std::vector<float> pdw_blob(num_q * fs * nns * 2);
    for (size_t q = 0; q < num_q; ++q) {
        const std::vector<float> & sm = q == 0 ? model.sm1 : model.sm2;
        const std::vector<float> & el = q == 0 ? model.el1 : model.el2;
        for (size_t k = 0; k < fs; ++k) {
            for (size_t p = 0; p < nns; ++p) {
                pdw_blob[((q * fs + k) * nns + p) * 2 + 0] = sm[p * fs + k];
                pdw_blob[((q * fs + k) * nns + p) * 2 + 1] = el[p * fs + k];
            }
        }
    }
    // predictor biases: (smB, elB) pairs in [q][p] order
    std::vector<float> pdb_blob(num_q * nns * 2);
    for (size_t q = 0; q < num_q; ++q) {
        const std::vector<float> & sm_b = q == 0 ? model.sm_b1 : model.sm_b2;
        const std::vector<float> & el_b = q == 0 ? model.el_b1 : model.el_b2;
        for (size_t p = 0; p < nns; ++p) {
            pdb_blob[(q * nns + p) * 2 + 0] = sm_b[p];
            pdb_blob[(q * nns + p) * 2 + 1] = el_b[p];
        }
    }

    {
        const auto result = get_device(device_id);
        if (std::holds_alternative<std::string>(result)) {
            return set_error(std::get<std::string>(result));
        }
        d->device = std::get<std::shared_ptr<VK_Device>>(result);
        d->device_id = device_id;
    }

    VkDevice dev = d->device->device;

    // Descriptor set layout: field / pad / interp / prescreener / weights /
    // biases / staging-out / pixel list / indirect struct.
    {
        VkDescriptorSetLayoutBinding bindings[9] {};
        for (uint32_t i = 0; i < 9; ++i) {
            bindings[i] = {
                i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr
            };
        }
        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 9,
            .pBindings = bindings
        };
        checkVK(vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &d->set_layout));
    }
    {
        VkPushConstantRange push_constant_range {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = 4 * sizeof(int32_t)
        };
        VkPipelineLayoutCreateInfo pipeline_layout_info {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &d->set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range
        };
        checkVK(vkCreatePipelineLayout(dev, &pipeline_layout_info, nullptr, &d->pipeline_layout));
    }
    {
        VkDescriptorPoolSize pool_size {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9 * static_cast<uint32_t>(d->num_streams)
        };
        VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = static_cast<uint32_t>(d->num_streams),
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size
        };
        checkVK(vkCreateDescriptorPool(dev, &pool_info, nullptr, &d->desc_pool));
    }

    // Shared weight buffers (map stays valid for the filter lifetime).
    if (const auto err = upload_weights(*d, d->ps_buf, d->ps_mem,
            ps_blob.data(), ps_blob.size() * sizeof(float))) {
        return set_error(*err);
    }
    if (const auto err = upload_weights(*d, d->pdw_buf, d->pdw_mem,
            pdw_blob.data(), pdw_blob.size() * sizeof(float))) {
        return set_error(*err);
    }
    if (const auto err = upload_weights(*d, d->pdb_buf, d->pdb_mem,
            pdb_blob.data(), pdb_blob.size() * sizeof(float))) {
        return set_error(*err);
    }

    // Shader modules for this bit depth.
    {
        const uint32_t * pad_code = nullptr;
        size_t pad_size = 0;
        const uint32_t * pre_code = nullptr;
        size_t pre_size = 0;
        const uint32_t * pred_code = nullptr;
        size_t pred_size = 0;
        const uint32_t * cnt_code = nullptr;
        size_t cnt_size = 0;
        const uint32_t * asm_code = nullptr;
        size_t asm_size = 0;
        if (d->elem_bytes == 2) {
            pad_code = nnedi3_16_pad_spv;   pad_size = nnedi3_16_pad_spv_size;
            pre_code = nnedi3_16_prescreen_spv; pre_size = nnedi3_16_prescreen_spv_size;
            pred_code = nnedi3_16_predict_spv; pred_size = nnedi3_16_predict_spv_size;
            cnt_code = nnedi3_16_count_spv; cnt_size = nnedi3_16_count_spv_size;
            asm_code = nnedi3_16_assemble_spv; asm_size = nnedi3_16_assemble_spv_size;
        } else {
            pad_code = nnedi3_32_pad_spv;   pad_size = nnedi3_32_pad_spv_size;
            pre_code = nnedi3_32_prescreen_spv; pre_size = nnedi3_32_prescreen_spv_size;
            pred_code = nnedi3_32_predict_spv; pred_size = nnedi3_32_predict_spv_size;
            cnt_code = nnedi3_32_count_spv; cnt_size = nnedi3_32_count_spv_size;
            asm_code = nnedi3_32_assemble_spv; asm_size = nnedi3_32_assemble_spv_size;
        }
        {
            const auto result = create_shader_module(*d->device, pad_code, pad_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pad_module = std::get<VkShaderModule>(result);
        }
        if (d->use_list) {
            const auto result = create_shader_module(*d->device, pre_code, pre_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pre_module = std::get<VkShaderModule>(result);
        }
        if (d->use_list) {
            const auto result = create_shader_module(*d->device, cnt_code, cnt_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->cnt_module = std::get<VkShaderModule>(result);
        }
        {
            const auto result = create_shader_module(*d->device, pred_code, pred_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->pred_module = std::get<VkShaderModule>(result);
        }
        {
            const auto result = create_shader_module(*d->device, asm_code, asm_size);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            d->asm_module = std::get<VkShaderModule>(result);
        }
    }

    // The cooperative predictor needs one 32-lane subgroup per 4 pixels.
    uint32_t pred_subgroup = 0;
    if (d->device->subgroup_size_control &&
        d->device->min_subgroup_size <= 32 && 32 <= d->device->max_subgroup_size) {
        pred_subgroup = 32;
    } else {
        return set_error("device cannot run 32-lane subgroups "
                         "(required by the predictor kernel).");
    }

    // Per-plane geometry and pipelines (deduplicated across identical planes).
    const uint32_t max_grid_x = d->device->limits.maxComputeWorkGroupCount[0];
    const uint32_t max_grid_y = d->device->limits.maxComputeWorkGroupCount[1];
    {
        struct Key { int w, rows, stride, pscrn, xdim, ydim, nns, qual; };
        std::array<Key, 3> keys {};
        std::array<VkPipeline, 3> pad_pipes {};
        std::array<VkPipeline, 3> pre_pipes {};
        std::array<VkPipeline, 3> pred_pipes {};
        std::array<VkPipeline, 3> cnt_pipes {};
        std::array<VkPipeline, 3> asm_pipes {};
        int n_keys = 0;
        for (int plane = 0; plane < fmt.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            auto & cfg = d->planes[plane];
            cfg.width = plane == 0 ? d->vi_out.width : d->vi_out.width >> fmt.subSamplingW;
            cfg.height = plane == 0 ? d->vi_out.height : d->vi_out.height >> fmt.subSamplingH;
            cfg.rows = cfg.height / 2;
            cfg.pad_stride = (cfg.width + MARGIN_H * 2 + 15) & ~15;
            cfg.pad_h = cfg.rows + MARGIN_V * 2;
            if (static_cast<int64_t>(cfg.width) * cfg.rows >= (int64_t(1) << 31) ||
                cfg.rows < 1 || cfg.width < 1) {
                return set_error("plane geometry out of range.");
            }
            cfg.pad_grid_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.width + MARGIN_H * 2) + 31) / 32, max_grid_x);
            cfg.pad_grid_y = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.pad_h) + 7) / 8, max_grid_y);
            // prescreen: one thread per pixel group (P=1 old, P=4 new)
            const uint32_t pps = d->pscrn == 1 ? 1 : 4;
            cfg.pre_grid_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.width) * static_cast<uint32_t>(cfg.rows) +
                 pps * 128 - 1) / (pps * 128), max_grid_x);
            // direct predict grid (both modes): full pixel coverage with
            // 1024-thread workgroups (see PREDICT entry); count-bounded by
            // the kernel's early return, always correct
            cfg.pred_grid_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.width) * static_cast<uint32_t>(cfg.rows) + 1023) / 1024,
                max_grid_x);
            // cooperative direct grid (pscrn==0): 4 subgroups x PXP pixels
            // per 128-thread workgroup (PXP mirrors the shader)
            {
                const int host_ppl = (d->nns + 31) / 32;
                const int host_fs = d->xdim * d->ydim;
                const int host_pxp = (host_ppl <= 2 && host_fs <= 128) ? 8 : 4;
                const uint32_t ppg = static_cast<uint32_t>(4 * host_pxp);
                cfg.pred_grid_direct_x = std::min<uint32_t>(
                    (static_cast<uint32_t>(cfg.width) * static_cast<uint32_t>(cfg.rows) + ppg - 1) / ppg,
                    max_grid_x);
            }
            cfg.asm_grid_x = std::min<uint32_t>(
                (static_cast<uint32_t>(cfg.width) * static_cast<uint32_t>(cfg.height) + 63) / 64,
                max_grid_x);

            int ki = 0;
            for (; ki < n_keys; ++ki) {
                if (keys[ki].w == cfg.width && keys[ki].rows == cfg.rows &&
                    keys[ki].stride == cfg.pad_stride &&
                    keys[ki].pscrn == d->pscrn && keys[ki].xdim == d->xdim &&
                    keys[ki].ydim == d->ydim && keys[ki].nns == d->nns &&
                    keys[ki].qual == d->qual) {
                    break;
                }
            }
            if (ki == n_keys) {
                const Nnedi3Spec spec {
                    cfg.width, cfg.rows, cfg.pad_stride, d->peak,
                    d->pscrn, d->xdim, d->ydim, d->nns, d->qual,
                    d->use_list ? 1 : 0
                };
                {
                    const auto result = create_pipeline(
                        *d->device, spec, d->pad_module, d->pipeline_layout);
                    if (std::holds_alternative<std::string>(result)) {
                        return set_error(std::get<std::string>(result));
                    }
                    pad_pipes[n_keys] = std::get<VkPipeline>(result);
                }
                if (d->use_list) {
                    const auto result = create_pipeline(
                        *d->device, spec, d->pre_module, d->pipeline_layout,
                        pred_subgroup);
                    if (std::holds_alternative<std::string>(result)) {
                        return set_error(std::get<std::string>(result));
                    }
                    pre_pipes[n_keys] = std::get<VkPipeline>(result);
                }
                if (d->use_list) {
                    const auto result = create_pipeline(
                        *d->device, spec, d->cnt_module, d->pipeline_layout);
                    if (std::holds_alternative<std::string>(result)) {
                        return set_error(std::get<std::string>(result));
                    }
                    cnt_pipes[n_keys] = std::get<VkPipeline>(result);
                }
                {
                    const auto result = create_pipeline(
                        *d->device, spec, d->pred_module, d->pipeline_layout,
                        pred_subgroup);
                    if (std::holds_alternative<std::string>(result)) {
                        return set_error(std::get<std::string>(result));
                    }
                    pred_pipes[n_keys] = std::get<VkPipeline>(result);
                }
                {
                    const auto result = create_pipeline(
                        *d->device, spec, d->asm_module, d->pipeline_layout);
                    if (std::holds_alternative<std::string>(result)) {
                        return set_error(std::get<std::string>(result));
                    }
                    asm_pipes[n_keys] = std::get<VkPipeline>(result);
                }
                keys[n_keys] = { cfg.width, cfg.rows, cfg.pad_stride, d->pscrn, d->xdim, d->ydim, d->nns, d->qual };
                ++n_keys;
            }
            cfg.pad_pipeline = pad_pipes[ki];
            cfg.pre_pipeline = pre_pipes[ki];
            cfg.pred_pipeline = pred_pipes[ki];
            cfg.cnt_pipeline = cnt_pipes[ki];
            cfg.asm_pipeline = asm_pipes[ki];
        }
        bool any = false;
        for (int plane = 0; plane < fmt.numPlanes; ++plane) {
            any |= d->process[plane];
        }
        if (!any) {
            return set_error("no planes to process.");
        }
    }

    // Buffer region offsets (raw bytes), each region 32-byte aligned. The
    // upload staging holds CPU-pre-padded planes (zero-copy GPU reads); the
    // download staging holds packed interp rows DMA'd from device-local dst.
    auto align32 = [](VkDeviceSize v) { return (v + 31) & ~VkDeviceSize(31); };
    {
        VkDeviceSize up = 0, down = 0, dst = 0, list = 0;
        for (int plane = 0; plane < fmt.numPlanes; ++plane) {
            if (!d->process[plane]) {
                continue;
            }
            auto & cfg = d->planes[plane];
            const VkDeviceSize field_bytes =
                static_cast<VkDeviceSize>(cfg.width) * cfg.rows * d->elem_bytes;
            const VkDeviceSize pad_bytes =
                static_cast<VkDeviceSize>(cfg.pad_stride) * cfg.pad_h * d->elem_bytes;
            const VkDeviceSize list_bytes =
                static_cast<VkDeviceSize>(cfg.width) * cfg.rows * sizeof(uint32_t);
            cfg.up_offset = align32(up);
            cfg.download_offset = align32(down);
            cfg.list_offset = align32(list);
            up = align32(cfg.up_offset + pad_bytes);
            down = align32(cfg.download_offset + field_bytes);
            list = align32(cfg.list_offset + list_bytes);
            cfg.pad_elem = static_cast<int32_t>(cfg.up_offset / d->elem_bytes);
            cfg.dst_elem = static_cast<int32_t>(align32(dst) / d->elem_bytes);
            cfg.list_elem = static_cast<int32_t>(cfg.list_offset / 4);
            dst = align32(dst) + field_bytes;
        }
        d->up_total = up;
        d->download_total = down;
        d->dst_total = dst;
        d->list_total = list;
    }


    // Download staging (GTT): packed interp rows + 16 B count-debug scratch.
    const VkDeviceSize staging_size =
        std::max<VkDeviceSize>(d->download_total + 16, 8);
    // Upload staging (VRAM-mapped): CPU-pre-padded planes, GPU reads direct.
    const VkDeviceSize up_size = std::max<VkDeviceSize>(d->up_total, 4);
    const VkDeviceSize dst_size = std::max<VkDeviceSize>(d->dst_total, 4);
    const VkDeviceSize list_size = std::max<VkDeviceSize>(d->list_total, 4);
    constexpr VkDeviceSize ind_size = 16;

    d->pool.semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
    d->pool.reserve(d->num_streams);

    uint32_t num_queues = std::min(
        d->num_streams, static_cast<int>(d->device->queue_count));

    for (int i = 0; i < d->num_streams; ++i) {
        Nnedi3Resource resource;

        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = staging_size,
                // TRANSFER_DST: D2H readback target, host-mapped for the
                // interleave copies (no GPU storage access anymore)
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.staging));
        }

        {
            const auto result = allocate_memory(
                *d->device, resource.staging,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.staging_mem = std::get<AllocatedMemory>(result).memory;
            resource.staging_type_index = std::get<AllocatedMemory>(result).type_index;
        }
        // Upload staging (VRAM-mapped): the host pre-pads into it, the GPU
        // reads the padded planes directly (zero-copy upload, STORAGE read).
        {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = up_size,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            checkVK(vkCreateBuffer(dev, &buffer_info, nullptr, &resource.up_staging));
        }
        {
            // VRAM-mapped upload: HOST_VISIBLE|COHERENT without CACHED lands
            // on the ReBAR type (write-combined, full-speed GPU reads); the
            // CACHED variant lands on GTT (slow GPU reads of every window
            // tap). NT stores make the CPU side fast on WC.
            const auto result = allocate_memory(
                *d->device, resource.up_staging,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return set_error(std::get<std::string>(result));
            }
            resource.up_mem = std::get<AllocatedMemory>(result).memory;
            resource.up_type_index = std::get<AllocatedMemory>(result).type_index;
        }
        auto make_device_buffer = [&](VkBuffer & buf, VkDeviceMemory & mem,
                                      VkDeviceSize size, const char * what,
                                      VkBufferUsageFlags extra_usage = 0)
            -> std::optional<std::string> {
            VkBufferCreateInfo buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = size,
                .usage = (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT) | extra_usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr
            };
            if (vkCreateBuffer(dev, &buffer_info, nullptr, &buf) != VK_SUCCESS) {
                return "vkCreateBuffer ("s + what + ") failed.";
            }
            const auto result = allocate_memory(
                *d->device, buf, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (std::holds_alternative<std::string>(result)) {
                return std::get<std::string>(result);
            }
            mem = std::get<AllocatedMemory>(result).memory;
            return std::nullopt;
        };

        if (const auto err = make_device_buffer(
                resource.dst_buf, resource.dst_mem, dst_size, "dst")) {
            return set_error(*err);
        }
        if (const auto err = make_device_buffer(
                resource.list_buf, resource.list_mem, list_size, "list")) {
            return set_error(*err);
        }
        if (const auto err = make_device_buffer(
                resource.ind_buf, resource.ind_mem, ind_size, "ind",
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)) {
            return set_error(*err);
        }

        {
            VkCommandPoolCreateInfo pool_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueFamilyIndex = d->device->queue_family
            };
            checkVK(vkCreateCommandPool(dev, &pool_info, nullptr, &resource.pool));
        }

        {
            VkCommandBufferAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = resource.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };
            checkVK(vkAllocateCommandBuffers(dev, &alloc_info, &resource.cmd));
        }

        {
            VkFenceCreateInfo fence_info {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };
            checkVK(vkCreateFence(dev, &fence_info, nullptr, &resource.fence));
        }

        {
            // TEMPORARY per-stage timestamps (remove after tuning)
            VkQueryPoolCreateInfo query_info {
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = 5,
                .pipelineStatistics = 0
            };
            checkVK(vkCreateQueryPool(dev, &query_info, nullptr, &resource.query_pool));
        }

        {
            VkDescriptorSetAllocateInfo alloc_info {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = d->desc_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &d->set_layout
            };
            checkVK(vkAllocateDescriptorSets(dev, &alloc_info, &resource.desc_set));
        }

        {
            VkDescriptorBufferInfo infos[9] {
                { resource.up_staging, 0, VK_WHOLE_SIZE },
                { resource.up_staging, 0, VK_WHOLE_SIZE },
                { resource.dst_buf, 0, VK_WHOLE_SIZE },
                { d->ps_buf, 0, VK_WHOLE_SIZE },
                { d->pdw_buf, 0, VK_WHOLE_SIZE },
                { d->pdb_buf, 0, VK_WHOLE_SIZE },
                { resource.staging, 0, VK_WHOLE_SIZE },
                { resource.list_buf, 0, VK_WHOLE_SIZE },
                { resource.ind_buf, 0, VK_WHOLE_SIZE },
            };
            VkWriteDescriptorSet writes[9] {};
            for (uint32_t b = 0; b < 9; ++b) {
                writes[b] = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = resource.desc_set,
                    .dstBinding = b,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &infos[b],
                    .pTexelBufferView = nullptr
                };
            }
            vkUpdateDescriptorSets(dev, 9, writes, 0, nullptr);
        }

        checkVK(vkMapMemory(dev, resource.staging_mem, 0, staging_size, 0,
            reinterpret_cast<void **>(&resource.map)));
        checkVK(vkMapMemory(dev, resource.up_mem, 0, up_size, 0,
            reinterpret_cast<void **>(&resource.up_map)));

        resource.queue = d->device->queues[i % num_queues].queue;
        resource.queue_lock = d->device->queues[i % num_queues].lock.get();

        d->pool.push(std::move(resource));
    }

    VSFilterDependency deps[1] = {{ d->node, d->field > 1 ? rpGeneral : rpStrictSpatial }};

    Nnedi3Data * data = d.release();

    vsapi->createVideoFilter(
        out, "NNEDI3", &data->vi_out,
        Nnedi3GetFrame, Nnedi3Free,
        fmParallel, deps, 1, data, core);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void vsfeel_register_nnedi3(const VSPLUGINAPI * vspapi, VSPlugin * plugin) {
    vspapi->registerFunction(
        "NNEDI3",
        "clip:vnode;"
        "field:int:opt;"
        "dh:int:opt;"
        "planes:int[]:opt;"
        "nsize:int:opt;"
        "nns:int:opt;"
        "qual:int:opt;"
        "etype:int:opt;"
        "pscrn:int:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        "clip:vnode;",
        Nnedi3Create, nullptr, plugin
    );
}