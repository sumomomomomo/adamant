#pragma once

#include "basalt/core/on_device.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
#include <vector>

namespace basalt::models::qwen4exp {

// A temporary stage keeps BF16 storage, with the main session's dequantized values.
inline bool mirror_prompt_hc8(const core::WeightTable& main, const core::WeightTable& stage,
                             int stage_device, std::string& error) {
    struct Job { kernels::Hc8View view; uint16_t* destination; };
    std::vector<Job> jobs;
    size_t elements = 0;
    for (const auto& [name, target] : stage.all()) {
        const core::WeightRef* source = main.find(name);
        kernels::Hc8View view;
        if (source == nullptr || !kernels::hc8_view(source->data, view) || !view.owned) continue;
        if (target.data == nullptr) continue;
        if (target.kind != core::WeightKind::Bf16InF32 || target.ne0 != view.cols || target.ne1 != view.rows) {
            error = "prompt stage: hyper-connection shape differs: " + name;
            return false;
        }
        jobs.push_back({view, (uint16_t*) target.data});
        elements = std::max(elements, (size_t) view.rows * view.cols);
    }
    if (jobs.empty()) return true;
    const core::OnDevice on0(0);
    uint16_t* scratch = nullptr;
    cudaError_t status = cudaMalloc((void**) &scratch, elements * sizeof(uint16_t));
    for (const Job& job : jobs) {
        if (status != cudaSuccess) break;
        kernels::hc8_dequant_bf16(job.view, 0, job.view.rows, scratch, nullptr);
        status = cudaStreamSynchronize(nullptr);
        if (status == cudaSuccess)
            status = cudaMemcpyPeer(job.destination, stage_device, scratch, 0,
                                    (size_t) job.view.rows * job.view.cols * sizeof(uint16_t));
    }
    if (scratch != nullptr) cudaFree(scratch);
    if (status == cudaSuccess) return true;
    error = std::string("prompt stage: hyper-connection copy: ") + cudaGetErrorString(status);
    return false;
}

}  // namespace basalt::models::qwen4exp
