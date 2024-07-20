#pragma once
#include <filesystem>
#include <array>
#include <utility>

#if BUILD_CUDA
#include <nvvm.h>
#include <cuda.h>
#include <nvtx3/nvToolsExt.h>


namespace cuda
{
// https://cuda.readthedocs.io/ko/latest/CUDA_ex/
class CudaContext
{
public:
    CudaContext();

    CUdeviceptr create_managed_buffer(std::size_t size) const;
    std::pair<CUmodule, CUfunction> create_kernel(const std::filesystem::path& path, std::string_view kernel_name) const;
    
    void launch_kernel(CUfunction kernel, std::array<std::uint32_t, 3> blocks, std::array<std::uint32_t, 3> threads_per_block, std::uint32_t slm_dynamic, std::vector<CUdeviceptr*> args) const;
    void synchronize();


private:
    CUcontext context_{};
    CUdevice device_{};
    CUstream stream_{};
};
}

#endif // #if BUILD_CUDA