#include "cuda_context.h"

#include <cassert>
#include <iostream>
#include <format>
#include <fstream>

#if BUILD_CUDA
namespace
{
// If 'err' is non-zero, emit an error message and exit.
#define CHECK_CUDA_ERROR(err) __check_cuda_errors(err, __FILE__, __LINE__)
static void __check_cuda_errors(CUresult err, const char* filename, int line)
{
    assert(filename);
    if (CUDA_SUCCESS != err)
    {
        const char* ename = NULL;
        const CUresult res = cuGetErrorName(err, &ename);
        const auto out_str = std::format("CUDA API ERROR: {}: {}, from file: {}, line: {}", std::uint32_t(err), (CUDA_SUCCESS == res) ? ename : "Unknown", filename, line);
        std::cerr << out_str << std::endl;
        std::exit(err);
    }
}

// Return a CUDA capable device or exit if one cannot be found.
static CUdevice cuda_device_init()
{
    CUresult err = cuInit(0);
    int deviceCount = 0;
    if (CUDA_SUCCESS == err)
    {
        CHECK_CUDA_ERROR(cuDeviceGetCount(&deviceCount));
    }

    if (deviceCount == 0)
    {
        std::cerr << "cudaDeviceInit error: no devices supporting CUDA" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Locate a CUDA supporting device and its name.
    CUdevice cuDevice = 0;
    CHECK_CUDA_ERROR(cuDeviceGet(&cuDevice, 0));
    char name[128];
    cuDeviceGetName(name, sizeof(name), cuDevice);
    std::cout << std::format("Using CUDA Device [0]: {}", name) << std::endl;;

    // Obtain the device's compute capability.
    int major = 0;
    CHECK_CUDA_ERROR(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    if (major < 5)
    {
        std::cerr << "Device 0 is not sm_50 or later" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return cuDevice;
}


static CUresult init_cuda(CUcontext& context, CUdevice& device, CUstream& stream)
{
    // Initialize.
    device = cuda_device_init();

    // Create a CUDA context on the device.
    CHECK_CUDA_ERROR(cuCtxCreate(&context, 0, device));

    CHECK_CUDA_ERROR(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    return CUDA_SUCCESS;
}


}

cuda::CudaContext::CudaContext()
{
    //auto domainHandle = nvtxDomainCreate("AI_Playground");
    assert(init_cuda(context_, device_, stream_) == CUDA_SUCCESS);
}

std::pair<CUmodule, CUfunction> cuda::CudaContext::create_kernel(const std::filesystem::path& path, std::string_view kernel_name) const
{
    // load ptx file
    std::ifstream ptx_file(path.c_str(), std::ios::binary);
    // copies all data into buffer
    std::vector<unsigned char> ptx(std::istreambuf_iterator<char>(ptx_file), {});
    ptx.push_back(0);  // has to be null terminated

    // Load the PTX.
    CUmodule module{};
    CHECK_CUDA_ERROR(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));

    //cuModuleEnumerateFunctions()

    // Locate the kernel entry point.
    CUfunction kernel{};
    CHECK_CUDA_ERROR(cuModuleGetFunction(&kernel, module, kernel_name.data()));

    return { module, kernel };
}

void cuda::CudaContext::launch_kernel(CUfunction kernel, std::array<std::uint32_t, 3> blocks, std::array<std::uint32_t, 3> threads_per_block, std::uint32_t slm_dynamic, std::vector<CUdeviceptr*> args) const
{
    CHECK_CUDA_ERROR(cuLaunchKernel(kernel, blocks[0], blocks[1], blocks[2], threads_per_block[0], threads_per_block[1], threads_per_block[2], slm_dynamic, stream_, (void**)args.data(), nullptr));
}

void cuda::CudaContext::synchronize()
{
    CHECK_CUDA_ERROR(cuStreamSynchronize(stream_));
}

CUdeviceptr cuda::CudaContext::create_managed_buffer(std::size_t size) const
{
    CUdeviceptr handle{};
    CHECK_CUDA_ERROR(cuMemAllocManaged(&handle, size, CU_MEM_ATTACH_GLOBAL));
    return handle;
}

#endif // #if BUILD_CUDA