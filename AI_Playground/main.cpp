#include "ioperator.h"

#include <iostream>
#include <sys/stat.h>
#include <format>
#include <vector>
#include <array>
#include <memory>


#include "quantized_gemm.h"

#include "dx12_context.h"
#include "cuda_context.h"

struct app_opts_t
{
    std::size_t execute_loop = 1;
};

int main()
{
    std::cout << "[AI_Playground] starting." << std::endl;
    app_opts_t opts{};

    std::unique_ptr<op::IOperator> op{};
    std::cout << "[AI_Playground] Creating quantized GEMM." << std::endl;
    op::QuantizedGemm::create_params_t cp{};
    cp.K = 512;
    cp.M = 512;
    cp.N = 512;
    cp.block_size = 32;
    op = std::make_unique<op::QuantizedGemm>(cp);

    std::vector<std::byte> result{};
#if BUILD_CUDA
    if (opts.run_cuda)
    {
        std::cout << "[AI_Playground] Executing CUDA." << std::endl;
        cuda::CudaContext cuda_ctx{};
        result = op->execute(&cuda_ctx, op::IOperator::execute_cuda_config_t{ opts.execute_loop });
    }
#endif  // #if BUILD_CUDA
    dx12::Dx12Context dx12_ctx{};
    if (result.empty())
    {
        std::cout << "[AI_Playground] Executing DML." << std::endl;
        result = op->execute(&dx12_ctx, op::IOperator::execute_dml_config_t{ opts.execute_loop, false });
    }

    std::cout << "[AI_Playground] Executing DML with MetaCommands disabled to capture reference data." << std::endl;
    std::vector<std::byte> result_reference = op->execute(&dx12_ctx, op::IOperator::execute_dml_config_t{ opts.execute_loop, true });
  
    std::cout << "[AI_Playground] Running conformance check." << std::endl;
    op->compare(result, result_reference);

    std::cout << "[AI_Playground] Finished." << std::endl;
    return 0;
}