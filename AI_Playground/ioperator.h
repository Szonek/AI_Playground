#pragma once
#include <cstdint>
#include <vector>

namespace dx12
{
class Dx12Context;
}

namespace cuda
{
class CudaContext;
}

namespace op
{
class IOperator
{
public:
    struct execute_dml_config_t
    {
        std::size_t iters = 1;
        bool disable_metacommands = false;
    };

    struct execute_cuda_config_t
    {
        std::size_t iters = 1;
    };

public:
    virtual std::vector<std::byte> execute(dx12::Dx12Context* dx_ctx, const execute_dml_config_t& config) = 0;
    virtual std::vector<std::byte> execute(cuda::CudaContext* cu_ctx, const execute_cuda_config_t& config) = 0;

    virtual bool compare(const std::vector<std::byte>& lhs, const std::vector<std::byte>& rhs) = 0;
};

}
