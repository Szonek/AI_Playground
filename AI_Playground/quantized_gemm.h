#pragma once
#include "ioperator.h"

#include <array>

namespace op
{

class QuantizedGemm : public IOperator
{
public:
    struct create_params_t
    {
        std::uint32_t M = 16;
        std::uint32_t K = 32;
        std::uint32_t N = 16;
        std::uint32_t block_size = 16;

        bool b_transposed = true;
    };
public:
    QuantizedGemm(const create_params_t& params);

    std::vector<std::byte> execute(dx12::Dx12Context* dml_ctx, const execute_dml_config_t& config) override;
    std::vector<std::byte> execute(cuda::CudaContext* cu_ctx, const execute_cuda_config_t& config) override;

    bool compare(const std::vector<std::byte>& lhs, const std::vector<std::byte>& rhs) override;

private:
    enum RESOURCE_INDEX
    {
        RESOURCE_INDEX_A,
        RESOURCE_INDEX_B,
        //RESOURCE_INDEX_C,
        RESOURCE_INDEX_B_QUANTIZATION_SCALE,
        RESOURCE_INDEX_B_QUANTIZATION_ZERO_POINT,
        // end of input resources
        RESOURCE_INDEX_OUT,
        // ..
        RESOURCE_INDEX_COUNT
    };

private:
    std::array<std::vector<std::byte>, RESOURCE_INDEX_COUNT> data_host_;
    const create_params_t params_;
};
}