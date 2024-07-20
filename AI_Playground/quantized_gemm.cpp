#include "quantized_gemm.h"
#include "dx12_context.h"
#include "cuda_context.h"

#include "DirectXMath.h"
#include "DirectXPackedVector.h"
using float16 = DirectX::PackedVector::HALF;

#include <iostream>

namespace
{
inline void fill_float16(std::span<std::byte> vec, float value)
{
    auto* f16 = reinterpret_cast<float16*>(vec.data());
    for (auto i = 0; i < vec.size() / sizeof(float16); i++)
    {
        f16[i] = DirectX::PackedVector::XMConvertFloatToHalf(value);
    }
}

inline void fill_uint4(std::span<std::byte> vec, std::uint8_t value)
{
    assert(value < (2 ^ 4));
    // https://stackoverflow.com/questions/44886203/interpret-int8-as-two-int4
    auto* u8 = reinterpret_cast<std::uint8_t*>(vec.data());

    for (auto i = 0; i < vec.size(); i++)
    {
        u8[i] = std::uint8_t(value);
        u8[i] |= (std::uint8_t(value) << 4);   // Move the high bits to the low bits
    }
}

}

op::QuantizedGemm::QuantizedGemm(const create_params_t& params)
    : params_(params)
{
    assert((params_.K / params_.block_size) != 0);
    assert(params_.b_transposed == true); // not supporting non-b tranposed yet

    const float dt_size = sizeof(float16);
    const float uint4_size = 0.5f;
    // A
    data_host_[RESOURCE_INDEX_A].resize(params_.M * params_.K * dt_size);
    fill_float16(data_host_[RESOURCE_INDEX_A], 1.0f);
    // B
    data_host_[RESOURCE_INDEX_B].resize(params_.K * params_.N * uint4_size);
    fill_uint4(data_host_[RESOURCE_INDEX_B], 1);
    // OUT
    data_host_[RESOURCE_INDEX_OUT].resize(params_.M * params_.N * dt_size);
    
    // B quantization params
    // B scales
    data_host_[RESOURCE_INDEX_B_QUANTIZATION_SCALE].resize(params_.N * (params_.K / params_.block_size) * dt_size);
    fill_float16(data_host_[RESOURCE_INDEX_B_QUANTIZATION_SCALE], 1.0f);
    // B zero points
    data_host_[RESOURCE_INDEX_B_QUANTIZATION_ZERO_POINT].resize(params_.N * (params_.K / params_.block_size) * uint4_size);
    fill_uint4(data_host_[RESOURCE_INDEX_B_QUANTIZATION_ZERO_POINT], 0);
}

std::vector<std::byte> op::QuantizedGemm::execute(dx12::Dx12Context* dx_ctx, const execute_dml_config_t& config)
{
    dml::Graph dml_graph = dx_ctx->create_graph();
    std::vector<dml::Expression> tensor_b_quantization_params(2);
    tensor_b_quantization_params[0] = dml::InputTensor(dml_graph, RESOURCE_INDEX_B_QUANTIZATION_SCALE, dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, { 1, 1, params_.N, params_.K / params_.block_size })); // transposed!!
    tensor_b_quantization_params[1] = dml::InputTensor(dml_graph, RESOURCE_INDEX_B_QUANTIZATION_ZERO_POINT, dml::TensorDesc(DML_TENSOR_DATA_TYPE_UINT4, DML_TENSOR_FLAG_NONE, { 1, 1, params_.N, params_.K / params_.block_size })); // transposed!!
    const auto tensor_a = dml::InputTensor(dml_graph, RESOURCE_INDEX_A, dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, { 1, 1, params_.M, params_.K }));
    const auto tensor_b = dml::InputTensor(dml_graph, RESOURCE_INDEX_B, dml::TensorDesc(DML_TENSOR_DATA_TYPE_UINT4, DML_TENSOR_FLAG_NONE, { 1, 1, params_.N, params_.K }));  // transposed!!
    const auto dequant_input_b = dml::Dequantize(tensor_b, tensor_b_quantization_params, DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT);
    std::vector<dml::Expression> outs(1);
    outs[0] = dml::GemmBuilder(tensor_a, dequant_input_b/*, tensor_c*/).Alpha(1.0f).Beta(1.0f).TransB(DML_MATRIX_TRANSFORM_TRANSPOSE).Build();
    
    auto exec_flags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    if (config.disable_metacommands)
    {
        exec_flags |= DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
    }
    // count inputs
    std::uint32_t inputs = 0;
    for (auto i = 0; i < RESOURCE_INDEX_OUT; i++)
    {
        if (!data_host_.empty())
        {
            inputs++;
        }
    }
    const auto compiled_op = dml_graph.Compile(exec_flags, outs, inputs);

    const auto dml_operator_initializer = dx_ctx->create_initalizer(compiled_op.Get());

    const auto initialize_binding_properties = dml_operator_initializer->GetBindingProperties();
    const auto execute_binding_properties = compiled_op->GetBindingProperties();
    const auto descriptor_count = max(
        initialize_binding_properties.RequiredDescriptorCount,
        execute_binding_properties.RequiredDescriptorCount);

    // Create descriptor heaps.
    const auto descriptor_heap = dx_ctx->create_heap(descriptor_count);
    dx_ctx->set_heap(descriptor_heap.Get());


    DML_BINDING_TABLE_DESC dml_binding_table_desc{};
    dml_binding_table_desc.Dispatchable = dml_operator_initializer.Get();
    dml_binding_table_desc.CPUDescriptorHandle = descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    dml_binding_table_desc.GPUDescriptorHandle = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    dml_binding_table_desc.SizeInDescriptors = descriptor_count;
    const auto dml_binding_table = dx_ctx->create_binding_table(dml_binding_table_desc);

    const auto temporary_resource_size = max(
        initialize_binding_properties.TemporaryResourceSize,
        execute_binding_properties.TemporaryResourceSize);
    const auto persistent_resource_size = execute_binding_properties.PersistentResourceSize;

    ComPtr<ID3D12Resource> temporary_buffer{};
    if (temporary_resource_size != 0)
    {
        temporary_buffer = dx_ctx->create_buffer(temporary_resource_size);

        if (initialize_binding_properties.TemporaryResourceSize != 0)
        {
            DML_BUFFER_BINDING buffer_binding{ temporary_buffer.Get(), 0, temporary_resource_size };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_binding_table->BindTemporaryResource(&binding_desc);
        }
    }

    ComPtr<ID3D12Resource> persistent_buffer{};
    if (persistent_resource_size != 0)
    {
        temporary_buffer = dx_ctx->create_buffer(persistent_resource_size);

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING buffer_binding{ persistent_buffer.Get(), 0, persistent_resource_size };
        DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
        dml_binding_table->BindOutputs(1, &binding_desc);
    }

    dx_ctx->record_dispatch(dml_operator_initializer.Get(), dml_binding_table.Get());
    dx_ctx->synchronize();

    // execute
    dx_ctx->set_heap(descriptor_heap.Get());

    dml_binding_table_desc.Dispatchable = compiled_op.Get();
    dml_binding_table_desc.CPUDescriptorHandle = descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    dml_binding_table_desc.GPUDescriptorHandle = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    dml_binding_table_desc.SizeInDescriptors = descriptor_count;

    dml_binding_table->Reset(&dml_binding_table_desc);

    if (temporary_resource_size != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ temporary_buffer.Get(), 0, temporary_resource_size };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dml_binding_table->BindTemporaryResource(&bindingDesc);
    }

    if (persistent_resource_size != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ persistent_buffer.Get(), 0, persistent_resource_size };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dml_binding_table->BindPersistentResource(&bindingDesc);
    }


    auto upload_buffer = dx_ctx->create_upload_buffer([this]() {
        std::size_t total_tensors_size = 0;
        for (const auto& dh : data_host_)
        {
            total_tensors_size += dh.size();
        }
        return total_tensors_size;
        }());

    std::byte* upload_ptr = nullptr;
    upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&upload_ptr));
    for (const auto& dh : data_host_)
    {
        if (dh.empty())
        {
            continue;
        }
        std::memcpy(upload_ptr, dh.data(), dh.size());
        upload_ptr += dh.size();
    }
    upload_buffer->Unmap(0, nullptr);
    dx_ctx->synchronize();

    std::array<ComPtr<ID3D12Resource>, RESOURCE_INDEX_COUNT> gpu_resources;
    std::size_t upload_heap_offset_counter = 0;
    for (auto i = 0; i < gpu_resources.size(); i++)
    {
        const auto& dh = data_host_[i];
        if (dh.empty())
        {
            continue;
        }
        gpu_resources[i] = dx_ctx->create_buffer(dh.size());
        dx_ctx->copy_buffer_region(dh.size(), gpu_resources[i].Get(), 0, upload_buffer.Get(), upload_heap_offset_counter);
        upload_heap_offset_counter += dh.size();
    }
    dx_ctx->synchronize();
    std::array<DML_BUFFER_BINDING, RESOURCE_INDEX_COUNT> bindings_buffer;
    for (auto i = 0; i < RESOURCE_INDEX_COUNT; i++)
    {
        const auto& dh = data_host_[i];
        if (dh.empty())
        {
            continue;
        }
        bindings_buffer[i] = DML_BUFFER_BINDING{ gpu_resources[i].Get(), 0, dh.size() };
    }
    std::vector<DML_BINDING_DESC> input_binding_desc_list{};
    for (int i = 0; i < RESOURCE_INDEX_OUT; i++)
    {
        const auto& dh = data_host_[i];
        if (dh.empty())
        {
            continue;
        }
        input_binding_desc_list.push_back({ DML_BINDING_TYPE_BUFFER, &bindings_buffer[i] });
    }
    dml_binding_table->BindInputs(input_binding_desc_list.size(), input_binding_desc_list.data());

    DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &bindings_buffer[RESOURCE_INDEX_OUT]};
    dml_binding_table->BindOutputs(1, &output_binding_desc);

    dx_ctx->set_heap(descriptor_heap.Get());
    dx_ctx->record_dispatch(compiled_op.Get(), dml_binding_table.Get());
    dx_ctx->synchronize();
    // readback result

    ComPtr<ID3D12Resource> readback_buffer = dx_ctx->create_readback_buffer(data_host_[RESOURCE_INDEX_OUT].size());
    dx_ctx->resource_state_transition(gpu_resources[RESOURCE_INDEX_OUT].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dx_ctx->resource_copy(readback_buffer.Get(), gpu_resources[RESOURCE_INDEX_OUT].Get());
    dx_ctx->synchronize();

    std::byte* ret_ptr = nullptr;
    readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&ret_ptr));
    std::vector<std::byte> ret(data_host_[RESOURCE_INDEX_OUT].size());
    std::memcpy(ret.data(), ret_ptr, ret.size());
    readback_buffer->Unmap(0, nullptr);

    //const auto* ret_f16 = reinterpret_cast<const float16*>(ret.data());
    //std::vector<float> ret_f32(ret.size() * 2);
    //DirectX::PackedVector::XMConvertHalfToFloatStream(ret_f32.data(), sizeof(float), ret_f16, sizeof(float16), ret.size() / 2);
    return ret;
}

std::vector<std::byte> op::QuantizedGemm::execute(cuda::CudaContext* cu_ctx, const execute_cuda_config_t& config)
{
#if BUILD_CUDA
    cu_ctx->create_kernel(std::filesystem::path("C:\\WORK\\AI_Playground\\AI_Playground\\kernels\\vec_add.ptx"), "_Z7vec_addPfS_S_");
#endif // #if BUILD_CUDA
    return std::vector<std::byte>();
}

bool op::QuantizedGemm::compare(const std::vector<std::byte>& lhs, const std::vector<std::byte>& rhs)
{
    assert(lhs.size() == rhs.size());
    const float16* data_f16 = reinterpret_cast<const float16*>(lhs.data());
    const float16* data_f16_ref = reinterpret_cast<const float16*>(rhs.data());
    for (int i = 0; i < lhs.size() / sizeof(float16); i++)
    {
        const auto data = DirectX::PackedVector::XMConvertHalfToFloat(data_f16[i]);
        const auto ref = DirectX::PackedVector::XMConvertHalfToFloat(data_f16_ref[i]);
        if (data != ref)
        {
            const auto str = std::format("Conformance failed. Data: {}, ref: {}, index: {}\n", data, ref, i);
            std::cout << str << std::endl;
            return false;
        }
    }

    std::cout << "Conformance passed" << std::endl;
    return true;
}
