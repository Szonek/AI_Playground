#include "dx12_context.h"

#include <iostream>
#include <format>

namespace
{


// If 'err' is non-zero, emit an error message and exit.
#define CHECK_D3D12_ERROR(err) __check_d3d12_errors(err, __FILE__, __LINE__)
static void __check_d3d12_errors(HRESULT err, const char* filename, int line)
{
    assert(filename);
    if (S_OK != err)
    {
        const char* ename = NULL;
        const auto out_str = std::format("D3D12 API ERROR: {}, from file: {}, line: {}", std::uint32_t(err), filename, line);
        std::cerr << out_str << std::endl;
        std::exit(err);
    }

}

void initialize_directD3D12(
    ComPtr<ID3D12Device>& d3D12Device,
    ComPtr<ID3D12CommandQueue>& commandQueue,
    ComPtr<ID3D12CommandAllocator>& commandAllocator,
    ComPtr<ID3D12GraphicsCommandList>& commandList)
{
#if defined(_DEBUG)
    ComPtr<ID3D12Debug> d3D12Debug;
    // Throws if the D3D12 debug layer is missing - you must install the Graphics Tools optional feature
    CHECK_D3D12_ERROR(D3D12GetDebugInterface(IID_PPV_ARGS(d3D12Debug.GetAddressOf())));
    d3D12Debug->EnableDebugLayer();
#endif


    ComPtr<IDXGIFactory4> dxgiFactory;
    CHECK_D3D12_ERROR(CreateDXGIFactory1(IID_PPV_ARGS(dxgiFactory.GetAddressOf())));

    ComPtr<IDXGIAdapter> dxgiAdapter;
    UINT adapterIndex{};
    HRESULT hr{};
    do
    {
        dxgiAdapter = nullptr;
        CHECK_D3D12_ERROR(dxgiFactory->EnumAdapters(adapterIndex, dxgiAdapter.ReleaseAndGetAddressOf()));
        ++adapterIndex;
        DXGI_ADAPTER_DESC desc{};
        dxgiAdapter->GetDesc(&desc);
        std::wcout << L"GPU: " << desc.Description << std::endl;

        hr = ::D3D12CreateDevice(
            dxgiAdapter.Get(),
            D3D_FEATURE_LEVEL_12_0,
            IID_PPV_ARGS(d3D12Device.ReleaseAndGetAddressOf()));
        if (hr == DXGI_ERROR_UNSUPPORTED) continue;
        CHECK_D3D12_ERROR(hr);
    } while (hr != S_OK);


    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    CHECK_D3D12_ERROR(d3D12Device->CreateCommandQueue(
        &commandQueueDesc,
        IID_PPV_ARGS(commandQueue.ReleaseAndGetAddressOf())));

    CHECK_D3D12_ERROR(d3D12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(commandAllocator.ReleaseAndGetAddressOf())));

    CHECK_D3D12_ERROR(d3D12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        commandAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(commandList.ReleaseAndGetAddressOf())));
}

void close_execute_reset_wait(
    ComPtr<ID3D12Device> d3D12Device,
    ComPtr<ID3D12CommandQueue> commandQueue,
    ComPtr<ID3D12CommandAllocator> commandAllocator,
    ComPtr<ID3D12GraphicsCommandList> commandList)
{
    CHECK_D3D12_ERROR(commandList->Close());

    ID3D12CommandList* commandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

    ComPtr<ID3D12Fence> d3D12Fence;
    CHECK_D3D12_ERROR(d3D12Device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(d3D12Fence.GetAddressOf())));

    auto fenceEventHandle = ::CreateEvent(nullptr, true, false, nullptr);

    CHECK_D3D12_ERROR(commandQueue->Signal(d3D12Fence.Get(), 1));
    CHECK_D3D12_ERROR(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle));

    ::WaitForSingleObjectEx(fenceEventHandle, INFINITE, FALSE);

    CHECK_D3D12_ERROR(commandAllocator->Reset());
    CHECK_D3D12_ERROR(commandList->Reset(commandAllocator.Get(), nullptr));
}
}

dx12::Dx12Context::Dx12Context()
{
    initialize_directD3D12(d3d12_device_, command_queue_, command_allocator_, command_list_);

    CHECK_D3D12_ERROR(DMLCreateDevice1(d3d12_device_.Get(), DML_CREATE_DEVICE_FLAG_DEBUG, DML_FEATURE_LEVEL_6_4, IID_PPV_ARGS(dml_device_.ReleaseAndGetAddressOf())));
    CHECK_D3D12_ERROR(dml_device_->CreateCommandRecorder(IID_PPV_ARGS(dml_cmd_recorder_.GetAddressOf())));
}

ComPtr<ID3D12DescriptorHeap> dx12::Dx12Context::create_heap(std::uint32_t descriptor_count) const
{
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap_desc{};
    descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptor_heap_desc.NumDescriptors = descriptor_count;
    descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK_D3D12_ERROR(d3d12_device_->CreateDescriptorHeap(
        &descriptor_heap_desc,
        IID_PPV_ARGS(descriptor_heap.GetAddressOf())));
    return descriptor_heap;
}

ComPtr<ID3D12Resource> dx12::Dx12Context::create_buffer(std::size_t size) const
{
    ComPtr<ID3D12Resource> ret{};
    const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    const auto desc = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    CHECK_D3D12_ERROR(d3d12_device_->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(ret.GetAddressOf())));

    return ret;
}

ComPtr<ID3D12Resource> dx12::Dx12Context::create_upload_buffer(std::size_t size) const
{
    ComPtr<ID3D12Resource> ret{};
    const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    const auto desc = CD3DX12_RESOURCE_DESC::Buffer(size,  D3D12_RESOURCE_FLAG_NONE);
    CHECK_D3D12_ERROR(d3d12_device_->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(ret.GetAddressOf())));
    return ret;
}

ComPtr<ID3D12Resource> dx12::Dx12Context::create_readback_buffer(std::size_t size) const
{
    ComPtr<ID3D12Resource> ret{};
    const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    const auto desc = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_NONE);
    CHECK_D3D12_ERROR(d3d12_device_->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(ret.GetAddressOf())));
    return ret;
}

void dx12::Dx12Context::set_heap(ID3D12DescriptorHeap* heap)
{
    command_list_->SetDescriptorHeaps(1, &heap);
}

void dx12::Dx12Context::synchronize()
{
    close_execute_reset_wait(d3d12_device_, command_queue_, command_allocator_, command_list_);
}


dml::Graph dx12::Dx12Context::create_graph() const
{
    return dml::Graph(dml_device_.Get());
}

ComPtr<IDMLOperatorInitializer> dx12::Dx12Context::create_initalizer(IDMLCompiledOperator* op) const
{
    IDMLCompiledOperator* dml_compiled_operators[] = { op };
    ComPtr<IDMLOperatorInitializer> dml_operator_initializer;
    CHECK_D3D12_ERROR(dml_device_->CreateOperatorInitializer(
        ARRAYSIZE(dml_compiled_operators),
        dml_compiled_operators,
        IID_PPV_ARGS(dml_operator_initializer.GetAddressOf())));
    return dml_operator_initializer;
}

ComPtr<IDMLBindingTable> dx12::Dx12Context::create_binding_table(const DML_BINDING_TABLE_DESC& desc) const
{
    ComPtr<IDMLBindingTable> dml_binding_table;
    CHECK_D3D12_ERROR(dml_device_->CreateBindingTable(
        &desc,
        IID_PPV_ARGS(dml_binding_table.GetAddressOf())));
    return dml_binding_table;
}

void dx12::Dx12Context::resource_copy(ID3D12Resource* dst, ID3D12Resource* src)
{
    command_list_->CopyResource(dst, src);
}

void dx12::Dx12Context::resource_state_transition(ID3D12Resource* rsc, D3D12_RESOURCE_STATES src, D3D12_RESOURCE_STATES dst)
{
    const auto barrier =CD3DX12_RESOURCE_BARRIER::Transition(rsc, src, dst);
    command_list_->ResourceBarrier(1, &barrier);
}

void dx12::Dx12Context::copy_buffer_region(std::size_t size, ID3D12Resource* dst, std::size_t dst_offset, ID3D12Resource* src, std::size_t src_offset)
{
    command_list_->CopyBufferRegion(dst, dst_offset, src, src_offset, size);
}

void dx12::Dx12Context::record_dispatch(IDMLDispatchable* dispatchable, IDMLBindingTable* binding_table)
{
    dml_cmd_recorder_->RecordDispatch(command_list_.Get(), dispatchable, binding_table);
}
