#pragma once

#include <span>
#include <optional>
#include <cstdint>
#include <cstdlib>


#include <dxgi1_6.h>
#include <wrl/client.h>
#define DML_TARGET_VERSION_USE_LATEST 1
#include <DirectMLX.h>
#include <d3dx12/d3dx12.h>

using Microsoft::WRL::ComPtr;

namespace dx12
{
class Dx12Context
{
public:
    Dx12Context();

    ID3D12Device* get_device() { return d3d12_device_.Get(); }
    ID3D12GraphicsCommandList* get_cmd_list() { return command_list_.Get(); }

    ComPtr<ID3D12DescriptorHeap> create_heap(std::uint32_t descriptors_count) const;
    ComPtr<ID3D12Resource> create_buffer(std::size_t size) const;
    ComPtr<ID3D12Resource> create_upload_buffer(std::size_t size) const;
    ComPtr<ID3D12Resource> create_readback_buffer(std::size_t size) const;

    void resource_copy(ID3D12Resource* dst, ID3D12Resource* src);
    void resource_state_transition(ID3D12Resource* rsc, D3D12_RESOURCE_STATES src, D3D12_RESOURCE_STATES dst);
    void copy_buffer_region(std::size_t size, ID3D12Resource* dst, std::size_t dst_offset, ID3D12Resource* src, std::size_t src_offset);
    void record_dispatch(IDMLDispatchable* dispatchable, IDMLBindingTable* binding_table);
    void set_heap(ID3D12DescriptorHeap* heap);
    void synchronize();

    dml::Graph create_graph() const;
    ComPtr<IDMLOperatorInitializer> create_initalizer(IDMLCompiledOperator* op) const;
    ComPtr<IDMLBindingTable> create_binding_table(const DML_BINDING_TABLE_DESC& desc) const;

private:
    ComPtr<ID3D12Device> d3d12_device_;
    ComPtr<ID3D12CommandQueue> command_queue_;
    ComPtr<ID3D12CommandAllocator> command_allocator_;
    ComPtr<ID3D12GraphicsCommandList> command_list_;

    // dml
    ComPtr<IDMLDevice> dml_device_;
    ComPtr<IDMLCommandRecorder> dml_cmd_recorder_;
};
}