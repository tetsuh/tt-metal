// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ringbuffer_trace_allocator.hpp"

#include "impl/context/metal_context.hpp"
#include "program/dispatch.hpp"

namespace tt::tt_metal {

RingbufferTraceAllocator::RingbufferTraceAllocator(
    uint32_t worker_ringbuffer_start,
    uint32_t worker_ringbuffer_size,
    uint32_t active_eth_ringbuffer_start,
    uint32_t active_eth_ringbuffer_size) {
    fmt::println(stderr, "Worker ringbuffer start: {}, size: {}, active eth ringbuffer start: {}, size: {}",
                    worker_ringbuffer_start, worker_ringbuffer_size,
                    active_eth_ringbuffer_start, active_eth_ringbuffer_size);
    for (auto& mgr : config_buffer_mgr_) {
        mgr.init_add_buffer(worker_ringbuffer_start, worker_ringbuffer_size);
        mgr.init_add_buffer(active_eth_ringbuffer_start, active_eth_ringbuffer_size);
        // Idle ethernet.
        mgr.init_add_buffer(0, 0);
        // Worker launch messages.
        mgr.init_add_buffer(0, launch_msg_buffer_num_entries - 1);
        // Ethernet launch messages.
        mgr.init_add_buffer(0, 1);
    }
}

void RingbufferTraceAllocator::allocate_trace_programs(std::vector<TraceNode*>& trace_nodes) {
    const auto& hal = MetalContext::instance().hal();
    DispatchArray<uint32_t> expected_workers_completed{};
    for (auto& node_ptr : trace_nodes) {
        auto& node = *node_ptr;
        auto& program = *node.program;
        auto sub_device_id = node.sub_device_id;
        auto sub_device_index = *sub_device_id;
        auto& sub_device_expected_workers_completed = expected_workers_completed[sub_device_index];
        program_dispatch::ProgramDispatchMetadata dispatch_metadata;
        // Reserve space for this program in the kernel config ring buffer
        program_dispatch::reserve_space_in_kernel_config_buffer(
            this->config_buffer_mgr_[sub_device_index],
            program.get_program_config_sizes(),
            ProgramBinaryStatus::Committed,
            node.num_workers,
            sub_device_expected_workers_completed,
            dispatch_metadata);
        uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        ProgramConfig& program_config = program.get_program_config(index);

        node.dispatch_metadata.binary_kernel_config_addrs = dispatch_metadata.kernel_config_addrs;
        node.dispatch_metadata.nonbinary_kernel_config_addrs = dispatch_metadata.kernel_config_addrs;
        node.dispatch_metadata.sync_count = dispatch_metadata.sync_count;
        node.dispatch_metadata.stall_first = dispatch_metadata.stall_first;
        node.dispatch_metadata.stall_before_program = dispatch_metadata.stall_before_program;
        if (sub_device_expected_workers_completed == 0) {
            // The first program to be dispatched should stall on 0, since there may be undetermined commands in the
            // ringbuffer before this we want to wait for. In particular in the mesh device case we can add go messages
            // for unused nodes before replaying the trace.
            node.dispatch_metadata.sync_count = 0;
            node.dispatch_metadata.stall_first = true;
        }

        // Allocate non-binaries before binaries for tensix. Non-tensix doesn't use a ringbuffer for binaries, so its
        // addresses don't need adjustment.
        node.dispatch_metadata.binary_kernel_config_addrs[index].addr += program_config.kernel_text_offset;
        sub_device_expected_workers_completed += node.num_workers;
    }
}

}  // namespace tt::tt_metal
