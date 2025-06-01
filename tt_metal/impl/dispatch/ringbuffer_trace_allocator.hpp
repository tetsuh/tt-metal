// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <api/tt-metalium/device.hpp>
#include <vector>
#include <optional>

#include "trace/trace_node.hpp"
#include "worker_config_buffer.hpp"

namespace tt::tt_metal {
class RingbufferTraceAllocator {
public:
    RingbufferTraceAllocator(
        uint32_t worker_ringbuffer_start,
        uint32_t worker_ringbuffer_size,
        uint32_t active_eth_ringbuffer_start,
        uint32_t active_eth_ringbuffer_size);

    void allocate_trace_programs(std::vector<TraceNode*>& trace_nodes);
  private:
    DispatchArray<WorkerConfigBufferMgr> config_buffer_mgr_;
};

}  // namespace tt::tt_metal
