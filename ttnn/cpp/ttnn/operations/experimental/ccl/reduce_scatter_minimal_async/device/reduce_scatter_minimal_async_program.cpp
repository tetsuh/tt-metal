// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async(
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};
    return reduce_scatter_minimal_async_helper(
        program,
        input_tensor,
        intermediate_tensor,
        sender_device,
        forward_device,
        backward_device,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        sub_device_id);
}

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const CoreCoord core_grid_offset) {
    auto mesh_device = input_tensor.mesh_device();
    const bool enable_async_output_tensor = false;
    const bool enable_persistent_fabric_mode = true;
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    bool fuse_op = false;

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {intermediate_tensor, output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    TT_ASSERT(!((topology == ccl::Topology::Linear) && fuse_op));
    if (topology == ccl::Topology::Ring && ring_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    // Get worker cores
    // 1 sender (reader + writer), and 2 receivers (forward/backward, each with a reader/writer)
    uint32_t num_senders_per_link = 1;
    uint32_t num_receivers_per_link = 2;
    const auto [sender_worker_core_range, sender_worker_cores] = choose_worker_cores(
        num_links, num_senders_per_link, enable_persistent_fabric_mode, mesh_device, sub_device_id, core_grid_offset);
    const auto [total_worker_core_range, total_worker_cores] = choose_worker_cores(
        num_links,
        (num_senders_per_link + num_receivers_per_link),
        enable_persistent_fabric_mode,
        mesh_device,
        sub_device_id,
        core_grid_offset);
    const auto receiver_worker_core_range = total_worker_core_range.subtract(sender_worker_core_range);
    const auto receiver_worker_cores = corerange_to_cores(receiver_worker_core_range, std::nullopt, true);
    std::set<CoreRange> receiver_forward_core_ranges;
    receiver_forward_core_ranges.insert(CoreRange(receiver_worker_cores[1]));
    CoreRangeSet receiver_forward_core_range_set = CoreRangeSet(receiver_forward_core_ranges);
    std::set<CoreRange> receiver_backward_core_ranges;
    receiver_backward_core_ranges.insert(CoreRange(receiver_worker_cores[0]));
    CoreRangeSet receiver_backward_core_range_set = CoreRangeSet(receiver_backward_core_ranges);
    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_1d_fabric_config().channel_buffer_size_bytes;
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_forward_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_sender_forward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_forward_cb_index, df}})
            .set_page_size(sender_forward_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_sender_forward_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_sender_forward_config);
    uint32_t sender_backward_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_sender_backward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_backward_cb_index, df}})
            .set_page_size(sender_backward_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_sender_backward_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_sender_backward_config);

    // CBs for transferring data between receiver_reader and receiver_writer
    uint32_t receiver_forward_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_receiver_forward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{receiver_forward_cb_index, df}})
            .set_page_size(receiver_forward_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_receiver_forward_workers =
        CreateCircularBuffer(program, receiver_forward_core_ranges, cb_receiver_forward_config);
    uint32_t receiver_backward_cb_index = tt::CB::c_in4;
    tt::tt_metal::CircularBufferConfig cb_receiver_backward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{receiver_backward_cb_index, df}})
            .set_page_size(receiver_backward_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_receiver_backward_workers =
        CreateCircularBuffer(program, receiver_backward_core_ranges, cb_receiver_backward_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);
    // Tensor Info
    const auto input_tensor_layout = input_tensor.buffer()->buffer_layout();
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto input_tensor_page_layout = input_tensor.layout();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto output_tensor_layout = output_tensor.buffer()->buffer_layout();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto output_tensor_page_layout = output_tensor.layout();
    const auto input_tensor_shape = input_tensor.get_padded_shape();
    const auto output_tensor_shape = output_tensor.get_padded_shape();
    const auto intermediate_tensor_buffer_type = intermediate_tensor.buffer()->buffer_type();

    // KERNEL CREATION
    // Reader
    auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_reader_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(input_tensor_buffer_type),         // input_buffer_type
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        sender_forward_cb_index,                                 // cb_forward_id
        sender_backward_cb_index,                                // cb_backward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // tensor0_page_size
        num_targets_forward,                                     // num_slices_forward_direction
        num_targets_backward,                                    // num_slices_backward_direction
        static_cast<uint32_t>(topology)                          // topology
    };
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
        "reduce_scatter_minimal_async_reader.cpp",
        sender_worker_core_range,
        sender_reader_kernel_config);

    // Writer
    auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_writer_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        reserved_packet_header_CB_index,                         // reserved_packet_header_cb_id
        num_packet_headers_storable,                             // num_packet_headers_storable
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
        sender_forward_cb_index,                                 // cb_forward_id
        sender_backward_cb_index,                                // cb_backward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // tensor0_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        dynamic_alternate,                                       // alternate
        fuse_op,                                                 // fused op
        static_cast<uint32_t>(topology)                          // topology
    };
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
        "reduce_scatter_minimal_async_writer.cpp",
        sender_worker_core_range,
        sender_writer_kernel_config);

    // Forward Receiver Kernels
    // Writer
    auto forward_receiver_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    forward_receiver_writer_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        receiver_forward_cb_index,                               // cb_forward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // output_tensor_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        1,                                                       // direction
        fuse_op,                                                 // fused op
    };
    auto worker_forward_receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
        "reduce_scatter_minimal_async_receiver_writer.cpp",
        receiver_forward_core_range_set,
        forward_receiver_writer_kernel_config);
    // Reader
    auto forward_receiver_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    forward_receiver_reader_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        receiver_forward_cb_index,                               // cb_forward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // output_tensor_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        1                                                        // direction
    };
    auto worker_forward_receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
        "reduce_scatter_minimal_async_receiver_reader.cpp",
        receiver_forward_core_range_set,
        forward_receiver_reader_kernel_config);

    // Backward Receiver Kernels
    // Writer
    auto backward_receiver_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    backward_receiver_writer_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        receiver_backward_cb_index,                              // cb_backward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // output_tensor_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        0,                                                       // direction
        fuse_op,                                                 // fused op
    };
    auto worker_backward_receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
        "reduce_scatter_minimal_async_receiver_writer.cpp",
        receiver_backward_core_range_set,
        backward_receiver_writer_kernel_config);
    // Reader
    auto backward_receiver_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    backward_receiver_reader_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        receiver_backward_cb_index,                              // cb_forward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // output_tensor_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        0                                                        // direction
    };
    auto worker_backward_receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
        "reduce_scatter_minimal_async_receiver_reader.cpp",
        receiver_backward_core_range_set,
        backward_receiver_reader_kernel_config);

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        // Set Sender Reader runtime args
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);

        TT_ASSERT(!(input_tensor_shape[3] % TILE_WIDTH));
        TT_ASSERT(!(output_tensor_shape[3] % TILE_WIDTH));
        uint32_t TILE_WIDTH = 32;
        uint32_t input_tensor_Wt = input_tensor_shape[3] / TILE_WIDTH;
        uint32_t output_tensor_Wt = output_tensor_shape[3] / TILE_WIDTH;

        std::set<CoreRange> receiver_forward_semaphore_core_ranges;
        receiver_forward_semaphore_core_ranges.insert(CoreRange(receiver_worker_cores[1]));
        receiver_forward_semaphore_core_ranges.insert(CoreRange(sender_worker_cores[0]));
        CoreRangeSet receiver_forward_semaphore_core_range_set = CoreRangeSet(receiver_forward_semaphore_core_ranges);
        auto sender_to_forward_receiver_semaphore_id =
            CreateSemaphore(program, receiver_forward_semaphore_core_range_set, 0);
        std::set<CoreRange> receiver_backward_semaphore_core_ranges;
        receiver_backward_semaphore_core_ranges.insert(CoreRange(receiver_worker_cores[0]));
        receiver_backward_semaphore_core_ranges.insert(CoreRange(sender_worker_cores[0]));
        CoreRangeSet receiver_backward_semaphore_core_range_set = CoreRangeSet(receiver_backward_semaphore_core_ranges);
        auto sender_to_backward_receiver_semaphore_id =
            CreateSemaphore(program, receiver_backward_semaphore_core_range_set, 0);
        std::vector<CoreCoord> receiver_worker_cores_noc;
        for (const auto& core : receiver_worker_cores) {
            receiver_worker_cores_noc.push_back(mesh_device->worker_core_from_logical_core(core));
        }

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),          // input_tensor_address
            intermediate_tensor.buffer()->address(),   // output_tensor_address
            input_tensor_Wt,                           // width in tiles of the output shard
            output_tensor_Wt,                          // width in tiles of entire output
            input_tile_id_start,                       // input_tile_id_start
            input_tile_id_end,                         // slice_num_pages
            ring_size,                                 // ring_size
            semaphore.at(0).address(),                 // out_ready_semaphore_forward
            semaphore.at(1).address(),                 // out_ready_semaphore_backward
            sender_to_forward_receiver_semaphore_id,   // signal_receiver_sem_forward
            sender_to_backward_receiver_semaphore_id,  // signal_receiver_sem_forward
            receiver_worker_cores_noc.at(1).x,         // forward receiver core x
            receiver_worker_cores_noc.at(1).y,         // forward receiver core y
            receiver_worker_cores_noc.at(0).x,         // backward receiver core x
            receiver_worker_cores_noc.at(0).y,         // backward receiver core y
        };
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set Sender Writer runtime args
        uint32_t out_ready_sem_wait_value = (dynamic_alternate ? (ring_size + 1) : ring_size) * num_links;

        std::vector<uint32_t> writer_rt_args = {
            intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
            output_tensor.buffer()->address(),        // output_tensor_address
            input_tensor_Wt,                          // width in tiles of the output shard
            output_tensor_Wt,                         // width in tiles of entire output
            input_tensor_num_pages,                   // slice_num_pages
            drain_sync_core.x,                        // out_ready_sem_noc0_x
            drain_sync_core.y,                        // out_ready_sem_noc0_y
            ring_size,                                // ring_size
            semaphore.at(0).address(),                // out_ready_semaphore_forward
            semaphore.at(1).address()                 // out_ready_semaphore_backward
        };
        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_device->id(), forward_device.value()->id(), link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_device->id(), backward_device.value()->id(), link, program, {core}, writer_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

        // Set Receiver runtime args
        // Reader
        std::vector<uint32_t> forward_receiver_reader_rt_args = {
            intermediate_tensor.buffer()->address(),  // input_tensor_address
            input_tensor_Wt,                          // width in tiles of the output shard
            output_tensor_Wt,                         // width in tiles of entire output
            input_tile_id_end,                        // slice_num_pages
            ring_size,                                // ring_size
            sender_to_forward_receiver_semaphore_id,  // signal_receiver_sem_forward
        };
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_forward_receiver_reader_kernel_id,
            {receiver_worker_cores[1]},
            forward_receiver_reader_rt_args);
        std::vector<uint32_t> backward_receiver_reader_rt_args = {
            intermediate_tensor.buffer()->address(),   // input_tensor_address
            input_tensor_Wt,                           // width in tiles of the output shard
            output_tensor_Wt,                          // width in tiles of entire output
            input_tile_id_end,                         // slice_num_pages
            ring_size,                                 // ring_size
            sender_to_backward_receiver_semaphore_id,  // signal_receiver_sem_backward
        };
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_backward_receiver_reader_kernel_id,
            {receiver_worker_cores[0]},
            backward_receiver_reader_rt_args);

        // Writer
        std::vector<uint32_t> forward_receiver_writer_rt_args = {
            output_tensor.buffer()->address(),  // output_tensor_address
            input_tensor_Wt,                    // width in tiles of the output shard
            output_tensor_Wt,                   // width in tiles of entire output
            input_tile_id_end,                  // slice_num_pages
            ring_size,                          // ring_size
        };
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_forward_receiver_writer_kernel_id,
            {receiver_worker_cores[1]},
            forward_receiver_writer_rt_args);
        std::vector<uint32_t> backward_receiver_writer_rt_args = {
            output_tensor.buffer()->address(),  // output_tensor_address
            input_tensor_Wt,                    // width in tiles of the output shard
            output_tensor_Wt,                   // width in tiles of entire output
            input_tile_id_end,                  // slice_num_pages
            ring_size,                          // ring_size
        };
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_backward_receiver_writer_kernel_id,
            {receiver_worker_cores[0]},
            backward_receiver_writer_rt_args);
    }

    auto override_runtime_arguments_callback = [worker_sender_reader_kernel_id,
                                                worker_sender_writer_kernel_id,
                                                worker_forward_receiver_reader_kernel_id,
                                                worker_forward_receiver_writer_kernel_id,
                                                worker_backward_receiver_reader_kernel_id,
                                                worker_backward_receiver_writer_kernel_id,
                                                sender_worker_cores,
                                                receiver_worker_cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&
                                                       optional_input_tensors,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[1];
        const auto& intermed = output_tensors[0];

        // update senders
        auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
        // update receivers
        auto& worker_forward_receiver_reader_runtime_args_by_core =
            GetRuntimeArgs(program, worker_forward_receiver_reader_kernel_id);
        auto& worker_forward_receiver_writer_runtime_args_by_core =
            GetRuntimeArgs(program, worker_forward_receiver_writer_kernel_id);
        auto& worker_backward_receiver_reader_runtime_args_by_core =
            GetRuntimeArgs(program, worker_backward_receiver_reader_kernel_id);
        auto& worker_backward_receiver_writer_runtime_args_by_core =
            GetRuntimeArgs(program, worker_backward_receiver_writer_kernel_id);
        for (const auto& core : sender_worker_cores) {
            // sender reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input.buffer()->address();
            worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
            // sender writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
            worker_writer_sender_runtime_args[1] = output.buffer()->address();
        }

        const auto& forward_receiver_core = receiver_worker_cores[1];
        // forward receiver reader
        auto& worker_forward_receiver_reader_runtime_args =
            worker_forward_receiver_reader_runtime_args_by_core[forward_receiver_core.x][forward_receiver_core.y];
        worker_forward_receiver_reader_runtime_args[0] = intermed.buffer()->address();
        // forward receiver writer
        auto& worker_forward_receiver_writer_runtime_args =
            worker_forward_receiver_writer_runtime_args_by_core[forward_receiver_core.x][forward_receiver_core.y];
        worker_forward_receiver_writer_runtime_args[0] = output.buffer()->address();
        const auto& backward_receiver_core = receiver_worker_cores[0];
        // backward receiver reader
        auto& worker_backward_receiver_reader_runtime_args =
            worker_backward_receiver_reader_runtime_args_by_core[backward_receiver_core.x][backward_receiver_core.y];
        worker_forward_receiver_reader_runtime_args[0] = intermed.buffer()->address();
        // backward receiver writer
        auto& worker_backward_receiver_writer_runtime_args =
            worker_backward_receiver_writer_runtime_args_by_core[backward_receiver_core.x][backward_receiver_core.y];
        worker_backward_receiver_writer_runtime_args[0] = output.buffer()->address();
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
