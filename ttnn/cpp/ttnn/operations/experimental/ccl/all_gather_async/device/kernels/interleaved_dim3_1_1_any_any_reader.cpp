// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

// inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//      DPRINT << "======" << ENDL();
//      for (uint8_t r = 0; r < 32; ++ r) {
//          SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
//          SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
//          DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " " << TileSlice(cb_id,
//          tile_id, sr_right, true, untilize) << ENDL();
//      }
//      DPRINT << "++++++" << ENDL();
// }

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType input_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr BufferType output_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb_forward_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_backward_id = get_compile_time_arg_val(4);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
constexpr bool fuse_op = get_compile_time_arg_val(9);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(10));
/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_forward = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_backward = get_arg_val<uint32_t>(arg_idx++);

    OpSignaler op_signaler_forward;
    OpSignaler op_signaler_backward;
    if constexpr (fuse_op) {
        op_signaler_forward = OpSignaler(arg_idx);
        op_signaler_backward = OpSignaler(arg_idx);
    }

    // Push out our local slice
    constexpr bool input_tensor_is_dram = input_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto input_tensor_addrgen = InterleavedAddrGenFast<input_tensor_is_dram>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_forward_id)};
    uint32_t pages_read_in_row = 0;
    uint32_t row_offset = 0;
    uint32_t tiles_read = 0;
    uint32_t tiles_to_read = slice_num_pages;
    uint32_t slice_Wt = input_tensor_Wt;
    uint32_t stride_Wt = 0;
    // DPRINT << "INPUT TILE ID START " << input_tile_id_start << ENDL();
    // DPRINT << "slice_Wt " << slice_Wt << ENDL();
    // DPRINT << "stride_Wt " << stride_Wt << ENDL();
    // DPRINT << "tiles_to_read " << tiles_to_read << ENDL();
    while (tiles_read < tiles_to_read) {
        cb_reserve_back(cb_forward_id, packet_size_in_pages);
        const uint32_t l1_write_addr_base = get_write_ptr(cb_forward_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
        for (uint32_t j = 0; j < num_pages_to_read; j++) {
            noc_async_read_tile(tiles_read, input_tensor_addrgen, l1_write_addr);
            l1_write_addr += input_tensor_page_size;
            tiles_read++;
        }

        noc_async_read_barrier();
        cb_push_back(cb_forward_id, packet_size_in_pages);
    }

    // DPRINT << "READER PAST WRITE LOCAL SEMAPHORE\n" << ENDL();
    // DPRINT << "MY CHIP ID " << my_chip_id << ENDL();
    // DPRINT << "num targets forward " << num_targets_forward_direction << ENDL();
    // DPRINT << "num targets backward " << num_targets_backward_direction << ENDL();

    constexpr bool output_tensor_is_dram = output_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto output_tensor_addrgen = InterleavedAddrGenFast<output_tensor_is_dram>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_forward_id)};
    uint32_t forward_slices_received = 0;
    uint32_t backward_slices_received = 0;
    uint32_t forward_slices_expected, backward_slices_expected;
    if (topology == Topology::Linear) {
        DPRINT << "READER TOPOLOGY LINEAR" << ENDL();
        forward_slices_expected = num_targets_forward_direction;
        backward_slices_expected = num_targets_backward_direction;
    } else if (topology == Topology::Ring) {
        DPRINT << "READER TOPOLOGY RING" << ENDL();
        forward_slices_expected = num_targets_backward_direction;
        backward_slices_expected = num_targets_forward_direction;
    }
    // DPRINT << "forward slices expected " << forward_slices_expected << ENDL();
    // DPRINT << "backward slices expected " << backward_slices_expected << ENDL();
    while (forward_slices_received < forward_slices_expected || backward_slices_received < backward_slices_expected) {
        DPRINT << "READER WHILE LOOP ITERATION\n" << ENDL();
        // Do i expect more from the right?
        // In the linear case, I expect num_targets_forward_direction slices from the right
        // In the ring case, I expect num_targets_backward_direction slices from the right
        if (forward_slices_received < forward_slices_expected) {
            DPRINT << "READER EXPECT " << forward_slices_expected << " from the right, have gotten "
                   << forward_slices_received << ENDL();
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_forward) <= forward_slices_received);
            DPRINT << "READER GOT slice " << forward_slices_received + 1 << " from the right" << ENDL();
            // Got it
            forward_slices_received++;

            uint32_t forward_chip_id = my_chip_id + forward_slices_received;
            uint32_t actual_forward_chip_id =
                (forward_chip_id >= ring_size) ? forward_chip_id - ring_size : forward_chip_id;
            if (fuse_op) {
                // Signal matmul to go
                op_signaler_forward.synchronize_workers_and_signal_op(actual_forward_chip_id);
            }

            // Should I forward what I got from the right to my left?
            // In the linear case, if I have any targets to my left, always forward
            // In the ring case, if I have received on the right less than my targets on the left, forward
            if ((topology == Topology::Linear && num_targets_backward_direction > 0) ||
                (topology == Topology::Ring && (forward_slices_received < (num_targets_backward_direction + 1)))) {
                DPRINT << "READER SEND WHAT I GOT FROM THE RIGHT TO THE LEFT " << ENDL();
                // read the next forward slice out of memory, and put it in CB
                uint32_t output_tile_id_start = actual_forward_chip_id * input_tensor_Wt;
                pages_read_in_row = 0;
                row_offset = 0;
                tiles_read = 0;
                tiles_to_read = slice_num_pages;
                slice_Wt = input_tensor_Wt;
                stride_Wt = output_tensor_Wt;
                while (tiles_read < tiles_to_read) {
                    cb_reserve_back(cb_backward_id, packet_size_in_pages);
                    size_t l1_write_addr = get_write_ptr(cb_backward_id);
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        noc_async_read_tile(
                            output_tile_id_start + row_offset + pages_read_in_row,
                            output_tensor_addrgen,
                            l1_write_addr);
                        l1_write_addr += input_tensor_page_size;
                        tiles_read++;

                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_backward_id, packet_size_in_pages);
                }
            }
        }

        // Do i expect more from the left?
        // In the linear case, I expect num_targets_backward_direction slices from the left
        // In the ring case, I expect num_targets_forward_direction slices from the right
        if (backward_slices_received < backward_slices_expected) {
            DPRINT << "READER EXPECT " << backward_slices_expected << " from the left, have gotten "
                   << backward_slices_received << ENDL();
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_backward) <= backward_slices_received);
            DPRINT << "READER GOT slice " << backward_slices_received + 1 << " from the left" << ENDL();
            // Got it
            backward_slices_received++;

            int backward_chip_id = my_chip_id - backward_slices_received;
            uint32_t actual_backward_chip_id = (backward_chip_id < 0) ? ring_size + backward_chip_id : backward_chip_id;
            if (fuse_op) {
                // Signal matmul to go
                op_signaler_forward.synchronize_workers_and_signal_op(actual_backward_chip_id);
            }

            // Should I forward what I got from the left to my right?
            // In the linear case, if I have any targets to my right, always forward
            // In the ring case, if I have received on the left less than my targets on the right, forward
            if ((topology == Topology::Linear && num_targets_forward_direction > 0) ||
                (topology == Topology::Ring && (backward_slices_received < (num_targets_forward_direction + 1)))) {
                DPRINT << "READER SEND WHAT I GOT FROM THE LEFT TO THE RIGHT " << ENDL();
                // read the next backward slice out of memory, and put it in CB
                uint32_t output_tile_id_start = actual_backward_chip_id * input_tensor_Wt;
                pages_read_in_row = 0;
                row_offset = 0;
                tiles_read = 0;
                tiles_to_read = slice_num_pages;
                slice_Wt = input_tensor_Wt;
                stride_Wt = output_tensor_Wt;
                while (tiles_read < tiles_to_read) {
                    cb_reserve_back(cb_forward_id, packet_size_in_pages);
                    size_t l1_write_addr = get_write_ptr(cb_forward_id);
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        noc_async_read_tile(
                            output_tile_id_start + row_offset + pages_read_in_row,
                            output_tensor_addrgen,
                            l1_write_addr);
                        l1_write_addr += input_tensor_page_size;
                        tiles_read++;

                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_forward_id, packet_size_in_pages);
                }
            }
        }
        DPRINT << "END OF WHILE LOOP " << ENDL();
    }
}
