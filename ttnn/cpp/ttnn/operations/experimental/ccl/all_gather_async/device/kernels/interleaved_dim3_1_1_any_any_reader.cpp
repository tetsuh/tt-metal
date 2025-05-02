// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType input_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr BufferType output_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_slices_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_slices_backward_direction = get_compile_time_arg_val(7);
constexpr bool fuse_op = get_compile_time_arg_val(8);
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
        .data_format = get_dataformat(cb0_id)};
    uint32_t pages_read_in_row = 0;
    uint32_t row_offset = 0;
    uint32_t tiles_read = 0;
    uint32_t tiles_to_read = slice_num_pages;
    uint32_t slice_Wt = input_tensor_Wt;
    uint32_t stride_Wt = input_tensor_Wt;
    while (tiles_read < tiles_to_read) {
        cb_reserve_back(cb0_id, packet_size_in_pages);
        size_t l1_write_addr = get_read_ptr(cb0_id);
        uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
        for (uint32_t j = 0; j < num_pages_to_read; j++) {
            noc_async_read_tile(
                input_tile_id_start + row_offset + pages_read_in_row, input_tensor_addrgen, l1_write_addr);
            l1_write_addr += input_tensor_page_size;
            tiles_read++;

            pages_read_in_row++;
            if (pages_read_in_row >= slice_Wt) {
                row_offset += stride_Wt;
                pages_read_in_row = 0;
            }
        }

        noc_async_read_barrier();
        cb_push_back(cb0_id, packet_size_in_pages);
    }

    constexpr bool output_tensor_is_dram = output_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto output_tensor_addrgen = InterleavedAddrGenFast<output_tensor_is_dram>{
        .bank_base_address = output_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb0_id)};
    uint32_t forward_slices_received = 0;
    uint32_t backward_slices_received = 0;
    while (forward_slices_received < num_slices_forward_direction ||
           backward_slices_received < num_slices_backward_direction) {
        if (forward_slices_received < num_slices_forward_direction) {
            while (!*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_forward));

            // Reset out_ready semaphore
            const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem_forward);
            noc_inline_dw_write(dest_noc_addr, 0);
            forward_slices_received++;

            uint32_t forward_chip_id = my_chip_id + forward_slices_received + 1;
            if (fuse_op) {
                // Signal matmul to go
                op_signaler_forward.synchronize_workers_and_signal_op(forward_chip_id);
            }

            if (forward_slices_received < num_slices_forward_direction) {
                // read the next forward slice out of memory, and put it in CB
                uint32_t output_tile_id_start = forward_chip_id * input_tensor_Wt;
                pages_read_in_row = 0;
                row_offset = 0;
                tiles_read = 0;
                tiles_to_read = slice_num_pages;
                slice_Wt = input_tensor_Wt;
                stride_Wt = output_tensor_Wt;
                while (tiles_read < tiles_to_read) {
                    cb_reserve_back(cb0_id, packet_size_in_pages);
                    size_t l1_write_addr = get_read_ptr(cb0_id);
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
                    cb_push_back(cb0_id, packet_size_in_pages);
                }
            }
        }

        if (backward_slices_received < num_slices_backward_direction) {
            while (!*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_backward));

            // Reset out_ready semaphore
            const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem_backward);
            noc_inline_dw_write(dest_noc_addr, 0);
            backward_slices_received++;

            uint32_t backward_chip_id = my_chip_id - backward_slices_received - 1;
            if (fuse_op) {
                // Signal matmul to go
                op_signaler_forward.synchronize_workers_and_signal_op(backward_chip_id);
            }

            if (backward_slices_received < num_slices_backward_direction) {
                // read the next backward slice out of memory, and put it in CB
                uint32_t output_tile_id_start = backward_chip_id * input_tensor_Wt;
                pages_read_in_row = 0;
                row_offset = 0;
                tiles_read = 0;
                tiles_to_read = slice_num_pages;
                slice_Wt = input_tensor_Wt;
                stride_Wt = output_tensor_Wt;
                while (tiles_read < tiles_to_read) {
                    cb_reserve_back(cb0_id, packet_size_in_pages);
                    size_t l1_write_addr = get_read_ptr(cb0_id);
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
                    cb_push_back(cb0_id, packet_size_in_pages);
                }
            }
        }
    }
}
