// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "ckernel.h"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
constexpr bool dynamic_alternate = get_compile_time_arg_val(9);
constexpr bool fuse_op = get_compile_time_arg_val(10);

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
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_forward = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_backward = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;

    if constexpr (fuse_op) {
        arg_idx = arg_for_fab;
        op_signaler_sender = OpSignaler(arg_idx);
    }

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_unicast(1);
    pkt_hdr_backward->to_chip_unicast(1);

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    uint32_t forward_writes = 0;
    uint32_t backward_writes = 0;

    // Write out the local slice to both DRAM and forward and backward
    uint32_t pages_read_in_row = 0;
    uint32_t row_offset = 0;
    uint32_t tiles_read = 0;
    uint32_t tiles_to_read = slice_num_pages;
    uint32_t tile_id_start = my_chip_id * input_tensor_Wt;
    while (tiles_read < tiles_to_read) {
        cb_wait_front(cb0_id, packet_size_in_pages);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
        uint32_t contig_pages_advanced = 1;  // always 1 for interleaved
        for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(
                tile_id_start + row_offset + pages_read_in_row, tensor0_addrgen, 0 /*offset*/, 0 /*noc_id*/);
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                contig_pages_advanced * tensor0_page_size);
            tiles_read++;
            pages_read_in_row++;
            if (pages_read_in_row >= input_tensor_Wt) {
                row_offset += output_tensor_Wt;
                pages_read_in_row = 0;
            }
        }

        cb_pop_front(cb0_id, packet_size_in_pages);
    }

    // 2. unicast output ready semaphore forward
    uint64_t out_ready_sem_noc_addr_in_pkt_forward =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_backward, 0);
    auto* pkt_hdr_fwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc_forward);
    pkt_hdr_fwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt_forward,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (forward)
    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
    pkt_hdr_fwd->to_chip_unicast(1);
    fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
        packet_header_buffer_seminc_forward, sizeof(PACKET_HEADER_TYPE));
    forward_writes++;
    // 2. unicast output ready semaphore backward
    uint64_t out_ready_sem_noc_addr_in_pkt_backward =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_forward, 0);
    auto* pkt_hdr_bwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc_backward);
    pkt_hdr_bwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt_backward,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (backward)
    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
    pkt_hdr_bwd->to_chip_unicast(1);
    fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
        packet_header_buffer_seminc_backward, sizeof(PACKET_HEADER_TYPE));
    backward_writes++;

    // increment locally
    if (fuse_op) {
        // Synchronize and signal that the local tensor slice is available
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
    }

    while (forward_writes < num_targets_forward_direction || backward_writes < num_targets_backward_direction) {
        // unicast backward
        if (backward_writes < num_targets_backward_direction) {
            pages_read_in_row = 0;
            row_offset = 0;
            tiles_read = 0;
            uint32_t slice_chip_id = my_chip_id + backward_writes;
            tile_id_start = slice_chip_id * input_tensor_Wt;
            tiles_to_read = slice_num_pages;
            while (tiles_read < tiles_to_read) {
                cb_wait_front(cb0_id, packet_size_in_pages);
                size_t l1_read_addr = get_read_ptr(cb0_id);
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                uint32_t contig_pages_advanced = 1;  // always 1 for interleaved
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint64_t noc0_dest_noc_addr = get_noc_addr(
                        tile_id_start + row_offset + pages_read_in_row, tensor0_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    write_and_advance_local_read_address_for_fabric_write_backward(
                        noc0_dest_noc_addr,
                        pkt_hdr_backward,
                        fabric_connection,
                        l1_read_addr,
                        contig_pages_advanced * tensor0_page_size);
                    tiles_read++;
                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt) {
                        row_offset += output_tensor_Wt;
                        pages_read_in_row = 0;
                    }
                }

                cb_pop_front(cb0_id, packet_size_in_pages);
            }

            // 2. unicast output ready semaphore backward
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_bwd->to_chip_unicast(1);
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc_backward, sizeof(PACKET_HEADER_TYPE));

            backward_writes++;
        }

        // unicast forward
        if (forward_writes < num_targets_forward_direction) {
            pages_read_in_row = 0;
            row_offset = 0;
            tiles_read = 0;
            uint32_t slice_chip_id = my_chip_id - forward_writes;
            tile_id_start = slice_chip_id * input_tensor_Wt;
            tiles_to_read = slice_num_pages;
            while (tiles_read < tiles_to_read) {
                cb_wait_front(cb0_id, packet_size_in_pages);
                size_t l1_read_addr = get_read_ptr(cb0_id);
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                uint32_t contig_pages_advanced = 1;  // always 1 for interleaved
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint64_t noc0_dest_noc_addr = get_noc_addr(
                        tile_id_start + row_offset + pages_read_in_row, tensor0_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    write_and_advance_local_read_address_for_fabric_write_forward(
                        noc0_dest_noc_addr,
                        pkt_hdr_forward,
                        fabric_connection,
                        l1_read_addr,
                        contig_pages_advanced * tensor0_page_size);

                    tiles_read++;
                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt) {
                        row_offset += output_tensor_Wt;
                        pages_read_in_row = 0;
                    }
                }

                cb_pop_front(cb0_id, packet_size_in_pages);
            }

            // 2. unicast output ready semaphore forward
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_fwd->to_chip_unicast(1);
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc_forward, sizeof(PACKET_HEADER_TYPE));

            forward_writes++;
        }
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
}
