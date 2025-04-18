// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

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
constexpr uint32_t num_max_targets = std::max(num_targets_forward_direction, num_targets_backward_direction);
constexpr uint32_t num_sync_targets_forward = dynamic_alternate ? num_max_targets : num_targets_forward_direction;
constexpr uint32_t num_sync_targets_backward = dynamic_alternate ? num_max_targets : num_targets_backward_direction;

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
    uint32_t tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_bank_addrs[ring_size];
    for (uint32_t device_id = 0; device_id < ring_size; device_id++) {
        out_ready_sem_bank_addrs[device_id] = get_arg_val<uint32_t>(arg_idx++);
    }
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

    /* Args for overlapped all gather */
    OpSignaler op_signaler_forward;
    OpSignaler op_signaler_backward;
    OpSignaler op_signaler_sender;

    if constexpr (fuse_op) {
        arg_idx = arg_for_fab;
        op_signaler_forward = OpSignaler(arg_idx);
        op_signaler_backward = OpSignaler(arg_idx);
        op_signaler_sender = OpSignaler(arg_idx);
    }

    // DPRINT << "ct args: \n";
    // DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    // DPRINT << "reserved_packet_header_cb_id: " << (uint32_t)reserved_packet_header_cb_id << "\n";
    // DPRINT << "num_packet_headers_storable: " << (uint32_t)num_packet_headers_storable << "\n";
    // DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    // DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    // DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    // DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    // DPRINT << "num_targets_forward_direction: " << (uint32_t)num_targets_forward_direction << "\n";
    // DPRINT << "num_targets_backward_direction: " << (uint32_t)num_targets_backward_direction << "\n";

    // DPRINT << "rt args: \n";
    // DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    // DPRINT << "tile_id_start: " << (uint32_t)tile_id_start << "\n";
    // DPRINT << "tile_id_end: " << (uint32_t)tile_id_end << "\n";
    // DPRINT << "wait_output_semaphore: " << (uint32_t)wait_output_semaphore << "\n";
    // DPRINT << "reset_global_semaphore: " << (uint32_t)reset_global_semaphore << "\n";
    // DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << "\n";
    // DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << "\n";
    // DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << "\n";
    // DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << "\n";

    // DPRINT << "arg_for_fab: " << (uint32_t)arg_for_fab << "\n";
    // DPRINT << "fabric_connection arg 0" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    // DPRINT << "fabric_connection arg 1" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    // DPRINT << "fabric_connection arg 2" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    // DPRINT << "fabric_connection arg 3" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    // DPRINT << "fabric_connection arg 4" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    DPRINT << "packet_header_buffer_addr_forward: " << (uint32_t)packet_header_buffer_addr_forward << "\n";
    DPRINT << "packet_header_buffer_addr_backward: " << (uint32_t)packet_header_buffer_addr_backward << "\n";
    DPRINT << "packet_header_buffer_seminc: " << (uint32_t)packet_header_buffer_seminc << "\n";

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
    pkt_hdr_backward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    // 1. mcast via fabric to remote tensor addresses
    uint32_t pages_read_in_row = 0;
    uint32_t row_offset = 0;
    uint32_t tiles_read = 0;
    while (tiles_read < tile_id_end) {
        // DPRINT << "tile_id: " << tiles_read << "\n";
        cb_wait_front(cb0_id, packet_size_in_pages);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint32_t num_pages_to_read = std::min(tile_id_end - tiles_read, packet_size_in_pages);
        uint32_t contig_pages_advanced = 1;  // always 1 for interleaved
        for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(
                tile_id_start + row_offset + pages_read_in_row, tensor0_addrgen, 0 /*offset*/, 0 /*noc_id*/);
            // DPRINT << "j: " << j << "\n";
            // DPRINT << "noc0_dest_noc_addr: " << noc0_dest_noc_addr << "\n";
            // DPRINT << "tile_id: " << tiles_read << "\n";
            // DPRINT << "row_offset: " << row_offset << "\n";
            // DPRINT << "pages_read_in_row: " << pages_read_in_row << "\n";
            // DPRINT << "real_tile_id: " << tile_id_start+row_offset+pages_read_in_row << "\n";

            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                contig_pages_advanced * tensor0_page_size);
            if constexpr (dynamic_alternate) {
                std::swap(
                    pkt_hdr_forward->routing_fields.value,
                    pkt_hdr_backward->routing_fields
                        .value);  // alternate the packet header distance for better balancing
            }
            tiles_read++;
            pages_read_in_row++;
            if (pages_read_in_row >= input_tensor_Wt) {
                row_offset += output_tensor_Wt;
                pages_read_in_row = 0;
            }
        }

        cb_pop_front(cb0_id, packet_size_in_pages);
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addrs[my_chip_id], 0);
    auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_forward)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_backward)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addrs[my_chip_id]);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);
    DPRINT << "inc done\n";
    if (fuse_op) {
        // Synchronize and signal that the local tensor slice is available
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
    }

    // 3. wait for mcast output ready semaphore
    uint32_t forward_chip_id = my_chip_id + 1;
    int backward_chip_id = my_chip_id - 1;

    uint32_t num_transfers_forward = 0;
    uint32_t num_transfers_backward = 0;
    uint32_t last_forward_chip;
    int last_backward_chip;
    if (fuse_op) {
        // match the ring configuration of matmul
        num_transfers_forward = (ring_size - 1) / 2;
        num_transfers_backward = (((ring_size - 1) - 1) / 2) + 1;
        last_forward_chip = my_chip_id + num_transfers_forward + 1;
        last_backward_chip = my_chip_id - num_transfers_backward;
    } else {
        // linear topology
        last_forward_chip = ring_size;
        last_backward_chip = 0;
    }
    if (wait_output_semaphore) {
        while ((forward_chip_id < last_forward_chip) || (backward_chip_id >= last_backward_chip)) {
            // Check forward
            if (forward_chip_id < last_forward_chip) {
                uint32_t actual_forward_chip_id =
                    (forward_chip_id >= ring_size) ? forward_chip_id - ring_size : forward_chip_id;
                if (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addrs[actual_forward_chip_id])) {
                    if (fuse_op) {
                        op_signaler_forward.synchronize_workers_and_signal_op(actual_forward_chip_id);
                    }
                    forward_chip_id++;
                }
            }
            // Check backward
            if (backward_chip_id >= last_backward_chip) {
                uint32_t actual_backward_chip_id =
                    (backward_chip_id < 0) ? ring_size + backward_chip_id : backward_chip_id;
                if (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addrs[actual_backward_chip_id])) {
                    if (fuse_op) {
                        op_signaler_backward.synchronize_workers_and_signal_op(actual_backward_chip_id);
                    }
                    backward_chip_id--;
                }
            }
        }
        DPRINT << "waitval done\n";
    }

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        for (uint32_t device_id = 0; device_id < ring_size; device_id++) {
            const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem_bank_addrs[device_id]);
            noc_inline_dw_write(dest_noc_addr, 0);
        }
        DPRINT << "reset done\n";
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
    DPRINT << "DONE \n";
}
