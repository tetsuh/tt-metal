// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Make n reads defined by num_reads
// Writes to Specified Circular Buffers in L1
// Expects n provided src_addr, src_noc_x, src_noc_y, and cb_id_in
void kernel_main() {
    DPRINT << "this is reader 2\n";
    const uint32_t num_pages = get_arg_val<uint32_t>(0);
    const uint32_t start_tensor = get_arg_val<uint32_t>(1);
    const uint32_t start_tensor_id = get_arg_val<uint32_t>(2);
    DPRINT << "num_pages: " << (uint32_t)num_pages << ", start_tensor: " << (uint32_t)start_tensor
           << ", start_tensor_id: " << (uint32_t)start_tensor_id << ENDL();

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(1);
    DPRINT << "cb_id_in: " << (uint32_t)cb_id_in << ", num_tensors: " << (uint32_t)num_tensors << ENDL();

    // ublocks size defined in pages
    constexpr uint32_t ublock_size_pages = 1;

    uint8_t l1_src_addr_gens_memblk[sizeof(InterleavedAddrGen<false>) * num_tensors];
    uint8_t dram_src_addr_gens_memblk[sizeof(InterleavedAddrGen<true>) * num_tensors];

    InterleavedAddrGen<false>* l1_src_addr_gens = reinterpret_cast<InterleavedAddrGen<false>*>(l1_src_addr_gens_memblk);
    InterleavedAddrGen<true>* dram_src_addr_gens =
        reinterpret_cast<InterleavedAddrGen<true>*>(dram_src_addr_gens_memblk);

    bool is_dram[num_tensors];
    uint32_t num_pages_per_block[num_tensors];
    uint32_t page_id_per_tensor[num_tensors];
    constexpr uint32_t src_addr_base_idx = 3;
    constexpr uint32_t is_dram_base_offset = num_tensors;
    constexpr uint32_t num_pages_per_block_base_offset = is_dram_base_offset + num_tensors;
    constexpr uint32_t page_size_per_tensor_offset = num_pages_per_block_base_offset + num_tensors;
    constexpr uint32_t page_id_per_tensor_offset = page_size_per_tensor_offset + num_tensors;
    tt_l1_ptr uint32_t* arg_ptr = (tt_l1_ptr uint32_t*)get_arg_addr(src_addr_base_idx);
    ""
    "
        const InterleavedAddrGen<1>
            s_test = {.bank_base_address = arg_ptr[0], .page_size = arg_ptr[page_size_per_tensor_offset]};
    cb_reserve_back(cb_id_in, ublock_size_pages);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in);
    noc_async_read_page(0, s_test, l1_write_addr);
    noc_async_read_barrier();
    ""
    "
        // volatile tt_l1_ptr uint32_t* dst_noc2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
        //     for (uint32_t value = 0; value < 32; value++) {
        //         DPRINT << "value at " << (uint32_t)value << " is: " << (uint32_t)dst_noc2[value] << ENDL();
        //     }

        // DPRINT << "after my check\n";

        for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t src_addr = arg_ptr[i];
        is_dram[i] = (bool)arg_ptr[is_dram_base_offset + i];
        num_pages_per_block[i] = arg_ptr[num_pages_per_block_base_offset + i];
        page_id_per_tensor[i] = arg_ptr[page_id_per_tensor_offset + i];
        DPRINT << "num_pages_per_block: " << (uint32_t)num_pages_per_block[i]
               << ", page_id_per_tensor: " << (uint32_t)page_id_per_tensor[i] << ENDL();
        if (is_dram[i]) {
            DPRINT << "src_addr: " << (uint32_t)src_addr
                   << ", is_dram: true, page_size: " << (uint32_t)arg_ptr[page_size_per_tensor_offset + i] << ENDL();
            new (&dram_src_addr_gens[i]) InterleavedAddrGen<true>{
                .bank_base_address = src_addr, .page_size = arg_ptr[page_size_per_tensor_offset + i]};
        } else {
            new (&l1_src_addr_gens[i]) InterleavedAddrGen<false>{
                .bank_base_address = src_addr, .page_size = arg_ptr[page_size_per_tensor_offset + i]};
        }
    }

    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    // FIX RM CONCAT WIDTH
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id_in, ublock_size_pages);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);
#ifdef WIDTH_CONCAT
        // For width concat we know we start at curr_tensor=0
        // num_pages_per_block[curr_tensor] is always one for width concat
        for (uint32_t j = 0; j < num_tensors; ++j) {
            if (is_dram[curr_tensor]) {
                noc_async_read_page(page_id_per_tensor[curr_tensor], dram_src_addr_gens[curr_tensor], l1_write_addr);
                l1_write_addr += dram_src_addr_gens[curr_tensor].page_size;
            } else {
                noc_async_read_page(page_id_per_tensor[curr_tensor], l1_src_addr_gens[curr_tensor], l1_write_addr);
                l1_write_addr += l1_src_addr_gens[curr_tensor].page_size;
            }
            page_id_per_tensor[curr_tensor]++;
            curr_tensor++;
        }
        curr_tensor = 0;
#else
        if (is_dram[curr_tensor]) {
            DPRINT << "page_id_per_tensor dram[curr_tensor] :" << (uint32_t)page_id_per_tensor[curr_tensor] << ENDL();
            DPRINT << "l1_write_addr: " << (uint32_t)l1_write_addr << ENDL();
            noc_async_read_page(page_id_per_tensor[curr_tensor], dram_src_addr_gens[curr_tensor], l1_write_addr);
        } else {
            DPRINT << "page_id_per_tensor l1[curr_tensor] :" << (uint32_t)page_id_per_tensor[curr_tensor] << ENDL();
            DPRINT << "l1_write_addr: " << (uint32_t)l1_write_addr << ENDL();
            noc_async_read_page(page_id_per_tensor[curr_tensor], l1_src_addr_gens[curr_tensor], l1_write_addr);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* dst_noc2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
        for (uint32_t value = 0; value < 32; value++) {
            DPRINT << "value at " << (uint32_t)value << " is: " << (uint32_t)dst_noc2[value] << ENDL();
        }

        page_id_per_tensor[curr_tensor]++;
        curr_tensor_id++;

        if (curr_tensor_id == num_pages_per_block[curr_tensor]) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
            }
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_id_in, ublock_size_pages);
    }
}
