// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

#include "debug/dprint.h"
#include "debug/dprint_pages.h"

namespace NAMESPACE {

// Helper constexpr function to compute num_blocks_per_col
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt) {
    for (uint32_t bct = 8; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }
    return 1;  // fallback, though unreachable if per_core_block_tile_cnt ≥ 1
}

void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    // Compute optimal num_blocks_per_col and block_ct_dim
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(per_core_block_tile_cnt);
    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

    // DPRINT << "block_ct_dim " << block_ct_dim << " full_ct_dim " << full_ct_dim << ENDL();

    pack_untilize_init<block_ct_dim, full_ct_dim>(src_cb_id, out_cb_id);

    cb_reserve_back(out_cb_id, full_ct_dim);

    for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
        cb_wait_front(src_cb_id, block_ct_dim);
        pack_untilize_block<block_ct_dim, full_ct_dim>(src_cb_id, 1, out_cb_id, b);
        cb_pop_front(src_cb_id, block_ct_dim);
    }

    cb_push_back(out_cb_id, full_ct_dim);

    pack_untilize_uninit(out_cb_id);
}
}  // namespace NAMESPACE
