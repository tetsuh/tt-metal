// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "debug/dprint_tensix.h"
#include "ttnn/cpp/ttnn/operations/examples/example/device/kernels/compute/utils.hpp"
#include "tensix_types.h"
#include "tensix.h"

namespace NAMESPACE {
void MAIN {
    DPRINT << "TR starts" << ENDL();
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    uint32_t src_addr = get_compile_time_arg_val(2);
    uint32_t dst_addr = get_compile_time_arg_val(3);
    DPRINT << (uint)(src_addr < (1024 * 1464)) << ENDL();
    DPRINT << (uint)(dst_addr < (1024 * 1464)) << ENDL();

    constexpr tt::CBIndex cb_in = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_1;

    Mover mover{};
    constexpr uint tile_els = 1024;
    constexpr uint face_els = 256;
    constexpr uint num_faces = 4;
    constexpr uint row_els = 16;
    constexpr uint num_rows = 16;
    constexpr uint byte_size = 4;
    constexpr uint tile_size_bytes = tile_els * byte_size;
    constexpr uint one_row_bytes = row_els * byte_size;

    init_sfpu(cb_in, cb_out);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_wait_front(cb_in, per_core_block_dim);
        cb_reserve_back(cb_out, per_core_block_dim);

        for (uint32_t tile_id = 0; tile_id < per_core_block_dim; ++tile_id) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, tile_id, tile_id);
            tile_regs_commit();

            write_through_pack_tile(tile_id, cb_out);
        }

        cb_push_back(cb_out, per_core_block_dim);
        cb_pop_front(cb_in, per_core_block_dim);
    }

    DPRINT << "TR ends" << ENDL();
}
}  // namespace NAMESPACE

// PACK(TTI_WRCFG(src_addr >> 4, false, THCON_SEC0_REG6_Source_address_ADDR32));
// PACK(TTI_WRCFG(dst_addr >> 4, false, THCON_SEC0_REG6_Destination_address_ADDR32));
// PACK(TTI_WRCFG(tile_size_bytes >> 4, false, THCON_SEC0_REG6_Buffer_size_ADDR32));
// PACK(TTI_WRCFG(mover.transfer_direction, false, THCON_SEC0_REG6_Transfer_direction_ADDR32));
// PACK(TTI_XMOV(0, 0));

// PACK(TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK)); // stall Thread Controller until pack becomes idle
// PACK(TTI_FLUSHDMA(0x8));
