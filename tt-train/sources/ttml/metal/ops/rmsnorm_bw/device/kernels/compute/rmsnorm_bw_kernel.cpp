// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/cb_api.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reg_api.h>
#include <debug/dprint.h>

#include <cstdint>

// TODO REMOVE UNNECESSARY INCLUDES
#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
// TODO: remove this using namespace. The functions should be accessible with out it.
using namespace ckernel;

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size =
    get_compile_time_arg_val(1);                          // Number of tiles in the inner dimention of the input tensor.
constexpr uint32_t mask_w = get_compile_time_arg_val(2);  // Unused atm.
constexpr uint32_t Wt = get_compile_time_arg_val(3);

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;  // Unused atm
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;  // Number of activations, i.e. c in the paper
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_6;
constexpr uint32_t cb_dL_dgamma_idx = tt::CBIndex::c_7;
// CBs with intermediate computations
constexpr uint32_t cb_scaled_gain = tt::CBIndex::c_8;
constexpr uint32_t cb_gained_dL_dout = tt::CBIndex::c_9;
constexpr uint32_t cb_scale = tt::CBIndex::c_10;
constexpr uint32_t cb_ms_a = tt::CBIndex::c_11;
constexpr uint32_t cb_c_by_ms_a = tt::CBIndex::c_12;
constexpr uint32_t cb_rhs = tt::CBIndex::c_13;
constexpr uint32_t cb_a_over_rms_a = tt::CBIndex::c_14;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_15;

constexpr uint32_t onetile = 1;

#ifdef DO_MASK_W  // Unsued atm
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

// Figure out why MAIN without ( )
void MAIN {
    cb_wait_front(cb_scaler_idx, onetile);
    cb_wait_front(cb_gamma_idx, onetile);

    if constexpr (do_mask_w) {
        // cb_wait_front(cb_mask_w_idx, onetile);
    }

    // We need to init_sfpu with cb_input_idx and cb_dL_da_idx, so that it knows how to handle the data formats. Since
    // all computations are done in bfloat16, we do not need to reconfigure the SFPU for each operation.
    init_sfpu(cb_input_idx, cb_dL_da_idx);

    // Should be here not sure exactly why, but it is needed to initialize the SFPU for the binary operations. TBC
    // later.
    binary_op_init_common(cb_input_idx, cb_gamma_idx, cb_dL_da_idx);

    // What is the purpose of reconfig_data_format here? It might work without it, but it is better to be sure that the
    // data format is correct. reconfig_data_format(cb_input_idx, cb_gamma_idx);

    // Notation:
    // _ · _ <- usual dot product
    // _ @ _ <- matrix multiplication
    // _ *. _ <- Hadamard product/eltwise multiplication with broadcasting
    // _ /. _ <- eltwise division with broadcasting
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // 1. Wait for the input tensor, rms_a and dL_out to be ready.
        cb_wait_front(cb_input_idx, Wt);
        // RMS(a) is a scalar, so we wait for one tile only.
        cb_wait_front(cb_rms_a_idx, onetile);
        cb_wait_front(cb_dL_out_idx, Wt);

        for (uint32_t col = 0; col < Wt; ++col) {
            // 2. Compute:
            // auto scaled_gain = ttnn::divide(
            //     g,
            //     rms_a,
            //     std::nullopt,
            //     std::nullopt,
            //     std::nullopt,
            //     none,
            //     none,
            //     none,
            //     false);  // [1,1,1,C] x [B,1,S,1] -> [B,1,S,C] (bcast)
            // auto gained_dL_dout = ttnn::multiply(
            //     scaled_gain,
            //     dL_dout,
            //     std::nullopt,
            //     std::nullopt,
            //     std::nullopt,
            //     none,
            //     none,
            //     none,
            //     false);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]
            const uint32_t rms_a_register = 0;
            const uint32_t scaled_gain_register = 1;
            tile_regs_acquire();

            copy_tile_init(cb_gamma_idx);
            copy_tile(cb_gamma_idx, /* tile_idx */ col, /* register_idx */ scaled_gain_register);
            copy_tile_init(cb_rms_a_idx);
            copy_tile(cb_rms_a_idx, /* tile_idx */ col, /* register_idx */ rms_a_register);
            div_binary_tile_init();
            div_binary_tile(scaled_gain_register, rms_a_register);

            // Let's pack scaled_gain to cb_scaled_gain, and multiply it with dL_out to get gained_dL_dout in FPU.
            cb_reserve_back(cb_scaled_gain, onetile);
            tile_regs_commit();
            tile_regs_wait();
            // Q: is this pack_reconfig_data_format necessary? It seems like it is not, but it is better to be sure.
            pack_reconfig_data_format(cb_scaled_gain);
            pack_tile(scaled_gain_register, cb_scaled_gain);
            tile_regs_release();
            cb_push_back(cb_scaled_gain, onetile);

            const uint32_t gained_dL_dout_register = 0;
            tile_regs_acquire();

            // We can use tile idx 0 for all cols as we popfront the cb_scaled_gain in each iteration.
            mul_tiles_init(cb_dL_out_idx, cb_scaled_gain);
            mul_tiles(
                cb_dL_out_idx,
                cb_scaled_gain,
                /* tile_idx */ 0,
                /* tile_idx */ 0,
                gained_dL_dout_register);
            cb_pop_front(cb_scaled_gain, onetile);

            // NOTE:
            // The order of commit and wait does not matter when they are next to each other, as they handle different
            // threads. Commit releases the lock for the math thread, allowing the pack thread to start working on the
            // data, while wait is for the pack thread to finish math. In principle, you can commit first and then wait,
            // or wait first and then commit. Logically, it makes sense to say the math procedure is finished (commit)
            // and then packing can start (wait), so commit first and then wait is preferred.
            cb_reserve_back(cb_gained_dL_dout, onetile);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_gained_dL_dout);
            pack_tile(gained_dL_dout_register, cb_gained_dL_dout);
            tile_regs_release();
            cb_push_back(cb_gained_dL_dout, onetile);

            // 3. Compute:
            // auto scale = ttml::ttnn_fixed::sum_over_dim(
            //     ttnn::multiply(a, gained_dL_dout, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            //     3);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C] -> [B,1,S,1]
            //
            // We will calculate scale iteratively reducting it to a single, scalar value after each step.
            const uint32_t scale_register = 0;            // destination register for the reduction
            const uint32_t scale_reduction_register = 1;  // register for the reduction
            tile_regs_acquire();

            // Perform elementwise multiplication and sum reduction in one step
            reduce_init<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_gained_dL_dout, cb_input_idx, cb_scale);

            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(  // (sum over inner dimension)
                cb_gained_dL_dout,                              // main input buffer
                cb_input_idx,                                   // scaler buffer (elementwise mul)
                /* tile_idx */ col,                             // tile index in main buffer
                /* tile_idx */ col,                             // tile index in scaler buffer
                scale_register);                                // destination register
            reduce_revert_delta<ReduceDim::REDUCE_ROW>(cb_gained_dL_dout);

            if (col == 0) {
                copy_dest_values_init();
                copy_dest_values(scale_reduction_register, scale_register);
            } else {
                copy_tile_init(cb_scale);
                copy_tile(cb_scale, /* tile_idx */ col, /* register_idx */ scale_reduction_register);
                // NOTE: Keep in mind that this is not the best idea to put everything in CB at one. L1 means only that
                // we read once, but we shouldn't use ~30 CBs filling them all with whole inner dimension. The reduction
                // that we do afterwards could be done here, reducing memory usage. Therefore we clean the cb_scale
                // after the first tile, so that we do not use so much memory for the cb_scale.
                cb_pop_front(cb_scale, onetile);
                add_binary_tile_init();
                add_binary_tile(scale_reduction_register, scale_register);
            }

            cb_reserve_back(cb_scale, onetile);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_scale);
            pack_tile(scale_reduction_register, cb_scale);
            tile_regs_release();
            cb_push_back(cb_scale, onetile);
        }
        // 4. Compute c_by_ms_a. This can be done outside above loop, because rms_a and c are constant across all tiles
        // in the row.
        //
        // auto ms_a = ttnn::square(rms_a);  // [B,1,S,1] -> [B,1,S,1]
        // auto c_by_ms_a = ttnn::multiply(
        //     ms_a, c, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);  // [B,1,S,1] x [1] ->
        // [B,1,S,1] (bcast)
        const uint32_t ms_a = 0;
        tile_regs_acquire();
        mul_tiles_init(cb_rms_a_idx, cb_rms_a_idx);
        mul_tiles(cb_rms_a_idx, cb_rms_a_idx, /* tile_idx */ 0, /* tile_idx */ 0, ms_a);

        // Now we have ms_a in ms_a register, we can pack it to cb_ms_a.
        cb_reserve_back(cb_ms_a, onetile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_ms_a);
        pack_tile(ms_a, cb_ms_a);
        tile_regs_release();
        cb_push_back(cb_ms_a, onetile);

        // Now we can calculate c_by_ms_a = ms_a * c, where c is the constant value in cb_scaler_idx.
        const uint32_t c_by_ms_a_register = 0;
        tile_regs_acquire();
        mul_tiles_init(cb_ms_a, cb_scaler_idx);
        mul_tiles(cb_ms_a, cb_scaler_idx, /* tile_idx */ 0, /* tile_idx */ 0, c_by_ms_a_register);

        // Now we have c_by_ms_a = rms_a^2 *. c in c_by_ms_a_register, we can pack it to cb_c_by_ms_a.
        cb_reserve_back(cb_c_by_ms_a, onetile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_c_by_ms_a);
        pack_tile(c_by_ms_a_register, cb_c_by_ms_a);
        tile_regs_release();
        cb_push_back(cb_c_by_ms_a, onetile);
        // We can pop_front cb_ms_a, since we do not need it anymore.
        cb_pop_front(cb_ms_a, onetile);

        // 5. Compute:
        // auto scaled_outer = ttnn::multiply(
        //     scale,
        //     a,
        //     std::nullopt,
        //     std::nullopt,
        //     std::nullopt,
        //     none,
        //     none,
        //     none,
        //     false);  // [B,1,S,1] x [B,1,S,C] -> [B,1,S,C] (bcast)
        // auto rhs = ttnn::divide(
        //     scaled_outer,
        //     c_by_ms_a,
        //     std::nullopt,
        //     std::nullopt,
        //     std::nullopt,
        //     none,
        //     none,
        //     none,
        //     false);  // [B,1,S,C] x [B,1,S,1] -> [B,1,S,C] (bcast)

        // We need to store in registers scale and c_by_ms_a, and iterate over all tiles in cb_input_idx to calculate
        // rhs for each tile.
        for (uint32_t col = 0; col < Wt; ++col) {
            // NOTE: I don't like the fact that we are coping c_by_ms_a to register every time, but idk if
            // there is a way to avoid it. We need to have it in register to perform the division.
            const uint32_t c_by_ms_a_register = 1;
            const uint32_t rhs_register = 0;
            tile_regs_acquire();

            mul_tiles_init(cb_input_idx, cb_scale);
            // We can use tile idx 0 for all cols as we have a reducted, single value in cb_scale.
            mul_tiles(cb_input_idx, cb_scale, /* tile_idx */ col, /* tile_idx */ 0, rhs_register);
            copy_tile_init(c_by_ms_a_register);
            copy_tile(cb_c_by_ms_a, /* tile_idx */ 0, /* register_idx */ c_by_ms_a_register);

            div_binary_tile_init();
            div_binary_tile(rhs_register, c_by_ms_a_register);

            // Now we have rhs in rhs_register, we can pack it to cb_rhs.
            cb_reserve_back(cb_rhs, onetile);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_rhs);
            pack_tile(rhs_register, cb_rhs);
            tile_regs_release();
            cb_push_back(cb_rhs, onetile);

            // 6. Compute:
            // auto dL_da = ttnn::subtract(
            //     gained_dL_dout,
            //     rhs,
            //     std::nullopt,
            //     std::nullopt,
            //     std::nullopt,
            //     none,
            //     none,
            //     none,
            //     false);  // [B,1,S,C] x [B,1,S,C] -> [B,1,S,C]
            const uint32_t dL_da_register = 0;
            tile_regs_acquire();
            cb_reserve_back(cb_dL_da_idx, onetile);
            sub_tiles_init(cb_gained_dL_dout, cb_rhs, false);
            sub_tiles(cb_gained_dL_dout, cb_rhs, /* tile_idx */ col, /* tile_idx */ 0, dL_da_register);
            // Now we have dL_da in dL_da_register, we can pack it to cb_dL_da_idx.
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_dL_da_idx);
            pack_tile(dL_da_register, cb_dL_da_idx);
            tile_regs_release();
            cb_push_back(cb_dL_da_idx, onetile);
            // We can pop_front rsh, since we do not need it anymore.
            cb_pop_front(cb_rhs, onetile);
        }

        // 7. Compute:
        // auto dL_dg_components = ttnn::multiply(
        //     dL_dout,
        //     ttnn::divide(a, rms_a, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
        //     std::nullopt,
        //     std::nullopt,
        //     std::nullopt,
        //     none,
        //     none,
        //     none,
        //     false);  // [B,1,S,C] x [B,1,S,1] -> [B,1,S,C] (bcast); checked by add_grad
        // auto dL_dg = ttnn::sum(
        //     dL_dg_components,
        //     /* dim_arg */ ttnn::SmallVector<int>{0, 1, 2},
        //     /* keep_dim */ true,
        //     /* output_mem_config */ std::nullopt,
        //     /*compute_kernel_config */ core::ComputeKernelConfig::precise());  // [B,1,S,C] -> [1,1,1,C]
        // NOTE: To compute dL_dg, we need to process all batches. Therefore, we will compute here only dL_dg_components
        // for each tile, and then store them in CB. The reduction will be done in a separate program.
        const uint32_t a_over_rms_a_register = 0;
        const uint32_t rms_a_register = 1;
        tile_regs_acquire();

        copy_tile_init(cb_input_idx);
        copy_tile(cb_input_idx, /* tile_idx */ 0, /* register_idx */ a_over_rms_a_register);
        copy_tile_init(cb_rms_a_idx);
        copy_tile(cb_rms_a_idx, /* tile_idx */ 0, /* register_idx */ rms_a_register);
        div_binary_tile_init();
        div_binary_tile(a_over_rms_a_register, rms_a_register);
        // Now we can pack it and perform the multiplication on FPU.
        cb_reserve_back(cb_a_over_rms_a, onetile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_a_over_rms_a);
        pack_tile(a_over_rms_a_register, cb_a_over_rms_a);
        tile_regs_release();
        cb_push_back(cb_a_over_rms_a, onetile);
        // Now we can perform the multiplication with dL_out.
        const uint32_t dL_dg_components_register = 0;
        tile_regs_acquire();
        mul_tiles_init(cb_dL_out_idx, cb_a_over_rms_a);
        // We can use tile idx 0 for all cols as we do not need to store all of the a over rms_a values, but only the
        // current tile value.
        mul_tiles(cb_dL_out_idx, cb_a_over_rms_a, /* tile_idx */ 0, /* tile_idx */ 0, dL_dg_components_register);
        // Now we have dL_dg_components in dL_dg_components_register, we can pack it to cb_dL_dgamma_components.
        cb_reserve_back(cb_dL_dgamma_components, onetile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_dL_dgamma_components);
        pack_tile(dL_dg_components_register, cb_dL_dgamma_components);
        tile_regs_release();
        cb_push_back(cb_dL_dgamma_components, onetile);
        // We can pop_front cb_a_over_rms_a, since we do not need it anymore.
        cb_pop_front(cb_a_over_rms_a, onetile);

        // TODO Make sure that we wait and resever for all necessary data in buffers! (probably not)
        // TODO2 Make sure if we do not need to reconfigure data format for any calculation. I neglected it for now.
    }
}

}  // namespace NAMESPACE
