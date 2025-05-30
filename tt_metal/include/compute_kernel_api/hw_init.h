// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"

#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#endif

namespace ckernel {

// clang-format off
/**
* Performs the necessary hardware initialization for all operations that will follow. Needs to be called once, before op-specific initialization function,
* e.g. reduce_init, tilize_init, etc. Meant to be called only once at the beginning of the compute kernel.
*
* Return value: None
*
* | Function Argument | Description                                                     | Type     | Valid Range                                    | Required |
* |-------------------|-----------------------------------------------------------------|----------|------------------------------------------------|----------|
* | icb0              | The identifier of the circular buffer (CB) containing operand A | uint32_t | 0 to 31                                        | True     |
* | icb1              | The identifier of the circular buffer (CB) containing operand B | uint32_t | 0 to 31                                        | True     |
* | ocb               | The identifier of the output circular buffer (CB)               | uint32_t | 0 to 31                                        | True     |
*/
// clang-format on
ALWI void hw_start_init(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated<false /*untilize*/, false /*skip_inputs*/>(icb0, icb1)));

    PACK((llk_pack_init<false /*untilize*/, false /*zero_output*/, false /*tilize*/>(ocb)));
    PACK((llk_pack_hw_configure_disaggregated<
          false /*untilize*/,
          DST_ACCUM_MODE,
          ReluType::NO_RELU,
          0 /*relu_treshold*/,
          false /*tilize*/>(ocb)));
    PACK((llk_pack_dest_init<false /*untilize*/, DST_ACCUM_MODE>(ocb)));
}

}  // namespace ckernel
