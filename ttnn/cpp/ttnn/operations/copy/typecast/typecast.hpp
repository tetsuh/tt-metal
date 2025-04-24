// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace copy {

struct Typecast {
    static Tensor invoke(
        const QueueId queue_id,
        const Tensor& input,
        const DataType& output_dtype,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static ttnn::Tensor invoke(
        const QueueId queue_id,
        const Tensor& input_tensor,
        const DataType& tt_input_dtype,
        const DataType& tt_output_dtype,
        const std::optional<MemoryConfig>& memory_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};
}  // namespace copy
}  // namespace operations

<<<<<<< HEAD
<<<<<<< HEAD
constexpr auto typecast = ttnn::register_operation<"ttnn::typecast", ttnn::operations::copy::Typecast>();
=======
constexpr auto typecast =
    ttnn::register_operation_with_auto_launch_op<"ttnn::typecast", ttnn::operations::copy::Typecast>();
>>>>>>> 555dc22387 (21127: typecast device op with sub_core_grids)
=======
constexpr auto typecast = ttnn::register_operation<"ttnn::typecast", ttnn::operations::copy::Typecast>();
>>>>>>> 87c8fe6fa5 (#21127: rebased)

}  // namespace ttnn
