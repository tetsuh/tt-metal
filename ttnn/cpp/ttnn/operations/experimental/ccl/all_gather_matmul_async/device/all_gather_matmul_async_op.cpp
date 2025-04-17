// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_op.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

/* All Gather Matmul fusion includes */
#include "cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "cpp/ttnn/operations/matmul/matmul.hpp"

namespace ttnn {
namespace ccl {
namespace all_gather_matmul_async_detail {

AllGatherMatmulAsync create_all_gather_matmul_async_struct(
    const ttnn::AllGatherAsync& all_gather_struct_input,
    const operations::matmul::Matmul& matmul_struct_input,
    const CoreCoord all_gather_core_grid_offset) {
    return ttnn::AllGatherMatmulAsync{all_gather_struct_input, matmul_struct_input, all_gather_core_grid_offset};
}

}  // namespace all_gather_matmul_async_detail
}  // namespace ccl

void AllGatherMatmulAsync::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 3, "AllGatherMatmulAsync requires 3 input tensors: [input, weight, all_gather_output]");

    auto& input_tensor = input_tensors[0];
    auto& all_gather_output_tensor = input_tensors[1];
    auto& weight_tensor = input_tensors[2];

    // All Gather validate
    this->all_gather_async_struct.validate({input_tensor});

    // Matmul validate.
    this->matmul_struct.validate({all_gather_output_tensor, weight_tensor}, optional_input_tensors, {});

    // All Gather Matmul validate
    TT_FATAL(
        this->all_gather_async_struct.dim == 3, "AllGatherMatmulAsync requires dim=3 for the AllGather operaitons.");
    TT_FATAL(
        input_tensor.get_padded_shape()[0] == 1 && input_tensor.get_padded_shape()[1] == 1,
        "AllGatherMatmulAsync requires input tensor to have batch size of 1.");
    std::visit(
        [&](const auto& config) {
            using ProgramConfigType = std::decay_t<decltype(config)>;
            if (not(std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
                    std::
                        is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>)) {
                TT_THROW(
                    "Unsupported MatmulProgramConfig type for AllGatherMatmulAsync. Needs to be 1D or 2D Multicast.");
            }
        },
        this->matmul_struct.program_config.value());

    const auto& all_gather_output_tensor_shard_spec = all_gather_output_tensor.shard_spec();
    if (all_gather_output_tensor_shard_spec.has_value()) {
        const auto& shard_grid = all_gather_output_tensor_shard_spec->grid.bounding_box();
        const auto& shard_grid_start = shard_grid.start_coord;
        const auto& shard_grid_end = shard_grid.end_coord;
        const uint32_t num_all_gather_output_shards = shard_builder::get_sharding_core_count(all_gather_output_tensor);
        TT_FATAL(
            this->all_gather_async_struct.ring_size == num_all_gather_output_shards,
            "AllGatherMatmulAsync requires number of tensor slices to equal the number of output shards of the "
            "all_gather.");
    }
}

std::vector<ttnn::TensorSpec> AllGatherMatmulAsync::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    // All Gather shape
    ttnn::TensorSpec all_gather_output_shape =
        this->all_gather_async_struct.compute_output_specs({input_tensors[0]})[0];

    // Matmul shape
    ttnn::TensorSpec matmul_output_specs =
        this->matmul_struct.compute_output_specs({input_tensors[1], input_tensors[2]}, {})[0];

    return {all_gather_output_shape, matmul_output_specs};
}

std::vector<Tensor> AllGatherMatmulAsync::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // All Gather output tensor
    auto& all_gather_output_tensor =
        input_tensors[1];  // this->all_gather_out_tensor =
                           // this->all_gather_async_struct.create_output_tensors(input_tensors)[0];

    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor =
        this->matmul_struct.create_output_tensors({input_tensors[1], input_tensors[2]})[0];

    return {all_gather_output_tensor, matmul_output_tensor};
}

operation::ProgramWithCallbacks AllGatherMatmulAsync::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    // Return the AllGatherMatmulAsync program with callbacks
    //  printf("1111111111111111111111 MATMUL ASYNC CREATE_PROGRAM %d\n",this->all_gather_async_struct.ring_index);
    return all_gather_matmul_async_multi_core_with_workers(
        input_tensors[0],   // input_tensor
        output_tensors[0],  // all_gather_output_tensor
        input_tensors[2],   // weight_tensor
        output_tensors[1],  // matmul_output_tensor

        /* All Gather Params */
        this->all_gather_async_struct.forward_device,
        this->all_gather_async_struct.backward_device,
        this->all_gather_async_struct.dim,
        this->all_gather_async_struct.num_links,
        this->all_gather_async_struct.ring_size,
        this->all_gather_async_struct.ring_index,
        this->all_gather_async_struct.topology,
        this->all_gather_async_struct.semaphore,
        this->all_gather_async_struct.sub_device_id,
        this->all_gather_async_struct.enable_persistent_fabric_mode,
        this->all_gather_core_grid_offset,

        /* Matmul Params */
        optional_input_tensors[0],  // Bias
        this->matmul_struct.bcast_batch.value(),
        this->matmul_struct.compute_kernel_config.value(),
        this->matmul_struct.program_config.value(),
        this->matmul_struct.untilize_out);
}

namespace operations {
namespace experimental {
namespace ccl {

std::vector<ttnn::Tensor> all_gather_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const uint32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const CoreCoord all_gather_core_grid_offset,
    const std::optional<const Tensor>& bias,
    const uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config_ag,
    const ttnn::ccl::Topology topology,
    std::optional<SubDeviceId> sub_device_id,
    bool enable_persistent_fabric_mode,
    const std::optional<MemoryConfig>& memory_config_mm,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    const std::optional<const std::string>& activation,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::CoreGrid> core_grid) {
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "AllGatherMatmulAsync is only supported for Fast Dispatch");

    std::vector<std::optional<const Tensor>> optional_input_tensors = {};
    std::vector<Tensor> output_tensors;
    auto devices = input_tensor.get_workers();
    if (bias.has_value()) {
        optional_input_tensors.push_back(bias.value());
        output_tensors = {
            ttnn::Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor}, {bias.value()})),
            ttnn::Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor}, {bias.value()}))};
    } else {
        optional_input_tensors.push_back(std::nullopt);
        output_tensors = {
            ttnn::Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor})),
            ttnn::Tensor(operation::get_workers_for_op_output({input_tensor, weight_tensor}))};
    }

    operation::launch_op(
        [dim,
         all_gather_core_grid_offset,
         num_links,
         memory_config_ag,
         topology,
         multi_device_global_semaphore,
         sub_device_id,
         enable_persistent_fabric_mode,
         memory_config_mm,
         transpose_a,
         transpose_b,
         dtype,
         program_config,
         activation,
         compute_kernel_config,
         core_grid,
         devices](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& input_tensor = input_tensors[0];
            const auto& weight_tensor = input_tensors[1];

            /* AllGather setup */
            ttnn::AllGatherAsync all_gather_async_struct =
                ttnn::ccl::all_gather_async_detail::create_all_gather_async_struct(
                    input_tensor,
                    dim,
                    num_links,
                    memory_config_ag,
                    devices,
                    topology,
                    multi_device_global_semaphore,
                    sub_device_id,
                    enable_persistent_fabric_mode);

            // Create the all gather output tensor used as input (activation) to the matmul
            ttnn::Tensor all_gather_out_tensor = all_gather_async_struct.create_output_tensors({input_tensor})[0];

            /* Matmul setup */
            bool user_run_batched =
                ttnn::operations::matmul::detail::is_input_batched(weight_tensor.get_logical_shape());
            std::optional<CoreCoord> user_core_coord;
            if (core_grid.has_value()) {
                user_core_coord = CoreCoord(core_grid->x, core_grid->y);
            }

            operations::matmul::Matmul matmul_struct = operations::matmul::create_matmul_struct(
                all_gather_out_tensor,
                weight_tensor,
                /*parameters=*/
                operations::matmul::Matmul{
                    program_config,
                    /*bcast_batch=*/std::nullopt,
                    memory_config_mm.value_or(input_tensor.memory_config()),
                    dtype.value_or(input_tensor.get_dtype()),
                    compute_kernel_config,
                    /*untilize_out=*/false,
                    user_core_coord,
                    ttnn::operations::matmul::get_fused_activation(activation),
                    user_run_batched,
                    transpose_a,
                    transpose_b,
                    /*output_tile=*/std::nullopt,
                    /*global_cb=*/std::nullopt});

            return operation::run(
                ttnn::ccl::all_gather_matmul_async_detail::create_all_gather_matmul_async_struct(
                    /* All Gather Params */
                    all_gather_async_struct,
                    /* Matmul params */
                    matmul_struct,
                    /* Fusion params */
                    all_gather_core_grid_offset),
                {input_tensor, all_gather_out_tensor, weight_tensor},
                optional_input_tensors);
        },
        {input_tensor, weight_tensor},
        output_tensors,
        optional_input_tensors);
    return {output_tensors[0], output_tensors[1]};
}

}  // namespace ccl
}  // namespace experimental
}  // namespace operations

}  // namespace ttnn
