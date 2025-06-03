// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <array>
#include <memory>
#include <optional>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

#include "ttnn/core.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp"
// #include "ttnn/operations/copy.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/quantization/quantization.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/unit_tests/gtests/utils.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

::ttnn::Tensor forward(::ttnn::Tensor v1, ::ttnn::Tensor v2, ttnn::IDevice* v3) {
    DeviceComputeKernelConfig compute_config =
        init_device_compute_kernel_config(tt::ARCH::WORMHOLE_B0, std::nullopt, MathFidelity::HiFi4, true, true, true);
    // ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
    ::ttnn::Tensor v4 = ttnn::transpose(
        v1, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v1, false);
    ::ttnn::Tensor v5 = ttnn::transpose(
        v4, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v4, false);
    ::ttnn::Tensor v6 = ttnn::reshape(v5, ::std::vector<int32_t>{1, 1, 50176, 3}, ::std::nullopt);
    ttnn::deallocate(v5, false);
    ::ttnn::Tensor v7 = ttnn::from_device(v6);
    ttnn::deallocate(v6, false);
    ::ttnn::Tensor v8 = ttnn::to_layout(
        v7,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY},
        static_cast<::ttnn::distributed::MeshDevice*>(nullptr));
    ttnn::deallocate(v7, false);
    ::ttnn::Tensor v9 = ttnn::to_device(
        v8, v3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v8, false);
    ::std::variant<
        ::ttnn::Tensor,
        ::std::tuple<::ttnn::Tensor, ::std::tuple<uint32_t, uint32_t>>,
        ::std::tuple<::ttnn::Tensor, ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>,
        ::std::tuple<
            ::ttnn::Tensor,
            ::std::tuple<uint32_t, uint32_t>,
            ::std::tuple<::ttnn::Tensor, ::std::optional<::ttnn::Tensor>>>>
        v10 = ttnn::conv2d(
            v9,
            v2,
            v3,
            3,
            768,
            1,
            224,
            224,
            ::std::array<uint32_t, 2>{16, 16},
            ::std::array<uint32_t, 2>{16, 16},
            ::std::array<uint32_t, 2>{0, 0},
            ::std::array<uint32_t, 2>{1, 1},
            1,
            ::std::nullopt,
            ::ttnn::operations::conv::conv2d::Conv2dConfig{
                .dtype = ::ttnn::DataType::FLOAT32,
                .weights_dtype = ::ttnn::DataType::FLOAT32,
                .always_preprocess_weights = true},
            compute_config,
            ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v11 = ::std::get<0>(v10);
    ttnn::deallocate(v9, false);
    ttnn::deallocate(v2, false);
    ::ttnn::Tensor v12 = ttnn::reshape(v11, ::std::vector<int32_t>{1, 14, 14, 768}, ::std::nullopt);
    ttnn::deallocate(v11, false);
    ::ttnn::Tensor v13 = ttnn::transpose(
        v12, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v12, false);
    ::ttnn::Tensor v14 = ttnn::transpose(
        v13, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ttnn::deallocate(v13, false);
    return v14;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_forward(ttnn::IDevice* v1) {
    // ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
    ::ttnn::Tensor v2 = ttnn::ones(
        ::ttnn::Shape({1, 3, 224, 224}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v3 = ttnn::to_device(
        v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v4 = ttnn::ones(
        ::ttnn::Shape({768, 3, 16, 16}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::ROW_MAJOR,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
    return std::make_tuple(v3, v4);
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor> create_rand_inputs_for_forward(ttnn::IDevice* v1) {
    // ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
    ::ttnn::Tensor v2 = ttnn::random::random(::ttnn::Shape({1, 3, 224, 224})).to_layout(Layout::TILE);
    // ::ttnn::DataType::FLOAT32,
    // ::ttnn::Layout::TILE,
    // ::std::nullopt,
    // ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v3 = ttnn::to_device(
        v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    ::ttnn::Tensor v4 = ttnn::random::random(::ttnn::Shape({768, 3, 16, 16})).to_layout(Layout::ROW_MAJOR);
    // ::ttnn::DataType::FLOAT32,
    // ::ttnn::Layout::ROW_MAJOR,
    // ::std::nullopt,
    // ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
    return std::make_tuple(v3, v4);
}

TEST(TTNNGraphRepro, TTNNGraphReproERROR) {
    const chip_id_t device_id = 0;
    const size_t L1_small_size = 32768;

    IDevice* device = CreateDevice(device_id, 1, L1_small_size);

    ::ttnn::Tensor input_tensor_device;
    ::ttnn::Tensor weights_tensor;
    std::tie(input_tensor_device, weights_tensor) = create_rand_inputs_for_forward(device);

    // Store host-side copies for reference
    ::ttnn::Tensor weights_tensor_host = weights_tensor;  // already on host

    std::cout << "input_tensor_device shape: " << input_tensor_device.get_logical_shape() << std::endl;
    std::cout << "weights_tensor_host shape: " << weights_tensor_host.get_logical_shape() << std::endl;
    ::ttnn::Tensor input_tensor_host = ttnn::from_device(input_tensor_device);
    std::vector<float> input_vector = input_tensor_host.to_vector<float>();
    std::vector<float> weights_vector = weights_tensor_host.to_vector<float>();

    ::ttnn::Tensor output_tensor_device = forward(input_tensor_device, weights_tensor, device);
    ::ttnn::Tensor output_tensor_host = ttnn::from_device(output_tensor_device);

    std::vector<float> output_vector = output_tensor_host.to_vector<float>();
    std::vector<float> reference_output = conv::conv2d::test::reference_implementation_conv2d(
        input_vector,
        weights_vector,
        3,         // input_channels
        768,       // output_channels
        1,         // batch_size
        224,       // input_height
        224,       // input_width
        {16, 16},  // kernel_size
        {16, 16},  // stride
        {0, 0}     // padding
    );
    //  print output reference output vector element by element
    std::cout << "Reference output vector: " << std::endl;
    for (const auto& val : reference_output) {
        std::cout << val << " ";
        break;
    }
    // print output vector element by element
    std::cout << "Output vector: " << std::endl;
    for (const auto& val : output_vector) {
        std::cout << val << " ";
        break;
    }
    std::cout << "PCC: " << test_utils::pcc(output_vector, reference_output) << std::endl;

    EXPECT_GT(test_utils::pcc(output_vector, reference_output), 0.99);

    CloseDevice(device);
}

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
