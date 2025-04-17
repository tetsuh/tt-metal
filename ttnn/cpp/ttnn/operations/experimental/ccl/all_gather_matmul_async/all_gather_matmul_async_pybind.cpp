// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_matmul_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

void py_bind_all_gather_matmul_async(pybind11::module& module) {
    namespace py = pybind11;

    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::all_gather_matmul_async,
        R"doc(all_gather_matmul_async(input_tensor: ttnn.Tensor, weight_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> (ttnn.Tensor, ttnn.Tensor)

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`weight_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)
            * :attr:`all_gather_core_grid_offset` (ttnn.CoreCoord): Core grid offset for the all-gather operation.

        Keyword Args:
            * :attr:`bias` (ttnn.Tensor): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config_ag` (Optional[ttnn.MemoryConfig]): Memory configuration for the All Gather operation.
            * :attr:`memory_config_mm` (Optional[ttnn.MemoryConfig]): Memory configuration for the Matmul operation.
            * :attr:`transpose_a` (bool)
            * :attr:`transpose_b` (bool)
            * :attr:`dtype` (Optional[DataType])
            * :attr:`program_config` (Optional[ttnn.MatmulProgramConfig])
            * :attr:`activation` (Optional[str])
            * :attr:`compute_kernel_config` (Optional[DeviceComputeKernelConfig])
            * :attr:`core_grid` (Optional[ttnn.CoreGrid])

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> weight_tensor = ttnn.from_torch(torch.tensor((2, 1), dtype=torch.bfloat16), device=device)
            >>> all_gathered_mm_in, mm_out = ttnn.all_gather_matmul_async(tensor, weight_tensor, dim=0, (0, 0))

        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"),
            py::arg("weight_tensor"),
            py::arg("dim"),
            py::arg("multi_device_global_semaphore"),
            py::arg("all_gather_core_grid_offset"),
            py::kw_only(),
            py::arg("bias") = std::nullopt,
            py::arg("num_links") = 1,
            py::arg("memory_config_ag") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("enable_persistent_fabric_mode") = false,
            py::arg("memory_config_mm") = std::nullopt,
            py::arg("transpose_a") = false,
            py::arg("transpose_b") = false,
            py::arg("dtype") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("activation") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("core_grid") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::ccl
