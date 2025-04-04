# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

from tests.ttnn.unit_tests.operations.ccl.test_all_gather_TG_post_commit import (
    run_line_all_gather_on_TG_with_mesh_tensor_along_rows,
)
from tests.ttnn.unit_tests.operations.ccl.test_ccl_async_TG_llama import (
    PREFETCHER_NOC1_RING,
    get_core_range_set,
)
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

USE_NON_FUSED = False
USE_LEGACY_ALLGATHER = True


def run_all_gather_impl(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    matmul_output_dim,
    matmul_weights_dtype,
    max_in0_block_w,
    use_bias,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    all_gather_topology,
    mem_config_weights=None,
    num_iters=1,
    trace_mode=False,
    create_persistent_fabric=True,
    teardown_persistent_fabric=True,
    wrap_fabric_around_mesh=False,
    enable_async=False,
):
    tile = (32, 32)

    # Set the default config
    if mem_config_weights is None:
        mem_config_weights = mem_config_ag

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape, dim, mem_config_ag, num_devices, num_links, ag_input_dtype, layout, tile
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    devices = t3k_mesh_device.get_devices()

    if not USE_LEGACY_ALLGATHER:
        enable_persistent_fabric = True
        if num_iters < 1:
            pytest.fail("num_iters must be >= 1")
        # Use Async mode based on test input config
        t3k_mesh_device.enable_async(enable_async)
        if enable_async:
            logger.info(f"Using Async Mode for All Gather Op Dispatch")

        ##### All gather setup #####
        compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
        if USE_NON_FUSED:
            ccl_sub_device_crs = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
            )
        worker_sub_device = ttnn.SubDevice(
            [
                ccl_sub_device_crs,
            ]
        )
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_stall_group = [worker_sub_device_id]
        if create_persistent_fabric:
            mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
                t3k_mesh_device,
                [worker_sub_device],
                0,
                0,
                enable_persistent_fabric,
                wrap_fabric_around_mesh=wrap_fabric_around_mesh,
            )
            t3k_mesh_device.set_sub_device_stall_group(sub_device_stall_group)

        # create global semaphore handles
        ccl_semaphore_handles = [
            create_global_semaphore_with_same_address(t3k_mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
        ]

    ##### All gather input setup #####
    logger.info(f"All gather output shape: {ag_output_shape}")
    logger.info(f"All gather dim: {dim}")

    input_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []
    _, _, _, hidden_dim = ag_output_shape

    for i in range(num_iters):
        ag_output_tensor = torch.rand(ag_output_shape).bfloat16()
        ag_output_tensor_goldens_list.append(ag_output_tensor)
        input_tensors = torch.chunk(ag_output_tensor, num_devices, dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(ttnn.Tensor(t, ag_input_dtype).to(layout).to(devices[i], mem_config_input))
        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

        input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Matmul weight setup #####
    if use_bias:
        weights_tensor = torch.randn([hidden_dim, matmul_output_dim * num_devices]).bfloat16()
        weights_tensor_padded = weights_tensor.unsqueeze(0).unsqueeze(0)
    else:
        weights_tensor = torch.randn([1, 1, hidden_dim, matmul_output_dim * num_devices]).bfloat16()
        weights_tensor_padded = weights_tensor
    weight_tt = ttnn.from_torch(
        weights_tensor_padded,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config_weights,
        mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=dim),
    )

    if use_bias:
        bias_tensor = torch.randn([1, matmul_output_dim * num_devices]).float()
        bias_tensor_padded = bias_tensor.unsqueeze(0).unsqueeze(0)
        bias_tt = ttnn.from_torch(
            bias_tensor_padded,
            dtype=matmul_weights_dtype,
            layout=layout,
            device=t3k_mesh_device,
            memory_config=mem_config_weights,
            mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=dim),
            tile=ttnn.Tile(tile),
        )
    else:
        bias_tt = None

    ##### Configs for ttnn.matmul #####
    if USE_NON_FUSED:
        core_grid = (8, 8)
    else:
        core_grid = (8, 4)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=min(max_in0_block_w, hidden_dim // 32 // core_grid[0]),  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(1, math.ceil(ag_output_shape[2] / 32 / core_grid[1])),  # M / TILE_HEIGHT / Grid_Size
        per_core_N=max(1, math.ceil(matmul_output_dim / 32 / core_grid[0])),  # N / TILE_WIDTH / Grid_Size
        transpose_mcast=False,
        fused_activation=None,  # ttnn.UnaryOpType.SILU,
        fuse_batch=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    ##### Perform torch ops #####
    torch_matmul_output_list = []
    for i in range(num_iters):
        if use_bias:
            matmul_output = torch.nn.functional.linear(
                ag_output_tensor_goldens_list[i], weights_tensor.T.contiguous(), bias_tensor
            )
        else:
            matmul_output = torch.matmul(ag_output_tensor_goldens_list[i], weights_tensor)
        torch_matmul_output_list.append(matmul_output)

    ##### Perform the TT ops #####
    tt_matmul_out_tensor_list = []
    for i in range(num_iters):
        if USE_NON_FUSED:
            if USE_LEGACY_ALLGATHER:
                tt_all_gather_out_tensor = ttnn.all_gather(
                    input_tensor_mesh_list[i],
                    dim,
                    num_links=num_links,
                    memory_config=mem_config_ag,
                )
            else:
                tt_all_gather_out_tensor = ttnn.experimental.all_gather_async(
                    input_tensor_mesh_list[i],
                    dim,
                    multi_device_global_semaphore=ccl_semaphore_handles[i],
                    num_links=num_links,
                    memory_config=mem_config_ag,
                    topology=all_gather_topology,
                    subdevice_id=worker_sub_device_id,
                    enable_persistent_fabric_mode=enable_persistent_fabric,
                )

            tt_matmul_out_tensor = ttnn.linear(
                tt_all_gather_out_tensor,
                weight_tt,
                bias=bias_tt,
                memory_config=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
        else:
            if USE_LEGACY_ALLGATHER:
                _, tt_matmul_out_tensor, _ = ttnn.experimental.all_gather_matmul(
                    input_tensor_mesh,
                    weight_tt,
                    dim,
                    (0, 4),
                    bias=bias_tt,
                    num_links=num_links,
                    memory_config_ag=mem_config_ag,
                    memory_config_mm=mem_config_mm,
                    program_config=program_config,
                    compute_kernel_config=compute_kernel_config,
                )
            else:
                assert True, "FUSED w/ FABRIC ALLGATHER not implemented"

        tt_matmul_out_tensor_list.append(tt_matmul_out_tensor)

    if not USE_LEGACY_ALLGATHER:
        logger.info(f"Waiting for op")
        ttnn.synchronize_devices(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    passed = True
    for i in range(num_iters):
        tt_out_tensor = tt_matmul_out_tensor_list[i]
        torch_out_tensor = torch_matmul_output_list[i]

        tt_mm_out = ttnn.from_device(tt_out_tensor)
        tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))
        eq, output = comp_pcc(tt_mm_out, torch_out_tensor)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED: {output}"

    if not USE_LEGACY_ALLGATHER:
        if enable_persistent_fabric and teardown_persistent_fabric:
            t3k_mesh_device.reset_sub_device_stall_group()
            teardown_fabric_interface(t3k_mesh_device)


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype",
    [
        (
            8,
            1,
            [1, 1, 4096, 2560],
            3,
            ttnn.TILE_LAYOUT,
            960,
            2,
            ttnn.bfloat16,
        ),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "use_bias",
    [
        True,
        #        False,
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag, mem_config_mm",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_async",
    [
        #        True,
        False,
    ],
)
def test_all_gather(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    matmul_output_dim,
    matmul_weights_dtype,
    max_in0_block_w,
    use_bias,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    enable_async,
):
    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        matmul_output_dim,
        matmul_weights_dtype,
        max_in0_block_w,
        use_bias,
        mem_config_input,
        mem_config_ag,
        mem_config_mm,
        all_gather_topology=ttnn.Topology.Linear,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    )
