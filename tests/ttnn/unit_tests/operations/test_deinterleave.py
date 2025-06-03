# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn._ttnn
from models.utility_functions import comp_allclose_and_pcc, comp_pcc
from loguru import logger
from enum import Enum

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)


class DeinterleaveMode(Enum):
    DeinterleaveBatch = 1
    DeinterleaveLocal = 2


class InputMode(Enum):
    Random = 1
    Debug = 2


def print_diff_to_file(torch_tensor, ttnn_tensor, filename):
    import numpy as np

    torch_tensor_fp32 = torch_tensor.to(torch.float32).cpu().numpy().reshape(-1, torch_tensor.shape[-1])
    ttnn_tensor_fp32 = ttnn_tensor.to(torch.float32).cpu().numpy().reshape(-1, torch_tensor.shape[-1])
    np.savetxt(f"{filename}_torch.csv", torch_tensor_fp32, delimiter=",")
    np.savetxt(f"{filename}_ttnn.csv", ttnn_tensor_fp32, delimiter=",")


def torch_deinterleave_to_batch(torch_input_nhwc, stride_hw):
    torch_deinterleaved_to_batch = torch.zeros(
        torch_input_nhwc.shape[0] * stride_hw[0] * stride_hw[1],
        torch_input_nhwc.shape[1] // stride_hw[0],
        torch_input_nhwc.shape[2] // stride_hw[1],
        torch_input_nhwc.shape[3],
    )

    logger.info(f"----torch_deinterleaved_to_batch shape: {torch_deinterleaved_to_batch.shape}")
    logger.info(f"----torch_input_nhwc shape: {torch_input_nhwc.shape}")
    for src_batch in range(torch_input_nhwc.shape[0]):
        for split_h in range(stride_hw[0]):
            for split_w in range(stride_hw[1]):
                batch_idx = src_batch * stride_hw[0] * stride_hw[1] + split_h * stride_hw[1] + split_w
                torch_deinterleaved_to_batch[batch_idx, :, :, :] = torch_input_nhwc[
                    src_batch,
                    split_h :: stride_hw[0],
                    split_w :: stride_hw[1],
                    :,
                ]
    return torch_deinterleaved_to_batch


def run_deinterleave(
    device,
    mode: DeinterleaveMode,
    shape_nhwc,
    input_memory_config,
    stride_hw,
    barrier_threshold=0,
    input_mode: InputMode = InputMode.Random,
):
    input_dtype = "bfloat16"

    if input_mode == InputMode.Random:
        torch_input = 2 * torch.rand(size=shape_nhwc, dtype=get_lib_dtype(torch, input_dtype)) - 1
    else:
        torch_input = torch.ones(size=shape_nhwc, dtype=get_lib_dtype(torch, input_dtype))
        for h in range(stride_hw[0]):
            for w in range(stride_hw[1]):
                torch_input[:, h :: stride_hw[0], w :: stride_hw[1], :] = 10 * (
                    h * stride_hw[1] + w
                )  # 0.5 * (h + 1) * (w + 1)

    # move to 1,1, NHW, C as in conv2ds
    torch_input_view = torch_input.reshape(1, 1, shape_nhwc[0] * shape_nhwc[1] * shape_nhwc[2], shape_nhwc[3])
    logger.info(f"----torch_input_view.shape={torch_input_view.shape}")
    logger.info(f"----torch_input_view={torch_input_view}")

    ttnn_input = ttnn.from_torch(
        torch_input_view,
        device=device,
        dtype=get_lib_dtype(ttnn, input_dtype),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_memory_config,
    ).to(device)

    compute_kernel_options = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    logger.info(f"----shard_shape {input_memory_config.shard_spec.shape}")
    logger.info(f"----shard_spec mode {input_memory_config.shard_spec.mode}")

    if mode == DeinterleaveMode.DeinterleaveBatch:
        golden_output = torch_deinterleave_to_batch(torch_input, stride_hw)

        ttnn_output = ttnn.experimental.deinterleave_to_batch(
            ttnn_input,
            compute_kernel_config=compute_kernel_options,
            stride_hw=stride_hw,
            input_height=shape_nhwc[1],
            input_width=shape_nhwc[2],
            barrier_threshold=barrier_threshold,
        )

        torch_output = ttnn.to_torch(ttnn_output)

        logger.info(f"----ttnn_output shape={ttnn_output.shape}")

        torch_output = torch_output.view(
            shape_nhwc[0] * stride_hw[0] * stride_hw[1],
            shape_nhwc[1] // stride_hw[0],
            shape_nhwc[2] // stride_hw[1],
            shape_nhwc[3],
        )

        logger.info(f"----golden_shape={golden_output.shape}")
        logger.info(f"----torch_shape={torch_output.shape}")

        if input_mode == InputMode.Debug:
            print_diff_to_file(
                golden_output,
                torch_output,
                f"deinterleave_to_batch_diff_{shape_nhwc[0]}_{shape_nhwc[1]}_{shape_nhwc[2]}_{shape_nhwc[3]}",
            )

        passing, out = comp_allclose_and_pcc(golden_output, torch_output, rtol=0.01, atol=0.01, pcc=0.999)
        logger.info(out)
        assert passing, out
    else:
        # Not implemented yet
        pass


@pytest.mark.parametrize(
    "shape, core_grid, stride_hw",
    [
        ([1, 1024, 128, 32], ttnn.CoreGrid(x=8, y=8), [2, 2]),
        ([1, 1024, 128, 48], ttnn.CoreGrid(x=8, y=8), [4, 4]),
        ([1, 1024, 128, 56], ttnn.CoreGrid(x=8, y=8), [8, 8]),
    ],
)
@pytest.mark.parametrize("deinterleave_mode", [DeinterleaveMode.DeinterleaveBatch])
def test_deinterleave_shape(
    device,
    shape,
    core_grid,
    stride_hw,
    deinterleave_mode,
    barrier_threshold=0,
):
    torch.manual_seed(2025)

    memory_config = ttnn.create_sharded_memory_config_(
        shape=[shape[0] * shape[1] * shape[2], shape[3]],
        core_grid=core_grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        strategy=ttnn.ShardStrategy.HEIGHT,
    )

    logger.info(f"Memory config: {memory_config}")

    run_deinterleave(device, deinterleave_mode, shape, memory_config, stride_hw, barrier_threshold)
