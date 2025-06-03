# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtnnConv1D:
    def __init__(
        self,
        conv,
        parameters,
        device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=None,  # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        fp32_accum=False,
        packer_l1_acc=False,
        deallocate_activation=True,
        math_fidelity=ttnn.MathFidelity.LoFi,
    ):
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size[0]
        self.padding = conv.padding[0]
        self.stride = conv.stride[0]
        self.groups = conv.groups
        self.conv_config = ttnn.Conv1dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_accum,
            packer_l1_acc=packer_l1_acc,
        )
        if "bias" in parameters and parameters["bias"] is not None:
            bias = ttnn.from_device(parameters.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(parameters.weight)
        self.weight = weight

    def __call__(self, x):
        input_length = x.shape[1]  # self.conv.input_length
        batch_size = 1  # self.conv.batch_size
        [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch_size,
            input_length=input_length,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return tt_output_tensor_on_device
