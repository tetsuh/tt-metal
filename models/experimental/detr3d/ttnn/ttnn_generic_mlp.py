# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.detr3d.ttnn.common import TtnnConv1D


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TttnnGenericMLP:
    def __init__(self, module, parameters, device):
        self.device = device
        self.parameters = parameters
        self.module = module
        print("module is", module.layers[0])
        print("parameters is", parameters)
        self.conv1 = TtnnConv1D(module.layers[0], parameters.layers[0], device)
        self.conv2 = TtnnConv1D(module.layers[3], parameters.layers[3], device)
        self.relu = ttnn.relu

    def __call__(self, input):
        out = self.conv1(input)
        p(out, "conv1 out")
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)

        out = ttnn.to_torch(out).squeeze(dim=0).permute(0, 2, 1)
        out = self.module.layers[1](out).unsqueeze(dim=0).permute(0, 1, 3, 2)
        out = ttnn.from_torch(
            out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        out = self.relu(out)
        out = self.conv2(out)
        p(out, "conv2 out")
        out = ttnn.to_torch(out).squeeze(dim=0).permute(0, 2, 1)
        out = self.module.layers[4](out).unsqueeze(dim=0).permute(0, 1, 3, 2)
        # out = torch.nn.BatchNorm1d(256,eps =self.module.layers[4].eps,momentum=self.module.layers[4].momentum)(out).unsqueeze(dim=0).permute(0,1,3,2)
        out = ttnn.from_torch(
            out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        out = self.relu(out)
        return out
        # ttnn bn
        # running_mean_tt = ttnn.from_torch(self.module.layers[1].running_mean,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=self.device)
        # running_var_tt = ttnn.from_torch(self.module.layers[1].running_var,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=self.device)
        # wt = ttnn.to_layout(self.parameters.layers[1].weight,layout=ttnn.TILE_LAYOUT)
        # bs = ttnn.to_layout(self.parameters.layers[1].bias,layout=ttnn.TILE_LAYOUT)
        # p(out,"bn1d input")
        # out = ttnn.batch_norm(out,weight=wt,bias=bs, running_mean=running_mean_tt,
        # running_var=running_var_tt)
