# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.detr3d.ttnn.ttnn_generic_mlp import TttnnGenericMLP
from models.experimental.detr3d.reference.detr3d_model import GenericMLP
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


@pytest.mark.parametrize(
    "input_dim,hidden_dims,output_dim,norm_fn_name,activation,use_conv,"
    "hidden_use_bias,output_use_bias,output_use_activation,output_use_norm,"
    "weight_init_name,dropout,x_shape",
    [
        (256, [256], 256, "bn1d", "relu", True, False, False, True, True, None, None, (1, 256, 1024)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_generic_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    norm_fn_name,
    activation,
    use_conv,
    hidden_use_bias,
    output_use_bias,
    output_use_activation,
    output_use_norm,
    weight_init_name,
    dropout,
    x_shape,
    device,
):
    torch_model = GenericMLP(
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name,
        activation,
        use_conv,
        dropout,
        hidden_use_bias,
        output_use_bias,
        output_use_activation,
        output_use_norm,
        weight_init_name,
    ).to(torch.bfloat16)
    torch_model.eval()
    x = torch.randn(x_shape, dtype=torch.bfloat16)
    torch_out = torch_model(x)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        # custom_preprocessor=custom_preprocessor,
        device=device,
    )
    print(
        "running var and mean values", torch_model.layers[1].running_mean.shape, torch_model.layers[1].running_var.shape
    )
    # ss
    print("param are", parameters)
    ttnn_model = TttnnGenericMLP(torch_model, parameters, device)
    ttnn_x = ttnn.from_torch(
        x.permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    p(ttnn_x, "ttnn input")
    ttnn_out = ttnn_model(ttnn_x)
    print("outputs are", ttnn_out.shape, torch_out.shape)
    assert_with_pcc(torch_out, ttnn.to_torch(ttnn_out).squeeze(dim=0).permute(0, 2, 1), 1.0)
