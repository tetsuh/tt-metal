# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import requests
import torch
from ultralytics import YOLO

from models.demos.yolov9c.reference.yolov9c import YoloV9


def load_coco_class_names():
    url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    path = f"models/demos/yolov4/demo/coco.names"
    response = requests.get(url)
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip().split("\n")
    except requests.RequestException:
        pass
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    raise Exception("Failed to fetch COCO class names from both online and local sources.")


def load_torch_model(use_weights_from_ultralytics=True, module=None, model_task="segment"):
    state_dict = None

    weights = "yolov9c-seg.pt" if model_task == "segment" else "yolov9c.pt"
    enable_segment = model_task == "segment"

    if use_weights_from_ultralytics:
        torch_model = YOLO(weights)  # Use "yolov9c.pt" weight for detection
        torch_model.eval()
        state_dict = torch_model.state_dict()

    model = YoloV9(enable_segment=enable_segment)
    state_dict = model.state_dict() if state_dict is None else state_dict

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2

    model.load_state_dict(new_state_dict)
    model.eval()

    return model
