# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov9c.demo.demo import save_seg_predictions_by_model
from models.demos.yolov9c.demo.demo_seg_utils import postprocess, preprocess
from models.demos.yolov9c.demo.demo_utils import load_coco_class_names
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner
from models.experimental.yolo_evaluation.yolo_common_evaluation import save_yolo_predictions_by_model
from models.experimental.yolo_evaluation.yolo_evaluation_utils import LoadImages
from models.experimental.yolo_evaluation.yolo_evaluation_utils import postprocess as obj_postprocess
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "source",
    [
        "models/demos/yolov9c/demo/image.png",
        # "models/sample_data/huggingface_cat_image.jpg", # Uncomment to run the demo with another image.
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",  # To run the demo with pre-trained weights
        # "False", # Uncomment to the run demo with random weights
    ],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",  # To run the demo for instance segmentation
        "detect",  # To run the demo for Object Detection
    ],
)
def test_demo_performant(
    device,
    source,
    use_weights_from_ultralytics,
    model_task,
    use_program_cache,
    reset_seeds,
    batch_size,
    act_dtype,
    weight_dtype,
):
    disable_persistent_kernel_cache()

    weights = "yolov9c-seg.pt" if model_task == "segment" else "yolov9c.pt"
    enable_segment = model_task == "segment"
    save_dir = "models/demos/yolov9c/demo/demo_trace_runs"
    dataset = LoadImages(path=source)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, res=(640, 640))
        performant_runner = YOLOv9PerformantRunner(
            device,
            batch_size,
            act_dtype,
            weight_dtype,
            model_task=model_task,
            resolution=(640, 640),
            model_location_generator=None,
            torch_input_tensor=im,
        )
        performant_runner._capture_yolov9_trace_2cqs()
        logger.info("Inferencing [TTNN] Model")

        preds = performant_runner.run(torch_input_tensor=im)
        preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)

        if enable_segment:
            detect1_out, detect2_out, detect3_out = [
                ttnn.to_torch(tensor, dtype=torch.float32) for tensor in preds[1][0]
            ]

            mask = ttnn.to_torch(preds[1][1], dtype=torch.float32)
            proto = ttnn.to_torch(preds[1][2], dtype=torch.float32)
            proto = proto.reshape((1, 160, 160, 32)).permute((0, 3, 1, 2))

            preds[1] = [[detect1_out, detect2_out, detect3_out], mask, proto]

            results = postprocess(preds, im, im0s, batch)
            for i in range(len(results)):
                save_seg_predictions_by_model(results[i], save_dir, paths[i], "tt_model")
        else:
            results = obj_postprocess(preds[0], im, im0s, batch, names)[0]
            save_yolo_predictions_by_model(results, save_dir, source, "tt_model")
        performant_runner.release()

    logger.info("Inference done")
