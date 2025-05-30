#!/usr/bin/env python3
import sys
import gi
import signal
import os
import time  # Import time module for FPS calculation & processing simulation
import numpy as np
import cv2
import torch

# os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase
import ttnn


from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_wormhole_b0, torch2tt_tensor, is_blackhole


from models.demos.yolov4.post_processing import (
    load_class_names,
    plot_boxes_cv2,
    post_processing,
)

# from models.experimental.yolov4.performant_files.mv2like_test_infra import create_test_infra
from models.utility_functions import (
    is_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch_random,
    profiler,
)
from models.perf.perf_utils import prep_perf_report


# Initialize GStreamer (only once)
Gst.init(None)


# --- Element Class Definition ---
class Yolov4(GstBase.BaseTransform):
    # Element metadata (for GStreamer)

    __gtype_name__ = "GstYolov4PythonBatching"

    __gstmetadata__ = (
        "Yolov4 Python",  # Long name
        "Filter/Effect/Converter",  # Classification
        "Prepends a configurable string to text buffer data",  # Description
        "Your Name <your.email@example.com>",  # Author
    )

    # Pad Templates
    _sink_template = Gst.PadTemplate.new("sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.new_any())
    _src_template = Gst.PadTemplate.new("src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any())
    __gsttemplates__ = (_src_template, _sink_template)  # Order can matter for some tools
    __gproperties__ = {
        "batch-size": (int, "Frequency", "Frequency of test signal", 1, 8, 1, GObject.ParamFlags.READWRITE)
    }

    def __init__(self):
        super().__init__()
        # Initialize properties from defaults if not set otherwise
        # self.batch_size = __gproperties__["batch-size"][3] # Default value
        self.model = None
        self.device = None

    def initialize_device(self):
        device_id = 0
        self.device = ttnn.CreateDevice(
            device_id,
            dispatch_core_config=self.get_dispatch_core_config(),
            l1_small_size=24576,
            trace_region_size=3211264,
            num_command_queues=2,
        )
        #        self.batch_size=1
        # ttnn.enable_program_cache(self.device)
        self.device.enable_program_cache()
        self.model = YOLOv4PerformantRunner(
            self.device,
            self.batch_size,
            ttnn.bfloat16,
            ttnn.bfloat16,
            resolution=(640, 640),
            model_location_generator=None,
        )
        print("########################################", batch_size)

    def get_dispatch_core_config(self):
        # TODO: 11059 move dispatch_core_type to device_params when all tests are updated to not use WH_ARCH_YAML env flag
        dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
        if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
            dispatch_core_type = ttnn.device.DispatchCoreType.ETH
        dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
        dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)

        return dispatch_core_config

    def _trace_release(self):
        ttnn.release_trace(self.device, self.tid)

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "batch-size":
            return self.batch_size  # Return the Python instance attribute
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == "batch-size":
            # Gst.info_object(self, f"Property 'batch-size' being set to: {value}")
            self.batch_size = value  # Store in Python instance attribute
            print("SET PROP", prop.name, value)
        else:
            raise AttributeError(f"Unknown property {prop.name}")

        self.initialize_device()

    def do_transform(self, inbuf: Gst.Buffer, outbuf: Gst.Buffer) -> Gst.FlowReturn:
        # print(self.batch_size)
        try:
            success, in_map_info = inbuf.map(Gst.MapFlags.READ)
            if not success:
                Gst.error("Yolov4: Failed to map input buffer")
                return Gst.FlowReturn.ERROR

            # original_data = in_map_info.data
            # prefix_bytes = self.prefix.encode('utf-8')
            # transformed_data = prefix_bytes + original_data
            print("SUCCESS")
            print(in_map_info.data)
            frame_data = np.frombuffer(in_map_info.data, dtype=np.uint8).reshape(320, 320, 3)
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            if type(frame_data) == np.ndarray and len(frame_data.shape) == 3:  # cv2 image
                frame_data = torch.from_numpy(frame_data.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(frame_data) == np.ndarray and len(frame_data.shape) == 4:
                frame_data = torch.from_numpy(frame_data.transpose(0, 3, 1, 2)).float().div(255.0)

            if self.batch_size > 1:
                n, c, h, w = frame_data.shape
                frame_data = frame_data.expand(self.batch_size, c, h, w)
            #            out, t = self.trace_run(frame_data)
            print(frame_data.shape)

            out = self.model.run(frame_data)
            print("Running model complete", out.shape)
            conf_thresh = 0.3
            nms_thresh = 0.4

            boxes = post_processing(img, conf_thresh, nms_thresh, output)
            namesfile = "models/demos/yolov4/resources/coco.names"
            class_names = load_class_names(namesfile)
            img = cv2.imread(imgfile)
            plot_boxes_cv2(img, boxes[0], "ttnn_yolov4_prediction_demo.jpg", class_names)

            inbuf.unmap(in_map_info)

            success, out_map_info = outbuf.map(Gst.MapFlags.WRITE)
            if not success:
                Gst.error("Yolov4: Failed to map output buffer for writing")
                return Gst.FlowReturn.ERROR

            outbuf.unmap(out_map_info)
            return Gst.FlowReturn.OK

        except Exception as e:
            Gst.error(f"Yolov4: Error in transform: {e}")
            # if "in_map_info" in locals() and in_map_info.buffer.is_mapped():
            #    in_map_info.buffer.unmap(in_map_info)
            # if "out_map_info" in locals() and out_map_info.buffer.is_mapped():
            #    out_map_info.buffer.unmap(out_map_info)
            return Gst.FlowReturn.ERROR


GObject.type_register(Yolov4)

__gstelementfactory__ = ("yolov4", Gst.Rank.NONE, Yolov4)

# The following are not strictly needed for __gstelementfactory__ but good for plugin tools
GST_PLUGIN_NAME = "yolov4plugin"
__gstplugininit__ = None  # Not using a plugin_init function with __gstelementfactory__
