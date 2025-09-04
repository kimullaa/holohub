from holoscan.core import Application, Operator, OperatorSpec, Tensor, MetadataPolicy
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)

import os
from holoscan.core import  Fragment
from holoscan.resources import UnboundedAllocator

import cupy as cp
import numpy as np

class InferierFragment(Fragment):
    def __init__(self, app, name, data, debug, source):
        super().__init__(app, name)
        self.data = data
        self.debug = debug
        self.source = source

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        inference_kwargs = self.kwargs("detection_inference")
        for k, v in inference_kwargs["model_path_map"].items():
            inference_kwargs["model_path_map"][k] = os.path.join(self.data, v)

        detection_inference = InferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="allocator"),
            **inference_kwargs,
        )

        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        detection_visualizer = HolovizOp(
            self,
            name="detection_visualizer",
            tensors=[
                dict(name="", type="color"),
                dict(
                    name="bbox",
                    type="rectangles",
                    opacity=0.5,
                    line_width=4,
                    color=[1.0, 0.0, 0.0, 1.0],
                ),
            ],
            headless=self.debug,
            **self.kwargs("detection_visualizer"),
        )

        if self.source == "replayer":
            # Operators
            detection_preprocessor_v4l2 = FormatConverterOp(
                self,
                pool=pool,
                name="detection_preprocessor",
                **self.kwargs("detection_preprocessor_v4l2"),
            )

            detection_preprocessor_common = FormatConverterOp(
                self,
                pool=pool,
                name="detection_preprocessor_common",
                **self.kwargs("detection_preprocessor_common"),
            )

            self.add_flow(detection_preprocessor_v4l2, detection_preprocessor_common, {("", "")})
            self.add_flow(detection_preprocessor_common, detection_inference, {("", "receivers")})
            self.add_flow(detection_preprocessor_v4l2, detection_visualizer, {("", "receivers")})

        if self.source == "v4l2":
            detection_preprocessor = FormatConverterOp(
                self,
                pool=pool,
                name="detection_preprocessor",
                **self.kwargs("detection_preprocessor_common"),
            )
            self.add_flow(detection_preprocessor, detection_inference, {("", "receivers")})
            self.add_flow(detection_preprocessor, detection_visualizer, {("", "receivers")})

        # Data flow
        self.add_flow(detection_inference, detection_postprocessor, {("transmitter", "")})
        # Connect the postprocessor to the visualizer
        self.add_flow(detection_postprocessor, detection_visualizer, {("outputs", "receivers")})
        self.add_flow(
            detection_postprocessor, detection_visualizer, {("output_specs", "input_specs")}
        )

        if (self.debug):
            console = ConsoleOp(self, name="console")
            self.add_flow(detection_postprocessor, console, {("output_specs", "specs")})



coco_label_map = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

class ConsoleOp(Operator):
    """Print the received text to the terminal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("specs")

    def compute(self, op_input, op_output, context):
        specs = op_input.receive("specs")
        for spec in specs:
            print(f"CONSOLE: {spec.text} \n")


class DetectionPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.

    in:
        inference_output_detection_boxes: Tensor of shape (nbatch, nboxes, ncoord)
            where:
            - nbatch: number of batches (typically 1)
            - nboxes: number of detected boxes
            - ncoord: coordinates (x1, y1, x2, y2) of each box
        inference_output_detection_scores: Tensor of shape (nbatch, nboxes)
            Confidence scores for each detected box
        inference_output_detection_classes: Tensor of shape (nbatch, nboxes)
            Predicted class (int) for each detected box

    outputs:
        bbox: (nbatch, nboxes*2, ncoord/2)
            where coordinates are scaled to [0,1] range
        bbox_label: (nbatch, nboxes, 2)
            label text coordinates, top-left of the bbox

    output_specs:
        HolovizOp.InputSpec("bbox_label", "text")
            a list of display label text
    """

    def __init__(self, *args, width=640, label_name_map=coco_label_map, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width
        self.label_name_map = label_name_map

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("output_specs")
        spec.output("outputs")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")
        # Convert input to numpy array (using CuPy)
        bboxes = cp.asarray(
            in_message.get("inference_output_detection_boxes")
        ).get()  # (nbatch, nboxes, ncoord)
        scores = cp.asarray(
            in_message.get("inference_output_detection_scores")
        ).get()  # (nbatch, nboxes)
        labels = cp.asarray(
            in_message.get("inference_output_detection_classes")
        ).get()  # (nbatch, nboxes)

        ix = scores.flatten() > 0  # Find the bbox with score >0
        if np.all(ix == False):
            bboxes = np.zeros([1, 2, 2], dtype=np.float32)
            labels = np.zeros([1, 1], dtype=np.float32)
            bboxes_reshape = bboxes
        else:
            bboxes = bboxes[:, ix, :]  # (nbatch, nboxes, 4)
            labels = labels[:, ix]
            # Make box shape compatible with Holoviz
            bboxes = bboxes / self.width  # The x, y need to be rescaled to [0,1]
            bboxes_reshape = np.reshape(bboxes, (1, -1, 2))  # (nbatch, nboxes*2, ncoord/2)

        bbox_label_text = [self.label_name_map[int(label)] for label in labels[0]]

        # Prepare output
        bbox_label = np.asarray([(b[0], b[1]) for b in bboxes[0]])  # Get the top-left coordinates
        out_message = {"bbox_label": bbox_label, "bbox": bboxes_reshape}

        spec_label = HolovizOp.InputSpec("bbox_label", "text")
        spec_label.text = bbox_label_text

        op_output.emit(out_message, "outputs")
        op_output.emit([spec_label], "output_specs")

