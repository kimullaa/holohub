
import cupy as cp
import cv2
import holoscan as hs
import numpy as np

import onnxruntime as ort
from holoscan.core import Application, Operator, OperatorSpec, Tensor
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    V4L2VideoCaptureOp,
    VideoStreamReplayerOp,
)

import random

import os
from holoscan.core import  Fragment
from holoscan.resources import UnboundedAllocator


class AnimeInferOp(Operator):
    def __init__(self, fragment, model_path, **kwargs):
        self.model_path = model_path
        super().__init__(fragment, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in_tensor")
        spec.output("output_image")
        spec.param("model_path")

    def initialize(self):
        self.session = ort.InferenceSession(self.model_path, providers = ['CUDAExecutionProvider','CPUExecutionProvider',])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def post_process(self, images, size):
        images = (np.squeeze(images) + 1.) / 2 * 255
        images = np.clip(images, 0, 255).astype(np.uint8)
        images = cv2.resize(images, size)

        return images

    def compute(self, op_input, op_output, context):
        input_tensor = op_input.receive("in_tensor")
        input_array = cp.asarray(input_tensor["preprocessed"])

        input_array = cp.expand_dims(input_array, axis=0)

        if self.model_path.endswith("_16.onnx"):
            result  = self.session.run([self.output_name], {self.input_name: cp.asnumpy(input_array).astype(np.float16) })
        else:
            result  = self.session.run([self.output_name], {self.input_name: cp.asnumpy(input_array).astype(np.float32) })
        converted = self.post_process(result[0], (input_array.shape[1], input_array.shape[2]) )

        out_message = { "image": Tensor.as_tensor(converted) }
        op_output.emit(out_message, "output_image")


class InferierFragment(Fragment):
    def __init__(self, app, name, model_path, source):
        super().__init__(app, name)
        self.model_path = model_path
        self.source = source

        if not os.path.exists(self.model_path):
            raise ValueError(f"Could not find video data: {model_path=}")

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        if self.source == "v4l2":
            preprocessor_v4l2 = FormatConverterOp(
                self,
                name="preprocessor",
                pool=pool,
                **self.kwargs("preprocessor_v4l2"),
            )
            preprocessor_common = FormatConverterOp(
                self,
                name="preprocessor_common",
                pool=pool,
                **self.kwargs("preprocessor_common"),
            )
            self.add_operator(preprocessor_v4l2)
            self.add_flow(preprocessor_v4l2, preprocessor_common)

        elif self.source == "replayer":
            preprocessor_common = FormatConverterOp(
                self,
                name="preprocessor",
                pool=pool,
                **self.kwargs("preprocessor_common"),
            )
            self.add_operator(preprocessor_common)

        animeInfer = AnimeInferOp(self, name="my infer", model_path=self.model_path)
        self.add_flow(preprocessor_common, animeInfer, {("tensor", "in_tensor")})
        holoviz = HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            window_title="Anime",
            **self.kwargs("holoviz"),
        )
        self.add_flow(animeInfer, holoviz, {("output_image", "receivers")})
