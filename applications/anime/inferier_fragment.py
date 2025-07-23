

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
from holoscan.resources import UnboundedAllocator

import random



import os
from holoscan.core import  Fragment
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
)
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from vlm_webapp_op import VLMWebAppOp


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

        result  = self.session.run([self.output_name], {self.input_name: cp.asnumpy(input_array).astype(np.float16) })
        converted = self.post_process(result[0], (input_array.shape[1], input_array.shape[2]) )

        out_message = { "image": Tensor.as_tensor(converted) }
        op_output.emit(out_message, "output_image")


class InferierFragment(Fragment):
    def __init__(self, app, name, model_path):
        super().__init__(app, name)
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise ValueError(f"Could not find video data: {model_path=}")

    def compose(self):

       preprocessor = FormatConverterOp(
            self,
            in_dtype=in_dtype,
            name="preprocessor",
            pool=pool,
            **self.kwargs("preprocessor"),
        )

        holoviz = HolovizOp(
            self,
            allocator=pool,
            name="holoviz",
            window_title="Anime",
            **self.kwargs("holoviz"),
        )

        animeInfer = AnimeInferOp(self, name="my infer", model_path=self.model_path)

        self.add_flow(preprocessor, animeInfer, {("tensor", "in_tensor")})
        self.add_flow(animeInfer, holoviz, {("output_image", "receivers")})

