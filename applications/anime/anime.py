# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser

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


class AnimeApp(Application):

    def __init__(self):
        """Initialize the Anime application"""
        super().__init__()

        # set name
        self.name = "Anime App"

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        source = VideoStreamReplayerOp(
            self,
            name="replayer_source",
            **self.kwargs("replayer"),
        )

        preprocessor = FormatConverterOp(
            self,
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

        animeInfer = AnimeInferOp(self, name="my infer", model_path="/app/anime/AnimeGANv3_Hayao_16.onnx")

        self.add_flow(source, preprocessor)
        self.add_flow(preprocessor, animeInfer, {("tensor", "in_tensor")})
        self.add_flow(animeInfer, holoviz, {("output_image", "receivers")})

if __name__ == "__main__":

    app = AnimeApp()
    config_file = os.path.join(os.path.dirname(__file__), "anime.yaml")
    app.config(config_file)
    app.run()
