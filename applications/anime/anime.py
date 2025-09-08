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

from holoscan.core import Tracker
import os
import argparse
from argparse import ArgumentParser
from video_input_fragment import VideoInputFragment
from video_capture_fragment import VideoCaptureFragment
from inferier_fragment import InferierFragment

from holoscan.core import Application

class AnimeApp(Application):

    def __init__(self, source):
        """Initialize the Anime application"""
        super().__init__()

        # set name
        self.name = "Anime App"
        self.source = source

    def compose(self):
        if self.source == "v4l2":
            source = VideoCaptureFragment(self, "video_cap_in")
            source_output = "v4l2_source.signal"

        elif self.source == "replayer":
            source = VideoInputFragment(self, "video_rep_in")
            source_output = "replayer_source.output"

        # inferier = InferierFragment(self, "inferier", "/app/anime/AnimeGANv3_PortraitSketch_25.onnx", in_dtype)
        # inferier = InferierFragment(self, "inferier", "/app/anime/AnimeGANv3_Hayao_36.onnx", in_dtype)
        inferier = InferierFragment(self, "inferier", "/app/anime/AnimeGANv3_Hayao_16.onnx", self.source)

        self.add_flow(source, inferier, {(source_output, "preprocessor")})


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(description="Anime Demo Application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help=("Input source: 'v4l2' for V4L2 device or 'replayer' for video stream replayer."),
    )
    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    app = AnimeApp(source=args.source)
    config_file = os.path.join(os.path.dirname(__file__), "anime.yaml")
    app.config(config_file)
    with Tracker(app, filename="tracker_anime.log") as trackers:
        try:
            app.run()
        except KeyboardInterrupt:
            for fragment_name, tracker in trackers.items():
                print(f"Fragment:{fragment_name}")
                tracker.print()

        for fragment_name, tracker in trackers.items():
            print(f"Fragment:{fragment_name}")
            tracker.print()
