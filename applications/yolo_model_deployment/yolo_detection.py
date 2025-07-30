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
import argparse
from argparse import ArgumentParser
from holoscan.core import Tracker
from holoscan.core import Application, Operator, OperatorSpec
from video_input_fragment import VideoInputFragment
from video_capture_fragment import VideoCaptureFragment
from inferier_fragment import InferierFragment
from holoviz_fragment import HolovizFragment



class YoloDetApp(Application):
    """
    YOLO Detection Application.

    This application performs object detection using a YOLO model. It supports
    video input from a replayer or a V4L2 device and visualizes the detection results.

    Parameters:
        video_dir (str): Path to the video directory.
        data (str): Path to the model data directory.
        source (str): Input source, either "replayer" or "v4l2".
    """

    def __init__(self, video_dir, data, debug, source="replayer"):
        super().__init__()
        self.name = "YOLO Detection App"
        self.source = source
        self.debug = debug

        # Set default paths if not provided
        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOHUB_DATA_PATH", "../data"), "yolo_model_deployment"
            )
        self.data = data

        if video_dir == "none":
            video_dir = data
        self.video_dir = video_dir

    def compose(self):
        # Resource allocator
        if self.source == "v4l2":
            source = VideoCaptureFragment(self, "video_cap_in")
            source_output = "v4l2_source.signal"
            in_dtype = "rgba8888"

        elif self.source == "replayer":
            source = VideoInputFragment(self, "video_rep_in", self.video_dir)
            source_output = "replayer_source.output"
            in_dtype = "rgb888"

        holoviz = HolovizFragment(self, "holoviz", self.debug)

        inferier = InferierFragment(self, "inferier", self.data, self.debug, in_dtype)
        self.add_flow(source, inferier, {(source_output, "detection_preprocessor")})
        self.add_flow(source, inferier, {(source_output, "detection_visualizer.receivers")})

        self.add_flow(source, holoviz, {(source_output, "detection_visualizer.receivers")})
        


def parse_args() -> argparse.Namespace:
    # Argument parser
    parser = ArgumentParser(description="YOLO Detection Demo Application.")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "yolo_detection.yaml"),
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help=("Input source: 'v4l2' for V4L2 device or 'replayer' for video stream replayer."),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help="Path to the model data directory.",
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        default="none",
        help="Path to the video directory.",
    )
    parser.add_argument(
        "-x",
        "--debug",
        action='store_true',
        default=False,
        help="enable CONSOLE log.",
    )

    args, _ = parser.parse_known_args()

    return args



if __name__ == "__main__":
    args = parse_args()

    app = YoloDetApp(video_dir=args.video_dir, data=args.data, source=args.source, debug=args.debug)
    app.config(args.config)
    with Tracker(app, filename="tracker_yolo.log") as trackers:
        try:
            app.run()
        except KeyboardInterrupt:
            for fragment_name, tracker in trackers.items():
                print(f"Fragment:{fragment_name}")
                tracker.print()

        for fragment_name, tracker in trackers.items():
            print(f"Fragment:{fragment_name}")
            tracker.print()
