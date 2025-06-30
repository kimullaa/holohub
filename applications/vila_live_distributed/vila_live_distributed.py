
import os
from argparse import ArgumentParser

from holoscan.core import Application

from video_input_fragment import VideoInputFragment
from video_capture_fragment import VideoCaptureFragment
from inferier_fragment import InferierFragment


class V4L2toVLM(Application):
    def __init__(self, data, source="v4l2", video_device="none"):
        """V4L2 to VLM app"""
        super().__init__()
        # set name
        self.name = "V4L2 to VLM app"
        self.source = source

        if data == "none":
            data = os.path.join(
                os.environ.get("HOLOHUB_DATA_DIR", "/workspace/holohub/data"), "vila_live_distributed"
            )

        self.sample_data_path = data
        self.video_device = video_device

    def compose(self):

        # V4L2 to capture usb camera input or replayer to replay video
        if self.source == "v4l2":
            source = VideoCaptureFragment(self, "video_cap_in", self.video_device)
            source_output = "v4l2_source.signal"

        elif self.source == "replayer":
            source = VideoInputFragment(self, "video_rep_in", self.sample_data_path)
            source_output = "replayer_source.output"

        inferier = InferierFragment(self, "inferier", self.sample_data_path)

        self.add_flow(source, inferier, {(source_output, "holoviz.receivers")})



def main():
    # Parse args
    parser = ArgumentParser(description="VILA live Distributed application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["v4l2", "replayer"],
        default="v4l2",
        help=(
            "If 'v4l2', uses the v4l2 device specified in the yaml file or "
            " --video_device if specified. "
            "If 'replayer', uses video stream replayer."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=os.environ.get(
            "HOLOSCAN_CONFIG_PATH",
            os.path.join(os.path.dirname(__file__), "vila_live_distributed.yaml"),
        ),

        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    parser.add_argument(
        "-v",
        "--video_device",
        default="none",
        help=("The video device to use.  By default the application will use /dev/video0"),
    )
    
    args, _ = parser.parse_known_args()


    app = V4L2toVLM(args.data, args.source, args.video_device)
    app.config(args.config)
    app.run()


if __name__ == "__main__":
    main()
