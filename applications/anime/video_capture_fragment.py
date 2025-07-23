import os

from holoscan.core import Fragment
from holoscan.operators import V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator

class VideoCaptureFragment(Fragment):
    def __init__(self, app, name, video_device="/dev/video0"):
        super().__init__(app, name)
        self.video_device = video_device

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        v4l2_args = self.kwargs("v4l2_source")
        if self.video_device != "none":
            v4l2_args["device"] = self.video_device

        input_op = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **v4l2_args,
            )
        self.add_operator(input_op)
