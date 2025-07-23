import os

from holoscan.core import Fragment
from holoscan.operators import V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator

class VideoCaptureFragment(Fragment):
    def __init__(self, app, name):
        super().__init__(app, name)

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        input_op = V4L2VideoCaptureOp(
                self,
                name="v4l2_source",
                allocator=pool,
                **self.kwargs("v4l2_source"),
            )
        self.add_operator(input_op)
