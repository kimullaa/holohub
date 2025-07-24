import os

from holoscan.core import Fragment
from holoscan.operators import VideoStreamReplayerOp


class VideoInputFragment(Fragment):
    def __init__(self, app, name):
        super().__init__(app, name)

    def compose(self):
        input_op = VideoStreamReplayerOp(
            self,
            name="replayer_source",
            **self.kwargs("replayer_source"),
        )
        self.add_operator(input_op)
