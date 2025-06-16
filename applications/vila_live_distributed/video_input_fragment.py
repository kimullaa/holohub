import os

from holoscan.core import Fragment
from holoscan.operators import VideoStreamReplayerOp


class VideoInputFragment(Fragment):
    def __init__(self, app, name, video_dir):
        super().__init__(app, name)
        self.video_dir = video_dir

        if not os.path.exists(self.video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

    def compose(self):
        input_op = VideoStreamReplayerOp(
            self,
            name="replayer_source",
            directory=self.video_dir,
            **self.kwargs("replayer_source"),
        )
        self.add_operator(input_op)
