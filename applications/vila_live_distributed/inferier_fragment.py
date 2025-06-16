
import os
from holoscan.core import  Fragment
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
)
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from vlm_webapp_op import VLMWebAppOp


class InferierFragment(Fragment):
    def __init__(self, app, name, video_dir):
        super().__init__(app, name)
        self.video_dir = video_dir

        if not os.path.exists(self.video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

    def compose(self):

        formatter_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        format_converter_vlm = FormatConverterOp(
            self,
            name="convert_video_to_tensor",
            in_dtype="rgba8888",
            out_dtype="rgb888",
            cuda_stream_pool=formatter_cuda_stream_pool,
            pool=UnboundedAllocator(self, name="FormatConverter allocator"),
        )

        holoviz_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            window_title="VILA Live",
            headless=True,
            enable_render_buffer_input=False,
            enable_render_buffer_output=True,
            allocator=UnboundedAllocator(self, name="Holoviz allocator"),
            cuda_stream_pool=holoviz_cuda_stream_pool,
            **self.kwargs("holoviz"),
        )

        # Initialize the VLM + WebApp operator
        web_server = VLMWebAppOp(self, name="VLMWebAppOp")

        self.add_flow(visualizer, format_converter_vlm, {("render_buffer_output", "source_video")})
        self.add_flow(format_converter_vlm, web_server, {("tensor", "video_stream")})

