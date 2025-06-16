import base64
import io
import time
from threading import Event, Thread
import cupy as cp
from holoscan.core import Operator, OperatorSpec
from PIL import Image
from vlm import VLM
from webserver import Webserver

class VLMWebAppOp(Operator):
    """
    VLMWebApp that using a local VLM model and a Flask web-app to display the results
    """

    def __init__(self, fragment, *args, **kwargs):
        self.server = Webserver()
        self.vlm = VLM()
        self.is_busy = Event()
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Start the Webserver on a background thread
        self.server.start()
        time.sleep(3)

    def setup(self, spec: OperatorSpec):
        spec.input("video_stream")

    def stop(self):
        pass

    def annotate_image(self, image_b64):
        self.is_busy.set()
        prompt = self.server.user_input.replace('"', "")
        for response in self.vlm.generate_response(prompt, image_b64):
            chat_history = [[prompt, response]]
            self.server.send_chat_history(chat_history)
        # time.sleep(3) # May be needed depending on the speed of the model
        self.is_busy.clear()

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("video_stream").get("")
        if in_message:
            # Create a b64 Image from the Holoscan Tensor
            cp_image = cp.from_dlpack(in_message)
            np_image = cp.asnumpy(cp_image)
            image = Image.fromarray(np_image)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")  # Save in JPEG format
            buffer.seek(0)
            image_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Check if we're currently running the VLM
            if not self.is_busy.is_set():
                thread = Thread(target=self.annotate_image, args=(image_b64,))
                thread.start()

            # Send the video frame to the web-app to be displayed
            payload = {"image_b64": image_b64}
            self.server.send_message(payload)

