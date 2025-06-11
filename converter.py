import onnx
from onnxconverter_common import float16

model = onnx.load("/app/anime/AnimeGANv3_Hayao_36.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "/app/anime/AnimeGANv3_Hayao_16.onnx")
