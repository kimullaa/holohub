import onnx
from onnxconverter_common import float16

model = onnx.load("/app/anime/AnimeGANv3_Hayao_36.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "/app/anime/AnimeGANv3_Hayao_16.onnx")

model = onnx.load("/app/anime/AnimeGANv3_PortraitSketch_25.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "/app/anime/AnimeGANv3_PortraitSketch_16.onnx")

model = onnx.load("/app/anime/AnimeGANv3_large_Ghibli_c1_e299.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "/app/anime/AnimeGANv3_large_Ghibli_c1_16.onnx")
