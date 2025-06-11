from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
import onnx

new_onnx_model = convert_float_to_float16_model_path('/app/anime/AnimeGANv3_Hayao_36.onnx')
onnx.save(new_onnx_model, '/app/anime/AnimeGANv3_Hayao_16.onnx')
