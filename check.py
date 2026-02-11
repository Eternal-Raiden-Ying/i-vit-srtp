import tensorrt as trt

TAEGET_ENGINE = "./QuantViT.engine"


"""
    QuantViT.engine的检查结果如下:
        embedding层：卷积中matmul是int8，加bias是fp，后续的cls token和pos embed都融合成一个算子了，判断不了内部的执行逻辑
        所有残差连接的地方用的fp，应该是所有quant_act(16)地方都用的fp执行，暂时没找到反例
        softmax算子的input output接口处都是int8， nvidia应该内部有优化，onnx里面是作为fp处理的，QDQ还是给融合成int8了（内部可能还是fp在算）
"""




logger = trt.Logger(trt.Logger.INFO)

# 1. 反序列化 engine
with open(TAEGET_ENGINE, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

print("Engine name:", engine.name)
print("Has implicit batch dim:", engine.has_implicit_batch_dimension)

print("=== Engine IO tensors ===")
try:
    num_tensors = engine.num_io_tensors
    for i in range(num_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"[{i}] {mode.name} {name}, shape={shape}, dtype={dtype}")
except AttributeError:
    print("This TensorRT Python wheel does not expose num_io_tensors/get_tensor_* APIs.")

inspector = engine.create_engine_inspector()
# print("=== Engine layers (DETAILED) ===")
# layer_idx = 0
# for layer_idx in range(engine.num_layers):
#         s = inspector.get_layer_information(layer_idx, trt.LayerInformationFormat.DETAILED)
#         print(f"--- Layer {layer_idx} ---\n")
#         print(s + "\n")
with open('QuantViT.json', 'w') as f:
    for layer_idx in range(engine.num_layers):
        s = inspector.get_layer_information(layer_idx, trt.LayerInformationFormat.JSON)
        f.write(f"--- Layer {layer_idx} ---\n")
        f.write(s + "\n")




