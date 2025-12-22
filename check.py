import tensorrt as trt

TAEGET_ENGINE = "test.engine"

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
with open('layer_info.json', 'w') as f:
    for layer_idx in range(engine.num_layers):
        s = inspector.get_layer_information(layer_idx, trt.LayerInformationFormat.JSON)
        f.write(f"--- Layer {layer_idx} ---\n")
        f.write(s + "\n")


