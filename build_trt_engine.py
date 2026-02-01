import json

import tensorrt as trt


ONNX_PATH = "simplified_QuantViT.onnx"
ENGINE_PATH = "QuantViT.engine"


def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print("Parsing ONNX file:", onnx_path)
    try:
        import onnx
        m = onnx.load(onnx_path)
        print("ONNX opset imports:", m.opset_import)
    except Exception as e:
        print("Warning: failed to load ONNX via python onnx:", e)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    print("Network inputs/outputs:")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print("Input", i, t.name, t.shape)
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print("Output", i, t.name, t.shape)

    config = builder.create_builder_config()
    workspace_size = 1 << 30  # 1GB
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    except AttributeError:
        config.max_workspace_size = workspace_size  # type: ignore[attr-defined]

    # 关键：打开 INT8，并把 profiling_verbosity 设为 DETAILED
    config.set_flag(trt.BuilderFlag.INT8)
    try:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    except AttributeError:
        # 对于不支持 ProfilingVerbosity 的旧 wheel，忽略
        print("ProfilingVerbosity not supported by this TensorRT Python wheel, layer info may be names-only.")

    # --- 为动态 batch 输入添加 OptimizationProfile（关键修改） ---
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    B, N, H, W = input_tensor.shape  # (-1, 16) -> feat_dim = 16

    min_shape = (1, N, H, W)
    opt_shape = (8, N, H, W)
    max_shape = (32, N, H, W)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    # -------------------------------------------------------------

    print("Network layers:")
    for i in range(network.num_layers):
        layer = network[i]
        print(i, layer.name, layer.type)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized network")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to deserialize CUDA engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("saved TensorRT engine to", engine_path)



if __name__ == "__main__":
    print(f"tensorRT version: {trt.__version__}")
    build_engine(ONNX_PATH, ENGINE_PATH)