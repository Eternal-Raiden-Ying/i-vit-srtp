import json

import tensorrt as trt


ONNX_PATH = "simple_mlp_fp.onnx"
QPARAMS_JSON = "simple_mlp_qparams.json"
ENGINE_PATH = "simple_mlp_int8.engine"


def load_qparams(path):
    with open(path, "r") as f:
        return json.load(f)


def _create_builder_safe(logger):
    """
    包一层，捕获 TensorRT 内部返回 nullptr 的情况，输出更明确的提示。
    """
    try:
        builder = trt.Builder(logger)
    except TypeError as e:
        raise RuntimeError(
            "Failed to create TensorRT Builder.\n"
            "This is usually an environment / version issue.\n"
            "Check that:\n"
            "  - TensorRT version matches your CUDA / cuDNN version.\n"
            "  - `trtexec` can run successfully from command line.\n"
            "  - Python bindings (pip/whl) come from the SAME TensorRT package.\n"
        ) from e

    if builder is None:
        raise RuntimeError(
            "TensorRT Builder is None (factory returned nullptr). "
            "This usually indicates a runtime / license / environment issue."
        )
    return builder


def _inspect_engine(engine: trt.ICudaEngine, path: str):
    """
    打印 engine 的 IO tensor 信息，帮助你检查 dtype / shape。
    新版 TensorRT 使用 num_io_tensors / get_tensor_* 接口。
    """
    with open(path, "w") as f:
        print("=== Engine IO tensors ===", file=f)
        try:
            num_tensors = engine.num_io_tensors
            for i in range(num_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)
                io_mode = engine.get_tensor_mode(name)  # INPUT / OUTPUT
                print(f"[{i}] {io_mode.name} {name}, shape={shape}, dtype={dtype}", file=f)
        except AttributeError:
            print("Engine inspection not supported by this TensorRT Python API.", file=f)

        print(
            "\nNOTE: For full graph & Q/DQ nodes, use `trtexec --loadEngine=... --dumpLayerInfo`.",
            file=f,
        )


def build_engine(onnx_path, qparams_json, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = _create_builder_safe(logger)
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
    _, feat_dim = input_tensor.shape  # (-1, 16) -> feat_dim = 16

    min_shape = (1, feat_dim)
    opt_shape = (8, feat_dim)
    max_shape = (32, feat_dim)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    # -------------------------------------------------------------

    # 这里只是为了调试保留 qparams 的读取
    qparams = load_qparams(qparams_json)
    print("Loaded qparams keys:", list(qparams.keys()))

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

    inspect_path = engine_path + ".summary.txt"
    _inspect_engine(engine, inspect_path)
    print("saved engine summary to", inspect_path)


if __name__ == "__main__":
    print(f"tensorRT version: {trt.__version__}")
    build_engine(ONNX_PATH, QPARAMS_JSON, ENGINE_PATH)