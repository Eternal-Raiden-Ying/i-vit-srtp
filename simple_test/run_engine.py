import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import numpy as np
import os

ENGINE_PATH = "simple_mlp_int8.engine"

def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def run_inference(engine, input_np: np.ndarray):
    context = engine.create_execution_context()

    # ----- 1. 获取输入/输出 tensor 名 -----
    input_tensors = [engine.get_tensor_name(i)
                     for i in range(engine.num_io_tensors)
                     if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_tensors = [engine.get_tensor_name(i)
                      for i in range(engine.num_io_tensors)
                      if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    assert len(input_tensors) == 1, "Expect exactly one input tensor"
    assert len(output_tensors) == 1, "Expect exactly one output tensor"

    input_name = input_tensors[0]
    output_name = output_tensors[0]

    # ----- 2. 设置输入 shape（处理动态 shape） -----
    input_shape = engine.get_tensor_shape(input_name)
    if -1 in input_shape:
        context.set_input_shape(input_name, input_np.shape)
    else:
        # 可选校验
        assert tuple(input_shape) == tuple(input_np.shape), \
            f"Input shape mismatch: engine {input_shape}, got {input_np.shape}"

    # ----- 3. 准备输出数组 -----
    output_shape = tuple(context.get_tensor_shape(output_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    output_np = np.empty(output_shape, dtype=output_dtype)

    # ----- 4. 分配 device buffer -----
    d_input = cuda.mem_alloc(input_np.nbytes)
    d_output = cuda.mem_alloc(output_np.nbytes)

    # ----- 5. CUDA stream -----
    stream = cuda.Stream()

    # H2D
    cuda.memcpy_htod_async(d_input, input_np, stream)

    # 绑定 tensor 地址
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # 执行推理（v10+）
    context.execute_async_v3(stream.handle)

    # D2H
    cuda.memcpy_dtoh_async(output_np, d_output, stream)
    stream.synchronize()

    return output_np

if __name__ == "__main__":
    assert os.path.exists(ENGINE_PATH), f"Engine not found: {ENGINE_PATH}"

    engine = load_engine(ENGINE_PATH)

    # 构造一个测试输入（根据你的模型改 shape）
    dummy_input = np.random.randn(1, 16).astype(np.float32)

    output = run_inference(engine, dummy_input)
    print("Output shape:", output.shape)
    print("Output:", output)