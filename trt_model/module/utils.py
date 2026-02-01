"""
TensorRT 工具函数模块
提供量化、激活函数、归一化等基础操作的TensorRT实现
"""

import tensorrt as trt
import numpy as np
import torch
import ctypes
from typing import Tuple, Optional, Dict, Any, Union


def get_scale_and_zero_point(
    qparams: Dict[str, Any],
    key: str,
    default_scale: Union[float, np.ndarray] = 1.0,
    default_zero_point: int = 0
) -> Tuple[Union[float, np.ndarray], int]:
    """
    从量化参数字典中获取scale和zero_point。

    支持两种形式：
    1) qparams["xxx.scale"] = 0.0078
    2) qparams["xxx"] = {"scale": 0.0078, "zero_point": 0}
    """
    if key not in qparams:
        return default_scale, default_zero_point

    value = qparams.get(key)
    if isinstance(value, dict):
        scale = value.get("scale", default_scale)
        zero_point = value.get("zero_point", default_zero_point)
        return scale, zero_point

    return value, default_zero_point


def normalize_scale(scale: Union[float, np.ndarray, list]) -> Union[float, np.ndarray]:
    if isinstance(scale, list):
        return np.array(scale, dtype=np.float32)
    return scale


def set_tensor_dynamic_range(
    tensor: trt.ITensor,
    scale: Union[float, np.ndarray]
) -> None:
    """
    为INT8推理设置Tensor动态范围。

    注意：TensorRT仅支持per-tensor动态范围。
    当scale为per-channel时，使用最大scale近似。
    """
    if isinstance(scale, np.ndarray):
        max_scale = float(np.max(scale))
    else:
        max_scale = float(scale)

    max_range = 127.0 * max_scale
    min_range = -max_range

    if hasattr(tensor, "set_dynamic_range"):
        tensor.set_dynamic_range(min_range, max_range)
    else:
        tensor.dynamic_range = (min_range, max_range)


def get_tensor_dynamic_range(tensor: trt.ITensor) -> Optional[Tuple[float, float]]:
    if hasattr(tensor, "dynamic_range"):
        try:
            return tensor.dynamic_range
        except Exception:
            raise AttributeError("Failed to get dynamic_range attribute")
    raise AttributeError("Object does not have dynamic_range attribute")


def set_tensor_dynamic_range_like(
    output: trt.ITensor,
    reference: trt.ITensor,
    scale: float = 1.0
) -> None:
    ref_range = get_tensor_dynamic_range(reference)
    if not ref_range:
        return
    min_r, max_r = ref_range
    max_abs = max(abs(min_r), abs(max_r)) * abs(scale)
    if max_abs <= 0:
        return
    if hasattr(output, "set_dynamic_range"):
        output.set_dynamic_range(-max_abs, max_abs)
    else:
        output.dynamic_range = (-max_abs, max_abs)


_REQUANT_PLUGIN_LOADED = False


def ensure_requant_plugin_loaded(dll_path: str) -> None:
    """
    Load the custom requant plugin DLL once.

    Args:
        dll_path: Absolute path to the built requant plugin DLL.
    """
    global _REQUANT_PLUGIN_LOADED
    if _REQUANT_PLUGIN_LOADED:
        return
    ctypes.CDLL(dll_path)
    _REQUANT_PLUGIN_LOADED = True


def scale_view(scale: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    """
    将scale数组reshape以便广播。

    Args:
        scale: 原始scale数组
        axis: 量化轴
        ndim: 目标张量维度数
    """
    shape = [1] * ndim
    shape[axis] = scale.size
    return scale.reshape(shape)


def quantize_to_int8(
    data: np.ndarray,
    scale: Union[float, np.ndarray],
    zero_point: int = 0,
    axis: int = 0,
    qmin: int = -128,
    qmax: int = 127
) -> np.ndarray:
    """
    将浮点权重按给定scale量化为int8。

    Args:
        data: 浮点权重
        scale: 量化scale（标量或per-channel）
        zero_point: 零点（默认0，保留接口）
    """
    scale = normalize_scale(scale)
    if isinstance(scale, np.ndarray):
        scale_reshape = scale_view(scale, axis, data.ndim)
        q = np.round(data / scale_reshape) + zero_point
    else:
        q = np.round(data / float(scale)) + zero_point

    q = np.clip(q, qmin, qmax)
    return q.astype(np.int8)

def quantize_to_int32(
    data: np.ndarray | torch.Tensor,
    scale: Union[float, np.ndarray],
    zero_point: int = 0,
    axis: int = 0,
    qmin: int = -2147483648,
    qmax: int = 2147483647
) ->  np.ndarray:
    """
    将浮点权重按给定scale量化为int32。

    Args:
        data: 浮点权重
        scale: 量化scale（标量或per-channel）
        zero_point: 零点（默认0，保留接口）
    """
    scale = normalize_scale(scale)
    if isinstance(scale, np.ndarray):
        scale_reshape = scale_view(scale, axis, data.ndim)
        q = np.round(data / scale_reshape) + zero_point
    else:
        q = np.round(data / float(scale)) + zero_point
    q = np.clip(q, qmin, qmax)
    return q.astype(np.int32)


def resolve_quantized_weight(
    weight_entry: Any,
    qparams: Dict[str, Any],
    weight_scale_key: str,
    weight_zp_key: Optional[str] = None
) -> Tuple[np.ndarray, Union[float, np.ndarray], int]:
    """
    解析权重量化参数。

    支持两种形式：
    1) weights[key] = np.ndarray (float或int8)，scale从qparams中取
    2) weights[key] = {"data": np.ndarray, "scale": ..., "zero_point": ...}
    """
    if isinstance(weight_entry, dict):
        weight_data = weight_entry.get("data")
        weight_scale = weight_entry.get("scale", 1.0)
        weight_zp = weight_entry.get("zero_point", 0)
    else:
        weight_data = weight_entry
        weight_scale, weight_zp = get_scale_and_zero_point(
            qparams, weight_scale_key, 1.0, 0
        )
        if weight_zp_key is not None:
            _, weight_zp = get_scale_and_zero_point(qparams, weight_zp_key, 1.0, weight_zp)

    return weight_data, normalize_scale(weight_scale), int(weight_zp)


def add_quantize_layer(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    scale: float | np.ndarray,
    name: str,
    axis: Optional[int] = None      # per-channel量化的轴
) -> trt.ITensor:
    """
    添加量化层 (Q节点)
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        scale: 量化缩放因子（float或numpy数组，用于per-channel）
        name: 层名称
        axis: per-channel量化的轴，None表示per-tensor量化
        
    Returns:
        量化后的张量
    """
    # 将scale转换为ITensor
    if isinstance(scale, (int, float)):
        # per-tensor量化
        scale_tensor = network.add_constant(
            shape=(1,),
            weights=trt.Weights(np.array([scale], dtype=np.float32))
        ).get_output(0)
    elif isinstance(scale, np.ndarray):
        # per-channel量化
        scale_tensor = network.add_constant(
            shape=scale.shape,
            weights=trt.Weights(scale.astype(np.float32))
        ).get_output(0)
    else:
        raise ValueError(f"Scale must be float or numpy.ndarray, recieved {type(scale)} yet")
    
    quant_layer = network.add_quantize(
        input=input_tensor,
        scale=scale_tensor
    )
    if axis is not None:
        quant_layer.axis = axis
    quant_layer.name = f"{name}_quantize"
    quant_layer.get_output(0).name = f"{name}_quantize_out"
    return quant_layer.get_output(0)


def add_dequantize_layer(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    scale: float | np.ndarray,
    name: str,
    axis: Optional[int] = None      # per-channel量化的轴
) -> trt.ITensor:
    """
    添加反量化层 (DQ节点)
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        scale: 反量化缩放因子（float或numpy数组，用于per-channel）
        name: 层名称
        axis: per-channel量化的轴，None表示per-tensor量化
        
    Returns:
        反量化后的张量
    """
    # 将scale转换为ITensor
    if isinstance(scale, (int, float)):
        # per-tensor量化
        scale_tensor = network.add_constant(
            shape=(1,),
            weights=trt.Weights(np.array([scale], dtype=np.float32))
        ).get_output(0)
    elif isinstance(scale, np.ndarray):
        # per-channel量化
        scale_tensor = network.add_constant(
            shape=scale.shape,
            weights=trt.Weights(scale.astype(np.float32))
        ).get_output(0)
    else:
        raise ValueError(f"Scale must be float or numpy.ndarray, recieved {type(scale)} yet")
    
    dequant_layer = network.add_dequantize(
        input=input_tensor,
        scale=scale_tensor
    )
    if axis is not None:
        dequant_layer.axis = axis
    dequant_layer.name = f"{name}_dequantize"
    dequant_layer.get_output(0).name = f"{name}_dequantize_out"
    return dequant_layer.get_output(0)


def add_qdq_layer(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    scale: float | np.ndarray,
    name: str,
    axis: Optional[int] = None
) -> trt.ITensor:
    """
    添加QDQ层对 (量化+反量化)
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        scale: 量化/反量化缩放因子（float或numpy数组）
        name: 层名称
        axis: per-channel量化的轴，None表示per-tensor量化
        
    Returns:
        经过QDQ处理的张量
    """
    # Q节点
    quant_out = add_quantize_layer(network, input_tensor, scale, name, axis)
    # DQ节点
    dequant_out = add_dequantize_layer(network, quant_out, scale, name, axis)
    return dequant_out

def add_requant_layer(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    input_scale: float,
    output_scale: float,
    bits: int = 8,
    name: str = "requant"
) -> trt.ITensor:
    """
    添加重新量化层 (Requant节点)
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        input_scale: 输入量化缩放因子
        output_scale: 输出量化缩放因子
        bits: 量化位宽 (默认8)
        name: 层名称
        
    Returns:
        重新量化后的张量
    """
    if bits != 8:
        raise ValueError(f"Requant plugin only supports 8-bit output, got bits={bits}")

    ensure_requant_plugin_loaded(r"./custom_op/requant_plugin.dll")

    registry = trt.get_plugin_registry()
    if registry is None:
        raise RuntimeError("TensorRT plugin registry is not available")

    creator = registry.get_plugin_creator("RequantInt32ToInt8", "1", "")
    if creator is None:
        raise RuntimeError(
            "RequantInt32ToInt8 plugin not found. "
            "Make sure the plugin DLL is built and loaded before building the network."
        )

    scale1 = np.array([float(input_scale)], dtype=np.float32)
    scale2 = np.array([float(output_scale)], dtype=np.float32)

    fields = [
        trt.PluginField("scale1", scale1, trt.PluginFieldType.FLOAT32),
        trt.PluginField("scale2", scale2, trt.PluginFieldType.FLOAT32),
    ]
    field_collection = trt.PluginFieldCollection(fields)

    plugin = creator.create_plugin(f"{name}_requant_plugin", field_collection)
    if plugin is None:
        raise RuntimeError("Failed to create RequantInt32ToInt8 plugin instance")

    requant_layer = network.add_plugin_v2([input_tensor], plugin)
    requant_layer.name = f"{name}_requantize"
    output = requant_layer.get_output(0)
    output.name = f"{name}_requantize_out"
    set_tensor_dynamic_range(output, output_scale)
    return output


def add_i32i32_elementwise_sum(
    network: trt.INetworkDefinition,
    input1: trt.ITensor,
    input2: trt.ITensor,
    name: str,
    add_div2: bool =False
) -> trt.ITensor:
    """
    添加INT32元素级加法层。
    输入输出均为INT32张量。
    """
    if not add_div2:
        add_layer = network.add_elementwise(
            input1=input1,
            input2=input2,
            op=trt.ElementWiseOperation.SUM
        )
        add_layer.name = f"{name}_int32_sum"
        add_layer.get_output(0).name = f"{name}_int32_sum_out"
        add_layer.precision = trt.DataType.INT32
        try:
            add_layer.set_output_type(0, trt.DataType.INT32)
        except Exception as exc:
            raise RuntimeError(f"Failed to set output type for {name}_int32_sum") from exc
        return add_layer.get_output(0)
    
    else:
        # INT32 floor-div by 2 for both inputs, then sum
        div_const = network.add_constant(
            shape=(1,),
            weights=trt.Weights(np.array([2], dtype=np.int32))
        )
        div_const.name = f"{name}_div2_const"
        div_tensor = div_const.get_output(0)
        div_tensor.name = f"{name}_div2_const_out"

        div_layer1 = network.add_elementwise(
            input1=input1,
            input2=div_tensor,
            op=trt.ElementWiseOperation.DIV
        )
        div_layer1.name = f"{name}_div2_a"
        div_layer1.get_output(0).name = f"{name}_div2_a_out"
        div_layer1.precision = trt.DataType.INT32
        try:
            div_layer1.set_output_type(0, trt.DataType.INT32)
        except Exception as exc:
            raise RuntimeError(f"Failed to set output type for {name}_div2_a") from exc

        div_layer2 = network.add_elementwise(
            input1=input2,
            input2=div_tensor,
            op=trt.ElementWiseOperation.DIV
        )
        div_layer2.name = f"{name}_div2_b"
        div_layer2.get_output(0).name = f"{name}_div2_b_out"
        div_layer2.precision = trt.DataType.INT32
        try:
            div_layer2.set_output_type(0, trt.DataType.INT32)
        except Exception as exc:
            raise RuntimeError(f"Failed to set output type for {name}_div2_b") from exc

        add_layer = network.add_elementwise(
            input1=div_layer1.get_output(0),
            input2=div_layer2.get_output(0),
            op=trt.ElementWiseOperation.SUM
        )
        add_layer.name = f"{name}_int32_sum"
        add_layer.get_output(0).name = f"{name}_int32_sum_out"
        add_layer.precision = trt.DataType.INT32
        try:
            add_layer.set_output_type(0, trt.DataType.INT32)
        except Exception as exc:
            raise RuntimeError(f"Failed to set output type for {name}_int32_sum") from exc
        return add_layer.get_output(0)


def add_i8i8_i32_linear(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    input_scale: float,
    weight_scale: float | np.ndarray,
    output_scale: float,
    name: str,
    use_per_channel: bool = False
) -> Tuple[trt.ITensor, float]:
    """
    添加INT8全连接层（支持per-channel量化）
    传入的weight和bias应为float，对接伪量化训练得到的浮点权重。
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        weight: 权重矩阵 (out_features, in_features)
        bias: 偏置向量
        input_scale: 输入量化缩放因子
        weight_scale: 权重量化缩放因子 (scalar或per-channel数组)
        output_scale: 输出量化缩放因子
        name: 层名称
        use_per_channel: 是否使用per-channel量化
        
    Returns:
        (输出张量, 输出缩放因子)
    """
    # 权重形状
    assert weight.ndim == 2, "Weight must be a 2D array, for convolution use add_int8_conv2d instead"
    out_features, in_features = weight.shape
    
    # if get_tensor_dynamic_range(input_tensor) is None:
    #     set_tensor_dynamic_range(input_tensor, input_scale)
    
    if use_per_channel:
        assert isinstance(weight_scale, np.ndarray) and weight_scale.size > 1, "weight_scale must be a numpy array for per-channel quantization"
    
    # 权重需要转置：从 (out_features, in_features) 转为 (in_features, out_features)
    # 这样 matmul: (..., in_features) @ (in_features, out_features) = (..., out_features)
    weight_transposed = np.ascontiguousarray(weight.T.astype(np.float32))  # (in_features, out_features)
    
    # 注意：转置后，per-channel应该在axis=1（输出通道维度）
    if use_per_channel:
        # Per-channel量化：转置后在axis=1（2D）或axis=2（3D）
        axis = 2 if len(input_tensor.shape) == 3 else 1
        weight_q = quantize_to_int8(
            data=weight_transposed,
            scale=weight_scale,
            axis=axis
        )
    else:
        # Per-tensor量化
        w_scale = weight_scale if isinstance(weight_scale, float) else float(weight_scale.mean())
        weight_q = quantize_to_int8(
            data=weight_transposed,
            scale=w_scale
        )
    weight_const = network.add_constant(
        shape=weight_q.shape,
        weights=trt.Weights(weight_q)
    )
    weight_const.name = f"{name}_weight_const"
    weight_tensor = weight_const.get_output(0)
    weight_tensor.name = f"{name}_weight_const_out"
    
    # 如果输入是3D，需要将权重reshape为3D以匹配维度
    if len(input_tensor.shape) == 3:
        # 输入是 (batch, seq_len, in_features)
        # 权重从 (in_features, out_features) reshape为 (1, in_features, out_features)
        shuffle = network.add_shuffle(weight_tensor)
        shuffle.reshape_dims = (1, weight_transposed.shape[0], weight_transposed.shape[1])
        shuffle.name = f"{name}_weight_reshape_broadcast"
        weight_tensor = shuffle.get_output(0)
        weight_tensor.name = f"{name}_weight_reshape_broadcast_out"

    
    # 全连接层（不需要transpose，因为权重已经转置）
    fc_layer = network.add_matrix_multiply(
        input0=input_tensor,
        op0=trt.MatrixOperation.NONE,
        input1=weight_tensor,
        op1=trt.MatrixOperation.NONE  # 不需要转置
    )
    fc_layer.name = f"{name}_matmul" 
    
    fc_layer.precision = trt.DataType.INT8
    fc_layer.set_output_type(0, trt.DataType.INT32)

    fc_output = fc_layer.get_output(0)
    fc_output.name = f"{name}_matmul_out"
    
    # 添加偏置
    if bias is not None:
        if len(input_tensor.shape) == 2:
            bias_shape = (1, out_features)
        elif len(input_tensor.shape) == 3:
            bias_shape = (1,1, out_features)
        elif len(input_tensor.shape) == 4:
            bias_shape = (1,1,1, out_features)
        else:
            raise ValueError(f"Unsupported input tensor ndim={len(input_tensor.shape)} for bias addition")
        
        bias_const = network.add_constant(
            shape=bias_shape,
            weights=trt.Weights(quantize_to_int32(bias, output_scale).astype(np.int32))
        )
        bias_const.name = f"{name}_bias_const"
        bias_tensor = bias_const.get_output(0)
        bias_tensor.name = f"{name}_bias_const_out"
        
        add_bias_output = add_i32i32_elementwise_sum(
            network,
            input1=fc_output,
            input2=bias_tensor,
            name=f"{name}_add_bias"
        )

        fc_output = add_bias_output
    
    return fc_output, output_scale


def add_int_layernorm(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    normalized_shape: int,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
    input_scale: float,
    output_scale: float,
    name: str,
    add_output_qdq: bool = True
) -> Tuple[trt.ITensor, float]:
    """
    添加整数LayerNorm层 (I-LayerNorm)
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        normalized_shape: 归一化的维度
        weight: gamma参数
        bias: beta参数
        eps: 数值稳定性小常数
        input_scale: 输入量化缩放因子
        output_scale: 输出量化缩放因子
        name: 层名称
        
    Returns:
        (输出张量, 输出缩放因子)
    """
    # 获取输入形状
    input_shape = input_tensor.shape
    if hasattr(input_shape, "nbDims"):
        ndim = input_shape.nbDims
    else:
        try:
            ndim = len(input_shape)
        except Exception:
            ndim = 0

    if ndim <= 0:
        # 本项目默认输入为 (B, N, C)
        ndim = 3

    # 计算均值 - Mean
    reduce_axes = 1 << (ndim - 1)  # 最后一个维度
    mean_layer = network.add_reduce(
        input=input_tensor,
        op=trt.ReduceOperation.AVG,
        axes=reduce_axes,
        keep_dims=True
    )
    mean_layer.name = f"{name}_mean"
    mean_output = mean_layer.get_output(0)
    
    # 计算 x - mean
    sub_layer = network.add_elementwise(
        input1=input_tensor,
        input2=mean_output,
        op=trt.ElementWiseOperation.SUB
    )
    sub_layer.name = f"{name}_sub_mean"
    centered = sub_layer.get_output(0)
    
    # 计算方差 - Variance
    pow_layer = network.add_elementwise(
        input1=centered,
        input2=centered,
        op=trt.ElementWiseOperation.PROD
    )
    pow_layer.name = f"{name}_square"
    squared = pow_layer.get_output(0)
    
    var_layer = network.add_reduce(
        input=squared,
        op=trt.ReduceOperation.AVG,
        axes=reduce_axes,
        keep_dims=True
    )
    var_layer.name = f"{name}_variance"
    variance = var_layer.get_output(0)
    
    # 计算 sqrt(var + eps)
    eps_shape = tuple([1] * ndim)
    eps_const = network.add_constant(
        shape=eps_shape,
        weights=trt.Weights(np.array([eps], dtype=np.float32))
    )
    eps_const.name = f"{name}_eps"
    set_tensor_dynamic_range(eps_const.get_output(0), eps if eps > 0 else 1e-6)
    set_tensor_dynamic_range(eps_const.get_output(0), eps if eps > 0 else 1e-6)
    
    add_eps_layer = network.add_elementwise(
        input1=variance,
        input2=eps_const.get_output(0),
        op=trt.ElementWiseOperation.SUM
    )
    add_eps_layer.name = f"{name}_add_eps"
    var_eps = add_eps_layer.get_output(0)
    
    sqrt_layer = network.add_unary(
        input=var_eps,
        op=trt.UnaryOperation.SQRT
    )
    sqrt_layer.name = f"{name}_sqrt"
    std = sqrt_layer.get_output(0)
    
    # 归一化: (x - mean) / std
    div_layer = network.add_elementwise(
        input1=centered,
        input2=std,
        op=trt.ElementWiseOperation.DIV
    )
    div_layer.name = f"{name}_normalize"
    normalized = div_layer.get_output(0)
    
    # 应用可学习参数: gamma * normalized + beta
    weight_const = network.add_constant(
        shape=(1, 1, normalized_shape),
        weights=trt.Weights(weight.reshape(1, 1, -1))
    )
    weight_const.name = f"{name}_gamma"
    set_tensor_dynamic_range(weight_const.get_output(0), float(np.max(np.abs(weight))) if weight.size else 1.0)
    set_tensor_dynamic_range(weight_const.get_output(0), float(np.max(np.abs(weight))) if weight.size else 1.0)
    
    scale_layer = network.add_elementwise(
        input1=normalized,
        input2=weight_const.get_output(0),
        op=trt.ElementWiseOperation.PROD
    )
    scale_layer.name = f"{name}_scale"
    scaled = scale_layer.get_output(0)
    
    bias_const = network.add_constant(
        shape=(1, 1, normalized_shape),
        weights=trt.Weights(bias.reshape(1, 1, -1))
    )
    bias_const.name = f"{name}_beta"
    set_tensor_dynamic_range(bias_const.get_output(0), float(np.max(np.abs(bias))) if bias.size else 1.0)
    set_tensor_dynamic_range(bias_const.get_output(0), float(np.max(np.abs(bias))) if bias.size else 1.0)
    
    add_bias_layer = network.add_elementwise(
        input1=scaled,
        input2=bias_const.get_output(0),
        op=trt.ElementWiseOperation.SUM
    )
    add_bias_layer.name = f"{name}_add_bias"
    output = add_bias_layer.get_output(0)
    
    # 添加QDQ节点
    output = add_qdq_layer(network, output, output_scale, name) if add_output_qdq else output
    
    return output, output_scale


def add_int_layernorm_direct(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    normalized_shape: int,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
    input_scale: float,
    output_scale: float,
    name: str,
    precision: Optional[trt.DataType] = None
) -> Tuple[trt.ITensor, float]:
    """
    添加LayerNorm（不使用QDQ，仅设置动态范围）。
    """
    # 复用原始实现，但去掉QDQ
    input_shape = input_tensor.shape
    if hasattr(input_shape, "nbDims"):
        ndim = input_shape.nbDims
    else:
        try:
            ndim = len(input_shape)
        except Exception:
            ndim = 0

    if ndim <= 0:
        # 本项目默认输入为 (B, N, C)
        ndim = 3

    # 默认归一化最后一维
    last_dim = ndim - 1
    reduce_axes = 1 << last_dim

    # 设置输入动态范围，避免缺失scale
    set_tensor_dynamic_range(input_tensor, input_scale)

    mean_layer = network.add_reduce(
        input=input_tensor,
        op=trt.ReduceOperation.AVG,
        axes=reduce_axes,
        keep_dims=True
    )
    mean_layer.name = f"{name}_mean"
    mean_output = mean_layer.get_output(0)
    set_tensor_dynamic_range_like(mean_output, input_tensor)

    sub_layer = network.add_elementwise(
        input1=input_tensor,
        input2=mean_output,
        op=trt.ElementWiseOperation.SUB
    )
    sub_layer.name = f"{name}_sub_mean"
    centered = sub_layer.get_output(0)
    set_tensor_dynamic_range_like(centered, input_tensor)

    pow_layer = network.add_elementwise(
        input1=centered,
        input2=centered,
        op=trt.ElementWiseOperation.PROD
    )
    pow_layer.name = f"{name}_square"
    squared = pow_layer.get_output(0)
    set_tensor_dynamic_range_like(squared, input_tensor)

    var_layer = network.add_reduce(
        input=squared,
        op=trt.ReduceOperation.AVG,
        axes=reduce_axes,
        keep_dims=True
    )
    var_layer.name = f"{name}_variance"
    variance = var_layer.get_output(0)
    set_tensor_dynamic_range_like(variance, input_tensor)

    eps_shape = tuple([1] * ndim)
    eps_const = network.add_constant(
        shape=eps_shape,
        weights=trt.Weights(np.array([eps], dtype=np.float32))
    )
    eps_const.name = f"{name}_eps"

    add_eps_layer = network.add_elementwise(
        input1=variance,
        input2=eps_const.get_output(0),
        op=trt.ElementWiseOperation.SUM
    )
    add_eps_layer.name = f"{name}_add_eps"
    var_eps = add_eps_layer.get_output(0)
    set_tensor_dynamic_range_like(var_eps, input_tensor)

    sqrt_layer = network.add_unary(
        input=var_eps,
        op=trt.UnaryOperation.SQRT
    )
    sqrt_layer.name = f"{name}_sqrt"
    std = sqrt_layer.get_output(0)
    set_tensor_dynamic_range_like(std, input_tensor)

    div_layer = network.add_elementwise(
        input1=centered,
        input2=std,
        op=trt.ElementWiseOperation.DIV
    )
    div_layer.name = f"{name}_normalize"
    normalized = div_layer.get_output(0)
    set_tensor_dynamic_range_like(normalized, input_tensor)

    weight_const = network.add_constant(
        shape=(1, 1, normalized_shape),
        weights=trt.Weights(weight.reshape(1, 1, -1))
    )
    weight_const.name = f"{name}_gamma"

    scale_layer = network.add_elementwise(
        input1=normalized,
        input2=weight_const.get_output(0),
        op=trt.ElementWiseOperation.PROD
    )
    scale_layer.name = f"{name}_scale"
    scaled = scale_layer.get_output(0)
    set_tensor_dynamic_range_like(scaled, input_tensor)

    bias_const = network.add_constant(
        shape=(1, 1, normalized_shape),
        weights=trt.Weights(bias.reshape(1, 1, -1))
    )
    bias_const.name = f"{name}_beta"

    add_bias_layer = network.add_elementwise(
        input1=scaled,
        input2=bias_const.get_output(0),
        op=trt.ElementWiseOperation.SUM
    )
    add_bias_layer.name = f"{name}_add_bias"
    output = add_bias_layer.get_output(0)
    set_tensor_dynamic_range(output, output_scale)

    if precision is not None:
        add_bias_layer.precision = precision

    set_tensor_dynamic_range(output, output_scale)

    return output, output_scale


def add_int_softmax(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    input_scale: float,
    output_scale: float,
    axis: int,
    name: str
) -> Tuple[trt.ITensor, float]:
    """
    添加整数Softmax层 (I-Softmax)
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        input_scale: 输入量化缩放因子
        output_scale: 输出量化缩放因子
        axis: Softmax操作的维度
        name: 层名称
        
    Returns:
        (输出张量, 输出缩放因子)
    """
    # 处理负数axis
    input_dims = input_tensor.shape
    if hasattr(input_dims, "nbDims"):
        ndim = input_dims.nbDims
    else:
        try:
            ndim = len(input_dims)
        except Exception:
            ndim = 0

    if axis < 0:
        if ndim > 0:
            axis = ndim + axis
        else:
            axis = 0

    if axis < 0:
        axis = 0
    
    # TensorRT的Softmax层
    softmax_layer = network.add_softmax(input=input_tensor)
    softmax_layer.axes = 1 << axis  # 转换为位掩码
    softmax_layer.name = f"{name}_softmax"
    output = softmax_layer.get_output(0)
    output.name = f"{name}_softmax_out"
    
    # 添加QDQ节点
    output = add_qdq_layer(network, output, output_scale, name)
    
    return output, output_scale


def add_int_softmax_direct(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    input_scale: float,
    output_scale: float,
    axis: int,
    name: str,
    precision: Optional[trt.DataType] = None
) -> Tuple[trt.ITensor, float]:
    """
    添加Softmax（不使用QDQ，仅设置动态范围）。
    """
    input_dims = input_tensor.shape
    if hasattr(input_dims, "nbDims"):
        ndim = input_dims.nbDims
    else:
        try:
            ndim = len(input_dims)
        except Exception:
            ndim = 0
    if axis < 0:
        if ndim > 0:
            axis = ndim + axis
        else:
            axis = 0

    if axis < 0:
        axis = 0

    softmax_layer = network.add_softmax(input=input_tensor)
    softmax_layer.axes = 1 << axis
    softmax_layer.name = f"{name}_softmax"
    output = softmax_layer.get_output(0)
    output.name = f"{name}_softmax_out"

    if precision is not None:
        softmax_layer.precision = precision

    set_tensor_dynamic_range(output, output_scale)

    return output, output_scale


def add_int_gelu(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    input_scale: float,
    output_scale: float,
    name: str
) -> Tuple[trt.ITensor, float]:
    """
    添加整数GELU激活函数 (I-GELU)
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        input_scale: 输入量化缩放因子
        output_scale: 输出量化缩放因子
        name: 层名称
        
    Returns:
        (输出张量, 输出缩放因子)
    """
    # 使用TensorRT的GELU plugin或近似实现
    # GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
    # 近似: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    # 使用TensorRT的activation layer
    gelu_layer = network.add_activation(
        input=input_tensor,
        type=trt.ActivationType.GELU_ERF  # 使用ERF版本的GELU
    )
    gelu_layer.name = f"{name}_gelu"
    output = gelu_layer.get_output(0)
    output.name = f"{name}_gelu_out"
    
    # 添加QDQ节点
    output = add_qdq_layer(network, output, output_scale, name)
    
    return output, output_scale


def add_int_gelu_direct(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    input_scale: float,
    output_scale: float,
    name: str,
    precision: Optional[trt.DataType] = None
) -> Tuple[trt.ITensor, float]:
    """
    添加GELU（不使用QDQ，仅设置动态范围）。
    """
    gelu_layer = network.add_activation(
        input=input_tensor,
        type=trt.ActivationType.GELU_ERF
    )
    gelu_layer.name = f"{name}_gelu"
    output = gelu_layer.get_output(0)
    output.name = f"{name}_gelu_out"

    if precision is not None:
        gelu_layer.precision = precision

    set_tensor_dynamic_range(output, output_scale)

    return output, output_scale


def add_int8_matmul(
    network: trt.INetworkDefinition,
    input1: trt.ITensor,
    input2: trt.ITensor,
    input1_scale: float,
    input2_scale: float,
    output_scale: float,
    name: str,
    add_output_qdq: bool = True
) -> Tuple[trt.ITensor, float]:
    """
    添加INT8矩阵乘法
    
    Args:
        network: TensorRT网络定义
        input1: 输入张量1
        input2: 输入张量2
        input1_scale: 输入1的量化缩放因子
        input2_scale: 输入2的量化缩放因子
        output_scale: 输出量化缩放因子
        name: 层名称
        
    Returns:
        (输出张量, 输出缩放因子)
    """
    # 矩阵乘法层
    matmul_layer = network.add_matrix_multiply(
        input0=input1,
        op0=trt.MatrixOperation.NONE,
        input1=input2,
        op1=trt.MatrixOperation.NONE
    )
    matmul_layer.name = f"{name}_matmul"
    output = matmul_layer.get_output(0)
    output.name = f"{name}_matmul_out"
    
    # 添加QDQ节点
    output = add_qdq_layer(network, output, output_scale, name) if add_output_qdq else output
    
    return output, output_scale


def add_int8_matmul_direct(
    network: trt.INetworkDefinition,
    input1: trt.ITensor,
    input2: trt.ITensor,
    input1_scale: float,
    input2_scale: float,
    output_scale: float,
    name: str,
    precision: Optional[trt.DataType] = trt.DataType.INT8
) -> Tuple[trt.ITensor, float]:
    """
    添加INT8矩阵乘法（不使用QDQ）。
    """
    set_tensor_dynamic_range(input1, input1_scale)
    set_tensor_dynamic_range(input2, input2_scale)

    matmul_layer = network.add_matrix_multiply(
        input0=input1,
        op0=trt.MatrixOperation.NONE,
        input1=input2,
        op1=trt.MatrixOperation.NONE
    )
    matmul_layer.name = f"{name}_matmul"

    if precision is not None:
        matmul_layer.precision = precision
        try:
            matmul_layer.set_output_type(0, precision)
        except Exception as exc:
            raise RuntimeError(f"Failed to set output type for {name}_matmul") from exc

    output = matmul_layer.get_output(0)
    output.name = f"{name}_matmul_out"

    set_tensor_dynamic_range(output, output_scale)

    return output, output_scale


def add_reshape(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    target_shape: Tuple[int, ...],
    name: str
) -> trt.ITensor:
    """
    添加Reshape层
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        target_shape: 目标形状
        name: 层名称
        
    Returns:
        重塑后的张量
    """
    shuffle_layer = network.add_shuffle(input=input_tensor)
    shuffle_layer.reshape_dims = target_shape
    shuffle_layer.name = f"{name}_reshape"
    output = shuffle_layer.get_output(0)
    output.name = f"{name}_reshape_out"
    set_tensor_dynamic_range_like(output, input_tensor)
    return output


def add_transpose(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    permutation: Tuple[int, ...],
    name: str
) -> trt.ITensor:
    """
    添加Transpose层
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        permutation: 维度置换顺序
        name: 层名称
        
    Returns:
        转置后的张量
    """
    shuffle_layer = network.add_shuffle(input=input_tensor)
    shuffle_layer.first_transpose = permutation
    shuffle_layer.name = f"{name}_transpose"
    output = shuffle_layer.get_output(0)
    output.name = f"{name}_transpose_out"
    set_tensor_dynamic_range_like(output, input_tensor)
    return output


def add_elementwise_add(
    network: trt.INetworkDefinition,
    input1: trt.ITensor,
    input2: trt.ITensor,
    name: str
) -> trt.ITensor:
    """
    添加逐元素加法层
    
    Args:
        network: TensorRT网络定义
        input1: 输入张量1
        input2: 输入张量2
        name: 层名称
        
    Returns:
        相加后的张量
    """
    add_layer = network.add_elementwise(
        input1=input1,
        input2=input2,
        op=trt.ElementWiseOperation.SUM
    )
    add_layer.name = f"{name}_add"
    output = add_layer.get_output(0)
    output.name = f"{name}_add_out"
    # Propagate a conservative dynamic range if available
    set_tensor_dynamic_range_like(output, input1)
    return output


def add_scale(
    network: trt.INetworkDefinition,
    input_tensor: trt.ITensor,
    scale: float,
    name: str
) -> trt.ITensor:
    """
    添加缩放层
    
    Args:
        network: TensorRT网络定义
        input_tensor: 输入张量
        scale: 缩放因子
        name: 层名称
        
    Returns:
        缩放后的张量
    """
    input_dims = input_tensor.shape
    if hasattr(input_dims, "nbDims"):
        ndim = input_dims.nbDims
    else:
        try:
            ndim = len(input_dims)
        except Exception:
            ndim = 0

    if ndim > 0:
        shape = tuple([1] * ndim)
    else:
        shape = (1,)

    scale_const = network.add_constant(
        shape=shape,
        weights=trt.Weights(np.array([scale], dtype=np.float32))
    )
    scale_const.name = f"{name}_scale_const"
    set_tensor_dynamic_range(scale_const.get_output(0), abs(scale) if scale != 0 else 1.0)
    
    scale_layer = network.add_elementwise(
        input1=input_tensor,
        input2=scale_const.get_output(0),
        op=trt.ElementWiseOperation.PROD
    )
    scale_layer.name = f"{name}_scale"
    output = scale_layer.get_output(0)
    output.name = f"{name}_scale_out"
    set_tensor_dynamic_range_like(output, input_tensor, scale=scale)
    return output


def load_quantization_params(qparams_file: str) -> Dict[str, Any]:
    """
    加载量化参数文件
    
    Args:
        qparams_file: 量化参数JSON文件路径
        
    Returns:
        量化参数字典
    """
    import json
    with open(qparams_file, 'r') as f:
        qparams = json.load(f)
    return qparams


def set_layer_precision(
    layer: trt.ILayer,
    precision: trt.DataType,
    output_index: int = 0
):
    """
    设置层的执行精度
    
    Args:
        layer: TensorRT层
        precision: 期望的精度 (trt.DataType.INT8, trt.DataType.FP16, trt.DataType.FP32)
        output_index: 输出索引
        
    使用示例:
        fc_layer = network.add_fully_connected(...)
        set_layer_precision(fc_layer, trt.DataType.INT8)
    """
    layer.precision = precision
    try:
        layer.set_output_type(output_index, precision)
    except Exception as exc:
        raise RuntimeError(f"Failed to set output type for {layer.name}") from exc


def set_layers_precision_by_pattern(
    network: trt.INetworkDefinition,
    pattern: str,
    precision: trt.DataType
):
    """
    根据层名称模式设置多个层的精度
    
    Args:
        network: TensorRT网络定义
        pattern: 层名称匹配模式（支持通配符）
        precision: 期望的精度
        
    使用示例:
        # 将所有attention层设置为FP16
        set_layers_precision_by_pattern(network, "*attn*", trt.DataType.FP16)
        
        # 将所有matmul层设置为INT8
        set_layers_precision_by_pattern(network, "*matmul*", trt.DataType.INT8)
    """
    import fnmatch
    
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if fnmatch.fnmatch(layer.name, pattern):
            set_layer_precision(layer, precision)


def get_precision_constraints_dict() -> Dict[str, trt.DataType]:
    """
    获取常见的精度约束配置字典
    
    Returns:
        精度配置字典，key为层名称模式，value为精度类型
        
    使用示例:
        precision_config = get_precision_constraints_dict()
        # 修改配置
        precision_config["*norm*"] = trt.DataType.FP16
        
        # 应用配置
        apply_precision_constraints(network, precision_config)
    """
    return {
        # 默认所有层使用INT8
        "*": trt.DataType.INT8,
        
        # LayerNorm可能需要FP16以保持精度
        "*norm*": trt.DataType.FP16,
        
        # Softmax通常需要FP16
        "*softmax*": trt.DataType.FP16,
        
        # 某些激活函数可能需要FP16
        "*gelu*": trt.DataType.FP16,
    }


def apply_precision_constraints(
    network: trt.INetworkDefinition,
    precision_config: Dict[str, trt.DataType]
):
    """
    应用精度约束配置到整个网络
    
    Args:
        network: TensorRT网络定义
        precision_config: 精度配置字典（层名称模式 -> 精度）
        
    使用示例:
        config = {
            "*matmul*": trt.DataType.INT8,
            "*norm*": trt.DataType.FP16,
            "*softmax*": trt.DataType.FP32,
        }
        apply_precision_constraints(network, config)
    """
    for pattern, precision in precision_config.items():
        set_layers_precision_by_pattern(network, pattern, precision)
