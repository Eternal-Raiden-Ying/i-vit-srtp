"""
TensorRT 工具函数模块
提供量化、激活函数、归一化等基础操作的TensorRT实现
"""

import tensorrt as trt
import numpy as np
import torch
import ctypes
from typing import Tuple, Optional, Dict, Any, Union



def normalize_scale(scale: Union[float, np.ndarray, list]) -> Union[float, np.ndarray]:
    if isinstance(scale, list):
        return np.array(scale, dtype=np.float32)
    return scale


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
    return q.astype(np.float32)

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
    add_layer = network.add_elementwise(
        input1=input1,
        input2=input2,
        op=trt.ElementWiseOperation.SUM
    )
    add_layer.name = f"{name}_int32_sum"
    add_layer.get_output(0).name = f"{name}_int32_sum_out"

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
    
    weight_cast = network.add_cast(weight_tensor, trt.DataType.INT8)
    weight_cast.name = f"{name}_weight_int8_cast"
    weight_cast_output = weight_cast.get_output(0)
    weight_cast_output.name = f"{name}_weight_int8_cast_out"
    
    # 如果输入是3D，需要将权重reshape为3D以匹配维度
    if len(input_tensor.shape) == 3:
        # 输入是 (batch, seq_len, in_features)
        # 权重从 (in_features, out_features) reshape为 (1, in_features, out_features)
        shuffle = network.add_shuffle(weight_cast_output)
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

    fc_output = fc_layer.get_output(0)
    fc_output.name = f"{name}_matmul_out"
    cast_layer = network.add_cast(fc_output, trt.DataType.INT32) 
    cast_layer.name = f"{name}_int32_cast"
    cast_output = cast_layer.get_output(0)
    cast_output.name = f"{name}_int32_cast_out"
    
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
            input1=cast_output,
            input2=bias_tensor,
            name=f"{name}_add_bias"
        )

        fc_output = add_bias_output
    
    return fc_output, output_scale

