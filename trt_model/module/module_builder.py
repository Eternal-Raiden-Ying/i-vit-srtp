"""
TensorRT 模块构建器
提供Attention和Block模块的TensorRT实现，对应PyTorch模型中的Attention和Block类
"""

import tensorrt as trt
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .utils import (
    add_int8_linear,
    add_int8_linear_direct,
    add_int_layernorm,
    add_int_layernorm_direct,
    add_int_softmax,
    add_int_softmax_direct,
    add_int_gelu,
    add_int_gelu_direct,
    add_int8_matmul,
    add_int8_matmul_direct,
    add_qdq_layer,
    add_reshape,
    add_transpose,
    add_elementwise_add,
    add_scale,
    get_scale_and_zero_point,
    normalize_scale,
    resolve_quantized_weight,
    set_tensor_dynamic_range
)


class AttentionBuilder:
    """
    对应PyTorch的Attention类
    构建ViT的Multi-Head Self-Attention模块
    """
    
    def __init__(
        self,
        network: trt.INetworkDefinition,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        qparams: Optional[Dict[str, Any]] = None,
        prefix: str = "attn",
        use_qdq: bool = False
    ):
        """
        Args:
            network: TensorRT网络定义
            dim: 嵌入维度
            num_heads: 注意力头数
            qkv_bias: QKV线性层是否使用偏置
            qk_scale: QK缩放因子，如果为None则使用head_dim ** -0.5
            qparams: 量化参数字典
            prefix: 层名称前缀
        """
        self.network = network
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5
        self.qparams = qparams or {}
        self.prefix = prefix
        self.use_qdq = use_qdq
        
    def build_export_mode(
        self,
        input_tensor: trt.ITensor,
        input_scale: float,
        weights: Dict[str, np.ndarray]
    ) -> Tuple[trt.ITensor, float]:
        """
        构建导出模式的Attention (分离的Q/K/V)
        对应PyTorch的export_forward方法
        
        Args:
            input_tensor: 输入张量 (B, N, C)
            input_scale: 输入量化缩放因子
            weights: 权重字典，包含q_linear, k_linear, v_linear, proj等
            
        Returns:
            (输出张量, 输出缩放因子)
        """
        input_dims = input_tensor.shape
        num_dims = None
        if hasattr(input_dims, "nbDims"):
            num_dims = input_dims.nbDims
        else:
            try:
                num_dims = len(input_dims)
            except Exception:
                num_dims = 0

        if num_dims >= 3:
            B, N, C = input_dims[0], input_dims[1], input_dims[2]
        else:
            # 动态shape或未知shape时，用0占位表示从输入拷贝
            B, N, C = 0, 0, 0
        
        # Q linear
        q_weight_entry = weights[f"{self.prefix}.q_linear.weight"]
        q_bias = weights.get(f"{self.prefix}.q_linear.bias") if self.qkv_bias else None
        q_weight, q_weight_scale, q_weight_zp = resolve_quantized_weight(
            q_weight_entry,
            self.qparams,
            f"{self.prefix}.q_linear.weight_scale"
        )
        q_weight_scale = normalize_scale(q_weight_scale)
        q_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.q_qact.scale",
            1.0
        )

        if self.use_qdq:
            q, q_scale = add_int8_linear(
                self.network, input_tensor, q_weight, q_bias,
                input_scale, q_weight_scale, q_output_scale,
                f"{self.prefix}_q",
                use_per_channel=True,
                add_input_qdq=False  # 输入已由QAct量化
            )
        else:
            q, q_scale = add_int8_linear_direct(
                self.network, input_tensor, q_weight, q_bias,
                input_scale, q_weight_scale, q_output_scale,
                f"{self.prefix}_q",
                weight_zero_point=q_weight_zp,
                use_per_channel=True, precision=trt.DataType.INT8
            )
        
        # K linear (scale已融合到权重中)
        k_weight_entry = weights[f"{self.prefix}.k_linear.weight"]
        k_bias = weights.get(f"{self.prefix}.k_linear.bias") if self.qkv_bias else None
        k_weight, k_weight_scale, k_weight_zp = resolve_quantized_weight(
            k_weight_entry,
            self.qparams,
            f"{self.prefix}.k_linear.weight_scale"
        )
        k_weight_scale = normalize_scale(k_weight_scale)

        # 融合scale: 对浮点权重直接乘，量化权重则放大scale
        if isinstance(k_weight, np.ndarray) and k_weight.dtype == np.int8:
            k_weight_scale = k_weight_scale * np.sqrt(self.scale)
        else:
            k_weight = k_weight * np.sqrt(self.scale)

        k_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.k_qact.scale",
            1.0
        )

        if self.use_qdq:
            k, k_scale = add_int8_linear(
                self.network, input_tensor, k_weight, k_bias,
                input_scale, k_weight_scale, k_output_scale,
                f"{self.prefix}_k",
                add_input_qdq=False
            )
        else:
            k, k_scale = add_int8_linear_direct(
                self.network, input_tensor, k_weight, k_bias,
                input_scale, k_weight_scale, k_output_scale,
                f"{self.prefix}_k",
                weight_zero_point=k_weight_zp, precision=trt.DataType.INT8
            )
        
        # V linear
        v_weight_entry = weights[f"{self.prefix}.v_linear.weight"]
        v_bias = weights.get(f"{self.prefix}.v_linear.bias") if self.qkv_bias else None
        v_weight, v_weight_scale, v_weight_zp = resolve_quantized_weight(
            v_weight_entry,
            self.qparams,
            f"{self.prefix}.v_linear.weight_scale"
        )
        v_weight_scale = normalize_scale(v_weight_scale)
        v_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.v_qact.scale",
            1.0
        )

        if self.use_qdq:
            v, v_scale = add_int8_linear(
                self.network, input_tensor, v_weight, v_bias,
                input_scale, v_weight_scale, v_output_scale,
                f"{self.prefix}_v",
                add_input_qdq=False
            )
        else:
            v, v_scale = add_int8_linear_direct(
                self.network, input_tensor, v_weight, v_bias,
                input_scale, v_weight_scale, v_output_scale,
                f"{self.prefix}_v",
                weight_zero_point=v_weight_zp, precision=trt.DataType.INT8
            )
        
        # Reshape Q: (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = add_reshape(self.network, q, (B, N, self.num_heads, self.head_dim), f"{self.prefix}_q_reshape")
        q = add_transpose(self.network, q, (0, 2, 1, 3), f"{self.prefix}_q_transpose")
        
        # Reshape K: (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, head_dim, N)
        k = add_reshape(self.network, k, (B, N, self.num_heads, self.head_dim), f"{self.prefix}_k_reshape")
        k = add_transpose(self.network, k, (0, 2, 3, 1), f"{self.prefix}_k_transpose")
        
        # Reshape V: (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        v = add_reshape(self.network, v, (B, N, self.num_heads, self.head_dim), f"{self.prefix}_v_reshape")
        v = add_transpose(self.network, v, (0, 2, 1, 3), f"{self.prefix}_v_transpose")
        
        # Q @ K^T -> attn (B, num_heads, N, N)
        attn_matmul1_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.matmul_1.output_scale",
            1.0
        )
        if self.use_qdq:
            attn, attn_scale = add_int8_matmul(
                self.network, q, k,
                q_scale, k_scale, attn_matmul1_scale,
                f"{self.prefix}_qk",
                add_output_qdq=False
            )
        else:
            attn, attn_scale = add_int8_matmul_direct(
                self.network, q, k,
                q_scale, k_scale, attn_matmul1_scale,
                f"{self.prefix}_qk", precision=trt.DataType.INT8
            )
        
        # QAct after matmul
        qact_attn_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact_attn1.scale",
            1.0
        )
        if self.use_qdq:
            attn = add_qdq_layer(self.network, attn, qact_attn_scale, f"{self.prefix}_qact_attn1")
        else:
            set_tensor_dynamic_range(attn, qact_attn_scale)
        attn_scale = qact_attn_scale
        
        # Softmax
        softmax_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact_softmax.scale",
            1.0
        )
        if self.use_qdq:
            attn, attn_scale = add_int_softmax(
                self.network, attn, attn_scale, softmax_scale,
                axis=-1, name=f"{self.prefix}"
            )
        else:
            attn, attn_scale = add_int_softmax_direct(
                self.network, attn, attn_scale, softmax_scale,
                axis=-1, name=f"{self.prefix}"
            )
        
        # attn @ V -> output (B, num_heads, N, head_dim)
        matmul2_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.matmul_2.output_scale",
            1.0
        )
        if self.use_qdq:
            x, x_scale = add_int8_matmul(
                self.network, attn, v,
                attn_scale, v_scale, matmul2_scale,
                f"{self.prefix}_attn_v",
                add_output_qdq=False
            )
        else:
            x, x_scale = add_int8_matmul_direct(
                self.network, attn, v,
                attn_scale, v_scale, matmul2_scale,
                f"{self.prefix}_attn_v"
            )
        
        # Transpose: (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim)
        x = add_transpose(self.network, x, (0, 2, 1, 3), f"{self.prefix}_output_transpose")
        
        # Reshape: (B, N, num_heads, head_dim) -> (B, N, C)
        x = add_reshape(self.network, x, (B, N, C), f"{self.prefix}_output_reshape")
        
        # QAct2
        qact2_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact2.scale",
            1.0
        )
        if self.use_qdq:
            x = add_qdq_layer(self.network, x, qact2_scale, f"{self.prefix}_qact2")
        else:
            set_tensor_dynamic_range(x, qact2_scale)
        x_scale = qact2_scale
        
        # Projection
        proj_weight_entry = weights[f"{self.prefix}.proj.weight"]
        proj_bias = weights.get(f"{self.prefix}.proj.bias")
        proj_weight, proj_weight_scale, proj_weight_zp = resolve_quantized_weight(
            proj_weight_entry,
            self.qparams,
            f"{self.prefix}.proj.weight_scale"
        )
        proj_weight_scale = normalize_scale(proj_weight_scale)
        proj_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact3.scale",
            1.0
        )
        
        if self.use_qdq:
            x, x_scale = add_int8_linear(
                self.network, x, proj_weight, proj_bias,
                x_scale, proj_weight_scale, proj_output_scale,
                f"{self.prefix}_proj"
            )
        else:
            x, x_scale = add_int8_linear_direct(
                self.network, x, proj_weight, proj_bias,
                x_scale, proj_weight_scale, proj_output_scale,
                f"{self.prefix}_proj",
                weight_zero_point=proj_weight_zp
            )
        
        return x, x_scale


class MlpBuilder:
    """
    对应PyTorch的Mlp类
    构建ViT的MLP模块 (FFN - Feed Forward Network)
    """
    
    def __init__(
        self,
        network: trt.INetworkDefinition,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        qparams: Optional[Dict[str, Any]] = None,
        prefix: str = "mlp",
        use_qdq: bool = False
    ):
        """
        Args:
            network: TensorRT网络定义
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度
            out_features: 输出特征维度，默认等于in_features
            qparams: 量化参数字典
            prefix: 层名称前缀
        """
        self.network = network
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.qparams = qparams or {}
        self.prefix = prefix
        self.use_qdq = use_qdq
        
    def build(
        self,
        input_tensor: trt.ITensor,
        input_scale: float,
        weights: Dict[str, np.ndarray]
    ) -> Tuple[trt.ITensor, float]:
        """
        构建MLP模块
        
        Args:
            input_tensor: 输入张量
            input_scale: 输入量化缩放因子
            weights: 权重字典，包含fc1, fc2等
            
        Returns:
            (输出张量, 输出缩放因子)
        """
        # FC1
        fc1_weight_entry = weights[f"{self.prefix}.fc1.weight"]
        fc1_bias = weights.get(f"{self.prefix}.fc1.bias")
        fc1_weight, fc1_weight_scale, fc1_weight_zp = resolve_quantized_weight(
            fc1_weight_entry,
            self.qparams,
            f"{self.prefix}.fc1.weight_scale"
        )
        fc1_weight_scale = normalize_scale(fc1_weight_scale)
        fc1_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact1.scale",
            1.0
        )
        
        if self.use_qdq:
            x, x_scale = add_int8_linear(
                self.network, input_tensor, fc1_weight, fc1_bias,
                input_scale, fc1_weight_scale, fc1_output_scale,
                f"{self.prefix}_fc1"
            )
        else:
            x, x_scale = add_int8_linear_direct(
                self.network, input_tensor, fc1_weight, fc1_bias,
                input_scale, fc1_weight_scale, fc1_output_scale,
                f"{self.prefix}_fc1",
                weight_zero_point=fc1_weight_zp
            )
        
        # GELU
        gelu_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.act.scale",
            1.0
        )
        if self.use_qdq:
            x, x_scale = add_int_gelu(
                self.network, x, x_scale, gelu_output_scale,
                f"{self.prefix}"
            )
        else:
            x, x_scale = add_int_gelu_direct(
                self.network, x, x_scale, gelu_output_scale,
                f"{self.prefix}"
            )
        
        # FC2
        fc2_weight_entry = weights[f"{self.prefix}.fc2.weight"]
        fc2_bias = weights.get(f"{self.prefix}.fc2.bias")
        fc2_weight, fc2_weight_scale, fc2_weight_zp = resolve_quantized_weight(
            fc2_weight_entry,
            self.qparams,
            f"{self.prefix}.fc2.weight_scale"
        )
        fc2_weight_scale = normalize_scale(fc2_weight_scale)
        fc2_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact2.scale",
            1.0
        )
        
        if self.use_qdq:
            x, x_scale = add_int8_linear(
                self.network, x, fc2_weight, fc2_bias,
                x_scale, fc2_weight_scale, fc2_output_scale,
                f"{self.prefix}_fc2"
            )
        else:
            x, x_scale = add_int8_linear_direct(
                self.network, x, fc2_weight, fc2_bias,
                x_scale, fc2_weight_scale, fc2_output_scale,
                f"{self.prefix}_fc2",
                weight_zero_point=fc2_weight_zp
            )
        
        return x, x_scale


class BlockBuilder:
    """
    对应PyTorch的Block类
    构建ViT的Transformer Block模块
    """
    
    def __init__(
        self,
        network: trt.INetworkDefinition,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        qparams: Optional[Dict[str, Any]] = None,
        prefix: str = "block",
        use_qdq: bool = False
    ):
        """
        Args:
            network: TensorRT网络定义
            dim: 嵌入维度
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度相对于dim的比例
            qkv_bias: QKV线性层是否使用偏置
            qk_scale: QK缩放因子
            qparams: 量化参数字典
            prefix: 层名称前缀
        """
        self.network = network
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.qparams = qparams or {}
        self.prefix = prefix
        self.use_qdq = use_qdq
        
        # 创建子模块构建器
        self.attn_builder = AttentionBuilder(
            network, dim, num_heads, qkv_bias, qk_scale,
            qparams, f"{prefix}.attn", use_qdq=use_qdq
        )
        
        self.mlp_builder = MlpBuilder(
            network, dim, self.mlp_hidden_dim, dim,
            qparams, f"{prefix}.mlp", use_qdq=use_qdq
        )
        
    def build(
        self,
        input_tensor: trt.ITensor,
        input_scale: float,
        weights: Dict[str, np.ndarray]
    ) -> Tuple[trt.ITensor, float]:
        """
        构建Block模块
        
        结构:
        x -> LayerNorm1 -> QAct1 -> Attention -> QAct2 (残差+量化) -> 
        -> LayerNorm2 -> QAct3 -> MLP -> QAct4 (残差+量化) -> output
        
        Args:
            input_tensor: 输入张量 (B, N, C)
            input_scale: 输入量化缩放因子
            weights: 权重字典
            
        Returns:
            (输出张量, 输出缩放因子)
        """
        x_1 = input_tensor
        act_scaling_factor_1 = input_scale
        
        # ===== Attention分支 =====
        # LayerNorm1
        norm1_weight = weights[f"{self.prefix}.norm1.weight"]
        norm1_bias = weights[f"{self.prefix}.norm1.bias"]
        norm1_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.norm1.output_scale",
            1.0
        )
        
        if self.use_qdq:
            x, act_scaling_factor = add_int_layernorm(
                self.network, x_1, self.dim,
                norm1_weight, norm1_bias, 1e-6,
                act_scaling_factor_1, norm1_output_scale,
                f"{self.prefix}_norm1",
                add_output_qdq=False
            )
        else:
            x, act_scaling_factor = add_int_layernorm_direct(
                self.network, x_1, self.dim,
                norm1_weight, norm1_bias, 1e-6,
                act_scaling_factor_1, norm1_output_scale,
                f"{self.prefix}_norm1"
            )
        
        # QAct1
        qact1_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact1.scale",
            1.0
        )
        if self.use_qdq:
            x = add_qdq_layer(self.network, x, qact1_scale, f"{self.prefix}_qact1")
        else:
            set_tensor_dynamic_range(x, qact1_scale)
        act_scaling_factor = qact1_scale
        
        # Attention
        x, act_scaling_factor = self.attn_builder.build_export_mode(
            x, act_scaling_factor, weights
        )
        
        # QAct2 (残差连接 + 量化)
        # 注意: 这里需要将x和x_1缩放到同一个量化域
        qact2_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact2.scale",
            1.0
        )
        
        # 将x_1缩放到qact2的量化域
        x_1_scaled = add_scale(self.network, x_1, act_scaling_factor_1 / qact2_scale, f"{self.prefix}_x1_scale")
        # 将x缩放到qact2的量化域
        x_scaled = add_scale(self.network, x, act_scaling_factor / qact2_scale, f"{self.prefix}_x_scale")
        # 残差连接
        x_2 = add_elementwise_add(self.network, x_scaled, x_1_scaled, f"{self.prefix}_residual1")
        # 量化
        if self.use_qdq:
            x_2 = add_qdq_layer(self.network, x_2, qact2_scale, f"{self.prefix}_qact2")
        else:
            set_tensor_dynamic_range(x_2, qact2_scale)
        act_scaling_factor_2 = qact2_scale
        
        # ===== MLP分支 =====
        # LayerNorm2
        norm2_weight = weights[f"{self.prefix}.norm2.weight"]
        norm2_bias = weights[f"{self.prefix}.norm2.bias"]
        norm2_output_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.norm2.output_scale",
            1.0
        )
        
        if self.use_qdq:
            x, act_scaling_factor = add_int_layernorm(
                self.network, x_2, self.dim,
                norm2_weight, norm2_bias, 1e-6,
                act_scaling_factor_2, norm2_output_scale,
                f"{self.prefix}_norm2",
                add_output_qdq=False
            )
        else:
            x, act_scaling_factor = add_int_layernorm_direct(
                self.network, x_2, self.dim,
                norm2_weight, norm2_bias, 1e-6,
                act_scaling_factor_2, norm2_output_scale,
                f"{self.prefix}_norm2"
            )
        
        # QAct3
        qact3_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact3.scale",
            1.0
        )
        if self.use_qdq:
            x = add_qdq_layer(self.network, x, qact3_scale, f"{self.prefix}_qact3")
        else:
            set_tensor_dynamic_range(x, qact3_scale)
        act_scaling_factor = qact3_scale
        
        # MLP
        x, act_scaling_factor = self.mlp_builder.build(
            x, act_scaling_factor, weights
        )
        
        # QAct4 (残差连接 + 量化)
        qact4_scale, _ = get_scale_and_zero_point(
            self.qparams,
            f"{self.prefix}.qact4.scale",
            1.0
        )
        
        # 将x_2缩放到qact4的量化域
        x_2_scaled = add_scale(self.network, x_2, act_scaling_factor_2 / qact4_scale, f"{self.prefix}_x2_scale")
        # 将x缩放到qact4的量化域
        x_scaled = add_scale(self.network, x, act_scaling_factor / qact4_scale, f"{self.prefix}_x_mlp_scale")
        # 残差连接
        x_out = add_elementwise_add(self.network, x_scaled, x_2_scaled, f"{self.prefix}_residual2")
        # 量化
        if self.use_qdq:
            x_out = add_qdq_layer(self.network, x_out, qact4_scale, f"{self.prefix}_qact4")
        else:
            set_tensor_dynamic_range(x_out, qact4_scale)
        act_scaling_factor_out = qact4_scale
        
        return x_out, act_scaling_factor_out
