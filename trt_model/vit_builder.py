"""
TensorRT Vision Transformer 构建器
提供完整的ViT模型构建，对应PyTorch模型中的VisionTransformer类
"""

import tensorrt as trt
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import json

from .module.module_builder import AttentionBuilder, MlpBuilder, BlockBuilder
from .module.utils import (
    add_int8_linear,
    add_int_layernorm,
    add_qdq_layer,
    add_reshape,
    add_transpose,
    add_elementwise_add,
    add_scale,
    load_quantization_params
)


class PatchEmbedBuilder:
    """
    对应PyTorch的PatchEmbed类
    将图像转换为patch embeddings
    """
    
    def __init__(
        self,
        network: trt.INetworkDefinition,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        qparams: Optional[Dict[str, Any]] = None,
        prefix: str = "patch_embed"
    ):
        """
        Args:
            network: TensorRT网络定义
            img_size: 输入图像大小
            patch_size: patch大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
            qparams: 量化参数字典
            prefix: 层名称前缀
        """
        self.network = network
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.qparams = qparams or {}
        self.prefix = prefix
        
    def build(
        self,
        input_tensor: trt.ITensor,
        input_scale: float,
        weights: Dict[str, np.ndarray]
    ) -> Tuple[trt.ITensor, float]:
        """
        构建PatchEmbed模块
        
        Args:
            input_tensor: 输入图像张量 (B, C, H, W)
            input_scale: 输入量化缩放因子
            weights: 权重字典，包含proj卷积层权重
            
        Returns:
            (输出张量 (B, N, embed_dim), 输出缩放因子)
        """
        # Projection: Conv2d with kernel_size=patch_size, stride=patch_size
        proj_weight = weights[f"{self.prefix}.proj.weight"]  # (embed_dim, in_chans, patch_size, patch_size)
        proj_bias = weights.get(f"{self.prefix}.proj.bias")
        proj_weight_scale = self.qparams.get(f"{self.prefix}.proj.weight_scale", 1.0)
        proj_output_scale = self.qparams.get(f"{self.prefix}.qact.scale", 1.0)
        
        # 添加卷积层
        conv_layer = self.network.add_convolution_nd(
            input=input_tensor,
            num_output_maps=self.embed_dim,
            kernel_shape=(self.patch_size, self.patch_size),
            kernel=trt.Weights(proj_weight),
            bias=trt.Weights(proj_bias) if proj_bias is not None else trt.Weights()
        )
        conv_layer.stride_nd = (self.patch_size, self.patch_size)
        conv_layer.name = f"{self.prefix}_proj_conv"
        x = conv_layer.get_output(0)
        x.name = f"{self.prefix}_proj_conv_out"
        
        # Flatten: (B, embed_dim, H//patch_size, W//patch_size) -> (B, embed_dim, N)
        B = input_tensor.shape[0]
        flatten_layer = self.network.add_shuffle(input=x)
        flatten_layer.reshape_dims = (B, self.embed_dim, self.num_patches)
        flatten_layer.name = f"{self.prefix}_flatten"
        x = flatten_layer.get_output(0)
        x.name = f"{self.prefix}_flatten_out"
        
        # Transpose: (B, embed_dim, N) -> (B, N, embed_dim)
        x = add_transpose(self.network, x, (0, 2, 1), f"{self.prefix}")
        
        # QAct
        x = add_qdq_layer(self.network, x, proj_output_scale, f"{self.prefix}_qact")
        act_scaling_factor = proj_output_scale
        
        return x, act_scaling_factor


class VisionTransformerBuilder:
    """
    对应PyTorch的VisionTransformer类
    构建完整的ViT模型
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        qparams_file: Optional[str] = None
    ):
        """
        Args:
            img_size: 输入图像大小
            patch_size: patch大小
            in_chans: 输入通道数
            num_classes: 分类类别数
            embed_dim: 嵌入维度
            depth: Transformer层数
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度相对于embed_dim的比例
            qkv_bias: QKV线性层是否使用偏置
            qk_scale: QK缩放因子
            qparams_file: 量化参数JSON文件路径
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        
        # 加载量化参数
        self.qparams = {}
        if qparams_file:
            self.qparams = load_quantization_params(qparams_file)
            
    def build(
        self,
        weights: Dict[str, np.ndarray],
        batch_size: int = 1,
        dtype: trt.DataType = trt.DataType.FLOAT
    ) -> Tuple[trt.INetworkDefinition, trt.IBuilder]:
        """
        构建完整的TensorRT网络
        
        Args:
            weights: 模型权重字典
            batch_size: 批次大小
            dtype: 数据类型
            
        Returns:
            (网络定义, builder)
        """
        # 创建builder和network
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # 定义输入
        input_shape = (batch_size, self.in_chans, self.img_size, self.img_size)
        input_tensor = network.add_input(
            name="input",
            dtype=dtype,
            shape=input_shape
        )
        
        # 构建网络
        output_tensor, output_scale = self._build_network(
            network, input_tensor, weights
        )
        
        # 标记输出
        network.mark_output(output_tensor)
        output_tensor.name = "output"
        
        return network, builder
    
    def _build_network(
        self,
        network: trt.INetworkDefinition,
        input_tensor: trt.ITensor,
        weights: Dict[str, np.ndarray]
    ) -> Tuple[trt.ITensor, float]:
        """
        构建网络的内部方法
        
        Args:
            network: TensorRT网络定义
            input_tensor: 输入张量
            weights: 模型权重字典
            
        Returns:
            (输出张量, 输出缩放因子)
        """
        B = input_tensor.shape[0]
        
        # ===== 输入量化 =====
        qact_input_scale = self.qparams.get("qact_input.scale", 1.0)
        x = add_qdq_layer(network, input_tensor, qact_input_scale, "qact_input")
        act_scaling_factor = qact_input_scale
        
        # ===== Patch Embedding =====
        patch_embed_builder = PatchEmbedBuilder(
            network, self.img_size, self.patch_size,
            self.in_chans, self.embed_dim, self.qparams
        )
        x, act_scaling_factor = patch_embed_builder.build(
            x, act_scaling_factor, weights
        )
        
        N = patch_embed_builder.num_patches
        
        # ===== CLS Token =====
        # 获取量化后的cls_token_integer
        cls_token_int = weights.get("cls_token_integer", weights["cls_token"])
        cls_token_const = network.add_constant(
            shape=(1, 1, self.embed_dim),
            weights=trt.Weights(cls_token_int)
        )
        cls_token_const.name = "cls_token"
        cls_token = cls_token_const.get_output(0)
        
        # 扩展到batch维度: (1, 1, embed_dim) -> (B, 1, embed_dim)
        # 使用Slice + Concatenate来实现expand
        cls_tokens_list = []
        for i in range(B):
            cls_tokens_list.append(cls_token)
        
        # Concatenate cls_token和patch_embeddings
        # x: (B, N, embed_dim), cls_token: (1, 1, embed_dim)
        # 需要先扩展cls_token到(B, 1, embed_dim)
        
        # 使用slice和concat来实现
        concat_layer = network.add_concatenation([cls_token, x])
        concat_layer.axis = 1
        concat_layer.name = "concat_cls_patches"
        x = concat_layer.get_output(0)  # (B, N+1, embed_dim)
        x.name = "cls_patches_concat_out"
        
        # ===== Position Embedding =====
        pos_embed = weights["pos_embed"]  # (1, N+1, embed_dim)
        
        # QAct_pos: 对pos_embed进行量化
        qact_pos_scale = self.qparams.get("qact_pos.scale", 1.0)
        pos_embed_const = network.add_constant(
            shape=(1, N + 1, self.embed_dim),
            weights=trt.Weights(pos_embed)
        )
        pos_embed_const.name = "pos_embed"
        x_pos = pos_embed_const.get_output(0)
        x_pos = add_qdq_layer(network, x_pos, qact_pos_scale, "qact_pos")
        act_scaling_factor_pos = qact_pos_scale
        
        # QAct1: 残差连接并量化 (x + pos_embed)
        qact1_scale = self.qparams.get("qact1.scale", 1.0)
        
        # 将x缩放到qact1的量化域
        x_scaled = add_scale(network, x, act_scaling_factor / qact1_scale, "x_scale_qact1")
        # 将x_pos缩放到qact1的量化域
        x_pos_scaled = add_scale(network, x_pos, act_scaling_factor_pos / qact1_scale, "x_pos_scale_qact1")
        # 残差连接
        x = add_elementwise_add(network, x_scaled, x_pos_scaled, "add_pos_embed")
        # 量化
        x = add_qdq_layer(network, x, qact1_scale, "qact1")
        act_scaling_factor = qact1_scale
        
        # ===== Transformer Blocks =====
        for i in range(self.depth):
            block_builder = BlockBuilder(
                network, self.embed_dim, self.num_heads,
                self.mlp_ratio, self.qkv_bias, self.qk_scale,
                self.qparams, f"blocks.{i}"
            )
            x, act_scaling_factor = block_builder.build(
                x, act_scaling_factor, weights
            )
        
        # ===== Final Norm =====
        norm_weight = weights["norm.weight"]
        norm_bias = weights["norm.bias"]
        norm_output_scale = self.qparams.get("norm.output_scale", 1.0)
        
        x, act_scaling_factor = add_int_layernorm(
            network, x, self.embed_dim,
            norm_weight, norm_bias, 1e-6,
            act_scaling_factor, norm_output_scale,
            "norm"
        )
        
        # ===== Extract CLS Token =====
        # x[:, 0] - 提取第一个token (CLS token)
        slice_layer = network.add_slice(
            input=x,
            start=(0, 0, 0),
            shape=(B, 1, self.embed_dim),
            stride=(1, 1, 1)
        )
        slice_layer.name = "extract_cls_token"
        x = slice_layer.get_output(0)
        x.name = "cls_token_extracted"
        
        # Squeeze: (B, 1, embed_dim) -> (B, embed_dim)
        x = add_reshape(network, x, (B, self.embed_dim), "squeeze_cls")
        
        # ===== QAct2 =====
        qact2_scale = self.qparams.get("qact2.scale", 1.0)
        x = add_qdq_layer(network, x, qact2_scale, "qact2")
        act_scaling_factor = qact2_scale
        
        # ===== Classification Head =====
        head_weight = weights["head.weight"]
        head_bias = weights.get("head.bias")
        head_weight_scale = self.qparams.get("head.weight_scale", 1.0)
        head_output_scale = self.qparams.get("act_out.scale", 1.0)
        
        x, act_scaling_factor = add_int8_linear(
            network, x, head_weight, head_bias,
            act_scaling_factor, head_weight_scale, head_output_scale,
            "head"
        )
        
        return x, act_scaling_factor
    
    def build_engine(
        self,
        weights: Dict[str, np.ndarray],
        batch_size: int = 1,
        dtype: trt.DataType = trt.DataType.FLOAT,
        workspace_size: int = 1 << 30,  # 1GB
        int8_mode: bool = True,
        fp16_mode: bool = False
    ) -> trt.ICudaEngine:
        """
        构建TensorRT引擎
        
        Args:
            weights: 模型权重字典
            batch_size: 批次大小
            dtype: 数据类型
            workspace_size: 工作空间大小
            int8_mode: 是否启用INT8模式
            fp16_mode: 是否启用FP16模式
            
        Returns:
            TensorRT引擎
        """
        network, builder = self.build(weights, batch_size, dtype)
        
        # 配置builder
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size
        
        if int8_mode:
            config.set_flag(trt.BuilderFlag.INT8)
            print("INT8 mode enabled")
            
        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
        
        # 构建引擎
        print("Building TensorRT engine...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
            
        print("TensorRT engine built successfully")
        return engine
    
    def save_engine(
        self,
        engine: trt.ICudaEngine,
        engine_file: str
    ):
        """
        保存TensorRT引擎到文件
        
        Args:
            engine: TensorRT引擎
            engine_file: 输出文件路径
        """
        with open(engine_file, "wb") as f:
            f.write(engine.serialize())
        print(f"Engine saved to {engine_file}")


def build_vit_base_patch16_224(
    weights: Dict[str, np.ndarray],
    qparams_file: str,
    batch_size: int = 1,
    int8_mode: bool = True
) -> trt.ICudaEngine:
    """
    构建ViT-Base/16模型的TensorRT引擎
    
    Args:
        weights: 模型权重字典
        qparams_file: 量化参数文件路径
        batch_size: 批次大小
        int8_mode: 是否启用INT8模式
        
    Returns:
        TensorRT引擎
    """
    builder = VisionTransformerBuilder(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qparams_file=qparams_file
    )
    
    engine = builder.build_engine(
        weights=weights,
        batch_size=batch_size,
        int8_mode=int8_mode
    )
    
    return engine


def build_vit_large_patch16_224(
    weights: Dict[str, np.ndarray],
    qparams_file: str,
    batch_size: int = 1,
    int8_mode: bool = True
) -> trt.ICudaEngine:
    """
    构建ViT-Large/16模型的TensorRT引擎
    
    Args:
        weights: 模型权重字典
        qparams_file: 量化参数文件路径
        batch_size: 批次大小
        int8_mode: 是否启用INT8模式
        
    Returns:
        TensorRT引擎
    """
    builder = VisionTransformerBuilder(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qparams_file=qparams_file
    )
    
    engine = builder.build_engine(
        weights=weights,
        batch_size=batch_size,
        int8_mode=int8_mode
    )
    
    return engine


def build_deit_small_patch16_224(
    weights: Dict[str, np.ndarray],
    qparams_file: str,
    batch_size: int = 1,
    int8_mode: bool = True
) -> trt.ICudaEngine:
    """
    构建DeiT-Small/16模型的TensorRT引擎
    
    Args:
        weights: 模型权重字典
        qparams_file: 量化参数文件路径
        batch_size: 批次大小
        int8_mode: 是否启用INT8模式
        
    Returns:
        TensorRT引擎
    """
    builder = VisionTransformerBuilder(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qparams_file=qparams_file
    )
    
    engine = builder.build_engine(
        weights=weights,
        batch_size=batch_size,
        int8_mode=int8_mode
    )
    
    return engine
