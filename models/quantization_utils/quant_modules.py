import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_utils import *

"""
    for full int8 inference, roadmap: pytorch -> onnx -> tensorrt
    1. export onnx map with QDQ node (QuantizeLinear, DeQuantizeLinear)
        we use (3) here
        (1) use torch.ao.quantize (mature tech but not suitable for custom quantization)
        (2) use pytorch-quantization (NVIDIA official, not explored yet)
        (3) use official op (torch.fake_quantize_per_tensor_affine / torch.fake_quantize_per_channel_affine)

        **NOTE**  QDQ node needs certain format
        input(float) -> Q -> (int8) -> DQ -> (float) -> FP ops -> Q -> (int8) -> DQ -> output(float)
                                                            -------------------------------
        after that, tensorRT automaticaly fuse ops (int8) ->|DQ -> (float) -> FP ops -> Q |-> (int8) into int8 ops
                                                            -------------------------------
        as a result, we don't need full int inference in pytorch model
        and what we need is to use fake quant to fulfuil QAT logic

        to achieve that, we add export_mode to every module for rightly export onnx map with QDQ node

    2. convert onnx QDQ model to int8 tensorrt engine with trtexec(full tensorRT) / python package(tensorrt already enough)

    3. check engine op, make sure all quantized ops are int8 ops

    4. run inference with tensorrt engine


    Question Log:
    onnx do not support 32bit QDQ, (qmin, qmax) only support (0-127, 0-255, -128-127)
"""


# TODO: quantlinear的per channel 维度可能不对  应该是对的（2.1）
#       quantact的per channel 根本用不到吧 （2.1）

class QuantLinear(nn.Linear):
    """
    用于量化给定 Linear 层权重的类。

    Parameters:
    ----------
    in_features : int
        输入特征的数量。
    out_features : int
        输出特征的数量。
    bias : bool, default True
        是否包含偏置。
    weight_bit : int, default 8
        量化权重的位宽。
    bias_bit : int, default None
        量化偏置的位宽。
    per_channel : bool, default True
        是否使用通道级别的量化。
    quant_mode : str, default 'symmetric'
        量化模式。'none' 表示不量化。
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 per_channel=True,
                 quant_mode='symmetric',
                 running_stat=True,
                 full_int_inference=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.running_stat = running_stat
        self.full_int_inference = full_int_inference

        self.export_mode = False

        # 根据量化模式设置量化函数
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric": # 根本没有这个部分的操作，这个I-ViT到底是哪里来的？
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        # 注册缓冲区
        """
        在PyTorch中，模型的参数通常是通过torch.nn.Parameter类型来管理的，这些参数会在优化过程中进行更新。
        但有时我们需要在模型中存储一些不需要优化的固定值，比如缩放因子、量化后的参数等，这时就可以使用register_buffer方法。
        """
        # NOTE 按照通道量化，所以缩放因子是一个和FC输出特征数量相同的向量 
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('bias_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_zp', torch.zeros_like(self.fc_scaling_factor, dtype=torch.int))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        self.running_stat = False
        
    def unfix(self):
        self.running_stat = True
        
    def full_int_model(self):
        self.full_int_inference = True
    
    def unfull_int_model(self):
        self.full_int_inference = False

    def forward(self, x, prev_act_scaling_factor=None):

        if self.export_mode:
            return self.export_forward(x, prev_act_scaling_factor)

        with torch.no_grad():
            if self.running_stat:
                if self.per_channel:
                    w = self.weight
                    # 计算每个通道的最小值和最大值
                    v = w.reshape(w.shape[0], -1)
                    cur_min = v.min(axis=1).values
                    cur_max = v.max(axis=1).values
                    self.min_val = cur_min
                    self.max_val = cur_max   
                                
                    # 计算线性量化参数
                    self.fc_scaling_factor = symmetric_linear_quantization_params(
                        self.weight_bit, self.min_val, self.max_val)
                    # 对权重进行线性量化
                    self.weight_integer = self.weight_function(
                        self.weight, self.weight_bit, self.fc_scaling_factor, True)
                    
                    # 计算偏置的缩放因子
                    # NOTE 偏执是加上的，所以量化因子需要再与激活值量化因子组合, 即输出对应的量化因子
                    self.bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor                
                    # 对偏置进行线性量化
                    if self.bias is not None:
                        self.bias_integer = self.weight_function(
                            self.bias, self.bias_bit, self.bias_scaling_factor, True)
                    else:
                        self.bias_integer = None
    
                else:
                    raise Exception('For weight, we only support per_channel quantization.')
                
                # just for a clean onnx map
                self.weight_zp = torch.zeros_like(self.fc_scaling_factor, dtype=torch.int, device=x.device)

        if self.full_int_inference:
            # TODO: check if x is already integer, because in full int inference mode, QuantLinear is not the first quantized layer
            return F.linear(x, weight=self.weight_integer, bias=self.bias_integer), self.bias_scaling_factor
        else:
            # 将输入按照缩放因子缩放
            prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
            # NOTE 没有取整操作，为伪量化
            x_int = torch.round(x / prev_act_scaling_factor)
            return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
               * self.bias_scaling_factor, self.bias_scaling_factor

    def export_forward(self, x, prev_act_scaling_factor=None):
        """
            export mode forward function, x is float tensor

        Parameters:
        x : dtype: torch.float
        prev_act_scaling_factor : scale
            (x/scale) already quantized (default int8)
        """
        assert self.running_stat is False and self.full_int_inference is False, "only export from fixed fake-quant model"
        assert self.quant_mode == "symmetric", "only support symmetric quantization for export"
        

        if self.per_channel:
            return (F.linear(
                x,
                weight=torch.fake_quantize_per_channel_affine(
                    self.weight, 
                    scale=self.fc_scaling_factor,
                    zero_point=self.weight_zp,
                    axis=0,
                    quant_max=2**(self.weight_bit -1) - 1,
                    quant_min=-(2**(self.weight_bit -1)),
                ),
                bias=self.bias if self.bias is not None else None
                ) , self.bias_scaling_factor
            )
        else:
            raise Exception('For weight, we only support per_channel quantization.')




class QuantAct(nn.Module):
    """
    Class to quantize given activations

    参数:
    ----------
    activation_bit : int
        用于量化激活的位宽。
    act_range_momentum : float，默认值为0.95
        更新激活量化范围的动量。
    running_stat : bool，默认值为True
        是否使用运行统计数据来计算激活量化范围。
    per_channel : bool，默认值为False
        是否进行通道级别的量化。
    channel_len : int，默认值为None
        在使用 per_channel 模式时指定通道长度。
    quant_mode : 'none' 或 'asymmetric'，默认值为 'none'
        量化模式。'none' 表示不进行量化。

    """

    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 quant_mode="symmetric",
                 full_int_inference = False):
        super(QuantAct, self).__init__()

        # 初始化参数
        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.full_int_inference = full_int_inference # 全整型推理，取消量化节点

        self.export_mode = False

        # 初始化缓冲区
        self.min_val = torch.zeros(1)
        self.max_val = torch.zeros(1)
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        self.register_buffer('act_zp', torch.zeros(1, dtype=torch.int))

        # 设置量化模式
        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("不支持的量化模式: {}".format(self.quant_mode))
        else:
            raise ValueError("未知的量化模式: {}".format(self.quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "quant_mode: {2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})," \
                "输出量化因子: {5:.2f}".format(self.__class__.__name__, self.activation_bit,
                                          self.quant_mode, self.x_min.item(), self.x_max.item(), self.act_scaling_factor.item())

    def fix(self):
        """
        通过设置运行统计数据来固定激活范围
        """
        self.running_stat = False

    def unfix(self):
        """
        通过设置运行统计数据来取消固定激活范围
        """
        self.running_stat = True
    
    def full_int_model(self):
        """
        使模型全部整型推理
        """
        self.full_int_inference = True
        
    def unfull_int_model(self):
        """
        取消模型全部整型推理
        """
        self.full_int_inference = False


    def forward(self, x,
                pre_act_scaling_factor=None,
                identity=None,
                identity_scaling_factor=None):
        
        if self.export_mode:
            return self.export_forward(x,
                pre_act_scaling_factor,
                identity,
                identity_scaling_factor)

        # 收集运行统计数据
        with torch.no_grad():
            # NOTE 用于残差链接并量化
            x_act = x if identity is None else identity + x
            
            # 如果需要更新运行统计数据
            if self.running_stat:
                # 如果输入张量是四维的（例如，批量大小、通道数、高度和宽度），则重新排列为通道在最后的形式
                if len(x_act.shape) == 4:
                    x_act = x_act.permute(0, 2, 3, 1)
                
                # 将输入张量展平为二维，并转置，以便每一列对应一个通道的数值
                # NOTE 如何具体展开的？如何具体的对输入图进行通道量化？
                v = x_act.reshape(-1, x_act.shape[-1])
                v = v.transpose(0, 1)

                # 计算每个通道的最小值和最大值
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                
                # 如果当前最小值和最大值尚未初始化
                # 检查当前最小值和最大值是否尚未初始化，即是否相等且都为零。
                if torch.eq(self.min_val, self.max_val).all():
                    # 将当前最小值和最大值作为初始值
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    # 如果当前最小值和最大值已经初始化，则使用动量法更新它们。
                    # 动量法通过加权当前值和历史值来更新最小值和最大值，其中 act_range_momentum 是动量因子（比例系数）。
                    # cur_min 是当前最小值，self.min_val 是历史最小值，act_range_momentum 是动量因子。
                    self.min_val = self.min_val * self.act_range_momentum + \
                                 cur_min * (1 - self.act_range_momentum)
                    self.max_val = self.max_val * self.act_range_momentum + \
                                   cur_max * (1 - self.act_range_momentum)
                # 将最大值和最小值限制在全局范围内
                # NOTE 限制在全局会导致失去通道化量化！
                # add per_channel option
                if not self.per_channel:
                    self.max_val = self.max_val.max()   
                    self.min_val = self.min_val.min()

                # 计算激活量化的缩放因子
                self.act_scaling_factor = symmetric_linear_quantization_params(
                    self.activation_bit, self.min_val, self.max_val)
                
                self.act_zp = torch.zeros_like(self.act_scaling_factor, dtype=torch.int, device=x.device)


        if pre_act_scaling_factor is None:
            # 仅根据输入张量的缩放因子进行量化
            quant_act_int = self.act_function(x, self.activation_bit, self.act_scaling_factor, False)
        else:
            # 根据输入张量的缩放因子和传入张量的缩放因子进行量化         
            quant_act_int = fixedpoint_mul.apply(
                x, pre_act_scaling_factor,
                self.activation_bit, self.quant_mode,
                self.act_scaling_factor,
                identity, identity_scaling_factor, self.full_int_inference)

        if self.full_int_inference:
            return quant_act_int, self.act_scaling_factor
        else:
            return quant_act_int * self.act_scaling_factor, self.act_scaling_factor

    def export_forward(self, x,
                pre_act_scaling_factor=None,
                identity=None,
                identity_scaling_factor=None):
        """
            temporaily not support DN quantize
        """
    
        assert self.running_stat is False and self.full_int_inference is False, "only export from fixed fake-quant model"
        assert self.quant_mode == "symmetric", "only support symmetric quantization for export"

        if self.activation_bit > 8:
            # 当成int32来处理 因为tensorRT除了自定义不支持int16格式
            x = x.div(self.act_scaling_factor) + self.act_zp
            if identity is not None:
                identity = identity.div(self.act_scaling_factor) + self.act_zp
                x = x + identity
            x = x.round().clamp(-(2**31), 2**31 -1)
            return x.to(torch.int32) * self.act_scaling_factor, self.act_scaling_factor

            

        if self.per_channel:
            if x.ndim == 4:
                channel_axis = 1
            elif x.ndim == 3:
                channel_axis = -1
            elif x.ndim == 2:
                channel_axis = -1
            else:
                raise ValueError("Unknown type for Qact")

            fake_quant_x = torch.fake_quantize_per_channel_affine(
                x,
                scale=self.act_scaling_factor,
                zero_point=self.act_zp,
                axis=channel_axis,
                quant_max=2**(self.activation_bit -1) - 1,
                quant_min=-(2**(self.activation_bit -1)),
            )

            if identity is None:
                return fake_quant_x, self.act_scaling_factor
            else:
                fake_quant_identity = torch.fake_quantize_per_channel_affine(
                    identity,
                    scale = self.act_scaling_factor,
                    zero_point=self.act_zp,
                    axis=channel_axis,
                    quant_max=2**(self.activation_bit -1) - 1,
                    quant_min=-(2**(self.activation_bit -1)),
                )
                return fake_quant_x + fake_quant_identity, self.act_scaling_factor
        
        else:
            fake_quant_x = torch.fake_quantize_per_tensor_affine(
                x,
                scale=self.act_scaling_factor,
                zero_point=self.act_zp,
                quant_max=2**(self.activation_bit -1) - 1,
                quant_min=-(2**(self.activation_bit -1)),
            )

            if identity is None:
                return fake_quant_x, self.act_scaling_factor
            else:
                fake_quant_identity = torch.fake_quantize_per_tensor_affine(
                    identity,
                    scale = self.act_scaling_factor,
                    zero_point=self.act_zp,
                    quant_max=2**(self.activation_bit -1) - 1,
                    quant_min=-(2**(self.activation_bit -1)),
                )
                return fake_quant_identity + fake_quant_x, self.act_scaling_factor


class QuantMatMul(nn.Module):
    """
    用于量化给定matmul层的权重的类
    """
    def __init__(self, running_stat=True, full_int_inference=False):
        super(QuantMatMul, self).__init__()
        # 注册一个缓冲区来存储激活的缩放因子
        self.register_buffer('act_scaling_factor', torch.zeros(1))
        # 状态变量
        self.running_stat = running_stat
        self.full_int_inference = full_int_inference
        self.export_mode = False
    
    def fix(self):
        self.running_stat = False

    def unfix(self):
        self.running_stat = True
    
    def full_int_model(self):
        self.full_int_inference = True
    
    def unfull_int_model(self):
        self.full_int_inference = False

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        if self.export_mode:
            #here A, B has already gone through fake-quant
            return A.matmul(B), self.act_scaling_factor
        
        if self.full_int_inference:
            # 如果是全整型推理，则输入的AB为整数
            return A @ B, self.act_scaling_factor
        # 对输入张量A和B进行量化
        # NOTE 没有取整操作，为伪量化
        A_int = torch.round(A / pre_act_scaling_factor_A)
        B_int = torch.round(B / pre_act_scaling_factor_B)
        # 计算输出的量化因子
        if self.running_stat:
            self.act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        # 返回量化后的乘积及其缩放因子
        return (A_int @ B_int) * self.act_scaling_factor, self.act_scaling_factor



class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 quant_mode="symmetric",
                 per_channel=True,
                 weight_percentile=0,
                 running_stat=True,
                 full_int_inference=False):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias
                                          )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.running_stat = running_stat
        self.full_int_inference = full_int_inference

        self.export_mode = False

        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))
        self.register_buffer('conv_opt_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_zp', torch.zeros_like(self.conv_scaling_factor, dtype=torch.int))

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        self.running_stat = False

    def unfix(self):
        self.running_stat = True
        
    def full_int_model(self):
        self.full_int_inference = True
    
    def unfull_int_model(self):
        self.full_int_inference = False

    def forward(self, x, pre_act_scaling_factor=None):
        if self.export_mode:
            return self.export_forward(x, pre_act_scaling_factor)

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))
        
        with torch.no_grad():
            if self.running_stat:
                w = self.weight
                if self.per_channel:
                    v = w.reshape(w.shape[0], -1)
                    cur_min = v.min(axis=1).values
                    cur_max = v.max(axis=1).values
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    raise Exception('For weight, we only support per_channel quantization.')

                self.conv_scaling_factor = symmetric_linear_quantization_params(
                    self.weight_bit, self.min_val, self.max_val)
                
                self.weight_zp = torch.zeros_like(self.conv_scaling_factor, dtype=torch.int, device=x.device)
        
        if self.running_stat:
            self.weight_integer = self.weight_function(
                self.weight, self.weight_bit, self.conv_scaling_factor, True)
            
            bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True)
            # 更新输出量化因子
            self.conv_opt_scaling_factor = bias_scaling_factor.view(1, -1, 1, 1)

        if self.full_int_inference:
            return (F.conv2d(x, self.weight_integer, self.bias_integer, 
                             self.stride, self.padding, self.dilation, self.groups), self.conv_opt_scaling_factor)
        else:
            pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
            x_int = torch.round(x / pre_act_scaling_factor)
            return (F.conv2d(x_int, self.weight_integer, self.bias_integer, 
                             self.stride, self.padding, self.dilation, self.groups) * self.conv_opt_scaling_factor, self.conv_opt_scaling_factor)
            
    def export_forward(self, x, pre_act_scaling_factor=None):
        """
            export mode forward function, x is float tensor

        Parameters:
        x : dtype: torch.float
        pre_act_scaling_factor : scale
            (x/scale) already quantized (default int8)
        """
        assert self.running_stat is False and self.full_int_inference is False, "only export from fixed fake-quant model"
        assert self.quant_mode == "symmetric", "only support symmetric quantization for export"
        
        if self.per_channel:
            return (F.conv2d(
                x,
                weight=torch.fake_quantize_per_channel_affine(
                    self.weight, 
                    scale=self.conv_scaling_factor,
                    zero_point=self.weight_zp,
                    axis=0,
                    quant_max=2**(self.weight_bit -1) - 1,
                    quant_min=-(2**(self.weight_bit -1)),
                ),
                bias=self.bias if self.bias is not None else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
                ) , self.conv_opt_scaling_factor
            )
        else:
            raise Exception('For weight, we only support per_channel quantization.')
        

class IntLayerNorm(nn.LayerNorm):
    """
    Implementation of I-LayerNorm
    Class to quantize given LayerNorm layer
    """
    def __init__(self, 
                normalized_shape, 
                eps=1e-5,
                elementwise_affine=True,
                running_stat=True,
                full_int_inference=False):
        super(IntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.dim_sqrt = None
        self.full_int_inference = full_int_inference
        self.running_stat = running_stat
        self.register_buffer('norm_scaling_factor', torch.zeros(1))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

        self.export_mode = False
        self.float_op = False

    def fix(self):
        self.running_stat = False

    def unfix(self):
        self.running_stat = True       
    
    def full_int_model(self):
        self.full_int_inference = True
    
    def unfull_int_model(self):
        self.full_int_inference = False
        

    def forward(self, x, scaling_factor=None):

        if self.export_mode or self.float_op:
            return self.export_forward(x, scaling_factor)

        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Normalization: computes mean and variance(std)
        # 如果整型推理则默认输入为上层传入的整型张量，否则默认为浮点张量需要伪量化
        x_int = x if self.full_int_inference else round_ste.apply(x / scaling_factor)
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2 ** 16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int/k))/2)
            k = k_1
        std_int = k

        factor = floor_ste.apply((2 ** 31-1) / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2 ** 30

        # scaling and shifting
        if self.running_stat: # 若unfix训练过程中，则更新bias_integer等
            bias = self.bias.data.detach() / (self.weight.data.detach())
            bias_int = floor_ste.apply(bias / scaling_factor)
            self.bias_integer = bias_int
        
        # TODO: here the logic of LN seems not correct
        y_int = y_int + self.bias_integer

        if self.full_int_inference:
            return y_int, self.norm_scaling_factor
        
        else:       
            scaling_factor = scaling_factor * self.weight
            x = y_int * scaling_factor
            self.norm_scaling_factor = scaling_factor
            return x, self.norm_scaling_factor

    def export_forward(self, x, scaling_factor=None):
        """
            export mode forward function, x is float tensor

        Parameters:
        x : dtype: torch.float
        scaling_factor : scale
            (x/scale) already quantized (default int8)
        """
        # x = x.to(torch.float32)
        mean = x.mean(axis=2, keepdim=True)
        y = x - mean
        var = torch.sum(torch.square(y), dim=2, keepdim=True) / x.shape[2]

        std = torch.sqrt(var + self.eps)
        y = y / std

        y = y * self.weight + self.bias

        return y, 1


class IntGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8,
                 full_int_inference=False,
                 running_stat=True):
        super(IntGELU, self).__init__()
        self.output_bit = output_bit
        self.running_stat = running_stat
        self.full_int_inference = full_int_inference

        self.n = 23  # sufficiently large integer
        #The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer('act_scaling_factor', torch.zeros(1))
        self.export_mode = False
        self.float_op = False

    def fix(self):
        self.running_stat = False
    def unfix(self):
        self.running_stat = True
    def full_int_model(self):
        self.full_int_inference = True
    def unfull_int_model(self):
        self.full_int_inference = False

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n

        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        if self.export_mode or self.float_op:
            return self.export_forward(x, scaling_factor)

        pre_x_int = x if self.full_int_inference else round_ste.apply(x / scaling_factor)
        scaling_factor_sig = scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig) # e^(x-x_max)

        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2**31-1)
        factor = floor_ste.apply((2 ** 31-1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(exp_int * factor / 2 ** (31-self.output_bit+1))

        if self.running_stat:
            sigmoid_scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit-1)]).cuda()
            scaling_factor = scaling_factor * sigmoid_scaling_factor
            self.act_scaling_factor = scaling_factor
            
        x_int = pre_x_int * sigmoid_int
        if self.full_int_inference:
            return x_int, self.act_scaling_factor
        else:
            
            return x_int * self.act_scaling_factor, self.act_scaling_factor

    def export_forward(self, x, scaling_factor=None):
        return F.gelu(x), 1


class IntSoftmax(nn.Module):
    """
    Implementation of Shiftmax
    Class to quantize given Softmax layer
    """

    def __init__(self, output_bit=8, full_int_inference=False):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit
        self.full_int_inference = full_int_inference

        self.n = 15  # sufficiently large integer
        #The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer('act_scaling_factor', torch.zeros(1))
        self.export_mode = False
        self.float_op = False

    def fix(self):
        pass

    def unfix(self):
        pass
    
    def full_int_model(self):
        self.full_int_inference = True
    def unfull_int_model(self):
        self.full_int_inference = False

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = floor_ste.apply(r/2 - x0_int)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        if self.export_mode or self.float_op:
            return self.export_forward(x, scaling_factor)

        x_int = x if self.full_int_inference else round_ste.apply(x / scaling_factor)
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum.clamp_max_(2**31-1)
        
        factor = floor_ste.apply((2**31-1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (31-self.output_bit+1))
        scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit-1)]).cuda()

        self.act_scaling_factor = scaling_factor
        if self.full_int_inference:
            return exp_int, self.act_scaling_factor
        else:
            return exp_int * self.act_scaling_factor, self.act_scaling_factor
    
    def export_forward(self, x, scaling_factor=None):
        return F.softmax(x, dim=-1), 1
