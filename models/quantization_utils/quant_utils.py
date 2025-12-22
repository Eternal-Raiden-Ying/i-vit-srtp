import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import bisect
from fractions import Fraction
import decimal
from decimal import Decimal
import time


def linear_quantize(input, scale, zero_point, is_weight):
    """
    使用给定的缩放因子和零点对单精度输入张量进行量化为整数。

    参数:
    ----------
    input: torch.Tensor
        要量化为整数的单精度输入张量
    scale: torch.Tensor
        量化的缩放因子
    zero_point: torch.Tensor
        量化的零点偏移
    is_weight: bool
        是否是权重张量
    """

    # 为卷积权重和激活重新调整缩放因子和零点的形状
    if is_weight:
        if len(input.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        elif len(input.shape) == 2:
            scale = scale.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        else:
            scale = scale.view(-1)
            zero_point = zero_point.view(-1)
    else:
        if len(input.shape) == 2:
            scale = scale.view(1, -1)
            zero_point = zero_point.view(1, -1)
        elif len(input.shape) == 3:
            scale = scale.view(1, 1, -1)
            zero_point = zero_point.view(1, 1, -1)
        elif len(input.shape) == 4:
            scale = scale.view(1, -1, 1, 1)
            zero_point = zero_point.view(1, -1, 1, 1)
        else:
            raise NotImplementedError

    # 计算量化后的张量，quantized = float / scale + zero_point
    # NOTE 使用量化因子量化后没有圆或取整操作，为伪量化
    return torch.round(1. / scale * input + zero_point)

def symmetric_linear_quantization_params(num_bits, min_val, max_val):
    """
    根据给定的量化范围计算对称量化的缩放因子。
    
    参数:
    ----------
    num_bits: int
        量化位数，即表示的位数。
    min_val: torch.Tensor
        量化范围的下限。
    max_val: torch.Tensor
        量化范围的上限。
    """

    # 在这部分代码中，我们不需要计算梯度，
    # 为了确保这一点，我们使用了 torch.no_grad()
    with torch.no_grad():
        # 计算量化的整数范围
        n = 2 ** (num_bits - 1) - 1
        # 机器浮点数的最小正值
        eps = torch.finfo(torch.float32).eps

        # 获取量化范围的绝对值的最大值, 非饱和量化。
        max_val = torch.max(-min_val, max_val)
        # 计算缩放因子
        scale = max_val / float(n)
        # 将缩放因子限制在最小正值以上，防止除0操作
        scale.clamp_(eps)

    return scale



class SymmetricQuantFunction(Function):
    """
    使用给定的范围和位宽对浮点值进行对称量化的类。
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale, is_weight):
        """
        前向传播函数

        参数:
        ----------
        x: torch.Tensor
            要量化的浮点张量
        k: int
            量化位宽
        specified_scale: torch.Tensor
            预先计算的用于缩放张量 x 的缩放因子
        is_weight: bool
            是否是权重张量
        """

        scale = specified_scale.cuda()

        zero_point = torch.tensor(0.).cuda()

        n = 2 ** (k - 1) - 1
        # 使用线性量化函数进行量化
        new_quant_x = linear_quantize(x, scale, zero_point, is_weight=is_weight)
        # 将量化结果限制在合适的范围内
        new_quant_x = torch.clamp(new_quant_x, -n-1, n)

        # 保存缩放因子和是否是权重张量的信息
        ctx.scale = scale
        ctx.is_weight = is_weight
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播函数

        参数:
        ----------
        grad_output: torch.Tensor
            梯度输出
        """

        scale = ctx.scale
        is_weight = ctx.is_weight
        # 根据张量的维度进行缩放因子的调整
        if is_weight:
            if len(grad_output.shape) == 4:
                scale = scale.view(-1, 1, 1, 1)
            elif len(grad_output.shape) == 2:
                scale = scale.view(-1, 1)
            else:
                scale = scale.view(-1)
        else:
            if len(grad_output.shape) == 2:
                scale = scale.view(1, -1)
            elif len(grad_output.shape) == 3:
                scale = scale.view(1, 1, -1)
            elif len(grad_output.shape) == 4:
                scale = scale.view(1, -1, 1, 1)
            else:
                raise NotImplementedError
        # 返回梯度
        return grad_output.clone() / scale, None, None, None


class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def batch_frexp(inputs, max_bit=7):
    """
    Decompose the scaling factor into mantissa and twos exponent.
    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2 ** max_bit)).quantize(Decimal('1'),
                                                                 rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = float(max_bit) - output_e

    return torch.from_numpy(output_m).cuda().view(shape_of_input), \
           torch.from_numpy(output_e).cuda().view(shape_of_input)


import torch

class fixedpoint_mul(Function):
    """
    执行可以与硬件上的整数算术匹配的定点算术的函数。
    参数：
    ----------
    pre_act: 输入张量
    pre_act_scaling_factor: 输入张量的缩放因子
    bit_num: 量化位宽
    quant_mode: 量化模式，'symmetric'或'asymmetric'
    z_scaling_factor: 输出张量的缩放因子
    identity: 身份张量
    identity_scaling_factor: 身份张量的缩放因子
    """

    @staticmethod
    def forward(ctx, pre_act, pre_act_scaling_factor,
                bit_num, quant_mode, z_scaling_factor,
                identity=None, identity_scaling_factor=None, full_int_inference=False):

        # 根据输入张量的形状选择不同的 reshape 函数
        if len(pre_act.shape) == 2:
            reshape = lambda x: x.view(1, -1)
        elif len(pre_act.shape) == 3:
            reshape = lambda x: x.view(1, 1, -1)
        elif len(pre_act.shape) == 4:
            reshape = lambda x: x.view(1, -1, 1, 1)
        else:
            raise NotImplementedError
        
        # 在上下文中存储身份张量
        ctx.identity = identity

        # 根据量化模式计算最大可表示整数
        if quant_mode == 'symmetric':
            # 带正负的量化
            n = 2 ** (bit_num - 1) - 1
        else:
            n = 2 ** bit_num - 1

        with torch.no_grad():
            # 将输入张量的缩放因子和身份张量的缩放因子调整为正确的形状
            # 例如，可能是按照通道量化的，卷积和全连接具有不同的结构通道
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor # 残差链接后的重新统计的量化因子

            # 对输入张量进行量化
            # NOTE 将x0的量化因子区域缩放到x0+x1（即残差链接后的位置）
            z_int = pre_act if full_int_inference else torch.round(pre_act / pre_act_scaling_factor) # 量化输入张量
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B # 
            new_scale = reshape(new_scale)

            # 缩放和量化输出张量
            m, e = batch_frexp(new_scale)
            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0 ** e))

            # 如果提供了身份张量，则进行相同的处理
            # NOTE 如果提供了残差链接的张量，identity即x1，按照x0方式量化
            if identity is not None:
                wx_int = identity if full_int_inference else torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                # NOTE 分子部分的量化不严格遵守硬件原则，而是使用double类型进行计算。
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0 ** e1))

                output = output1 + output

            # 根据量化位宽和模式对输出张量进行截断
            if bit_num in [4, 8, 16, 32]:
                if quant_mode == 'symmetric':
                    return torch.clamp(output.type(torch.float), -n-1, n)
                else:
                    return torch.clamp(output.type(torch.float), 0, n)
            else:
                return output.type(torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        # 计算输入张量的梯度
        identity_grad = None
        # 对梯度进行量化的理解
        # fixedpoint_mul在QuantAct中进行使用，在fixedpoint_mul的输出为整型，所以其对应的反向传播应该也是传播整型的梯度
        # NOTE 原始代码中没有对反量化的梯度进行取整
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        # 返回梯度中，因为使用了full_int_inference标志位，所以需要二外在最后补充一个None来保持梯度的一致性
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, \
               identity_grad, None, None

