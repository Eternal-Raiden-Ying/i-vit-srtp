from collections import OrderedDict
from functools import partial

import torch
from torch import nn

from .layers_quant import PatchEmbed, Mlp, DropPath, trunc_normal_
from .quantization_utils import QuantLinear, QuantAct, QuantConv2d, IntLayerNorm, IntSoftmax, IntGELU, QuantMatMul
from .utils import load_weights_from_npz


__all__ = ['SimpleModel']

class SimpleModel(nn.Module):
    """
        simple model for testing pytorch -> onnx -> tensorrt roadmap
    """
    def __init__(
            self,
            in_features=16,
            hidden_features=64,
            out_features=32,
            fc_bias=True,
            drop=0.0):
        super().__init__()
        self.qact_input = QuantAct()
        self.fc1 = QuantLinear(
            in_features,
            hidden_features,
            bias=fc_bias
        )
        
        self.qact_fc1 = QuantAct()
        # self.act = IntGELU()
        # self.qact_gelu = QuantAct()
        self.fc2 = QuantLinear(
            hidden_features,
            out_features,
            bias=fc_bias
        )
        self.qact_fc2 = QuantAct()
        self.drop = nn.Dropout(drop)
        self.softmax = IntSoftmax(output_bit=8)

    def forward(self, x):
        x, act_scaling_factor = self.qact_input(x, None)
        x, act_scaling_factor = self.fc1(x, act_scaling_factor)
        x, act_scaling_factor = self.qact_fc1(x, act_scaling_factor)
        # x, act_scaling_factor = self.act(x, act_scaling_factor)
        # x, act_scaling_factor = self.qact_gelu(x, act_scaling_factor)
        # x = self.drop(x)
        x, act_scaling_factor = self.softmax(x, act_scaling_factor)
        x, act_scaling_factor = self.fc2(x, act_scaling_factor)
        x, act_scaling_factor = self.qact_fc2(x, act_scaling_factor)
        # x = self.drop(x)
        return x