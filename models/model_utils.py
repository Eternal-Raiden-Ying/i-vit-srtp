import torch.nn as nn
from .quantization_utils import *
from .vit_quant import Attention, VisionTransformer

"""
进行纯整型推理的重要脚本，
使用遍历递归将模块和类中的更新量化因子标志位(running_stat)，
和纯整型推理标志位（full_int_inference)进行调整。
"""

classes_to_fix = (QuantAct, QuantConv2d, QuantLinear, QuantMatMul, IntGELU, IntLayerNorm, IntSoftmax)
def freeze_model(model):
    """
    冻结模型的激活范围。递归调用 layer.fix()
    """
    model.running_int = False
    for child in model.children():
        # 如果子模块是QuantLinear的实例，则调用full_int_model方法
        if isinstance(child, classes_to_fix):
            child.fix()
        # 递归遍历子模块
        freeze_model(child)
        
def unfreeze_model(model):
    """
    解冻模型的激活范围。递归调用 layer.unfix()
    """
    model.running_int = True
    for child in model.children():
        # 如果子模块是QuantLinear的实例，则调用full_int_model方法
        if isinstance(child, classes_to_fix):
            child.unfix()
        # 递归遍历子模块
        unfreeze_model(child)        

# 遍历模型的所有子模块
classes_to_int = (QuantAct, QuantConv2d, QuantLinear, QuantMatMul, IntGELU, IntLayerNorm, IntSoftmax, Attention)
# 遍历模型的所有子模块
def int_model(model):
    model.full_int_inference = True
    for child in model.children():
        # 如果子模块是QuantLinear的实例，则调用full_int_model方法
        if isinstance(child, classes_to_int):
            child.full_int_model()
        # 递归遍历子模块
        int_model(child)

def un_int_model(model):
    model.full_int_inference = False
    for child in model.children():
        # 如果子模块是QuantLinear的实例，则调用full_int_model方法
        if isinstance(child, classes_to_int):
            child.unfull_int_model()
        # 递归遍历子模块
        un_int_model(child)