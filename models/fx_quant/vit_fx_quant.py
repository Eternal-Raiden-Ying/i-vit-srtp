"""
    something more to do
    TODO:
"""


"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import copy
import os.path
from functools import partial
from collections import OrderedDict

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def DN_quantize(num: float, c: int = 2):
    n = 0
    if num >= 1:
        while num >= 1:
            num /= 2
            n -= 1
    elif num < 0.5:
        while num < 0.5:
            num *= 2
            n += 1
    assert 0.5 <= num < 1, ["DN_quantization failure!"]
    b: int = math.floor(num * 2**c)
    n += c
    return b, n


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    # 仅在训练时，dropout一部分x
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class MinMaxObserver(nn.Module):

    def __init__(self,dtype, quant_min, quant_max, qscheme,eps=torch.finfo(torch.float32).eps):
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = eps
        self.max_val = torch.tensor(float("-inf"))
        self.min_val = torch.tensor(float("inf"))
        if self.dtype != torch.quint8 and self.dtype != torch.qint8:
            raise NotImplementedError(
                "Do not support dtype except torch.qint8 and torch.quint8"
            )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def calculate_qparam(self):
        if self.qscheme == torch.per_tensor_symmetric:
            abs_max = max(abs(self.max_val),abs(self.min_val))
            scale = 2*abs_max/(self.quant_max-self.quant_min)
            # scale = 2*abs_max/(self.quant_max-self.quant_min+self.eps)
            zp = math.floor((self.quant_max+self.quant_min)/2)
        elif self.qscheme == torch.per_tensor_affine:
            scale = (self.max_val - self.min_val)/(self.quant_max - self.quant_min)
            # scale = (self.max_val - self.min_val)/(self.quant_max - self.quant_min + self.eps)
            zp = round(self.quant_max - self.max_val / scale)
        else:
            raise NotImplementedError(
                "Do not support qscheme except torch.per_tensor_symmetric and torch.tensor_affine"
            )
        return scale, zp


class DropPath(nn.Module):
    """
    Dropout when drop_prob is True
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 原始224*224 图像  每个 patch 16*16  224/16 =14，所以整张图片可以被分成 14*14 = 196 个 patch, embed_dim=768=3*16*16
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # nn.Identity()是恒等映射，即此处是否需要归一化

    def forward(self, x):
        B, C, H, W = x.shape
        # x.shape usually (1,3,224,224)
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # proj:(1,3,224,224) -> (1,768,14,14)
        # flatten: [B, C, H, W] -> [B, C, HW]: (1,768,196)
        # transpose: [B, C, HW] -> [B, HW, C]:(1,196,768)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim 768
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # each head handling a portion of the total embedding space
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 这里默认QKV的维度都是total_embed_dim，实际可以是其他维度值
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]   
        # +1指的是class_token作为第一个token
        # x已经embedding好的
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #! 生成QKV叠加
        
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # from plot_utils import plot_heat_map
        # Q = torch.concatenate([q[0][i].clone().detach() for i in range(q.shape[1])], dim=-1)
        # K = torch.concatenate([k[0][i].clone().detach() for i in range(q.shape[1])], dim=-1)
        # plot_heat_map((Q @ (K.permute(1,0))).cpu())

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  #! Q*K转置/根号dim
        attn = attn.softmax(dim=-1)  #  normalizes the attention scores
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        """
            attn中对加入的class token也做了softmax，这合理吗，class token用于？
            并使用attn中的max indices替换了v中原先的class token，此时v第二个维度的num_patches+1的‘+1’似乎就没有特殊的意义了
            这个版本中，multi-head，不同的heads会在对应的子空间中选出不同的相似度最高的indices，但是不同head拼接起来时indices可以不同，再选出的v也是由不同indices的v的片段拼接而成，甚至是v的class token
        """
        x = self.proj(x)  # 映射回原特征维度，混合不同head的信息
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    # in_features -> hidden_features -> out_features
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        '''
        在 PyTorch 中 nn.Parameter 是一种特殊的张量，用于标记为模型的可学习参数。
        nn.Parameter 会自动被注册为模型的一部分，因此在执行 反向传播 时，这些参数会被计算梯度并更新。
        '''

        # 逐层增加dropout概率直到最后一层概率为drop_path_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])    #! 堆叠12层Encoder
        self.norm = norm_layer(embed_dim)

        # Representation layer
        '''
        假设 Transformer 的输出 embed_dim = 768，而你希望用一个大小为 512 的表示来进行分类，那么你可以将 representation_size 设置为 512。表示层将会有以下作用：
        输入：一个 768 维度的特征向量。
        输出：一个 512 维度的表示向量。
        处理：通过一个线性层 nn.Linear(768, 512) 和一个 Tanh 激活函数对特征进行投影和非线性变换。
        最终，这个 512 维的表示向量将用于分类头的输入。
        '''
        if representation_size and not distilled:
            self.has_logits = True  # TODO： not used here, check if code was deleted
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)   #! 分类头，就是一个线性层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None: # 不使用蒸馏机制
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768] #! 加上了class token 
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)  # 额外拼接一个 distillation token

        x = self.pos_drop(x + self.pos_embed) # 位置编码嵌入
        # 在embed前嵌入位置信息是否会更合理，直接相加会使模型难以区分，虽然在embed前嵌入也会存在这个问题（在卷积中融合临近域的信息，但是这一过程是否可逆，不可逆则必然丢失信息）
        # 另外，嵌入的pos_embed作为一个可学习参数，不同patch的embedding_dim上相互独立，但作为pos意义本身，只应该在不同的patch上区分，TODO：检查pos_embed在不同patch上的差异

        x = self.blocks(x)
        x = self.norm(x)
        #  最后一个block有冗余的计算量，虽然占比不多
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])  # 取出第一个class token
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        #torch.Size([1, 768])
        
        if self.head_dist is not None:  # 采用蒸馏模型
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            
        # torch.Size([batch, num_class])
        return x


class QAttention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim 768
                 input_observer=None,
                 output_observer=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,  # 计算softmax
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(QAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # each head handling a portion of the total embedding space
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # ! 这里默认QKV的维度都是total_embed_dim，实际可以是其他维度值
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.input_observer = input_observer
        self.output_observer = output_observer
        self.module_quantized = False
        self.show_details:dict = dict()

        self.input_scale = None
        self.input_zp = None
        self.weight_scale = None
        self.weight_zp = None
        self.bias_scale = None
        self.bias_zp = None
        self.output_scale = None
        self.output_zp = None

    def forward(self, x):

        if not self.module_quantized:

            if "qkv_input" in self.show_details.keys():
                [path, count] = self.show_details["qkv_input"]
                assert count > 0,["count should be positive"]
                self.save_data(name=f"qkv_input{count}.txt",
                               path=path,
                               data=x)
                count -= 1
                if not count:
                    self.show_details.pop("qkv_input")
            # [batch_size, num_patches + 1, total_embed_dim]
            # +1指的是class_token作为第一个token
            # x已经embedding好的
            B, N, C = x.shape

            # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
            # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
            # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # ! 生成QKV叠加
            if "qkv" in self.show_details.keys():
                [path, count] = self.show_details["qkv"]
                assert count > 0, ["count should be positive"]
                self.save_data(name=f"qkv{count}.txt",
                               path=path,
                               data=qkv)
                count -= 1
                if not count:
                    self.show_details.pop("qkv")

            # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
            # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
            attn = q @ k.transpose(-2, -1)
            if "qkT" in self.show_details.keys():
                [path, count] = self.show_details["qkT"]
                assert count > 0, ["count should be positive"]
                self.save_data(name=f"qkT{count}.txt",
                               path=path,
                               data=attn)
                count -= 1
                if not count:
                    self.show_details.pop("qkT")

            attn = attn*self.scale
            attn = attn.softmax(dim=-1)  # normalizes the attention scores
            attn = self.attn_drop(attn)

            # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
            # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
            # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            if "qkv_output" in self.show_details.keys():
                [path, count] = self.show_details["qkv_output"]
                assert count > 0, ["count should be positive"]
                self.save_data(name=f"qkv_output{count}.txt",
                               path=path,
                               data=x)
                count -= 1
                if not count:
                    self.show_details.pop("qkv_output")

            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        else:
            #待修改
            if "qkv_input" in self.show_details.keys():
                [path, count] = self.show_details["qkv_input"]
                assert count > 0,["count should be positive"]
                self.save_data(name=f"qkv_input{count}.txt",
                               path=path,
                               data=x)
                count -= 1
                if not count:
                    self.show_details.pop("qkv_input")

            x = self._quant(x,self.input_scale,self.input_zp)
            if "qkv_input_quant" in self.show_details.keys():
                [path, count] = self.show_details["qkv_input_quant"]
                assert count > 0,["count should be positive"]
                self.save_data(name=f"qkv_input_quant{count}.txt",
                               path=path,
                               data=x)
                count -= 1
                if not count:
                    self.show_details.pop("qkv_input_quant")

            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # ! 生成QKV叠加
            if "qkv" in self.show_details.keys():
                [path, count] = self.show_details["qkv"]
                assert count > 0, ["count should be positive"]
                self.save_data(name=f"qkv{count}.txt",
                               path=path,
                               data=qkv)
                count -= 1
                if not count:
                    self.show_details.pop("qkv")
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1))
            if "qkT" in self.show_details.keys():
                [path, count] = self.show_details["qkT"]
                assert count > 0, ["count should be positive"]
                self.save_data(name=f"qkT{count}.txt",
                               path=path,
                               data=attn)
                count -= 1
                if not count:
                    self.show_details.pop("qkT")

            b, n = DN_quantize(self.scale)
            attn = torch.round((attn * b) / (2**n))
            attn = attn.softmax(dim=-1)  # normalizes the attention scores
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            if "qkv_output" in self.show_details.keys():
                [path, count] = self.show_details["qkv_output"]
                assert count > 0, ["count should be positive"]
                self.save_data(name=f"qkv_output{count}.txt",
                               path=path,
                               data=x)
                count -= 1
                if not count:
                    self.show_details.pop("qkv_output")

            x = x*(self.input_scale**3)*(self.weight_scale**3)
            if "qkv_output_dequant" in self.show_details.keys():
                [path, count] = self.show_details["qkv_output_dequant"]
                assert count > 0, ["count should be positive"]
                self.save_data(name=f"qkv_output_dequant{count}.txt",
                               path=path,
                               data=x)
                count -= 1
                if not count:
                    self.show_details.pop("qkv_output_dequant")

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    def quantize_weight(self):
        assert hasattr(self, "state_dict"), ["Linear does not have attr state_dict"]
        weight_float = self.state_dict()["qkv.weight"].detach()  # avoid keeping autograd tape
        # bias_float = self.state_dict()["qkv.bias"].detach()
        min_val_w, max_val_w = torch.aminmax(weight_float)
        # min_val_b, max_val_b = torch.aminmax(bias_float)
        self.weight_scale = 2*max(abs(min_val_w), abs(max_val_w))/255
        self.weight_zp = 0
        # self.bias_scale = 2*max(abs(min_val_b), abs(max_val_b))/255
        # self.bias_zp = 0
        self.state_dict()["qkv.weight"].copy_(
            torch.round(self.state_dict()["qkv.weight"]/self.weight_scale + self.weight_zp)
        )
        # self.state_dict()["qkv.bias"].copy_(
        #     torch.round(self.state_dict()["qkv.bias"] / self.bias_scale + self.bias_zp)
        # )

    def calibrate(self, x):
        assert not self.module_quantized,["The module has already been quantized"]
        assert isinstance(self.input_observer, MinMaxObserver), ["Attention does not get an input_observer!"]
        assert isinstance(self.output_observer, MinMaxObserver), ["Attention does not get an output_observer!"]
        assert self.weight_scale, ["qkv_weight matrix has not been quantized"]
        # x.size(): torch.Size([1, 197, 768])
        x = self.input_observer.forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # ! 生成QKV叠加
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # ! Q*K转置/根号dim

        attn = attn.softmax(dim=-1)  # normalizes the attention scores
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = x.mul(self.weight_scale)
        x = self.output_observer.forward(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def quantize(self):
        assert isinstance(self.input_observer, MinMaxObserver), ["Attention does not get an input_observer!"]
        assert isinstance(self.output_observer, MinMaxObserver), ["Attention does not get an output_observer!"]

        self.input_scale, self.input_zp = self.input_observer.calculate_qparam()
        self.output_scale, self.output_zp = self.output_observer.calculate_qparam()

        # self.input_zp = self.output_zp = 0
        # self.input_scale = self.input_observer.max_val/self.input_observer.quant_max
        # self.output_scale = self.output_observer.max_val/self.output_observer.quant_max
        self.module_quantized = True

    def _quant(self, x, scale, zp):
        x = torch.round(x/scale + zp)
        return x

    def _dequant(self, x, scale, zp):
        x = scale*(x-zp)
        return x

    @staticmethod
    def save_data(name, path, data):
        if os.path.exists(path) is False:
            os.makedirs(path)
        with open(os.path.join(path, name), 'w') as file:
            for row in data:
                file.writelines(' '.join(map(str, row.tolist())))


class QBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 input_observer=None,
                 output_observer=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(QBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = QAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               input_observer=input_observer,output_observer=output_observer,
                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

        self.qkv_calibrate = False
        self.show_details: dict = dict()

    def forward(self, x):
        if len(self.show_details):
            self.attn.show_details = copy.deepcopy(self.show_details)
            self.show_details = dict()

        if not self.qkv_calibrate:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn.calibrate(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class QVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, # 取消qkv的bias
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,act_layer=None,
                 input_observer=MinMaxObserver(
                     dtype=torch.quint8,quant_min=0,quant_max=255,qscheme=torch.per_tensor_symmetric
                 ),
                 output_observer=MinMaxObserver(
                     dtype=torch.quint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_symmetric
                 )):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(QVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.show_detail_opts: dict = dict()
        '''
        在 PyTorch 中 nn.Parameter 是一种特殊的张量，用于标记为模型的可学习参数。
        nn.Parameter 会自动被注册为模型的一部分，因此在执行 反向传播 时，这些参数会被计算梯度并更新。
        '''

        # 逐层增加dropout概率直到最后一层概率为drop_path_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            QBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                   input_observer=input_observer, output_observer= output_observer,
                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                   norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])  # ! 堆叠12层Encoder
        self.norm = norm_layer(embed_dim)

        # Representation layer
        '''
        假设 Transformer 的输出 embed_dim = 768，而你希望用一个大小为 512 的表示来进行分类，那么你可以将 representation_size 设置为 512。表示层将会有以下作用：
        输入：一个 768 维度的特征向量。
        输出：一个 512 维度的表示向量。
        处理：通过一个线性层 nn.Linear(768, 512) 和一个 Tanh 激活函数对特征进行投影和非线性变换。
        最终，这个 512 维的表示向量将用于分类头的输入。
        '''
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)   #! 分类头，就是一个线性层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        if len(self.show_detail_opts):
            if "EmbeddedInput" in self.show_detail_opts.keys():
                [path, count] = self.show_detail_opts["EmbeddedInput"]
                self.save_data(name=f"EmbeddedInput{count}.txt",
                               path=path,
                               data=x)
                count -= 1
                if not count:
                    self.show_detail_opts.pop("EmbeddedInput")
            if "blocks" in self.show_detail_opts.keys():
                for key in self.show_detail_opts["blocks"].keys():
                    index = int(key)
                    self.blocks[index].show_details[self.show_detail_opts["blocks"][key][0]] = [
                        self.show_detail_opts["blocks"][key][1]+f"/block{index}",
                        self.show_detail_opts["blocks"][key][2]
                    ]
                self.show_detail_opts.pop("blocks")

        # [B, C, H, W] -> [B, num_patches, embed_dim]

        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:  # 不使用蒸馏机制
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768] #! 加上了class token
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x),
                          dim=1)  # 额外拼接一个 distillation token

        x = self.pos_drop(x + self.pos_embed)  # 位置编码嵌入
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])  # 取出第一个class token
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        # torch.Size([1, 768])

        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)

        # torch.Size([batch, num_class])
        return x

    def prepare(self):
        for block in self.blocks:
            block.attn.quantize_weight()
            block.qkv_calibrate = True

    def quantize(self):
        for block in self.blocks:
            block.qkv_calibrate = False
            block.attn.quantize()

    def show_details(self, opt: str, path:str, count:int=1):
        """
        show data process details
        :param opt:
            "EmbeddedInput“， ”blocks.[0-11].attn.qkv_input“, "blocks.[0-11].attn.qkv","blocks.[0-11].attn.qkT",
            "blocks.[0-11].attn.qkv_output"
        :return:
        """

        if opt == "EmbeddedInput":
            self.show_detail_opts["EmbeddedInput"] = [path, count]
        if "blocks" in opt:
            opts = opt.split('.')
            assert str.isdigit(opts[1]), ["opt in wrong form"]
            assert opts[2] == 'attn' and len(opts) == 4, ["support qkv only"]
            if "blocks" not in self.show_detail_opts.keys():
                self.show_detail_opts["blocks"] = dict()
            self.show_detail_opts["blocks"][opts[1]] = [opts[3], path, count]

    @staticmethod
    def save_data(name, path, data):
        if os.path.exists(path) is False:
            os.makedirs(path)
        with open(os.path.join(path, name), 'w') as file:
            for row in data:
                file.writelines(' '.join(map(str, row.tolist())))


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01) # 从截断正态分布中抽取进行初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out") # kaiming 正态分布
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_without_bias(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                              qkv_bias=False)
    return model


def vit_base_patch16_224_int8(num_classes: int = 1000):
    model = QVisionTransformer(img_size=224,
                               patch_size=16,
                               embed_dim=768,
                               depth=12,
                               num_heads=12,
                               representation_size=None,
                               num_classes=num_classes)
    return model

def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model


