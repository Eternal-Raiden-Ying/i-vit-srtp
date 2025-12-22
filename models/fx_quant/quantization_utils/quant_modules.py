import torch
import torch.nn as nn


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