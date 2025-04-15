############################################################################################
# Reference: https://huggingface.co/blog/annotated-diffusion
############################################################################################


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from torch import einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


############################################
# Helper Functions
############################################

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    """
    Implements a residual/skip connection module that helps to mitigate gradient vanishing/explosion.

    Mechanism:
        * For an input x and a transformation function fn(x), the residual block computes:
            H(x) = fn(x) + x
        * fn(x) = H(x) - x, whcih learns the difference between input and output.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    """Up-sampling module."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """Down-sampling module."""
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


############################################
# Position embeddings
############################################

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Employs sinusoidal position embeddings to encode sequential information.

    As the parameters of the neural network are shared across time,
    this makes the neural network "know" at which particular time step it is operating.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


############################################
# ResNet/ConvNeXT block
############################################

class WeightStandardizedConv2d(nn.Conv2d):
    """
    A extend version of nn.Conv2d module based on the paper from https://arxiv.org/abs/1903.10520
    
    During each forward process, the convolutional weights is standardized with the following equation:
        - (weight - mean) / sqrt(var + eps) 
    Suitable for coupling with group normalization.
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    """
    A basic building block with:
        - 2D convolution (with padding)
        - Group normalization
        - Optional conditioning through scale and shift parameters
        - SiLU (Sigmoid Linear Unit) activation function
    """
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    A residual block with time embeddings.
    
    Reference:
        https://arxiv.org/abs/1512.03385
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


############################################
# Attention Modules
############################################

class Attention(nn.Module):
    """
    A multi-head self-attention module that flattens the spatial dimensions 
    to work with 2d feature maps.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """A improved attention module that reduces computational complexity."""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Cross-attention module used to incorporate text embeddings in conditioning U-Net."""
    def __init__(self, query_dim, context_dim, heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, -1).transpose(1, 2)
        x_norm = self.norm(x_flat)
        heads = self.heads
        
        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), (q, k, v))
        
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        out = out + x_flat 
        out = out.transpose(1, 2).reshape(b, c, h, w)
        
        return out


class LinearCrossAttention(nn.Module):
    """Cross-attention module with linear complexity to reduce computational complexity."""
    def __init__(self, query_dim, context_dim, heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, -1).transpose(1, 2)
        x_norm = self.norm(x_flat)
        heads = self.heads
        
        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), (q, k, v))
        
        q = q * self.scale
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        
        context_matrix = einsum('b h j d, b h j e -> b h d e', k, v)
        
        out = einsum('b h i d, b h d e -> b h i e', q, context_matrix)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        out = out + x_flat
        out = out.transpose(1, 2).reshape(b, c, h, w)
        
        return out


############################################
# Group Normalization
############################################

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


############################################
# Conditional U-Net
############################################

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        self_condition=False,
        resnet_block_groups=4,
        context_dim=None,
        use_linear_attn=True,
        use_cross_attention=True
    ):
        super().__init__()

        # Determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.use_cross_attention = use_cross_attention
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Choose attention modules
        attn_klass = LinearAttention if use_linear_attn else Attention
        cross_attn_klass = LinearCrossAttention if use_linear_attn else CrossAttention

        # Time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Downsampling
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, attn_klass(dim_in))),
                        Residual(PreNorm(dim_in, cross_attn_klass(dim_in, context_dim, heads=4))) 
                        if context_dim is not None and use_cross_attention else nn.Identity(),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )
        
        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, attn_klass(mid_dim)))
        if context_dim is not None and use_cross_attention:
            self.mid_cross_attn = Residual(PreNorm(mid_dim, cross_attn_klass(mid_dim, context_dim, heads=8))) 
        else:
             self.mid_cross_attn = nn.Identity()
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, attn_klass(dim_out))),
                        Residual(PreNorm(dim_out, cross_attn_klass(dim_out, context_dim, heads=4)))
                        if context_dim is not None and use_cross_attention else nn.Identity(),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, context=None, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
            
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        
        h = []
        
        # Downsampling
        for block1, block2, attn, cross_attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            x = attn(x)
            # If context provided, apply cross attention
            if self.use_cross_attention and (context is not None):
                x = cross_attn(x, context)
            h.append(x)
            
            x = downsample(x)
        
        # Bottlenect
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        if self.use_cross_attention and (context is not None):
            x = self.mid_cross_attn(x, context)
        x = self.mid_block2(x, t)
        
        # Upsampling
        for block1, block2, attn, cross_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            if context is not None:
                x = cross_attn(x, context)
                
            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
    