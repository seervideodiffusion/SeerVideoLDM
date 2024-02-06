# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from rotary_embedding_torch import RotaryEmbedding
from xformers.components.attention import AttentionMask
from einops import rearrange
CLIP_IMG_DIM = 1024
MAX_WIN_SIZE = 8
MAX_RATIO = 4
MIN_WIN_SIZE = 4
FIX_FRAME_WIN_SIZE = 5
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
def window_partition(x, window_size):
    """
    Args:
        x: (B, F, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, F*window_size*window_size, C)
    """
    B, F, H, W, C = x.shape
    x = x.view(B, F, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(2, 4, 0, 1, 3, 5, 6).contiguous().view(-1, F*window_size*window_size, C)
    return windows
    
def window_reverse(windows, window_size, F, H, W):
    """
    Args:
        windows: (num_windows*B, F*window_size*window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, F*H*W, C)
    """
    C = windows.shape[-1]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(H // window_size, W // window_size, B, F, window_size, window_size, C)
    x = x.permute(2, 3, 0, 4, 1, 5, 6).contiguous().view(B, F*H*W, C)
    return x

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
        return x

@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            Hidden states conditioned on `encoder_hidden_states` input. If discrete, returns probability distributions
            for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class SpatialTransformer3D(nn.Module):
    """
    Transformer block for video-like data. First, project the input (aka embedding) and reshape to b, t, n, d. Then apply
    2D transformer action, Cross-Attention, and temporal attention. Finally, reshape to video tensor
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None, temporal=False, text_frame_condition = False, causal = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.proj_in = InflatedConv3d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        if text_frame_condition:
            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTextTransformerBlock3D(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                    for d in range(depth)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock3D(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, temporal=temporal, causal = causal)
                    for d in range(depth)
                ]
            )
        self.proj_out = InflatedConv3d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = zero_module(self.proj_out)
    
    def forward(self, x, context=None, cond_frame = 0, return_attn = False):
        b, c, f, h, w = x.shape
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            if return_attn:
                x,attn = block(x, context=context, cond_frame = cond_frame, return_attn = return_attn)
            else:
                x = block(x, context=context, cond_frame = cond_frame)
        x = x.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        x = self.proj_out(x) + x_in
        if return_attn:
            return Transformer3DModelOutput(sample=x)['sample'],attn
        else:
            return Transformer3DModelOutput(sample=x)['sample']

    def set_weight(self, layer):
        self.norm = layer.norm
        self.proj_in = layer.proj_in
        self.transformer_blocks = layer.transformer_blocks
        self.proj_out = layer.proj_out

class LinearTransformer3D(nn.Module):
    """
    Linear Transformer block for CLIP text embedding. Propagate learnable text tokens along text sequence / temporal dimension.
    The basic layer of FSText Decomposer for learning sub-instruction embedding
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None, image_space = False, temporal=[False,], cross_attention_only = False):
        super().__init__()
        self.cross_attention_only = cross_attention_only
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.transformer_blocks = nn.ModuleList(
            [
                BasicLinearTransformerBlock3D(in_channels, inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, temporal=temporal[d], image_space = image_space, cross_attention_only=cross_attention_only)
                for d in range(depth)
            ]
        )
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        for i,block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        return Transformer3DModelOutput(sample=x)['sample']

    def set_weight(self, layer):
        self.transformer_blocks = layer.transformer_blocks

class BasicTransformerBlock3D(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True, temporal=False, attention_bias = False, upcast_attention = False, causal = False):
        super().__init__()
        if temporal:
            self.attn1 = WindowSTempAttention(
                query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention, temporal = temporal, causal = causal
            )  # is a self-attention
            
        else:
            self.attn1 = CrossAttention(
                query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention
            )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, activation_fn="geglu")
        if not temporal:
            self.attn2 = CrossAttention(
                query_dim=dim, cross_attention_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention
            )  # is self-attn if context is none
            self.norm2 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.temporal = temporal
        
    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if not self.temporal:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, x, context=None, cond_frame = 0, return_attn = False):
        b, c, f, h, w = x.shape
        if self.temporal:
            x = x.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c).contiguous()
        else:
            x = x.permute(0, 2, 3, 4, 1).reshape(b * f, h * w, c).contiguous()
        x_norm = self.norm1(x)
        x_norm = x_norm.reshape(b, f, h, w, c).contiguous()
        x = self.attn1(x_norm).contiguous() + x
        if self.temporal:
            if cond_frame>0:
                x0 = x[:,:cond_frame*h*w,:]
                x = x[:,cond_frame*h*w:,:]
            x = self.ff(self.norm3(x)) + x
            if cond_frame>0:
                x = torch.cat([x0,x],dim=1)
            x = x.reshape(b, f, h, w, c).contiguous()
            return x
        attn = None
        if not context is None:
            x = x.reshape(b, f * h * w, c).contiguous()
            if return_attn:
                residual = x
                x,attn = self.attn2(self.norm2(x), context=context, return_attn = return_attn)
                x = x.contiguous()+residual
                attn = attn.reshape(b,-1, f, h, w, context.shape[-2])
            else:
                x = self.attn2(self.norm2(x), context=context).contiguous() + x
        x = self.ff(self.norm3(x)) + x
        if return_attn:
            return x,attn
        else:
            return x

class BasicTextTransformerBlock3D(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True, temporal=False, attention_bias = False, upcast_attention = False,):
        super().__init__()
        self.attn1 = CrossAttention(
                query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention
            )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, activation_fn="geglu")
        self.attn2 = CrossAttention(
            query_dim=dim, cross_attention_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention, temporal = False
        )  # is self-attn if context is none
        self.norm2 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.temporal = temporal
        self.checkpoint = checkpoint
        
    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if not self.temporal:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, x, context=None, cond_frame = 0, return_attn = False):
        b, c, f, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b * f, h * w, c).contiguous()
        x = self.attn1(self.norm1(x)).contiguous() + x
        attn = None
        if not context is None:
            c1 = context.shape[-1]
            context = context.reshape(b * f, -1, c1)
            if return_attn:
                residual = x
                x,attn = self.attn2(self.norm2(x), context=context, return_attn = return_attn)
                x = x.contiguous()+residual
                attn = attn.reshape(b, f,-1, h, w, context.shape[-2]).permute(0,2,1,3,4,5)
            else:
                x = self.attn2(self.norm2(x), context=context).contiguous() + x
        x = self.ff(self.norm3(x)) + x
        if return_attn:
            return x,attn
        else:
            return x
class BasicLinearTransformerBlock3D(nn.Module):
    
    def __init__(self, in_channels, dim, n_heads, d_head, dropout=0.0, context_dim=None, image_space = False, gated_ff=True, checkpoint=True, temporal=False, attention_bias = False, upcast_attention = False, cross_attention_only = False):
        super().__init__()
        self.attn1 = None
        if not cross_attention_only:
            self.attn1 = CrossAttention(
                query_dim=in_channels, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention, temporal = temporal, causal = True
            )  # is a self-attention
        self.ff = FeedForward(in_channels if temporal else dim, dropout=dropout, activation_fn="geglu") 
        self.vision_projection = None
        if not temporal:
            if in_channels!=dim:
                self.attn2 = CrossAttention(
                    query_dim=dim, inp_dim = in_channels, cross_attention_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention
                )  # is self-attn if context is none
                self.vision_projection = nn.Linear(in_channels, dim, bias=False)
            else:
                self.attn2 = CrossAttention(
                    query_dim=dim, cross_attention_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention
                )  # is self-attn if context is none
            self.norm2 = nn.LayerNorm(in_channels)
        if not cross_attention_only:
            self.norm1 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels if temporal else dim)
        self.checkpoint = checkpoint
        self.temporal = temporal
        self.image_space = image_space
        self.context_dim=context_dim
        
    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            if not self.attn1 is None:
                self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if not self.temporal:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, x, context=None):
        b, f, l, c = x.shape
        if not self.attn1 is None:
            if self.temporal:
                if self.image_space:
                    x = x.reshape(b, f * l, c).contiguous()
                else:
                    x = x.permute(0, 2, 1, 3).reshape(b * l, f, c).contiguous()
            else:
                x = x.reshape(b * f, l, c).contiguous()
            x = self.attn1(self.norm1(x)).contiguous() + x
            if self.temporal:
                x = self.ff(self.norm3(x)) + x
                if self.image_space:
                    x = x.reshape(b, f, l, c).contiguous()
                else:
                    x = x.reshape(b, l, f, c).permute(0, 2, 1, 3).contiguous()
                return x
        if not context is None:
            if len(context.shape)==3:
                x = x.reshape(b, f * l, c).contiguous()
            elif len(context.shape)==4:
                f0,l0,c0 = context.shape[1:]
                if not f0==f:
                    x = x.reshape(b, f, l, c).contiguous()
                    x0 = x[:,:f0,:,:]
                    x1 = x[:,f0:,:,:]
                    x = x0.reshape(b*f0, l, c)
                else:
                    x = x.reshape(b*f0, l, c)
                context = context.reshape(b*f0, l0, c0)
        else:
            x = x.reshape(b * f, l, c).contiguous()
        if not self.vision_projection is None:
            x_temp = self.vision_projection(x)
            c = x_temp.shape[-1]
        x = self.attn2(self.norm2(x), context=context).contiguous() + (x if self.vision_projection is None else x_temp)
        if (not context is None) and context.shape[-1] == CLIP_IMG_DIM and (not f0==f):
            x = torch.cat([x.reshape(b, f0, l, c),x1],dim=1)
        x = self.ff(self.norm3(x)) + x
        x = x.reshape(b, f, l, c)
        return x
        
class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        temporal (`bool`, *optional*, defaults to False):
            Set to `True` for implementing SWAT spatial temporal attention, with.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        temporal: bool = False, 
        causal: bool = False, 
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        inp_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.added_kv_proj_dim = added_kv_proj_dim
        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        if temporal:
            self.rotary_emb = RotaryEmbedding(min(32, dim_head))
        self.temporal = temporal
        self.causal = causal

        self.to_q = nn.Linear(query_dim if inp_dim is None else inp_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))
    
    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def forward(self, hidden_states, context=None, mask=None, return_attn = False):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]
        if self.temporal:
            if self.causal:
                i, j = query.shape[-2],key.shape[-2]
                mask = torch.ones((i, j), dtype = torch.bool, device = query.device).tril(j - i)
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        if self.temporal:
            query = self.rotary_emb.rotate_queries_or_keys(query)
            key = self.rotary_emb.rotate_queries_or_keys(key)
        # TODO(PVP) - mask is currently never used. Remember to re-implement when used
        
        # attention, what we cannot get enough of
        if (not return_attn) and self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, mask = mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                if return_attn:
                    hidden_states,attn = self._attention(query, key, value, mask = mask, return_attn = return_attn)
                else:
                    hidden_states = self._attention(query, key, value, mask = mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, mask = mask)
        
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        if return_attn:
            return hidden_states,attn
        else:
            return hidden_states

    def _attention(self, query, key, value, mask = None, return_attn = False):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        if not mask is None: 
            max_neg_value = -torch.finfo(attention_scores.dtype).max
            attention_scores = attention_scores.masked_fill(~mask, max_neg_value)
        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        if return_attn:
            return hidden_states,attention_scores
        else:
            return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, mask = None):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        if not mask is None:
            mask = xformers.ops.LowerTriangularMask()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

class WindowSTempAttention(CrossAttention):
    def forward(self, hidden_states, attention_mask=None, video_length=None):
        b,f,h,w,_ = hidden_states.shape
        hidden_states = hidden_states.reshape(b, f * h * w, -1)
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)
        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        if self.temporal:
            query = self.rotary_emb.rotate_queries_or_keys(query)
            key = self.rotary_emb.rotate_queries_or_keys(key)

        query = rearrange(query, "b (f d) c -> b f d c", f=f)
        key = rearrange(key, "b (f d) c -> b f d c", f=f)
        value = rearrange(value, "b (f d) c -> b f d c", f=f)
        
        c = query.shape[-1]
        f1 = query.shape[1]
        f2 = key.shape[1]
        #print(query.shape)
        if h > MIN_WIN_SIZE:
            query = query.reshape(-1, f1, h, w, c)
            key = key.reshape(-1, f2, h, w, c)
            value = value.reshape(-1, f2, h, w, c)
            if (h//MAX_WIN_SIZE)>=MAX_RATIO:
                win_size = MAX_WIN_SIZE
            else:
                win_size = MIN_WIN_SIZE
            query_window = window_partition(query, window_size=win_size)
            key_window = window_partition(key, window_size=win_size)
            value_window = window_partition(value, window_size=win_size)
            batch_size, sequence_length, _ = value_window.shape
            mask_h = win_size
            mask_w = win_size
        else:
            query_window = query.reshape(-1, f1*h*w, c)
            key_window = key.reshape(-1, f2*h*w, c)
            value_window = value.reshape(-1, f2*h*w, c)
            mask_h = h
            mask_w = w
        if self.temporal:
            if self.causal:
                i, j = query.shape[-2],key.shape[-2]
                attention_mask = torch.ones((i, j), dtype = torch.bool, device = query.device).tril(j - i)
        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query_window, key_window, value_window, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query_window, key_window, value_window, attention_mask)
            else:
                hidden_states = self._sliced_attention(query_window, key_window, value_window, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        if h>MIN_WIN_SIZE:
            hidden_states = window_reverse(hidden_states, window_size=win_size, F=f, H = h, W = w)
        return hidden_states

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))
        
    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x
