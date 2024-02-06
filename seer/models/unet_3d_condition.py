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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat, reduce

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .attention import  LinearTransformer3D
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    UNetMidBlock3DCrossAttn,
    DownBlock3D,
    UpBlock3D,
    get_down_block,
    get_up_block,
)

MAX_LENGTH  = 1024
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class SeerUNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(
        self,
        sample_size=None,
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=1280,
        attention_head_dim=8,

    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        
        down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D")
        up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D")
        downsample_padding=1
        # input
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            if down_block_type == "CrossAttnDownBlock3D":
                down_block = get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim,
                    downsample_padding=downsample_padding,
                    text_frame_condition=True,
                    causal = True,
                )
            else:
                down_block = get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim,
                    downsample_padding=downsample_padding,
                )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            text_frame_condition=True,
            causal = True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "CrossAttnUpBlock3D":
                up_block = get_up_block(
                    up_block_type,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim,
                    text_frame_condition=True,
                    causal = True,
                )
            else:
                up_block = get_up_block(
                    up_block_type,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim,
                )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
    """
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('SpatialTransformer3D') != -1:
            print(m)
            m.apply(m._init_weights)
            m.to(self.dtype)
    """
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, UNetMidBlock3DCrossAttn, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        context: torch.Tensor,
        cond_frame:int = 0,
        return_attn:bool = False,
    ) -> torch.FloatTensor:
        if return_attn:
            attn_list = []
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension
        timesteps = timesteps.broadcast_to(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)
        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:

            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                if return_attn:
                    sample, res_samples, attn_map = downsample_block(
                        hidden_states=sample, temb=emb, encoder_hidden_states=context, cond_frame = cond_frame, return_attn = return_attn
                    )
                    attn_list.append(attn_map)
                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample, temb=emb, encoder_hidden_states=context, cond_frame = cond_frame
                    )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples
        # 4. mid
        if return_attn:
            sample,attn_map = self.mid_block(sample, emb, encoder_hidden_states=context, cond_frame = cond_frame, return_attn = return_attn)
            attn_list.append(attn_map)
        else:
            sample = self.mid_block(sample, emb, encoder_hidden_states=context, cond_frame = cond_frame)

        # 5. up
        for upsample_block in self.up_blocks:

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                if return_attn:
                    sample,attn_map = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=context,
                        cond_frame = cond_frame, 
                        return_attn = return_attn,
                    )
                    attn_list.append(attn_map)
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=context,
                        cond_frame = cond_frame, 
                    )
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)

        # 6. post-process
        # make sure hidden states is in float32
        # when running in half-precision
        sample = self.conv_norm_out(sample.float()).type(sample.dtype)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        output = sample #{"sample": sample}
        if return_attn:
            return output,attn_list
        else:
            return output


class FSTextTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    def __init__(
        self,
        num_frames=None,
        in_channels=768,
        out_channels=768,
        n_heads = 8,
        num_layers=2,
        cross_attention_dim=768,
    ):
        super().__init__()
        d_head = out_channels//n_heads
        self.learnable_query = nn.Parameter(torch.zeros(1, 1, 1, out_channels))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, MAX_LENGTH, out_channels))
        self.trf_blocks = nn.ModuleList([])
        self.num_frames = num_frames
        for n in range(num_layers):
            self.trf_blocks.append(LinearTransformer3D(in_channels = in_channels, n_heads = n_heads, d_head = d_head,\
                depth = 2, context_dim = cross_attention_dim, temporal = [False, True]))
        self.norm = nn.LayerNorm(out_channels)
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (LinearTransformer3D)):
            module.gradient_checkpointing = value
    def set_numframe(self,num_frames):
        self.num_frames = num_frames
    def forward(
        self,
        context: torch.Tensor,
    ) -> torch.FloatTensor:
        b,l,c = context.shape
        sample = self.learnable_query.expand(b,self.num_frames,l,-1)
        pos_embed = self.pos_embed[:,:,:l,:]
        if self.pos_embed.shape[1]!=self.num_frames:
            pos_embed = F.interpolate(pos_embed.permute(0,3,1,2), size=(self.num_frames,l)).permute(0,2,3,1)
        x = sample + pos_embed
        for trf_blk in self.trf_blocks:
            x = trf_blk(x,[context,None])
        x = self.norm(x)
        output = x
        return output

