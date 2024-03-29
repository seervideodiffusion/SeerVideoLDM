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
import numpy as np
import torch
from torch import nn

from .attention import SpatialTransformer3D
from .resnet import ResnetBlock3D, Downsample3D, Upsample3D


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    text_frame_condition=False,
    causal = False,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
        )
    
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            text_frame_condition = text_frame_condition,
            causal = causal,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    text_frame_condition = False,
    causal = False,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            text_frame_condition=text_frame_condition,
            causal = causal,
        )
    raise ValueError(f"{up_block_type} does not exist.")

class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type="default",
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        text_frame_condition=False,
        causal = False,
        **kwargs,
    ):
        super().__init__()

        self.attention_type = attention_type
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        temporal_attentions = []
        for _ in range(num_layers):
            attentions.append(
                SpatialTransformer3D(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    text_frame_condition=text_frame_condition,
                )
            )
            temporal_attentions.append(
                SpatialTransformer3D(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=None,
                    temporal=True,
                    causal = causal,
                )
            )
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.temporal_attentions = nn.ModuleList(temporal_attentions)
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, fusion = False, cond_frame = 0, return_attn = False):
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self.resnets[0]), hidden_states, temb)
        else:
            hidden_states = self.resnets[0](hidden_states, temb)
        for attn, temp_attn, resnet in zip(self.attentions, self.temporal_attentions, self.resnets[1:]):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                if fusion:
                    hidden_states_0 = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn), hidden_states, encoder_hidden_states
                    )
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states_1 = temp_attn(hidden_states)
                    else:
                        hidden_states_1 = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn), hidden_states, cond_frame
                        )
                    hidden_states = 0.5*(hidden_states_0+hidden_states_1)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn), hidden_states, encoder_hidden_states
                    )
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states = temp_attn(hidden_states)
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn), hidden_states, cond_frame
                        )
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                
            else:
                if fusion:
                    hidden_states_0 = attn(hidden_states, encoder_hidden_states)
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states_1 = temp_attn(hidden_states)
                    else:
                        hidden_states_1 = temp_attn(hidden_states, cond_frame = cond_frame)
                    hidden_states = 0.5*(hidden_states_0+hidden_states_1)
                else:
                    if return_attn:
                        hidden_states, attn_map = attn(hidden_states, encoder_hidden_states, return_attn=return_attn)
                    else:
                        hidden_states = attn(hidden_states, encoder_hidden_states)
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states = temp_attn(hidden_states)
                    else:
                        hidden_states = temp_attn(hidden_states, cond_frame = cond_frame)
                hidden_states = resnet(hidden_states, temb)
        if return_attn:
            return hidden_states, attn_map
        else:
            return hidden_states

class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        text_frame_condition=False,
        causal=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        temporal_attentions = []

        self.attention_type = attention_type

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SpatialTransformer3D(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    text_frame_condition=text_frame_condition,
                )
            )
            temporal_attentions.append(
                SpatialTransformer3D(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=None,
                    temporal = True,
                    causal = causal,
                ) #if i == (num_layers-1) else nn.Identity()
            )
        self.attentions = nn.ModuleList(attentions)
        self.temporal_attentions = nn.ModuleList(temporal_attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, fusion = False, cond_frame = 0, return_attn = False):
        output_states = ()

        for resnet, attn, temp_attn in zip(self.resnets, self.attentions, self.temporal_attentions):
            
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                if fusion:
                    hidden_states_0 = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn), hidden_states, encoder_hidden_states
                    )
                    if isinstance(temp_attn, nn.Identity):
                            hidden_states_1 = temp_attn(hidden_states)
                    else:
                        hidden_states_1 = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn), hidden_states, cond_frame
                        )
                    hidden_states = 0.5*(hidden_states_0 + hidden_states_1)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn), hidden_states, encoder_hidden_states
                    )
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states = temp_attn(hidden_states)
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn), hidden_states, cond_frame
                        )
            else:
                hidden_states = resnet(hidden_states, temb)
                if fusion:
                    hidden_states_0 = attn(hidden_states, context=encoder_hidden_states)
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states_1 = temp_attn(hidden_states)
                    else:
                        hidden_states_1 = temp_attn(hidden_states, cond_frame = cond_frame)
                    hidden_states = 0.5*(hidden_states_0 + hidden_states_1)
                else:
                    if return_attn:
                        hidden_states,attn_map = attn(hidden_states, context=encoder_hidden_states, return_attn=return_attn)
                    else:
                        hidden_states = attn(hidden_states, context=encoder_hidden_states)
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states = temp_attn(hidden_states)
                    else:
                        hidden_states = temp_attn(hidden_states, cond_frame = cond_frame)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        if return_attn and (attn_map is not None):
            return hidden_states, output_states, attn_map
        else:
            return hidden_states, output_states

class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_upsample=True,
        text_frame_condition=False,
        causal = False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        temporal_attentions = []

        self.attention_type = attention_type

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                SpatialTransformer3D(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    text_frame_condition = text_frame_condition,
                )
            )
            temporal_attentions.append(
                SpatialTransformer3D(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=None,
                    temporal = True,
                    causal = causal,
                ) #if i == (num_layers-1) else nn.Identity()
            )
        self.attentions = nn.ModuleList(attentions)
        self.temporal_attentions = nn.ModuleList(temporal_attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None, fusion = False, cond_frame = 0, return_attn = False):
        for resnet, attn, temp_attn in zip(self.resnets, self.attentions, self.temporal_attentions):
            
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                
                if fusion:
                    hidden_states_0 = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn), hidden_states, encoder_hidden_states
                    )
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states_1 = temp_attn(hidden_states)
                    else:
                        hidden_states_1 = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn), hidden_states, cond_frame
                        )
                    hidden_states = 0.5*(hidden_states_0 + hidden_states_1)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn), hidden_states, encoder_hidden_states
                    )
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states = temp_attn(hidden_states)
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(temp_attn), hidden_states, cond_frame
                        )
            else:
                hidden_states = resnet(hidden_states, temb)
                if fusion:
                    hidden_states_0 = attn(hidden_states, context=encoder_hidden_states)
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states_1 = temp_attn(hidden_states)
                    else:
                        hidden_states_1 = temp_attn(hidden_states, cond_frame = cond_frame)
                    hidden_states = 0.5*(hidden_states_0 + hidden_states_1)
                else:
                    if return_attn:
                        hidden_states,attn_map = attn(hidden_states, context=encoder_hidden_states, return_attn=return_attn)
                    else:
                        hidden_states = attn(hidden_states, context=encoder_hidden_states)
                    if isinstance(temp_attn, nn.Identity):
                        hidden_states = temp_attn(hidden_states)
                    else:
                        hidden_states = temp_attn(hidden_states, cond_frame = cond_frame)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        if return_attn and (attn_map is not None):
            return hidden_states, attn_map
        else:
            return hidden_states

class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
