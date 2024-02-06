import argparse
import itertools
import math
import os
import os.path as osp
import random
from pathlib import Path
from typing import Optional
from einops import rearrange, repeat, reduce
import json
from functools import partial, wraps
from multiprocessing import Process

import numpy as np
import torch
from torchvision import transforms as T
from torch.utils import data
from pathlib import Path
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from seer.models.unet_3d_condition import SeerUNet, FSTextTransformer
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from ldm.models.diffusion.ddim_video import DDIMSampler
from utils.ddim_sampling_utils import ddim_sample, save_visualization
from omegaconf import OmegaConf

logger = get_logger(__name__)


def cycle(dl):
    while True:
        for data in dl:
            yield data

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def main(args):
    if args.data_dir is None:
        raise ValueError("You must specify a data directory.")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    global_step = 0
    # Load models and create wrapper for stable diffusion
    text_tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    sunet = SeerUNet.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        low_cpu_mem_usage = False,
    )
    fstext_model = FSTextTransformer(num_frames = 16, num_layers = 8)
    fstext_model.set_numframe(args.num_frames)
    sampler = DDIMSampler(device = accelerator.device)
    if is_xformers_available():
        try:
            sunet.enable_xformers_memory_efficient_attention()
            fstext_model.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    
    if args.dataset == 'bridgedata':
        from dataset.bridgedata import Dataset
    elif args.dataset == 'sthv2':
        from dataset.sthv2 import Dataset
    elif args.dataset == 'epickitchen':
        from dataset.epickitchen import Dataset
    else:
        NotImplementedError
    ds_val = Dataset(args.data_dir, args.resolution, val_batch_size = args.val_batch_size, channels = 3, num_frames = args.num_frames, split = 'val', normalize = False)

    print(f'found {len(ds_val)} videos as gif files at {args.data_dir}')
    assert len(ds_val) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

    val_dl = torch.utils.data.DataLoader(ds_val, batch_size = args.val_batch_size, shuffle=False)
    
    sunet, fstext_model, val_dl = accelerator.prepare(
        sunet, fstext_model, val_dl
    )
    load_path = os.path.join(args.output_dir, f"learned_sdunet-steps-{args.saved_global_step}")
    load_path_file = os.path.join(args.output_dir, f"learned_sdunet-steps-{args.saved_global_step}.pt")
    if os.path.exists(load_path):
        print("loading")
        fstext_state_dict = torch.load(os.path.join(load_path,'pytorch_model_1.bin'), map_location="cpu")
        msg = fstext_model.load_state_dict(fstext_state_dict, strict=True)
        print(msg)
        sunet_state_dict = torch.load(os.path.join(load_path,'pytorch_model.bin'), map_location="cpu")
        msg = sunet.load_state_dict(sunet_state_dict, strict=True)
        print(msg)
    if os.path.exists(load_path_file):
        print("loading steps")
        state_dict = torch.load(load_path_file)
        global_step = state_dict["global_step"]


    # Freeze all models
    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(sunet.parameters())
    freeze_params(fstext_model.parameters())
    # Move vae and text encoder to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Keep all models in eval model during inference
    vae.eval()
    text_encoder.eval()
    sunet.eval()
    fstext_model.eval()
    
    val_iter = iter(cycle(val_dl))
    for i_ter in range(args.sample_iter):
        data_val = next(val_iter)
        video_val, input_text_val = data_val

        cond_input = text_tokenizer(
                    input_text_val,
                    padding="max_length",
                    max_length=text_tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
        text_cond_emb_val = text_encoder(
                cond_input.input_ids.to(accelerator.device),
                attention_mask=cond_input.attention_mask.to(accelerator.device),
        )
        text_cond_emb_val = text_cond_emb_val[0]
        cond_input_empty = text_tokenizer(
                    len(input_text_val)*[''],
                    padding="max_length",
                    max_length=text_tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
        text_empty_emb_val = text_encoder(
                cond_input_empty.input_ids.to(accelerator.device),
                attention_mask=cond_input_empty.attention_mask.to(accelerator.device),
        )
        text_empty_emb_val = text_empty_emb_val[0]
        
        x0_image_val = video_val[:,:,:args.cond_frames,:,:] #first frame
        images_val = video_val[:,:,args.cond_frames:,:,:] #future frames
        b, c, f1, h, w = x0_image_val.shape
        f2 = images_val.shape[2]
        x0_image_val = rearrange(x0_image_val, 'b c f h w -> (b f) c h w')
        images_val = rearrange(images_val, 'b c f h w -> (b f) c h w')
        latents_val = vae.encode(images_val).latent_dist.sample().detach()
        latents_x0_val = vae.encode(x0_image_val).latent_dist.sample().detach()
        latents_val = latents_val * 0.18215
        latents_x0_val = latents_x0_val * 0.18215
        latents_x0_val = rearrange(latents_x0_val, '(b f) c h w -> b c f h w', f=f1)
        latents_val = rearrange(latents_val, '(b f) c h w -> b c f h w', f=f2)

        text_seq_cond_emb = fstext_model(context=text_cond_emb_val).detach()
        text_empty_emb_val = text_empty_emb_val.unsqueeze(1).expand(-1,text_seq_cond_emb.shape[1],-1,-1)
        _,c_l,_,h_l,w_l=latents_val.shape
        f = f1+f2
        noise_val = torch.randn((b,c_l,f2,h_l,w_l)).to(latents_val.device)
        for j in range(args.num_samples):
            x_samples_ddim = ddim_sample(sampler, sunet, vae, shape=(b,c_l,f2,h_l,w_l), c=text_seq_cond_emb, start_code=noise_val, x0_emb=latents_x0_val,\
                                            ddim_steps=args.ddim_steps, scale=args.scale, uc=text_empty_emb_val)

            save_visualization(accelerator, vae, x_samples_ddim, video_latent=latents_val,video=video_val,results_folder=args.output_dir,\
                                global_step=global_step+i_ter*10+j, num_sample_rows = args.n_rows)

            noise_val = torch.randn((b,c_l,f2,h_l,w_l)).to(latents_val.device)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    args = parser.parse_args()
    args = OmegaConf.load(args.config)
    main(args)