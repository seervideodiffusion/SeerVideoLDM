import argparse
import math
import os
from pathlib import Path
from typing import Optional
from einops import rearrange, repeat, reduce
from multiprocessing import Process

import numpy as np
import torch
from torchvision import transforms as T, utils
from torch.utils import data
from pathlib import Path
import torch.nn.functional as F
import torch.utils.checkpoint
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from seer.models.unet_3d_condition import SeerUNet, FSTextTransformer
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from multiprocessing import Process

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
logger = get_logger(__name__)


def cycle(dl):
    while True:
        for data in dl:
            yield data
# utils
@torch.no_grad()
def concat_all_gather(accelerator,tensor):
    tensors_gather = accelerator.gather(tensor)

    return tensors_gather
    
def save_progress(unet, accelerator, args, save_path):
    logger.info("Saving param")
    unet_instance = accelerator.unwrap_model(unet)
    learned_dict = {"model": unet_instance.state_dict()}
    torch.save(learned_dict, save_path)

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def freeze_params(params):
    for param in params:
        param.requires_grad = False


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, save_seq=True):
        self.momentum = momentum
        self.save_seq = save_seq
        if self.save_seq:
            self.vals, self.steps = [], []
        self.reset()

    def ckpt(self):
        return {'vals':self.vals,'avg':self.avg,'steps':self.steps}

    def load(self,dict_ckpt):
        self.vals = dict_ckpt['vals']
        if len(self.vals)>0:
            self.val = self.vals[-1]
        self.avg = dict_ckpt['avg']
        self.steps = dict_ckpt['steps']

    def reset(self):
        self.val, self.avg = None, 0

    def update(self, val, step=None):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)

    def synchronize_and_update(self, accelerator, val, step=None):
        """
        Warning: does not synchronize the deque!
        """
        val = accelerator.reduce(val, reduction = 'mean')
        val = val.item()
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)
        return val

def main(args):
    if args.data_dir is None:
        raise ValueError("You must specify a data directory.")
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    trainable_modules = (
        "temporal_attentions",
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    save_epoch = 0
    global_step = 0
    lr_meter = RunningAverageMeter()
    losses_train = RunningAverageMeter()
    
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
    # Load models and create wrapper for stable diffusion
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
    fstext_init_state_dict = torch.load(args.fstext_init_ckpt, map_location="cpu")
    msg = fstext_model.load_state_dict(fstext_init_state_dict, strict=False)
    del fstext_init_state_dict
    fstext_model.set_numframe(args.num_frames)
    sunet.requires_grad_(False)
    for name, module in sunet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if is_xformers_available():
        try:
            sunet.enable_xformers_memory_efficient_attention()
            fstext_model.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    
    # Freeze vae and CLIP Text encoder
    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    parameters = list(filter(lambda p: p.requires_grad, sunet.parameters()))+list(fstext_model.parameters())
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # Initialize the optimizer
    optimizer = optimizer_class(
        parameters,  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.dataset == 'bridgedata':
        from dataset.bridgedata import Dataset
    elif args.dataset == 'sthv2':
        from dataset.sthv2 import Dataset
    elif args.dataset == 'epickitchen':
        from dataset.epickitchen import Dataset
    else:
        NotImplementedError
    ds_train = Dataset(args.data_dir, args.resolution, val_batch_size = args.val_batch_size, channels = 3, num_frames = args.num_frames, split = 'train', normalize = False)

    print(f'found {len(ds_train)} videos as gif files at {args.data_dir}')
    assert len(ds_train) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size = args.train_batch_size,num_workers=args.num_workers, pin_memory=True, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * accelerator.num_processes * args.gradient_accumulation_steps,
    )
    
    sunet, fstext_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        sunet, fstext_model, optimizer, train_dataloader, lr_scheduler
    )
    load_path = os.path.join(args.output_dir, f"learned_sdunet-steps-{args.saved_global_step}")
    load_path_file = os.path.join(args.output_dir, f"learned_sdunet-steps-{args.saved_global_step}.pt")
    if os.path.exists(load_path):
        print("loading")
        accelerator.load_state(load_path)
    if os.path.exists(load_path_file):
        print("loading steps")
        state_dict = torch.load(load_path_file)
        global_step = state_dict["global_step"]
        save_epoch = state_dict["epoch"]
        lr_meter.load(state_dict['lr_meter'])
        losses_train.load(state_dict['losses_train'])
        print(global_step)

    
    # Move vae and Text encoder to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Keep vae and Text encoder in eval model as we don't train these
    vae.eval()
    text_encoder.eval()
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("sd_sunet_finetune")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(ds_train)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step,args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(save_epoch,args.num_train_epochs):
        sunet.train()
        fstext_model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(sunet):
                video, input_text_prompts = batch

                cond_input = text_tokenizer(
                            input_text_prompts,
                            padding="max_length",
                            max_length=text_tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                text_cond_emb = text_encoder(
                        cond_input.input_ids.to(accelerator.device),
                        attention_mask=cond_input.attention_mask.to(accelerator.device),
                )
                text_cond_emb = text_cond_emb[0]

                x0_image = video[:,:,:args.cond_frames,:,:] #first frame
                images = video[:,:,args.cond_frames:,:,:] #future frames
                _, c, f1, h, w = x0_image.shape
                f2 = images.shape[2]
                x0_image = rearrange(x0_image, 'b c f h w -> (b f) c h w')
                images = rearrange(images, 'b c f h w -> (b f) c h w')

                #with autocast(enabled = (args.mixed_precision=="fp16")):
                with accelerator.autocast():
                    text_seq_cond_emb = fstext_model(context=text_cond_emb)
                    if args.text_loss: #the initialization optimization of FSText decomposer
                        loss_text = F.mse_loss(text_seq_cond_emb.mean(1), text_cond_emb.clone().detach(), reduction="none").mean([1, 2]).mean()
                    
                    latents = vae.encode(images).latent_dist.sample().detach()
                    latents_x0 = vae.encode(x0_image).latent_dist.sample().detach()
                    latents = latents * 0.18215
                    latents_x0 = latents_x0 * 0.18215
                    latents_x0 = rearrange(latents_x0, '(b f) c h w -> b c f h w', f=f1)
                    latents = rearrange(latents, '(b f) c h w -> b c f h w', f=f2)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()
                    
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents = torch.cat([latents_x0,noisy_latents],dim=2)

                    model_pred = sunet(noisy_latents, timesteps, text_seq_cond_emb, args.cond_frames)#.sample
                    
                    model_pred = model_pred[:,:,args.cond_frames:,:,:]
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    if args.text_loss:
                        loss = F.mse_loss(model_pred, target, reduction="none").mean([1, 2, 3, 4]).mean()+loss_text
                    else:
                        loss = F.mse_loss(model_pred, target, reduction="none").mean([1, 2, 3, 4]).mean()
                    
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(sunet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                losses_train.synchronize_and_update(accelerator, loss, global_step)
                lr_meter.update(lr_scheduler.get_last_lr()[0],global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:  
                    save_path = os.path.join(args.output_dir, f"learned_sdunet-steps-{global_step}")
                    accelerator.save_state(save_path)
                    save_path_file = os.path.join(args.output_dir, f"learned_sdunet-steps-{global_step}.pt")
                    accelerator.save({"epoch":epoch,"global_step":global_step, 'lr_meter': lr_meter.ckpt(), 'losses_train': losses_train.ckpt()},save_path_file)
                    # Plot graphs
                    try:
                        plot_graphs_process.join()
                    except:
                        pass
                    plot_graphs_process = Process(target=plot_graphs, args=(losses_train, lr_meter, args.output_dir,))
                    plot_graphs_process.start()
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
    accelerator.end_training()

def plot_graphs(losses_train, lr_meter, log_folder):
    # Losses
    plt.plot(losses_train.steps, losses_train.vals, label='Train')
    plt.xlabel("Steps")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(log_folder, 'loss.png'))
    plt.yscale("log")
    plt.savefig(os.path.join(log_folder, 'loss_log.png'))
    plt.clf()
    plt.close()
    # LR
    plt.plot(lr_meter.steps, lr_meter.vals)
    plt.xlabel("Steps")
    plt.ylabel("LR")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    plt.savefig(os.path.join(log_folder, 'lr.png'))
    plt.clf()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()
    args = OmegaConf.load(args.config)
    main(args)
