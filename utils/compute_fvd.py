import os
import functools
import argparse
from .download import load_i3d_pretrained
from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from .fvd import get_fvd_logits, frechet_distance, polynomial_mmd

SPLIT_BATCH = 2

# utils
@torch.no_grad()
def concat_all_gather(accelerator,tensor):
    tensors_gather = accelerator.gather(tensor)
    return tensors_gather

def all_gather(tensor):
    rank, size = dist.get_rank(), dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)

@torch.no_grad()
def eval_video_fvd(accelerator, i3d, pred, gt, fake_embeddings_stack, real_embeddings_stack):
    rank, size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0
    fake = pred
    fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    fake = (fake * 255).astype('uint8')
    fake_embeddings_stack.append(get_fvd_logits(fake, i3d=i3d, device=accelerator.device))
    real = gt
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings_stack.append(get_fvd_logits(real, i3d=i3d, device=accelerator.device))
    fake_embeddings = torch.cat(fake_embeddings_stack)
    real_embeddings = torch.cat(real_embeddings_stack)

    fvd = frechet_distance(fake_embeddings, real_embeddings)
    kvd = polynomial_mmd(fake_embeddings.cpu(), real_embeddings.cpu())
    return fvd.item(), kvd.item(), fake_embeddings_stack, real_embeddings_stack

@torch.no_grad()
def eval_recon_fvd(i3d, vq_enc, loader, device):
    rank, size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0

    batch = next(iter(loader)).to(device)
    fake_embeddings = []
    for i in range(0, batch.shape[0], MAX_BATCH):
        fake = vq_enc.sample(batch[i:i+MAX_BATCH])
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        fake = (fake * 255).astype('uint8')
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    real = batch.to(device)

    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake_embeddings = all_gather(fake_embeddings)
    real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_embeddings.shape[0] == 256

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    return fvd.item()

@torch.no_grad()
def eval_sample_fvd(i3d, videon, unet, loader, device, cond_scale = 1.):
    rank, size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0

    batch = next(iter(loader)).to(device)
    fake_embeddings = []
    recon_embeddings = []
    for i in range(0, batch.shape[0], MAX_BATCH):
        pred_videos_tensor, ori_videos_tensor, cond_frames= videon.sample(batch[i:i+MAX_BATCH],unet,return_loss=False, cond_scale = cond_scale)
        fake = torch.cat([cond_frames,pred_videos_tensor],dim=2)
        recon = torch.cat([cond_frames,ori_videos_tensor[1]],dim=2)
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        fake = (fake * 255).astype('uint8')
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))

        recon = recon.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        recon = (recon * 255).astype('uint8')
        recon_embeddings.append(get_fvd_logits(recon, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)
    recon_embeddings = torch.cat(recon_embeddings)

    real = batch.to(device)

    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake_embeddings = all_gather(fake_embeddings)
    recon_embeddings = all_gather(recon_embeddings)
    real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_embeddings.shape[0] == recon_embeddings.shape[0] == 256

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    fvd_star = frechet_distance(fake_embeddings.clone(), recon_embeddings)
    return fvd.item(), fvd_star.item()

@torch.no_grad()
def eval_sample_text_fvd(i3d, videon, unet, loader, device, cond_scale = 1.):
    rank, size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0

    batch = next(iter(loader))
    fake_embeddings = []
    recon_embeddings = []
    for i in range(0, batch[0].shape[0], MAX_BATCH):
        print(i)
        video_val, text_cond_emb_val = batch
        text_cond_emb_val = text_cond_emb_val[:,0,:,:]
        video_val, text_cond_emb_val = video_val[i:i+MAX_BATCH].to(device), text_cond_emb_val[i:i+MAX_BATCH].to(device)
        pred_videos_tensor, ori_videos_tensor, cond_frames= videon.sample(video_val, text_cond_emb_val, unet, return_loss=False, cond_scale = cond_scale)
        fake = torch.cat([cond_frames,pred_videos_tensor],dim=2)
        recon = torch.cat([cond_frames,ori_videos_tensor[1]],dim=2)
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        fake = (fake * 255).astype('uint8')
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))

        recon = recon.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        recon = (recon * 255).astype('uint8')
        recon_embeddings.append(get_fvd_logits(recon, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)
    recon_embeddings = torch.cat(recon_embeddings)

    real = batch[0].to(device)

    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake_embeddings = all_gather(fake_embeddings)
    recon_embeddings = all_gather(recon_embeddings)
    real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_embeddings.shape[0] == recon_embeddings.shape[0] == 256

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    fvd_star = frechet_distance(fake_embeddings.clone(), recon_embeddings)
    return fvd.item(), fvd_star.item()

@torch.no_grad()
def eval_dalle2_sample_fvd(i3d, videon, unet, loader, device, cond_scale = 1.):
    rank, size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0

    batch = next(iter(loader))
    fake_embeddings = []
    recon_embeddings = []
    for i in range(0, batch.shape[0], MAX_BATCH):
        video_val, token_ids_val, img_cond_emb_val = batch[i:i+MAX_BATCH]
        video_val, token_ids_val, img_cond_emb_val = video_val.to(device), token_ids_val.to(device), img_cond_emb_val.to(device)
        token_ids_val = token_ids_val[:,0,:]
        pred_videos_tensor, ori_videos_tensor, cond_frames= videon.sample(video_val, img_cond_emb_val,unet,return_loss=False, cond_scale = cond_scale)
        fake = torch.cat([cond_frames,pred_videos_tensor],dim=2)
        recon = torch.cat([cond_frames,ori_videos_tensor[1]],dim=2)
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        fake = (fake * 255).astype('uint8')
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))

        recon = recon.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        recon = (recon * 255).astype('uint8')
        recon_embeddings.append(get_fvd_logits(recon, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)
    recon_embeddings = torch.cat(recon_embeddings)

    real = batch.to(device)

    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake_embeddings = all_gather(fake_embeddings)
    recon_embeddings = all_gather(recon_embeddings)
    real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_embeddings.shape[0] == recon_embeddings.shape[0] == 256

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    fvd_star = frechet_distance(fake_embeddings.clone(), recon_embeddings)
    return fvd.item(), fvd_star.item()

@torch.no_grad()
def eval_recon_fvd_256(i3d, vq_enc, loader, device):
    rank, size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0

    batch = next(iter(loader)).to(device)
    fake_embeddings = []
    for i in range(0, batch.shape[0], MAX_BATCH):
        fake = []
        for j in range(0,MAX_BATCH,SPLIT_BATCH):
            fake.append(vq_enc.sample(batch[i+j:i+j+SPLIT_BATCH]))
        fake = torch.cat(fake, dim=0)
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        fake = (fake * 255).astype('uint8')
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    real_embeddings = []
    for i in range(0, batch.shape[0], MAX_BATCH):
        real = batch[i:i+MAX_BATCH]
        real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        real = (real * 255).astype('uint8')
        real_embeddings.append(get_fvd_logits(real, i3d=i3d, device=device))
    real_embeddings = torch.cat(real_embeddings)

    fake_embeddings = all_gather(fake_embeddings)
    real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_embeddings.shape[0] == 256

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    return fvd.item()

@torch.no_grad()
def eval_sample_fvd_256(i3d, videon, unet, loader, device, cond_scale = 1.):
    rank, size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0

    batch = next(iter(loader)).to(device)
    fake_embeddings = []
    recon_embeddings = []
    for i in range(0, batch.shape[0], MAX_BATCH):
        fake = []
        recon = []
        for j in range(0,MAX_BATCH,SPLIT_BATCH):
            pred_videos_tensor, ori_videos_tensor, cond_frames= videon.sample(batch[i+j:i+j+SPLIT_BATCH],unet,return_loss=False)
            fake.append(torch.cat([cond_frames,pred_videos_tensor],dim=2))
            recon.append(torch.cat([cond_frames,ori_videos_tensor[1]],dim=2))
        fake = torch.cat(fake, dim=0)
        recon = torch.cat(recon, dim=0)
        print(fake.shape,recon.shape)
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        fake = (fake * 255).astype('uint8')
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))

        recon = recon.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        recon = (recon * 255).astype('uint8')
        recon_embeddings.append(get_fvd_logits(recon, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)
    recon_embeddings = torch.cat(recon_embeddings)

    real = batch.to(device)

    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake_embeddings = all_gather(fake_embeddings)
    recon_embeddings = all_gather(recon_embeddings)
    real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_embeddings.shape[0] == recon_embeddings.shape[0] == 256

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    fvd_star = frechet_distance(fake_embeddings.clone(), recon_embeddings)
    return fvd.item(), fvd_star.item()


import math
import logging
import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np
logger = logging.getLogger(__name__)

def inception_score(classifier, samples, y_score_stack, batchsize=100, splits=10, eps=1e-20):
    """Compute the inception score for given images.
    Default batchsize is 100 and split size is 10. Please refer to the
    official implementation. It is recommended to to use at least 50000
    images to obtain a reliable score.
    Reference:
    https://github.com/openai/improved-gan/blob/master/inception_score/classifier.py
    """
    n = samples.shape[0]
    n_batches = int(math.ceil(float(n) / float(batchsize)))
    xp = classifier.xp

    # print('Batch size:', batchsize)
    # print('Total number of images:', n)
    # print('Total number of batches:', n_batches)

    # Compute the softmax predicitions for for all images, split into batches
    # in order to fit in memory
    y_score_stack = []
    for i in range(n_batches):
        logger.info('Running batch %i/%i...', i + 1, n_batches)
        batch_start = (i * batchsize)
        batch_end = min((i + 1) * batchsize, n)
        samples_batch = samples[batch_start:batch_end]
        #samples_batch = samples
        samples_batch = xp.asarray(samples_batch)  # To GPU if using CuPy
        samples_batch = Variable(samples_batch)
        # Feed images to the inception module to get the softmax predictions
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = F.softmax(classifier(samples_batch))
        y_score_stack.append(y.data.astype(xp.float64))
    n  = len(y_score_stack)
    # Compute the inception score based on the softmax predictions of the
    # inception module.
    scores = xp.empty((splits), dtype=xp.float64)  # Split inception scores
    for i in range(splits):
        if n<splits:
            part = F.concat(y_score_stack[i : i + 1],axis =0).data.astype(xp.float64)
        else:
            part = F.concat(y_score_stack[(i * n // splits):((i + 1) * n // splits)],axis =0).data.astype(xp.float64)
        part = part + eps  # to avoid convergence
        kl = part * (xp.log(part) -
                     xp.log(xp.expand_dims(xp.mean(part, 0), 0)))
        kl = xp.mean(xp.sum(kl, 1))
        scores[i] = xp.exp(kl)
        if n<=i+1:
            break
    if n<splits:
        return xp.mean(scores[:n]), xp.std(scores[:n]), y_score_stack
    else:
        return xp.mean(scores), xp.std(scores), y_score_stack