import torch
from torchvision.utils import make_grid, save_image
from einops import rearrange, repeat, reduce
from torchvision import transforms as T
import torch.nn.functional as F
import imageio
import os

@torch.no_grad()
def concat_all_gather(accelerator,tensor):
    tensors_gather = accelerator.gather(tensor)
    return tensors_gather

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

@torch.no_grad()
def ddim_sample(sampler,unet,vae,shape,c,start_code,x0_emb,ddim_steps=10,scale=1.0, uc=None):
    frames = x0_emb.shape[2]
    if scale == 1.0:
        uc = None
    samples_ddim, _ = sampler.sample(unet = unet,
                                        S=ddim_steps,
                                        conditioning=c,
                                        batch_size=shape[0],
                                        shape=shape[1:],
                                        x0_emb = x0_emb,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=0.0,
                                        x_T=start_code,
                                        is_3d = True)
    samples_ddim = rearrange(samples_ddim, 'n c f h w -> (n f) c h w')
    samples_ddim = 1 / 0.18215 * samples_ddim
    x_samples_ddim = vae.decode(samples_ddim).sample
    x_samples_ddim = rearrange(x_samples_ddim, '(n f) c h w -> n c f h w', f = shape[2])
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    return x_samples_ddim



def save_visualization(accelerator, vae, x_samples_ddim, video_latent, video, results_folder, global_step, num_sample_rows=2):
    f = video_latent.shape[2]
    video_latent = rearrange(video_latent, 'n c f h w -> (n f) c h w')
    video_latent = 1 / 0.18215 * video_latent 
    video_recon = vae.decode(video_latent).sample
    video_recon = rearrange(video_recon, '(n f) c h w -> n c f h w', f = f)
    video_recon = (video_recon + 1.0) / 2.0

    f0 = video.shape[2] - f
    ori_videos_tensor = video[:,:,f0:,:,:]
    ori_videos_cond_tensor = video[:,:,:f0,:,:]
    ori_videos_tensor = (ori_videos_tensor + 1.0) / 2.0
    ori_videos_cond_tensor = (ori_videos_cond_tensor + 1.0) / 2.0

    all_videos_tensor = F.pad(concat_all_gather(accelerator,x_samples_ddim.contiguous()), (2, 2, 2, 2))
    recon_videos_tensor = F.pad(concat_all_gather(accelerator,video_recon.contiguous()), (2, 2, 2, 2))
    ori_videos_tensor = F.pad(concat_all_gather(accelerator,ori_videos_tensor.contiguous()), (2, 2, 2, 2))
    ori_videos_cond_tensor = F.pad(concat_all_gather(accelerator,ori_videos_cond_tensor.contiguous()), (2, 2, 2, 2))

    one_gif = rearrange(all_videos_tensor, '(i j) c f h w -> c f (i h) (j w)', i = num_sample_rows)
    one_gif_ori = rearrange(ori_videos_tensor, '(i j) c f h w -> c f (i h) (j w)', i = num_sample_rows)
    one_gif = (one_gif.permute(1, 2, 3, 0).cpu().numpy()*255).astype('uint8')
    one_gif_ori = (one_gif_ori.permute(1, 2, 3, 0).cpu().numpy()*255).astype('uint8')

    video_path = os.path.join(results_folder, f'{global_step}.gif')
    ori_video_path = os.path.join(results_folder, f'ori_{global_step}.gif')
    imageio.mimwrite(video_path, one_gif, fps=4)
    imageio.mimwrite(ori_video_path, one_gif_ori, fps=4)
    
    recon_videos_tensor = rearrange(recon_videos_tensor, 'b c f h w -> b c h (f w)').to('cpu')
    ori_videos_tensor = rearrange(ori_videos_tensor, 'b c f h w -> b c h (f w)').to('cpu')
    reali = torch.cat([ori_videos_tensor,recon_videos_tensor], dim=-2)
    pred = rearrange(all_videos_tensor, 'b c f h w -> b c h (f w)').to('cpu')
    cond = rearrange(ori_videos_cond_tensor, 'b c f h w -> b c h (f w)').to('cpu')

    reali_pre = torch.cat([reali, pred], dim=-2)
    cond_expand = cond.repeat(1,1,3,1)

    padding = 0.5 * torch.ones(len(reali_pre), reali_pre.shape[1], reali_pre.shape[2], 4)
    padding_red, padding_green = torch.ones(len(reali_pre), reali_pre.shape[1], reali_pre.shape[2], 4), torch.ones(len(reali_pre), reali_pre.shape[1], reali_pre.shape[2], 4)
    padding_red[:, [1, 2]], padding_green[:, [0, 2]] = 0, 0

    data = torch.cat([cond_expand, padding_green, reali_pre, padding_red], dim=-1)
    nrow = 1
    image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
    
    save_image(image_grid, os.path.join(results_folder, 'image_grid_{}.png'.format(int(global_step))))


def save_visualization_onegif(accelerator, vae, x_samples_ddim, x0_image, sample_id, image_path, num_sample_rows=1):
    f = x_samples_ddim.shape[2]
    f0 = x0_image.shape[2]
    base_file_name = image_path.rsplit('.',1)[0]

    all_videos_tensor = F.pad(concat_all_gather(accelerator,x_samples_ddim.contiguous()), (2, 2, 2, 2))
    ori_videos_cond_tensor = F.pad(concat_all_gather(accelerator,x0_image.contiguous()), (2, 2, 2, 2))

    one_gif = rearrange(torch.cat([ori_videos_cond_tensor,all_videos_tensor],dim=2), '(i j) c f h w -> c f (i h) (j w)', i = num_sample_rows)
    one_gif = (one_gif.permute(1, 2, 3, 0).cpu().numpy()*255).astype('uint8')

    video_path = base_file_name+'_{}.gif'.format(int(sample_id))
    imageio.mimwrite(video_path, one_gif, fps=4)
    
    pred = rearrange(all_videos_tensor, 'b c f h w -> b c h (f w)').to('cpu')
    cond = rearrange(ori_videos_cond_tensor, 'b c f h w -> b c h (f w)').to('cpu')

    reali_pre = pred
    cond_expand = cond

    padding = 0.5 * torch.ones(len(reali_pre), reali_pre.shape[1], reali_pre.shape[2], 4)
    padding_red, padding_green = torch.ones(len(reali_pre), reali_pre.shape[1], reali_pre.shape[2], 4), torch.ones(len(reali_pre), reali_pre.shape[1], reali_pre.shape[2], 4)
    padding_red[:, [1, 2]], padding_green[:, [0, 2]] = 0, 0

    data = torch.cat([cond_expand, padding_green, reali_pre, padding_red], dim=-1)
    nrow = 1
    image_grid = make_grid(data, nrow=nrow, padding=6, pad_value=0.5)
    
    save_image(image_grid, base_file_name+'_grid_{}.png'.format(int(sample_id)))

