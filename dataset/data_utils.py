import torch.nn.functional as F
import torch
from torchvision import transforms as T
from PIL import Image

def list_to_tensor(pathlist, channels = 3, transform = T.ToTensor()):
    tensors = []
    for path in pathlist:
        tensors.append(transform(Image.open(path)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    c,f,h,w = t.shape

    if f == frames:
        return t

    if f > frames:
        t = F.interpolate(t.unsqueeze(0),size=(frames,h,w),mode='trilinear').squeeze(0).contiguous()
        return t

    return F.pad(t, (0, 0, 0, 0, 0, frames - f, 0, 0))